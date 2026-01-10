"""
Learning Module for Meaning Chain
==================================

Implements entropy-based learning where concepts learn their parameters
(τ, g, j) from accumulated adjective observations.

Theory:
    τ = 1 + 5 × (1 - H_norm)

    Where H_norm = H / log₂(variety) is normalized Shannon entropy
    of the adjective distribution.

    - High entropy (many varied adjectives) → low τ → abstract
    - Low entropy (few concentrated adjectives) → high τ → concrete

Usage:
    from graph.learning import ConceptLearner

    learner = ConceptLearner(graph)
    learner.observe_adjective("serendipity", "happy", count=3)
    learner.observe_adjective("serendipity", "unexpected", count=2)
    learner.update_concept("serendipity")  # Recomputes τ, g, j from entropy
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class AdjObservation:
    """Single adjective observation for a concept."""
    adjective: str
    count: int
    source: str = "unknown"  # "book", "conversation", "corpus"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConceptState:
    """
    Current state of a learned concept.

    Tracks adjective distribution and derived parameters.
    """
    word: str
    adj_counts: Dict[str, int] = field(default_factory=dict)  # {adj: count}
    total_count: int = 0
    variety: int = 0

    # Derived parameters (recomputed on update)
    h_adj: float = 0.0           # Shannon entropy
    h_adj_norm: float = 0.0      # Normalized entropy [0, 1]
    tau: float = 3.0             # τ = 1 + 5 × (1 - h_norm)
    g: float = 0.0               # Goodness (projection onto good direction)
    j: List[float] = field(default_factory=lambda: [0.0] * 5)  # 5D j-vector

    # Learning metadata
    n_observations: int = 0      # Number of observation events
    confidence: float = 0.1      # [0.1, 1.0] grows with observations
    learned_from: str = "unknown"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for Neo4j storage."""
        return {
            "word": self.word,
            "tau": self.tau,
            "g": self.g,
            "j": self.j,
            "variety": self.variety,
            "h_adj": self.h_adj,
            "h_adj_norm": self.h_adj_norm,
            "total_count": self.total_count,
            "n_observations": self.n_observations,
            "confidence": self.confidence,
            "learned_from": self.learned_from,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


class EntropyCalculator:
    """
    Computes Shannon entropy and derived τ from distributions.

    Stateless utility class - can be used independently.
    """

    @staticmethod
    def shannon_entropy(counts: Dict[str, int]) -> float:
        """
        Compute Shannon entropy: H = -Σ p(x) log₂ p(x)

        Args:
            counts: {item: count} distribution

        Returns:
            Shannon entropy in bits
        """
        if not counts:
            return 0.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    @staticmethod
    def normalized_entropy(counts: Dict[str, int]) -> float:
        """
        Compute normalized entropy: H_norm = H / H_max

        Where H_max = log₂(variety)

        Returns:
            Normalized entropy in [0, 1]
        """
        if not counts or len(counts) <= 1:
            return 0.0

        h = EntropyCalculator.shannon_entropy(counts)
        h_max = math.log2(len(counts))

        return h / h_max if h_max > 0 else 0.0

    @staticmethod
    def tau_from_entropy(h_norm: float) -> float:
        """
        Derive τ from normalized entropy.

        Formula: τ = 1 + 5 × (1 - h_norm)

        - h_norm = 1 (max entropy) → τ = 1 (abstract)
        - h_norm = 0 (min entropy) → τ = 6 (concrete)

        Args:
            h_norm: Normalized entropy [0, 1]

        Returns:
            τ value [1, 6]
        """
        return 1.0 + 5.0 * (1.0 - h_norm)

    @staticmethod
    def confidence_from_observations(n: int, saturation: int = 100) -> float:
        """
        Compute confidence from observation count.

        Uses exponential saturation: conf = 0.1 + 0.9 × (1 - e^(-n/saturation))

        Args:
            n: Number of observations
            saturation: Observations for ~63% of max confidence

        Returns:
            Confidence in [0.1, 1.0]
        """
        if n <= 0:
            return 0.1

        return 0.1 + 0.9 * (1.0 - math.exp(-n / saturation))


class VectorCalculator:
    """
    Computes j-vectors and goodness from adjective clouds.

    Requires adjective vectors for centroid computation.
    """

    # J-space dimensions
    J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']

    # "Good" direction in j-space (computed from good-evil, love-hate pairs)
    J_GOOD = np.array([-0.48, -0.36, -0.17, +0.71, +0.33])

    def __init__(self, adj_vectors: Dict[str, np.ndarray] = None):
        """
        Initialize with adjective vectors.

        Args:
            adj_vectors: {adjective: 5D j-vector}
        """
        self.adj_vectors = adj_vectors or {}

    def load_vectors_from_loader(self, data_loader):
        """Load adjective vectors from DataLoader."""
        vectors = data_loader.load_word_vectors()

        for word, v in vectors.items():
            if v.get('j'):
                j_arr = np.array([v['j'].get(d, 0) for d in self.J_DIMS])
                self.adj_vectors[word] = j_arr

    def compute_j_centroid(self, adj_counts: Dict[str, int]) -> np.ndarray:
        """
        Compute j-space centroid as weighted average of adjective vectors.

        Args:
            adj_counts: {adjective: count}

        Returns:
            5D j-vector centroid
        """
        if not adj_counts:
            return np.zeros(5)

        total = sum(adj_counts.values())
        if total == 0:
            return np.zeros(5)

        centroid = np.zeros(5)
        weight_sum = 0.0

        for adj, count in adj_counts.items():
            if adj in self.adj_vectors:
                weight = count / total
                centroid += weight * self.adj_vectors[adj]
                weight_sum += weight

        # Normalize by weight sum (in case some adjectives not found)
        if weight_sum > 0:
            centroid /= weight_sum

        return centroid

    def compute_goodness(self, j_vector: np.ndarray) -> float:
        """
        Compute goodness as projection onto good direction.

        g = j · j_good / ||j_good||

        Args:
            j_vector: 5D j-vector

        Returns:
            Goodness value (unbounded, typically [-2, +2])
        """
        if np.linalg.norm(j_vector) < 1e-10:
            return 0.0

        j_good_norm = np.linalg.norm(self.J_GOOD)
        if j_good_norm < 1e-10:
            return 0.0

        return float(np.dot(j_vector, self.J_GOOD) / j_good_norm)


class ConceptLearner:
    """
    Main learning class for concepts.

    Tracks adjective observations and updates concept parameters.
    Designed to be used with MeaningGraph but can work standalone.

    Usage:
        learner = ConceptLearner()
        learner.observe_adjective("serendipity", "happy", count=3)
        learner.observe_adjective("serendipity", "unexpected", count=2)
        state = learner.get_concept_state("serendipity")
        # state.tau, state.g, state.j are now derived from entropy
    """

    def __init__(self, vector_calculator: VectorCalculator = None):
        """
        Initialize learner.

        Args:
            vector_calculator: For computing j-vectors. If None, uses default.
        """
        self.entropy_calc = EntropyCalculator()
        self.vector_calc = vector_calculator or VectorCalculator()

        # In-memory state (can be persisted to Neo4j)
        self._concepts: Dict[str, ConceptState] = {}

        # Observation history (for debugging/analysis)
        self._observations: List[AdjObservation] = []
        self._observation_limit = 10000  # Keep last N observations

    def load_vectors(self, data_loader):
        """Load adjective vectors from DataLoader."""
        self.vector_calc.load_vectors_from_loader(data_loader)

    def observe_adjective(self, noun: str, adjective: str,
                          count: int = 1, source: str = "unknown") -> ConceptState:
        """
        Record an adjective observation for a noun.

        This is the main learning entry point. Each observation:
        1. Updates the adjective count distribution
        2. Recomputes entropy and derived τ
        3. Updates j-vector and goodness

        Args:
            noun: The noun being described
            adjective: The adjective describing it
            count: Number of occurrences
            source: Origin of observation ("book", "conversation", "corpus")

        Returns:
            Updated ConceptState
        """
        # Get or create concept state
        if noun not in self._concepts:
            self._concepts[noun] = ConceptState(
                word=noun,
                learned_from=source
            )

        state = self._concepts[noun]

        # Update adjective counts
        if adjective not in state.adj_counts:
            state.adj_counts[adjective] = 0
        state.adj_counts[adjective] += count

        # Update totals
        state.total_count += count
        state.variety = len(state.adj_counts)
        state.n_observations += 1

        # Track observation
        obs = AdjObservation(adjective, count, source)
        self._observations.append(obs)
        if len(self._observations) > self._observation_limit:
            self._observations = self._observations[-self._observation_limit:]

        # Recompute derived parameters
        self._update_derived_params(state)

        return state

    def observe_batch(self, noun: str, adj_counts: Dict[str, int],
                      source: str = "unknown") -> ConceptState:
        """
        Record multiple adjective observations at once.

        More efficient than calling observe_adjective repeatedly.

        Args:
            noun: The noun being described
            adj_counts: {adjective: count}
            source: Origin of observations

        Returns:
            Updated ConceptState
        """
        if noun not in self._concepts:
            self._concepts[noun] = ConceptState(
                word=noun,
                learned_from=source
            )

        state = self._concepts[noun]

        # Merge counts
        for adj, count in adj_counts.items():
            if adj not in state.adj_counts:
                state.adj_counts[adj] = 0
            state.adj_counts[adj] += count
            state.total_count += count

        state.variety = len(state.adj_counts)
        state.n_observations += len(adj_counts)

        # Recompute derived parameters
        self._update_derived_params(state)

        return state

    def _update_derived_params(self, state: ConceptState):
        """
        Recompute τ, g, j from current adjective distribution.

        Called automatically after observations.
        """
        # Compute entropy
        state.h_adj = self.entropy_calc.shannon_entropy(state.adj_counts)
        state.h_adj_norm = self.entropy_calc.normalized_entropy(state.adj_counts)

        # Derive τ from entropy
        state.tau = self.entropy_calc.tau_from_entropy(state.h_adj_norm)

        # Compute j-vector centroid
        j_vector = self.vector_calc.compute_j_centroid(state.adj_counts)
        state.j = j_vector.tolist()

        # Compute goodness
        state.g = self.vector_calc.compute_goodness(j_vector)

        # Update confidence
        state.confidence = self.entropy_calc.confidence_from_observations(
            state.n_observations
        )

        # Update timestamp
        state.last_updated = datetime.now()

    def get_concept_state(self, noun: str) -> Optional[ConceptState]:
        """Get current state of a learned concept."""
        return self._concepts.get(noun)

    def has_concept(self, noun: str) -> bool:
        """Check if concept has been learned."""
        return noun in self._concepts

    def get_all_concepts(self) -> Dict[str, ConceptState]:
        """Get all learned concepts."""
        return self._concepts.copy()

    def get_concepts_by_source(self, source: str) -> List[ConceptState]:
        """Get concepts learned from a specific source."""
        return [s for s in self._concepts.values() if s.learned_from == source]

    def get_stats(self) -> Dict:
        """Get learning statistics."""
        if not self._concepts:
            return {
                "total_concepts": 0,
                "total_observations": 0,
                "avg_variety": 0,
                "avg_tau": 0,
                "sources": {}
            }

        sources = defaultdict(int)
        for c in self._concepts.values():
            sources[c.learned_from] += 1

        return {
            "total_concepts": len(self._concepts),
            "total_observations": sum(c.n_observations for c in self._concepts.values()),
            "avg_variety": sum(c.variety for c in self._concepts.values()) / len(self._concepts),
            "avg_tau": sum(c.tau for c in self._concepts.values()) / len(self._concepts),
            "sources": dict(sources)
        }

    def export_for_neo4j(self) -> List[Dict]:
        """
        Export all concepts as dicts ready for Neo4j batch insert.

        Returns:
            List of concept dictionaries
        """
        return [state.to_dict() for state in self._concepts.values()]

    def clear(self):
        """Clear all learned concepts."""
        self._concepts.clear()
        self._observations.clear()


class Neo4jLearningStore:
    """
    Neo4j storage for adjective observations and learned concepts.

    Schema:
        (:Concept) - semantic state (extended with learning properties)
        (:Adjective) - adjective node for tracking observations
        (:Concept)-[:DESCRIBED_BY {count, source}]->(:Adjective)
        (:Concept)-[:VIA {weight, last_used, last_reinforced}]->(:Concept)

    This allows:
        - Tracking all adj-noun observations in the graph
        - Recomputing τ, g, j from the actual distribution
        - Sharing adjective vectors across concepts
        - Weight decay (forgetting) based on time since last use
    """

    def __init__(self, driver):
        """
        Initialize with Neo4j driver.

        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
        self.entropy_calc = EntropyCalculator()
        self.vector_calc = VectorCalculator()

        # Import weight dynamics
        from .weight_dynamics import (
            W_MIN, W_MAX, LAMBDA_FORGET, DORMANCY_THRESHOLD,
            decay_weight, decay_statistics
        )
        self.W_MIN = W_MIN
        self.W_MAX = W_MAX
        self.LAMBDA_FORGET = LAMBDA_FORGET
        self.DORMANCY_THRESHOLD = DORMANCY_THRESHOLD
        self._decay_weight = decay_weight
        self._decay_statistics = decay_statistics

    def setup_schema(self):
        """Create indexes and constraints for learning."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Adjective node constraint
            session.run("""
                CREATE CONSTRAINT adjective_word IF NOT EXISTS
                FOR (a:Adjective) REQUIRE a.word IS UNIQUE
            """)

            # Index on DESCRIBED_BY for fast aggregation
            session.run("""
                CREATE INDEX described_by_count IF NOT EXISTS
                FOR ()-[r:DESCRIBED_BY]-() ON (r.count)
            """)

            # Index on learned concepts
            session.run("""
                CREATE INDEX concept_learned IF NOT EXISTS
                FOR (c:Concept) ON (c.learned)
            """)

            print("[Learning] Schema setup complete")

    def observe_adjective(self, noun: str, adjective: str,
                          count: int = 1, source: str = "unknown"):
        """
        Record an adjective observation in Neo4j.

        Creates/updates:
        - (:Adjective {word}) node
        - (:Concept)-[:DESCRIBED_BY {count, source}]->(:Adjective)
        """
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MERGE (n:Concept {word: $noun})
                ON CREATE SET n.learned = true, n.pos = 'noun'

                MERGE (a:Adjective {word: $adj})

                MERGE (n)-[r:DESCRIBED_BY]->(a)
                ON CREATE SET
                    r.count = $count,
                    r.source = $source,
                    r.created_at = datetime()
                ON MATCH SET
                    r.count = r.count + $count,
                    r.last_updated = datetime()
            """, noun=noun, adj=adjective, count=count, source=source)

    def observe_batch(self, noun: str, adj_counts: Dict[str, int],
                      source: str = "unknown"):
        """
        Record multiple adjective observations efficiently.

        Args:
            noun: The noun being described
            adj_counts: {adjective: count}
            source: Origin of observations
        """
        if not self.driver or not adj_counts:
            return

        # Convert to list for UNWIND
        observations = [
            {"adj": adj, "count": count}
            for adj, count in adj_counts.items()
        ]

        with self.driver.session() as session:
            session.run("""
                MERGE (n:Concept {word: $noun})
                ON CREATE SET n.learned = true, n.pos = 'noun'

                WITH n
                UNWIND $observations AS obs

                MERGE (a:Adjective {word: obs.adj})

                MERGE (n)-[r:DESCRIBED_BY]->(a)
                ON CREATE SET
                    r.count = obs.count,
                    r.source = $source,
                    r.created_at = datetime()
                ON MATCH SET
                    r.count = r.count + obs.count,
                    r.last_updated = datetime()
            """, noun=noun, observations=observations, source=source)

    def get_adj_distribution(self, noun: str) -> Dict[str, int]:
        """
        Get the full adjective distribution for a noun from Neo4j.

        Returns:
            {adjective: count}
        """
        if not self.driver:
            return {}

        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Concept {word: $noun})-[r:DESCRIBED_BY]->(a:Adjective)
                RETURN a.word as adj, r.count as count
                ORDER BY r.count DESC
            """, noun=noun)

            return {record["adj"]: record["count"] for record in result}

    def update_concept_from_distribution(self, noun: str,
                                         j_vectors: Dict[str, np.ndarray] = None):
        """
        Recompute and update concept's τ, g, j from its adjective distribution.

        This is the key learning operation:
        1. Fetch adjective distribution from Neo4j
        2. Compute entropy → τ
        3. Compute j-centroid → g
        4. Update concept node

        Args:
            noun: Concept to update
            j_vectors: {adjective: 5D j-vector} for centroid computation
        """
        if not self.driver:
            return None

        # Get distribution
        adj_counts = self.get_adj_distribution(noun)
        if not adj_counts:
            return None

        # Compute entropy and τ
        h_adj = self.entropy_calc.shannon_entropy(adj_counts)
        h_adj_norm = self.entropy_calc.normalized_entropy(adj_counts)
        tau = self.entropy_calc.tau_from_entropy(h_adj_norm)

        # Compute j-vector and g
        if j_vectors:
            self.vector_calc.adj_vectors = j_vectors

        j_vector = self.vector_calc.compute_j_centroid(adj_counts)
        g = self.vector_calc.compute_goodness(j_vector)

        # Compute confidence
        n_observations = len(adj_counts)
        confidence = self.entropy_calc.confidence_from_observations(n_observations)

        variety = len(adj_counts)
        total_count = sum(adj_counts.values())

        # Update in Neo4j
        with self.driver.session() as session:
            session.run("""
                MATCH (c:Concept {word: $word})
                SET c.tau = $tau,
                    c.g = $g,
                    c.j = $j,
                    c.h_adj = $h_adj,
                    c.h_adj_norm = $h_adj_norm,
                    c.variety = $variety,
                    c.total_count = $total_count,
                    c.confidence = $confidence,
                    c.learned = true,
                    c.last_computed = datetime()
            """, word=noun, tau=tau, g=g, j=j_vector.tolist(),
                 h_adj=h_adj, h_adj_norm=h_adj_norm,
                 variety=variety, total_count=total_count,
                 confidence=confidence)

        return {
            "word": noun,
            "tau": tau,
            "g": g,
            "j": j_vector.tolist(),
            "variety": variety,
            "h_adj": h_adj,
            "h_adj_norm": h_adj_norm,
            "confidence": confidence
        }

    def learn_concept(self, noun: str, adj_counts: Dict[str, int],
                      source: str = "unknown",
                      j_vectors: Dict[str, np.ndarray] = None) -> Dict:
        """
        Complete learning operation: observe + update.

        This is the main entry point for learning a new concept.

        Args:
            noun: Concept to learn
            adj_counts: {adjective: count}
            source: Origin of observations
            j_vectors: Adjective vectors for centroid computation

        Returns:
            Updated concept properties
        """
        # Store observations
        self.observe_batch(noun, adj_counts, source)

        # Recompute from distribution
        return self.update_concept_from_distribution(noun, j_vectors)

    def get_learned_concepts(self, min_variety: int = 3) -> List[Dict]:
        """
        Get all learned concepts with their properties.

        Args:
            min_variety: Minimum adjective variety to include

        Returns:
            List of concept dictionaries
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE c.learned = true AND c.variety >= $min_variety
                RETURN c.word as word, c.tau as tau, c.g as g,
                       c.j as j, c.variety as variety,
                       c.confidence as confidence
                ORDER BY c.variety DESC
            """, min_variety=min_variety)

            return [dict(record) for record in result]

    def get_stats(self) -> Dict:
        """Get learning statistics from Neo4j."""
        if not self.driver:
            return {"error": "Not connected"}

        with self.driver.session() as session:
            # Learned concepts count
            result = session.run("""
                MATCH (c:Concept)
                WHERE c.learned = true
                RETURN count(c) as learned_concepts,
                       avg(c.variety) as avg_variety,
                       avg(c.tau) as avg_tau,
                       avg(c.confidence) as avg_confidence
            """)
            concept_stats = result.single()

            # Adjective count
            result = session.run("MATCH (a:Adjective) RETURN count(a) as count")
            adj_count = result.single()["count"]

            # Observation count
            result = session.run("""
                MATCH ()-[r:DESCRIBED_BY]->()
                RETURN count(r) as edges, sum(r.count) as total_observations
            """)
            obs_stats = result.single()

            return {
                "learned_concepts": concept_stats["learned_concepts"],
                "adjectives": adj_count,
                "observation_edges": obs_stats["edges"],
                "total_observations": obs_stats["total_observations"],
                "avg_variety": concept_stats["avg_variety"],
                "avg_tau": concept_stats["avg_tau"],
                "avg_confidence": concept_stats["avg_confidence"]
            }

    # =========================================================================
    # WEIGHT DECAY (FORGETTING) METHODS
    # =========================================================================

    def apply_decay(self, days_elapsed: float = 1.0,
                    dry_run: bool = False) -> Dict:
        """
        Apply forgetting decay to all VIA edge weights.

        Formula: w(t+dt) = w_min + (w - w_min) * e^(-lambda * dt)

        This implements the "nightly decay" where edges not recently
        reinforced gradually lose weight toward w_min (0.1).

        Args:
            days_elapsed: Days since last decay (default 1.0 for nightly)
            dry_run: If True, compute but don't apply changes

        Returns:
            Statistics about the decay operation
        """
        if not self.driver:
            return {"error": "Not connected"}

        import math
        decay_factor = math.exp(-self.LAMBDA_FORGET * days_elapsed)

        with self.driver.session() as session:
            if dry_run:
                # Preview what would happen
                result = session.run("""
                    MATCH ()-[r:VIA]->()
                    WHERE r.weight > $w_min
                    WITH r,
                         r.weight as w_before,
                         $w_min + (r.weight - $w_min) * $decay_factor as w_after
                    RETURN count(r) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           sum(w_before - w_after) as total_decay,
                           avg(w_before) as avg_before,
                           avg(w_after) as avg_after,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=self.W_MIN, decay_factor=decay_factor,
                     threshold=self.DORMANCY_THRESHOLD)

                record = result.single()
                return {
                    "dry_run": True,
                    "days_elapsed": days_elapsed,
                    "decay_factor": decay_factor,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "total_decay": record["total_decay"],
                    "avg_weight_before": record["avg_before"],
                    "avg_weight_after": record["avg_after"],
                    "newly_dormant": record["newly_dormant"]
                }
            else:
                # Actually apply the decay
                result = session.run("""
                    MATCH ()-[r:VIA]->()
                    WHERE r.weight > $w_min
                    WITH r,
                         r.weight as w_before,
                         $w_min + (r.weight - $w_min) * $decay_factor as w_after
                    SET r.weight = w_after,
                        r.last_decay = datetime()
                    RETURN count(r) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           sum(w_before - w_after) as total_decay,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=self.W_MIN, decay_factor=decay_factor,
                     threshold=self.DORMANCY_THRESHOLD)

                record = result.single()
                return {
                    "dry_run": False,
                    "days_elapsed": days_elapsed,
                    "decay_factor": decay_factor,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "total_decay": record["total_decay"],
                    "newly_dormant": record["newly_dormant"],
                    "applied_at": datetime.now().isoformat()
                }

    def apply_decay_since_last_use(self, dry_run: bool = False) -> Dict:
        """
        Apply decay based on each edge's individual last_used timestamp.

        More accurate than apply_decay() - decays each edge based on
        how long since it was actually used.

        Args:
            dry_run: If True, compute but don't apply changes

        Returns:
            Statistics about the decay operation
        """
        if not self.driver:
            return {"error": "Not connected"}

        with self.driver.session() as session:
            if dry_run:
                result = session.run("""
                    MATCH ()-[r:VIA]->()
                    WHERE r.weight > $w_min AND r.last_used IS NOT NULL
                    WITH r,
                         r.weight as w_before,
                         duration.inDays(r.last_used, datetime()).days as days_since,
                         $w_min + (r.weight - $w_min) * exp(-$lambda * duration.inDays(r.last_used, datetime()).days) as w_after
                    RETURN count(r) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           avg(days_since) as avg_days_since_use,
                           max(days_since) as max_days_since_use,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=self.W_MIN, lambda_val=self.LAMBDA_FORGET,
                     threshold=self.DORMANCY_THRESHOLD)

                record = result.single()
                if record["edge_count"] == 0:
                    return {
                        "dry_run": True,
                        "edges_with_timestamp": 0,
                        "note": "No edges have last_used timestamp set"
                    }

                return {
                    "dry_run": True,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "avg_days_since_use": record["avg_days_since_use"],
                    "max_days_since_use": record["max_days_since_use"],
                    "newly_dormant": record["newly_dormant"]
                }
            else:
                # Apply decay based on individual timestamps
                result = session.run("""
                    MATCH ()-[r:VIA]->()
                    WHERE r.weight > $w_min AND r.last_used IS NOT NULL
                    WITH r,
                         r.weight as w_before,
                         duration.inDays(r.last_used, datetime()).days as days_since
                    WITH r, w_before, days_since,
                         $w_min + (w_before - $w_min) * exp(-$lambda * days_since) as w_after
                    SET r.weight = w_after,
                        r.last_decay = datetime()
                    RETURN count(r) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=self.W_MIN, lambda_val=self.LAMBDA_FORGET,
                     threshold=self.DORMANCY_THRESHOLD)

                record = result.single()
                return {
                    "dry_run": False,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "newly_dormant": record["newly_dormant"],
                    "applied_at": datetime.now().isoformat()
                }

    def mark_edge_used(self, subject: str, verb: str, obj: str):
        """
        Mark a VIA edge as recently used (updates last_used timestamp).

        Call this during navigation to track edge usage for decay.

        Args:
            subject: Source concept word
            verb: The verb on the edge
            obj: Target concept word
        """
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MATCH (s:Concept {word: $subject})
                      -[r:VIA {verb: $verb}]->
                      (o:Concept {word: $obj})
                SET r.last_used = datetime()
            """, subject=subject, verb=verb, obj=obj)

    def mark_edges_used_batch(self, edges: List[Tuple[str, str, str]]):
        """
        Mark multiple edges as recently used.

        Args:
            edges: List of (subject, verb, object) tuples
        """
        if not self.driver or not edges:
            return

        edge_data = [
            {"subject": s, "verb": v, "object": o}
            for s, v, o in edges
        ]

        with self.driver.session() as session:
            session.run("""
                UNWIND $edges AS e
                MATCH (s:Concept {word: e.subject})
                      -[r:VIA {verb: e.verb}]->
                      (o:Concept {word: e.object})
                SET r.last_used = datetime()
            """, edges=edge_data)

    def get_decay_stats(self) -> Dict:
        """
        Get statistics about edge weights and decay state.

        Returns:
            Dictionary with weight distribution and dormancy stats
        """
        if not self.driver:
            return {"error": "Not connected"}

        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r:VIA]->()
                RETURN count(r) as total_edges,
                       avg(r.weight) as avg_weight,
                       min(r.weight) as min_weight,
                       max(r.weight) as max_weight,
                       sum(CASE WHEN r.weight <= $threshold THEN 1 ELSE 0 END) as dormant_count,
                       sum(CASE WHEN r.weight > $threshold THEN 1 ELSE 0 END) as active_count,
                       sum(CASE WHEN r.weight >= 0.9 THEN 1 ELSE 0 END) as saturated_count,
                       sum(CASE WHEN r.last_used IS NOT NULL THEN 1 ELSE 0 END) as edges_with_timestamp,
                       sum(CASE WHEN r.last_decay IS NOT NULL THEN 1 ELSE 0 END) as edges_with_decay_record
            """, threshold=self.DORMANCY_THRESHOLD)

            record = result.single()
            total = record["total_edges"]

            return {
                "total_edges": total,
                "avg_weight": record["avg_weight"],
                "min_weight": record["min_weight"],
                "max_weight": record["max_weight"],
                "dormant_count": record["dormant_count"],
                "active_count": record["active_count"],
                "saturated_count": record["saturated_count"],
                "dormant_percentage": (record["dormant_count"] / total * 100) if total > 0 else 0,
                "edges_with_timestamp": record["edges_with_timestamp"],
                "edges_with_decay_record": record["edges_with_decay_record"],
                "dormancy_threshold": self.DORMANCY_THRESHOLD
            }

    def get_weight_distribution(self, buckets: int = 10) -> List[Dict]:
        """
        Get histogram of edge weights.

        Args:
            buckets: Number of histogram buckets

        Returns:
            List of {range_start, range_end, count} dicts
        """
        if not self.driver:
            return []

        bucket_size = (self.W_MAX - self.W_MIN) / buckets
        distribution = []

        with self.driver.session() as session:
            for i in range(buckets):
                low = self.W_MIN + i * bucket_size
                high = low + bucket_size

                result = session.run("""
                    MATCH ()-[r:VIA]->()
                    WHERE r.weight >= $low AND r.weight < $high
                    RETURN count(r) as count
                """, low=low, high=high)

                record = result.single()
                distribution.append({
                    "range_start": round(low, 2),
                    "range_end": round(high, 2),
                    "count": record["count"]
                })

        return distribution

    def get_stale_edges(self, days_threshold: int = 30,
                        limit: int = 100) -> List[Dict]:
        """
        Get edges that haven't been used in a while.

        Args:
            days_threshold: Days of inactivity to consider stale
            limit: Maximum edges to return

        Returns:
            List of stale edge information
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Concept)-[r:VIA]->(o:Concept)
                WHERE r.last_used IS NOT NULL
                  AND duration.inDays(r.last_used, datetime()).days >= $days
                RETURN s.word as subject,
                       r.verb as verb,
                       o.word as object,
                       r.weight as weight,
                       duration.inDays(r.last_used, datetime()).days as days_since_use
                ORDER BY days_since_use DESC
                LIMIT $limit
            """, days=days_threshold, limit=limit)

            return [dict(record) for record in result]

    def initialize_last_used_timestamps(self):
        """
        Initialize last_used timestamps for edges that don't have them.

        Sets last_used to created_at if available, otherwise to now.
        This is useful when first enabling timestamp-based decay.
        """
        if not self.driver:
            return {"error": "Not connected"}

        with self.driver.session() as session:
            # Use created_at if available
            result1 = session.run("""
                MATCH ()-[r:VIA]->()
                WHERE r.last_used IS NULL AND r.created_at IS NOT NULL
                SET r.last_used = r.created_at
                RETURN count(r) as updated
            """)
            from_created = result1.single()["updated"]

            # For edges without created_at, use last_reinforced
            result2 = session.run("""
                MATCH ()-[r:VIA]->()
                WHERE r.last_used IS NULL AND r.last_reinforced IS NOT NULL
                SET r.last_used = r.last_reinforced
                RETURN count(r) as updated
            """)
            from_reinforced = result2.single()["updated"]

            # For remaining edges, set to now
            result3 = session.run("""
                MATCH ()-[r:VIA]->()
                WHERE r.last_used IS NULL
                SET r.last_used = datetime()
                RETURN count(r) as updated
            """)
            from_now = result3.single()["updated"]

            return {
                "initialized_from_created_at": from_created,
                "initialized_from_last_reinforced": from_reinforced,
                "initialized_to_now": from_now,
                "total_initialized": from_created + from_reinforced + from_now
            }


class LearningGraphAdapter:
    """
    High-level adapter combining in-memory learning with Neo4j persistence.

    Use this for applications that need both fast in-memory access
    and persistent storage.
    """

    def __init__(self, graph, auto_sync: bool = True):
        """
        Initialize adapter.

        Args:
            graph: MeaningGraph instance
            auto_sync: If True, sync to Neo4j after each observation
        """
        self.graph = graph
        self.auto_sync = auto_sync

        # In-memory learner for fast access
        self.learner = ConceptLearner()

        # Neo4j store for persistence
        self.store = None
        if graph and graph.is_connected():
            self.store = Neo4jLearningStore(graph.driver)

    def setup_schema(self):
        """Setup Neo4j schema for learning."""
        if self.store:
            self.store.setup_schema()

    def observe_adjective(self, noun: str, adjective: str,
                          count: int = 1, source: str = "unknown") -> ConceptState:
        """
        Record an adjective observation.

        Updates both in-memory state and Neo4j.
        """
        # In-memory update
        state = self.learner.observe_adjective(noun, adjective, count, source)

        # Neo4j update (if auto_sync)
        if self.auto_sync and self.store:
            self.store.observe_adjective(noun, adjective, count, source)

        return state

    def observe_batch(self, noun: str, adj_counts: Dict[str, int],
                      source: str = "unknown") -> ConceptState:
        """Record multiple adjective observations."""
        state = self.learner.observe_batch(noun, adj_counts, source)

        if self.auto_sync and self.store:
            self.store.observe_batch(noun, adj_counts, source)

        return state

    def update_concept(self, noun: str,
                       j_vectors: Dict[str, np.ndarray] = None) -> Optional[Dict]:
        """
        Recompute concept parameters from Neo4j distribution.

        This loads the full distribution from Neo4j and recomputes.
        """
        if self.store:
            return self.store.update_concept_from_distribution(noun, j_vectors)
        return None

    def sync_to_neo4j(self, noun: str):
        """
        Force sync a concept's observations to Neo4j.

        Use when auto_sync is False.
        """
        if not self.store:
            return

        state = self.learner.get_concept_state(noun)
        if state:
            self.store.observe_batch(noun, state.adj_counts, state.learned_from)

    def sync_all(self):
        """Force sync all in-memory concepts to Neo4j."""
        if not self.store:
            return

        for noun, state in self.learner.get_all_concepts().items():
            self.store.observe_batch(noun, state.adj_counts, state.learned_from)

    def load_from_neo4j(self, noun: str) -> Optional[ConceptState]:
        """
        Load a concept's distribution from Neo4j into memory.
        """
        if not self.store:
            return None

        adj_counts = self.store.get_adj_distribution(noun)
        if adj_counts:
            return self.learner.observe_batch(noun, adj_counts, "neo4j")
        return None

    def get_stats(self) -> Dict:
        """Get combined stats from memory and Neo4j."""
        memory_stats = self.learner.get_stats()

        if self.store:
            neo4j_stats = self.store.get_stats()
            return {
                "memory": memory_stats,
                "neo4j": neo4j_stats
            }

        return {"memory": memory_stats}


# Convenience functions
def create_learner_with_vectors(data_loader) -> ConceptLearner:
    """
    Create a ConceptLearner with adjective vectors loaded.

    Args:
        data_loader: DataLoader instance from core.data_loader

    Returns:
        ConceptLearner ready to use
    """
    learner = ConceptLearner()
    learner.load_vectors(data_loader)
    return learner


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("ConceptLearner Test")
    print("=" * 60)

    # Create learner
    learner = ConceptLearner()

    # Simulate learning "serendipity"
    print("\n1. Learning 'serendipity' from adjective observations:")

    # First observations
    learner.observe_adjective("serendipity", "happy", count=5, source="book")
    learner.observe_adjective("serendipity", "unexpected", count=3, source="book")
    learner.observe_adjective("serendipity", "fortunate", count=2, source="book")

    state = learner.get_concept_state("serendipity")
    print(f"   After 3 adj types:")
    print(f"   - variety: {state.variety}")
    print(f"   - entropy: {state.h_adj:.3f} (norm: {state.h_adj_norm:.3f})")
    print(f"   - τ: {state.tau:.2f}")
    print(f"   - confidence: {state.confidence:.3f}")

    # More observations - entropy increases, τ decreases
    learner.observe_adjective("serendipity", "wonderful", count=2, source="conversation")
    learner.observe_adjective("serendipity", "rare", count=1, source="conversation")
    learner.observe_adjective("serendipity", "magical", count=1, source="conversation")

    state = learner.get_concept_state("serendipity")
    print(f"\n   After 6 adj types:")
    print(f"   - variety: {state.variety}")
    print(f"   - entropy: {state.h_adj:.3f} (norm: {state.h_adj_norm:.3f})")
    print(f"   - τ: {state.tau:.2f}")
    print(f"   - confidence: {state.confidence:.3f}")

    # Test batch observation
    print("\n2. Batch observation for 'ephemeral':")
    learner.observe_batch("ephemeral", {
        "brief": 10,
        "fleeting": 8,
        "transient": 5,
        "momentary": 3
    }, source="corpus")

    state = learner.get_concept_state("ephemeral")
    print(f"   - variety: {state.variety}")
    print(f"   - entropy: {state.h_adj:.3f} (norm: {state.h_adj_norm:.3f})")
    print(f"   - τ: {state.tau:.2f}")

    # Test with concentrated distribution (low entropy → high τ)
    print("\n3. Concentrated distribution for 'red' (should be high τ):")
    learner.observe_batch("red_thing", {
        "red": 100,
        "bright": 2,
        "dark": 1
    }, source="test")

    state = learner.get_concept_state("red_thing")
    print(f"   - variety: {state.variety}")
    print(f"   - entropy: {state.h_adj:.3f} (norm: {state.h_adj_norm:.3f})")
    print(f"   - τ: {state.tau:.2f} (high = concrete)")

    # Stats
    print("\n4. Learning stats:")
    stats = learner.get_stats()
    print(f"   - Total concepts: {stats['total_concepts']}")
    print(f"   - Total observations: {stats['total_observations']}")
    print(f"   - Avg variety: {stats['avg_variety']:.1f}")
    print(f"   - Avg τ: {stats['avg_tau']:.2f}")
    print(f"   - Sources: {stats['sources']}")

    print("\n" + "=" * 60)
    print("Test complete!")
