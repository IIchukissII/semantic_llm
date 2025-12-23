"""
Neo4j Experience Graph

Store experience as a weighted graph where:
- Nodes = Semantic states (word, g, τ, j)
- Edges = Walked transitions (weighted by frequency)
- Weight = "Breadth of way" (frequently walked = easier to navigate)

Theory coherence:
- Experience IS the graph (personal subgraph of Wholeness)
- "Broad way" = high-weight edges
- Tunneling probability ∝ adjacency to experience
- Navigation confidence ∝ familiarity (visits) + edge weight

"Only believe what was lived is knowledge"
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from neo4j import GraphDatabase

# Import from layers
import sys
_THIS_FILE = Path(__file__).resolve()
_EXPERIENCE_KNOWLEDGE = _THIS_FILE.parent.parent

sys.path.insert(0, str(_EXPERIENCE_KNOWLEDGE))

from layers.core import Wholeness, SemanticState
from layers.dynamics import WeightDynamics, WeightConfig, initial_weight


@dataclass
class GraphConfig:
    """Neo4j connection config."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "experience123"  # From docker-compose.yml
    database: str = "neo4j"  # Default database


class ExperienceGraph:
    """
    Neo4j-backed experience graph.

    Theory-coherent storage of walked paths through semantic space.
    Now with learning/forgetting dynamics.
    """

    def __init__(self, config: GraphConfig = None, wholeness: Wholeness = None,
                 dynamics: WeightDynamics = None):
        self.config = config or GraphConfig()
        self.driver = None
        self.wholeness = wholeness
        self.dynamics = dynamics or WeightDynamics()

        self._connect()

    def _connect(self):
        """Connect to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password)
            )
            self.driver.verify_connectivity()
            print(f"Connected to Neo4j at {self.config.uri}")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            print("Make sure Neo4j is running and credentials are correct.")
            self.driver = None

    def close(self):
        """Close connection."""
        if self.driver:
            self.driver.close()

    def setup_schema(self):
        """Create indexes and constraints."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Unique constraint on word
            session.run("""
                CREATE CONSTRAINT semantic_state_word IF NOT EXISTS
                FOR (s:SemanticState) REQUIRE s.word IS UNIQUE
            """)

            # Index for goodness queries
            session.run("""
                CREATE INDEX semantic_state_goodness IF NOT EXISTS
                FOR (s:SemanticState) ON (s.goodness)
            """)

            # Index for tau queries
            session.run("""
                CREATE INDEX semantic_state_tau IF NOT EXISTS
                FOR (s:SemanticState) ON (s.tau)
            """)

            # Index for last_visited (for decay queries)
            session.run("""
                CREATE INDEX semantic_state_last_visited IF NOT EXISTS
                FOR (s:SemanticState) ON (s.last_visited)
            """)

            # Index for learned_from (corpus vs conversation)
            session.run("""
                CREATE INDEX semantic_state_source IF NOT EXISTS
                FOR (s:SemanticState) ON (s.learned_from)
            """)

            print("Schema setup complete")

    def clear_experience(self):
        """Clear all experience (start fresh)."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Experience cleared")

    def _get_state(self, word: str) -> Optional[SemanticState]:
        """Get semantic state from wholeness."""
        if self.wholeness and word in self.wholeness:
            return self.wholeness.states[word]
        return None

    def read_book(self, filepath: str, book_id: str = None) -> Dict:
        """
        Read a book and store experience in graph.

        Returns statistics about the reading.
        """
        if not self.driver:
            return {"error": "Not connected to Neo4j"}

        if book_id is None:
            book_id = Path(filepath).stem
            if " - " in book_id:
                book_id = book_id.split(" - ", 1)[1]

        print(f"\nReading: {book_id}")

        # Load text
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Skip header/footer
        text = text[len(text)//20:-len(text)//20]

        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        print(f"  Total words: {len(words)}")

        # Filter to semantic words
        semantic_words = [w for w in words if self._get_state(w)]
        print(f"  Semantic words: {len(semantic_words)}")

        # Build transitions
        transitions = {}
        visits = {}

        prev = None
        for word in semantic_words:
            visits[word] = visits.get(word, 0) + 1

            if prev is not None:
                key = (prev, word)
                transitions[key] = transitions.get(key, 0) + 1

            prev = word

        print(f"  Unique states: {len(visits)}")
        print(f"  Unique transitions: {len(transitions)}")

        # Store in Neo4j (batch for efficiency)
        self._store_states(visits, book_id)
        self._store_transitions(transitions, book_id)

        return {
            "book_id": book_id,
            "total_words": len(words),
            "semantic_words": len(semantic_words),
            "unique_states": len(visits),
            "unique_transitions": len(transitions)
        }

    def _store_states(self, visits: Dict[str, int], book_id: str, source: str = "corpus"):
        """Store or update semantic states (batched for performance)."""
        if not self.driver:
            return

        now = datetime.now().isoformat()

        # Prepare batch data
        batch = []
        for word, count in visits.items():
            state = self._get_state(word)
            if not state:
                continue
            batch.append({
                'word': word,
                'goodness': float(state.goodness),
                'tau': float(state.tau),
                'j': state.j.tolist(),
                'count': count,
                'book_id': book_id,
                'now': now,
                'source': source
            })

        # Batch insert using UNWIND
        with self.driver.session() as session:
            session.run("""
                UNWIND $batch AS item
                MERGE (s:SemanticState {word: item.word})
                ON CREATE SET
                    s.goodness = item.goodness,
                    s.tau = item.tau,
                    s.j = item.j,
                    s.visits = item.count,
                    s.books = [item.book_id],
                    s.created_at = item.now,
                    s.last_visited = item.now,
                    s.learned_from = item.source
                ON MATCH SET
                    s.visits = s.visits + item.count,
                    s.last_visited = item.now,
                    s.books = CASE
                        WHEN NOT item.book_id IN s.books
                        THEN s.books + item.book_id
                        ELSE s.books
                    END
            """, batch=batch)

    def _store_transitions(self, transitions: Dict[Tuple[str, str], int], book_id: str):
        """Store or update transitions (edges) - batched for performance."""
        if not self.driver:
            return

        now = datetime.now().isoformat()
        # Initial weight for corpus-derived transitions
        init_weight = self.dynamics.config.w_max

        # Prepare batch data
        batch = [
            {'from_word': f, 'to_word': t, 'count': c, 'book_id': book_id,
             'now': now, 'init_weight': init_weight}
            for (f, t), c in transitions.items()
        ]

        # Batch insert using UNWIND (in chunks to avoid memory issues)
        chunk_size = 5000
        with self.driver.session() as session:
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i+chunk_size]
                session.run("""
                    UNWIND $batch AS item
                    MATCH (a:SemanticState {word: item.from_word})
                    MATCH (b:SemanticState {word: item.to_word})
                    MERGE (a)-[t:TRANSITION]->(b)
                    ON CREATE SET
                        t.weight = item.init_weight,
                        t.raw_count = item.count,
                        t.books = [item.book_id],
                        t.created_at = item.now,
                        t.last_updated = item.now
                    ON MATCH SET
                        t.raw_count = coalesce(t.raw_count, 0) + item.count,
                        t.last_updated = item.now,
                        t.books = CASE
                            WHEN NOT item.book_id IN t.books
                            THEN t.books + item.book_id
                            ELSE t.books
                        END
                """, batch=chunk)

    def get_stats(self) -> Dict:
        """Get experience graph statistics."""
        if not self.driver:
            return {"error": "Not connected"}

        with self.driver.session() as session:
            # Node count
            result = session.run("MATCH (s:SemanticState) RETURN count(s) as count")
            node_count = result.single()["count"]

            # Edge count
            result = session.run("MATCH ()-[t:TRANSITION]->() RETURN count(t) as count")
            edge_count = result.single()["count"]

            # Total visits
            result = session.run("MATCH (s:SemanticState) RETURN sum(s.visits) as total")
            total_visits = result.single()["total"] or 0

            # Total weight
            result = session.run("MATCH ()-[t:TRANSITION]->() RETURN sum(t.weight) as total")
            total_weight = result.single()["total"] or 0

            # Books
            result = session.run("""
                MATCH (s:SemanticState)
                UNWIND s.books as book
                RETURN DISTINCT book
            """)
            books = [r["book"] for r in result]

            return {
                "nodes": node_count,
                "edges": edge_count,
                "total_visits": total_visits,
                "total_transitions": total_weight,
                "books": books
            }

    # =========================================================================
    # Theory-coherent queries
    # =========================================================================

    def knows(self, word: str) -> bool:
        """Have I been to this state? (Experience check)"""
        if not self.driver:
            return False

        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState {word: $word})
                RETURN s.visits as visits
            """, word=word)
            record = result.single()
            return record is not None and record["visits"] > 0

    def familiarity(self, word: str) -> float:
        """How familiar am I with this state? [0, 1]"""
        if not self.driver:
            return 0.0

        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState {word: $word})
                RETURN s.visits as visits
            """, word=word)
            record = result.single()
            if not record:
                return 0.0

            visits = record["visits"]
            # Log scale familiarity, cap at 1.0
            return min(1.0, np.log1p(visits) / np.log1p(100))

    def has_walked(self, from_word: str, to_word: str) -> Tuple[bool, int]:
        """Have I walked this transition? Returns (walked, weight)."""
        if not self.driver:
            return False, 0

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:SemanticState {word: $from_word})
                      -[t:TRANSITION]->
                      (b:SemanticState {word: $to_word})
                RETURN t.weight as weight
            """, from_word=from_word, to_word=to_word)
            record = result.single()
            if record:
                return True, record["weight"]
            return False, 0

    def can_tunnel(self, target: str, believe: float = 0.5) -> Tuple[bool, float]:
        """
        Can I tunnel to this target?

        Theory: Can only tunnel to states connected to experience.
        Probability based on:
        - Direct experience (been there) → high probability
        - Adjacent to experience → lower probability
        - Not connected → cannot tunnel
        """
        if not self.driver:
            return False, 0.0

        # Check if target is in wholeness
        if not self._get_state(target):
            return False, 0.0

        # Direct experience
        if self.knows(target):
            fam = self.familiarity(target)
            p = 0.5 + 0.5 * fam
            return True, p * believe

        # Check adjacency to experience
        with self.driver.session() as session:
            # How many of my known states are neighbors of target in wholeness?
            # (This requires wholeness neighbor info)
            if self.wholeness:
                target_neighbors = self.wholeness.neighbors(target)
                if not target_neighbors:
                    target_neighbors = set(self.wholeness.semantic_neighbors(target, 0.4))

                if target_neighbors:
                    # Count how many neighbors I know
                    result = session.run("""
                        MATCH (s:SemanticState)
                        WHERE s.word IN $neighbors
                        RETURN count(s) as known_count
                    """, neighbors=list(target_neighbors))
                    record = result.single()
                    known_count = record["known_count"] if record else 0

                    if known_count > 0:
                        adjacency = known_count / len(target_neighbors)
                        p = adjacency * believe * 0.3
                        return True, p

        return False, 0.0

    def navigation_confidence(self, from_word: str, to_word: str) -> float:
        """
        How confident am I navigating from A to B?

        Theory: Confidence based on:
        - Have walked this exact path → high (0.8-1.0)
        - Know both endpoints → medium (0.4-0.7)
        - Know only start → low (0.1)

        Weight of edge increases confidence ("broad way").
        """
        if not self.driver:
            return 0.0

        if not self.knows(from_word):
            return 0.0

        # Check if walked this transition
        walked, weight = self.has_walked(from_word, to_word)

        if walked:
            # Broad way: higher weight = higher confidence
            weight_bonus = min(0.2, np.log1p(weight) / np.log1p(100) * 0.2)
            return 0.8 + weight_bonus

        if self.knows(to_word):
            fam = self.familiarity(to_word)
            return 0.4 + 0.3 * fam

        return 0.1

    def find_path(self, from_word: str, to_word: str,
                  max_length: int = 5) -> List[Tuple[str, float]]:
        """
        Find path through experience from A to B.

        Uses weighted shortest path (lower weight = less traveled = harder).
        Returns [(word, confidence), ...] or empty if no path.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            # Use inverse weight as cost (less traveled = higher cost)
            result = session.run("""
                MATCH (start:SemanticState {word: $from_word}),
                      (end:SemanticState {word: $to_word}),
                      path = shortestPath((start)-[:TRANSITION*1..""" + str(max_length) + """]->(end))
                RETURN [n in nodes(path) | n.word] as words,
                       [r in relationships(path) | r.weight] as weights
            """, from_word=from_word, to_word=to_word)

            record = result.single()
            if not record:
                return []

            words = record["words"]
            weights = record["weights"]

            # Build path with confidence at each step
            path = []
            for i, word in enumerate(words):
                if i < len(weights):
                    # Confidence based on edge weight
                    w = weights[i]
                    conf = 0.5 + 0.5 * min(1.0, np.log1p(w) / np.log1p(100))
                else:
                    conf = 1.0  # Destination
                path.append((word, conf))

            return path

    def suggest_next(self, current: str, goal: str = "good",
                     top_k: int = 5) -> List[Tuple[str, float, int]]:
        """
        Suggest next steps from current position.

        Returns [(word, delta_g, weight), ...] sorted by:
        - Direction toward goal (good: higher g, evil: lower g)
        - Edge weight (prefer broad ways)
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (current:SemanticState {word: $current})
                      -[t:TRANSITION]->
                      (next:SemanticState)
                RETURN next.word as word,
                       next.goodness as g,
                       current.goodness as current_g,
                       t.weight as weight,
                       next.tau as tau
                ORDER BY t.weight DESC
                LIMIT 50
            """, current=current)

            suggestions = []
            for record in result:
                word = record["word"]
                g = record["g"]
                current_g = record["current_g"]
                weight = record["weight"]
                tau = record["tau"]

                # Skip function words (low tau)
                if tau < 1.5 or len(word) < 4:
                    continue

                delta_g = g - current_g

                # Score: direction toward goal + weight bonus
                if goal == "good":
                    direction_score = delta_g
                else:
                    direction_score = -delta_g

                weight_score = np.log1p(weight) * 0.1
                score = direction_score + weight_score

                suggestions.append((word, score, weight))

            # Sort by score
            suggestions.sort(key=lambda x: -x[1])
            return suggestions[:top_k]

    def get_broad_ways(self, min_weight: int = 10, limit: int = 20) -> List[Dict]:
        """
        Find the "broad ways" - most frequently walked paths.

        These are the well-trodden routes through semantic space.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:SemanticState)-[t:TRANSITION]->(b:SemanticState)
                WHERE t.weight >= $min_weight
                RETURN a.word as from_word,
                       b.word as to_word,
                       t.weight as weight,
                       a.goodness as from_g,
                       b.goodness as to_g,
                       b.goodness - a.goodness as delta_g
                ORDER BY t.weight DESC
                LIMIT $limit
            """, min_weight=min_weight, limit=limit)

            ways = []
            for record in result:
                ways.append({
                    "from": record["from_word"],
                    "to": record["to_word"],
                    "weight": record["weight"],
                    "from_g": record["from_g"],
                    "to_g": record["to_g"],
                    "delta_g": record["delta_g"]
                })

            return ways

    def get_most_visited(self, limit: int = 20) -> List[Dict]:
        """Get most visited states."""
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState)
                WHERE s.tau > 1.5 AND size(s.word) >= 4
                RETURN s.word as word,
                       s.visits as visits,
                       s.goodness as g,
                       s.tau as tau,
                       s.books as books
                ORDER BY s.visits DESC
                LIMIT $limit
            """, limit=limit)

            return [dict(record) for record in result]

    # =========================================================================
    # Learning and Forgetting Dynamics
    # =========================================================================

    def reinforce_transition(self, from_word: str, to_word: str,
                             reinforcements: int = 1) -> Optional[float]:
        """
        Apply learning to a transition (path walked).

        Uses dynamics formula: w → w_max exponentially.

        Args:
            from_word: Source state
            to_word: Target state
            reinforcements: Number of reinforcement events

        Returns:
            New weight after learning, or None if edge doesn't exist
        """
        if not self.driver:
            return None

        now = datetime.now().isoformat()

        with self.driver.session() as session:
            # Get current weight
            result = session.run("""
                MATCH (a:SemanticState {word: $from_word})
                      -[t:TRANSITION]->
                      (b:SemanticState {word: $to_word})
                RETURN t.weight as weight
            """, from_word=from_word, to_word=to_word)

            record = result.single()
            if not record:
                return None

            current_weight = record["weight"] or self.dynamics.config.w_min
            new_weight = self.dynamics.learn(current_weight, reinforcements)

            # Update weight and timestamp
            session.run("""
                MATCH (a:SemanticState {word: $from_word})
                      -[t:TRANSITION]->
                      (b:SemanticState {word: $to_word})
                SET t.weight = $new_weight,
                    t.last_updated = $now,
                    t.reinforcements = coalesce(t.reinforcements, 0) + $count
            """, from_word=from_word, to_word=to_word,
                 new_weight=new_weight, now=now, count=reinforcements)

            return new_weight

    def apply_decay(self, max_age_days: int = 30) -> Dict:
        """
        Apply forgetting to all edges not updated recently.

        Called during Sleep to decay unused paths.

        Args:
            max_age_days: Only decay edges older than this (optimization)

        Returns:
            Statistics about decay applied
        """
        if not self.driver:
            return {"error": "Not connected"}

        now = datetime.now()
        stats = {"edges_processed": 0, "edges_decayed": 0, "total_decay": 0.0}

        with self.driver.session() as session:
            # Get edges that need decay
            result = session.run("""
                MATCH (a:SemanticState)-[t:TRANSITION]->(b:SemanticState)
                WHERE t.last_updated IS NOT NULL
                RETURN a.word as from_word,
                       b.word as to_word,
                       t.weight as weight,
                       t.last_updated as last_updated
            """)

            updates = []
            for record in result:
                stats["edges_processed"] += 1

                last_updated_str = record["last_updated"]
                if not last_updated_str:
                    continue

                # Parse timestamp
                try:
                    last_updated = datetime.fromisoformat(last_updated_str)
                except:
                    continue

                # Calculate days elapsed
                days_elapsed = (now - last_updated).total_seconds() / 86400

                # Skip if recently updated
                if days_elapsed < 1:
                    continue

                current_weight = record["weight"] or self.dynamics.config.w_max
                new_weight = self.dynamics.forget(current_weight, days_elapsed)

                # Only update if weight actually changed
                if abs(new_weight - current_weight) > 0.01:
                    decay_amount = current_weight - new_weight
                    stats["edges_decayed"] += 1
                    stats["total_decay"] += decay_amount

                    updates.append({
                        "from_word": record["from_word"],
                        "to_word": record["to_word"],
                        "new_weight": new_weight
                    })

            # Batch update decayed weights
            now_str = now.isoformat()
            for update in updates:
                session.run("""
                    MATCH (a:SemanticState {word: $from_word})
                          -[t:TRANSITION]->
                          (b:SemanticState {word: $to_word})
                    SET t.weight = $new_weight,
                        t.decay_applied = $now
                """, from_word=update["from_word"], to_word=update["to_word"],
                     new_weight=update["new_weight"], now=now_str)

        return stats

    def learn_new_word(self, word: str, context_words: List[str],
                       source: str = "conversation") -> Optional[Dict]:
        """
        Learn a new word from context.

        Estimates τ and g from surrounding known words.
        Creates node with low initial weight (needs reinforcement).

        Args:
            word: New word to learn
            context_words: Known words appearing near this word
            source: "conversation" | "context"

        Returns:
            Dict with created state info, or None if failed
        """
        if not self.driver:
            return None

        # Check if word already exists
        if self.knows(word):
            return {"exists": True, "word": word}

        # Filter to known context words
        known_context = [w for w in context_words if self.knows(w)]

        if not known_context:
            return None  # Cannot learn without context

        # Gather properties from context
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState)
                WHERE s.word IN $words
                RETURN avg(s.tau) as avg_tau,
                       avg(s.goodness) as avg_g,
                       collect(s.j) as j_vectors
            """, words=known_context)

            record = result.single()
            if not record:
                return None

            # Estimate properties from context
            estimated_tau = record["avg_tau"] or 3.0
            estimated_g = record["avg_g"] or 0.0

            # Average j-vectors if available
            j_vectors = record["j_vectors"]
            if j_vectors and len(j_vectors) > 0:
                estimated_j = np.mean([np.array(j) for j in j_vectors if j], axis=0).tolist()
            else:
                estimated_j = [0.0] * 5  # Default j-vector

            # Create new state with low initial weight
            now = datetime.now().isoformat()
            init_weight = self.dynamics.initial_weight(source)

            session.run("""
                CREATE (s:SemanticState {
                    word: $word,
                    tau: $tau,
                    goodness: $g,
                    j: $j,
                    visits: 1,
                    books: [],
                    created_at: $now,
                    last_visited: $now,
                    learned_from: $source,
                    confidence: $confidence
                })
            """, word=word, tau=estimated_tau, g=estimated_g, j=estimated_j,
                 now=now, source=source, confidence=0.1)

            # Create transitions from context words
            for ctx_word in known_context[:5]:  # Limit connections
                session.run("""
                    MATCH (a:SemanticState {word: $ctx_word})
                    MATCH (b:SemanticState {word: $word})
                    MERGE (a)-[t:TRANSITION]->(b)
                    ON CREATE SET
                        t.weight = $init_weight,
                        t.created_at = $now,
                        t.last_updated = $now,
                        t.source = 'context'
                    MERGE (b)-[t2:TRANSITION]->(a)
                    ON CREATE SET
                        t2.weight = $init_weight,
                        t2.created_at = $now,
                        t2.last_updated = $now,
                        t2.source = 'context'
                """, ctx_word=ctx_word, word=word, init_weight=init_weight, now=now)

            return {
                "word": word,
                "tau": estimated_tau,
                "goodness": estimated_g,
                "source": source,
                "context": known_context,
                "confidence": 0.1,
                "created": True
            }

    def get_dormant_edges(self, threshold: float = None) -> List[Dict]:
        """
        Get edges that have decayed to dormant state.

        Dormant = weight below threshold, exists but not active in navigation.
        """
        if not self.driver:
            return []

        threshold = threshold or (2 * self.dynamics.config.w_min)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:SemanticState)-[t:TRANSITION]->(b:SemanticState)
                WHERE t.weight <= $threshold
                RETURN a.word as from_word,
                       b.word as to_word,
                       t.weight as weight,
                       t.last_updated as last_updated
                ORDER BY t.weight ASC
                LIMIT 100
            """, threshold=threshold)

            return [dict(record) for record in result]

    def migrate_weights(self) -> Dict:
        """
        Migrate existing raw count weights to normalized [w_min, w_max] scale.

        Uses log scaling: normalized = w_min + (w_max - w_min) * log(1+count) / log(1+max_count)

        Returns:
            Statistics about the migration
        """
        if not self.driver:
            return {"error": "Not connected"}

        stats = {"edges_migrated": 0, "max_old_weight": 0, "min_old_weight": float('inf')}
        now = datetime.now().isoformat()

        with self.driver.session() as session:
            # First, find the max weight to use as reference
            result = session.run("""
                MATCH ()-[t:TRANSITION]->()
                WHERE t.weight IS NOT NULL
                RETURN max(t.weight) as max_w, min(t.weight) as min_w, count(t) as total
            """)
            record = result.single()

            if not record or not record["max_w"]:
                return {"edges_migrated": 0, "message": "No edges to migrate"}

            max_old = float(record["max_w"])
            min_old = float(record["min_w"])
            total = record["total"]

            stats["max_old_weight"] = max_old
            stats["min_old_weight"] = min_old
            stats["total_edges"] = total

            # Check if already migrated (weights in [0, 2] range suggest normalized)
            if max_old <= 2.0:
                return {"edges_migrated": 0, "message": "Weights appear already normalized"}

            # Migrate using log scaling
            w_min = self.dynamics.config.w_min
            w_max = self.dynamics.config.w_max
            log_max = np.log1p(max_old)

            # Batch update all edges
            result = session.run("""
                MATCH ()-[t:TRANSITION]->()
                WHERE t.weight IS NOT NULL AND t.weight > 2.0
                RETURN id(t) as edge_id, t.weight as old_weight
            """)

            updates = []
            for record in result:
                edge_id = record["edge_id"]
                old_weight = float(record["old_weight"])

                # Log-scale normalization
                normalized = w_min + (w_max - w_min) * np.log1p(old_weight) / log_max
                normalized = max(w_min, min(w_max, normalized))  # Clamp

                updates.append({
                    "edge_id": edge_id,
                    "old_weight": old_weight,
                    "new_weight": normalized
                })

            # Apply updates
            for update in updates:
                session.run("""
                    MATCH ()-[t:TRANSITION]->()
                    WHERE id(t) = $edge_id
                    SET t.weight = $new_weight,
                        t.raw_count = $old_weight,
                        t.last_updated = $now,
                        t.migrated = true
                """, edge_id=update["edge_id"], new_weight=update["new_weight"],
                     old_weight=update["old_weight"], now=now)

            stats["edges_migrated"] = len(updates)

            # Show sample of migrations
            if updates:
                sample = updates[:5]
                stats["sample"] = [(u["old_weight"], u["new_weight"]) for u in sample]

        return stats


def load_library(graph: ExperienceGraph, book_dir: str, limit: int = None):
    """Load multiple books into experience graph."""
    book_path = Path(book_dir)
    books = sorted(book_path.glob("*.txt"))

    if limit:
        books = books[:limit]

    print(f"\nLoading {len(books)} books...")
    print("=" * 60)

    stats_all = []
    for book in books:
        try:
            stats = graph.read_book(str(book))
            stats_all.append(stats)
        except Exception as e:
            print(f"  Error reading {book.name}: {e}")

    print("\n" + "=" * 60)
    print("LIBRARY LOADED")
    print("=" * 60)

    total_stats = graph.get_stats()
    print(f"\nTotal experience:")
    print(f"  States: {total_stats['nodes']}")
    print(f"  Transitions: {total_stats['edges']}")
    print(f"  Total visits: {total_stats['total_visits']}")
    print(f"  Books: {len(total_stats['books'])}")

    return stats_all


def demo():
    """Demonstrate the experience graph."""
    print("=" * 70)
    print("EXPERIENCE GRAPH DEMO")
    print("=" * 70)

    # Load wholeness
    wholeness = Wholeness()

    # Create graph (uses defaults from docker-compose)
    config = GraphConfig()

    graph = ExperienceGraph(config, wholeness)

    if not graph.driver:
        print("\nCannot connect to Neo4j. Please ensure:")
        print("  1. Neo4j is running")
        print("  2. Credentials are correct")
        print("  3. Database 'experience' exists")
        return

    # Setup schema
    graph.setup_schema()

    # Clear and reload
    graph.clear_experience()

    # Read some books
    gutenberg = Path("/home/chukiss/text_project/data/gutenberg")
    books = [
        "Alighieri, Dante - The Divine Comedy.txt",
        "Dostoevsky, Fyodor - Crime and Punishment.txt",
        "Conrad, Joseph - Heart of Darkness.txt",
        "Bronte, Charlotte - Jane Eyre.txt",
    ]

    for book in books:
        path = gutenberg / book
        if path.exists():
            graph.read_book(str(path))

    # Show stats
    print("\n" + "-" * 70)
    print("EXPERIENCE STATISTICS")
    print("-" * 70)

    stats = graph.get_stats()
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Total visits: {stats['total_visits']}")
    print(f"  Books: {stats['books']}")

    # Show broad ways
    print("\n" + "-" * 70)
    print("BROAD WAYS (most traveled paths)")
    print("-" * 70)

    ways = graph.get_broad_ways(min_weight=50, limit=10)
    for way in ways:
        print(f"  {way['from']:15} → {way['to']:15} (w={way['weight']:4}, Δg={way['delta_g']:+.2f})")

    # Show most visited
    print("\n" + "-" * 70)
    print("MOST VISITED STATES")
    print("-" * 70)

    visited = graph.get_most_visited(limit=10)
    for v in visited:
        print(f"  {v['word']:15} visits={v['visits']:<5} g={v['g']:+.2f}")

    # Test navigation
    print("\n" + "-" * 70)
    print("NAVIGATION TEST")
    print("-" * 70)

    tests = [
        ("darkness", "light"),
        ("fear", "hope"),
        ("sin", "redemption"),
        ("love", "hate"),
    ]

    for from_w, to_w in tests:
        conf = graph.navigation_confidence(from_w, to_w)
        path = graph.find_path(from_w, to_w)
        path_str = " → ".join([p[0] for p in path]) if path else "No path"
        print(f"  {from_w} → {to_w}: conf={conf:.2f}, path={path_str[:50]}")

    # Test suggestions
    print("\n" + "-" * 70)
    print("SUGGESTIONS FROM 'darkness' (toward good)")
    print("-" * 70)

    suggestions = graph.suggest_next("darkness", "good")
    for word, score, weight in suggestions:
        print(f"  → {word} (score={score:+.2f}, weight={weight})")

    graph.close()


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Experience Graph - Neo4j backend")
    parser.add_argument("command", choices=["demo", "load", "stats", "paths", "clear"],
                       help="Command to run")
    parser.add_argument("--books", type=int, default=None,
                       help="Number of books to load (default: all)")
    parser.add_argument("--book-dir", default="/home/chukiss/text_project/data/gutenberg",
                       help="Directory with books")

    args = parser.parse_args()

    # Load wholeness
    print("Loading semantic space...")
    wholeness = Wholeness()

    # Connect to Neo4j
    config = GraphConfig()
    graph = ExperienceGraph(config, wholeness)

    if not graph.driver:
        print("\nNeo4j not available. Start with:")
        print("  docker-compose up -d")
        return

    if args.command == "demo":
        graph.close()
        demo()

    elif args.command == "load":
        # Load full library
        graph.setup_schema()
        load_library(graph, args.book_dir, limit=args.books)
        graph.close()

    elif args.command == "stats":
        stats = graph.get_stats()
        print("\n" + "=" * 60)
        print("EXPERIENCE GRAPH STATISTICS")
        print("=" * 60)
        print(f"  States: {stats['nodes']}")
        print(f"  Transitions: {stats['edges']}")
        print(f"  Total visits: {stats['total_visits']}")
        print(f"  Total transitions walked: {stats['total_transitions']}")
        print(f"  Books: {len(stats['books'])}")
        for book in stats['books']:
            print(f"    - {book}")

        print("\n--- Most Visited States ---")
        for v in graph.get_most_visited(10):
            print(f"  {v['word']:15} visits={v['visits']:<5} g={v['g']:+.2f}")

        print("\n--- Broad Ways ---")
        for way in graph.get_broad_ways(min_weight=20, limit=10):
            print(f"  {way['from']:12} → {way['to']:12} (w={way['weight']:4}, Δg={way['delta_g']:+.2f})")

        graph.close()

    elif args.command == "paths":
        # Interactive path finding
        print("\nPath finder (type 'quit' to exit)")
        while True:
            try:
                query = input("\nFrom → To: ").strip()
                if query.lower() == 'quit':
                    break
                parts = query.split()
                if len(parts) >= 2:
                    from_w, to_w = parts[0], parts[-1]
                    path = graph.find_path(from_w, to_w)
                    if path:
                        print(f"  Path: {' → '.join([p[0] for p in path])}")
                        conf = graph.navigation_confidence(from_w, to_w)
                        print(f"  Confidence: {conf:.2f}")
                    else:
                        print("  No path found")
            except (EOFError, KeyboardInterrupt):
                break
        graph.close()

    elif args.command == "clear":
        confirm = input("Clear all experience? (yes/no): ")
        if confirm.lower() == "yes":
            graph.clear_experience()
        graph.close()


if __name__ == "__main__":
    main()
