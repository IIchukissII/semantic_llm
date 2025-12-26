"""
MeaningGraph - Neo4j backend for intent-driven semantic navigation.

Separate from main experience_knowledge database.
Uses verb-mediated transitions (VIA relationships) for intent collapse.

Schema:
    (:Concept) -[:VIA {verb}]-> (:Concept)
    (:VerbOperator) -[:OPERATES_ON]-> (:Concept)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from neo4j import GraphDatabase


@dataclass
class GraphConfig:
    """Neo4j connection config for meaning_chain."""
    uri: str = "bolt://localhost:7688"  # Different port from main!
    user: str = "neo4j"
    password: str = "meaning123"
    database: str = "neo4j"


class MeaningGraph:
    """
    Neo4j graph for verb-mediated semantic navigation.

    Key differences from experience_knowledge:
    - Explicit verbs on edges (VIA relationship)
    - VerbOperator nodes with j-vectors
    - Intent-driven query methods
    """

    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()
        self.driver = None
        self._connect()

    def _connect(self):
        """Connect to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password)
            )
            self.driver.verify_connectivity()
            print(f"[MeaningGraph] Connected to {self.config.uri}")
        except Exception as e:
            print(f"[MeaningGraph] Connection failed: {e}")
            print("  Start with: docker-compose -f config/docker-compose.yml up -d")
            self.driver = None

    def close(self):
        """Close connection."""
        if self.driver:
            self.driver.close()

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.driver is not None

    # =========================================================================
    # Schema Setup
    # =========================================================================

    def setup_schema(self):
        """Create indexes and constraints."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Concept constraints
            session.run("""
                CREATE CONSTRAINT concept_word IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.word IS UNIQUE
            """)

            # VerbOperator constraints
            session.run("""
                CREATE CONSTRAINT verb_operator IF NOT EXISTS
                FOR (v:VerbOperator) REQUIRE v.verb IS UNIQUE
            """)

            # Indexes for queries
            session.run("""
                CREATE INDEX concept_g IF NOT EXISTS
                FOR (c:Concept) ON (c.g)
            """)
            session.run("""
                CREATE INDEX concept_tau IF NOT EXISTS
                FOR (c:Concept) ON (c.tau)
            """)

            print("[MeaningGraph] Schema setup complete")

    def clear_all(self):
        """Clear entire graph (use with caution)."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("[MeaningGraph] Graph cleared")

    # =========================================================================
    # Node Creation
    # =========================================================================

    def create_concept(self, word: str, g: float, tau: float,
                       j: List[float], pos: str = "noun"):
        """Create a Concept node."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MERGE (c:Concept {word: $word})
                ON CREATE SET
                    c.g = $g,
                    c.tau = $tau,
                    c.j = $j,
                    c.pos = $pos
                ON MATCH SET
                    c.g = $g,
                    c.tau = $tau,
                    c.j = $j,
                    c.pos = $pos
            """, word=word, g=g, tau=tau, j=j, pos=pos)

    def create_concept_batch(self, concepts: List[Dict]):
        """Create multiple Concept nodes efficiently."""
        if not self.driver or not concepts:
            return

        with self.driver.session() as session:
            session.run("""
                UNWIND $batch AS item
                MERGE (c:Concept {word: item.word})
                ON CREATE SET
                    c.g = item.g,
                    c.tau = item.tau,
                    c.j = item.j,
                    c.pos = item.pos
            """, batch=concepts)

    def create_verb_operator(self, verb: str, j: List[float],
                             magnitude: float, objects: List[str] = None):
        """Create a VerbOperator node."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MERGE (v:VerbOperator {verb: $verb})
                ON CREATE SET
                    v.j = $j,
                    v.magnitude = $magnitude,
                    v.objects = $objects
                ON MATCH SET
                    v.j = $j,
                    v.magnitude = $magnitude,
                    v.objects = $objects
            """, verb=verb, j=j, magnitude=magnitude, objects=objects or [])

    def create_verb_operator_batch(self, verbs: List[Dict]):
        """Create multiple VerbOperator nodes efficiently."""
        if not self.driver or not verbs:
            return

        with self.driver.session() as session:
            session.run("""
                UNWIND $batch AS item
                MERGE (v:VerbOperator {verb: item.verb})
                ON CREATE SET
                    v.j = item.j,
                    v.magnitude = item.magnitude,
                    v.objects = item.objects
            """, batch=verbs)

    # =========================================================================
    # Relationship Creation
    # =========================================================================

    def create_via(self, subject: str, object_word: str, verb: str,
                   weight: float = 1.0, count: int = 1, source: str = "svo"):
        """Create a VIA relationship (verb-mediated transition)."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MATCH (s:Concept {word: $subject})
                MATCH (o:Concept {word: $object})
                MERGE (s)-[r:VIA {verb: $verb}]->(o)
                ON CREATE SET
                    r.weight = $weight,
                    r.count = $count,
                    r.source = $source
                ON MATCH SET
                    r.count = r.count + $count,
                    r.weight = CASE
                        WHEN r.weight < $weight THEN $weight
                        ELSE r.weight
                    END
            """, subject=subject, object=object_word, verb=verb,
                 weight=weight, count=count, source=source)

    def create_via_batch(self, transitions: List[Dict]):
        """Create multiple VIA relationships efficiently."""
        if not self.driver or not transitions:
            return

        # Process in chunks
        chunk_size = 5000
        with self.driver.session() as session:
            for i in range(0, len(transitions), chunk_size):
                chunk = transitions[i:i+chunk_size]
                session.run("""
                    UNWIND $batch AS item
                    MATCH (s:Concept {word: item.subject})
                    MATCH (o:Concept {word: item.object})
                    MERGE (s)-[r:VIA {verb: item.verb}]->(o)
                    ON CREATE SET
                        r.weight = item.weight,
                        r.count = item.count,
                        r.source = item.source
                    ON MATCH SET
                        r.count = r.count + item.count
                """, batch=chunk)

    def create_operates_on(self, verb: str, concept: str, weight: float = 1.0):
        """Create OPERATES_ON relationship (verb -> concept it acts upon)."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MATCH (v:VerbOperator {verb: $verb})
                MATCH (c:Concept {word: $concept})
                MERGE (v)-[r:OPERATES_ON]->(c)
                ON CREATE SET r.weight = $weight
            """, verb=verb, concept=concept, weight=weight)

    def create_operates_on_batch(self, relations: List[Dict]):
        """Create multiple OPERATES_ON relationships efficiently."""
        if not self.driver or not relations:
            return

        with self.driver.session() as session:
            session.run("""
                UNWIND $batch AS item
                MATCH (v:VerbOperator {verb: item.verb})
                MATCH (c:Concept {word: item.concept})
                MERGE (v)-[r:OPERATES_ON]->(c)
                ON CREATE SET r.weight = item.weight
            """, batch=relations)

    # =========================================================================
    # Intent-Driven Queries
    # =========================================================================

    def get_verb_targets(self, verbs: List[str]) -> Set[str]:
        """
        Get all concepts that intent verbs operate on.

        This defines the "intent space" - concepts aligned with user's verbs.
        """
        if not self.driver or not verbs:
            return set()

        with self.driver.session() as session:
            result = session.run("""
                MATCH (v:VerbOperator)-[:OPERATES_ON]->(c:Concept)
                WHERE v.verb IN $verbs
                RETURN DISTINCT c.word as word
            """, verbs=verbs)

            return {record["word"] for record in result}

    def get_intent_transitions(self, word: str, intent_verbs: Set[str],
                                intent_targets: Set[str],
                                limit: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get transitions from word, collapsed by intent.

        Returns transitions where:
        1. The verb matches an intent verb, OR
        2. The target is in intent_targets (what intent verbs operate on)

        Returns: [(verb, target_word, score), ...]
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            # Query with intent filtering
            result = session.run("""
                MATCH (source:Concept {word: $word})
                      -[r:VIA]->
                      (target:Concept)
                WHERE target.tau > 1.5
                  AND size(target.word) >= 3
                WITH r, target,
                     CASE WHEN r.verb IN $intent_verbs THEN 1.0 ELSE 0.0 END as verb_match,
                     CASE WHEN target.word IN $intent_targets THEN 0.8 ELSE 0.0 END as target_match
                WHERE verb_match > 0 OR target_match > 0
                RETURN r.verb as verb,
                       target.word as target,
                       target.g as g,
                       r.weight as weight,
                       verb_match + target_match as intent_score
                ORDER BY intent_score DESC, r.weight DESC
                LIMIT $limit
            """, word=word, intent_verbs=list(intent_verbs),
                 intent_targets=list(intent_targets), limit=limit)

            transitions = []
            for record in result:
                score = record["intent_score"] * 0.5 + record["weight"] * 0.3 + max(0, record["g"]) * 0.2
                transitions.append((record["verb"], record["target"], score))

            return transitions

    def get_all_transitions(self, word: str, limit: int = 20) -> List[Tuple[str, str, float]]:
        """
        Get all transitions from word (no intent filtering).

        Fallback when no intent matches.
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (source:Concept {word: $word})
                      -[r:VIA]->
                      (target:Concept)
                WHERE target.tau > 1.5
                  AND size(target.word) >= 3
                RETURN r.verb as verb,
                       target.word as target,
                       target.g as g,
                       r.weight as weight
                ORDER BY r.weight DESC
                LIMIT $limit
            """, word=word, limit=limit)

            return [(r["verb"], r["target"], r["weight"]) for r in result]

    def get_concept(self, word: str) -> Optional[Dict]:
        """Get concept properties."""
        if not self.driver:
            return None

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {word: $word})
                RETURN c.g as g, c.tau as tau, c.j as j, c.pos as pos
            """, word=word)

            record = result.single()
            if record:
                return {
                    "word": word,
                    "g": record["g"],
                    "tau": record["tau"],
                    "j": record["j"],
                    "pos": record["pos"]
                }
            return None

    def has_concept(self, word: str) -> bool:
        """Check if concept exists."""
        if not self.driver:
            return False

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {word: $word})
                RETURN count(c) > 0 as exists
            """, word=word)
            record = result.single()
            return record["exists"] if record else False

    # =========================================================================
    # Learning Methods
    # =========================================================================

    def get_learning_store(self):
        """
        Get Neo4jLearningStore for this graph.

        Lazy initialization - created on first access.
        """
        if not hasattr(self, '_learning_store'):
            from .learning import Neo4jLearningStore
            self._learning_store = Neo4jLearningStore(self.driver) if self.driver else None
        return self._learning_store

    def setup_learning_schema(self):
        """Setup schema for learning (Adjective nodes, DESCRIBED_BY edges)."""
        store = self.get_learning_store()
        if store:
            store.setup_schema()

    def learn_concept(self, noun: str, adj_counts: Dict[str, int],
                      source: str = "unknown",
                      j_vectors: Dict[str, np.ndarray] = None) -> Optional[Dict]:
        """
        Learn a new concept from adjective observations.

        This is the main learning entry point:
        1. Stores adjective observations in Neo4j
        2. Computes entropy → τ
        3. Computes j-centroid → g
        4. Creates/updates Concept node

        Args:
            noun: Concept to learn
            adj_counts: {adjective: count} distribution
            source: Origin ("book", "conversation", etc.)
            j_vectors: Adjective j-vectors for centroid computation

        Returns:
            Updated concept properties dict
        """
        store = self.get_learning_store()
        if store:
            return store.learn_concept(noun, adj_counts, source, j_vectors)
        return None

    def observe_adjective(self, noun: str, adjective: str,
                          count: int = 1, source: str = "unknown"):
        """
        Record a single adjective observation.

        Use learn_concept() with update_concept() for full learning.
        This just records the observation without recomputing τ.
        """
        store = self.get_learning_store()
        if store:
            store.observe_adjective(noun, adjective, count, source)

    def update_learned_concept(self, noun: str,
                               j_vectors: Dict[str, np.ndarray] = None) -> Optional[Dict]:
        """
        Recompute concept's τ, g, j from its adjective distribution.

        Call after batch observations to update derived parameters.
        """
        store = self.get_learning_store()
        if store:
            return store.update_concept_from_distribution(noun, j_vectors)
        return None

    def get_learned_concepts(self, min_variety: int = 3) -> List[Dict]:
        """Get all concepts that were learned (not from corpus)."""
        store = self.get_learning_store()
        if store:
            return store.get_learned_concepts(min_variety)
        return []

    def get_adj_distribution(self, noun: str) -> Dict[str, int]:
        """Get adjective distribution for a concept."""
        store = self.get_learning_store()
        if store:
            return store.get_adj_distribution(noun)
        return {}

    def get_learning_stats(self) -> Dict:
        """Get learning-specific statistics."""
        store = self.get_learning_store()
        if store:
            return store.get_stats()
        return {"error": "Learning store not available"}

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        if not self.driver:
            return {"error": "Not connected"}

        with self.driver.session() as session:
            # Concept count
            result = session.run("MATCH (c:Concept) RETURN count(c) as count")
            concepts = result.single()["count"]

            # VerbOperator count
            result = session.run("MATCH (v:VerbOperator) RETURN count(v) as count")
            verbs = result.single()["count"]

            # VIA count
            result = session.run("MATCH ()-[r:VIA]->() RETURN count(r) as count")
            via_edges = result.single()["count"]

            # OPERATES_ON count
            result = session.run("MATCH ()-[r:OPERATES_ON]->() RETURN count(r) as count")
            operates_edges = result.single()["count"]

            # Sample verbs
            result = session.run("""
                MATCH ()-[r:VIA]->()
                RETURN DISTINCT r.verb as verb
                LIMIT 10
            """)
            sample_verbs = [r["verb"] for r in result]

            return {
                "concepts": concepts,
                "verb_operators": verbs,
                "via_edges": via_edges,
                "operates_on_edges": operates_edges,
                "sample_verbs": sample_verbs
            }


def main():
    """Test the MeaningGraph."""
    print("=" * 60)
    print("MeaningGraph Test")
    print("=" * 60)

    graph = MeaningGraph()

    if not graph.is_connected():
        print("\nNot connected. Start Neo4j with:")
        print("  cd config && docker-compose up -d")
        return

    # Setup
    graph.setup_schema()

    # Show stats
    stats = graph.get_stats()
    print(f"\nStats: {stats}")

    graph.close()


if __name__ == "__main__":
    main()
