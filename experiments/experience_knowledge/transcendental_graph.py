"""
Transcendental Graph: Expand experience with semantic structure.

Adds to experience graph:
- SEMANTIC_NEIGHBOR: j-vector similarity
- SPIN_PAIR: word ↔ antiword (love ↔ hate)
- VERB_CONNECTS: verb transitions (love → kiss, hate → attack)
- GOODNESS_ARC: computed moral trajectory

This expands the navigable space beyond just walked paths.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from neo4j import GraphDatabase

import sys
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent
_SEMANTIC_LLM = _EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_EXPERIMENT_DIR))

from core import Wholeness
from graph_experience import GraphConfig, ExperienceGraph


class TranscendentalGraph(ExperienceGraph):
    """
    Extends ExperienceGraph with transcendental structure.

    Adds relationship types:
    - :SEMANTIC_NEIGHBOR {similarity: float}  - j-vector proximity
    - :SPIN_PAIR {cos: float, delta_tau: float}  - word opposites
    - :VERB_CONNECTS {verb: string, count: int}  - verb transitions
    """

    def __init__(self, config: GraphConfig = None, wholeness: Wholeness = None):
        super().__init__(config, wholeness)

    def setup_transcendental_schema(self):
        """Create indexes for transcendental relationships."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Index for semantic neighbors
            session.run("""
                CREATE INDEX semantic_neighbor_sim IF NOT EXISTS
                FOR ()-[r:SEMANTIC_NEIGHBOR]-() ON (r.similarity)
            """)

            # Index for spin pairs
            session.run("""
                CREATE INDEX spin_pair_cos IF NOT EXISTS
                FOR ()-[r:SPIN_PAIR]-() ON (r.cos)
            """)

            print("Transcendental schema setup complete")

    def load_semantic_neighbors(self, threshold: float = 0.3, max_neighbors: int = 20):
        """
        Load semantic neighbors based on j-vector similarity.

        This expands navigable space beyond walked paths.
        """
        if not self.driver or not self.wholeness:
            return

        print(f"\nLoading semantic neighbors (threshold={threshold})...")

        # Get all states in experience
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState)
                RETURN s.word as word, s.j as j
            """)
            states = [(r["word"], np.array(r["j"])) for r in result]

        print(f"  Computing similarities for {len(states)} states...")

        # Compute neighbors (this is O(n²) but limited to experienced states)
        neighbors = []
        for i, (w1, j1) in enumerate(states):
            if i % 500 == 0:
                print(f"    {i}/{len(states)}...")

            distances = []
            for j, (w2, j2) in enumerate(states):
                if i != j:
                    dist = np.linalg.norm(j1 - j2)
                    if dist < threshold:
                        distances.append((w2, 1.0 - dist/threshold))  # Convert to similarity

            # Keep top-k neighbors
            distances.sort(key=lambda x: -x[1])
            for w2, sim in distances[:max_neighbors]:
                neighbors.append({
                    'from_word': w1,
                    'to_word': w2,
                    'similarity': float(sim)
                })

        print(f"  Found {len(neighbors)} neighbor pairs")

        # Batch insert
        chunk_size = 5000
        with self.driver.session() as session:
            for i in range(0, len(neighbors), chunk_size):
                chunk = neighbors[i:i+chunk_size]
                session.run("""
                    UNWIND $batch AS item
                    MATCH (a:SemanticState {word: item.from_word})
                    MATCH (b:SemanticState {word: item.to_word})
                    MERGE (a)-[r:SEMANTIC_NEIGHBOR]->(b)
                    SET r.similarity = item.similarity
                """, batch=chunk)

        print(f"  Stored {len(neighbors)} SEMANTIC_NEIGHBOR relationships")

    def load_spin_pairs(self):
        """
        Load spin pairs (word ↔ antiword) from QuantumCore.

        These are transcendental opposites in moral space.
        """
        if not self.driver or not self.wholeness:
            return

        print("\nLoading spin pairs...")

        # Load from QuantumCore
        try:
            import importlib.util

            hybrid_llm_path = _SEMANTIC_LLM / "core" / "hybrid_llm.py"
            spec = importlib.util.spec_from_file_location("hybrid_llm", hybrid_llm_path)
            hybrid_llm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hybrid_llm)

            core = hybrid_llm.QuantumCore()

            pairs = []
            for word, antiword, cos, delta_tau in core.spin_pairs:
                # Check if both words are in our experience
                pairs.append({
                    'word': word,
                    'antiword': antiword,
                    'cos': float(cos),
                    'delta_tau': float(delta_tau)
                })

            print(f"  Found {len(pairs)} spin pairs in QuantumCore")

            # Store in Neo4j (only if both endpoints exist)
            with self.driver.session() as session:
                result = session.run("""
                    UNWIND $pairs AS p
                    MATCH (a:SemanticState {word: p.word})
                    MATCH (b:SemanticState {word: p.antiword})
                    MERGE (a)-[r:SPIN_PAIR]-(b)
                    SET r.cos = p.cos, r.delta_tau = p.delta_tau
                    RETURN count(r) as count
                """, pairs=pairs)
                count = result.single()["count"]
                print(f"  Stored {count} SPIN_PAIR relationships")

        except Exception as e:
            print(f"  Error loading spin pairs: {e}")

    def load_verb_connections(self):
        """
        Load verb-based connections from QuantumCore.

        E.g., if "love" and "kiss" both appear with verb "feels",
        they are connected: (love)-[:VERB_CONNECTS {verb: "feels"}]->(kiss)
        """
        if not self.driver or not self.wholeness:
            return

        print("\nLoading verb connections...")

        try:
            import importlib.util

            hybrid_llm_path = _SEMANTIC_LLM / "core" / "hybrid_llm.py"
            spec = importlib.util.spec_from_file_location("hybrid_llm", hybrid_llm_path)
            hybrid_llm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hybrid_llm)

            core = hybrid_llm.QuantumCore()

            # Get verbs and their objects
            verb_connections = []
            for verb, objects in core.verb_objects.items():
                obj_list = list(objects)
                # Connect objects that share this verb
                for i, obj1 in enumerate(obj_list[:50]):  # Limit per verb
                    for obj2 in obj_list[i+1:i+10]:
                        verb_connections.append({
                            'word1': obj1,
                            'word2': obj2,
                            'verb': verb
                        })

            print(f"  Found {len(verb_connections)} verb connections")

            # Batch insert (only if both endpoints exist in experience)
            chunk_size = 5000
            total_stored = 0
            with self.driver.session() as session:
                for i in range(0, len(verb_connections), chunk_size):
                    chunk = verb_connections[i:i+chunk_size]
                    result = session.run("""
                        UNWIND $batch AS item
                        MATCH (a:SemanticState {word: item.word1})
                        MATCH (b:SemanticState {word: item.word2})
                        MERGE (a)-[r:VERB_CONNECTS]-(b)
                        ON CREATE SET r.verbs = [item.verb]
                        ON MATCH SET r.verbs = CASE
                            WHEN NOT item.verb IN r.verbs
                            THEN r.verbs + item.verb
                            ELSE r.verbs
                        END
                        RETURN count(r) as count
                    """, batch=chunk)
                    total_stored += result.single()["count"]

            print(f"  Stored {total_stored} VERB_CONNECTS relationships")

        except Exception as e:
            print(f"  Error loading verb connections: {e}")

    def compute_goodness_arcs(self):
        """
        Compute and store goodness arcs (moral trajectories).

        An arc exists when there's a multi-step path with consistent
        goodness change (ascending or descending).
        """
        if not self.driver:
            return

        print("\nComputing goodness arcs...")

        with self.driver.session() as session:
            # Find ascending arcs (darkness → light type paths)
            result = session.run("""
                MATCH path = (start:SemanticState)-[:TRANSITION*2..4]->(end:SemanticState)
                WHERE start.goodness < -0.2 AND end.goodness > 0.2
                  AND end.goodness - start.goodness > 0.5
                WITH start, end,
                     end.goodness - start.goodness as delta_g,
                     length(path) as path_length
                MERGE (start)-[a:GOODNESS_ARC]->(end)
                SET a.delta_g = delta_g,
                    a.direction = 'ascending',
                    a.path_length = path_length
                RETURN count(a) as count
            """)
            ascending = result.single()["count"]
            print(f"  Found {ascending} ascending arcs")

            # Find descending arcs (light → darkness type paths)
            result = session.run("""
                MATCH path = (start:SemanticState)-[:TRANSITION*2..4]->(end:SemanticState)
                WHERE start.goodness > 0.2 AND end.goodness < -0.2
                  AND start.goodness - end.goodness > 0.5
                WITH start, end,
                     end.goodness - start.goodness as delta_g,
                     length(path) as path_length
                MERGE (start)-[a:GOODNESS_ARC]->(end)
                SET a.delta_g = delta_g,
                    a.direction = 'descending',
                    a.path_length = path_length
                RETURN count(a) as count
            """)
            descending = result.single()["count"]
            print(f"  Found {descending} descending arcs")

    def expand_tunneling(self, target: str, believe: float = 0.5) -> Tuple[bool, float, str]:
        """
        Enhanced tunneling using transcendental structure.

        Now can tunnel via:
        1. Direct experience (walked path) - highest probability
        2. Semantic neighbors - medium probability
        3. Spin pairs - low probability (risky jump)
        4. Verb connections - medium probability

        Returns (can_tunnel, probability, via)
        """
        if not self.driver:
            return False, 0.0, "none"

        # First check direct experience
        can, prob = self.can_tunnel(target)
        if can and prob > 0.3:
            return True, prob, "experience"

        # Check semantic neighbors
        with self.driver.session() as session:
            result = session.run("""
                MATCH (known:SemanticState)-[r:SEMANTIC_NEIGHBOR]-(target:SemanticState {word: $target})
                WHERE known.visits > 0
                RETURN max(r.similarity) as max_sim, count(known) as known_neighbors
            """, target=target)
            record = result.single()
            if record and record["max_sim"]:
                sim = record["max_sim"]
                prob = sim * believe * 0.5
                return True, prob, "semantic_neighbor"

            # Check spin pairs (risky but possible)
            result = session.run("""
                MATCH (known:SemanticState)-[r:SPIN_PAIR]-(target:SemanticState {word: $target})
                WHERE known.visits > 0
                RETURN r.cos as cos, known.word as via_word
                LIMIT 1
            """, target=target)
            record = result.single()
            if record:
                # Negative cos means opposite - higher risk
                cos = record["cos"]
                prob = (1 - abs(cos)) * believe * 0.3  # Lower prob for opposites
                return True, prob, f"spin_pair_via_{record['via_word']}"

            # Check verb connections
            result = session.run("""
                MATCH (known:SemanticState)-[r:VERB_CONNECTS]-(target:SemanticState {word: $target})
                WHERE known.visits > 0
                RETURN r.verbs as verbs, count(*) as connection_count
            """, target=target)
            record = result.single()
            if record and record["connection_count"] > 0:
                prob = min(0.4, record["connection_count"] * 0.1) * believe
                return True, prob, "verb_connection"

        return False, 0.0, "none"

    def find_redemption_path(self, start: str, end: str) -> Dict:
        """
        Find redemption path (moral journey) between concepts.

        Uses all relationship types weighted by reliability.
        """
        if not self.driver:
            return {"error": "Not connected"}

        with self.driver.session() as session:
            # Try to find a path using any relationship type
            result = session.run("""
                MATCH (start:SemanticState {word: $start}),
                      (end:SemanticState {word: $end})
                CALL {
                    WITH start, end
                    MATCH path = shortestPath((start)-[:TRANSITION|SEMANTIC_NEIGHBOR|VERB_CONNECTS*1..6]->(end))
                    RETURN path, length(path) as len
                    UNION
                    WITH start, end
                    MATCH path = shortestPath((start)-[:TRANSITION|SEMANTIC_NEIGHBOR|VERB_CONNECTS*1..6]-(end))
                    RETURN path, length(path) as len
                }
                RETURN [n in nodes(path) | n.word] as words,
                       [n in nodes(path) | n.goodness] as goodness,
                       [type(r) for r in relationships(path)] as rel_types,
                       len
                ORDER BY len
                LIMIT 1
            """, start=start, end=end)

            record = result.single()
            if not record:
                return {"found": False, "start": start, "end": end}

            words = record["words"]
            goodness = record["goodness"]
            rel_types = record["rel_types"]

            return {
                "found": True,
                "start": start,
                "end": end,
                "path": words,
                "goodness_trajectory": goodness,
                "delta_g": goodness[-1] - goodness[0] if goodness else 0,
                "relationship_types": rel_types,
                "length": record["len"]
            }

    def get_transcendental_stats(self) -> Dict:
        """Get stats including transcendental relationships."""
        stats = self.get_stats()

        if not self.driver:
            return stats

        with self.driver.session() as session:
            # Count semantic neighbors
            result = session.run("MATCH ()-[r:SEMANTIC_NEIGHBOR]->() RETURN count(r) as count")
            stats["semantic_neighbors"] = result.single()["count"]

            # Count spin pairs
            result = session.run("MATCH ()-[r:SPIN_PAIR]-() RETURN count(r) as count")
            stats["spin_pairs"] = result.single()["count"]

            # Count verb connections
            result = session.run("MATCH ()-[r:VERB_CONNECTS]-() RETURN count(r) as count")
            stats["verb_connections"] = result.single()["count"]

            # Count goodness arcs
            result = session.run("MATCH ()-[r:GOODNESS_ARC]->() RETURN count(r) as count")
            stats["goodness_arcs"] = result.single()["count"]

        return stats


def expand_graph():
    """Expand experience graph with transcendental structure."""
    print("=" * 70)
    print("EXPANDING TRANSCENDENTAL GRAPH")
    print("=" * 70)

    # Load wholeness
    wholeness = Wholeness()

    # Create transcendental graph
    config = GraphConfig()
    graph = TranscendentalGraph(config, wholeness)

    if not graph.driver:
        print("\nNeo4j not available. Start with: docker-compose up -d")
        return

    # Setup schema
    graph.setup_transcendental_schema()

    # Expand with transcendental relationships
    graph.load_spin_pairs()
    graph.load_verb_connections()
    graph.load_semantic_neighbors(threshold=0.35, max_neighbors=10)
    graph.compute_goodness_arcs()

    # Show expanded stats
    print("\n" + "=" * 70)
    print("TRANSCENDENTAL GRAPH STATISTICS")
    print("=" * 70)

    stats = graph.get_transcendental_stats()
    print(f"""
    Experience (walked paths):
      States: {stats['nodes']}
      Transitions: {stats['edges']}
      Books: {len(stats.get('books', []))}

    Transcendental (expanded):
      Semantic neighbors: {stats.get('semantic_neighbors', 0)}
      Spin pairs: {stats.get('spin_pairs', 0)}
      Verb connections: {stats.get('verb_connections', 0)}
      Goodness arcs: {stats.get('goodness_arcs', 0)}

    Total navigable space: {stats['edges'] + stats.get('semantic_neighbors', 0) + stats.get('verb_connections', 0)} relationships
    """)

    # Test expanded tunneling
    print("\n" + "-" * 70)
    print("EXPANDED TUNNELING TEST")
    print("-" * 70)

    test_targets = ["redemption", "transcendence", "enlightenment", "salvation", "damnation"]
    for target in test_targets:
        can, prob, via = graph.expand_tunneling(target)
        print(f"  {target:15} {'Yes' if can else 'No':5} p={prob:.2f} via={via}")

    # Test redemption paths
    print("\n" + "-" * 70)
    print("REDEMPTION PATHS")
    print("-" * 70)

    paths_to_find = [
        ("darkness", "light"),
        ("sin", "redemption"),
        ("despair", "hope"),
        ("hate", "love"),
    ]

    for start, end in paths_to_find:
        result = graph.find_redemption_path(start, end)
        if result.get("found"):
            path_str = " → ".join(result["path"][:6])
            if len(result["path"]) > 6:
                path_str += "..."
            print(f"  {start} → {end}: Δg={result['delta_g']:+.2f}")
            print(f"    Path: {path_str}")
        else:
            print(f"  {start} → {end}: No path found")

    graph.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcendental Graph")
    parser.add_argument("command", choices=["expand", "stats", "tunnel", "path"],
                       help="Command to run")
    parser.add_argument("--target", help="Target word for tunneling")
    parser.add_argument("--start", help="Start word for path")
    parser.add_argument("--end", help="End word for path")

    args = parser.parse_args()

    if args.command == "expand":
        expand_graph()
    elif args.command == "stats":
        wholeness = Wholeness()
        graph = TranscendentalGraph(wholeness=wholeness)
        if graph.driver:
            stats = graph.get_transcendental_stats()
            print(f"Nodes: {stats['nodes']}")
            print(f"Transitions: {stats['edges']}")
            print(f"Semantic neighbors: {stats.get('semantic_neighbors', 0)}")
            print(f"Spin pairs: {stats.get('spin_pairs', 0)}")
            print(f"Verb connections: {stats.get('verb_connections', 0)}")
            graph.close()
    elif args.command == "tunnel":
        if args.target:
            wholeness = Wholeness()
            graph = TranscendentalGraph(wholeness=wholeness)
            if graph.driver:
                can, prob, via = graph.expand_tunneling(args.target)
                print(f"Can tunnel to '{args.target}': {can}, p={prob:.2f}, via={via}")
                graph.close()
    elif args.command == "path":
        if args.start and args.end:
            wholeness = Wholeness()
            graph = TranscendentalGraph(wholeness=wholeness)
            if graph.driver:
                result = graph.find_redemption_path(args.start, args.end)
                print(result)
                graph.close()
