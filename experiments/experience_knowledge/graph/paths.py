"""
Explored Paths: Transcendental structure discovered through navigation.

Two distinct concepts:
- Experience = walked paths (from reading books) - "I was here"
- Explored = discovered paths (from navigation) - "I found this route exists"

The transcendental graph stores EXPLORED territory:
- Paths that work (validated navigation)
- Paths that fail (dead ends)
- Discovered connections (not from books)

Theory:
- Walking creates Experience (direct knowledge)
- Navigating creates Exploration (map knowledge)
- Experience + Exploration = Full Knowledge
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from neo4j import GraphDatabase
import json

import sys
_THIS_FILE = Path(__file__).resolve()
_GRAPH_DIR = _THIS_FILE.parent
_EXPERIENCE_KNOWLEDGE = _GRAPH_DIR.parent

sys.path.insert(0, str(_EXPERIENCE_KNOWLEDGE))

from layers.core import Wholeness
from graph.experience import GraphConfig


@dataclass
class ExploredPath:
    """A discovered path through semantic space."""
    start: str
    end: str
    path: List[str]
    goodness_trajectory: List[float]
    delta_g: float
    valid: bool  # Does this path actually work?
    discovery_method: str  # "navigation", "tunneling", "inference"
    confidence: float


class TranscendentalGraph:
    """
    Store explored/discovered paths through semantic space.

    Separate from Experience (walked paths).
    This is the MAP, not the FOOTPRINTS.
    """

    def __init__(self, config: GraphConfig = None, wholeness: Wholeness = None):
        self.config = config or GraphConfig()
        self.driver = None
        self.wholeness = wholeness

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
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def setup_schema(self):
        """Create schema for transcendental graph."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Explored path relationship
            session.run("""
                CREATE INDEX explored_path_confidence IF NOT EXISTS
                FOR ()-[r:EXPLORED_PATH]-() ON (r.confidence)
            """)

            # Discovery node (tracks exploration events)
            session.run("""
                CREATE CONSTRAINT discovery_id IF NOT EXISTS
                FOR (d:Discovery) REQUIRE d.id IS UNIQUE
            """)

            print("Transcendental schema ready")

    # =========================================================================
    # EXPLORATION METHODS
    # =========================================================================

    def explore_path(self, start: str, end: str, method: str = "navigation") -> Optional[ExploredPath]:
        """
        Attempt to explore a path from start to end.

        If path exists (in experience), record it in transcendental graph.
        This is DISCOVERY - finding that a route exists.
        """
        if not self.driver:
            return None

        # Try to find path through experience (walked paths)
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:SemanticState {word: $start}),
                      (end:SemanticState {word: $end}),
                      path = shortestPath((start)-[:TRANSITION*1..6]->(end))
                RETURN [n in nodes(path) | n.word] as words,
                       [n in nodes(path) | n.goodness] as goodness
            """, start=start, end=end)

            record = result.single()
            if not record:
                # Path doesn't exist in experience
                return None

            words = record["words"]
            goodness = record["goodness"]

            explored = ExploredPath(
                start=start,
                end=end,
                path=words,
                goodness_trajectory=goodness,
                delta_g=goodness[-1] - goodness[0] if goodness else 0,
                valid=True,
                discovery_method=method,
                confidence=0.8  # High confidence - path exists in experience
            )

            # Store in transcendental graph
            self._store_explored_path(explored)

            return explored

    def _store_explored_path(self, path: ExploredPath):
        """Store explored path in transcendental graph."""
        if not self.driver:
            return

        with self.driver.session() as session:
            # Create/update EXPLORED_PATH relationship
            session.run("""
                MATCH (start:SemanticState {word: $start}),
                      (end:SemanticState {word: $end})
                MERGE (start)-[r:EXPLORED_PATH]->(end)
                SET r.path = $path,
                    r.goodness_trajectory = $trajectory,
                    r.delta_g = $delta_g,
                    r.valid = $valid,
                    r.method = $method,
                    r.confidence = $confidence,
                    r.explored_at = datetime()
            """,
            start=path.start,
            end=path.end,
            path=path.path,
            trajectory=path.goodness_trajectory,
            delta_g=path.delta_g,
            valid=path.valid,
            method=path.discovery_method,
            confidence=path.confidence
            )

    def explore_territory(self, center: str, radius: int = 2) -> Dict:
        """
        Explore territory around a concept.

        Discovers all reachable states within radius steps.
        This expands the transcendental map.
        """
        if not self.driver:
            return {}

        with self.driver.session() as session:
            # Find all states reachable within radius
            result = session.run("""
                MATCH (center:SemanticState {word: $center})
                CALL {
                    WITH center
                    MATCH path = (center)-[:TRANSITION*1..""" + str(radius) + """]->(reached:SemanticState)
                    RETURN DISTINCT reached.word as word,
                           reached.goodness as g,
                           length(path) as distance
                }
                RETURN word, g, distance
                ORDER BY distance, g DESC
            """, center=center)

            territory = {
                "center": center,
                "radius": radius,
                "discovered": []
            }

            for record in result:
                territory["discovered"].append({
                    "word": record["word"],
                    "goodness": record["g"],
                    "distance": record["distance"]
                })

                # Record exploration
                self._record_discovery(center, record["word"], record["distance"])

            return territory

    def _record_discovery(self, from_word: str, to_word: str, distance: int):
        """Record a discovery in the transcendental graph."""
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run("""
                MATCH (from:SemanticState {word: $from_word}),
                      (to:SemanticState {word: $to_word})
                MERGE (from)-[r:DISCOVERED]->(to)
                ON CREATE SET r.distance = $distance, r.first_seen = datetime()
                ON MATCH SET r.times_seen = coalesce(r.times_seen, 0) + 1
            """, from_word=from_word, to_word=to_word, distance=distance)

    def explore_redemption_arcs(self, min_delta_g: float = 0.5) -> List[ExploredPath]:
        """
        Discover all redemption arcs (paths from negative to positive g).

        These are the meaningful moral trajectories in the explored territory.
        """
        if not self.driver:
            return []

        print(f"\nExploring redemption arcs (Δg > {min_delta_g})...")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:SemanticState)
                WHERE start.goodness < -0.1 AND start.tau > 1.5
                MATCH (end:SemanticState)
                WHERE end.goodness > 0.3 AND end.tau > 1.5
                  AND end.goodness - start.goodness > $min_delta_g
                WITH start, end
                MATCH path = shortestPath((start)-[:TRANSITION*1..5]->(end))
                RETURN start.word as start_word,
                       end.word as end_word,
                       [n in nodes(path) | n.word] as words,
                       [n in nodes(path) | n.goodness] as goodness
                LIMIT 100
            """, min_delta_g=min_delta_g)

            arcs = []
            for record in result:
                words = record["words"]
                goodness = record["goodness"]

                arc = ExploredPath(
                    start=record["start_word"],
                    end=record["end_word"],
                    path=words,
                    goodness_trajectory=goodness,
                    delta_g=goodness[-1] - goodness[0],
                    valid=True,
                    discovery_method="redemption_search",
                    confidence=0.9
                )
                arcs.append(arc)

                # Store in transcendental graph
                self._store_explored_path(arc)

            print(f"  Discovered {len(arcs)} redemption arcs")
            return arcs

    def explore_descent_arcs(self, min_delta_g: float = -0.5) -> List[ExploredPath]:
        """
        Discover all descent arcs (paths from positive to negative g).

        These are the fall/corruption trajectories.
        """
        if not self.driver:
            return []

        print(f"\nExploring descent arcs (Δg < {min_delta_g})...")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:SemanticState)
                WHERE start.goodness > 0.2 AND start.tau > 1.5
                MATCH (end:SemanticState)
                WHERE end.goodness < -0.1 AND end.tau > 1.5
                  AND end.goodness - start.goodness < $min_delta_g
                WITH start, end
                MATCH path = shortestPath((start)-[:TRANSITION*1..5]->(end))
                RETURN start.word as start_word,
                       end.word as end_word,
                       [n in nodes(path) | n.word] as words,
                       [n in nodes(path) | n.goodness] as goodness
                LIMIT 100
            """, min_delta_g=min_delta_g)

            arcs = []
            for record in result:
                words = record["words"]
                goodness = record["goodness"]

                arc = ExploredPath(
                    start=record["start_word"],
                    end=record["end_word"],
                    path=words,
                    goodness_trajectory=goodness,
                    delta_g=goodness[-1] - goodness[0],
                    valid=True,
                    discovery_method="descent_search",
                    confidence=0.9
                )
                arcs.append(arc)
                self._store_explored_path(arc)

            print(f"  Discovered {len(arcs)} descent arcs")
            return arcs

    def get_explored_stats(self) -> Dict:
        """Get statistics about explored territory."""
        if not self.driver:
            return {}

        with self.driver.session() as session:
            # Count explored paths
            result = session.run("""
                MATCH ()-[r:EXPLORED_PATH]->()
                RETURN count(r) as explored_paths,
                       avg(r.delta_g) as avg_delta_g,
                       count(CASE WHEN r.delta_g > 0 THEN 1 END) as redemption_count,
                       count(CASE WHEN r.delta_g < 0 THEN 1 END) as descent_count
            """)
            record = result.single()

            # Count discoveries
            result2 = session.run("""
                MATCH ()-[r:DISCOVERED]->()
                RETURN count(r) as discoveries
            """)
            discoveries = result2.single()["discoveries"]

            return {
                "explored_paths": record["explored_paths"],
                "avg_delta_g": record["avg_delta_g"],
                "redemption_arcs": record["redemption_count"],
                "descent_arcs": record["descent_count"],
                "discoveries": discoveries
            }

    def get_explored_path(self, start: str, end: str) -> Optional[ExploredPath]:
        """Get previously explored path if it exists."""
        if not self.driver:
            return None

        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:SemanticState {word: $start})
                      -[r:EXPLORED_PATH]->
                      (end:SemanticState {word: $end})
                RETURN r.path as path,
                       r.goodness_trajectory as trajectory,
                       r.delta_g as delta_g,
                       r.valid as valid,
                       r.method as method,
                       r.confidence as confidence
            """, start=start, end=end)

            record = result.single()
            if not record:
                return None

            return ExploredPath(
                start=start,
                end=end,
                path=record["path"],
                goodness_trajectory=record["trajectory"],
                delta_g=record["delta_g"],
                valid=record["valid"],
                discovery_method=record["method"],
                confidence=record["confidence"]
            )


def explore_full_territory():
    """Explore the full territory and build transcendental map."""
    print("=" * 70)
    print("BUILDING TRANSCENDENTAL MAP")
    print("=" * 70)

    # Load wholeness
    wholeness = Wholeness()

    # Create transcendental graph
    config = GraphConfig()
    graph = TranscendentalGraph(config, wholeness)

    if not graph.driver:
        print("\nNeo4j not available. Start with: docker-compose up -d")
        return

    graph.setup_schema()

    # Explore redemption arcs
    redemption_arcs = graph.explore_redemption_arcs(min_delta_g=0.4)

    # Explore descent arcs
    descent_arcs = graph.explore_descent_arcs(min_delta_g=-0.4)

    # Explore territory around key concepts
    key_concepts = ["love", "hate", "fear", "hope", "light", "darkness", "truth", "wisdom"]
    for concept in key_concepts:
        print(f"\nExploring territory around '{concept}'...")
        territory = graph.explore_territory(concept, radius=3)
        print(f"  Discovered {len(territory.get('discovered', []))} reachable states")

    # Show stats
    print("\n" + "=" * 70)
    print("TRANSCENDENTAL MAP STATISTICS")
    print("=" * 70)

    stats = graph.get_explored_stats()
    print(f"""
    Explored territory:
      Total explored paths: {stats['explored_paths']}
      Redemption arcs: {stats['redemption_arcs']}
      Descent arcs: {stats['descent_arcs']}
      Average Δg: {stats['avg_delta_g']:.3f if stats['avg_delta_g'] else 'N/A'}
      Total discoveries: {stats['discoveries']}
    """)

    # Show example paths
    print("\n" + "-" * 70)
    print("EXAMPLE REDEMPTION ARCS")
    print("-" * 70)

    for arc in redemption_arcs[:5]:
        path_str = " → ".join(arc.path[:6])
        print(f"  {arc.start} → {arc.end}: Δg={arc.delta_g:+.2f}")
        print(f"    {path_str}")

    print("\n" + "-" * 70)
    print("EXAMPLE DESCENT ARCS")
    print("-" * 70)

    for arc in descent_arcs[:5]:
        path_str = " → ".join(arc.path[:6])
        print(f"  {arc.start} → {arc.end}: Δg={arc.delta_g:+.2f}")
        print(f"    {path_str}")

    graph.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcendental Graph - Explored Paths")
    parser.add_argument("command", choices=["explore", "stats", "path"],
                       help="Command to run")
    parser.add_argument("--start", help="Start word")
    parser.add_argument("--end", help="End word")

    args = parser.parse_args()

    if args.command == "explore":
        explore_full_territory()
    elif args.command == "stats":
        wholeness = Wholeness()
        graph = TranscendentalGraph(wholeness=wholeness)
        if graph.driver:
            stats = graph.get_explored_stats()
            print(f"Explored paths: {stats['explored_paths']}")
            print(f"Redemption arcs: {stats['redemption_arcs']}")
            print(f"Descent arcs: {stats['descent_arcs']}")
            print(f"Discoveries: {stats['discoveries']}")
            graph.close()
    elif args.command == "path":
        if args.start and args.end:
            wholeness = Wholeness()
            graph = TranscendentalGraph(wholeness=wholeness)
            if graph.driver:
                # First try to get existing explored path
                path = graph.get_explored_path(args.start, args.end)
                if path:
                    print(f"Previously explored: {' → '.join(path.path)}")
                    print(f"Δg={path.delta_g:+.2f}, confidence={path.confidence}")
                else:
                    # Try to explore new path
                    path = graph.explore_path(args.start, args.end)
                    if path:
                        print(f"Newly explored: {' → '.join(path.path)}")
                        print(f"Δg={path.delta_g:+.2f}")
                    else:
                        print("No path found")
                graph.close()
