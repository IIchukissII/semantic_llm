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
from neo4j import GraphDatabase

# Import from layers
import sys
_THIS_FILE = Path(__file__).resolve()
_EXPERIENCE_KNOWLEDGE = _THIS_FILE.parent.parent

sys.path.insert(0, str(_EXPERIENCE_KNOWLEDGE))

from layers.core import Wholeness, SemanticState


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

    def _store_states(self, visits: Dict[str, int], book_id: str):
        """Store or update semantic states (batched for performance)."""
        if not self.driver:
            return

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
                'book_id': book_id
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
                    s.books = [item.book_id]
                ON MATCH SET
                    s.visits = s.visits + item.count,
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

        # Prepare batch data
        batch = [
            {'from_word': f, 'to_word': t, 'count': c, 'book_id': book_id}
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
                        t.weight = item.count,
                        t.books = [item.book_id]
                    ON MATCH SET
                        t.weight = t.weight + item.count,
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
