"""
Load Full Semantic Space into Neo4j

The semantic space is STABLE - derived from corpus analysis.
Loading it once gives us the complete map for navigation.

Structure:
- (:SemanticState) - All 24,524 words with g, tau, j
- (:SemanticState)-[:SEMANTIC_NEIGHBOR]->() - j-vector proximity
- (:SemanticState)-[:SPIN_PAIR]-() - word ↔ antiword

Experience and Transcendental are overlays on top.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
import sys

_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent
_SEMANTIC_LLM = _EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))

from graph_experience import GraphConfig


def load_quantum_core():
    """Load QuantumCore directly."""
    import importlib.util

    data_loader_path = _SEMANTIC_LLM / "core" / "data_loader.py"
    hybrid_llm_path = _SEMANTIC_LLM / "core" / "hybrid_llm.py"

    spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
    data_loader_module = importlib.util.module_from_spec(spec)
    sys.modules['core.data_loader'] = data_loader_module
    spec.loader.exec_module(data_loader_module)

    spec = importlib.util.spec_from_file_location("hybrid_llm", hybrid_llm_path)
    hybrid_llm_module = importlib.util.module_from_spec(spec)
    sys.modules['core.hybrid_llm'] = hybrid_llm_module
    spec.loader.exec_module(hybrid_llm_module)

    print("Loading QuantumCore...")
    core = hybrid_llm_module.QuantumCore()
    print(f"  {len(core.states)} states")
    print(f"  {len(core.verb_objects)} verbs")
    print(f"  {len(core.spin_pairs)} spin pairs")

    return core


def load_full_space(clear_first: bool = False):
    """Load entire semantic space into Neo4j."""
    print("=" * 70)
    print("LOADING FULL SEMANTIC SPACE INTO NEO4J")
    print("=" * 70)

    # Connect to Neo4j
    config = GraphConfig()
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.user, config.password)
    )

    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {config.uri}")
    except Exception as e:
        print(f"Cannot connect to Neo4j: {e}")
        print("Start with: docker-compose up -d")
        return

    # Load QuantumCore
    core = load_quantum_core()

    with driver.session() as session:
        if clear_first:
            print("\nClearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")

        # Setup indexes
        print("\nCreating indexes...")
        session.run("""
            CREATE CONSTRAINT semantic_state_word IF NOT EXISTS
            FOR (s:SemanticState) REQUIRE s.word IS UNIQUE
        """)
        session.run("""
            CREATE INDEX semantic_state_goodness IF NOT EXISTS
            FOR (s:SemanticState) ON (s.goodness)
        """)
        session.run("""
            CREATE INDEX semantic_state_tau IF NOT EXISTS
            FOR (s:SemanticState) ON (s.tau)
        """)

        # Load all states in batches
        print("\nLoading semantic states...")
        states = list(core.states.items())
        batch_size = 1000
        total_loaded = 0

        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size]
            batch_data = []

            for word, state in batch:
                batch_data.append({
                    'word': word,
                    'goodness': float(state.goodness),
                    'tau': float(state.tau),
                    'j': state.j.tolist(),
                    'is_cloud': getattr(state, 'is_cloud', True),
                    'visits': 0  # No experience yet
                })

            session.run("""
                UNWIND $batch AS item
                MERGE (s:SemanticState {word: item.word})
                ON CREATE SET
                    s.goodness = item.goodness,
                    s.tau = item.tau,
                    s.j = item.j,
                    s.is_cloud = item.is_cloud,
                    s.visits = 0,
                    s.books = []
                ON MATCH SET
                    s.goodness = item.goodness,
                    s.tau = item.tau,
                    s.j = item.j,
                    s.is_cloud = item.is_cloud
            """, batch=batch_data)

            total_loaded += len(batch)
            if total_loaded % 5000 == 0:
                print(f"  Loaded {total_loaded}/{len(states)} states...")

        print(f"  Total: {total_loaded} states loaded")

        # Load spin pairs
        print("\nLoading spin pairs...")
        spin_batch = []
        seen_pairs = set()
        for word, spin_pair in core.spin_pairs.items():
            # Each SpinPair has: base, prefixed, j_cosine, delta_tau
            pair_key = tuple(sorted([spin_pair.base, spin_pair.prefixed]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            spin_batch.append({
                'word': spin_pair.base,
                'antiword': spin_pair.prefixed,
                'cos': float(spin_pair.j_cosine),
                'delta_tau': float(spin_pair.delta_tau)
            })

        session.run("""
            UNWIND $batch AS item
            MATCH (a:SemanticState {word: item.word})
            MATCH (b:SemanticState {word: item.antiword})
            MERGE (a)-[r:SPIN_PAIR]-(b)
            SET r.cos = item.cos, r.delta_tau = item.delta_tau
        """, batch=spin_batch)
        print(f"  {len(spin_batch)} spin pairs loaded")

        # Load verb connections (sample to avoid too many edges)
        print("\nLoading verb connections...")
        verb_count = 0
        for verb, objects in list(core.verb_objects.items())[:500]:  # Top 500 verbs
            obj_list = list(objects)[:20]  # Top 20 objects per verb
            if len(obj_list) < 2:
                continue

            verb_batch = []
            for i, obj1 in enumerate(obj_list):
                for obj2 in obj_list[i+1:]:
                    verb_batch.append({
                        'word1': obj1,
                        'word2': obj2,
                        'verb': verb
                    })

            if verb_batch:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (a:SemanticState {word: item.word1})
                    MATCH (b:SemanticState {word: item.word2})
                    MERGE (a)-[r:VERB_CONNECTS]-(b)
                    ON CREATE SET r.verbs = [item.verb]
                    ON MATCH SET r.verbs =
                        CASE WHEN NOT item.verb IN r.verbs
                        THEN r.verbs + item.verb ELSE r.verbs END
                """, batch=verb_batch)
                verb_count += len(verb_batch)

        print(f"  {verb_count} verb connections loaded")

        # Show final stats
        print("\n" + "=" * 70)
        print("SEMANTIC SPACE LOADED")
        print("=" * 70)

        result = session.run("""
            MATCH (s:SemanticState)
            RETURN count(s) as nodes,
                   avg(s.goodness) as avg_g,
                   min(s.goodness) as min_g,
                   max(s.goodness) as max_g,
                   avg(s.tau) as avg_tau
        """)
        stats = result.single()

        result2 = session.run("MATCH ()-[r:SPIN_PAIR]-() RETURN count(r)/2 as pairs")
        spin_pairs = result2.single()["pairs"]

        result3 = session.run("MATCH ()-[r:VERB_CONNECTS]-() RETURN count(r)/2 as connections")
        verb_connections = result3.single()["connections"]

        print(f"""
    Semantic Space:
      Total states: {stats['nodes']}
      Goodness range: [{stats['min_g']:.2f}, {stats['max_g']:.2f}]
      Average goodness: {stats['avg_g']:.3f}
      Average tau: {stats['avg_tau']:.2f}

    Relationships:
      Spin pairs: {spin_pairs}
      Verb connections: {verb_connections}

    The full map is now in Neo4j.
    Experience (visits, books) starts at 0 - read books to gain experience.
        """)

    driver.close()


def show_space_stats():
    """Show current semantic space stats from Neo4j."""
    config = GraphConfig()
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.user, config.password)
    )

    with driver.session() as session:
        # Basic stats
        result = session.run("""
            MATCH (s:SemanticState)
            RETURN count(s) as total,
                   count(CASE WHEN s.visits > 0 THEN 1 END) as experienced,
                   sum(s.visits) as total_visits
        """)
        stats = result.single()

        # Relationship counts
        result2 = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
        """)
        rels = {r["rel_type"]: r["count"] for r in result2}

        # Goodness distribution
        result3 = session.run("""
            MATCH (s:SemanticState)
            RETURN
                count(CASE WHEN s.goodness > 0.5 THEN 1 END) as very_good,
                count(CASE WHEN s.goodness > 0 AND s.goodness <= 0.5 THEN 1 END) as good,
                count(CASE WHEN s.goodness >= -0.5 AND s.goodness <= 0 THEN 1 END) as neutral_bad,
                count(CASE WHEN s.goodness < -0.5 THEN 1 END) as very_bad
        """)
        dist = result3.single()

        print(f"""
SEMANTIC SPACE IN NEO4J
=======================

States:
  Total: {stats['total']}
  Experienced: {stats['experienced']} ({100*stats['experienced']/stats['total']:.1f}%)
  Total visits: {stats['total_visits']}

Relationships:
  TRANSITION: {rels.get('TRANSITION', 0)} (from reading)
  SPIN_PAIR: {rels.get('SPIN_PAIR', 0)//2} (word ↔ antiword)
  VERB_CONNECTS: {rels.get('VERB_CONNECTS', 0)//2} (shared verbs)
  EXPLORED_PATH: {rels.get('EXPLORED_PATH', 0)} (discovered routes)
  DISCOVERED: {rels.get('DISCOVERED', 0)} (explored territory)

Goodness Distribution:
  Very good (g > 0.5): {dist['very_good']}
  Good (0 < g ≤ 0.5): {dist['good']}
  Neutral/Bad (-0.5 ≤ g ≤ 0): {dist['neutral_bad']}
  Very bad (g < -0.5): {dist['very_bad']}
        """)

    driver.close()


def query_word(word: str):
    """Query a specific word from the semantic space."""
    config = GraphConfig()
    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.user, config.password)
    )

    with driver.session() as session:
        result = session.run("""
            MATCH (s:SemanticState {word: $word})
            OPTIONAL MATCH (s)-[sp:SPIN_PAIR]-(anti:SemanticState)
            OPTIONAL MATCH (s)-[:TRANSITION]->(next:SemanticState)
            RETURN s.word as word,
                   s.goodness as g,
                   s.tau as tau,
                   s.visits as visits,
                   s.books as books,
                   collect(DISTINCT anti.word)[0..3] as spin_pairs,
                   collect(DISTINCT next.word)[0..5] as transitions
        """, word=word)

        record = result.single()
        if not record:
            print(f"Word '{word}' not found in semantic space")
            return

        print(f"""
WORD: {record['word']}
================
Goodness: {record['g']:+.3f}
Tau: {record['tau']:.2f}
Visits: {record['visits']}
Books: {record['books'] if record['books'] else 'None'}

Spin pairs (antonyms): {', '.join(record['spin_pairs']) if record['spin_pairs'] else 'None'}
Transitions (walked to): {', '.join(record['transitions']) if record['transitions'] else 'None'}
        """)

    driver.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load semantic space into Neo4j")
    parser.add_argument("command", choices=["load", "stats", "query", "clear"],
                       help="Command to run")
    parser.add_argument("--word", help="Word to query")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")

    args = parser.parse_args()

    if args.command == "load":
        load_full_space(clear_first=args.clear)
    elif args.command == "stats":
        show_space_stats()
    elif args.command == "query":
        if args.word:
            query_word(args.word)
        else:
            print("Use --word to specify word to query")
    elif args.command == "clear":
        config = GraphConfig()
        driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("All data cleared")
        driver.close()
