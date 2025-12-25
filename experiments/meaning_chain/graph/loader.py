"""
Loader - Populate MeaningGraph from semantic data.

Sources:
1. Semantic space (Wholeness) - concepts with g, tau, j
2. SVO patterns - subject-verb-object triplets
3. Verb operators - verb j-vectors
4. Verb objects - what verbs typically operate on
"""

import sys
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass

# Add paths
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_EXPERIMENTS = _MEANING_CHAIN.parent
_SEMANTIC_LLM = _EXPERIMENTS.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from graph.meaning_graph import MeaningGraph, GraphConfig


@dataclass
class LoaderConfig:
    """Configuration for loading."""
    min_tau: float = 1.5          # Skip function words
    min_word_length: int = 3      # Skip very short words
    max_concepts: int = 50000     # Limit concepts (0 = no limit)
    max_svo_per_subject: int = 20 # Limit SVO patterns per subject
    verbose: bool = True


class MeaningGraphLoader:
    """
    Load semantic data into MeaningGraph.

    Populates:
    - Concepts from Wholeness (semantic space)
    - VIA relationships from SVO patterns
    - VerbOperators from verb_operators
    - OPERATES_ON from verb_objects
    """

    def __init__(self, graph: MeaningGraph = None,
                 data_loader: DataLoader = None,
                 config: LoaderConfig = None):
        self.graph = graph or MeaningGraph()
        self.loader = data_loader or DataLoader()
        self.config = config or LoaderConfig()

        # Track loaded data
        self.concepts_loaded: Set[str] = set()
        self.verbs_loaded: Set[str] = set()

    def _log(self, message: str):
        """Log if verbose."""
        if self.config.verbose:
            print(message)

    def load_all(self, clear_first: bool = False):
        """
        Load all data into graph.

        Args:
            clear_first: Clear graph before loading
        """
        if not self.graph.is_connected():
            print("[Loader] Not connected to Neo4j")
            return

        if clear_first:
            self._log("[Loader] Clearing existing data...")
            self.graph.clear_all()

        self._log("[Loader] Setting up schema...")
        self.graph.setup_schema()

        self._log("\n[Loader] Loading concepts...")
        self.load_concepts()

        self._log("\n[Loader] Loading verb operators...")
        self.load_verb_operators()

        self._log("\n[Loader] Loading SVO patterns (VIA relationships)...")
        self.load_svo_patterns()

        self._log("\n[Loader] Loading verb-object relationships (OPERATES_ON)...")
        self.load_verb_objects()

        # Final stats
        stats = self.graph.get_stats()
        self._log("\n" + "=" * 60)
        self._log("LOADING COMPLETE")
        self._log("=" * 60)
        self._log(f"  Concepts: {stats['concepts']}")
        self._log(f"  Verb Operators: {stats['verb_operators']}")
        self._log(f"  VIA edges: {stats['via_edges']}")
        self._log(f"  OPERATES_ON edges: {stats['operates_on_edges']}")
        self._log(f"  Sample verbs: {stats['sample_verbs'][:5]}")

    def load_concepts(self):
        """Load concepts from semantic space."""
        # Try to load from Wholeness (full semantic space)
        try:
            sys.path.insert(0, str(_EXPERIMENTS / "experience_knowledge"))
            from layers.core import Wholeness
            wholeness = Wholeness()

            concepts = []
            for word, state in wholeness.states.items():
                # Filter
                if state.tau < self.config.min_tau:
                    continue
                if len(word) < self.config.min_word_length:
                    continue

                concepts.append({
                    "word": word,
                    "g": float(state.goodness),
                    "tau": float(state.tau),
                    "j": state.j.tolist(),
                    "pos": "noun"  # Wholeness is mainly nouns
                })
                self.concepts_loaded.add(word)

                if self.config.max_concepts > 0 and len(concepts) >= self.config.max_concepts:
                    break

            self._log(f"  Loading {len(concepts)} concepts from Wholeness...")
            self.graph.create_concept_batch(concepts)
            self._log(f"  Done: {len(concepts)} concepts loaded")
            return

        except ImportError:
            self._log("  Wholeness not available, falling back to word_vectors...")

        # Fallback to word vectors
        word_vectors = self.loader.load_word_vectors()

        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']
        concepts = []

        for word, data in word_vectors.items():
            # Filter
            tau = data.get('tau', 3.0)
            if tau < self.config.min_tau:
                continue
            if len(word) < self.config.min_word_length:
                continue
            if not data.get('j'):
                continue

            j_vec = [data['j'].get(d, 0) for d in j_dims]
            g = sum(j_vec) / len(j_vec)  # Approximate goodness

            concepts.append({
                "word": word,
                "g": g,
                "tau": tau,
                "j": j_vec,
                "pos": "noun"
            })
            self.concepts_loaded.add(word)

            if self.config.max_concepts > 0 and len(concepts) >= self.config.max_concepts:
                break

        self._log(f"  Loading {len(concepts)} concepts from word_vectors...")
        self.graph.create_concept_batch(concepts)
        self._log(f"  Done: {len(concepts)} concepts loaded")

    def load_verb_operators(self):
        """Load verb operators with j-vectors."""
        verb_operators = self.loader.load_verb_operators()

        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']
        verbs = []

        for verb, data in verb_operators.items():
            if 'vector' not in data:
                continue

            j_vec = [data['vector'].get(d, 0) for d in j_dims]
            magnitude = data.get('magnitude', 1.0)

            verbs.append({
                "verb": verb,
                "j": j_vec,
                "magnitude": magnitude,
                "objects": []  # Will be filled by verb_objects
            })
            self.verbs_loaded.add(verb)

        self._log(f"  Loading {len(verbs)} verb operators...")
        self.graph.create_verb_operator_batch(verbs)
        self._log(f"  Done: {len(verbs)} verb operators loaded")

    def load_svo_patterns(self):
        """Load SVO patterns as VIA relationships."""
        svo_patterns = self.loader.load_svo_patterns()

        transitions = []
        subjects_processed = 0

        for subject, patterns in svo_patterns.items():
            # Skip if subject not in concepts
            if subject not in self.concepts_loaded:
                continue

            subjects_processed += 1
            count = 0

            for verb, obj in patterns:
                # Skip if object not in concepts
                if obj not in self.concepts_loaded:
                    continue

                transitions.append({
                    "subject": subject,
                    "object": obj,
                    "verb": verb,
                    "weight": 1.0,
                    "count": 1,
                    "source": "svo"
                })
                count += 1

                if count >= self.config.max_svo_per_subject:
                    break

        self._log(f"  Loading {len(transitions)} VIA relationships...")
        self._log(f"  From {subjects_processed} subjects")
        self.graph.create_via_batch(transitions)
        self._log(f"  Done: {len(transitions)} VIA edges created")

    def load_verb_objects(self):
        """Load verb-object relationships (OPERATES_ON)."""
        verb_objects = self.loader.load_verb_objects()

        relations = []

        for verb, objects in verb_objects.items():
            # Skip if verb not loaded
            if verb not in self.verbs_loaded:
                continue

            for obj in objects[:10]:  # Limit per verb
                # Skip if object not in concepts
                if obj not in self.concepts_loaded:
                    continue

                relations.append({
                    "verb": verb,
                    "concept": obj,
                    "weight": 1.0
                })

        self._log(f"  Loading {len(relations)} OPERATES_ON relationships...")
        self.graph.create_operates_on_batch(relations)
        self._log(f"  Done: {len(relations)} OPERATES_ON edges created")

    def close(self):
        """Close connections."""
        if self.graph:
            self.graph.close()


def main():
    """CLI for loading data."""
    import argparse

    parser = argparse.ArgumentParser(description="Load data into MeaningGraph")
    parser.add_argument("command", choices=["load", "stats", "clear"],
                        help="Command to run")
    parser.add_argument("--clear", action="store_true",
                        help="Clear graph before loading")
    parser.add_argument("--max-concepts", type=int, default=50000,
                        help="Maximum concepts to load")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output")

    args = parser.parse_args()

    config = LoaderConfig(
        max_concepts=args.max_concepts,
        verbose=not args.quiet
    )

    graph = MeaningGraph()
    loader = MeaningGraphLoader(graph, config=config)

    if args.command == "load":
        loader.load_all(clear_first=args.clear)

    elif args.command == "stats":
        if graph.is_connected():
            stats = graph.get_stats()
            print("\n" + "=" * 60)
            print("MeaningGraph Statistics")
            print("=" * 60)
            print(f"  Concepts: {stats['concepts']}")
            print(f"  Verb Operators: {stats['verb_operators']}")
            print(f"  VIA edges: {stats['via_edges']}")
            print(f"  OPERATES_ON edges: {stats['operates_on_edges']}")
            print(f"  Sample verbs: {stats['sample_verbs']}")

    elif args.command == "clear":
        if graph.is_connected():
            confirm = input("Clear entire MeaningGraph? (yes/no): ")
            if confirm.lower() == "yes":
                graph.clear_all()

    loader.close()


if __name__ == "__main__":
    main()
