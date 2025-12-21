"""
Semantic Loader: Load real semantic data from QuantumCore.

Bridges the semantic RL environment with the full 19K word semantic space.
"""

import sys
from pathlib import Path

# Add path to find QuantumCore BEFORE other imports
# semantic_rl/src/core/semantic_loader.py -> semantic_llm/
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM_PATH = _THIS_FILE.parent.parent.parent.parent.parent
if str(_SEMANTIC_LLM_PATH) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_LLM_PATH))

import numpy as np
from typing import Optional, List, Tuple

from .semantic_state import SemanticState, SemanticGraph, Transition


def _import_quantum_core():
    """Import QuantumCore from the semantic_llm package."""

    # Compute path dynamically to avoid import-time issues
    this_file = Path(__file__).resolve()
    semantic_llm_path = this_file.parent.parent.parent.parent.parent

    # Verify path
    hybrid_llm_file = semantic_llm_path / "core" / "hybrid_llm.py"
    if not hybrid_llm_file.exists():
        print(f"Cannot find: {hybrid_llm_file}")
        return None

    # Temporarily modify sys.path to prioritize semantic_llm
    original_path = sys.path.copy()

    try:
        # Put semantic_llm at front, remove any semantic_rl paths
        sys.path = [str(semantic_llm_path)] + [
            p for p in original_path if 'semantic_rl' not in p
        ]

        # Import using exec to avoid module caching issues
        import importlib
        if 'core.hybrid_llm' in sys.modules:
            del sys.modules['core.hybrid_llm']
        if 'core.data_loader' in sys.modules:
            del sys.modules['core.data_loader']
        if 'core' in sys.modules:
            del sys.modules['core']

        from core.hybrid_llm import QuantumCore
        return QuantumCore

    except Exception as e:
        import traceback
        print(f"Import error: {e}")
        traceback.print_exc()
        return None
    finally:
        # Restore original path
        sys.path = original_path


class SemanticLoader:
    """
    Loads semantic data from the QuantumCore system.

    Provides:
    - 19,055 word states with τ, g, j properties
    - 2,444 verbs for transitions
    - 93 spin pairs for tunneling
    """

    def __init__(self):
        self.core = None
        self.graph = None
        self._load_core()

    def _load_core(self):
        """Load the QuantumCore."""
        try:
            QuantumCore = _import_quantum_core()
            if QuantumCore is None:
                raise ImportError("Could not find hybrid_llm.py")

            print("Loading QuantumCore...")
            self.core = QuantumCore()
            print(f"  Loaded {len(self.core.states)} states")
            print(f"  Loaded {len(self.core.verb_objects)} verb-object pairs")
            print(f"  Loaded {len(self.core.spin_pairs)} spin pairs")
        except Exception as e:
            print(f"Warning: Could not load QuantumCore: {e}")
            import traceback
            traceback.print_exc()
            print("Using fallback semantic data...")
            self.core = None

    def build_graph(self,
                    max_states: int = 1000,
                    sample_verbs: int = 100) -> SemanticGraph:
        """
        Build SemanticGraph from QuantumCore data.

        Args:
            max_states: Maximum number of states to include
            sample_verbs: Number of verbs to sample for transitions

        Returns:
            SemanticGraph populated with real semantic data
        """
        graph = SemanticGraph()

        if self.core is None:
            print("No QuantumCore available, returning empty graph")
            return graph

        # Select states (prioritize interesting ones)
        selected_words = self._select_interesting_states(max_states)
        print(f"Selected {len(selected_words)} states for graph")

        # Add states
        for word in selected_words:
            state_data = self.core.states.get(word)
            if state_data is None:
                continue

            # QuantumCore's SemanticState has 'j' (5D array), not 'j_norm'
            # Expand to 16D if needed for our local SemanticState
            j_vector = state_data.j
            if len(j_vector) < 16:
                # Pad with zeros to 16D
                j_vector = np.concatenate([j_vector, np.zeros(16 - len(j_vector))])

            state = SemanticState(
                word=word,
                tau=state_data.tau,
                goodness=state_data.goodness,
                j_vector=j_vector,
                believe_modifier=self._compute_believe_modifier(state_data)
            )
            graph.add_state(state)

        # Add transitions (verb connections)
        self._add_transitions(graph, selected_words, sample_verbs)

        print(f"Graph built: {len(graph.states)} states, "
              f"{sum(len(t) for t in graph.transitions.values())} transitions")

        self.graph = graph
        return graph

    def _select_interesting_states(self, max_states: int) -> List[str]:
        """Select interesting states for the graph."""
        if self.core is None:
            return []

        # Priority words (concepts we want to include)
        priority = [
            # Journey concepts
            "darkness", "light", "fear", "courage", "hope", "despair",
            "struggle", "wisdom", "truth", "love", "hate", "peace", "war",
            "life", "death", "birth", "growth", "decay", "change",
            # Abstract concepts
            "beauty", "ugliness", "good", "evil", "order", "chaos",
            "freedom", "slavery", "justice", "injustice", "mercy", "cruelty",
            # Emotional states
            "joy", "sadness", "anger", "calm", "anxiety", "serenity",
            "trust", "betrayal", "loneliness", "connection", "isolation",
            # Action concepts
            "creation", "destruction", "discovery", "loss", "victory", "defeat",
        ]

        selected = []

        # Add priority words that exist
        for word in priority:
            if word in self.core.states:
                selected.append(word)

        # Add spin pair words (for tunneling)
        # spin_pairs is a Dict[str, SpinPair] with .base and .prefixed
        seen_pairs = set()
        for word, pair in self.core.spin_pairs.items():
            pair_key = tuple(sorted([pair.base, pair.prefixed]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            if pair.base in self.core.states and pair.base not in selected:
                selected.append(pair.base)
            if pair.prefixed in self.core.states and pair.prefixed not in selected:
                selected.append(pair.prefixed)
            if len(selected) >= max_states:
                break

        # Fill remaining with high-variance goodness words
        if len(selected) < max_states:
            all_words = list(self.core.states.keys())
            # Sort by absolute goodness (interesting extremes)
            all_words.sort(key=lambda w: abs(self.core.states[w].goodness), reverse=True)

            for word in all_words:
                if word not in selected:
                    selected.append(word)
                if len(selected) >= max_states:
                    break

        return selected[:max_states]

    def _compute_believe_modifier(self, state_data) -> float:
        """Compute believe modifier from state properties."""
        # Positive goodness boosts believe
        # High abstraction slightly reduces (need grounding)
        g_effect = state_data.goodness * 0.2
        tau_effect = -state_data.tau * 0.05
        return np.clip(g_effect + tau_effect, -0.3, 0.3)

    def _add_transitions(self, graph: SemanticGraph,
                         words: List[str],
                         sample_verbs: int):
        """Add verb transitions to the graph."""
        if self.core is None:
            return

        import random

        # verb_objects: verb -> [list of objects]
        all_verbs = list(self.core.verb_objects.keys())
        sampled_verbs = random.sample(all_verbs, min(sample_verbs, len(all_verbs)))

        words_set = set(words)

        # For each sampled verb, create transitions
        for verb in sampled_verbs:
            objects = self.core.verb_objects.get(verb, [])
            if not objects:
                continue

            # Find objects that are in our word set
            valid_objects = [o for o in objects if o in words_set]
            if not valid_objects:
                continue

            # Create transitions from other words to these objects
            for to_word in valid_objects[:5]:  # Limit per verb
                to_state = graph.get_state(to_word)
                if to_state is None:
                    continue

                # Pick random source words
                for from_word in random.sample(words, min(3, len(words))):
                    if from_word != to_word:
                        from_state = graph.get_state(from_word)
                        if from_state:
                            delta_g = to_state.goodness - from_state.goodness
                            graph.add_transition(Transition(
                                verb=verb,
                                from_state=from_word,
                                to_state=to_word,
                                delta_g=delta_g
                            ))

    def get_state(self, word: str) -> Optional[SemanticState]:
        """Get a semantic state by word."""
        if self.graph and word in self.graph.states:
            return self.graph.states[word]

        if self.core and word in self.core.states:
            state_data = self.core.states[word]
            j_vector = state_data.j
            if len(j_vector) < 16:
                j_vector = np.concatenate([j_vector, np.zeros(16 - len(j_vector))])
            return SemanticState(
                word=word,
                tau=state_data.tau,
                goodness=state_data.goodness,
                j_vector=j_vector
            )

        return None

    def get_spin_pairs(self) -> List[Tuple[str, str]]:
        """Get spin pairs for tunneling."""
        if self.core:
            # spin_pairs is Dict[str, SpinPair], extract unique (base, prefixed) tuples
            seen = set()
            pairs = []
            for pair in self.core.spin_pairs.values():
                pair_key = tuple(sorted([pair.base, pair.prefixed]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs.append((pair.base, pair.prefixed))
            return pairs
        return []

    def tunnel_probability(self, word1: str, word2: str) -> float:
        """Compute tunneling probability between two words."""
        s1 = self.get_state(word1)
        s2 = self.get_state(word2)

        if s1 is None or s2 is None:
            return 0.0

        return s1.tunnel_probability(s2)


# Quick test
if __name__ == "__main__":
    loader = SemanticLoader()
    graph = loader.build_graph(max_states=100, sample_verbs=50)

    print("\nSample states:")
    for word in ["darkness", "hope", "wisdom", "fear", "love"]:
        state = graph.get_state(word)
        if state:
            print(f"  {state}")

    print("\nSample transitions from 'darkness':")
    neighbors = graph.get_neighbors("darkness")
    for to_word, verb, delta_g in neighbors[:5]:
        print(f"  --{verb}--> {to_word} (Δg={delta_g:+.2f})")

    print("\nSpin pairs (tunneling):")
    for w1, w2 in loader.get_spin_pairs()[:5]:
        p = loader.tunnel_probability(w1, w2)
        print(f"  {w1} ↔ {w2} (P={p:.3f})")
