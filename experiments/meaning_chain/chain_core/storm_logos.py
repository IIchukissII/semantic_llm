"""
Storm-Logos: Biological-Inspired Meaning Emergence

When a human is asked a question:
1. STORM (Neocortex): Many thoughts arise chaotically - associations, fragments, memories
2. LOGOS (Pattern): These are projected onto meaning structure, pattern emerges
3. RESPONSE: One coherent output crystallizes from the pattern

This replaces brute-force candidate generation with principled emergence.

"The storm of thoughts finds its logos in the structure of meaning"
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
import sys

# Add parent paths for imports
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from graph.meaning_graph import MeaningGraph
from models.types import MeaningNode, MeaningTree, SemanticProperties


@dataclass
class StormState:
    """A thought in the storm - concept with its activation."""
    word: str
    g: float
    tau: float
    j: np.ndarray
    activation: float = 1.0  # How strongly this thought was activated


@dataclass
class StormResult:
    """Result of the storm phase - all activated thoughts."""
    seeds: List[str]
    thoughts: List[StormState]
    visit_counts: Dict[str, int]  # How many times each concept was visited
    total_steps: int

    @property
    def resonance(self) -> Dict[str, float]:
        """Normalized activation of each concept."""
        if not self.visit_counts:
            return {}
        max_count = max(self.visit_counts.values())
        return {w: c / max_count for w, c in self.visit_counts.items()}

    @property
    def top_concepts(self) -> List[Tuple[str, int]]:
        """Most visited concepts (highest resonance)."""
        return sorted(self.visit_counts.items(), key=lambda x: -x[1])


@dataclass
class LogosPattern:
    """The meaningful pattern extracted from storm."""
    core_concepts: List[str]      # Highest resonance concepts
    convergence_point: Optional[str]  # Where thoughts meet
    j_center: np.ndarray          # Center of meaning in j-space
    g_direction: float            # Overall goodness direction
    tau_level: float              # Abstraction level of pattern
    coherence: float              # How coherent is the pattern [0, 1]


class Storm:
    """
    Storm phase: Generate chaotic thought associations.

    Like the neocortex firing - many parallel walks through
    semantic space, sampling probabilistically. No analysis,
    just activation spreading.
    """

    def __init__(self, temperature: float = 1.5):
        """
        Args:
            temperature: Controls storm chaos
                - Low T (0.5): Focused, constrained associations
                - High T (2.0): Wild, divergent associations
        """
        self.temperature = temperature
        self.loader = DataLoader()
        self._graph = None

    def _init_graph(self) -> bool:
        if self._graph is not None:
            return self._graph.is_connected()
        try:
            self._graph = MeaningGraph()
            return self._graph.is_connected()
        except:
            return False

    def _get_concept(self, word: str) -> Optional[StormState]:
        """Get storm state for a concept."""
        if not self._init_graph():
            return None
        concept = self._graph.get_concept(word)
        if not concept:
            return None
        return StormState(
            word=word,
            g=concept.get('g', 0.0),
            tau=concept.get('tau', 3.0),
            j=np.array(concept.get('j', [0]*5)),
            activation=1.0
        )

    def _get_transitions(self, word: str) -> List[Tuple[str, float]]:
        """Get transitions with weights."""
        if not self._init_graph():
            return []
        transitions = self._graph.get_all_transitions(word, limit=15)
        return [(t[1], t[2]) for t in transitions]

    def _sample_next(self, transitions: List[Tuple[str, float]]) -> Optional[str]:
        """Sample next concept using Boltzmann."""
        if not transitions:
            return None
        words = [t[0] for t in transitions]
        weights = np.array([t[1] for t in transitions])
        T = max(0.01, self.temperature)
        exp_w = np.exp(weights / T)
        probs = exp_w / np.sum(exp_w)
        return np.random.choice(words, p=probs)

    def generate(self, seeds: List[str], n_walks: int = 5,
                 steps_per_walk: int = 8) -> StormResult:
        """
        Generate storm of thoughts from seed concepts.

        Args:
            seeds: Starting concepts (from query decomposition)
            n_walks: Number of parallel walks per seed
            steps_per_walk: Steps in each walk

        Returns:
            StormResult with all activated thoughts
        """
        thoughts = []
        visit_counts = Counter()
        total_steps = 0

        for seed in seeds:
            seed_state = self._get_concept(seed)
            if not seed_state:
                continue

            # Multiple walks from this seed
            for walk in range(n_walks):
                current = seed
                visit_counts[current] += 1

                for step in range(steps_per_walk):
                    transitions = self._get_transitions(current)
                    next_word = self._sample_next(transitions)

                    if not next_word:
                        break

                    # Record the thought
                    state = self._get_concept(next_word)
                    if state:
                        # Activation decays with distance from seed
                        state.activation = 1.0 / (1.0 + step * 0.2)
                        thoughts.append(state)
                        visit_counts[next_word] += 1
                        current = next_word
                        total_steps += 1

        return StormResult(
            seeds=seeds,
            thoughts=thoughts,
            visit_counts=dict(visit_counts),
            total_steps=total_steps
        )

    def close(self):
        if self._graph:
            self._graph.close()


class Logos:
    """
    Logos phase: The LENS that focuses storm onto meaning.

    Like an optical lens focuses light rays into a coherent image,
    Logos focuses chaotic thoughts through the structure of meaning:

    - j_good: The "good" direction - ethical/aesthetic focus
    - Intent: What the user wants - purpose focus
    - Tau: Abstraction level - clarity focus

    Storm thoughts pass through this lens, and only those aligned
    with the focus emerge in the pattern.
    """

    def __init__(self):
        self.loader = DataLoader()
        self._j_good = None

    def _load_j_good(self):
        if self._j_good is None:
            self._j_good = np.array(self.loader.get_j_good())

    def _compute_focus_score(self, thought: StormState,
                              intent_j: Optional[np.ndarray] = None) -> float:
        """
        Compute how well a thought passes through the lens.

        Score = alignment with focus direction × goodness × activation

        High score = thought is focused, relevant, good
        Low score = thought is scattered, irrelevant, or negative
        """
        score = thought.activation  # Base: how strongly activated

        # Goodness lens: prefer positive g
        g_factor = (thought.g + 1) / 2  # Map [-1,1] to [0,1]
        score *= (0.5 + 0.5 * g_factor)  # Range [0.5, 1.0]

        # J-good lens: alignment with "the good"
        if thought.j is not None and np.linalg.norm(thought.j) > 1e-6:
            j_norm = thought.j / np.linalg.norm(thought.j)
            j_good_align = np.dot(j_norm, self._j_good)
            j_factor = (j_good_align + 1) / 2  # Map [-1,1] to [0,1]
            score *= (0.5 + 0.5 * j_factor)

        # Intent lens: alignment with user's intent direction
        if intent_j is not None and np.linalg.norm(intent_j) > 1e-6:
            if thought.j is not None and np.linalg.norm(thought.j) > 1e-6:
                j_norm = thought.j / np.linalg.norm(thought.j)
                intent_norm = intent_j / np.linalg.norm(intent_j)
                intent_align = np.dot(j_norm, intent_norm)
                intent_factor = (intent_align + 1) / 2
                score *= (0.5 + 0.5 * intent_factor)

        return score

    def focus(self, storm: StormResult,
              intent_j: Optional[np.ndarray] = None,
              focus_threshold: float = 0.3) -> LogosPattern:
        """
        Focus storm through the meaning lens.

        Like light through a lens, only aligned thoughts pass through.

        Args:
            storm: Chaotic thoughts from storm phase
            intent_j: Intent direction in j-space (from verbs)
            focus_threshold: Minimum focus score to include

        Returns:
            LogosPattern with focused, coherent concepts
        """
        self._load_j_good()

        if not storm.thoughts:
            return LogosPattern(
                core_concepts=storm.seeds,
                convergence_point=None,
                j_center=np.zeros(5),
                g_direction=0.0,
                tau_level=3.0,
                coherence=0.0
            )

        # Compute focus score for each thought
        focus_scores = {}
        for thought in storm.thoughts:
            score = self._compute_focus_score(thought, intent_j)
            # Accumulate scores for same word (resonance × focus)
            if thought.word in focus_scores:
                focus_scores[thought.word] += score
            else:
                focus_scores[thought.word] = score

        # Normalize by max
        if focus_scores:
            max_score = max(focus_scores.values())
            focus_scores = {w: s / max_score for w, s in focus_scores.items()}

        # Filter through lens: only focused thoughts pass
        focused_concepts = [
            (word, score) for word, score in focus_scores.items()
            if score >= focus_threshold
        ]
        focused_concepts.sort(key=lambda x: -x[1])  # Highest focus first

        # Seeds go first (they're the query concepts), then focused concepts
        core_concepts = list(storm.seeds)  # Start with seeds
        for word, score in focused_concepts[:7]:
            if word not in core_concepts:
                core_concepts.append(word)

        # Convergence: highest focused non-seed concept
        convergence_point = None
        for word, score in focused_concepts:
            if word not in storm.seeds:
                convergence_point = word
                break

        # Compute pattern properties from focused thoughts only
        focused_thoughts = [t for t in storm.thoughts if t.word in core_concepts]

        if focused_thoughts:
            j_vectors = [t.j for t in focused_thoughts if t.j is not None]
            weights = [focus_scores.get(t.word, 0) for t in focused_thoughts if t.j is not None]

            if j_vectors and weights:
                weights = np.array(weights)
                weights = weights / (np.sum(weights) + 1e-6)
                j_center = np.average(j_vectors, axis=0, weights=weights)
                g_direction = np.average([t.g for t in focused_thoughts], weights=weights)
                tau_level = np.average([t.tau for t in focused_thoughts], weights=weights)

                # Coherence: how well-focused is the result?
                j_norm = j_center / (np.linalg.norm(j_center) + 1e-6)
                alignments = [
                    np.dot(j / (np.linalg.norm(j) + 1e-6), j_norm)
                    for j in j_vectors
                ]
                coherence = (np.mean(alignments) + 1) / 2
            else:
                j_center = np.zeros(5)
                g_direction = 0.0
                tau_level = 3.0
                coherence = 0.0
        else:
            j_center = np.zeros(5)
            g_direction = 0.0
            tau_level = 3.0
            coherence = 0.0

        return LogosPattern(
            core_concepts=core_concepts,
            convergence_point=convergence_point,
            j_center=j_center,
            g_direction=g_direction,
            tau_level=tau_level,
            coherence=coherence
        )

    # Keep old method for compatibility
    def extract_pattern(self, storm: StormResult,
                        min_resonance: float = 0.3) -> LogosPattern:
        """Legacy method - calls focus()."""
        return self.focus(storm, intent_j=None, focus_threshold=min_resonance)


class StormLogosBuilder:
    """
    Builds meaning tree using Storm-Logos architecture.

    Instead of brute-force candidate generation:
    1. Storm: Generate chaotic associations (neocortex firing)
    2. Logos: Focus through meaning lens (pattern recognition)
    3. Build: Create tree from focused pattern
    """

    def __init__(self, storm_temperature: float = 1.5,
                 n_walks: int = 5, steps_per_walk: int = 8):
        self.storm = Storm(temperature=storm_temperature)
        self.logos = Logos()
        self.loader = DataLoader()
        self.n_walks = n_walks
        self.steps_per_walk = steps_per_walk
        self._graph = None

    def _init_graph(self) -> bool:
        if self._graph is not None:
            return self._graph.is_connected()
        try:
            self._graph = MeaningGraph()
            return self._graph.is_connected()
        except:
            return False

    def _get_semantic_properties(self, word: str) -> SemanticProperties:
        """Get semantic properties for a word."""
        if self._init_graph():
            concept = self._graph.get_concept(word)
            if concept:
                return SemanticProperties(
                    g=concept['g'],
                    tau=concept['tau'],
                    j=np.array(concept['j']) if concept['j'] else np.zeros(5)
                )
        return SemanticProperties(g=0.0, tau=3.0)

    def _compute_intent_j(self, verbs: List[str]) -> Optional[np.ndarray]:
        """Compute intent direction in j-space from verbs."""
        if not verbs:
            return None

        verb_ops = self.loader.load_verb_operators()
        intent_j = np.zeros(5)
        count = 0
        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

        for verb in verbs:
            if verb in verb_ops:
                vec = verb_ops[verb].get('vector', {})
                if isinstance(vec, dict):
                    # Vector is a dict with dimension names
                    try:
                        j_vec = np.array([vec.get(d, 0) for d in j_dims], dtype=float)
                        intent_j += j_vec
                        count += 1
                    except (ValueError, TypeError):
                        pass
                elif isinstance(vec, (list, tuple)) and len(vec) == 5:
                    # Vector is a list
                    try:
                        intent_j += np.array(vec, dtype=float)
                        count += 1
                    except (ValueError, TypeError):
                        pass

        if count > 0 and np.linalg.norm(intent_j) > 1e-6:
            return intent_j / np.linalg.norm(intent_j)
        return None

    def build(self, nouns: List[str], verbs: List[str],
              source_text: str = "") -> Tuple[MeaningTree, LogosPattern]:
        """
        Build meaning tree using storm-logos.

        Args:
            nouns: Seed concepts
            verbs: Intent operators (the lens direction)
            source_text: Original query

        Returns:
            (MeaningTree, LogosPattern) - the tree and the pattern it emerged from
        """
        # Phase 1: STORM (neocortex firing)
        # Let thoughts spread chaotically from seeds
        storm_result = self.storm.generate(
            seeds=nouns,
            n_walks=self.n_walks,
            steps_per_walk=self.steps_per_walk
        )

        # Compute intent direction from verbs (the lens)
        intent_j = self._compute_intent_j(verbs)

        # Phase 2: LOGOS (focus through meaning lens)
        # Verbs act as lens direction - filter to intent-aligned thoughts
        pattern = self.logos.focus(storm_result, intent_j=intent_j)

        # Phase 3: BUILD TREE
        # Create focused tree from pattern
        tree = self._build_from_pattern(pattern, source_text)

        return tree, pattern

    def _build_from_pattern(self, pattern: LogosPattern,
                            source_text: str) -> MeaningTree:
        """Build tree from extracted pattern - the focused structure."""
        tree = MeaningTree(
            roots=[],
            max_depth=2,  # Shallow tree - pattern already filtered
            source_text=source_text
        )

        if not pattern.core_concepts:
            return tree

        # Use convergence point as root if available and in core concepts
        if pattern.convergence_point and pattern.convergence_point in pattern.core_concepts:
            root_word = pattern.convergence_point
        else:
            root_word = pattern.core_concepts[0]

        # Verify root exists in graph
        if not self._init_graph() or not self._graph.has_concept(root_word):
            # Fallback to first available core concept
            for word in pattern.core_concepts:
                if self._graph.has_concept(word):
                    root_word = word
                    break

        props = self._get_semantic_properties(root_word)
        root_node = MeaningNode(
            word=root_word,
            properties=props,
            depth=0
        )

        # Add other core concepts as children
        seen = {root_word}
        for child_word in pattern.core_concepts:
            if child_word in seen:
                continue
            if len(root_node.children) >= 4:
                break
            if not self._graph.has_concept(child_word):
                continue

            # Try to find actual verb connection
            verb = self._find_connection(root_word, child_word)

            child_props = self._get_semantic_properties(child_word)
            child_node = MeaningNode(
                word=child_word,
                properties=child_props,
                depth=1,
                verb_from_parent=verb
            )
            root_node.children.append(child_node)
            seen.add(child_word)

        tree.roots.append(root_node)
        return tree

    def _find_connection(self, from_word: str, to_word: str) -> str:
        """Find verb connecting two concepts, or return generic."""
        if not self._init_graph():
            return "relates"

        transitions = self._graph.get_all_transitions(from_word, limit=50)
        for verb, target, _ in transitions:
            if target == to_word:
                return verb

        return "relates"

    def close(self):
        self.storm.close()
        if self._graph:
            self._graph.close()


def demo():
    """Demonstrate storm-logos architecture."""
    print("=" * 60)
    print("STORM-LOGOS: Biological Meaning Emergence")
    print("=" * 60)

    builder = StormLogosBuilder(
        storm_temperature=1.5,
        n_walks=5,
        steps_per_walk=8
    )

    # Test query
    nouns = ["dream", "love"]
    verbs = ["understand", "find"]

    print(f"\nSeeds: {nouns}")
    print(f"Intent (lens): {verbs}")

    print("\n--- Phase 1: STORM (neocortex firing) ---")
    storm = builder.storm.generate(nouns, n_walks=5, steps_per_walk=8)
    print(f"Total thoughts: {len(storm.thoughts)}")
    print(f"Unique concepts: {len(storm.visit_counts)}")
    print(f"Raw resonance (before lens):")
    for word, count in storm.top_concepts[:8]:
        bar = "█" * min(count, 20)
        print(f"  {word:15s} {bar} ({count})")

    print("\n--- Phase 2: LOGOS (focus through lens) ---")
    intent_j = builder._compute_intent_j(verbs)
    print(f"Intent j-direction: {intent_j}")
    pattern = builder.logos.focus(storm, intent_j=intent_j)
    print(f"\nAfter focusing through lens:")
    print(f"  Core concepts: {pattern.core_concepts[:5]}")
    print(f"  Convergence: {pattern.convergence_point}")
    print(f"  Coherence: {pattern.coherence:.2f}")
    print(f"  G direction: {pattern.g_direction:+.2f}")
    print(f"  Tau level: {pattern.tau_level:.1f}")

    print("\n--- Phase 3: BUILD (focused tree) ---")
    tree, _ = builder.build(nouns, verbs, "what does love mean in dreams?")
    print(tree.pretty_print())

    builder.close()


if __name__ == "__main__":
    demo()
