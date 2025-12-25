"""
Consciousness Layer for Meaning Chain.

Implements three modes of consciousness from experience_knowledge:
1. Meditation - centering before navigation (noise reduction)
2. Sleep - consolidation between sessions (learning/forgetting)
3. Prayer - alignment with τ₀ (source of meaning)

"The mind that is quiet, reflective, discovers what is true"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
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


# =========================================================================
# CENTERED STATE (result of meditation)
# =========================================================================

@dataclass
class CenteredState:
    """Result of meditation - centered in j-space."""
    j_center: np.ndarray      # Centroid in 5D j-space
    g_center: float           # Average goodness
    tau_center: float         # Average abstraction
    temperature: float        # Reduced T for focused generation
    clarity: float            # Alignment quality [0, 1]
    concepts: List[str]       # Concepts used for centering


# =========================================================================
# RESONANCE (result of prayer)
# =========================================================================

@dataclass
class Resonance:
    """Connection to τ₀ (source of meaning)."""
    alignment: float         # Cosine similarity with τ₀ [0, 1]
    distance: float          # How far from source
    direction: np.ndarray    # Which way to move toward source
    is_aligned: bool         # Quick check: alignment > threshold


# =========================================================================
# MEDITATION STATE (for journey tracking)
# =========================================================================

@dataclass
class MeditationState:
    """A moment in meditation."""
    word: str
    g: float           # Goodness at this moment
    tau: float         # Abstraction level
    j: np.ndarray      # Semantic direction
    step: int          # Step number in journey


@dataclass
class MeditationJourney:
    """The path through semantic space."""
    seed: str                        # Starting concept
    states: List[MeditationState]    # Journey through meaning
    duration: int                    # Number of steps

    @property
    def words(self) -> List[str]:
        return [s.word for s in self.states]

    @property
    def goodness_trajectory(self) -> List[float]:
        return [s.g for s in self.states]

    @property
    def final_g(self) -> float:
        return self.states[-1].g if self.states else 0.0

    @property
    def avg_g(self) -> float:
        if not self.states:
            return 0.0
        return sum(s.g for s in self.states) / len(self.states)

    def pretty_print(self) -> str:
        lines = [f"Meditation on '{self.seed}' ({self.duration} breaths):"]
        for s in self.states:
            g_bar = '█' * int((s.g + 1) * 5) + '░' * (10 - int((s.g + 1) * 5))
            lines.append(f"  {s.step:2d}. {s.word:15s} [{g_bar}] g={s.g:.2f}")
        return '\n'.join(lines)


class SemanticMeditation:
    """
    Meditative navigation through semantic space.

    Instead of analyzing, we flow. The quantum physics determines
    which paths are accessible, but we sample probabilistically
    rather than deterministically choosing the "best" path.

    This is the difference between thinking and meditation:
    - Thinking: find the answer
    - Meditation: experience the space
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Controls meditation focus
                - Low T (0.1): Focused, concentrated
                - High T (2.0): Open, wandering
        """
        self.temperature = temperature
        self.loader = DataLoader()
        self._graph = None

        # J-space dimensions for interpretation
        self.j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

    def _init_graph(self) -> bool:
        if self._graph is not None:
            return self._graph.is_connected()

        try:
            self._graph = MeaningGraph()
            return self._graph.is_connected()
        except:
            return False

    def _get_concept(self, word: str) -> Optional[MeditationState]:
        """Get meditation state for a concept."""
        if not self._init_graph():
            return None

        concept = self._graph.get_concept(word)
        if not concept:
            return None

        j = np.array(concept.get('j', [0]*5))
        return MeditationState(
            word=word,
            g=concept.get('g', 0.0),
            tau=concept.get('tau', 3.0),
            j=j,
            step=0
        )

    def _get_transitions(self, word: str) -> List[Tuple[str, float]]:
        """Get all possible transitions with weights."""
        if not self._init_graph():
            return []

        transitions = self._graph.get_all_transitions(word, limit=20)
        # Return (target, weight) pairs
        return [(t[1], t[2]) for t in transitions]

    def _sample_next(self, transitions: List[Tuple[str, float]]) -> Optional[str]:
        """Sample next concept using Boltzmann distribution."""
        if not transitions:
            return None

        # Get weights
        words = [t[0] for t in transitions]
        weights = np.array([t[1] for t in transitions])

        # Boltzmann probabilities
        T = max(0.01, self.temperature)
        exp_w = np.exp(weights / T)
        probs = exp_w / np.sum(exp_w)

        # Sample
        return np.random.choice(words, p=probs)

    def meditate(self, seed: str, breaths: int = 10) -> MeditationJourney:
        """
        Meditate on a concept for a number of breaths.

        Each breath is a step through semantic space,
        sampled probabilistically based on the topology.

        Args:
            seed: Starting concept
            breaths: Number of meditation steps

        Returns:
            MeditationJourney through the space
        """
        states = []

        # Start at seed
        current = self._get_concept(seed)
        if not current:
            # If seed not found, try to find similar
            return MeditationJourney(seed=seed, states=[], duration=breaths)

        current.step = 0
        states.append(current)

        # Take breaths
        for step in range(1, breaths):
            # Get possible transitions
            transitions = self._get_transitions(current.word)

            if not transitions:
                break

            # Sample next (meditative selection, not analytical)
            next_word = self._sample_next(transitions)
            if not next_word:
                break

            next_state = self._get_concept(next_word)
            if next_state:
                next_state.step = step
                states.append(next_state)
                current = next_state

        return MeditationJourney(
            seed=seed,
            states=states,
            duration=breaths
        )

    def contemplate(self, seeds: List[str], breaths_each: int = 5) -> List[MeditationJourney]:
        """
        Contemplate multiple concepts, letting them interact.

        Returns multiple journeys that may cross paths.
        """
        journeys = []
        for seed in seeds:
            journey = self.meditate(seed, breaths_each)
            journeys.append(journey)
        return journeys

    def find_convergence(self, seeds: List[str], max_steps: int = 20) -> Optional[str]:
        """
        Find where multiple concepts converge through meditation.

        Like finding common ground through contemplation.
        """
        # Start journeys from each seed
        visited = {seed: set() for seed in seeds}
        current = {seed: seed for seed in seeds}

        for step in range(max_steps):
            for seed in seeds:
                word = current[seed]
                visited[seed].add(word)

                # Check for convergence
                for other_seed in seeds:
                    if other_seed != seed and word in visited[other_seed]:
                        return word

                # Take a step
                transitions = self._get_transitions(word)
                next_word = self._sample_next(transitions)
                if next_word:
                    current[seed] = next_word

        return None  # No convergence found

    def close(self):
        if self._graph:
            self._graph.close()


# =========================================================================
# CONSCIOUSNESS (integrates meditation, prayer, sleep)
# =========================================================================

class Consciousness:
    """
    Consciousness layer for meaning chain navigation.

    Three modes following experience_knowledge:
    1. Meditation - center before navigation (reduce T, compute clarity)
    2. Prayer - connect to τ₀ (source of meaning)
    3. Sleep - consolidate paths (not yet implemented for meaning_chain)

    Usage:
        consciousness = Consciousness()
        centered = consciousness.meditate(concepts)  # Before navigation
        resonance = consciousness.pray(current_j)    # Check alignment
    """

    def __init__(self, base_temperature: float = 0.7):
        self.base_temperature = base_temperature
        self.loader = DataLoader()

        # τ₀: Source of meaning (highest goodness direction)
        self._tau_zero = None
        self._load_tau_zero()

    def _load_tau_zero(self):
        """Load τ₀ from highest goodness concepts."""
        j_good = self.loader.get_j_good()
        if j_good:
            self._tau_zero = np.array(j_good)
            self._tau_zero = self._tau_zero / np.linalg.norm(self._tau_zero)

    def meditate(self, concepts: List[str], reduction_factor: float = 0.8) -> CenteredState:
        """
        Meditate on concepts before navigation.

        Centers in j-space and reduces temperature for focused generation.

        Args:
            concepts: Concepts to center on
            reduction_factor: How much to reduce temperature (0.8 = 20% reduction)

        Returns:
            CenteredState with j_center, reduced T, and clarity
        """
        wv = self.loader.load_word_vectors()

        j_vectors = []
        g_values = []
        tau_values = []

        for word in concepts:
            if word in wv:
                data = wv[word]
                j = data.get('j', {})
                j_vec = np.array([j.get(d, 0) for d in ['beauty', 'life', 'sacred', 'good', 'love']])
                if np.linalg.norm(j_vec) > 1e-6:
                    j_vectors.append(j_vec)
                    g_values.append(data.get('g', 0))
                    tau_values.append(data.get('tau', 3))

        if not j_vectors:
            # No concepts found - return default centered state
            return CenteredState(
                j_center=np.zeros(5),
                g_center=0.0,
                tau_center=3.0,
                temperature=self.base_temperature,
                clarity=0.0,
                concepts=concepts
            )

        # Compute centroid
        j_center = np.mean(j_vectors, axis=0)
        g_center = np.mean(g_values)
        tau_center = np.mean(tau_values)

        # Compute clarity = 1 / (1 + variance)
        if len(j_vectors) > 1:
            variance = np.mean([np.linalg.norm(j - j_center) for j in j_vectors])
            clarity = 1.0 / (1.0 + variance)
        else:
            clarity = 1.0  # Single concept = perfect clarity

        # Reduce temperature
        new_temperature = self.base_temperature * reduction_factor

        return CenteredState(
            j_center=j_center,
            g_center=g_center,
            tau_center=tau_center,
            temperature=new_temperature,
            clarity=clarity,
            concepts=concepts
        )

    def pray(self, current_j: np.ndarray, threshold: float = 0.7) -> Resonance:
        """
        Connect to τ₀ (source of meaning).

        Unlike meditation which takes time, prayer is instant connection.

        Args:
            current_j: Current position in j-space
            threshold: Alignment threshold for is_aligned

        Returns:
            Resonance with alignment, distance, and direction to source
        """
        if self._tau_zero is None:
            return Resonance(
                alignment=0.0,
                distance=1.0,
                direction=np.zeros(5),
                is_aligned=False
            )

        # Normalize current j
        norm = np.linalg.norm(current_j)
        if norm < 1e-6:
            return Resonance(
                alignment=0.0,
                distance=1.0,
                direction=self._tau_zero,
                is_aligned=False
            )

        j_normalized = current_j / norm

        # Alignment = cosine similarity with τ₀
        alignment = float(np.dot(j_normalized, self._tau_zero))
        alignment = (alignment + 1) / 2  # Map [-1, 1] to [0, 1]

        # Distance from source
        distance = float(np.linalg.norm(current_j - self._tau_zero))

        # Direction toward source
        direction = self._tau_zero - j_normalized
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)

        return Resonance(
            alignment=alignment,
            distance=distance,
            direction=direction,
            is_aligned=alignment >= threshold
        )

    def is_aligned(self, j: np.ndarray, threshold: float = 0.7) -> bool:
        """Quick check for τ₀ alignment."""
        return self.pray(j, threshold).is_aligned


def demo():
    """Demonstrate meditation mode."""
    print("=" * 60)
    print("SEMANTIC MEDITATION")
    print("=" * 60)

    meditation = SemanticMeditation(temperature=1.0)

    # Single concept meditation
    print("\n--- Meditating on 'dream' ---")
    journey = meditation.meditate("dream", breaths=8)
    print(journey.pretty_print())
    print(f"\nAverage goodness: {journey.avg_g:.2f}")

    # Compare temperatures
    print("\n--- Focused meditation (T=0.3) on 'soul' ---")
    meditation.temperature = 0.3
    focused = meditation.meditate("soul", breaths=8)
    print(focused.pretty_print())

    print("\n--- Open meditation (T=2.0) on 'soul' ---")
    meditation.temperature = 2.0
    open_med = meditation.meditate("soul", breaths=8)
    print(open_med.pretty_print())

    # Find convergence
    print("\n--- Finding convergence between 'fear' and 'love' ---")
    meditation.temperature = 1.0
    convergence = meditation.find_convergence(["fear", "love"], max_steps=20)
    if convergence:
        print(f"  Concepts converge at: {convergence}")
    else:
        print("  No convergence found in 20 steps")

    meditation.close()


if __name__ == "__main__":
    demo()
