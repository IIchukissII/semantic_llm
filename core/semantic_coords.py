"""
Semantic Coordinates: (A, S, τ) Encoding API

The minimal semantic representation:
- Words encoded as (A, S, τ) - 3 numbers
- Verbs encoded as (ΔA, ΔS) - 2 numbers

Compression: 100-250x vs standard embeddings.

This is the LOOKUP-BASED API (O(1) encoding).
For neural network training, see semantic_bottleneck.py.

Usage:
    from core.semantic_coords import BottleneckEncoder

    encoder = BottleneckEncoder()

    # Encode word
    coord = encoder.encode_word("truth")
    print(coord)  # SemanticCoord(A=0.865, S=-0.175, tau=2.40, word='truth')

    # Encode verb
    op = encoder.encode_verb("love")
    print(op)  # VerbOperator(dA=-0.172, dS=0.165, verb='love')

    # Apply verb transformation
    new_coord = encoder.apply(coord, "love")
    # Or: new_coord = coord + op

    # Find nearest words
    neighbors = encoder.nearest(coord, k=5)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import json
import numpy as np
import math


# Principal Component Vectors (validated exp10b)
PC1_AFFIRMATION = np.array([-0.448, -0.519, -0.118, -0.480, -0.534])
PC2_SACRED = np.array([-0.513, +0.128, -0.732, +0.420, +0.090])
DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


@dataclass
class SemanticCoord:
    """
    Semantic coordinate in (A, S, τ) space.

    A = Affirmation (PC1, 83.3% variance): life, love, good, beauty
    S = Sacred (PC2, 11.7% variance): transcendence dimension
    τ = Tau (abstraction level): 1=concrete, 7=transcendental

    The Veil is at τ = e ≈ 2.718.
    """
    A: float           # Affirmation [-3, +3]
    S: float           # Sacred [-2, +2]
    tau: float         # Abstraction [1, 7]
    word: str = ""     # Source word (optional)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [A, S, τ]."""
        return np.array([self.A, self.S, self.tau])

    def to_2d(self) -> np.ndarray:
        """Convert to 2D array [A, S] (for operators)."""
        return np.array([self.A, self.S])

    def distance(self, other: SemanticCoord) -> float:
        """Euclidean distance in (A, S, τ) space."""
        return math.sqrt(
            (self.A - other.A) ** 2 +
            (self.S - other.S) ** 2 +
            (self.tau - other.tau) ** 2
        )

    def distance_2d(self, other: SemanticCoord) -> float:
        """Euclidean distance in (A, S) space only."""
        return math.sqrt(
            (self.A - other.A) ** 2 +
            (self.S - other.S) ** 2
        )

    def __add__(self, op: VerbOperator) -> SemanticCoord:
        """Apply verb operator: coord + verb → new_coord."""
        return SemanticCoord(
            A=self.A + op.dA,
            S=self.S + op.dS,
            tau=self.tau,  # τ unchanged by verb
            word=f"{op.verb}({self.word})" if self.word else ""
        )

    def __sub__(self, other: SemanticCoord) -> Tuple[float, float, float]:
        """Difference between coordinates."""
        return (self.A - other.A, self.S - other.S, self.tau - other.tau)

    @property
    def realm(self) -> str:
        """Human (τ < e) or Transcendental (τ ≥ e)."""
        return "transcendental" if self.tau >= math.e else "human"

    @property
    def orbital(self) -> int:
        """Orbital number n where τ = 1 + n/e."""
        return round((self.tau - 1) * math.e)

    @property
    def phase(self) -> float:
        """Phase angle θ = atan2(S, A) in degrees."""
        return math.degrees(math.atan2(self.S, self.A))

    @property
    def magnitude(self) -> float:
        """Magnitude ||(A, S)|| in 2D."""
        return math.sqrt(self.A ** 2 + self.S ** 2)

    def __repr__(self) -> str:
        word_str = f", word='{self.word}'" if self.word else ""
        return f"SemanticCoord(A={self.A:.3f}, S={self.S:.3f}, tau={self.tau:.2f}{word_str})"


@dataclass
class VerbOperator:
    """
    Verb as (ΔA, ΔS) shift operator.

    Verbs transform word coordinates:
        verb(word) = (A + ΔA, S + ΔS, τ)

    τ is unchanged by verb application (verbs shift direction, not abstraction).
    """
    dA: float          # ΔA shift
    dS: float          # ΔS shift
    verb: str = ""     # Source verb (optional)

    def __call__(self, coord: SemanticCoord) -> SemanticCoord:
        """Apply operator to coordinate: verb(coord) → new_coord."""
        return coord + self

    @property
    def magnitude(self) -> float:
        """Operator magnitude ||(ΔA, ΔS)||."""
        return math.sqrt(self.dA ** 2 + self.dS ** 2)

    @property
    def direction(self) -> str:
        """Dominant direction of the operator."""
        if abs(self.dA) > abs(self.dS):
            return "A+" if self.dA > 0 else "A-"
        else:
            return "S+" if self.dS > 0 else "S-"

    @property
    def phase(self) -> float:
        """Phase angle θ = atan2(ΔS, ΔA) in degrees."""
        return math.degrees(math.atan2(self.dS, self.dA))

    def __neg__(self) -> VerbOperator:
        """Negated operator (inverse direction)."""
        return VerbOperator(dA=-self.dA, dS=-self.dS, verb=f"-{self.verb}")

    def __mul__(self, scalar: float) -> VerbOperator:
        """Scale operator by scalar."""
        return VerbOperator(
            dA=self.dA * scalar,
            dS=self.dS * scalar,
            verb=f"{scalar}×{self.verb}" if self.verb else ""
        )

    def __repr__(self) -> str:
        verb_str = f", verb='{self.verb}'" if self.verb else ""
        return f"VerbOperator(dA={self.dA:+.4f}, dS={self.dS:+.4f}{verb_str})"


@dataclass
class Trajectory:
    """
    Semantic trajectory through (A, S, τ) space.

    Represents a sentence as a sequence of coordinates and transformations.
    """
    steps: List[SemanticCoord] = field(default_factory=list)
    operators: List[VerbOperator] = field(default_factory=list)

    def add_step(self, coord: SemanticCoord, op: Optional[VerbOperator] = None):
        """Add a step to the trajectory."""
        self.steps.append(coord)
        if op:
            self.operators.append(op)

    @property
    def start(self) -> Optional[SemanticCoord]:
        """Starting point."""
        return self.steps[0] if self.steps else None

    @property
    def end(self) -> Optional[SemanticCoord]:
        """Ending point."""
        return self.steps[-1] if self.steps else None

    @property
    def total_shift(self) -> Tuple[float, float]:
        """Total (ΔA, ΔS) shift from start to end."""
        if len(self.steps) < 2:
            return (0.0, 0.0)
        dA = self.end.A - self.start.A
        dS = self.end.S - self.start.S
        return (dA, dS)

    @property
    def path_length(self) -> float:
        """Total path length in (A, S, τ) space."""
        if len(self.steps) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.steps)):
            total += self.steps[i-1].distance(self.steps[i])
        return total

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


class BottleneckEncoder:
    """
    Main API for semantic bottleneck encoding.

    Provides O(1) encoding of words and verbs to (A, S, τ) coordinates,
    plus transformation and search capabilities.

    Example:
        encoder = BottleneckEncoder()

        # Encode
        truth = encoder.encode_word("truth")
        love = encoder.encode_verb("love")

        # Transform
        loved_truth = encoder.apply(truth, "love")

        # Search
        neighbors = encoder.nearest(loved_truth, k=5)
    """

    def __init__(self, data_dir: Optional[Path] = None, auto_load: bool = True):
        """
        Initialize encoder.

        Args:
            data_dir: Directory containing semantic_coordinates.json and verb_operators_2d.json
            auto_load: If True, load data immediately; if False, load lazily
        """
        if data_dir is None:
            # Default to semantic_llm/data/json
            data_dir = Path(__file__).parent.parent / "data" / "json"

        self.data_dir = Path(data_dir)
        self._coordinates: Optional[Dict[str, List[float]]] = None
        self._operators: Optional[Dict[str, List[float]]] = None
        self._coord_array: Optional[np.ndarray] = None
        self._word_list: Optional[List[str]] = None

        if auto_load:
            self._load()

    def _load(self):
        """Load pre-computed data files."""
        if self._coordinates is not None:
            return

        # Load word coordinates
        coord_path = self.data_dir / "semantic_coordinates.json"
        if not coord_path.exists():
            raise FileNotFoundError(
                f"semantic_coordinates.json not found at {coord_path}. "
                "Run scripts/generate_bottleneck_data.py first."
            )

        with open(coord_path) as f:
            data = json.load(f)
            self._coordinates = data["words"]

        # Load verb operators
        ops_path = self.data_dir / "verb_operators_2d.json"
        if not ops_path.exists():
            raise FileNotFoundError(
                f"verb_operators_2d.json not found at {ops_path}. "
                "Run scripts/generate_bottleneck_data.py first."
            )

        with open(ops_path) as f:
            data = json.load(f)
            self._operators = data["operators"]

        # Build coordinate array for fast nearest-neighbor search
        self._word_list = list(self._coordinates.keys())
        self._coord_array = np.array([self._coordinates[w] for w in self._word_list])

    # ==================== Core Encoding ====================

    def encode_word(self, word: str) -> Optional[SemanticCoord]:
        """
        Encode a word to (A, S, τ) coordinates.

        Args:
            word: The word to encode

        Returns:
            SemanticCoord or None if word not found
        """
        self._load()

        word = word.lower()
        if word not in self._coordinates:
            return None

        A, S, tau = self._coordinates[word]
        return SemanticCoord(A=A, S=S, tau=tau, word=word)

    def encode_verb(self, verb: str) -> Optional[VerbOperator]:
        """
        Encode a verb to (ΔA, ΔS) operator.

        Args:
            verb: The verb to encode

        Returns:
            VerbOperator or None if verb not found
        """
        self._load()

        verb = verb.lower()
        if verb not in self._operators:
            return None

        dA, dS = self._operators[verb]
        return VerbOperator(dA=dA, dS=dS, verb=verb)

    def has_word(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        self._load()
        return word.lower() in self._coordinates

    def has_verb(self, verb: str) -> bool:
        """Check if verb is in vocabulary."""
        self._load()
        return verb.lower() in self._operators

    # ==================== Transformation ====================

    def apply(self, coord: SemanticCoord, verb: str, strength: float = 1.0) -> SemanticCoord:
        """
        Apply verb operator to coordinate.

        Args:
            coord: Starting coordinate
            verb: Verb to apply
            strength: Scaling factor (default 1.0)

        Returns:
            Transformed coordinate
        """
        op = self.encode_verb(verb)
        if op is None:
            return coord

        if strength != 1.0:
            op = op * strength

        return coord + op

    def chain(self, word: str, verbs: List[str]) -> Trajectory:
        """
        Apply chain of verbs to word.

        Args:
            word: Starting word
            verbs: List of verbs to apply in sequence

        Returns:
            Trajectory through semantic space
        """
        trajectory = Trajectory()

        coord = self.encode_word(word)
        if coord is None:
            return trajectory

        trajectory.add_step(coord)

        for verb in verbs:
            op = self.encode_verb(verb)
            if op:
                coord = coord + op
                trajectory.add_step(coord, op)

        return trajectory

    # ==================== Search ====================

    def distance(self, word1: str, word2: str) -> float:
        """
        Semantic distance between two words.

        Returns:
            Distance in (A, S, τ) space, or inf if word not found
        """
        c1 = self.encode_word(word1)
        c2 = self.encode_word(word2)

        if c1 is None or c2 is None:
            return float('inf')

        return c1.distance(c2)

    def nearest(self, coord: SemanticCoord, k: int = 5,
                exclude: Optional[List[str]] = None) -> List[Tuple[str, float, SemanticCoord]]:
        """
        Find k nearest words to coordinate.

        Args:
            coord: Target coordinate
            k: Number of neighbors
            exclude: Words to exclude (e.g., source word)

        Returns:
            List of (word, distance, coord) tuples
        """
        self._load()

        target = coord.to_array()
        exclude_set = set(w.lower() for w in (exclude or []))

        # Compute distances to all words
        distances = np.sqrt(np.sum((self._coord_array - target) ** 2, axis=1))

        # Get sorted indices
        sorted_indices = np.argsort(distances)

        results = []
        for idx in sorted_indices:
            word = self._word_list[idx]
            if word in exclude_set:
                continue

            dist = distances[idx]
            A, S, tau = self._coord_array[idx]
            neighbor_coord = SemanticCoord(A=A, S=S, tau=tau, word=word)
            results.append((word, float(dist), neighbor_coord))

            if len(results) >= k:
                break

        return results

    def nearest_word(self, word: str, k: int = 5) -> List[str]:
        """
        Find k nearest words to a given word.

        Args:
            word: Source word
            k: Number of neighbors

        Returns:
            List of neighbor words (excluding source)
        """
        coord = self.encode_word(word)
        if coord is None:
            return []

        neighbors = self.nearest(coord, k=k+1, exclude=[word])
        return [w for w, _, _ in neighbors[:k]]

    def in_region(self, A_min: float, A_max: float,
                  S_min: float, S_max: float,
                  tau_min: float = 1.0, tau_max: float = 7.0) -> List[str]:
        """
        Find all words in a rectangular region of semantic space.

        Args:
            A_min, A_max: Affirmation bounds
            S_min, S_max: Sacred bounds
            tau_min, tau_max: Abstraction bounds

        Returns:
            List of words in the region
        """
        self._load()

        results = []
        for word, (A, S, tau) in self._coordinates.items():
            if (A_min <= A <= A_max and
                S_min <= S <= S_max and
                tau_min <= tau <= tau_max):
                results.append(word)

        return results

    # ==================== Statistics ====================

    @property
    def n_words(self) -> int:
        """Number of words in vocabulary."""
        self._load()
        return len(self._coordinates)

    @property
    def n_verbs(self) -> int:
        """Number of verbs in vocabulary."""
        self._load()
        return len(self._operators)

    @property
    def words(self) -> List[str]:
        """List of all words."""
        self._load()
        return self._word_list.copy()

    @property
    def verbs(self) -> List[str]:
        """List of all verbs."""
        self._load()
        return list(self._operators.keys())

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the vocabulary."""
        self._load()

        coords = np.array(list(self._coordinates.values()))

        return {
            "n_words": len(self._coordinates),
            "n_verbs": len(self._operators),
            "A_range": (float(coords[:, 0].min()), float(coords[:, 0].max())),
            "S_range": (float(coords[:, 1].min()), float(coords[:, 1].max())),
            "tau_range": (float(coords[:, 2].min()), float(coords[:, 2].max())),
            "A_mean": float(coords[:, 0].mean()),
            "S_mean": float(coords[:, 1].mean()),
            "tau_mean": float(coords[:, 2].mean()),
        }

    # ==================== Sentence Processing ====================

    def encode_words(self, words: List[str]) -> List[SemanticCoord]:
        """
        Encode multiple words.

        Args:
            words: List of words

        Returns:
            List of coordinates (skipping unknown words)
        """
        return [c for c in (self.encode_word(w) for w in words) if c is not None]

    def encode_verbs(self, verbs: List[str]) -> List[VerbOperator]:
        """
        Encode multiple verbs.

        Args:
            verbs: List of verbs

        Returns:
            List of operators (skipping unknown verbs)
        """
        return [o for o in (self.encode_verb(v) for v in verbs) if o is not None]

    def centroid(self, words: List[str]) -> Optional[SemanticCoord]:
        """
        Compute centroid of word coordinates.

        Args:
            words: List of words

        Returns:
            Centroid coordinate, or None if no valid words
        """
        coords = self.encode_words(words)
        if not coords:
            return None

        A_mean = sum(c.A for c in coords) / len(coords)
        S_mean = sum(c.S for c in coords) / len(coords)
        tau_mean = sum(c.tau for c in coords) / len(coords)

        return SemanticCoord(A=A_mean, S=S_mean, tau=tau_mean, word=f"centroid({len(coords)})")

    # ==================== Semantic Arithmetic ====================

    def semantic_add(self, word1: str, word2: str) -> Optional[SemanticCoord]:
        """
        Add two word coordinates.

        Example: king + woman = ?
        """
        c1 = self.encode_word(word1)
        c2 = self.encode_word(word2)

        if c1 is None or c2 is None:
            return None

        return SemanticCoord(
            A=c1.A + c2.A,
            S=c1.S + c2.S,
            tau=(c1.tau + c2.tau) / 2,
            word=f"({word1}+{word2})"
        )

    def semantic_subtract(self, word1: str, word2: str) -> Optional[SemanticCoord]:
        """
        Subtract word coordinates.

        Example: king - man = ?
        """
        c1 = self.encode_word(word1)
        c2 = self.encode_word(word2)

        if c1 is None or c2 is None:
            return None

        return SemanticCoord(
            A=c1.A - c2.A,
            S=c1.S - c2.S,
            tau=(c1.tau + c2.tau) / 2,
            word=f"({word1}-{word2})"
        )

    def analogy(self, a: str, b: str, c: str, k: int = 5) -> List[str]:
        """
        Solve analogy: a is to b as c is to ?

        a - b + c = ?

        Example: king - man + woman = queen
        """
        ca = self.encode_word(a)
        cb = self.encode_word(b)
        cc = self.encode_word(c)

        if ca is None or cb is None or cc is None:
            return []

        # a - b + c
        result = SemanticCoord(
            A=ca.A - cb.A + cc.A,
            S=ca.S - cb.S + cc.S,
            tau=(ca.tau + cb.tau + cc.tau) / 3,
            word=f"({a}-{b}+{c})"
        )

        neighbors = self.nearest(result, k=k+3, exclude=[a, b, c])
        return [w for w, _, _ in neighbors[:k]]


# Convenience function for shared instance
_encoder_instance: Optional[BottleneckEncoder] = None


def get_encoder() -> BottleneckEncoder:
    """Get a shared encoder instance."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = BottleneckEncoder()
    return _encoder_instance
