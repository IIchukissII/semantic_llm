"""
Semantic Quantum Numbers: (n, θ, r) Encoding

The minimal quantized representation of semantic meaning:
- n = orbital (abstraction level): 4 bits (0-15)
- θ = phase (direction): 4 bits (16 directions)
- r = intensity (magnitude): 4 bits (16 levels)

Total: 12 bits = 1.5 bytes per word

This encodes 22,486 words in ~34 KB.

Theory:
    Like quantum mechanics uses (n, l, m) to describe electron states,
    semantic space uses (n, θ, r) to describe word meanings:

    n = energy level (abstraction)
        n=0,1,2: abstract (love, truth, beauty)
        n=5+: concrete (chair, stone, water)

    θ = phase angle (direction in j-space)
        θ=0°: affirming
        θ=90°: sacred
        θ=180°: negating
        θ=-90°: profane

    r = magnitude (intensity of transcendental character)
        r≈0: neutral (common words)
        r>>1: strongly characterized

Usage:
    from core.semantic_quantum import QuantumEncoder

    encoder = QuantumEncoder()

    # Encode word to 12 bits
    bits = encoder.encode("truth")
    print(bits)  # QuantumWord(n=4, theta_idx=15, r_idx=3)

    # Decode back
    A, S, tau = encoder.decode(bits)


    # Apply verb transformation
    result = encoder.apply_verb("truth", "love")
    print(result)  # New quantum position after transformation

    # Chain of verbs
    trajectory = encoder.chain("truth", ["love", "seek", "find"])
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import json
import math
import numpy as np

# Constants
E = math.e
N_BITS = 4  # bits per dimension
N_LEVELS = 2 ** N_BITS  # 16 levels

# Quantization ranges (from data analysis)
N_MIN, N_MAX = 0, 15
THETA_MIN, THETA_MAX = -math.pi, math.pi
R_MIN, R_MAX = 0.0, 4.0


@dataclass
class QuantumVerb:
    """
    Verb operator in quantum semantic space.

    Verbs shift position in Cartesian (A, S) space:
        (A', S') = (A + ΔA, S + ΔS)

    Then re-quantized to (n, θ', r').
    """
    dA: float           # Affirmation shift
    dS: float           # Sacred shift
    verb: str = ""      # Source verb

    @property
    def magnitude(self) -> float:
        """Magnitude of the shift vector."""
        return math.sqrt(self.dA**2 + self.dS**2)

    @property
    def direction(self) -> float:
        """Direction of shift in radians."""
        return math.atan2(self.dS, self.dA)

    @property
    def direction_deg(self) -> float:
        """Direction of shift in degrees."""
        return math.degrees(self.direction)

    def __repr__(self) -> str:
        return f"QuantumVerb({self.verb}: ΔA={self.dA:+.3f}, ΔS={self.dS:+.3f}, |Δ|={self.magnitude:.3f})"


@dataclass
class QuantumTrajectory:
    """A path through quantum semantic space."""
    steps: List[QuantumWord] = field(default_factory=list)
    verbs: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def start(self) -> Optional[QuantumWord]:
        return self.steps[0] if self.steps else None

    @property
    def end(self) -> Optional[QuantumWord]:
        return self.steps[-1] if self.steps else None

    @property
    def total_shift(self) -> Tuple[float, float]:
        """Total (ΔA, ΔS) from start to end."""
        if len(self.steps) < 2:
            return (0.0, 0.0)
        return (self.end.A - self.start.A, self.end.S - self.start.S)

    @property
    def orbital_change(self) -> int:
        """Change in orbital level (Δn)."""
        if len(self.steps) < 2:
            return 0
        return self.end.n - self.start.n

    @property
    def phase_change(self) -> float:
        """Change in phase angle (Δθ) in degrees."""
        if len(self.steps) < 2:
            return 0.0
        return self.end.theta_deg - self.start.theta_deg


@dataclass
class QuantumWord:
    """
    Quantized semantic representation of a word.

    Total: 12 bits (n: 4, θ: 4, r: 4)
    """
    n: int           # orbital (0-15)
    theta_idx: int   # phase index (0-15 → -180° to +180°)
    r_idx: int       # intensity index (0-15 → 0.0 to 4.0)
    word: str = ""   # source word

    @property
    def theta(self) -> float:
        """Phase angle in radians."""
        return THETA_MIN + (self.theta_idx / (N_LEVELS - 1)) * (THETA_MAX - THETA_MIN)

    @property
    def theta_deg(self) -> float:
        """Phase angle in degrees."""
        return math.degrees(self.theta)

    @property
    def r(self) -> float:
        """Intensity magnitude."""
        return R_MIN + (self.r_idx / (N_LEVELS - 1)) * (R_MAX - R_MIN)

    @property
    def tau(self) -> float:
        """Abstraction level τ = 1 + n/e."""
        return 1 + self.n / E

    @property
    def A(self) -> float:
        """Affirmation (reconstructed from polar)."""
        return self.r * math.cos(self.theta)

    @property
    def S(self) -> float:
        """Sacred (reconstructed from polar)."""
        return self.r * math.sin(self.theta)

    def to_bits(self) -> int:
        """Pack into 12-bit integer."""
        return (self.n << 8) | (self.theta_idx << 4) | self.r_idx

    @classmethod
    def from_bits(cls, bits: int, word: str = "") -> QuantumWord:
        """Unpack from 12-bit integer."""
        n = (bits >> 8) & 0xF
        theta_idx = (bits >> 4) & 0xF
        r_idx = bits & 0xF
        return cls(n=n, theta_idx=theta_idx, r_idx=r_idx, word=word)

    def to_hex(self) -> str:
        """Convert to 3-character hex string."""
        return f"{self.to_bits():03X}"

    @classmethod
    def from_hex(cls, hex_str: str, word: str = "") -> QuantumWord:
        """Create from hex string."""
        return cls.from_bits(int(hex_str, 16), word)

    def __repr__(self) -> str:
        return (f"QuantumWord(n={self.n}, θ={self.theta_deg:+.0f}°, r={self.r:.2f}, "
                f"word='{self.word}')")


class QuantumEncoder:
    """
    Encoder for 12-bit semantic quantum numbers.

    Converts between:
    - Cartesian: (A, S, τ) - 3 floats
    - Quantum: (n, θ, r) - 3 floats
    - Quantized: 12 bits packed
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "json"
        self.data_dir = Path(data_dir)

        self._quantized: Optional[Dict[str, int]] = None
        self._word_list: Optional[List[str]] = None
        self._verbs: Optional[Dict[str, Tuple[float, float]]] = None

    def _load(self):
        """Load or generate quantized data."""
        if self._quantized is not None:
            return

        quant_path = self.data_dir / "semantic_quantum.json"

        if quant_path.exists():
            with open(quant_path) as f:
                data = json.load(f)
                self._quantized = {w: int(h, 16) for w, h in data["words"].items()}
                self._word_list = list(self._quantized.keys())
        else:
            # Generate from coordinates
            self._generate_quantized()

    def _generate_quantized(self):
        """Generate quantized encoding from semantic coordinates."""
        coord_path = self.data_dir / "semantic_coordinates.json"

        if not coord_path.exists():
            raise FileNotFoundError(
                "semantic_coordinates.json not found. "
                "Run scripts/generate_bottleneck_data.py first."
            )

        with open(coord_path) as f:
            data = json.load(f)
            coords = data["words"]

        self._quantized = {}

        for word, (A, S, tau) in coords.items():
            # Convert to polar quantum numbers
            r = math.sqrt(A**2 + S**2)
            theta = math.atan2(S, A)
            n = round((tau - 1) * E)

            # Clamp to valid ranges
            n = max(N_MIN, min(N_MAX, n))
            r = max(R_MIN, min(R_MAX, r))

            # Quantize
            theta_idx = round((theta - THETA_MIN) / (THETA_MAX - THETA_MIN) * (N_LEVELS - 1))
            theta_idx = max(0, min(N_LEVELS - 1, theta_idx))

            r_idx = round((r - R_MIN) / (R_MAX - R_MIN) * (N_LEVELS - 1))
            r_idx = max(0, min(N_LEVELS - 1, r_idx))

            # Pack into 12 bits
            bits = (n << 8) | (theta_idx << 4) | r_idx
            self._quantized[word] = bits

        self._word_list = list(self._quantized.keys())

        # Save for future use
        self._save_quantized()

    def _save_quantized(self):
        """Save quantized data to JSON."""
        data = {
            "version": "1.0",
            "encoding": "12-bit (n:4, theta:4, r:4)",
            "n_words": len(self._quantized),
            "size_bytes": len(self._quantized) * 2,  # 12 bits ≈ 2 bytes with padding
            "words": {w: f"{b:03X}" for w, b in self._quantized.items()}
        }

        quant_path = self.data_dir / "semantic_quantum.json"
        with open(quant_path, "w") as f:
            json.dump(data, f)

        print(f"[QuantumEncoder] Saved {len(self._quantized)} words to {quant_path.name}")

    # ==================== Encoding ====================

    def encode(self, word: str) -> Optional[QuantumWord]:
        """Encode word to quantized representation."""
        self._load()

        word = word.lower()
        if word not in self._quantized:
            return None

        bits = self._quantized[word]
        return QuantumWord.from_bits(bits, word)

    def encode_to_bits(self, word: str) -> Optional[int]:
        """Encode word to 12-bit integer."""
        self._load()
        return self._quantized.get(word.lower())

    def encode_to_hex(self, word: str) -> Optional[str]:
        """Encode word to 3-character hex string."""
        bits = self.encode_to_bits(word)
        if bits is None:
            return None
        return f"{bits:03X}"

    # ==================== Decoding ====================

    def decode(self, qword: QuantumWord) -> Tuple[float, float, float]:
        """Decode quantized word to (A, S, τ)."""
        return qword.A, qword.S, qword.tau

    def decode_bits(self, bits: int) -> Tuple[float, float, float]:
        """Decode 12-bit integer to (A, S, τ)."""
        qword = QuantumWord.from_bits(bits)
        return self.decode(qword)

    def decode_hex(self, hex_str: str) -> Tuple[float, float, float]:
        """Decode hex string to (A, S, τ)."""
        return self.decode_bits(int(hex_str, 16))

    # ==================== Search ====================

    def nearest(self, qword: QuantumWord, k: int = 5) -> List[Tuple[str, float, QuantumWord]]:
        """Find k nearest words in quantized space."""
        self._load()

        distances = []
        for word, bits in self._quantized.items():
            other = QuantumWord.from_bits(bits, word)

            # Distance in (n, θ, r) space
            dn = (qword.n - other.n) ** 2

            # Angular distance (handle wrap-around)
            dtheta = abs(qword.theta - other.theta)
            if dtheta > math.pi:
                dtheta = 2 * math.pi - dtheta
            dtheta = (dtheta / math.pi) ** 2  # normalize

            dr = ((qword.r - other.r) / R_MAX) ** 2

            dist = math.sqrt(dn + dtheta + dr)
            distances.append((word, dist, other))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    # ==================== Verb Operators ====================

    def _load_verbs(self):
        """Load verb operators from JSON."""
        if self._verbs is not None:
            return

        verb_path = self.data_dir / "verb_operators_2d.json"

        if not verb_path.exists():
            raise FileNotFoundError(
                "verb_operators_2d.json not found. "
                "Run scripts/generate_bottleneck_data.py first."
            )

        with open(verb_path) as f:
            data = json.load(f)
            self._verbs = {v: tuple(op) for v, op in data["operators"].items()}

    def encode_verb(self, verb: str) -> Optional[QuantumVerb]:
        """Encode verb to operator."""
        self._load_verbs()

        verb = verb.lower()
        if verb not in self._verbs:
            return None

        dA, dS = self._verbs[verb]
        return QuantumVerb(dA=dA, dS=dS, verb=verb)

    @property
    def n_verbs(self) -> int:
        """Number of verb operators."""
        self._load_verbs()
        return len(self._verbs)

    @property
    def verbs(self) -> List[str]:
        """List of available verbs."""
        self._load_verbs()
        return list(self._verbs.keys())

    def _quantize_coords(self, A: float, S: float, tau: float) -> QuantumWord:
        """Convert (A, S, τ) to quantized QuantumWord."""
        # Convert to polar
        r = math.sqrt(A**2 + S**2)
        theta = math.atan2(S, A)
        n = round((tau - 1) * E)

        # Clamp to valid ranges
        n = max(N_MIN, min(N_MAX, n))
        r = max(R_MIN, min(R_MAX, r))

        # Quantize
        theta_idx = round((theta - THETA_MIN) / (THETA_MAX - THETA_MIN) * (N_LEVELS - 1))
        theta_idx = max(0, min(N_LEVELS - 1, theta_idx))

        r_idx = round((r - R_MIN) / (R_MAX - R_MIN) * (N_LEVELS - 1))
        r_idx = max(0, min(N_LEVELS - 1, r_idx))

        return QuantumWord(n=n, theta_idx=theta_idx, r_idx=r_idx)

    def apply_verb(self, word: str, verb: str, strength: float = 1.0) -> Optional[QuantumWord]:
        """
        Apply verb operator to word.

        word(verb) → new quantum position

        Args:
            word: Starting word
            verb: Verb operator to apply
            strength: Scaling factor for verb effect (default 1.0)

        Returns:
            New QuantumWord at transformed position
        """
        qw = self.encode(word)
        vb = self.encode_verb(verb)

        if qw is None or vb is None:
            return None

        # Apply in Cartesian space
        new_A = qw.A + vb.dA * strength
        new_S = qw.S + vb.dS * strength
        tau = qw.tau  # τ preserved

        # Re-quantize
        result = self._quantize_coords(new_A, new_S, tau)
        result.word = f"{word}+{verb}"

        return result

    def apply_verb_to_quantum(self, qw: QuantumWord, verb: str, strength: float = 1.0) -> Optional[QuantumWord]:
        """Apply verb to an existing quantum position."""
        vb = self.encode_verb(verb)
        if vb is None:
            return None

        new_A = qw.A + vb.dA * strength
        new_S = qw.S + vb.dS * strength

        result = self._quantize_coords(new_A, new_S, qw.tau)
        result.word = f"{qw.word}+{verb}"

        return result

    def chain(self, start_word: str, verbs: List[str], strength: float = 1.0) -> QuantumTrajectory:
        """
        Apply chain of verbs to navigate through semantic space.

        Args:
            start_word: Starting word
            verbs: List of verbs to apply in sequence
            strength: Scaling factor for verb effects

        Returns:
            QuantumTrajectory with all steps
        """
        trajectory = QuantumTrajectory(verbs=verbs)

        qw = self.encode(start_word)
        if qw is None:
            return trajectory

        trajectory.steps.append(qw)

        current = qw
        for verb in verbs:
            next_qw = self.apply_verb_to_quantum(current, verb, strength)
            if next_qw is None:
                break
            trajectory.steps.append(next_qw)
            current = next_qw

        return trajectory

    def navigate(self, start_word: str, target_theta: float = None,
                 target_n: int = None, target_r: float = None,
                 max_steps: int = 5) -> List[Tuple[str, QuantumWord]]:
        """
        Find verbs that navigate toward target coordinates.

        Args:
            start_word: Starting word
            target_theta: Target phase angle in degrees (optional)
            target_n: Target orbital level (optional)
            target_r: Target intensity (optional)
            max_steps: Maximum verbs to try

        Returns:
            List of (verb, resulting_position) sorted by distance to target
        """
        self._load_verbs()

        qw = self.encode(start_word)
        if qw is None:
            return []

        results = []

        for verb in self._verbs.keys():
            next_qw = self.apply_verb_to_quantum(qw, verb)
            if next_qw is None:
                continue

            # Compute distance to target
            dist = 0.0
            if target_theta is not None:
                dtheta = abs(next_qw.theta_deg - target_theta)
                if dtheta > 180:
                    dtheta = 360 - dtheta
                dist += (dtheta / 180) ** 2
            if target_n is not None:
                dist += (next_qw.n - target_n) ** 2
            if target_r is not None:
                dist += ((next_qw.r - target_r) / R_MAX) ** 2

            results.append((verb, next_qw, math.sqrt(dist)))

        results.sort(key=lambda x: x[2])
        return [(v, qw) for v, qw, _ in results[:max_steps]]

    # ==================== Statistics ====================

    @property
    def n_words(self) -> int:
        self._load()
        return len(self._quantized)

    def stats(self) -> Dict:
        """Get encoding statistics."""
        self._load()

        ns = []
        thetas = []
        rs = []

        for bits in self._quantized.values():
            qw = QuantumWord.from_bits(bits)
            ns.append(qw.n)
            thetas.append(qw.theta_idx)
            rs.append(qw.r_idx)

        return {
            "n_words": len(self._quantized),
            "bits_per_word": 12,
            "bytes_total": len(self._quantized) * 2,
            "n_distribution": {i: ns.count(i) for i in range(N_LEVELS) if ns.count(i) > 0},
            "mean_n": np.mean(ns),
            "mean_theta_idx": np.mean(thetas),
            "mean_r_idx": np.mean(rs),
        }


def generate_quantum_encoding():
    """Generate the quantized encoding file."""
    print("=" * 60)
    print("GENERATING 12-BIT SEMANTIC QUANTUM ENCODING")
    print("=" * 60)
    print()

    encoder = QuantumEncoder()
    encoder._load()

    print(f"Encoded {encoder.n_words} words")
    print()

    # Test some words
    test_words = ["truth", "love", "god", "death", "wisdom", "life", "peace", "war"]

    print("Sample encodings:")
    print(f"{'Word':<10} {'Hex':>5} {'n':>3} {'θ°':>6} {'r':>5} | {'A':>6} {'S':>6} {'τ':>5}")
    print("-" * 60)

    for word in test_words:
        qw = encoder.encode(word)
        if qw:
            print(f"{word:<10} {qw.to_hex():>5} {qw.n:>3} {qw.theta_deg:>+6.0f} {qw.r:>5.2f} | "
                  f"{qw.A:>+6.2f} {qw.S:>+6.2f} {qw.tau:>5.2f}")

    print()

    # Size comparison
    stats = encoder.stats()
    print("Size comparison:")
    print(f"  12-bit quantized:     {stats['bytes_total']:,} bytes ({stats['bytes_total']/1024:.1f} KB)")
    print(f"  3×float32 (original): {encoder.n_words * 12:,} bytes ({encoder.n_words * 12/1024:.1f} KB)")
    print(f"  Compression ratio:    {encoder.n_words * 12 / stats['bytes_total']:.1f}x")
    print()


if __name__ == "__main__":
    generate_quantum_encoding()
