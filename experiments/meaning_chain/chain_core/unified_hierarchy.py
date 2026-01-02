#!/usr/bin/env python3
"""
Unified Semantic Hierarchy
==========================

5-LEVEL STRUCTURE (like physics: quarks → hadrons → nuclei → atoms → molecules)

LEVEL 1: TRANSCENDENTALS (Source)
─────────────────────────────────
  Space:     (A, S) = 2D
  Units:     Pure qualities
  Objects:   beauty, life, sacred, good, love
  Law:       Source, no dynamics

LEVEL 2: WORDS
──────────────
  Space:     (n, θ, r) = 3 quantum numbers
  Units:     nouns, adjectives, verbs
  Laws:      - Boltzmann: P ∝ exp(-Δn/kT)
             - Gravity: φ = λn - μA
  Derivation:
    - Adjectives: direct projection from Level 1
    - Nouns: weighted centroid of adjectives
    - Verbs: phase shift operators (Δθ, Δr)

LEVEL 3: BONDS
──────────────
  Space:     Bipartite graph (adj ↔ noun)
  Units:     adj-noun pairs
  Laws:      - PT1 saturation: b/ν = (b/ν)_max × (1 - e^(-ν/τ_ν))
             - Zipf, Heaps
  Derivation:
    - n = f(variety) via entropy
    - θ, r = weighted average of adj (θ, r)

LEVEL 4: SENTENCES
──────────────────
  Space:     Trajectories in (n, θ, r)
  Units:     SVO sequences
  Laws:      - Intent collapse
             - Coherence = cos(Δθ)
  Derivation:
    - sentence = integration over words

LEVEL 5: DIALOGUE
─────────────────
  Space:     Navigation
  Units:     Exchanges
  Laws:      - Storm-Logos
             - Paradox chain reaction
             - Σ = C + 0.1P
  Derivation:
    - dialogue = sequence of sentences

KEY INSIGHT: Each level has emergence, not full reduction.
"""

import sys
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict

# Constants
E = math.e  # Euler's constant
KT = E ** (-1/5)  # Semantic temperature ≈ 0.819

# PC vectors for (A, S) projection - from PCA on j-vectors
# A = "Affirmation" axis: PC1 from embedding space
# S = "Sacred" axis: PC2 from embedding space
#
# These axes capture STATISTICAL variance in semantic space.
# Labels are approximate - geometry (angles, distances) is what matters.
#
# Verbs are OPERATORS on nouns - both must use the SAME coordinate system.
PC1_AFFIRMATION = np.array([-0.448, -0.519, -0.118, -0.480, -0.534])
PC2_SACRED = np.array([-0.513, +0.128, -0.732, +0.420, +0.090])

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']

# Global mean of verb j-vectors (The Pirate Insight)
# Raw j-vectors are biased toward this mean, making opposites look similar (99% cosine).
# Centering by subtracting this mean reveals true opposition (love/hate → -0.22).
J_GLOBAL_MEAN = np.array([-0.82, -0.97, -0.92, -0.80, -0.95])


# =============================================================================
# LEVEL 1: TRANSCENDENTALS
# =============================================================================

@dataclass
class Transcendental:
    """
    Level 1: Pure quality in 2D space.

    These are the 5 source dimensions that project down to everything else.
    They exist in a 2D reduced space: (A, S).

    A = Affirmation axis (PC1 from embedding space)
    S = Sacred axis (PC2 from embedding space)
    """
    name: str
    A: float  # Affirmation component
    S: float  # Sacred component

    @property
    def theta(self) -> float:
        """Phase angle in radians."""
        return math.atan2(self.S, self.A)

    @property
    def r(self) -> float:
        """Magnitude."""
        return math.sqrt(self.A**2 + self.S**2)


# The 5 transcendentals as unit vectors in their respective dimensions
# Projected onto (A, S) axes from PCA
TRANSCENDENTALS = {
    'beauty': Transcendental('beauty', A=-0.448, S=-0.513),
    'life':   Transcendental('life',   A=-0.519, S=+0.128),
    'sacred': Transcendental('sacred', A=-0.118, S=-0.732),
    'good':   Transcendental('good',   A=-0.480, S=+0.420),
    'love':   Transcendental('love',   A=-0.534, S=+0.090),
}


# =============================================================================
# LEVEL 2: WORDS (Quantum Numbers)
# =============================================================================

@dataclass
class QuantumWord:
    """
    Level 2: Word with quantum numbers (n, θ, r).

    TWO COORDINATE SYSTEMS:
    ───────────────────────
    1. DERIVED (usage patterns): from adjective centroids (bond space)
       - θ_derived, r_derived: weighted average of adjective coordinates
       - Captures HOW words are USED

    2. RAW (semantic meaning): from neural embeddings
       - θ_raw, r_raw: direct projection from j-vector
       - Captures WHAT words MEAN

    TYPES:
    - Adjective: direct projection from transcendentals
    - Noun: has both derived and raw coordinates
    - Verb: phase shift operator (Δθ, Δr)
    """
    word: str
    word_type: str  # 'noun', 'adj', 'verb'

    # Quantum numbers (DERIVED from bonds - usage patterns)
    n: float        # Orbital level [0-15], derived from entropy
    theta: float    # Phase angle in radians (derived)
    r: float        # Magnitude/intensity (derived)

    # RAW coordinates (semantic meaning from neural embeddings)
    theta_raw: float = 0.0    # Phase angle from original j-vector
    r_raw: float = 0.0        # Magnitude from original j-vector

    # Source data (for nouns)
    variety: int = 0          # Number of unique adjectives
    h_norm: float = 0.0       # Normalized entropy
    coverage: float = 1.0     # Fraction of adjectives with known vectors

    # For verbs: the shift operator
    delta_theta: float = 0.0  # Phase shift applied to objects
    delta_r: float = 0.0      # Magnitude shift

    @property
    def A(self) -> float:
        """Affirmation (Cartesian from polar)."""
        return self.r * math.cos(self.theta)

    @property
    def S(self) -> float:
        """Sacred (Cartesian from polar)."""
        return self.r * math.sin(self.theta)

    @property
    def tau(self) -> float:
        """
        Abstraction level τ ∈ [1, 6].
        Derived from orbital: τ = 1 + n/e
        """
        return 1.0 + self.n / E

    @classmethod
    def from_cartesian(cls, word: str, word_type: str, A: float, S: float, n: float,
                       A_raw: float = None, S_raw: float = None, **kwargs):
        """
        Create from Cartesian coordinates.

        Args:
            A, S: Derived coordinates (from centroids)
            A_raw, S_raw: Raw coordinates (from neural embeddings)
        """
        theta = math.atan2(S, A)
        r = math.sqrt(A**2 + S**2)

        # Raw coordinates (default to derived if not provided)
        if A_raw is not None and S_raw is not None:
            theta_raw = math.atan2(S_raw, A_raw)
            r_raw = math.sqrt(A_raw**2 + S_raw**2)
        else:
            theta_raw = theta
            r_raw = r

        return cls(word=word, word_type=word_type, n=n, theta=theta, r=r,
                   theta_raw=theta_raw, r_raw=r_raw, **kwargs)

    def boltzmann_weight(self, target_n: float) -> float:
        """
        Boltzmann weight for transition to target orbital.
        P ∝ exp(-Δn/kT)
        """
        delta_n = abs(target_n - self.n)
        return math.exp(-delta_n / KT)

    def gravity_potential(self, lambda_: float = 1.0, mu: float = 0.5) -> float:
        """
        Gravity potential: φ = λn - μA
        Words "fall" toward low n (concrete) and high A (affirming).
        """
        return lambda_ * self.n - mu * self.A


@dataclass
class VerbOperator:
    """
    Verb as MOMENTUM operator - direction of transformation.

    Verbs don't occupy positions - they are DIRECTIONS that transform positions.
    Like momentum in physics: p = m × v (direction of motion)

    verb(noun) → noun' where:
        n' = n + Δn      (orbital shift: abstraction level)
        θ' = θ + Δθ      (phase rotation: semantic direction)
        r' = r + Δr      (magnitude change: intensity)

    The verb's centered j-vector (i = j - j_mean) gives the DIRECTION.
    """
    verb: str
    delta_theta: float  # Phase rotation (radians) - direction in (A,S) space
    delta_r: float      # Magnitude shift
    delta_n: float = 0.0  # Orbital shift (abstraction: + = abstract, - = concrete)

    # The verb's direction vector (from centered j)
    theta: float = 0.0  # Direction angle
    r: float = 0.0      # Direction magnitude

    def apply(self, noun: QuantumWord) -> Tuple[float, float, float]:
        """
        Apply verb operator to noun, return new (n, θ, r).
        """
        new_n = noun.n + self.delta_n

        new_theta = noun.theta + self.delta_theta
        # Normalize to [-π, π]
        while new_theta > math.pi:
            new_theta -= 2 * math.pi
        while new_theta < -math.pi:
            new_theta += 2 * math.pi

        new_r = noun.r + self.delta_r
        return new_n, new_theta, max(0, new_r)

    def apply_theta_r(self, noun: QuantumWord) -> Tuple[float, float]:
        """Legacy: return just (θ, r) for compatibility."""
        _, theta, r = self.apply(noun)
        return theta, r


# =============================================================================
# LEVEL 3: BONDS (Graph Structure)
# =============================================================================

@dataclass
class BondSpace:
    """
    Level 3: Bipartite graph of adj-noun bonds.

    Laws:
    - PT1 saturation: b/ν = (b/ν)_max × (1 - e^(-ν/τ_ν))
    - Entropy → orbital: n = 5 × (1 - H_norm)

    This level derives n for nouns from bond statistics.
    """
    # adj -> noun -> count
    bonds: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    # noun -> {adj: count}
    noun_profiles: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_bond(self, adj: str, noun: str, count: int = 1):
        """Add observation of adj-noun bond."""
        self.bonds[adj][noun] += count
        if noun not in self.noun_profiles:
            self.noun_profiles[noun] = defaultdict(int)
        self.noun_profiles[noun][adj] += count

    def get_noun_profile(self, noun: str) -> Dict[str, int]:
        """Get adjective profile for noun."""
        return dict(self.noun_profiles.get(noun, {}))

    def compute_entropy(self, profile: Dict[str, int]) -> Tuple[float, float, int]:
        """
        Compute Shannon entropy of adjective distribution.

        Returns:
            (H, H_norm, variety)
        """
        if not profile:
            return 0.0, 0.0, 0

        total = sum(profile.values())
        if total == 0:
            return 0.0, 0.0, 0

        variety = len(profile)
        if variety <= 1:
            return 0.0, 0.0, variety

        H = 0.0
        for count in profile.values():
            if count > 0:
                p = count / total
                H -= p * math.log2(p)

        H_max = math.log2(variety)
        H_norm = H / H_max

        return H, H_norm, variety

    def entropy_to_orbital(self, H_norm: float) -> float:
        """
        Derive orbital n from normalized entropy.

        n = 5 × (1 - H_norm)

        High entropy → low n → abstract
        Low entropy → high n → concrete
        """
        return 5.0 * (1.0 - H_norm)


# =============================================================================
# UNIFIED BUILDER: Derive all levels from bonds
# =============================================================================

class UnifiedHierarchy:
    """
    Build the complete hierarchy from bond space.

    Flow:
    1. Load adjective j-vectors (Level 1 projections)
    2. Load adj-noun bonds (Level 3)
    3. Derive noun (n, θ, r) from bonds (Level 2)
    4. Derive verb operators from verb j-vectors
    """

    def __init__(self):
        self.adjectives: Dict[str, QuantumWord] = {}
        self.nouns: Dict[str, QuantumWord] = {}
        self.verbs: Dict[str, VerbOperator] = {}
        self.bond_space = BondSpace()

    def load_adjective_vectors(self, word_vectors: Dict):
        """
        Load adjectives from word vectors (Level 1 → Level 2 projection).

        Adjectives are DIRECT projections from transcendentals.
        """
        for word, data in word_vectors.items():
            if not data.get('j'):
                continue

            # Only process adjectives
            wtype = data.get('word_type')
            if wtype not in ('adjective', 2):
                continue

            j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])
            A = float(np.dot(j_5d, PC1_AFFIRMATION))
            S = float(np.dot(j_5d, PC2_SACRED))

            # Adjectives have no orbital from bonds (direct projection)
            # Use tau from data if available, else default to 3
            tau = data.get('tau', 3.0)
            n = (tau - 1) * E  # Inverse of τ = 1 + n/e

            self.adjectives[word] = QuantumWord.from_cartesian(
                word=word,
                word_type='adj',
                A=A, S=S, n=n
            )

        print(f"[Hierarchy] Loaded {len(self.adjectives)} adjectives")

    def load_all_word_vectors_as_adjectives(self, word_vectors: Dict):
        """
        Treat ALL words with j-vectors as potential adjectives for centroid computation.

        This expands coverage since many adj-noun bonds use words not tagged as adjectives.
        """
        for word, data in word_vectors.items():
            if not data.get('j'):
                continue
            if word in self.adjectives:
                continue

            j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])
            A = float(np.dot(j_5d, PC1_AFFIRMATION))
            S = float(np.dot(j_5d, PC2_SACRED))

            tau = data.get('tau', 3.0)
            n = (tau - 1) * E

            wtype = data.get('word_type', 'unknown')
            if wtype in (0, 'noun'):
                wtype_str = 'noun'
            elif wtype in (1, 'verb'):
                wtype_str = 'verb'
            elif wtype in (2, 'adjective'):
                wtype_str = 'adj'
            else:
                wtype_str = 'unknown'

            self.adjectives[word] = QuantumWord.from_cartesian(
                word=word,
                word_type=wtype_str,
                A=A, S=S, n=n
            )

        print(f"[Hierarchy] Expanded to {len(self.adjectives)} words with j-vectors")

    def load_bonds(self, adj_profiles: Dict[str, Dict[str, int]]):
        """
        Load adj-noun bonds (Level 3).
        """
        for noun, profile in adj_profiles.items():
            self.bond_space.noun_profiles[noun] = dict(profile)

        print(f"[Hierarchy] Loaded bonds for {len(adj_profiles)} nouns")

    def derive_noun_coordinates(self, word_vectors: Dict = None,
                                 min_adjectives: int = 5, min_coverage: float = 0.2):
        """
        Derive noun (n, θ, r) from adjective centroids (Level 3 → Level 2).

        This computes TWO coordinate systems:
            DERIVED: from adjective centroids (usage patterns)
            RAW: from original j-vectors (semantic meaning)

        Args:
            word_vectors: Original word vectors for RAW coordinates
            min_adjectives: Minimum adjectives per noun
            min_coverage: Minimum coverage threshold
        """
        derived_count = 0
        skipped_few_adj = 0
        skipped_low_coverage = 0

        for noun, profile in self.bond_space.noun_profiles.items():
            # Check minimum adjectives
            if len(profile) < min_adjectives:
                skipped_few_adj += 1
                continue

            # Compute entropy → orbital n
            H, H_norm, variety = self.bond_space.compute_entropy(profile)
            n = self.bond_space.entropy_to_orbital(H_norm)

            # Compute weighted centroid of adjective (A, S) - DERIVED
            total = sum(profile.values())
            if total == 0:
                continue

            A_sum = 0.0
            S_sum = 0.0
            weight_sum = 0.0
            found = 0

            for adj, count in profile.items():
                if adj in self.adjectives:
                    weight = count / total
                    adj_word = self.adjectives[adj]
                    A_sum += weight * adj_word.A
                    S_sum += weight * adj_word.S
                    weight_sum += weight
                    found += 1

            # Check coverage
            coverage = found / len(profile) if len(profile) > 0 else 0
            if coverage < min_coverage:
                skipped_low_coverage += 1
                continue

            if weight_sum < 0.01:
                continue

            # Normalize - DERIVED coordinates
            A_derived = A_sum / weight_sum
            S_derived = S_sum / weight_sum

            # Get RAW coordinates from original word vectors
            A_raw = None
            S_raw = None
            if word_vectors and noun in word_vectors and word_vectors[noun].get('j'):
                data = word_vectors[noun]
                j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])
                A_raw = float(np.dot(j_5d, PC1_AFFIRMATION))
                S_raw = float(np.dot(j_5d, PC2_SACRED))

            # Create noun with BOTH coordinate systems
            self.nouns[noun] = QuantumWord.from_cartesian(
                word=noun,
                word_type='noun',
                A=A_derived, S=S_derived, n=n,
                A_raw=A_raw, S_raw=S_raw,
                variety=variety,
                h_norm=H_norm,
                coverage=coverage
            )
            derived_count += 1

        print(f"[Hierarchy] Derived coordinates for {derived_count} nouns")
        print(f"           Skipped {skipped_few_adj} (< {min_adjectives} adj)")
        print(f"           Skipped {skipped_low_coverage} (coverage < {min_coverage})")

    def derive_verb_operators(self, word_vectors: Dict, verb_objects: Dict[str, List[str]]):
        """
        Derive verb operators (Δn, Δθ, Δr) from PHASE-SHIFTED j-vectors.

        VERBS AS MOMENTUM:
        ──────────────────
        Verbs are DIRECTIONS that transform noun positions.
        Like momentum p = m×v in physics.

        THE PIRATE INSIGHT:
        Raw j-vectors are biased → opposites look similar (99%).
        CENTERING reveals true direction of push.

        OPERATOR COMPONENTS:
          Δn: orbital shift (abstraction level)
              - Δn > 0: abstracts (toward ideas)
              - Δn < 0: grounds (toward concrete)
              - Derived from: sacred vs life components

          Δθ: phase rotation (semantic direction)
              - Direction of push in (A, S) space

          Δr: magnitude change (intensity)
        """
        for word, data in word_vectors.items():
            if not data.get('j'):
                continue

            wtype = data.get('word_type')
            if wtype not in ('verb', 1):
                continue

            # Raw j-vector [beauty, life, sacred, good, love]
            j_raw = np.array([data['j'].get(d, 0) for d in J_DIMS])

            # PHASE SHIFT: center by subtracting global mean
            j_centered = j_raw - J_GLOBAL_MEAN

            # Project centered j onto (A, S) axes
            A = float(np.dot(j_centered, PC1_AFFIRMATION))
            S = float(np.dot(j_centered, PC2_SACRED))

            theta = math.atan2(S, A)
            r = math.sqrt(A**2 + S**2)

            # ORBITAL SHIFT (Δn): sacred abstracts, life grounds
            # j_centered[2] = sacred component, j_centered[1] = life component
            delta_n = (j_centered[2] - j_centered[1]) * 0.1  # Scale factor

            # Scaling for phase/magnitude shifts
            scale = 0.1

            self.verbs[word] = VerbOperator(
                verb=word,
                delta_theta=theta * scale,
                delta_r=r * scale,
                delta_n=delta_n,
                theta=theta,
                r=r
            )

        print(f"[Hierarchy] Derived operators for {len(self.verbs)} verbs")

    def get_word(self, word: str) -> Optional[QuantumWord]:
        """Get quantum word (noun, adj, or verb as QuantumWord)."""
        if word in self.nouns:
            return self.nouns[word]
        if word in self.adjectives:
            return self.adjectives[word]
        return None

    def get_verb(self, verb: str) -> Optional[VerbOperator]:
        """Get verb operator."""
        return self.verbs.get(verb)

    def export_coordinates(self) -> Dict[str, Dict]:
        """
        Export all coordinates (both DERIVED and RAW).

        Format: {word: {
            'n': float,
            'theta': float,      # DERIVED (usage patterns)
            'r': float,          # DERIVED
            'theta_raw': float,  # RAW (semantic meaning)
            'r_raw': float,      # RAW
            ...
        }}
        """
        result = {}

        for word, qw in self.nouns.items():
            result[word] = {
                'word_type': 'noun',
                'n': qw.n,
                # DERIVED coordinates (usage patterns)
                'theta': qw.theta,
                'theta_deg': math.degrees(qw.theta),
                'r': qw.r,
                'A': qw.A,
                'S': qw.S,
                # RAW coordinates (semantic meaning)
                'theta_raw': qw.theta_raw,
                'theta_raw_deg': math.degrees(qw.theta_raw),
                'r_raw': qw.r_raw,
                'A_raw': qw.r_raw * math.cos(qw.theta_raw),
                'S_raw': qw.r_raw * math.sin(qw.theta_raw),
                # Derived from n
                'tau': qw.tau,
                # Metadata
                'variety': qw.variety,
                'h_norm': qw.h_norm,
                'coverage': qw.coverage
            }

        for word, qw in self.adjectives.items():
            if word not in result:  # Don't overwrite nouns
                result[word] = {
                    'word_type': qw.word_type,
                    'n': qw.n,
                    'theta': qw.theta,
                    'theta_deg': math.degrees(qw.theta),
                    'r': qw.r,
                    'A': qw.A,
                    'S': qw.S,
                    'tau': qw.tau
                }

        return result

    def get_statistics(self) -> Dict:
        """Get statistics about the hierarchy."""
        noun_n = [n.n for n in self.nouns.values()]
        noun_r = [n.r for n in self.nouns.values()]
        noun_coverage = [n.coverage for n in self.nouns.values()]

        return {
            'n_adjectives': len(self.adjectives),
            'n_nouns': len(self.nouns),
            'n_verbs': len(self.verbs),
            'noun_n_mean': np.mean(noun_n) if noun_n else 0,
            'noun_n_std': np.std(noun_n) if noun_n else 0,
            'noun_r_mean': np.mean(noun_r) if noun_r else 0,
            'noun_r_std': np.std(noun_r) if noun_r else 0,
            'noun_coverage_mean': np.mean(noun_coverage) if noun_coverage else 0,
        }


# =============================================================================
# LEVEL LAWS SUMMARY
# =============================================================================

LEVEL_LAWS = """
LEVEL 1: TRANSCENDENTALS
  - No dynamics (source)
  - (A, S) are the 2D basis

LEVEL 2: WORDS
  - Boltzmann: P(transition) ∝ exp(-Δn/kT), kT = e^(-1/5) ≈ 0.819
  - Gravity: φ = λn - μA (words fall toward concrete + affirming)
  - Orbital quantization: n = 0, 1, 2, ... (discrete abstraction)

LEVEL 3: BONDS
  - PT1 saturation: b/ν = (b/ν)_max × (1 - e^(-ν/τ_ν)), R² = 0.9919
  - Zipf's law: frequency ∝ 1/rank
  - Heaps' law: vocabulary ∝ tokens^β
  - Entropy → orbital: n = 5 × (1 - H_norm)

LEVEL 4: SENTENCES
  - Intent collapse: verbs collapse meaning exploration
  - Coherence: C = cos(Δθ) between adjacent words
  - SVO as trajectory: subject → verb(operator) → object

LEVEL 5: DIALOGUE
  - Storm-Logos: chaotic exploration → meaning lens
  - Paradox chain reaction: λ ≈ 3-7, supercritical amplification
  - Energy conservation: Σ = C + 0.1P = e^(1/5) ≈ 1.22
  - kT × Σ = 1 (reciprocal relationship)
"""


# =============================================================================
# SEMANTIC NAVIGATION (Level 4-5)
# =============================================================================

class SemanticNavigator:
    """
    Navigate semantic space using the unified hierarchy.

    Provides:
    - Similarity computation (combining orbital + phase + magnitude)
    - Word neighborhood finding
    - Path coherence computation
    - Query decomposition
    """

    def __init__(self, hierarchy: UnifiedHierarchy, stopwords_path: Optional[Path] = None):
        self.hierarchy = hierarchy
        self._stopwords = self._load_stopwords(stopwords_path)

    def _load_stopwords(self, path: Optional[Path] = None) -> set:
        """Load stopwords from file or use minimal default."""
        if path and path.exists():
            with open(path) as f:
                return {line.strip().lower() for line in f if line.strip()}

        # Minimal functional word set (articles, prepositions, pronouns)
        return {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'and', 'or',
            'but', 'if', 'then', 'so', 'no', 'not', 'just', 'very', 'how', 'why'
        }

    def similarity(self, w1: QuantumWord, w2: QuantumWord,
                   use_raw: bool = False, include_orbital: bool = True) -> float:
        """
        Compute semantic similarity combining:
        - Boltzmann weight for orbital transition: exp(-Δn/kT)
        - Phase coherence: cos(Δθ)
        - Magnitude similarity: min(r)/max(r)

        Args:
            w1, w2: Words to compare
            use_raw: Use RAW (semantic) vs DERIVED (usage) coordinates
            include_orbital: Whether to weight by orbital proximity

        Returns:
            Similarity in range [-1, 1] (or [0, 1] with orbital weighting)
        """
        if use_raw:
            theta1, r1 = w1.theta_raw, w1.r_raw
            theta2, r2 = w2.theta_raw, w2.r_raw
        else:
            theta1, r1 = w1.theta, w1.r
            theta2, r2 = w2.theta, w2.r

        # Phase coherence: cos(Δθ) ∈ [-1, 1]
        phase_coherence = math.cos(theta1 - theta2)

        # Magnitude similarity: min/max ∈ [0, 1]
        if max(r1, r2) > 0.01:
            mag_sim = min(r1, r2) / max(r1, r2)
        else:
            mag_sim = 1.0

        # Base similarity
        base_sim = phase_coherence * mag_sim

        if include_orbital:
            # Boltzmann weight: exp(-Δn/kT) ∈ [0, 1]
            boltzmann = math.exp(-abs(w1.n - w2.n) / KT)
            return boltzmann * (1 + base_sim) / 2  # Scale to [0, 1]
        else:
            return base_sim

    def find_neighbors(self, word: str, n_results: int = 10,
                       use_raw: bool = False, same_orbital: bool = False,
                       max_orbital_diff: float = 1.0) -> List[Tuple[str, float, float]]:
        """
        Find semantically related words.

        Args:
            word: Source word
            n_results: Number of results
            use_raw: Use RAW (semantic) vs DERIVED (usage) coordinates
            same_orbital: Restrict to similar orbital levels
            max_orbital_diff: Maximum orbital difference (if same_orbital=True)

        Returns:
            List of (word, similarity, orbital_n) tuples
        """
        source = self.hierarchy.get_word(word)
        if not source:
            return []

        results = []
        for target_word, target in self.hierarchy.nouns.items():
            if target_word == word:
                continue

            # Optional orbital filter
            if same_orbital and abs(source.n - target.n) > max_orbital_diff:
                continue

            sim = self.similarity(source, target, use_raw=use_raw, include_orbital=True)
            results.append((target_word, sim, target.n))

        results.sort(key=lambda x: -x[1])
        return results[:n_results]

    def coherence(self, w1: QuantumWord, w2: QuantumWord, use_raw: bool = False) -> float:
        """Compute phase coherence: cos(Δθ)."""
        if use_raw:
            delta_theta = w1.theta_raw - w2.theta_raw
        else:
            delta_theta = w1.theta - w2.theta
        return math.cos(delta_theta)

    def singularity_metric(self, w1: QuantumWord, w2: QuantumWord) -> float:
        """
        Compute singularity metric using Weierstrass half-angle formula.

        tan((θ₁-θ₂)/2) reveals whether concepts are:
        - SINGULARITY (|tan| < 0.5): Same transcendental position, different names
          Examples: life/death, peace/war, truth/lie, beginning/end
        - TRANSITIONAL (0.5 < |tan| < 1.5): Partial overlap
        - TRUE OPPOSITES (|tan| > 1.5): Different orientations toward being
          Examples: love/hate, good/evil, god/devil

        Uses RAW coordinates (semantic meaning).

        Returns:
            tan((θ₁-θ₂)/2) - smaller absolute value = more singular
        """
        delta_theta = w1.theta_raw - w2.theta_raw
        return math.tan(delta_theta / 2)

    def classify_pair(self, word1: str, word2: str) -> Tuple[str, float]:
        """
        Classify a word pair as singularity, transitional, or true opposites.

        Returns:
            (classification, singularity_metric)
        """
        qw1 = self.hierarchy.get_word(word1)
        qw2 = self.hierarchy.get_word(word2)

        if not qw1 or not qw2:
            return ('unknown', float('nan'))

        metric = self.singularity_metric(qw1, qw2)
        abs_metric = abs(metric)

        if abs_metric < 0.5:
            return ('singularity', metric)
        elif abs_metric < 1.5:
            return ('transitional', metric)
        else:
            return ('true_opposites', metric)

    # =========================================================================
    # COMBINATION RULES (Chemistry Analogy)
    # =========================================================================
    #
    # ATOM:  (n, l, m) → valence, electronegativity, energy levels
    # WORD:  (n, θ, r) → resonance, energy, intensity
    #
    # Validated from bond space (565,202 bonds):
    #   - Resonant bonds (Δθ < 30°):     1.80x enriched
    #   - Antiresonant bonds (Δθ > 150°): 1.23x enriched
    #   - Mean Δn = +1.7 (adj at higher n than noun)
    # =========================================================================

    # Thresholds (in radians)
    RESONANT_THRESHOLD = math.radians(30)      # |Δθ| < 30°
    ORTHOGONAL_LOW = math.radians(60)          # 60° < |Δθ| < 120°
    ORTHOGONAL_HIGH = math.radians(120)
    ANTIRESONANT_THRESHOLD = math.radians(150)  # |Δθ| > 150°

    def bond_type(self, w1: QuantumWord, w2: QuantumWord) -> str:
        """
        Classify bond type based on phase difference.

        Returns:
            'resonant'     - Δθ < 30° (усиление, 1.8x enriched)
            'orthogonal'   - 60° < Δθ < 120° (независимость)
            'antiresonant' - Δθ > 150° (парадокс, 1.23x enriched)
            'mixed'        - other
        """
        delta_theta = abs(w1.theta - w2.theta)
        if delta_theta > math.pi:
            delta_theta = 2 * math.pi - delta_theta

        if delta_theta < self.RESONANT_THRESHOLD:
            return 'resonant'
        elif self.ORTHOGONAL_LOW < delta_theta < self.ORTHOGONAL_HIGH:
            return 'orthogonal'
        elif delta_theta > self.ANTIRESONANT_THRESHOLD:
            return 'antiresonant'
        else:
            return 'mixed'

    def bond_strength(self, w1: QuantumWord, w2: QuantumWord) -> float:
        """
        Compute bond strength combining resonance and energy.

        Formula:
            S = cos(Δθ) × exp(-|Δn|/kT)

        Higher S = stronger bond (more likely to occur).

        Returns:
            Bond strength in range [-1, 1]
        """
        # Resonance component: cos(Δθ)
        delta_theta = w1.theta - w2.theta
        resonance = math.cos(delta_theta)

        # Energy component: Boltzmann factor
        delta_n = abs(w1.n - w2.n)
        energy = math.exp(-delta_n / KT)

        return resonance * energy

    def can_combine(self, adj: str, noun: str) -> Dict:
        """
        Check if adjective-noun combination is valid.

        Based on validated rules:
            1. RESONANCE: Δθ < 30° preferred (1.8x)
            2. ENERGY: |Δn| < 2 for 54.6% of bonds
            3. ANTIRESONANCE: Δθ > 150° also valid (paradox)

        Returns:
            {
                'valid': bool,
                'bond_type': str,
                'bond_strength': float,
                'delta_theta_deg': float,
                'delta_n': float,
                'reason': str
            }
        """
        qw_adj = self.hierarchy.adjectives.get(adj)
        qw_noun = self.hierarchy.get_word(noun)

        if not qw_adj:
            return {'valid': False, 'reason': f"Unknown adjective: {adj}"}
        if not qw_noun:
            return {'valid': False, 'reason': f"Unknown noun: {noun}"}

        delta_theta = qw_adj.theta - qw_noun.theta
        delta_theta_normalized = delta_theta
        while delta_theta_normalized > math.pi:
            delta_theta_normalized -= 2 * math.pi
        while delta_theta_normalized < -math.pi:
            delta_theta_normalized += 2 * math.pi

        delta_n = qw_adj.n - qw_noun.n
        bond_t = self.bond_type(qw_adj, qw_noun)
        strength = self.bond_strength(qw_adj, qw_noun)

        result = {
            'valid': True,
            'bond_type': bond_t,
            'bond_strength': strength,
            'delta_theta_deg': math.degrees(delta_theta_normalized),
            'delta_n': delta_n,
        }

        # Determine validity and reason
        if bond_t == 'resonant':
            result['reason'] = 'RESONANT: phase alignment amplifies meaning'
        elif bond_t == 'antiresonant':
            result['reason'] = 'ANTIRESONANT: paradox creates new meaning'
        elif bond_t == 'orthogonal':
            result['reason'] = 'ORTHOGONAL: independent combination'
        else:
            result['reason'] = 'MIXED: standard combination'

        return result

    def combine(self, adj: str, noun: str) -> Dict:
        """
        Combine adjective + noun and compute resulting semantic position.

        RESONANT (Δθ ≈ 0):      усиление - amplify in same direction
        ORTHOGONAL (Δθ ≈ 90):   combination - vector addition
        ANTIRESONANT (Δθ ≈ 180): парадокс - synthesis required

        Returns:
            {
                'phrase': str,
                'bond_type': str,
                'result_theta': float,
                'result_r': float,
                'result_n': float,
                'effect': str
            }
        """
        info = self.can_combine(adj, noun)
        if not info['valid']:
            return {'phrase': f"{adj} {noun}", 'error': info['reason']}

        qw_adj = self.hierarchy.adjectives.get(adj)
        qw_noun = self.hierarchy.get_word(noun)

        bond_t = info['bond_type']

        if bond_t == 'resonant':
            # RESONANT: amplify in noun's direction, increase r
            result_theta = qw_noun.theta
            result_r = qw_noun.r + 0.5 * qw_adj.r  # Amplification
            effect = 'AMPLIFICATION: meaning intensified'

        elif bond_t == 'antiresonant':
            # ANTIRESONANT: paradox - average creates new meaning
            result_theta = (qw_adj.theta + qw_noun.theta) / 2
            result_r = math.sqrt(qw_adj.r * qw_noun.r)  # Geometric mean
            effect = 'PARADOX: new meaning synthesized'

        elif bond_t == 'orthogonal':
            # ORTHOGONAL: vector addition
            A1, S1 = qw_adj.A, qw_adj.S
            A2, S2 = qw_noun.A, qw_noun.S
            A_sum = A1 + A2
            S_sum = S1 + S2
            result_theta = math.atan2(S_sum, A_sum)
            result_r = math.sqrt(A_sum**2 + S_sum**2)
            effect = 'COMBINATION: independent meanings merged'

        else:  # mixed
            # Weighted average
            w1, w2 = qw_adj.r, qw_noun.r
            total = w1 + w2
            result_theta = (w1 * qw_adj.theta + w2 * qw_noun.theta) / total
            result_r = (qw_adj.r + qw_noun.r) / 2
            effect = 'BLEND: weighted combination'

        # n is inherited from noun (noun determines abstraction level)
        result_n = qw_noun.n

        return {
            'phrase': f"{adj} {noun}",
            'bond_type': bond_t,
            'result_theta_deg': math.degrees(result_theta),
            'result_r': result_r,
            'result_n': result_n,
            'effect': effect,
            'bond_strength': info['bond_strength']
        }

    def combine_nouns(self, n1: str, n2: str) -> Dict:
        """
        Combine noun + noun (compound noun).

        Uses TWO analyses:
        1. DERIVED (usage): How often these words appear together
           → Determines syntactic compatibility
        2. RAW (semantic): Weierstrass singularity metric
           → Determines semantic relationship

        Combination types:
        - RESONANT + SINGULARITY:    "life death" - same thing, amplify
        - RESONANT + TRUE_OPPOSITES: "love hate" - tension compound
        - ORTHOGONAL + any:          novel compound (independent parts)
        - ANTIRESONANT + any:        paradox compound

        Returns:
            {
                'compound': str,
                'usage_bond': str,      # resonant/orthogonal/antiresonant
                'semantic_type': str,   # singularity/transitional/true_opposites
                'delta_theta_usage': float,
                'singularity_metric': float,
                'effect': str,
                'result_theta': float,
                'result_r': float,
                'result_n': float
            }
        """
        qw1 = self.hierarchy.get_word(n1)
        qw2 = self.hierarchy.get_word(n2)

        if not qw1:
            return {'compound': f"{n1} {n2}", 'error': f"Unknown: {n1}"}
        if not qw2:
            return {'compound': f"{n1} {n2}", 'error': f"Unknown: {n2}"}

        # DERIVED analysis (usage patterns)
        delta_theta_usage = qw1.theta - qw2.theta
        while delta_theta_usage > math.pi:
            delta_theta_usage -= 2 * math.pi
        while delta_theta_usage < -math.pi:
            delta_theta_usage += 2 * math.pi

        delta_theta_abs = abs(delta_theta_usage)
        if delta_theta_abs < self.RESONANT_THRESHOLD:
            usage_bond = 'resonant'
        elif self.ORTHOGONAL_LOW < delta_theta_abs < self.ORTHOGONAL_HIGH:
            usage_bond = 'orthogonal'
        elif delta_theta_abs > self.ANTIRESONANT_THRESHOLD:
            usage_bond = 'antiresonant'
        else:
            usage_bond = 'mixed'

        # RAW analysis (semantic relationship via Weierstrass)
        delta_theta_raw = qw1.theta_raw - qw2.theta_raw
        singularity_metric = math.tan(delta_theta_raw / 2)

        if abs(singularity_metric) < 0.5:
            semantic_type = 'singularity'
        elif abs(singularity_metric) > 1.5:
            semantic_type = 'true_opposites'
        else:
            semantic_type = 'transitional'

        # Determine effect and compute result
        if semantic_type == 'singularity':
            if usage_bond == 'resonant':
                effect = 'UNIFIED: same essence, maximum reinforcement'
                # Average position, increased magnitude
                result_theta = (qw1.theta + qw2.theta) / 2
                result_r = qw1.r + qw2.r
            else:
                effect = 'FACETED: same essence, different perspectives'
                result_theta = (qw1.theta + qw2.theta) / 2
                result_r = max(qw1.r, qw2.r)

        elif semantic_type == 'true_opposites':
            if usage_bond == 'resonant':
                effect = 'TENSION: opposites frequently paired, dialectic potential'
                # Midpoint with reduced magnitude (tension)
                result_theta = (qw1.theta + qw2.theta) / 2
                result_r = abs(qw1.r - qw2.r)
            elif usage_bond == 'antiresonant':
                effect = 'PARADOX MAXIMUM: usage and meaning both oppose'
                result_theta = (qw1.theta_raw + qw2.theta_raw) / 2
                result_r = math.sqrt(qw1.r * qw2.r)
            else:
                effect = 'CONTRAST: opposites in independent combination'
                A_sum = qw1.A + qw2.A
                S_sum = qw1.S + qw2.S
                result_theta = math.atan2(S_sum, A_sum)
                result_r = math.sqrt(A_sum**2 + S_sum**2)

        else:  # transitional
            effect = 'BLEND: partial overlap, standard compound'
            w1, w2 = qw1.r, qw2.r
            total = w1 + w2
            result_theta = (w1 * qw1.theta + w2 * qw2.theta) / total
            result_r = (qw1.r + qw2.r) / 2

        # n inherited from first noun (head of compound)
        result_n = qw1.n

        # Bond strength (usage-based)
        delta_n = abs(qw1.n - qw2.n)
        strength = math.cos(delta_theta_usage) * math.exp(-delta_n / KT)

        return {
            'compound': f"{n1} {n2}",
            'usage_bond': usage_bond,
            'semantic_type': semantic_type,
            'delta_theta_usage_deg': math.degrees(delta_theta_usage),
            'singularity_metric': singularity_metric,
            'strength': strength,
            'effect': effect,
            'result_theta_deg': math.degrees(result_theta),
            'result_r': result_r,
            'result_n': result_n
        }

    def dialectic(self, thesis: str, antithesis: str) -> Dict:
        """
        Dialectical engine for semantic movement.

        TRUE OPPOSITES (|S| > 1.5):
            thesis + antithesis → synthesis
            Hegelian movement: find concept that transcends both

        SINGULARITIES (|S| < 0.5):
            No synthesis possible - they're the same thing.
            Instead: UNFOLD the common essence, reveal unity.

        TRANSITIONAL (0.5 < |S| < 1.5):
            Partial synthesis possible.

        Returns:
            {
                'thesis': str,
                'antithesis': str,
                'classification': str,
                'metric': float,
                'operation': str,  # 'synthesis' | 'unfold' | 'partial'
                'result': list of candidate words
            }
        """
        classification, metric = self.classify_pair(thesis, antithesis)

        qw1 = self.hierarchy.get_word(thesis)
        qw2 = self.hierarchy.get_word(antithesis)

        if not qw1 or not qw2:
            return {
                'thesis': thesis,
                'antithesis': antithesis,
                'classification': 'unknown',
                'metric': float('nan'),
                'operation': 'none',
                'result': []
            }

        result = {
            'thesis': thesis,
            'antithesis': antithesis,
            'classification': classification,
            'metric': metric,
        }

        if classification == 'true_opposites':
            # SYNTHESIS: find concept at midpoint that transcends both
            result['operation'] = 'synthesis'
            mid_theta = (qw1.theta_raw + qw2.theta_raw) / 2
            mid_r = (qw1.r_raw + qw2.r_raw) / 2
            mid_n = (qw1.n + qw2.n) / 2

            # Find words near the synthesis point
            candidates = []
            for word, qw in self.hierarchy.nouns.items():
                if word in (thesis, antithesis):
                    continue
                # Distance to midpoint
                d_theta = abs(qw.theta_raw - mid_theta)
                if d_theta > math.pi:
                    d_theta = 2 * math.pi - d_theta
                d_r = abs(qw.r_raw - mid_r)
                d_n = abs(qw.n - mid_n)
                dist = d_theta + 0.3 * d_r + 0.5 * d_n
                candidates.append((word, dist))

            candidates.sort(key=lambda x: x[1])
            result['result'] = [w for w, _ in candidates[:10]]

        elif classification == 'singularity':
            # UNFOLD: reveal the common essence
            result['operation'] = 'unfold'
            # Find words that are singularities with BOTH thesis and antithesis
            common_theta = (qw1.theta_raw + qw2.theta_raw) / 2
            common_n = (qw1.n + qw2.n) / 2

            candidates = []
            for word, qw in self.hierarchy.nouns.items():
                if word in (thesis, antithesis):
                    continue
                # Check if singularity with both
                m1 = abs(math.tan((qw.theta_raw - qw1.theta_raw) / 2))
                m2 = abs(math.tan((qw.theta_raw - qw2.theta_raw) / 2))
                if m1 < 0.5 and m2 < 0.5:
                    # Also similar orbital
                    if abs(qw.n - common_n) < 1.0:
                        avg_metric = (m1 + m2) / 2
                        candidates.append((word, avg_metric))

            candidates.sort(key=lambda x: x[1])
            result['result'] = [w for w, _ in candidates[:10]]

        else:  # transitional
            # PARTIAL SYNTHESIS: weighted midpoint
            result['operation'] = 'partial_synthesis'
            weight = 1.0 - (abs(metric) - 0.5)  # Higher weight = closer to singularity
            mid_theta = (qw1.theta_raw + qw2.theta_raw) / 2
            mid_n = (qw1.n + qw2.n) / 2

            candidates = []
            for word, qw in self.hierarchy.nouns.items():
                if word in (thesis, antithesis):
                    continue
                d_theta = abs(qw.theta_raw - mid_theta)
                if d_theta > math.pi:
                    d_theta = 2 * math.pi - d_theta
                d_n = abs(qw.n - mid_n)
                dist = d_theta + 0.5 * d_n
                candidates.append((word, dist))

            candidates.sort(key=lambda x: x[1])
            result['result'] = [w for w, _ in candidates[:10]]

        return result

    def path_coherence(self, words: List[str], use_raw: bool = False) -> Tuple[float, List[float]]:
        """
        Compute coherence along a path of words.

        Returns:
            (average_coherence, list_of_step_coherences)
        """
        if len(words) < 2:
            return 1.0, []

        step_coherences = []
        for i in range(len(words) - 1):
            w1 = self.hierarchy.get_word(words[i])
            w2 = self.hierarchy.get_word(words[i + 1])
            if w1 and w2:
                step_coherences.append(self.coherence(w1, w2, use_raw=use_raw))
            else:
                step_coherences.append(0.0)

        avg = sum(step_coherences) / len(step_coherences) if step_coherences else 0.0
        return avg, step_coherences

    def decompose_query(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Decompose query into content nouns and verbs.

        Returns:
            (nouns, verbs)
        """
        words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()

        nouns = []
        verbs = []

        for word in words:
            if word in self._stopwords:
                continue

            # Check if it's a verb in our hierarchy
            if self.hierarchy.get_verb(word):
                verbs.append(word)

            # Check if it's a noun in our derived set
            if word in self.hierarchy.nouns:
                nouns.append(word)

        return nouns, verbs

    def find_semantic_neighbors(self, word: str, n_results: int = 10,
                                  max_orbital_diff: float = 0.5) -> List[Tuple[str, float, float]]:
        """
        Find words with similar MEANING (not just usage).

        Uses RAW coordinates with tight orbital constraint.
        This finds words that occupy similar semantic space.

        Args:
            word: Source word
            n_results: Number of results
            max_orbital_diff: Maximum orbital difference (tighter = more similar)

        Returns:
            List of (word, similarity, orbital_n) tuples
        """
        source = self.hierarchy.get_word(word)
        if not source:
            return []

        results = []
        for target_word, target in self.hierarchy.nouns.items():
            if target_word == word:
                continue

            # Tight orbital constraint
            if abs(source.n - target.n) > max_orbital_diff:
                continue

            # Use RAW coordinates for semantic meaning
            sim = self.similarity(source, target, use_raw=True, include_orbital=True)
            results.append((target_word, sim, target.n))

        results.sort(key=lambda x: -x[1])
        return results[:n_results]

    def find_usage_neighbors(self, word: str, n_results: int = 10,
                             max_orbital_diff: float = 0.5) -> List[Tuple[str, float, float]]:
        """
        Find words used in similar CONTEXTS (not necessarily similar meaning).

        Uses DERIVED coordinates with tight orbital constraint.
        This finds words that appear in similar syntactic/discourse contexts.

        Args:
            word: Source word
            n_results: Number of results
            max_orbital_diff: Maximum orbital difference

        Returns:
            List of (word, similarity, orbital_n) tuples
        """
        source = self.hierarchy.get_word(word)
        if not source:
            return []

        results = []
        for target_word, target in self.hierarchy.nouns.items():
            if target_word == word:
                continue

            # Tight orbital constraint
            if abs(source.n - target.n) > max_orbital_diff:
                continue

            # Use DERIVED coordinates for usage patterns
            sim = self.similarity(source, target, use_raw=False, include_orbital=True)
            results.append((target_word, sim, target.n))

        results.sort(key=lambda x: -x[1])
        return results[:n_results]

    def navigate(self, query: str) -> Dict:
        """
        Navigate a query through semantic space.

        Returns a rich result with:
        - Decomposed query (nouns, verbs)
        - Seed coordinates
        - Neighbors in DERIVED and RAW systems
        - Verb transformations
        - Path coherence
        """
        nouns, verbs = self.decompose_query(query)

        result = {
            'query': query,
            'nouns': nouns,
            'verbs': verbs,
            'seeds': {},
            'neighbors': {},
            'transformations': {},
            'path_coherence': {}
        }

        # Seed coordinates
        for word in nouns:
            qw = self.hierarchy.get_word(word)
            if qw:
                result['seeds'][word] = {
                    'n': qw.n,
                    'tau': qw.tau,
                    'theta_derived_deg': math.degrees(qw.theta),
                    'r_derived': qw.r,
                    'theta_raw_deg': math.degrees(qw.theta_raw),
                    'r_raw': qw.r_raw
                }

        # Neighbors for first 2 seeds (using constrained search)
        for seed in nouns[:2]:
            result['neighbors'][seed] = {
                'semantic': self.find_semantic_neighbors(seed, n_results=8),
                'usage': self.find_usage_neighbors(seed, n_results=8)
            }

        # Verb transformations
        if nouns and verbs:
            seed = nouns[0]
            qw = self.hierarchy.get_word(seed)
            for verb in verbs[:2]:
                vop = self.hierarchy.get_verb(verb)
                if qw and vop:
                    new_theta, new_r = vop.apply(qw)
                    result['transformations'][f'{verb}({seed})'] = {
                        'before': {'theta_deg': math.degrees(qw.theta), 'r': qw.r},
                        'after': {'theta_deg': math.degrees(new_theta), 'r': new_r},
                        'shift': {
                            'delta_theta_deg': math.degrees(new_theta - qw.theta),
                            'delta_r': new_r - qw.r
                        }
                    }

        # Path coherence
        if len(nouns) >= 2:
            avg_der, steps_der = self.path_coherence(nouns, use_raw=False)
            avg_raw, steps_raw = self.path_coherence(nouns, use_raw=True)
            result['path_coherence'] = {
                'derived': {'average': avg_der, 'steps': steps_der},
                'raw': {'average': avg_raw, 'steps': steps_raw}
            }

        return result


def build_hierarchy(data_loader) -> UnifiedHierarchy:
    """
    Build complete hierarchy from data loader.

    TWO COORDINATE SYSTEMS:
    ───────────────────────
    - DERIVED (θ, r): from adjective centroids - captures USAGE patterns
    - RAW (θ_raw, r_raw): from neural embeddings - captures SEMANTIC meaning

    Usage:
        from core.data_loader import DataLoader
        from chain_core.unified_hierarchy import build_hierarchy

        loader = DataLoader()
        hierarchy = build_hierarchy(loader)

        # Get noun with BOTH coordinate systems
        word = hierarchy.get_word('love')
        print(f"love DERIVED: θ={math.degrees(word.theta):.1f}° (usage)")
        print(f"love RAW: θ={math.degrees(word.theta_raw):.1f}° (semantic)")
    """
    hierarchy = UnifiedHierarchy()

    # Load all word vectors (including adjectives)
    print("[Building Hierarchy]")
    word_vectors = data_loader.load_word_vectors()

    # Step 1: Load adjectives (Level 1 → Level 2)
    hierarchy.load_adjective_vectors(word_vectors)

    # Step 2: Expand with all words that have j-vectors
    hierarchy.load_all_word_vectors_as_adjectives(word_vectors)

    # Step 3: Load bonds (Level 3)
    adj_profiles = data_loader.load_noun_adj_profiles()
    hierarchy.load_bonds(adj_profiles)

    # Step 4: Derive noun coordinates (Level 3 → Level 2)
    # Pass word_vectors for RAW coordinates
    hierarchy.derive_noun_coordinates(word_vectors=word_vectors)

    # Step 5: Derive verb operators
    verb_objects = data_loader.load_verb_objects()
    hierarchy.derive_verb_operators(word_vectors, verb_objects)

    # Print summary
    stats = hierarchy.get_statistics()
    print(f"\n[Hierarchy Complete]")
    print(f"  Adjectives: {stats['n_adjectives']}")
    print(f"  Nouns (derived): {stats['n_nouns']}")
    print(f"  Verbs: {stats['n_verbs']}")
    print(f"  Noun n: {stats['noun_n_mean']:.2f} ± {stats['noun_n_std']:.2f}")
    print(f"  Noun r (derived): {stats['noun_r_mean']:.3f} ± {stats['noun_r_std']:.3f}")
    print(f"  Coverage: {stats['noun_coverage_mean']:.1%}")

    return hierarchy


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from core.data_loader import DataLoader

    print("=" * 70)
    print("UNIFIED SEMANTIC HIERARCHY")
    print("=" * 70)
    print(LEVEL_LAWS)
    print("=" * 70)

    loader = DataLoader()
    hierarchy = build_hierarchy(loader)

    # Test some words - show BOTH coordinate systems
    print("\n" + "=" * 70)
    print("TWO COORDINATE SYSTEMS")
    print("=" * 70)
    print("""
    DERIVED (θ, r): from adjective centroids → USAGE patterns
    RAW (θ_raw, r_raw): from neural embeddings → SEMANTIC meaning
    """)

    test_words = ['love', 'hate', 'good', 'evil', 'peace', 'war', 'life', 'death',
                  'god', 'man', 'woman', 'beauty']

    print(f"{'Word':<10} {'n':>5} │ {'θ_der°':>8} {'r_der':>6} │ {'θ_raw°':>8} {'r_raw':>6} │ {'Δθ°':>7}")
    print("-" * 70)

    for word in test_words:
        qw = hierarchy.get_word(word)
        if qw:
            delta_theta = math.degrees(qw.theta - qw.theta_raw)
            print(f"{word:<10} {qw.n:>5.2f} │ {math.degrees(qw.theta):>8.1f} {qw.r:>6.3f} │ "
                  f"{math.degrees(qw.theta_raw):>8.1f} {qw.r_raw:>6.3f} │ {delta_theta:>7.1f}")
        else:
            print(f"{word:<10} -- not found --")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: DERIVED vs RAW")
    print("=" * 70)
    print("""
    DERIVED coordinates cluster together (similar usage patterns)
    RAW coordinates show semantic separation (love ≠ hate)

    For NAVIGATION (Level 4-5): use DERIVED (follows usage patterns)
    For SEMANTIC ANALYSIS: use RAW (captures meaning differences)
    """)

    # Test verb operators
    print("\n" + "=" * 70)
    print("VERB OPERATORS (Phase Shifts)")
    print("=" * 70)

    test_verbs = ['create', 'destroy', 'give', 'take', 'help', 'hurt', 'love', 'hate']

    print(f"\n{'Verb':<12} {'Δθ°':>8} {'Δr':>8} {'Effect':>30}")
    print("-" * 60)

    for verb in test_verbs:
        vop = hierarchy.get_verb(verb)
        if vop:
            if vop.delta_theta > 0:
                effect = f"shifts toward sacred (+{math.degrees(vop.delta_theta):.1f}°)"
            elif vop.delta_theta < 0:
                effect = f"shifts toward profane ({math.degrees(vop.delta_theta):.1f}°)"
            else:
                effect = "no phase shift"
            print(f"{verb:<12} {math.degrees(vop.delta_theta):>8.2f} {vop.delta_r:>8.3f} {effect:>30}")
        else:
            print(f"{verb:<12} -- not found --")
