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

# PC vectors for (A, S) projection
PC1_AFFIRMATION = np.array([-0.448, -0.519, -0.118, -0.480, -0.534])
PC2_SACRED = np.array([-0.513, +0.128, -0.732, +0.420, +0.090])

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


# =============================================================================
# LEVEL 1: TRANSCENDENTALS
# =============================================================================

@dataclass
class Transcendental:
    """
    Level 1: Pure quality in 2D space.

    These are the 5 source dimensions that project down to everything else.
    They exist in a 2D reduced space: (A, S).
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
    Verb as a phase shift operator.

    Verbs don't occupy positions - they TRANSFORM positions.
    verb(noun) → noun' where:
        θ' = θ + Δθ
        r' = r × (1 + Δr/r_verb)
    """
    verb: str
    delta_theta: float  # Phase shift in radians
    delta_r: float      # Magnitude shift factor

    # The verb's own character (for computing shifts)
    theta: float = 0.0  # Base phase
    r: float = 0.0      # Base magnitude

    def apply(self, noun: QuantumWord) -> Tuple[float, float]:
        """
        Apply verb operator to noun, return new (θ, r).
        """
        new_theta = noun.theta + self.delta_theta
        # Normalize to [-π, π]
        while new_theta > math.pi:
            new_theta -= 2 * math.pi
        while new_theta < -math.pi:
            new_theta += 2 * math.pi

        new_r = noun.r + self.delta_r
        return new_theta, max(0, new_r)


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
        Derive verb operators (Δθ, Δr) from verb j-vectors and typical objects.

        A verb's effect is the DIFFERENCE between its character and neutral.
        Verbs shift nouns toward their own character.
        """
        for word, data in word_vectors.items():
            if not data.get('j'):
                continue

            wtype = data.get('word_type')
            if wtype not in ('verb', 1):
                continue

            j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])
            A = float(np.dot(j_5d, PC1_AFFIRMATION))
            S = float(np.dot(j_5d, PC2_SACRED))

            theta = math.atan2(S, A)
            r = math.sqrt(A**2 + S**2)

            # The verb operator shifts toward its own direction
            # Δθ = verb's theta (pulls target toward verb's phase)
            # Δr = verb's r (adds intensity)
            #
            # Scaling: we want small shifts, so divide by typical magnitude
            scale = 0.1  # Verbs have ~10% effect per application

            self.verbs[word] = VerbOperator(
                verb=word,
                delta_theta=theta * scale,
                delta_r=r * scale,
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
