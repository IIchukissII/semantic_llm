"""
j-Space: Transcendental Vector Space Utilities
===============================================

This module provides utilities for working with j-vectors (transcendental space).

THE 2D REDUCTION (Validated Jan 2026):
    5 transcendentals (beauty, life, sacred, good, love) reduce to 2 dimensions:
    - PC1 (83.3%): AFFIRMATION = beauty + life + good + love
    - PC2 (11.7%): SACRED (partially orthogonal)

    The correlation matrix shows:
    - life ↔ love: r = 0.954 (nearly identical)
    - sacred ↔ good: r = 0.133 (nearly orthogonal)

    This means beauty, life, good, love are ONE thing (The Affirmation).
    Sacred is a separate, orthogonal dimension (Transcendence).

Usage:
    from chain_core.j_space import JSpace

    js = JSpace()
    j_5d = js.get_j_vector(word_data)           # 5D vector
    a, s = js.project_2d(j_5d)                   # 2D projection
    sim = js.similarity_2d(word1_data, word2_data)  # 2D similarity
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union

# The 5 transcendental dimensions
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']

# The 11 surface dimensions (i-space)
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']

# Principal Component vectors (from exp10c validation)
# These project 5D j-space to 2D (Affirmation, Sacred)
PC1_AFFIRMATION = np.array([-0.448, -0.519, -0.118, -0.480, -0.534])
PC2_SACRED = np.array([-0.513, +0.128, -0.732, +0.420, +0.090])

# Variance explained by each component
VAR_AFFIRMATION = 0.833  # 83.3%
VAR_SACRED = 0.117       # 11.7%

# Correlation constants
CORR_LIFE_LOVE = 0.954
CORR_SACRED_GOOD = 0.133


class JSpace:
    """
    Utility class for j-vector (transcendental space) operations.

    Supports both 5D (original) and 2D (reduced) representations.
    """

    def __init__(self, j_mean: Optional[np.ndarray] = None):
        """
        Initialize JSpace.

        Args:
            j_mean: Optional mean j-vector for phase shift centering.
                   If not provided, will be computed on first use.
        """
        self._j_mean = j_mean
        self._pc1 = PC1_AFFIRMATION
        self._pc2 = PC2_SACRED

    @staticmethod
    def get_j_vector(word_data: Dict) -> np.ndarray:
        """
        Extract 5D j-vector from word data.

        Args:
            word_data: Dictionary with 'j' key containing {dim: value}

        Returns:
            5D numpy array [beauty, life, sacred, good, love]
        """
        j_dict = word_data.get('j', {})
        return np.array([j_dict.get(dim, 0.0) for dim in J_DIMS])

    @staticmethod
    def get_i_vector(word_data: Dict) -> np.ndarray:
        """
        Extract 11D i-vector (surface dimensions) from word data.

        Args:
            word_data: Dictionary with 'i' key containing {dim: value}

        Returns:
            11D numpy array
        """
        i_dict = word_data.get('i', {})
        return np.array([i_dict.get(dim, 0.0) for dim in I_DIMS])

    @staticmethod
    def get_tau(word_data: Dict) -> float:
        """Get τ (abstractness) from word data."""
        return word_data.get('tau', 2.5)

    def project_2d(self, j_5d: np.ndarray,
                   centered: bool = False,
                   j_mean: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Project 5D j-vector to 2D (Affirmation, Sacred).

        Args:
            j_5d: 5D j-vector
            centered: If True, subtract j_mean before projection
            j_mean: Mean vector for centering (uses stored if not provided)

        Returns:
            Tuple (affirmation_score, sacred_score)
        """
        if centered:
            mean = j_mean if j_mean is not None else self._j_mean
            if mean is not None:
                j_5d = j_5d - mean

        affirmation = float(np.dot(j_5d, self._pc1))
        sacred = float(np.dot(j_5d, self._pc2))

        return affirmation, sacred

    def project_2d_batch(self, j_vectors: np.ndarray,
                         centered: bool = False) -> np.ndarray:
        """
        Project multiple 5D j-vectors to 2D.

        Args:
            j_vectors: Array of shape (N, 5)
            centered: If True, subtract j_mean

        Returns:
            Array of shape (N, 2) with [affirmation, sacred] columns
        """
        if centered and self._j_mean is not None:
            j_vectors = j_vectors - self._j_mean

        result = np.zeros((len(j_vectors), 2))
        result[:, 0] = j_vectors @ self._pc1
        result[:, 1] = j_vectors @ self._pc2

        return result

    def similarity_2d(self,
                      word1_data: Dict,
                      word2_data: Dict,
                      include_tau: bool = True) -> float:
        """
        Compute similarity using 2D projection.

        Formula: sim = α×(A₁×A₂) + β×(S₁×S₂) + γ×τ_proximity
        With α=0.83, β=0.12, γ=0.05 (from variance explained)

        Args:
            word1_data, word2_data: Word data dictionaries
            include_tau: If True, include τ-proximity in similarity

        Returns:
            Similarity score
        """
        j1 = self.get_j_vector(word1_data)
        j2 = self.get_j_vector(word2_data)

        a1, s1 = self.project_2d(j1)
        a2, s2 = self.project_2d(j2)

        # Normalize for cosine-like similarity
        mag1 = np.sqrt(a1**2 + s1**2)
        mag2 = np.sqrt(a2**2 + s2**2)

        if mag1 < 1e-10 or mag2 < 1e-10:
            return 0.0

        # Weighted dot product
        sim = VAR_AFFIRMATION * (a1 * a2) + VAR_SACRED * (s1 * s2)
        sim = sim / (mag1 * mag2)

        if include_tau:
            tau1 = self.get_tau(word1_data)
            tau2 = self.get_tau(word2_data)
            tau_prox = 1.0 - abs(tau1 - tau2) / 5.0
            sim = 0.95 * sim + 0.05 * tau_prox

        return sim

    def phase_2d(self, j_5d: np.ndarray) -> float:
        """
        Compute phase angle in 2D space.

        θ = atan2(Sacred, Affirmation)

        θ = 0°   → Pure affirmation
        θ = 90°  → Pure sacred
        θ = 180° → Pure negation
        θ = 270° → Pure profane

        Args:
            j_5d: 5D j-vector

        Returns:
            Phase angle in radians
        """
        a, s = self.project_2d(j_5d)
        return np.arctan2(s, a)

    def entropy_2d(self, j_5d: np.ndarray) -> float:
        """
        Compute Shannon entropy in 2D space.

        H = -p_A×log₂(p_A) - p_S×log₂(p_S)
        where p_A = |A|/(|A|+|S|), p_S = |S|/(|A|+|S|)

        Max entropy: 1 bit (when |A| = |S|)

        Args:
            j_5d: 5D j-vector

        Returns:
            Entropy in bits (0 to 1)
        """
        a, s = self.project_2d(j_5d)

        abs_a = abs(a)
        abs_s = abs(s)
        total = abs_a + abs_s

        if total < 1e-10:
            return 0.0

        p_a = abs_a / total
        p_s = abs_s / total

        entropy = 0.0
        if p_a > 1e-10:
            entropy -= p_a * np.log2(p_a)
        if p_s > 1e-10:
            entropy -= p_s * np.log2(p_s)

        return entropy

    def quadrant(self, j_5d: np.ndarray) -> int:
        """
        Determine which quadrant the j-vector falls in.

        Q1: Affirmation+, Sacred+  (41% of words)
        Q2: Affirmation-, Sacred+  (9%)
        Q3: Affirmation-, Sacred-  (41%)
        Q4: Affirmation+, Sacred-  (10%)

        Args:
            j_5d: 5D j-vector

        Returns:
            Quadrant number (1-4)
        """
        a, s = self.project_2d(j_5d)

        if a >= 0 and s >= 0:
            return 1
        elif a < 0 and s >= 0:
            return 2
        elif a < 0 and s < 0:
            return 3
        else:
            return 4

    def is_dialectical_opposite(self, j1_5d: np.ndarray, j2_5d: np.ndarray) -> bool:
        """
        Check if two j-vectors are in dialectically opposite quadrants.

        Opposite pairs: Q1↔Q3, Q2↔Q4

        Args:
            j1_5d, j2_5d: 5D j-vectors

        Returns:
            True if in opposite quadrants
        """
        q1 = self.quadrant(j1_5d)
        q2 = self.quadrant(j2_5d)

        return (q1 == 1 and q2 == 3) or (q1 == 3 and q2 == 1) or \
               (q1 == 2 and q2 == 4) or (q1 == 4 and q2 == 2)

    def compute_mean(self, word_vectors: Dict[str, Dict]) -> np.ndarray:
        """
        Compute mean j-vector from word vectors dictionary.

        Args:
            word_vectors: {word: word_data} dictionary

        Returns:
            Mean j-vector (5D)
        """
        all_j = []
        for word, data in word_vectors.items():
            j = self.get_j_vector(data)
            if np.linalg.norm(j) > 0.01:
                all_j.append(j)

        self._j_mean = np.mean(all_j, axis=0)
        return self._j_mean

    def decompose_word(self, word_data: Dict) -> Dict:
        """
        Full decomposition of a word using 2D + τ model.

        Args:
            word_data: Word data dictionary

        Returns:
            Dictionary with all decomposition components
        """
        j_5d = self.get_j_vector(word_data)
        i_11d = self.get_i_vector(word_data)
        tau = self.get_tau(word_data)

        affirmation, sacred = self.project_2d(j_5d)
        phase = self.phase_2d(j_5d)
        entropy = self.entropy_2d(j_5d)
        quad = self.quadrant(j_5d)

        return {
            # Original 5D
            'j_5d': j_5d,
            'i_11d': i_11d,
            'tau': tau,

            # 2D reduction
            'affirmation': affirmation,
            'sacred': sacred,

            # Derived quantities
            'phase': phase,
            'phase_deg': np.degrees(phase),
            'entropy_2d': entropy,
            'quadrant': quad,
            'magnitude_2d': np.sqrt(affirmation**2 + sacred**2),

            # Interpretation
            'is_affirming': affirmation > 0,
            'is_sacred': sacred > 0,
        }


# Convenience function for quick access
def get_j_vector(word_data: Dict) -> np.ndarray:
    """Quick access to j-vector extraction."""
    return JSpace.get_j_vector(word_data)


def project_2d(j_5d: np.ndarray) -> Tuple[float, float]:
    """Quick access to 2D projection."""
    return JSpace().project_2d(j_5d)


# Singleton instance for common use
_default_jspace = None

def get_jspace() -> JSpace:
    """Get the default JSpace instance."""
    global _default_jspace
    if _default_jspace is None:
        _default_jspace = JSpace()
    return _default_jspace
