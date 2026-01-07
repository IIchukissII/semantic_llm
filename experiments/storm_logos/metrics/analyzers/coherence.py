"""Coherence Analyzer: Measure trajectory coherence."""

from typing import List, Optional
import numpy as np

from ...data.models import Bond, Trajectory, SemanticState
from ...semantic.physics import coherence as compute_coherence


class CoherenceAnalyzer:
    """Analyze coherence of semantic trajectories.

    Coherence = how well the path holds together.

    Measured via:
    1. Pairwise cosine similarity in A-S plane
    2. Boltzmann probability of τ transitions
    3. Smoothness of movement
    """

    def analyze(self, trajectory: Trajectory = None,
                bonds: List[Bond] = None) -> float:
        """Analyze coherence of trajectory or bonds.

        Args:
            trajectory: Trajectory to analyze
            bonds: List of bonds (alternative to trajectory)

        Returns:
            Coherence score (0 to 1)
        """
        if trajectory:
            bonds = trajectory.bonds

        if not bonds or len(bonds) < 2:
            return 1.0

        coherences = []
        for i in range(1, len(bonds)):
            prev = bonds[i-1]
            curr = bonds[i]

            # Create state from previous bond
            state = SemanticState(A=prev.A, S=prev.S, tau=prev.tau)
            coh = compute_coherence(state, curr)
            coherences.append(coh)

        # Mean coherence
        mean_coh = np.mean(coherences)

        # Normalize to 0-1 (coherence can be -1 to 1)
        return (mean_coh + 1) / 2

    def analyze_window(self, bonds: List[Bond], window: int = 5) -> float:
        """Analyze coherence over sliding window.

        Args:
            bonds: List of bonds
            window: Window size

        Returns:
            Mean windowed coherence
        """
        if len(bonds) < 2:
            return 1.0

        window_coherences = []
        for i in range(len(bonds) - window + 1):
            window_bonds = bonds[i:i+window]
            coh = self.analyze(bonds=window_bonds)
            window_coherences.append(coh)

        return np.mean(window_coherences) if window_coherences else 1.0

    def compute_noise_ratio(self, trajectory: Trajectory) -> float:
        """Compute noise ratio = 1 - coherence.

        Higher noise = less coherent path.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Noise ratio (0 to 1)
        """
        coherence = self.analyze(trajectory)
        return 1 - coherence

    def compute_transition_probability(self, bonds: List[Bond]) -> float:
        """Compute mean transition probability.

        Uses Boltzmann distribution for τ transitions.

        Args:
            bonds: List of bonds

        Returns:
            Mean transition probability
        """
        if len(bonds) < 2:
            return 1.0

        from ...config import KT
        import math

        probs = []
        for i in range(1, len(bonds)):
            delta_tau = abs(bonds[i].tau - bonds[i-1].tau)
            prob = math.exp(-delta_tau / KT)
            probs.append(prob)

        return np.mean(probs)

    def analyze_smoothness(self, trajectory: Trajectory) -> float:
        """Analyze smoothness of trajectory.

        Smooth = gradual changes.
        Jumpy = sudden large changes.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Smoothness score (0 to 1, higher = smoother)
        """
        if len(trajectory.bonds) < 3:
            return 1.0

        coords = trajectory.get_coords()

        # Compute second derivative (acceleration)
        velocities = np.diff(coords, axis=0)
        accelerations = np.diff(velocities, axis=0)

        # Smoothness = inverse of acceleration magnitude
        acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        mean_acc = np.mean(acc_magnitudes)

        # Normalize: high acceleration = low smoothness
        smoothness = 1 / (1 + mean_acc)

        return smoothness
