"""Tau Analyzer: Track abstraction level dynamics."""

from typing import List, Optional, Tuple
import numpy as np

from ...data.models import Bond, Trajectory, SemanticState


class TauAnalyzer:
    """Analyze τ (abstraction level) dynamics.

    τ tracks:
    - How abstract/concrete the discourse is
    - Movement between abstraction levels
    - Variance (narrative "breathing")
    """

    def analyze(self, trajectory: Trajectory = None,
                bonds: List[Bond] = None,
                states: List[SemanticState] = None) -> dict:
        """Analyze τ dynamics.

        Args:
            trajectory: Trajectory to analyze
            bonds: List of bonds (alternative)
            states: List of states (alternative)

        Returns:
            Dictionary with tau metrics
        """
        # Extract tau values
        if trajectory:
            tau_vals = [b.tau for b in trajectory.bonds]
        elif bonds:
            tau_vals = [b.tau for b in bonds]
        elif states:
            tau_vals = [s.tau for s in states]
        else:
            return {
                'tau_mean': 2.5,
                'tau_variance': 0.0,
                'tau_slope': 0.0,
                'tau_min': 2.5,
                'tau_max': 2.5,
            }

        if not tau_vals:
            return {
                'tau_mean': 2.5,
                'tau_variance': 0.0,
                'tau_slope': 0.0,
                'tau_min': 2.5,
                'tau_max': 2.5,
            }

        tau = np.array(tau_vals)

        return {
            'tau_mean': np.mean(tau),
            'tau_variance': np.var(tau),
            'tau_slope': self._compute_slope(tau),
            'tau_min': np.min(tau),
            'tau_max': np.max(tau),
            'tau_range': np.max(tau) - np.min(tau),
        }

    def _compute_slope(self, tau: np.ndarray) -> float:
        """Compute linear slope of tau values."""
        if len(tau) < 2:
            return 0.0

        x = np.arange(len(tau))
        covariance = np.cov(x, tau)[0, 1]
        variance = np.var(x)

        if variance < 1e-6:
            return 0.0

        return covariance / variance

    def compute_breathing(self, trajectory: Trajectory,
                          window: int = 5) -> float:
        """Compute "breathing" metric: variance over sliding windows.

        High breathing = text moves between abstract and concrete.
        Low breathing = flat, monotonous abstraction level.

        Args:
            trajectory: Trajectory to analyze
            window: Window size

        Returns:
            Breathing score (variance of variances)
        """
        if len(trajectory.bonds) < window * 2:
            return 0.0

        tau_vals = [b.tau for b in trajectory.bonds]
        tau = np.array(tau_vals)

        window_variances = []
        for i in range(len(tau) - window + 1):
            w = tau[i:i+window]
            window_variances.append(np.var(w))

        return np.mean(window_variances)

    def detect_boundary(self, prev_tau: float, curr_tau: float,
                        threshold: float = 0.5) -> bool:
        """Detect if there's a sentence boundary (large τ jump).

        Args:
            prev_tau: Previous τ value
            curr_tau: Current τ value
            threshold: Jump threshold

        Returns:
            True if boundary detected
        """
        return abs(curr_tau - prev_tau) > threshold

    def compute_autocorrelation(self, trajectory: Trajectory,
                                lag: int = 1) -> float:
        """Compute autocorrelation of τ values.

        High autocorrelation = persistent abstraction levels.
        Low autocorrelation = rapidly changing.

        Args:
            trajectory: Trajectory to analyze
            lag: Lag for autocorrelation

        Returns:
            Autocorrelation coefficient
        """
        if len(trajectory.bonds) < lag + 2:
            return 0.0

        tau_vals = np.array([b.tau for b in trajectory.bonds])

        # Compute autocorrelation
        n = len(tau_vals)
        mean = np.mean(tau_vals)
        var = np.var(tau_vals)

        if var < 1e-6:
            return 1.0

        autocorr = np.sum(
            (tau_vals[:-lag] - mean) * (tau_vals[lag:] - mean)
        ) / (n * var)

        return autocorr
