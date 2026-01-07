"""State Extractor: Extract metrics from semantic state and trajectory."""

from typing import List, Tuple, Optional
import numpy as np

from ...data.models import SemanticState, Trajectory, Bond


class StateExtractor:
    """Extract metrics from semantic states and trajectories."""

    def extract_from_trajectory(self, trajectory: Trajectory) -> dict:
        """Extract metrics from a trajectory.

        Args:
            trajectory: Trajectory of bonds

        Returns:
            Dictionary of metrics
        """
        if not trajectory.bonds:
            return {
                'n_bonds': 0,
                'tau_mean': 2.5,
                'tau_variance': 0.0,
                'tau_slope': 0.0,
                'A_mean': 0.0,
                'S_mean': 0.0,
            }

        coords = trajectory.get_coords()

        # τ statistics
        tau = coords[:, 2]
        tau_mean = np.mean(tau)
        tau_variance = np.var(tau)
        tau_slope = self._compute_slope(tau)

        # A and S statistics
        A = coords[:, 0]
        S = coords[:, 1]

        return {
            'n_bonds': len(trajectory.bonds),
            'tau_mean': tau_mean,
            'tau_variance': tau_variance,
            'tau_slope': tau_slope,
            'A_mean': np.mean(A),
            'A_variance': np.var(A),
            'S_mean': np.mean(S),
            'S_variance': np.var(S),
        }

    def _compute_slope(self, values: np.ndarray) -> float:
        """Compute linear slope of values."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        covariance = np.cov(x, values)[0, 1]
        variance = np.var(x)

        if variance < 1e-6:
            return 0.0

        return covariance / variance

    def extract_from_state(self, state: SemanticState,
                           history: Optional[List[SemanticState]] = None) -> dict:
        """Extract metrics from current state and optional history.

        Args:
            state: Current semantic state
            history: Optional history of states

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'A_position': state.A,
            'S_position': state.S,
            'tau_position': state.tau,
            'irony': state.irony,
            'sarcasm': state.sarcasm,
            'intensity': state.intensity,
        }

        if history and len(history) >= 2:
            # Compute velocity
            prev = history[-2]
            metrics['velocity_A'] = state.A - prev.A
            metrics['velocity_S'] = state.S - prev.S
            metrics['velocity_tau'] = state.tau - prev.tau

            # Compute history stats
            A_vals = [s.A for s in history]
            S_vals = [s.S for s in history]
            tau_vals = [s.tau for s in history]

            metrics['tau_mean'] = np.mean(tau_vals)
            metrics['tau_variance'] = np.var(tau_vals)
            metrics['tau_slope'] = self._compute_slope(np.array(tau_vals))

        return metrics

    def compute_transition_metrics(self, prev: Bond, curr: Bond) -> dict:
        """Compute metrics for a single transition.

        Args:
            prev: Previous bond
            curr: Current bond

        Returns:
            Dictionary of transition metrics
        """
        return {
            'delta_A': abs(curr.A - prev.A),
            'delta_S': abs(curr.S - prev.S),
            'delta_tau': abs(curr.tau - prev.tau),
            'distance': np.linalg.norm(
                np.array([curr.A, curr.S, curr.tau]) -
                np.array([prev.A, prev.S, prev.tau])
            ),
        }
