"""Analyzer Application: Analysis-only agent.

Analyzes text/trajectories without generation.
"""

from typing import List, Optional

from ..data.models import Trajectory, Metrics, SemanticState
from ..metrics.engine import MetricsEngine
from ..semantic.dialectic import Dialectic


class Analyzer:
    """Analysis-only application.

    Provides:
    - Text analysis (metrics, defenses, dialectic)
    - Trajectory analysis
    - Genre classification
    """

    def __init__(self):
        self.metrics = MetricsEngine()
        self.dialectic = Dialectic()

    def analyze_text(self, text: str) -> dict:
        """Full analysis of text.

        Args:
            text: Text to analyze

        Returns:
            Complete analysis dictionary
        """
        # Metrics
        m = self.metrics.measure(text=text)

        # Dialectic
        state = SemanticState(A=m.A_position, S=m.S_position, tau=m.tau_mean)
        dial = self.dialectic.analyze(state)

        return {
            'metrics': m.as_dict(),
            'position': {
                'A': m.A_position,
                'S': m.S_position,
                'tau': m.tau_mean,
            },
            'defenses': m.defenses,
            'dialectic': dial,
            'coherence': m.coherence,
            'irony': m.irony,
            'tension': m.tension_score,
        }

    def analyze_trajectory(self, trajectory: Trajectory) -> dict:
        """Analyze a trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Trajectory analysis
        """
        m = self.metrics.measure(trajectory=trajectory)

        return {
            'n_bonds': len(trajectory.bonds),
            'coherence': m.coherence,
            'noise_ratio': m.noise_ratio,
            'tau_mean': m.tau_mean,
            'tau_variance': m.tau_variance,
            'tau_slope': m.tau_slope,
            'tension': m.tension_score,
        }

    def classify_genre(self, trajectory: Trajectory) -> dict:
        """Classify genre from trajectory.

        Based on boundary jump patterns:
        - τ ratio
        - A ratio
        - S ratio

        Args:
            trajectory: Trajectory to classify

        Returns:
            Genre classification
        """
        from ..metrics.analyzers.boundary import BoundaryAnalyzer
        boundary = BoundaryAnalyzer()

        # Get boundary and within-segment stats
        boundary_stats = boundary.compute_boundary_jumps(trajectory)
        within_stats = boundary.compute_within_segment_stats(trajectory)

        # Compute ratios
        tau_ratio = self._safe_ratio(
            boundary_stats.get('mean_tau_jump', 0),
            within_stats.get('mean_within_tau', 0.1)
        )
        A_ratio = self._safe_ratio(
            boundary_stats.get('mean_A_jump', 0),
            within_stats.get('mean_within_A', 0.1)
        )
        S_ratio = self._safe_ratio(
            boundary_stats.get('mean_S_jump', 0),
            within_stats.get('mean_within_S', 0.1)
        )

        # Genre centroids (from validation)
        genres = {
            'dramatic': (0.88, 1.18, 1.11),
            'ironic': (0.88, 1.14, 0.99),
            'balanced': (1.01, 1.06, 1.06),
        }

        # Find closest genre
        feature = (tau_ratio, A_ratio, S_ratio)
        best_genre = 'balanced'
        best_dist = float('inf')

        for genre, centroid in genres.items():
            dist = sum((f - c)**2 for f, c in zip(feature, centroid))
            if dist < best_dist:
                best_dist = dist
                best_genre = genre

        return {
            'genre': best_genre,
            'confidence': 1.0 / (1.0 + best_dist),
            'features': {
                'tau_ratio': tau_ratio,
                'A_ratio': A_ratio,
                'S_ratio': S_ratio,
            }
        }

    def _safe_ratio(self, num: float, denom: float) -> float:
        """Compute ratio with safety."""
        if abs(denom) < 0.01:
            return 1.0
        return num / denom

    def compare(self, text1: str, text2: str) -> dict:
        """Compare two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Comparison results
        """
        a1 = self.analyze_text(text1)
        a2 = self.analyze_text(text2)

        return {
            'text1': a1,
            'text2': a2,
            'differences': {
                'A_diff': a2['position']['A'] - a1['position']['A'],
                'S_diff': a2['position']['S'] - a1['position']['S'],
                'tau_diff': a2['position']['tau'] - a1['position']['tau'],
                'coherence_diff': a2['coherence'] - a1['coherence'],
                'irony_diff': a2['irony'] - a1['irony'],
            }
        }
