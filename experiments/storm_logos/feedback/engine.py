"""Feedback Engine: Orchestrates error computation."""

from typing import Dict, Optional

from ..data.models import Metrics, Errors
from .comparator import Comparator
from .targets import HOMEOSTATIC_TARGETS, get_targets


class FeedbackEngine:
    """Feedback Engine: computes errors for adaptive control.

    Main loop:
        1. Receive metrics
        2. Compare against targets
        3. Output errors
        4. Errors drive adaptive controller
    """

    def __init__(self, targets: Dict[str, float] = None):
        self.targets = targets or HOMEOSTATIC_TARGETS.copy()
        self.comparator = Comparator()
        self._history: list = []

    def compute_errors(self, metrics: Metrics) -> Errors:
        """Compute errors from metrics.

        Args:
            metrics: Current metrics from MetricsEngine

        Returns:
            Errors for AdaptiveController
        """
        errors = self.comparator.compute(metrics, self.targets)

        # Track history
        self._history.append({
            'metrics': metrics.as_dict(),
            'errors': errors.as_dict(),
        })

        # Limit history size
        if len(self._history) > 100:
            self._history.pop(0)

        return errors

    def set_targets(self, targets: Dict[str, float]):
        """Update homeostatic targets.

        Args:
            targets: New target values
        """
        self.targets.update(targets)

    def use_preset(self, name: str):
        """Use a preset target set.

        Args:
            name: 'therapeutic', 'generation', 'balanced'
        """
        self.targets = get_targets(name)

    def reset(self):
        """Reset all accumulators."""
        self.comparator.reset_integral()
        self._history.clear()

    def get_integral(self) -> Dict[str, float]:
        """Get accumulated error integrals."""
        return self.comparator.get_all_integrals()

    def get_history(self, n: int = None) -> list:
        """Get error history.

        Args:
            n: Number of recent entries (None = all)

        Returns:
            List of {metrics, errors} dicts
        """
        if n:
            return self._history[-n:]
        return self._history


# ============================================================================
# SINGLETON
# ============================================================================

_engine_instance: Optional[FeedbackEngine] = None


def get_feedback_engine() -> FeedbackEngine:
    """Get singleton FeedbackEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FeedbackEngine()
    return _engine_instance
