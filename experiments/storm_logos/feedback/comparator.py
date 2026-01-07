"""Comparator: Compute errors between metrics and targets."""

from typing import Dict
from collections import defaultdict

from ..data.models import Metrics, Errors


class Comparator:
    """Compare metrics against homeostatic targets.

    Computes:
        - Error = target - current
        - Integral = accumulated error
        - Derivative = rate of change
    """

    def __init__(self):
        self._integral: Dict[str, float] = defaultdict(float)
        self._prev_errors: Dict[str, float] = {}

    def compute(self, metrics: Metrics, targets: Dict[str, float]) -> Errors:
        """Compute errors between metrics and targets.

        Args:
            metrics: Current metrics
            targets: Homeostatic targets

        Returns:
            Errors object
        """
        errors = Errors()
        metrics_dict = metrics.as_dict()

        # Compute error for each target
        for metric, target in targets.items():
            if metric in metrics_dict:
                current = metrics_dict[metric]
                if isinstance(current, (int, float)):
                    error = target - current

                    # Store in appropriate field
                    if metric == 'coherence':
                        errors.coherence_error = error
                    elif metric == 'irony':
                        errors.irony_error = error
                    elif metric == 'tension':
                        errors.tension_error = error
                    elif metric == 'tau_slope':
                        errors.tau_slope_error = error
                    elif metric == 'tau_variance':
                        errors.tau_variance_error = error
                    elif metric == 'noise_ratio':
                        errors.noise_ratio_error = error

                    # Update integral
                    self._integral[metric] += error
                    errors.integral[metric] = self._integral[metric]

                    # Compute derivative
                    if metric in self._prev_errors:
                        errors.derivative[metric] = error - self._prev_errors[metric]
                    else:
                        errors.derivative[metric] = 0.0

                    self._prev_errors[metric] = error

        return errors

    def reset_integral(self, metric: str = None):
        """Reset integral accumulator.

        Args:
            metric: Specific metric to reset, or None for all
        """
        if metric:
            self._integral[metric] = 0.0
        else:
            self._integral.clear()

    def get_integral(self, metric: str) -> float:
        """Get accumulated integral for a metric."""
        return self._integral.get(metric, 0.0)

    def get_all_integrals(self) -> Dict[str, float]:
        """Get all integral values."""
        return dict(self._integral)
