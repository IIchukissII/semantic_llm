"""PI Controller: Proportional-Integral control for parameters.

Implements:
    Δp = η × error × direction + κ × integral × direction
    p_new = clamp(p + Δp, range)
"""

from typing import Dict
from collections import defaultdict

from ..data.models import Errors, Parameters
from .parameters import AdaptationRule, ADAPTIVE_PARAMETERS


class PIController:
    """Proportional-Integral controller for adaptive parameters.

    For each parameter p with rule:
        - Compute error from target metric
        - P term: η × error × direction
        - I term: κ × integral × direction
        - Update: p_new = clamp(p + P + I, range)
    """

    def __init__(self, rules: Dict[str, AdaptationRule] = None):
        self.rules = rules or ADAPTIVE_PARAMETERS
        self._integral: Dict[str, float] = defaultdict(float)

    def adapt(self, params: Parameters, errors: Errors) -> Parameters:
        """Adapt parameters based on errors.

        Args:
            params: Current parameters
            errors: Errors from FeedbackEngine

        Returns:
            Updated parameters
        """
        params_dict = params.as_dict()
        errors_dict = errors.as_dict()

        new_params = params_dict.copy()

        for name, rule in self.rules.items():
            if name not in params_dict:
                continue

            # Get error for target metric
            error_key = f'{rule.target_metric}_error'
            if error_key in errors_dict:
                error = errors_dict[error_key]
            else:
                # Try to compute from integral
                error = errors.integral.get(rule.target_metric, 0.0)

            if error == 0:
                continue

            # P term
            p_term = rule.eta * error * rule.direction

            # I term
            self._integral[name] += error
            i_term = rule.kappa * self._integral[name] * rule.direction

            # Update
            delta = p_term + i_term
            new_val = params_dict[name] + delta
            new_params[name] = rule.clamp(new_val)

        return Parameters(**new_params)

    def reset(self, param: str = None):
        """Reset integral accumulators.

        Args:
            param: Specific parameter, or None for all
        """
        if param:
            self._integral[param] = 0.0
        else:
            self._integral.clear()

    def get_integral(self, param: str) -> float:
        """Get integral value for a parameter."""
        return self._integral.get(param, 0.0)

    def set_rules(self, rules: Dict[str, AdaptationRule]):
        """Update adaptation rules."""
        self.rules = rules


class PIDController(PIController):
    """PID Controller with derivative term.

    Adds: D term = μ × derivative × direction
    """

    def __init__(self, rules: Dict[str, AdaptationRule] = None):
        super().__init__(rules)
        self._prev_errors: Dict[str, float] = {}

    def adapt(self, params: Parameters, errors: Errors) -> Parameters:
        """Adapt with PID control."""
        params_dict = params.as_dict()
        errors_dict = errors.as_dict()

        new_params = params_dict.copy()

        for name, rule in self.rules.items():
            if name not in params_dict:
                continue

            error_key = f'{rule.target_metric}_error'
            error = errors_dict.get(error_key, 0.0)

            if error == 0:
                continue

            # P term
            p_term = rule.eta * error * rule.direction

            # I term
            self._integral[name] += error
            i_term = rule.kappa * self._integral[name] * rule.direction

            # D term (derivative)
            prev = self._prev_errors.get(name, error)
            derivative = error - prev
            d_term = 0.05 * derivative * rule.direction  # Fixed D coefficient
            self._prev_errors[name] = error

            # Update
            delta = p_term + i_term + d_term
            new_val = params_dict[name] + delta
            new_params[name] = rule.clamp(new_val)

        return Parameters(**new_params)
