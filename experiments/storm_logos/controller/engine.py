"""Adaptive Controller Engine: Orchestrates parameter adaptation."""

from typing import Dict, Optional

from ..data.models import Errors, Parameters
from .parameters import ADAPTIVE_PARAMETERS, get_initial_params
from .rules import get_rules
from .pi_controller import PIController


class AdaptiveController:
    """Adaptive Controller: adjusts generation parameters based on feedback.

    Usage:
        controller = AdaptiveController()
        new_params = controller.adapt(errors)
    """

    def __init__(self, context: str = 'default'):
        self.context = context
        self.rules = get_rules(context)
        self.pi = PIController(self.rules)
        self._params = Parameters(**get_initial_params())
        self._history: list = []

    @property
    def parameters(self) -> Parameters:
        """Current parameters."""
        return self._params

    def adapt(self, errors: Errors) -> Parameters:
        """Adapt parameters based on errors.

        Args:
            errors: Errors from FeedbackEngine

        Returns:
            Updated parameters
        """
        new_params = self.pi.adapt(self._params, errors)

        # Track history
        self._history.append({
            'before': self._params.as_dict(),
            'errors': errors.as_dict(),
            'after': new_params.as_dict(),
        })

        if len(self._history) > 100:
            self._history.pop(0)

        self._params = new_params
        return new_params

    def get_parameters(self) -> Parameters:
        """Get current parameters."""
        return self._params

    def set_parameters(self, params: Parameters):
        """Manually set parameters."""
        self._params = params

    def reset(self):
        """Reset to initial parameters."""
        self._params = Parameters(**get_initial_params())
        self.pi.reset()
        self._history.clear()

    def set_context(self, context: str):
        """Switch adaptation context.

        Args:
            context: 'default', 'therapeutic', 'generation'
        """
        self.context = context
        self.rules = get_rules(context)
        self.pi.set_rules(self.rules)

    def get_history(self, n: int = None) -> list:
        """Get adaptation history."""
        if n:
            return self._history[-n:]
        return self._history


# ============================================================================
# SINGLETON
# ============================================================================

_controller_instance: Optional[AdaptiveController] = None


def get_controller() -> AdaptiveController:
    """Get singleton AdaptiveController instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = AdaptiveController()
    return _controller_instance
