"""
Weight Dynamics: Learning and Forgetting

Unified formula for both directions:
    dw/dt = λ · (w_target - w)

Discrete form:
    w(t+dt) = w_target + (w_current - w_target) · e^(-λ·dt)

Learning:   w_target = w_max  → weight rises toward maximum
Forgetting: w_target = w_min  → weight decays toward minimum (never zero)

Same curve, two directions:
    Learning:   w_min → w_max (capacitor charging)
    Forgetting: w_current → w_min (capacitor discharging)

"Knowledge is never lost, only harder to access."
"""

from dataclasses import dataclass
from typing import Tuple
from math import exp


@dataclass
class WeightConfig:
    """Configuration for weight dynamics."""
    w_min: float = 0.1       # Floor - never fully forgotten
    w_max: float = 1.0       # Ceiling - fully learned (normalized scale)
    lambda_learn: float = 0.3    # Learning rate (per reinforcement)
    lambda_forget: float = 0.05  # Forgetting rate (per day) - slower than learning


class WeightDynamics:
    """
    Unified weight dynamics for learning and forgetting.

    The same exponential approach formula used in both directions:
    - Learning: weight approaches w_max
    - Forgetting: weight decays toward w_min

    Key insight: Learning is fast, forgetting is slow.
    This matches human memory - easy to learn, slow to fully forget.
    """

    def __init__(self, config: WeightConfig = None):
        self.config = config or WeightConfig()

    def _compute(self, w_current: float, w_target: float,
                 lambda_rate: float, dt: float) -> float:
        """
        Core formula: w(t+dt) = w_target + (w_current - w_target) · e^(-λ·dt)

        Args:
            w_current: Current weight value
            w_target: Target weight (w_max for learning, w_min for forgetting)
            lambda_rate: Rate constant
            dt: Time delta (reinforcement count for learning, days for forgetting)

        Returns:
            New weight value
        """
        decay_factor = exp(-lambda_rate * dt)
        return w_target + (w_current - w_target) * decay_factor

    def learn(self, w_current: float, reinforcements: int = 1) -> float:
        """
        Apply learning: move weight toward w_max.

        Args:
            w_current: Current weight
            reinforcements: Number of reinforcement events (default 1)

        Returns:
            New weight after learning
        """
        return self._compute(
            w_current=w_current,
            w_target=self.config.w_max,
            lambda_rate=self.config.lambda_learn,
            dt=reinforcements
        )

    def forget(self, w_current: float, days_elapsed: float) -> float:
        """
        Apply forgetting: decay weight toward w_min.

        Args:
            w_current: Current weight
            days_elapsed: Days since last reinforcement

        Returns:
            New weight after decay (never below w_min)
        """
        new_weight = self._compute(
            w_current=w_current,
            w_target=self.config.w_min,
            lambda_rate=self.config.lambda_forget,
            dt=days_elapsed
        )
        # Ensure we never go below minimum
        return max(new_weight, self.config.w_min)

    def compute_delta(self, w_current: float, w_target: float,
                      lambda_rate: float, dt: float) -> float:
        """
        Compute the change in weight (for logging/reporting).

        Returns:
            Delta weight (positive for learning, negative for forgetting)
        """
        new_weight = self._compute(w_current, w_target, lambda_rate, dt)
        return new_weight - w_current

    def effective_weight(self, w_stored: float, days_since_update: float) -> float:
        """
        Compute effective weight considering time decay.

        This can be used for real-time queries without modifying stored weight.
        Actual weight update happens during Sleep.

        Args:
            w_stored: Stored weight in database
            days_since_update: Days since last update

        Returns:
            Effective weight for navigation decisions
        """
        return self.forget(w_stored, days_since_update)

    def is_dormant(self, w_current: float, threshold: float = None) -> bool:
        """
        Check if weight has decayed to dormant state.

        Dormant = exists but not actively used in navigation.

        Args:
            w_current: Current weight
            threshold: Dormancy threshold (default: 2 * w_min)

        Returns:
            True if weight is in dormant range
        """
        threshold = threshold or (2 * self.config.w_min)
        return w_current <= threshold

    def initial_weight(self, source: str) -> float:
        """
        Get initial weight for new word based on source.

        Args:
            source: "corpus" | "conversation" | "context"

        Returns:
            Initial weight value
        """
        if source == "corpus":
            return self.config.w_max  # Corpus words start strong
        elif source == "conversation":
            return self.config.w_min * 2  # Conversation words start low
        else:  # "context" or unknown
            return self.config.w_min  # Context-inferred words start at minimum

    def summary(self, w_current: float, days_elapsed: float = 0) -> dict:
        """
        Get summary of weight state for debugging/logging.
        """
        effective = self.effective_weight(w_current, days_elapsed) if days_elapsed > 0 else w_current
        return {
            'stored': w_current,
            'effective': effective,
            'days_elapsed': days_elapsed,
            'is_dormant': self.is_dormant(effective),
            'strength_pct': (effective - self.config.w_min) / (self.config.w_max - self.config.w_min) * 100
        }


# Convenience instance with default config
default_dynamics = WeightDynamics()


def learn(w_current: float, reinforcements: int = 1) -> float:
    """Convenience function using default dynamics."""
    return default_dynamics.learn(w_current, reinforcements)


def forget(w_current: float, days_elapsed: float) -> float:
    """Convenience function using default dynamics."""
    return default_dynamics.forget(w_current, days_elapsed)


def initial_weight(source: str) -> float:
    """Convenience function using default dynamics."""
    return default_dynamics.initial_weight(source)
