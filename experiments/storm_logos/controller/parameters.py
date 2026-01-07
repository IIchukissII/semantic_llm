"""Parameter Definitions: Adaptive parameters with ranges and rules."""

from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class AdaptationRule:
    """Rule for adapting a single parameter.

    The parameter is adjusted based on the target metric's error.
    """
    parameter: str          # Parameter name (e.g., "storm_radius")
    target_metric: str      # Metric to optimize (e.g., "coherence")
    target_value: float     # Target value for metric
    direction: int          # +1 if param should increase with error, -1 otherwise
    eta: float              # Learning rate (P term coefficient)
    kappa: float            # Integral coefficient (I term)
    range: Tuple[float, float]  # Valid parameter range (min, max)

    def clamp(self, value: float) -> float:
        """Clamp value to valid range."""
        return max(self.range[0], min(self.range[1], value))


# Default adaptation rules
ADAPTIVE_PARAMETERS: Dict[str, AdaptationRule] = {
    'storm_radius': AdaptationRule(
        parameter='storm_radius',
        target_metric='coherence',
        target_value=0.70,
        direction=+1,   # Higher radius when coherence is low
        eta=0.10,
        kappa=0.02,
        range=(0.5, 2.0),
    ),

    'dialectic_tension': AdaptationRule(
        parameter='dialectic_tension',
        target_metric='tau_variance',
        target_value=0.80,
        direction=+1,   # Higher tension when variance is low
        eta=0.15,
        kappa=0.03,
        range=(0.0, 1.0),
    ),

    'chain_decay': AdaptationRule(
        parameter='chain_decay',
        target_metric='noise_ratio',
        target_value=0.20,
        direction=-1,   # Lower decay (forget faster) when noise is high
        eta=0.08,
        kappa=0.02,
        range=(0.5, 0.95),
    ),

    'gravity_strength': AdaptationRule(
        parameter='gravity_strength',
        target_metric='tau_slope',
        target_value=-0.10,
        direction=+1,   # Higher gravity when slope is too positive
        eta=0.12,
        kappa=0.02,
        range=(0.0, 1.0),
    ),

    'mirror_depth': AdaptationRule(
        parameter='mirror_depth',
        target_metric='irony',
        target_value=0.15,
        direction=-1,   # Lower depth when irony is high
        eta=0.10,
        kappa=0.03,
        range=(0.0, 1.0),
    ),

    'antithesis_weight': AdaptationRule(
        parameter='antithesis_weight',
        target_metric='tension',
        target_value=0.60,
        direction=+1,   # Higher weight when tension is low
        eta=0.12,
        kappa=0.02,
        range=(0.0, 1.0),
    ),

    'coherence_threshold': AdaptationRule(
        parameter='coherence_threshold',
        target_metric='coherence',
        target_value=0.70,
        direction=-1,   # Lower threshold when coherence is high
        eta=0.08,
        kappa=0.02,
        range=(0.1, 0.6),
    ),
}


def get_default_params() -> Dict[str, float]:
    """Get default parameter values (midpoint of ranges)."""
    params = {}
    for name, rule in ADAPTIVE_PARAMETERS.items():
        params[name] = (rule.range[0] + rule.range[1]) / 2
    return params


def get_initial_params() -> Dict[str, float]:
    """Get initial parameter values."""
    return {
        'storm_radius': 1.0,
        'dialectic_tension': 0.5,
        'chain_decay': 0.85,
        'gravity_strength': 0.5,
        'mirror_depth': 0.5,
        'antithesis_weight': 0.5,
        'coherence_threshold': 0.3,
    }
