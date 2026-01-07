"""Homeostatic Targets: The equilibrium the system seeks.

These are the target values for each metric.
Deviations trigger adaptation.
"""

from dataclasses import dataclass
from typing import Dict


# Default homeostatic targets
HOMEOSTATIC_TARGETS = {
    'coherence': 0.70,      # Trajectory coherence
    'irony': 0.15,          # Irony level (low = open)
    'tension': 0.60,        # Dialectical tension
    'tau_variance': 0.80,   # τ breathing
    'noise_ratio': 0.20,    # Noise in trajectory
    'tau_slope': -0.10,     # Slight grounding trend
}


@dataclass
class TargetSet:
    """A set of homeostatic targets.

    Different contexts may have different targets.
    """
    name: str
    targets: Dict[str, float]

    def get(self, metric: str, default: float = 0.5) -> float:
        return self.targets.get(metric, default)


# Predefined target sets
THERAPEUTIC_TARGETS = TargetSet(
    name='therapeutic',
    targets={
        'coherence': 0.70,
        'irony': 0.15,      # Low irony = patient open
        'tension': 0.55,    # Moderate tension
        'tau_variance': 0.70,
        'noise_ratio': 0.25,
        'tau_slope': -0.10,
    }
)

GENERATION_TARGETS = TargetSet(
    name='generation',
    targets={
        'coherence': 0.75,
        'irony': 0.20,
        'tension': 0.65,    # Higher tension for drama
        'tau_variance': 0.85,
        'noise_ratio': 0.15,
        'tau_slope': -0.05,
    }
)

BALANCED_TARGETS = TargetSet(
    name='balanced',
    targets={
        'coherence': 0.70,
        'irony': 0.15,
        'tension': 0.50,
        'tau_variance': 0.70,
        'noise_ratio': 0.20,
        'tau_slope': 0.00,
    }
)


def get_targets(name: str = 'default') -> Dict[str, float]:
    """Get target set by name.

    Args:
        name: 'default', 'therapeutic', 'generation', 'balanced'

    Returns:
        Dictionary of targets
    """
    if name == 'therapeutic':
        return THERAPEUTIC_TARGETS.targets
    elif name == 'generation':
        return GENERATION_TARGETS.targets
    elif name == 'balanced':
        return BALANCED_TARGETS.targets
    else:
        return HOMEOSTATIC_TARGETS
