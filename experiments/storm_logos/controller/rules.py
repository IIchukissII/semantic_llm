"""Adaptation Rules: How parameters respond to errors."""

from typing import Dict, List
from dataclasses import dataclass

from .parameters import AdaptationRule, ADAPTIVE_PARAMETERS


@dataclass
class RuleSet:
    """A collection of adaptation rules for a context."""
    name: str
    rules: Dict[str, AdaptationRule]

    def get(self, param: str) -> AdaptationRule:
        return self.rules.get(param)


# Therapeutic context: prioritize irony and grounding
THERAPEUTIC_RULES = RuleSet(
    name='therapeutic',
    rules={
        **ADAPTIVE_PARAMETERS,
        'mirror_depth': AdaptationRule(
            parameter='mirror_depth',
            target_metric='irony',
            target_value=0.15,
            direction=-1,
            eta=0.15,   # Faster adaptation
            kappa=0.04,
            range=(0.0, 1.0),
        ),
        'gravity_strength': AdaptationRule(
            parameter='gravity_strength',
            target_metric='tau_slope',
            target_value=-0.15,  # Stronger grounding
            direction=+1,
            eta=0.15,
            kappa=0.03,
            range=(0.0, 1.0),
        ),
    }
)

# Generation context: prioritize coherence and tension
GENERATION_RULES = RuleSet(
    name='generation',
    rules={
        **ADAPTIVE_PARAMETERS,
        'storm_radius': AdaptationRule(
            parameter='storm_radius',
            target_metric='coherence',
            target_value=0.75,  # Higher coherence target
            direction=+1,
            eta=0.12,
            kappa=0.03,
            range=(0.4, 1.5),
        ),
        'dialectic_tension': AdaptationRule(
            parameter='dialectic_tension',
            target_metric='tension',
            target_value=0.65,  # Higher tension
            direction=+1,
            eta=0.18,
            kappa=0.04,
            range=(0.0, 1.0),
        ),
    }
)


def get_rules(context: str = 'default') -> Dict[str, AdaptationRule]:
    """Get adaptation rules for context.

    Args:
        context: 'default', 'therapeutic', 'generation'

    Returns:
        Dictionary of rules
    """
    if context == 'therapeutic':
        return THERAPEUTIC_RULES.rules
    elif context == 'generation':
        return GENERATION_RULES.rules
    else:
        return ADAPTIVE_PARAMETERS
