"""Semantic Layer: Storm, Dialectic, Chain Reaction.

Core semantic operations:
    - Storm: Explosion of candidates around current position
    - Dialectic: Thesis-antithesis filtering for tension
    - Chain: Resonance scoring and winner selection
    - Physics: Gravity, RC dynamics, Boltzmann
"""

from .physics import (
    gravity_potential,
    gravity_force,
    rc_update,
    boltzmann_factor,
    transition_probability,
)
from .state import StateManager
from .storm import Storm
from .dialectic import Dialectic
from .chain import ChainReaction
