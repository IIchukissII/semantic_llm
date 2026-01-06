"""Core: Data structures, physics, and data layer."""

from .models import SemanticState, ConversationTrajectory
from .physics import (
    KT, LAMBDA, MU, HEALTH, TARGETS,
    gravity_potential, gravity_force, resistance,
    rc_update, therapeutic_vector,
)
from .data import SemanticData, get_data, WordCoordinates, Bond

__all__ = [
    # Models
    'SemanticState',
    'ConversationTrajectory',

    # Physics
    'KT', 'LAMBDA', 'MU', 'HEALTH', 'TARGETS',
    'gravity_potential', 'gravity_force', 'resistance',
    'rc_update', 'therapeutic_vector',

    # Data
    'SemanticData', 'get_data', 'WordCoordinates', 'Bond',
]
