"""
Semantic Connections: Parallel & Series Architecture
=====================================================

This module implements electrical circuit analogies for semantic navigation:
- Series connections (sequential deepening)
- Parallel connections (simultaneous expansion)
- Impedance matching
- Resonance and oscillation
- Frequency-based filtering

Theory: If semantic space follows physics (PT1 dynamics, Boltzmann),
then concepts can be connected like electrical components.
"""

from .connection_types import (
    SeriesConnection,
    ParallelConnection,
    HybridConnection,
    ConnectionResult,
)
from .impedance import (
    SemanticImpedance,
    compute_impedance,
    reflection_coefficient,
    impedance_match_quality,
    QueryImpedance,
    compute_query_impedance,
    ImpedanceMatcher,
    VERB_TAU_TARGETS,
    VERB_J_DIRECTIONS,
)
from .oscillators import (
    SemanticOscillator,
    compute_resonance_frequency,
    compute_inductance,
    compute_capacitance,
    PT1Dynamics,
    RCCircuit,
    RLCCircuit,
    E,
    KT_NATURAL,
    TAU_SATURATION,
    MAX_BOND_RATIO,
)
from .filters import (
    SemanticFilter,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
)

__all__ = [
    # Connection types
    'SeriesConnection',
    'ParallelConnection',
    'HybridConnection',
    'ConnectionResult',
    # Impedance
    'SemanticImpedance',
    'compute_impedance',
    'reflection_coefficient',
    'impedance_match_quality',
    'QueryImpedance',
    'compute_query_impedance',
    'ImpedanceMatcher',
    'VERB_TAU_TARGETS',
    'VERB_J_DIRECTIONS',
    # Oscillators
    'SemanticOscillator',
    'compute_resonance_frequency',
    'compute_inductance',
    'compute_capacitance',
    'PT1Dynamics',
    'RCCircuit',
    'RLCCircuit',
    'E',
    'KT_NATURAL',
    'TAU_SATURATION',
    'MAX_BOND_RATIO',
    # Filters
    'SemanticFilter',
    'LowPassFilter',
    'HighPassFilter',
    'BandPassFilter',
]
