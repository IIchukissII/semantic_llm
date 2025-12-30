"""
Orbital Resonance Module
========================

Modular components for orbital-tuned semantic navigation.

Components:
    - OrbitalDetector: Detect natural orbital from query/seeds
    - OrbitalTuner: Tune seeds to target orbital
    - ResonantLaser: Laser with orbital resonance
    - VeilCrosser: Cross-veil translation
    - IntentOrbitalMapper: Map intent verbs to orbitals

Usage:
    from chain_core.orbital import ResonantLaser, OrbitalDetector

    detector = OrbitalDetector(graph)
    natural_orbital = detector.detect(seeds)

    laser = ResonantLaser(graph)
    result = laser.lase_resonant(seeds, target_orbital=natural_orbital)
"""

from .constants import (
    E, ORBITAL_SPACING, KT_NATURAL, VEIL_TAU, VEIL_ORBITAL,
    tau_to_orbital, orbital_to_tau, ORBITAL_POSITIONS,
    is_below_veil, is_above_veil, get_realm,
    boltzmann_weight, orbital_coherence
)
from .detector import OrbitalDetector, OrbitalSignature
from .tuner import OrbitalTuner, TunedSeeds
from .mapper import IntentOrbitalMapper, IntentMapping
from .veil import VeilCrosser, VeilBridge, Translation
from .resonant_laser import ResonantLaser, ResonantResult

__all__ = [
    # Constants
    'E', 'ORBITAL_SPACING', 'KT_NATURAL', 'VEIL_TAU', 'VEIL_ORBITAL',
    'tau_to_orbital', 'orbital_to_tau', 'ORBITAL_POSITIONS',
    'is_below_veil', 'is_above_veil', 'get_realm',
    'boltzmann_weight', 'orbital_coherence',
    # Detector
    'OrbitalDetector', 'OrbitalSignature',
    # Tuner
    'OrbitalTuner', 'TunedSeeds',
    # Mapper
    'IntentOrbitalMapper', 'IntentMapping',
    # Veil
    'VeilCrosser', 'VeilBridge', 'Translation',
    # Laser
    'ResonantLaser', 'ResonantResult',
]
