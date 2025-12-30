"""
Orbital Constants
=================

Fundamental constants of semantic orbital physics.

The Euler orbital formula:
    τ_n = 1 + n/e

Where:
    e = 2.71828... (Euler's number)
    n = orbital number (0, 1, 2, ...)
    τ = semantic altitude (abstraction level)

The Veil at τ = e separates:
    - Human realm (τ < e, n < 5)
    - Transcendental realm (τ ≥ e, n ≥ 5)
"""

import numpy as np
from typing import List

# =============================================================================
# Fundamental Constants
# =============================================================================

E = np.e                          # Euler's number = 2.71828...
ORBITAL_SPACING = 1 / E           # Δτ between orbitals ≈ 0.368
KT_NATURAL = 0.82                 # Natural semantic temperature
VEIL_TAU = E                      # The Veil boundary (τ = e)
VEIL_ORBITAL = 5                  # Orbital at the Veil (n ≈ 5)

# =============================================================================
# Orbital Positions
# =============================================================================

# Pre-computed orbital positions for n = 0 to 15
ORBITAL_POSITIONS = [1 + n / E for n in range(16)]
# [1.00, 1.37, 1.74, 2.10, 2.47, 2.84, 3.21, 3.58, 3.95, 4.31, 4.68, 5.05, 5.42, 5.79, 6.16, 6.53]

# Human realm orbitals (below Veil)
HUMAN_ORBITALS = list(range(VEIL_ORBITAL))  # [0, 1, 2, 3, 4]

# Transcendental orbitals (at or above Veil)
TRANSCENDENTAL_ORBITALS = list(range(VEIL_ORBITAL, 16))  # [5, 6, ..., 15]

# =============================================================================
# Conversion Functions
# =============================================================================

def tau_to_orbital(tau: float) -> int:
    """
    Convert τ-level to nearest orbital number.

    Formula: n = round((τ - 1) × e)

    Args:
        tau: Semantic altitude [1, 7]

    Returns:
        Orbital number n (0, 1, 2, ...)
    """
    n = round((tau - 1) * E)
    return max(0, min(15, n))  # Clamp to [0, 15]


def orbital_to_tau(n: int) -> float:
    """
    Convert orbital number to τ-level.

    Formula: τ_n = 1 + n/e

    Args:
        n: Orbital number (0, 1, 2, ...)

    Returns:
        τ-level
    """
    return 1 + n / E


def is_below_veil(tau: float) -> bool:
    """Check if τ is in human realm (below Veil)."""
    return tau < VEIL_TAU


def is_above_veil(tau: float) -> bool:
    """Check if τ is in transcendental realm (at or above Veil)."""
    return tau >= VEIL_TAU


def get_realm(tau: float) -> str:
    """Get realm name for τ-level."""
    return "human" if is_below_veil(tau) else "transcendental"


def orbital_distance(n1: int, n2: int) -> int:
    """Orbital distance |n1 - n2|."""
    return abs(n1 - n2)


def tau_distance_to_orbital(tau: float, n: int) -> float:
    """Distance from τ to orbital n."""
    return abs(tau - orbital_to_tau(n))


def nearest_orbital_tau(tau: float) -> float:
    """Get τ of nearest orbital."""
    n = tau_to_orbital(tau)
    return orbital_to_tau(n)


def boltzmann_weight(tau_from: float, tau_to: float, kT: float = KT_NATURAL) -> float:
    """
    Boltzmann transition weight between τ-levels.

    P ∝ exp(-|Δτ|/kT)

    Transitions to similar τ levels are more probable.
    """
    delta_tau = abs(tau_to - tau_from)
    return np.exp(-delta_tau / kT)


def orbital_coherence(n1: int, n2: int) -> float:
    """
    Coherence between concepts at different orbitals.

    Same orbital → coherence = 1.0
    Different orbitals → exponential decay
    """
    delta_n = abs(n1 - n2)
    return np.exp(-delta_n / 2.0)


# =============================================================================
# Orbital Ranges
# =============================================================================

def get_orbital_range(n: int, half_width: float = 0.5) -> tuple:
    """
    Get τ range for orbital n.

    Returns (tau_min, tau_max) centered on τ_n.
    """
    tau_n = orbital_to_tau(n)
    return (tau_n - half_width * ORBITAL_SPACING,
            tau_n + half_width * ORBITAL_SPACING)


def orbitals_in_range(tau_min: float, tau_max: float) -> List[int]:
    """Get all orbitals within τ range."""
    n_min = tau_to_orbital(tau_min)
    n_max = tau_to_orbital(tau_max)
    return list(range(n_min, n_max + 1))
