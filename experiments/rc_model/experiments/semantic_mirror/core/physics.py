"""Semantic Physics: Gravity, RC dynamics, constants.

The physics of semantic space:
- Gravity φ = λτ - μA pulls toward concrete (+τ) and good (+A)
- RC dynamics: state accumulates like capacitor charge
- kT = e^(-1/5) ≈ 0.819 sets fluctuation scale
"""

import math
from typing import Tuple

from .models import SemanticState


# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

# Thermal constant: sets scale of semantic fluctuations
KT = math.exp(-1/5)  # ≈ 0.819

# Gravity constants
LAMBDA = 0.5  # Pull toward concrete (lower τ)
MU = 0.5      # Pull toward good (higher A)

# RC dynamics
TAU_RC = 5  # RC time constant (turns)
DECAY = 0.05  # State decay rate


# ════════════════════════════════════════════════════════════════
# THERAPEUTIC TARGETS
# ════════════════════════════════════════════════════════════════

# Health: grounded, positive, meaningful
HEALTH = SemanticState(A=0.3, S=0.2, tau=2.0)

# Genre targets (for generation)
TARGETS = {
    'dramatic': SemanticState(A=0.4, S=0.3, tau=1.8),
    'ironic': SemanticState(A=0.2, S=-0.2, tau=2.5),
    'balanced': SemanticState(A=0.1, S=0.1, tau=2.8),
}


# ════════════════════════════════════════════════════════════════
# GRAVITY
# ════════════════════════════════════════════════════════════════

def gravity_potential(state: SemanticState) -> float:
    """Compute gravity potential φ = λτ - μA.

    Lower potential = more stable (concrete, good).
    """
    return LAMBDA * state.tau - MU * state.A


def gravity_force(state: SemanticState) -> Tuple[float, float, float]:
    """Compute gravity force vector (dA/dt, dS/dt, dτ/dt).

    Gravity pulls toward:
    - Higher A (more positive/affirming)
    - Lower τ (more concrete/grounded)
    - S is free (meaning can be found anywhere)
    """
    return (
        MU,       # Force toward +A (good)
        0.0,      # No force on S
        -LAMBDA,  # Force toward -τ (concrete)
    )


def resistance(velocity: Tuple[float, float, float], state: SemanticState) -> float:
    """Compute resistance = motion against therapeutic gravity.

    High resistance = patient fighting the natural flow toward health.
    """
    gravity = gravity_force(state)

    # Resistance = negative dot product with gravity
    # (if moving against gravity, resistance is positive)
    dot = (
        velocity[0] * gravity[0] +
        velocity[2] * gravity[2]  # S doesn't contribute
    )

    # Normalize by gravity magnitude
    gravity_mag = math.sqrt(gravity[0]**2 + gravity[2]**2)
    if gravity_mag < 0.01:
        return 0.0

    # Resistance is negative alignment (0 to 1 scale)
    alignment = dot / gravity_mag
    return max(0.0, min(1.0, -alignment))


# ════════════════════════════════════════════════════════════════
# RC DYNAMICS
# ════════════════════════════════════════════════════════════════

def rc_update(current: SemanticState, measured: SemanticState,
              dt: float = 0.3) -> SemanticState:
    """RC dynamics update: blend current position with new measurement.

    Like capacitor charging: Q(t+dt) = Q(t) + dt*(Q_new - Q(t)) - decay*Q(t)
    """
    return SemanticState(
        A=current.A + dt * (measured.A - current.A) - DECAY * current.A,
        S=current.S + dt * (measured.S - current.S) - DECAY * current.S,
        tau=current.tau + dt * (measured.tau - current.tau),
        irony=current.irony + dt * (measured.irony - current.irony),
        sarcasm=current.sarcasm + dt * (measured.sarcasm - current.sarcasm),
        emotion=measured.emotion,
        intensity=measured.intensity,
    )


def therapeutic_vector(current: SemanticState,
                       target: SemanticState = None) -> Tuple[float, float, float]:
    """Compute direction toward health (unit vector)."""
    target = target or HEALTH

    delta_A = target.A - current.A
    delta_S = target.S - current.S
    delta_tau = target.tau - current.tau

    magnitude = math.sqrt(delta_A**2 + delta_S**2 + delta_tau**2)
    if magnitude < 0.01:
        return (0.0, 0.0, 0.0)

    return (
        delta_A / magnitude,
        delta_S / magnitude,
        delta_tau / magnitude
    )
