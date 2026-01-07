"""Semantic Physics: Gravity, RC dynamics, Boltzmann.

The physics of semantic space:
- Gravity: φ = λτ - μA pulls toward concrete (+τ) and good (+A)
- RC dynamics: state accumulates like capacitor charge
- Boltzmann: transition probabilities based on energy differences
"""

import math
from typing import Tuple
import numpy as np

from ..config import get_config, KT, LAMBDA, MU, DECAY, Q_MAX, DT
from ..data.models import SemanticState, Bond


# ============================================================================
# GRAVITY
# ============================================================================

def gravity_potential(state: SemanticState) -> float:
    """Compute gravity potential φ = λτ - μA.

    Lower potential = more stable (concrete, good).

    Args:
        state: Current semantic state

    Returns:
        Gravity potential φ
    """
    return LAMBDA * state.tau - MU * state.A


def gravity_force(state: SemanticState = None) -> Tuple[float, float, float]:
    """Compute gravity force vector (dA/dt, dS/dt, dτ/dt).

    Gravity pulls toward:
    - Higher A (more positive/affirming)
    - Lower τ (more concrete/grounded)
    - S is free (meaning can be found anywhere)

    Returns:
        (force_A, force_S, force_tau)
    """
    return (
        MU,       # Force toward +A (good)
        0.0,      # No force on S
        -LAMBDA,  # Force toward -τ (concrete)
    )


def resistance(velocity: Tuple[float, float, float],
               state: SemanticState) -> float:
    """Compute resistance = motion against therapeutic gravity.

    High resistance = patient fighting the natural flow toward health.

    Args:
        velocity: (dA, dS, dτ) velocity vector
        state: Current state

    Returns:
        Resistance value (0 to 1)
    """
    gravity = gravity_force(state)

    # Resistance = negative dot product with gravity
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


# ============================================================================
# RC DYNAMICS
# ============================================================================

def rc_update(current: SemanticState, measured: SemanticState,
              dt: float = DT, decay: float = DECAY) -> SemanticState:
    """RC dynamics update: blend current position with new measurement.

    Like capacitor charging:
        Q(t+dt) = Q(t) + dt*(Q_new - Q(t)) - decay*Q(t)

    Args:
        current: Current state Q(t)
        measured: New measurement
        dt: Time step
        decay: Decay rate

    Returns:
        Updated state Q(t+dt)
    """
    return SemanticState(
        A=current.A + dt * (measured.A - current.A) - decay * current.A,
        S=current.S + dt * (measured.S - current.S) - decay * current.S,
        tau=current.tau + dt * (measured.tau - current.tau),
        irony=current.irony + dt * (measured.irony - current.irony),
        sarcasm=current.sarcasm + dt * (measured.sarcasm - current.sarcasm),
        emotion=measured.emotion,
        intensity=measured.intensity,
    )


def rc_update_exact(Q: np.ndarray, target: np.ndarray,
                    dt: float = DT, decay: float = DECAY,
                    Q_max: float = Q_MAX) -> np.ndarray:
    """Exact RC update from THEORY with saturation.

    dQ_x/dt = (x_w - Q_x) × (1 - |Q_x|/Q_max) - Q_x × decay

    Three components:
        (x_w - Q_x)         — attraction to input
        (1 - |Q_x|/Q_max)   — saturation
        Q_x × decay         — forgetting

    Args:
        Q: Current state [Q_A, Q_S, Q_τ]
        target: Target [A, S, τ]
        dt: Time step
        decay: Forgetting rate
        Q_max: Saturation limit

    Returns:
        Updated Q
    """
    Q_new = Q.copy()

    for i in range(3):
        x_w = target[i]
        Q_x = Q[i]

        # Attraction to input
        attraction = x_w - Q_x

        # Saturation factor (slows down near Q_max)
        saturation = 1 - abs(Q_x) / Q_max
        saturation = max(0, saturation)

        # Forgetting
        forgetting = Q_x * decay

        # Update
        dQ = attraction * saturation * dt - forgetting * dt
        Q_new[i] = Q_x + dQ

    return Q_new


# ============================================================================
# BOLTZMANN DISTRIBUTION
# ============================================================================

def boltzmann_factor(delta_tau: float, kT: float = KT) -> float:
    """Compute Boltzmann factor for τ transition.

    exp(-|Δτ|/kT)

    Args:
        delta_tau: Change in abstraction level
        kT: Boltzmann temperature

    Returns:
        Boltzmann factor (0 to 1)
    """
    return math.exp(-abs(delta_tau) / kT)


def gaussian_factor(delta: float, sigma: float = 0.5) -> float:
    """Compute Gaussian factor for A or S transition.

    exp(-Δ²/σ²)

    Args:
        delta: Change in coordinate
        sigma: Width of Gaussian

    Returns:
        Gaussian factor (0 to 1)
    """
    return math.exp(-delta**2 / sigma**2)


def transition_probability(Q: SemanticState, target: Bond,
                           kT: float = KT, sigma: float = 0.5) -> float:
    """Compute transition probability from Q to target bond.

    P(next | Q) ∝ exp(-|Δτ|/kT) × exp(-|ΔA|²/σ²) × exp(-|ΔS|²/σ²)

    Args:
        Q: Current state
        target: Target bond
        kT: Boltzmann temperature
        sigma: Width for A/S Gaussians

    Returns:
        Unnormalized probability
    """
    # Boltzmann for τ
    delta_tau = abs(target.tau - Q.tau)
    p_tau = boltzmann_factor(delta_tau, kT)

    # Gaussian for A
    delta_A = target.A - Q.A
    p_A = gaussian_factor(delta_A, sigma)

    # Gaussian for S
    delta_S = target.S - Q.S
    p_S = gaussian_factor(delta_S, sigma)

    return p_tau * p_A * p_S


# ============================================================================
# ZIPF DISTRIBUTION
# ============================================================================

def zipf_factor(variety: int, tau: float,
                alpha_0: float = 2.5, alpha_1: float = -1.4) -> float:
    """Compute Zipf factor for bond frequency.

    v^(-α(τ)) where α(τ) = α₀ + α₁×τ

    Args:
        variety: Bond frequency
        tau: Current abstraction level
        alpha_0: Baseline Zipf exponent
        alpha_1: τ-dependent Zipf term

    Returns:
        Zipf factor
    """
    if variety <= 0:
        return 0.0

    alpha = alpha_0 + alpha_1 * tau
    return variety ** (-alpha)


# ============================================================================
# COHERENCE
# ============================================================================

def coherence(state: SemanticState, bond: Bond) -> float:
    """Compute coherence between state and bond.

    Uses cosine similarity in A-S plane.

    Args:
        state: Current state
        bond: Candidate bond

    Returns:
        Coherence value (-1 to 1)
    """
    # State vector (A, S)
    state_vec = np.array([state.A, state.S])
    state_mag = np.linalg.norm(state_vec)

    # Bond vector (A, S)
    bond_vec = np.array([bond.A, bond.S])
    bond_mag = np.linalg.norm(bond_vec)

    if state_mag < 0.01 or bond_mag < 0.01:
        return 0.0

    # Cosine similarity
    return np.dot(state_vec, bond_vec) / (state_mag * bond_mag)


def trajectory_coherence(bonds: list, window: int = 5) -> float:
    """Compute coherence of a trajectory.

    Average pairwise coherence over sliding window.

    Args:
        bonds: List of bonds
        window: Window size

    Returns:
        Mean coherence
    """
    if len(bonds) < 2:
        return 1.0

    coherences = []
    for i in range(1, len(bonds)):
        # Create state from previous bond
        prev = bonds[i-1]
        state = SemanticState(A=prev.A, S=prev.S, tau=prev.tau)
        coh = coherence(state, bonds[i])
        coherences.append(coh)

    return np.mean(coherences) if coherences else 1.0


# ============================================================================
# THERAPEUTIC VECTOR
# ============================================================================

def therapeutic_vector(current: SemanticState,
                       target: SemanticState = None) -> Tuple[float, float, float]:
    """Compute direction toward health (unit vector).

    Args:
        current: Current state
        target: Health target (uses default if None)

    Returns:
        Unit vector (dA, dS, dτ) toward health
    """
    if target is None:
        config = get_config()
        target = SemanticState(
            A=config.health.A,
            S=config.health.S,
            tau=config.health.tau
        )

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


# ============================================================================
# MASTER EQUATION
# ============================================================================

def master_score(Q: SemanticState, bond: Bond,
                 kT: float = KT, sigma: float = 0.5,
                 gravity_weight: float = 0.5) -> float:
    """Master equation for bond scoring.

    P(bond | Q) ∝ exp(-|Δτ|/kT) × v^(-α(τ)) × exp(-φ/kT) × coherence

    Args:
        Q: Current state
        bond: Candidate bond
        kT: Boltzmann temperature
        sigma: Width for A/S Gaussians
        gravity_weight: Weight for gravity term

    Returns:
        Score (higher = better)
    """
    # Boltzmann factor for τ
    delta_tau = abs(bond.tau - Q.tau)
    boltz = boltzmann_factor(delta_tau, kT)

    # Zipf factor
    zipf = zipf_factor(bond.variety, Q.tau)

    # Gravity factor
    gravity_bond = SemanticState(A=bond.A, S=bond.S, tau=bond.tau)
    phi = gravity_potential(gravity_bond)
    gravity = math.exp(-phi * gravity_weight / kT)

    # Coherence
    coh = coherence(Q, bond)
    coh_factor = (1 + coh) / 2  # Map from [-1,1] to [0,1]

    return boltz * zipf * gravity * coh_factor
