"""Semantic RC-Model v2: Exact formulas from THEORY.

State = (Q_A, Q_S, Q_τ) - Three Capacitors

Dynamics:
    dQ_A/dt = (A_w - Q_A) × (1 - |Q_A|/Q_max) - Q_A × decay
    dQ_S/dt = (S_w - Q_S) × (1 - |Q_S|/Q_max) - Q_S × decay
    dQ_τ/dt = (τ_w - Q_τ) × (1 - |Q_τ|/Q_max) - Q_τ × decay

Transition Probability:
    P(next | S) ∝ exp(-|Δτ|/kT) × exp(-|ΔA|²/σ²) × exp(-|ΔS|²/σ²)

Constants:
    kT = e^(-1/5) ≈ 0.82
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math
import json
from pathlib import Path

from .coord_loader import CoordLoader, get_loader
from .bond_extractor import TextBonds, Bond


# Fundamental constants
KT = math.exp(-1/5)   # ≈ 0.819 (Boltzmann temperature)
Q_MAX = 2.0           # Saturation limit


def load_intensity_dict() -> dict:
    """Load adjective intensity dictionary."""
    intensity_path = Path(__file__).parent.parent / "data" / "intensity_dict.json"
    if intensity_path.exists():
        with open(intensity_path) as f:
            return json.load(f)
    return {}


@dataclass
class RCStateV2:
    """State of RC model v2."""
    Q: np.ndarray  # [Q_A, Q_S, Q_τ]
    bond_index: int
    sentence_index: int
    bond: Optional[Bond] = None
    input_coords: Optional[tuple[float, float, float]] = None  # (A, S, τ) input

    @property
    def Q_A(self) -> float:
        return self.Q[0]

    @property
    def Q_S(self) -> float:
        return self.Q[1]

    @property
    def Q_tau(self) -> float:
        return self.Q[2]


@dataclass
class TrajectoryV2:
    """Trajectory of RC model v2."""
    states: list[RCStateV2] = field(default_factory=list)
    sentence_boundaries: list[int] = field(default_factory=list)
    skipped_words: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.states)

    @property
    def Q_array(self) -> np.ndarray:
        """All Q values as (N, 3) array: [Q_A, Q_S, Q_τ]."""
        if not self.states:
            return np.array([]).reshape(0, 3)
        return np.array([s.Q for s in self.states])

    @property
    def Q_A(self) -> np.ndarray:
        return self.Q_array[:, 0]

    @property
    def Q_S(self) -> np.ndarray:
        return self.Q_array[:, 1]

    @property
    def Q_tau(self) -> np.ndarray:
        return self.Q_array[:, 2]


class SemanticRCv2:
    """Semantic RC model v2 with exact formulas.

    State: S(t) = (Q_A, Q_S, Q_τ)

    Dynamics (from THEORY):
        dQ_x/dt = (x_w - Q_x) × (1 - |Q_x|/Q_max) - Q_x × decay

    Three components:
        (x_w - Q_x)         — attraction to input
        (1 - |Q_x|/Q_max)   — saturation
        Q_x × decay         — forgetting
    """

    def __init__(
        self,
        kT: float = KT,
        Q_max: float = Q_MAX,
        decay: float = 0.05,
        dt: float = 0.5,
        sigma: float = 1.0,
        coord_loader: Optional[CoordLoader] = None,
        intensity_scale: float = 0.2,
        intensity_dict: Optional[dict] = None,
    ):
        """Initialize RC model v2.

        Args:
            kT: Boltzmann temperature (e^(-1/5) ≈ 0.82)
            Q_max: Saturation limit
            decay: Forgetting rate
            dt: Time step
            sigma: Width for A/S Gaussians
            coord_loader: Coordinate loader
            intensity_scale: Adjective intensity effect
            intensity_dict: Adjective intensities
        """
        self.kT = kT
        self.Q_max = Q_max
        self.decay = decay
        self.dt = dt
        self.sigma = sigma
        self.loader = coord_loader or get_loader()
        self.intensity_scale = intensity_scale
        self.intensity_dict = intensity_dict if intensity_dict is not None else load_intensity_dict()

        # State: [Q_A, Q_S, Q_τ]
        self.Q = np.zeros(3)

    def reset(self):
        """Reset state to zero."""
        self.Q = np.zeros(3)

    def update(self, A: float, S: float, tau: float) -> np.ndarray:
        """Update Q using exact THEORY formulas.

        dQ_x/dt = (x_w - Q_x) × (1 - |Q_x|/Q_max) - Q_x × decay

        Args:
            A: Target Affirmation
            S: Target Sacred
            tau: Target Abstraction

        Returns:
            Updated Q state [Q_A, Q_S, Q_τ]
        """
        Q_A, Q_S, Q_tau = self.Q
        targets = np.array([A, S, tau])

        for i in range(3):
            x_w = targets[i]
            Q_x = self.Q[i]

            # Attraction to input
            attraction = x_w - Q_x

            # Saturation factor (slows down near Q_max)
            saturation = 1 - abs(Q_x) / self.Q_max
            saturation = max(0, saturation)  # Clamp to non-negative

            # Forgetting
            forgetting = Q_x * self.decay

            # Update
            dQ = attraction * saturation * self.dt - forgetting * self.dt
            self.Q[i] = Q_x + dQ

        return self.Q.copy()

    def transition_probability(self, A: float, S: float, tau: float) -> float:
        """Compute transition probability to target.

        P(next | S) ∝ exp(-|Δτ|/kT) × exp(-|ΔA|²/σ²) × exp(-|ΔS|²/σ²)

        Args:
            A, S, tau: Target coordinates

        Returns:
            Unnormalized probability
        """
        Q_A, Q_S, Q_tau = self.Q

        # Boltzmann for τ
        delta_tau = abs(tau - Q_tau)
        p_tau = math.exp(-delta_tau / self.kT)

        # Gaussian for A
        delta_A = A - Q_A
        p_A = math.exp(-delta_A**2 / self.sigma**2)

        # Gaussian for S
        delta_S = S - Q_S
        p_S = math.exp(-delta_S**2 / self.sigma**2)

        return p_tau * p_A * p_S

    def process_bond(self, bond: Bond) -> Optional[tuple[np.ndarray, tuple[float, float, float]]]:
        """Process bond and return updated Q plus raw input.

        Args:
            bond: (noun, adj) bond

        Returns:
            (Q, input_coords) or None
        """
        coord = self.loader.get(bond.noun)
        if coord is None:
            return None

        A, S, tau = coord.A, coord.S, coord.tau

        # Apply adjective intensity if present
        if bond.adj and bond.adj.lower() in self.intensity_dict:
            delta_r = self.intensity_dict[bond.adj.lower()]['delta_r']
            # Scale r = sqrt(A² + S²) by intensity
            r = math.sqrt(A**2 + S**2)
            theta = math.atan2(S, A)
            r_new = r * (1 + self.intensity_scale * delta_r)
            r_new = max(0.01, min(r_new, 3.0))
            A = r_new * math.cos(theta)
            S = r_new * math.sin(theta)

        input_coords = (A, S, tau)
        Q = self.update(A, S, tau)
        return (Q, input_coords)

    def process_text(self, text_bonds: TextBonds) -> TrajectoryV2:
        """Process text and return trajectory.

        Args:
            text_bonds: Extracted bonds

        Returns:
            TrajectoryV2
        """
        self.reset()
        trajectory = TrajectoryV2()

        bond_index = 0
        for sent_idx, sentence in enumerate(text_bonds.sentences):
            for bond in sentence.bonds:
                result = self.process_bond(bond)

                if result is not None:
                    Q, input_coords = result
                    trajectory.states.append(RCStateV2(
                        Q=Q,
                        bond_index=bond_index,
                        sentence_index=sent_idx,
                        bond=bond,
                        input_coords=input_coords,
                    ))
                    bond_index += 1
                else:
                    trajectory.skipped_words.append(bond.noun)

            if bond_index > 0:
                trajectory.sentence_boundaries.append(bond_index)

        return trajectory


def compute_metrics_v2(trajectory: TrajectoryV2) -> dict:
    """Compute metrics for v2 trajectory."""
    Q = trajectory.Q_array
    if len(Q) < 2:
        return {'n_bonds': len(Q)}

    # Raw input transitions
    input_coords = []
    for state in trajectory.states:
        if state.input_coords:
            input_coords.append(state.input_coords)
    input_coords = np.array(input_coords)

    # Compute |ΔA|, |ΔS|, |Δτ| on raw inputs
    if len(input_coords) >= 2:
        delta_A = np.abs(np.diff(input_coords[:, 0]))
        delta_S = np.abs(np.diff(input_coords[:, 1]))
        delta_tau = np.abs(np.diff(input_coords[:, 2]))
    else:
        delta_A = delta_S = delta_tau = np.array([0])

    # Coherence metrics
    # exp(-|Δτ|/kT) averaged
    boltz = np.exp(-delta_tau / KT)
    mean_boltz = np.mean(boltz)

    # exp(-|ΔA|²/σ²) averaged (using σ=0.5)
    sigma = 0.5
    gauss_A = np.exp(-delta_A**2 / sigma**2)
    gauss_S = np.exp(-delta_S**2 / sigma**2)

    return {
        'n_bonds': len(Q),
        'mean_delta_tau': np.mean(delta_tau),
        'mean_delta_A': np.mean(delta_A),
        'mean_delta_S': np.mean(delta_S),
        'mean_boltz_tau': mean_boltz,
        'mean_gauss_A': np.mean(gauss_A),
        'mean_gauss_S': np.mean(gauss_S),
        'Q_A_mean': Q[:, 0].mean(),
        'Q_S_mean': Q[:, 1].mean(),
        'Q_tau_mean': Q[:, 2].mean(),
    }
