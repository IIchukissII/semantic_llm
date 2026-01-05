"""Semantic RC-Model: Quantum number dynamics for (n, θ, r).

State = (Q_n, Q_θ, Q_r)

Physics from THEORY.md:
    - Boltzmann for n (orbital): P ∝ exp(-|Δn|/kT)
    - Coherence for θ (phase): C = cos(Δθ)
    - Bond strength: S = cos(Δθ) × exp(-|Δn|/kT)

Constants:
    kT = e^(-1/5) ≈ 0.819 (semantic temperature)
    Σ = e^(1/5) ≈ 1.22 (energy budget)
    kT × Σ = 1 (thermodynamic unity)

Intensity modulation (from 3M+ bond corpus):
    - Adjectives have learned intensity Δr from bond statistics
    - r_final = r_noun × (1 + intensity_scale × Δr_adj)
    - Δr > 0: amplifiers ("luckier", "cherish", "uncounted")
    - Δr < 0: dampeners ("cyclonic", "cataleptic", "foetid")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math
import json
from pathlib import Path

from .coord_loader import CoordLoader, get_loader
from .bond_extractor import TextBonds, Bond


def load_intensity_dict() -> dict:
    """Load adjective intensity dictionary from file."""
    intensity_path = Path(__file__).parent.parent / "data" / "intensity_dict.json"
    if intensity_path.exists():
        with open(intensity_path) as f:
            return json.load(f)
    return {}


# Fundamental constants from semantic physics
KT = math.exp(-1/5)   # ≈ 0.819 (Boltzmann temperature)
SIGMA = math.exp(1/5)  # ≈ 1.22 (energy budget)
E = math.e             # ≈ 2.718 (the Veil)


@dataclass
class RCState:
    """State of the RC model at a point in time."""
    Q: np.ndarray  # [Q_n, Q_θ, Q_r] - quantum number charges (smoothed)
    bond_index: int
    sentence_index: int
    bond: Optional[Bond] = None
    input_coords: Optional[tuple[float, float, float]] = None  # Raw (n, θ, r) input

    @property
    def Q_n(self) -> float:
        """Orbital charge (abstraction level)."""
        return self.Q[0]

    @property
    def Q_theta(self) -> float:
        """Phase charge (semantic direction)."""
        return self.Q[1]

    @property
    def Q_r(self) -> float:
        """Magnitude charge (intensity)."""
        return self.Q[2]

    @property
    def tau(self) -> float:
        """Abstraction level τ = 1 + n/e."""
        return 1 + self.Q_n / E

    @property
    def A(self) -> float:
        """Affirmation = r × cos(θ)."""
        return self.Q_r * math.cos(self.Q_theta)

    @property
    def S(self) -> float:
        """Sacred = r × sin(θ)."""
        return self.Q_r * math.sin(self.Q_theta)

    @property
    def potential(self) -> float:
        """Semantic potential Φ = r (magnitude in (A,S) space)."""
        return self.Q_r

    # Legacy compatibility
    @property
    def Q_A(self) -> float:
        return self.A

    @property
    def Q_S(self) -> float:
        return self.S

    @property
    def Q_tau(self) -> float:
        return self.tau


@dataclass
class Trajectory:
    """Full trajectory of RC model through text."""
    states: list[RCState] = field(default_factory=list)
    sentence_boundaries: list[int] = field(default_factory=list)
    skipped_words: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.states)

    @property
    def Q_array(self) -> np.ndarray:
        """All Q values as (N, 3) array: [Q_n, Q_θ, Q_r]."""
        if not self.states:
            return np.array([]).reshape(0, 3)
        return np.array([s.Q for s in self.states])

    @property
    def Q_n(self) -> np.ndarray:
        """Orbital (n) trajectory."""
        return self.Q_array[:, 0]

    @property
    def Q_theta(self) -> np.ndarray:
        """Phase (θ) trajectory."""
        return self.Q_array[:, 1]

    @property
    def Q_r(self) -> np.ndarray:
        """Magnitude (r) trajectory."""
        return self.Q_array[:, 2]

    @property
    def tau(self) -> np.ndarray:
        """Abstraction level τ = 1 + n/e."""
        return 1 + self.Q_n / E

    @property
    def A(self) -> np.ndarray:
        """Affirmation = r × cos(θ)."""
        return self.Q_r * np.cos(self.Q_theta)

    @property
    def S(self) -> np.ndarray:
        """Sacred = r × sin(θ)."""
        return self.Q_r * np.sin(self.Q_theta)

    @property
    def potential(self) -> np.ndarray:
        """Semantic potential Φ(t) = r (magnitude)."""
        return self.Q_r

    @property
    def coherence(self) -> np.ndarray:
        """Local coherence C = cos(Δθ) between consecutive states."""
        if len(self.states) < 2:
            return np.array([1.0])
        theta = self.Q_theta
        delta_theta = np.diff(theta)
        return np.concatenate([[1.0], np.cos(delta_theta)])

    @property
    def bond_strength(self) -> np.ndarray:
        """Bond strength S = cos(Δθ) × exp(-|Δn|/kT) between consecutive states."""
        if len(self.states) < 2:
            return np.array([1.0])
        n = self.Q_n
        theta = self.Q_theta
        delta_n = np.abs(np.diff(n))
        delta_theta = np.diff(theta)
        boltz = np.exp(-delta_n / KT)
        coh = np.cos(delta_theta)
        return np.concatenate([[1.0], coh * boltz])

    # Legacy compatibility (map to Cartesian)
    @property
    def Q_A(self) -> np.ndarray:
        return self.A

    @property
    def Q_S(self) -> np.ndarray:
        return self.S

    @property
    def Q_tau(self) -> np.ndarray:
        return self.tau


class SemanticRC:
    """Semantic RC model with quantum number dynamics.

    Implements physics from THEORY.md:
        - State: (Q_n, Q_θ, Q_r) - quantum number charges
        - Boltzmann for n: transition cost exp(-|Δn|/kT)
        - Phase dynamics for θ: coherence C = cos(Δθ)
        - Intensity for r: magnitude in (A,S) space

    RC dynamics with semantic physics:
        dQ_n/dt = (n_target - Q_n) × boltzmann_weight - Q_n × decay
        dQ_θ/dt = angular_diff(θ_target, Q_θ) × coherence - Q_θ × decay
        dQ_r/dt = (r_target - Q_r) × intensity_factor - Q_r × decay
    """

    def __init__(
        self,
        kT: float = KT,
        n_max: float = 5.0,
        decay: float = 0.05,
        dt: float = 0.5,
        coord_loader: Optional[CoordLoader] = None,
        intensity_scale: float = 0.2,
        intensity_dict: Optional[dict] = None,
    ):
        """Initialize RC model with semantic physics.

        Args:
            kT: Boltzmann temperature (default: e^(-1/5) ≈ 0.819)
            n_max: Maximum orbital level for normalization
            decay: Forgetting rate per time step
            dt: Time step for dynamics (larger = faster response)
            coord_loader: Coordinate loader (uses default if None)
            intensity_scale: How much adjective intensity affects r (0=none, 1=full)
            intensity_dict: Adjective intensity dictionary (loads default if None)
        """
        self.kT = kT
        self.n_max = n_max
        self.decay = decay
        self.dt = dt
        self.loader = coord_loader or get_loader()
        self.intensity_scale = intensity_scale
        self.intensity_dict = intensity_dict if intensity_dict is not None else load_intensity_dict()

        # Current state: [Q_n, Q_θ, Q_r]
        self.Q = np.zeros(3)

    def reset(self):
        """Reset state to zero."""
        self.Q = np.zeros(3)

    def _angular_diff(self, theta1: float, theta2: float) -> float:
        """Compute signed angular difference (handles wrap-around)."""
        diff = theta1 - theta2
        # Wrap to [-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def update(self, n: float, theta: float, r: float) -> np.ndarray:
        """Update Q given target quantum numbers.

        Args:
            n: Target orbital level
            theta: Target phase angle
            r: Target magnitude

        Returns:
            Updated Q state [Q_n, Q_θ, Q_r]
        """
        # Current state
        Q_n, Q_theta, Q_r = self.Q

        # 1. ORBITAL (n): Boltzmann-weighted update
        #    Larger Δn → smaller update (hard to jump levels)
        delta_n = n - Q_n
        boltzmann_weight = math.exp(-abs(delta_n) / self.kT)
        dQ_n = delta_n * boltzmann_weight * self.dt

        # 2. PHASE (θ): Coherence-weighted update
        #    Update is weighted by how coherent the transition is
        delta_theta = self._angular_diff(theta, Q_theta)
        coherence = math.cos(delta_theta)  # C = cos(Δθ) from theory
        # Use absolute coherence for update strength, signed delta for direction
        dQ_theta = delta_theta * abs(coherence) * self.dt

        # 3. MAGNITUDE (r): Direct RC dynamics
        delta_r = r - Q_r
        dQ_r = delta_r * self.dt

        # Apply decay (forgetting)
        decay_factor = self.decay * self.dt

        # Update state
        self.Q[0] = Q_n + dQ_n - Q_n * decay_factor
        self.Q[1] = Q_theta + dQ_theta - Q_theta * decay_factor
        self.Q[2] = Q_r + dQ_r - Q_r * decay_factor

        # Keep theta in [-π, π]
        while self.Q[1] > math.pi:
            self.Q[1] -= 2 * math.pi
        while self.Q[1] < -math.pi:
            self.Q[1] += 2 * math.pi

        return self.Q.copy()

    def bond_strength(self, n: float, theta: float) -> float:
        """Compute bond strength from current state to target.

        S = cos(Δθ) × exp(-|Δn|/kT)

        From THEORY.md Section 2.8
        """
        delta_n = abs(n - self.Q[0])
        delta_theta = self._angular_diff(theta, self.Q[1])
        return math.cos(delta_theta) * math.exp(-delta_n / self.kT)

    def process_bond(self, bond: Bond) -> Optional[tuple[np.ndarray, tuple[float, float, float]]]:
        """Process a single bond and return updated Q plus raw input.

        Args:
            bond: Bond to process

        Returns:
            (Updated Q, input_coords) or None if word not in dictionary
        """
        # Get noun coordinates
        coord = self.loader.get(bond.noun)
        if coord is None:
            return None

        n, theta, r = coord.n, coord.theta, coord.r

        # If adjective exists, combine using bond rules
        if bond.adj:
            adj_coord = self.loader.get(bond.adj)
            if adj_coord:
                # Bond combines noun and adjective
                # Phase: weighted toward resonance
                # Magnitude: combined intensity
                adj_n, adj_theta, adj_r = adj_coord.n, adj_coord.theta, adj_coord.r

                # Compute bond strength for combination weighting
                delta_theta = self._angular_diff(theta, adj_theta)
                resonance = (1 + math.cos(delta_theta)) / 2  # 0 to 1

                # Combined coordinates (weighted by resonance)
                n = (n + adj_n * resonance) / (1 + resonance)
                # For theta: move toward midpoint, weighted by resonance
                theta = theta + delta_theta * resonance * 0.5
                # Magnitude: boost for resonant pairs
                r = (r + adj_r) / 2 * (1 + 0.2 * resonance)

            # Apply learned intensity modulation from bond statistics
            adj_lower = bond.adj.lower()
            if adj_lower in self.intensity_dict:
                delta_r = self.intensity_dict[adj_lower]['delta_r']
                # r_final = r × (1 + scale × Δr)
                # Δr > 0: amplifiers increase r
                # Δr < 0: dampeners decrease r
                r = r * (1 + self.intensity_scale * delta_r)
                # Clamp r to reasonable bounds
                r = max(0.01, min(r, 3.0))

        # Store raw input before RC smoothing
        input_coords = (n, theta, r)
        Q = self.update(n, theta, r)
        return (Q, input_coords)

    def process_text(self, text_bonds: TextBonds) -> Trajectory:
        """Process text and return trajectory.

        Args:
            text_bonds: Extracted bonds with sentence structure

        Returns:
            Trajectory with all states and sentence boundaries
        """
        self.reset()
        trajectory = Trajectory()

        bond_index = 0
        for sent_idx, sentence in enumerate(text_bonds.sentences):
            for bond in sentence.bonds:
                result = self.process_bond(bond)

                if result is not None:
                    Q, input_coords = result
                    trajectory.states.append(RCState(
                        Q=Q,
                        bond_index=bond_index,
                        sentence_index=sent_idx,
                        bond=bond,
                        input_coords=input_coords,
                    ))
                    bond_index += 1
                else:
                    trajectory.skipped_words.append(bond.noun)

            # Mark sentence boundary
            if bond_index > 0:
                trajectory.sentence_boundaries.append(bond_index)

        return trajectory

    def transition_probability(
        self,
        target: np.ndarray,
        sigma: float = 1.0,
    ) -> float:
        """Compute transition probability to target.

        P(target | Q) ∝ exp(-|Δτ|/kT) × exp(-|ΔA|²/σ²) × exp(-|ΔS|²/σ²)

        Args:
            target: Target (A, S, τ) coordinates
            sigma: Width parameter for A/S Gaussian

        Returns:
            Unnormalized transition probability
        """
        delta = target - self.Q

        # Boltzmann factor for τ
        p_tau = np.exp(-np.abs(delta[2]) / self.kT)

        # Gaussian factors for A and S
        p_A = np.exp(-delta[0]**2 / sigma**2)
        p_S = np.exp(-delta[1]**2 / sigma**2)

        return p_tau * p_A * p_S


def process_text(text: str, **kwargs) -> Trajectory:
    """Convenience function: text → trajectory.

    Args:
        text: Input text
        **kwargs: Arguments for SemanticRC

    Returns:
        Trajectory through semantic space
    """
    from .bond_extractor import extract_bonds

    bonds = extract_bonds(text)
    rc = SemanticRC(**kwargs)
    return rc.process_text(bonds)
