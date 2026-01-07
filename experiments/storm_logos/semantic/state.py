"""State Manager: Q state tracking and updates.

Manages the semantic state Q = (Q_A, Q_S, Q_τ) as it evolves
through RC dynamics.
"""

from typing import List, Optional, Tuple
import numpy as np

from ..data.models import SemanticState, Bond, Trajectory
from ..config import get_config, KT, Q_MAX, DECAY, DT
from .physics import rc_update, rc_update_exact


class StateManager:
    """Manages semantic state Q through conversation/generation.

    State evolves via RC dynamics:
        dQ/dt = (input - Q) × (1 - |Q|/Q_max) - Q × decay

    Also tracks history for metrics computation.
    """

    def __init__(self,
                 kT: float = KT,
                 Q_max: float = Q_MAX,
                 decay: float = DECAY,
                 dt: float = DT,
                 history_size: int = 100):
        self.kT = kT
        self.Q_max = Q_max
        self.decay = decay
        self.dt = dt
        self.history_size = history_size

        # Current state
        self._Q = np.zeros(3)  # [Q_A, Q_S, Q_τ]

        # History
        self._history: List[np.ndarray] = []
        self._bonds: List[Bond] = []

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def Q(self) -> np.ndarray:
        """Current state as numpy array [Q_A, Q_S, Q_τ]."""
        return self._Q.copy()

    @property
    def Q_A(self) -> float:
        return self._Q[0]

    @property
    def Q_S(self) -> float:
        return self._Q[1]

    @property
    def Q_tau(self) -> float:
        return self._Q[2]

    @property
    def state(self) -> SemanticState:
        """Current state as SemanticState object."""
        return SemanticState(A=self.Q_A, S=self.Q_S, tau=self.Q_tau)

    @property
    def history(self) -> List[np.ndarray]:
        """History of Q values."""
        return self._history

    @property
    def bonds(self) -> List[Bond]:
        """History of processed bonds."""
        return self._bonds

    # ========================================================================
    # STATE OPERATIONS
    # ========================================================================

    def reset(self, initial: Optional[SemanticState] = None):
        """Reset state to initial values."""
        if initial:
            self._Q = np.array([initial.A, initial.S, initial.tau])
        else:
            self._Q = np.zeros(3)

        self._history.clear()
        self._bonds.clear()

    def set(self, A: float = None, S: float = None, tau: float = None):
        """Set state components directly."""
        if A is not None:
            self._Q[0] = A
        if S is not None:
            self._Q[1] = S
        if tau is not None:
            self._Q[2] = tau

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
        target = np.array([A, S, tau])
        self._Q = rc_update_exact(
            self._Q, target,
            dt=self.dt, decay=self.decay, Q_max=self.Q_max
        )

        # Track history
        self._history.append(self._Q.copy())
        if len(self._history) > self.history_size:
            self._history.pop(0)

        return self._Q.copy()

    def process_bond(self, bond: Bond) -> np.ndarray:
        """Process a bond and update state.

        Args:
            bond: Bond to process

        Returns:
            Updated Q state
        """
        self._bonds.append(bond)
        if len(self._bonds) > self.history_size:
            self._bonds.pop(0)

        return self.update(bond.A, bond.S, bond.tau)

    def process_trajectory(self, trajectory: Trajectory) -> List[np.ndarray]:
        """Process entire trajectory.

        Args:
            trajectory: Trajectory of bonds

        Returns:
            List of Q states after each bond
        """
        states = []
        for bond in trajectory.bonds:
            Q = self.process_bond(bond)
            states.append(Q)
        return states

    # ========================================================================
    # METRICS
    # ========================================================================

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity (rate of change)."""
        if len(self._history) < 2:
            return (0.0, 0.0, 0.0)

        prev = self._history[-2]
        curr = self._history[-1]

        return (
            curr[0] - prev[0],
            curr[1] - prev[1],
            curr[2] - prev[2],
        )

    def get_mean(self, window: int = 10) -> np.ndarray:
        """Get mean state over recent window."""
        if not self._history:
            return np.zeros(3)

        recent = self._history[-window:]
        return np.mean(recent, axis=0)

    def get_variance(self, window: int = 10) -> np.ndarray:
        """Get variance of state over recent window."""
        if len(self._history) < 2:
            return np.zeros(3)

        recent = self._history[-window:]
        return np.var(recent, axis=0)

    def get_slope(self, window: int = 10) -> Tuple[float, float, float]:
        """Get slope (trend) of each dimension."""
        if len(self._history) < 2:
            return (0.0, 0.0, 0.0)

        recent = np.array(self._history[-window:])
        n = len(recent)
        x = np.arange(n)

        slopes = []
        for dim in range(3):
            y = recent[:, dim]
            # Simple linear regression slope
            slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-6)
            slopes.append(slope)

        return tuple(slopes)

    def distance_to(self, target: SemanticState) -> float:
        """Euclidean distance to target state."""
        return np.linalg.norm(
            self._Q - np.array([target.A, target.S, target.tau])
        )

    # ========================================================================
    # TRAJECTORY
    # ========================================================================

    def to_trajectory(self) -> Trajectory:
        """Convert history to Trajectory object."""
        trajectory = Trajectory()
        trajectory.bonds = self._bonds.copy()

        for Q in self._history:
            trajectory.states.append(SemanticState(
                A=Q[0], S=Q[1], tau=Q[2]
            ))

        return trajectory

    def get_recent_bonds(self, n: int = 10) -> List[Bond]:
        """Get last N processed bonds."""
        return self._bonds[-n:]
