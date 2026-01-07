"""Tension Analyzer: Measure dialectical tension."""

from typing import List, Optional
import math

from ...data.models import Bond, Trajectory, SemanticState
from ...semantic.dialectic import Dialectic


class TensionAnalyzer:
    """Analyze dialectical tension in trajectory.

    Tension = distance between thesis and antithesis.
    High tension = meaningful discourse.
    Low tension = flat, one-sided.
    """

    def __init__(self):
        self.dialectic = Dialectic()

    def analyze(self, trajectory: Trajectory = None,
                state: SemanticState = None) -> float:
        """Analyze tension.

        Args:
            trajectory: Trajectory to analyze
            state: Current state to analyze

        Returns:
            Tension score (0 to 1)
        """
        if state:
            analysis = self.dialectic.analyze(state)
            # Normalize tension to 0-1 range
            # Max tension is sqrt(2^2 + 2^2 + 4^2) ≈ 4.9
            return min(analysis['tension'] / 4.9, 1.0)

        if trajectory and trajectory.current_state:
            return self.analyze(state=trajectory.current_state)

        if trajectory and trajectory.bonds:
            # Create state from last bond
            last = trajectory.bonds[-1]
            state = SemanticState(A=last.A, S=last.S, tau=last.tau)
            return self.analyze(state=state)

        return 0.5  # Default neutral tension

    def analyze_trajectory_tension(self, trajectory: Trajectory) -> dict:
        """Analyze tension across entire trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dictionary with tension metrics
        """
        if not trajectory.bonds:
            return {
                'mean_tension': 0.5,
                'tension_variance': 0.0,
                'max_tension': 0.5,
            }

        tensions = []
        for bond in trajectory.bonds:
            state = SemanticState(A=bond.A, S=bond.S, tau=bond.tau)
            tension = self.analyze(state=state)
            tensions.append(tension)

        import numpy as np
        return {
            'mean_tension': np.mean(tensions),
            'tension_variance': np.var(tensions),
            'max_tension': np.max(tensions),
            'min_tension': np.min(tensions),
        }

    def compute_holding_score(self, trajectory: Trajectory) -> float:
        """Compute how well trajectory holds tension.

        Good holding = maintains tension without collapsing to one pole.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Holding score (0 to 1)
        """
        if len(trajectory.bonds) < 3:
            return 0.5

        import numpy as np

        # Get A and S values
        A_vals = np.array([b.A for b in trajectory.bonds])
        S_vals = np.array([b.S for b in trajectory.bonds])

        # Check if values stay near both poles or collapse to one
        A_variance = np.var(A_vals)
        S_variance = np.var(S_vals)

        # High variance = moves between poles = holding tension
        # Low variance = stuck at one pole = not holding

        holding = (A_variance + S_variance) / 2

        # Normalize (max expected variance is ~1)
        return min(holding, 1.0)

    def detect_collapse(self, trajectory: Trajectory,
                        window: int = 5) -> bool:
        """Detect if tension has collapsed.

        Collapse = variance drops significantly.

        Args:
            trajectory: Trajectory to analyze
            window: Window to analyze

        Returns:
            True if tension collapsed
        """
        if len(trajectory.bonds) < window * 2:
            return False

        import numpy as np

        bonds = trajectory.bonds

        # Compare early vs late variance
        early = bonds[:window]
        late = bonds[-window:]

        early_var = np.var([b.A for b in early]) + np.var([b.S for b in early])
        late_var = np.var([b.A for b in late]) + np.var([b.S for b in late])

        # Collapse if late variance is much lower
        return late_var < early_var * 0.3
