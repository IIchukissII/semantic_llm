"""Navigator Application: Semantic navigation agent.

Navigates through semantic space with goal-directed behavior.
"""

from typing import Optional, List, Tuple
import math

from ..data.models import SemanticState, Bond
from ..semantic.storm import Storm, get_storm
from ..semantic.physics import therapeutic_vector
from ..semantic.state import StateManager


class Navigator:
    """Semantic navigation agent.

    Navigates from current position to goal in (A, S, τ) space.
    Uses Storm for candidate generation and physics for direction.
    """

    def __init__(self):
        self.storm = get_storm()
        self.state = StateManager()
        self._path: List[Bond] = []

    def navigate(self, start: SemanticState,
                 goal: SemanticState,
                 max_steps: int = 20) -> List[Bond]:
        """Navigate from start to goal.

        Args:
            start: Starting position
            goal: Goal position
            max_steps: Maximum steps

        Returns:
            Path (list of bonds)
        """
        self.state.reset(start)
        self._path = []

        for _ in range(max_steps):
            # Check if reached goal
            if self._at_goal(goal):
                break

            # Get next step
            next_bond = self._step_toward(goal)
            if next_bond:
                self._path.append(next_bond)
                self.state.process_bond(next_bond)
            else:
                break  # No valid moves

        return self._path

    def _at_goal(self, goal: SemanticState, threshold: float = 0.3) -> bool:
        """Check if current position is at goal."""
        return self.state.distance_to(goal) < threshold

    def _step_toward(self, goal: SemanticState) -> Optional[Bond]:
        """Take one step toward goal."""
        current = self.state.state

        # Get direction vector
        direction = therapeutic_vector(current, goal)

        # Target position
        target_A = current.A + direction[0] * 0.3
        target_S = current.S + direction[1] * 0.3
        target_tau = current.tau + direction[2] * 0.3

        # Get candidates near target
        candidates = self.storm.get_candidates_by_coords(
            target_A, target_S, target_tau, radius=0.5
        )

        if not candidates:
            # Widen search
            candidates = self.storm.explode(current, radius=1.0)

        if not candidates:
            return None

        # Select best candidate (closest to direction)
        best = None
        best_score = -float('inf')

        for bond in candidates:
            # Score = how well bond aligns with direction
            delta = (
                (bond.A - current.A) * direction[0] +
                (bond.S - current.S) * direction[1] +
                (bond.tau - current.tau) * direction[2]
            )
            if delta > best_score:
                best_score = delta
                best = bond

        return best

    def navigate_with_waypoints(self, start: SemanticState,
                                 waypoints: List[SemanticState],
                                 goal: SemanticState) -> List[Bond]:
        """Navigate through waypoints.

        Args:
            start: Starting position
            waypoints: Intermediate waypoints
            goal: Final goal

        Returns:
            Full path
        """
        full_path = []
        current = start

        for waypoint in waypoints:
            segment = self.navigate(current, waypoint)
            full_path.extend(segment)
            current = self.state.state

        # Final segment to goal
        final = self.navigate(current, goal)
        full_path.extend(final)

        return full_path

    def get_path(self) -> List[Bond]:
        """Get current navigation path."""
        return self._path

    def get_position(self) -> SemanticState:
        """Get current position."""
        return self.state.state

    def reset(self):
        """Reset navigator."""
        self.state.reset()
        self._path = []
