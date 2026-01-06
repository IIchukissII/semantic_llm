"""Semantic Models: State and Trajectory data classes.

Core data structures for tracking position in (A, S, τ) space.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SemanticState:
    """Position in semantic space with emotional markers."""
    A: float = 0.0      # Affirmation axis (-1 to +1)
    S: float = 0.0      # Sacred axis (-1 to +1)
    tau: float = 2.5    # Abstraction level (0.5 to 4.5)

    # Detected markers
    irony: float = 0.0       # 0-1 irony strength
    sarcasm: float = 0.0     # 0-1 sarcasm strength
    emotion: str = "neutral"  # primary emotion
    intensity: float = 0.0   # emotional intensity

    def copy(self) -> 'SemanticState':
        return SemanticState(
            A=self.A, S=self.S, tau=self.tau,
            irony=self.irony, sarcasm=self.sarcasm,
            emotion=self.emotion, intensity=self.intensity
        )

    def distance_to(self, other: 'SemanticState') -> float:
        """Euclidean distance in semantic space."""
        return (
            (self.A - other.A)**2 +
            (self.S - other.S)**2 +
            (self.tau - other.tau)**2
        ) ** 0.5

    def __repr__(self) -> str:
        return f"State(A={self.A:+.2f}, S={self.S:+.2f}, τ={self.tau:.2f})"


@dataclass
class ConversationTrajectory:
    """Track conversation path through semantic space.

    Implements RC dynamics: state accumulates like charge on capacitor.
    """
    history: List[SemanticState] = field(default_factory=list)
    window_size: int = 10  # RC memory window

    def add(self, state: SemanticState):
        """Add new state to trajectory."""
        self.history.append(state.copy())

    @property
    def current(self) -> Optional[SemanticState]:
        """Most recent state."""
        return self.history[-1] if self.history else None

    @property
    def previous(self) -> Optional[SemanticState]:
        """Previous state."""
        return self.history[-2] if len(self.history) >= 2 else None

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Semantic velocity: direction of movement (dA, dS, dτ)."""
        if len(self.history) < 2:
            return (0.0, 0.0, 0.0)
        prev, curr = self.history[-2], self.history[-1]
        return (
            curr.A - prev.A,
            curr.S - prev.S,
            curr.tau - prev.tau
        )

    @property
    def mean_state(self) -> SemanticState:
        """Average state over recent window (RC charge)."""
        if not self.history:
            return SemanticState()
        window = self.history[-self.window_size:]
        n = len(window)
        return SemanticState(
            A=sum(s.A for s in window) / n,
            S=sum(s.S for s in window) / n,
            tau=sum(s.tau for s in window) / n,
            irony=sum(s.irony for s in window) / n,
            sarcasm=sum(s.sarcasm for s in window) / n,
        )

    @property
    def n_turns(self) -> int:
        """Number of turns in conversation."""
        return len(self.history)

    def clear(self):
        """Reset trajectory."""
        self.history.clear()
