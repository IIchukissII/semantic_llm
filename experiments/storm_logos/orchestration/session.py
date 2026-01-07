"""Session: Manages state across a generation/therapy session."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ..data.models import SemanticState, Trajectory, Metrics, Parameters


@dataclass
class Session:
    """Session state container.

    Tracks everything across a session:
    - Semantic position Q
    - Parameter history
    - Metrics history
    - Generated content
    """
    # Current state
    Q: SemanticState = field(default_factory=SemanticState)
    parameters: Parameters = field(default_factory=Parameters)

    # History
    state_history: List[SemanticState] = field(default_factory=list)
    metrics_history: List[Metrics] = field(default_factory=list)
    params_history: List[Parameters] = field(default_factory=list)

    # Content
    trajectory: Trajectory = field(default_factory=Trajectory)
    generated_text: List[str] = field(default_factory=list)

    # Metadata
    context: str = 'default'
    genre: str = 'balanced'
    turn_count: int = 0

    def update_state(self, new_state: SemanticState):
        """Update current state and add to history."""
        self.state_history.append(self.Q.copy())
        self.Q = new_state
        self.turn_count += 1

    def update_parameters(self, new_params: Parameters):
        """Update parameters and add to history."""
        self.params_history.append(self.parameters.copy())
        self.parameters = new_params

    def add_metrics(self, metrics: Metrics):
        """Add metrics to history."""
        self.metrics_history.append(metrics)

    def add_text(self, text: str):
        """Add generated text."""
        self.generated_text.append(text)

    def reset(self):
        """Reset session."""
        self.Q = SemanticState()
        self.parameters = Parameters()
        self.state_history.clear()
        self.metrics_history.clear()
        self.params_history.clear()
        self.trajectory = Trajectory()
        self.generated_text.clear()
        self.turn_count = 0

    def get_recent_states(self, n: int = 10) -> List[SemanticState]:
        """Get recent state history."""
        return self.state_history[-n:]

    def get_recent_metrics(self, n: int = 10) -> List[Metrics]:
        """Get recent metrics history."""
        return self.metrics_history[-n:]

    def to_dict(self) -> Dict:
        """Convert session to dictionary for serialization."""
        return {
            'Q': {'A': self.Q.A, 'S': self.Q.S, 'tau': self.Q.tau},
            'parameters': self.parameters.as_dict(),
            'context': self.context,
            'genre': self.genre,
            'turn_count': self.turn_count,
            'n_states': len(self.state_history),
            'n_metrics': len(self.metrics_history),
            'n_texts': len(self.generated_text),
        }
