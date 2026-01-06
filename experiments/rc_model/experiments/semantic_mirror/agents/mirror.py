"""Semantic Mirror: The Psychoanalyst Agent.

Main agent that:
1. DETECTS: where human is (irony, sarcasm, emotions, defenses)
2. TRACKS: trajectory through semantic space
3. GUIDES: uses gravity to move toward good (+A) and sacred (+S)

Uses dialectical engine for thesis → antithesis → synthesis.
"""

from typing import Dict, Optional

from ..core import (
    SemanticState, ConversationTrajectory,
    get_data, SemanticData,
    HEALTH, therapeutic_vector,
)
from ..detection import SemanticDetector
from ..analysis import SemanticAnalyzer


class SemanticMirror:
    """Psychoanalyst agent: understand + guide toward good/sacred."""

    def __init__(self, data: SemanticData = None):
        """Initialize mirror with semantic data.

        Args:
            data: SemanticData instance (uses singleton if None)
        """
        self.data = data or get_data()
        self.detector = SemanticDetector(self.data)
        self.analyzer = SemanticAnalyzer()
        self.trajectory = ConversationTrajectory()

    # ════════════════════════════════════════════════════════════════
    # CORE API
    # ════════════════════════════════════════════════════════════════

    def observe(self, text: str) -> SemanticState:
        """Observe human text, update trajectory, return state.

        This is the main entry point for each turn of conversation.
        """
        state = self.detector.detect(text)
        self.trajectory.add(state)
        return state

    def diagnose(self) -> Dict:
        """Get psychoanalytic diagnosis of current state."""
        return self.analyzer.diagnose(self.trajectory)

    def dialectic(self) -> Dict:
        """Get dialectical analysis: thesis → antithesis → synthesis."""
        return self.analyzer.dialectic(self.trajectory)

    def therapeutic_direction(self) -> Dict:
        """Get direction toward health as vector."""
        curr = self.trajectory.current
        if not curr:
            return {'status': 'no_data'}

        vec = therapeutic_vector(curr)
        return {
            'vector': {'dA': vec[0], 'dS': vec[1], 'dtau': vec[2]},
            'current': {'A': curr.A, 'S': curr.S, 'tau': curr.tau},
            'target': {'A': HEALTH.A, 'S': HEALTH.S, 'tau': HEALTH.tau},
            'distance': curr.distance_to(HEALTH),
        }

    # ════════════════════════════════════════════════════════════════
    # CONTEXT
    # ════════════════════════════════════════════════════════════════

    def get_context(self) -> Dict:
        """Get full conversation context for response generation."""
        curr = self.trajectory.current
        mean = self.trajectory.mean_state
        vel = self.trajectory.velocity

        return {
            'current': {
                'A': curr.A if curr else 0,
                'S': curr.S if curr else 0,
                'tau': curr.tau if curr else 2.5,
                'irony': curr.irony if curr else 0,
                'sarcasm': curr.sarcasm if curr else 0,
                'emotion': curr.emotion if curr else 'neutral',
                'intensity': curr.intensity if curr else 0,
            },
            'trajectory': {
                'mean_A': mean.A,
                'mean_S': mean.S,
                'mean_tau': mean.tau,
                'mean_irony': mean.irony,
                'velocity': vel,
                'n_turns': self.trajectory.n_turns,
            },
            'interpretation': self._interpret(),
        }

    def _interpret(self) -> str:
        """Human-readable interpretation of current state."""
        curr = self.trajectory.current
        if not curr:
            return "No conversation yet."

        parts = []

        # Emotional state
        if curr.emotion != 'neutral':
            parts.append(f"Emotion: {curr.emotion} ({curr.intensity:.0%})")

        # Irony/sarcasm
        if curr.irony > 0.3:
            parts.append(f"Irony detected ({curr.irony:.0%})")
        if curr.sarcasm > 0.3:
            parts.append(f"Sarcasm detected ({curr.sarcasm:.0%})")

        # Semantic position
        if curr.A > 0.3:
            parts.append("Affirming/positive tone")
        elif curr.A < -0.3:
            parts.append("Negating/critical tone")

        if curr.S > 0.2:
            parts.append("Elevated/philosophical")
        elif curr.S < -0.2:
            parts.append("Mundane/everyday")

        if curr.tau < 2.0:
            parts.append("Concrete/specific")
        elif curr.tau > 3.0:
            parts.append("Abstract/general")

        # Trajectory
        vel = self.trajectory.velocity
        if abs(vel[0]) > 0.2:
            direction = "more positive" if vel[0] > 0 else "more negative"
            parts.append(f"Moving {direction}")

        return " | ".join(parts) if parts else "Neutral state"

    # ════════════════════════════════════════════════════════════════
    # SESSION MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def reset(self):
        """Reset session (clear trajectory)."""
        self.trajectory.clear()

    def summary(self) -> Dict:
        """Get session summary."""
        diagnosis = self.diagnose()
        mean = self.trajectory.mean_state

        return {
            'n_turns': self.trajectory.n_turns,
            'mean_position': {
                'A': mean.A,
                'S': mean.S,
                'tau': mean.tau,
            },
            'mean_irony': mean.irony,
            'defenses_observed': diagnosis.get('defenses', []),
            'overall_resistance': diagnosis.get('resistance', 0),
            'distance_to_health': diagnosis.get('distance_to_health', 0),
        }

    @property
    def health_target(self) -> SemanticState:
        """Health target state."""
        return HEALTH


# ════════════════════════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════════════════════════

_mirror_instance: Optional[SemanticMirror] = None


def get_mirror() -> SemanticMirror:
    """Get singleton SemanticMirror instance."""
    global _mirror_instance
    if _mirror_instance is None:
        _mirror_instance = SemanticMirror()
    return _mirror_instance
