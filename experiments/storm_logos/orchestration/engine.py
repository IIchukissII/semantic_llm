"""Orchestrator: High-level interface to the system."""

from typing import Optional, List

from ..data.models import SemanticState, Trajectory
from .session import Session
from .loop import MainLoop


class Orchestrator:
    """High-level orchestrator for Storm-Logos.

    Provides simplified interface for:
    - Generation mode
    - Therapy mode
    - Analysis mode
    """

    def __init__(self):
        self.loop = MainLoop()

    # ========================================================================
    # GENERATION MODE
    # ========================================================================

    def generate(self, genre: str = 'balanced',
                 n_sentences: int = 3,
                 n_bonds_per_sentence: int = 4) -> str:
        """Generate text in specified genre.

        Args:
            genre: 'dramatic', 'ironic', 'balanced'
            n_sentences: Number of sentences
            n_bonds_per_sentence: Bonds per sentence

        Returns:
            Generated text
        """
        self.loop.reset(context='generation')
        self.loop.session.genre = genre

        # Run generation steps
        total_bonds = n_sentences * n_bonds_per_sentence
        self.loop.run(n_steps=total_bonds)

        # Render to text
        trajectory = self.loop.generation.get_trajectory()
        skeleton = self._trajectory_to_skeleton(
            trajectory, n_bonds_per_sentence
        )

        return self.loop.generation.renderer.render(skeleton, genre=genre)

    def _trajectory_to_skeleton(self, trajectory: Trajectory,
                                bonds_per_sent: int) -> List[List]:
        """Convert flat trajectory to nested skeleton."""
        skeleton = []
        current_sent = []

        for bond in trajectory.bonds:
            current_sent.append(bond)
            if len(current_sent) >= bonds_per_sent:
                skeleton.append(current_sent)
                current_sent = []

        if current_sent:
            skeleton.append(current_sent)

        return skeleton

    # ========================================================================
    # THERAPY MODE
    # ========================================================================

    def therapy_step(self, patient_text: str) -> dict:
        """Process one therapy turn.

        Args:
            patient_text: Patient's utterance

        Returns:
            Dictionary with analysis and suggested response direction
        """
        self.loop.reset(context='therapeutic')

        # Analyze patient
        result = self.loop.step(input_text=patient_text)

        # Get dialectical analysis
        from ..semantic.dialectic import get_dialectic
        dialectic = get_dialectic()

        current_state = self.loop.session.Q
        analysis = dialectic.analyze(current_state)

        return {
            'metrics': result['metrics'],
            'dialectic': analysis,
            'suggested_direction': analysis.get('intervention', {}),
            'parameters': result['parameters'],
        }

    # ========================================================================
    # ANALYSIS MODE
    # ========================================================================

    def analyze(self, text: str) -> dict:
        """Analyze text without generation.

        Args:
            text: Text to analyze

        Returns:
            Full analysis dictionary
        """
        # Measure
        metrics = self.loop.metrics.measure(text=text)

        # Dialectic
        from ..semantic.dialectic import get_dialectic
        dialectic = get_dialectic()

        # Create state from metrics
        state = SemanticState(
            A=metrics.A_position,
            S=metrics.S_position,
        )
        dial_analysis = dialectic.analyze(state)

        return {
            'metrics': metrics.as_dict(),
            'dialectic': dial_analysis,
            'defenses': metrics.defenses,
        }

    # ========================================================================
    # STATE
    # ========================================================================

    def get_session(self) -> Session:
        """Get current session."""
        return self.loop.session

    def reset(self):
        """Reset orchestrator."""
        self.loop.reset()


# ============================================================================
# SINGLETON
# ============================================================================

_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get singleton Orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance
