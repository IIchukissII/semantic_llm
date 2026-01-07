"""Generator Application: Text generation agent.

Uses Storm-Logos for semantic skeleton generation + LLM rendering.
"""

from typing import Optional, List

from ..data.models import SemanticState, Parameters, Bond
from ..orchestration.engine import Orchestrator, get_orchestrator
from ..generation.engine import GenerationEngine
from ..config import get_config, GenreParams


class Generator:
    """Text generation application.

    Generates text with specified genre characteristics:
    - dramatic: High tension, emotional jumps
    - ironic: Mundane sacred, controlled tension
    - balanced: Measured, equilibrium-seeking
    """

    def __init__(self):
        self.orchestrator = get_orchestrator()
        self.engine = GenerationEngine()
        self.config = get_config()

    def generate(self, genre: str = 'balanced',
                 n_sentences: int = 3,
                 seed: Optional[str] = None) -> str:
        """Generate text.

        Args:
            genre: 'dramatic', 'ironic', 'balanced'
            n_sentences: Number of sentences
            seed: Optional seed text/phrase

        Returns:
            Generated text
        """
        # Get genre parameters
        genre_params = self.config.get_genre(genre)

        # Compute seed state
        if seed:
            seed_state = self._seed_to_state(seed)
        else:
            seed_state = self._get_genre_seed(genre)

        # Generate skeleton
        skeleton = self.engine.generate_skeleton(
            genre=genre,
            n_sentences=n_sentences,
            seed_state=seed_state,
        )

        # Render to text
        return self.engine.renderer.render(skeleton, genre=genre)

    def generate_skeleton(self, genre: str = 'balanced',
                          n_sentences: int = 3) -> List[List[Bond]]:
        """Generate skeleton only (no rendering).

        Args:
            genre: Genre name
            n_sentences: Number of sentences

        Returns:
            Skeleton (list of sentences, each a list of bonds)
        """
        seed_state = self._get_genre_seed(genre)
        return self.engine.generate_skeleton(
            genre=genre,
            n_sentences=n_sentences,
            seed_state=seed_state,
        )

    def _seed_to_state(self, seed: str) -> SemanticState:
        """Convert seed text to initial state."""
        from ..metrics.engine import MetricsEngine
        metrics = MetricsEngine()
        m = metrics.measure(text=seed)
        return SemanticState(
            A=m.A_position,
            S=m.S_position,
            tau=m.tau_mean if m.tau_mean else 2.5,
        )

    def _get_genre_seed(self, genre: str) -> SemanticState:
        """Get appropriate seed state for genre."""
        if genre == 'dramatic':
            return SemanticState(A=0.3, S=0.3, tau=2.0)
        elif genre == 'ironic':
            return SemanticState(A=0.1, S=-0.1, tau=3.0)
        else:  # balanced
            return SemanticState(A=0.0, S=0.1, tau=2.5)

    def generate_with_params(self, genre: str,
                             params: Parameters,
                             n_sentences: int = 3) -> str:
        """Generate with explicit parameters.

        Args:
            genre: Genre name
            params: Generation parameters
            n_sentences: Number of sentences

        Returns:
            Generated text
        """
        seed_state = self._get_genre_seed(genre)
        skeleton = self.engine.generate_skeleton(
            genre=genre,
            n_sentences=n_sentences,
            seed_state=seed_state,
            params=params,
        )
        return self.engine.renderer.render(skeleton, genre=genre)

    def reset(self):
        """Reset generator state."""
        self.engine.reset()
