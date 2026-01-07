"""Generation Engine: Orchestrates the generation pipeline."""

from typing import List, Optional

from ..data.models import (
    Bond, SemanticState, GenerationResult, Parameters, Trajectory
)
from ..config import get_config, GenreParams, GENRE_PRESETS
from .pipeline import Pipeline
from .renderer import Renderer


class GenerationEngine:
    """Generation Engine: full generation with rendering.

    Usage:
        engine = GenerationEngine()
        text = engine.generate(genre='dramatic', n_sentences=3)
    """

    def __init__(self, renderer: Optional[Renderer] = None):
        self.pipeline = Pipeline()
        self.renderer = renderer or Renderer()
        self.config = get_config()

    def generate_skeleton(self, genre: str = 'balanced',
                          n_sentences: int = 3,
                          seed_state: Optional[SemanticState] = None,
                          params: Optional[Parameters] = None) -> List[List[Bond]]:
        """Generate semantic skeleton.

        Args:
            genre: 'dramatic', 'ironic', 'balanced'
            n_sentences: Number of sentences
            seed_state: Starting state (optional)
            params: Override parameters (optional)

        Returns:
            Skeleton (list of sentences, each a list of bonds)
        """
        # Get genre params
        genre_params = self.config.get_genre(genre)

        # Starting state
        if seed_state is None:
            seed_state = SemanticState(A=0.0, S=0.0, tau=3.0)

        # Parameters
        if params is None:
            params = Parameters(
                storm_radius=genre_params.R_storm,
                coherence_threshold=genre_params.coh_threshold,
            )

        return self.pipeline.generate_skeleton(
            Q=seed_state,
            params=params,
            n_sentences=n_sentences,
            bonds_per_sentence=genre_params.bonds_per_sentence,
        )

    def generate(self, genre: str = 'balanced',
                 n_sentences: int = 3,
                 seed_state: Optional[SemanticState] = None,
                 params: Optional[Parameters] = None) -> str:
        """Generate text.

        Args:
            genre: 'dramatic', 'ironic', 'balanced'
            n_sentences: Number of sentences
            seed_state: Starting state (optional)
            params: Override parameters (optional)

        Returns:
            Generated text
        """
        skeleton = self.generate_skeleton(
            genre=genre,
            n_sentences=n_sentences,
            seed_state=seed_state,
            params=params,
        )

        return self.renderer.render(skeleton, genre=genre)

    def generate_next(self, params: Parameters) -> GenerationResult:
        """Generate single next bond.

        Args:
            params: Generation parameters

        Returns:
            GenerationResult
        """
        Q = self.pipeline.state.state
        history = self.pipeline.state.get_recent_bonds()
        return self.pipeline.generate_next(Q, history, params)

    def get_trajectory(self) -> Trajectory:
        """Get current trajectory."""
        return self.pipeline.to_trajectory()

    def reset(self, seed_state: Optional[SemanticState] = None):
        """Reset engine state."""
        self.pipeline.reset(seed_state)


# ============================================================================
# SINGLETON
# ============================================================================

_engine_instance: Optional[GenerationEngine] = None


def get_generation_engine() -> GenerationEngine:
    """Get singleton GenerationEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = GenerationEngine()
    return _engine_instance
