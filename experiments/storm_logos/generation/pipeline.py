"""Generation Pipeline: Storm -> Dialectic -> Chain.

The core generation algorithm:
1. STORM: Explode candidates
2. DIALECTIC: Filter by tension
3. CHAIN: Select via resonance
4. UPDATE: Advance Q state
"""

from typing import List, Optional

from ..data.models import Bond, SemanticState, GenerationResult, Parameters, Trajectory
from ..semantic.storm import Storm, get_storm
from ..semantic.dialectic import Dialectic, get_dialectic
from ..semantic.chain import ChainReaction, get_chain
from ..semantic.state import StateManager


class Pipeline:
    """Generation pipeline: Storm -> Dialectic -> Chain.

    Generates semantic skeletons (bond sequences) for LLM rendering.
    """

    def __init__(self,
                 storm: Optional[Storm] = None,
                 dialectic: Optional[Dialectic] = None,
                 chain: Optional[ChainReaction] = None):
        self.storm = storm or get_storm()
        self.dialectic = dialectic or get_dialectic()
        self.chain = chain or get_chain()
        self.state = StateManager()

    def generate_next(self, Q: SemanticState,
                      history: List[Bond],
                      params: Parameters) -> GenerationResult:
        """Generate next bond.

        Args:
            Q: Current state
            history: Recent bond history
            params: Adaptive parameters

        Returns:
            GenerationResult with bond, new state, metadata
        """
        # 1. STORM: Explode candidates
        candidates = self.storm.explode(
            Q,
            radius=params.storm_radius,
        )

        if not candidates:
            # Fallback: widen search
            candidates = self.storm.explode(Q, radius=params.storm_radius * 2)

        # 2. DIALECTIC: Filter by tension
        filtered = self.dialectic.filter(
            candidates,
            Q,
            tension_weight=params.dialectic_tension,
            coherence_threshold=params.coherence_threshold,
        )

        if not filtered:
            # Fallback: use all candidates
            filtered = candidates

        # 3. CHAIN: Select via resonance
        winner = self.chain.select(
            filtered,
            history,
            decay=params.chain_decay,
        )

        # 4. UPDATE: Advance state
        new_state = self.state.state
        self.state.process_bond(winner)
        new_state = self.state.state

        return GenerationResult(
            bond=winner,
            new_state=new_state,
            candidates_count=len(candidates),
            filtered_count=len(filtered),
            winner_score=self.chain._score(winner, history),
        )

    def generate_sentence(self, Q: SemanticState,
                          params: Parameters,
                          n_bonds: int = 4) -> List[Bond]:
        """Generate a sentence (sequence of bonds).

        Args:
            Q: Starting state
            params: Adaptive parameters
            n_bonds: Number of bonds per sentence

        Returns:
            List of bonds forming a sentence
        """
        self.state.reset(Q)
        history = []
        sentence = []

        for _ in range(n_bonds):
            result = self.generate_next(
                self.state.state,
                history,
                params,
            )
            sentence.append(result.bond)
            history.append(result.bond)

            # Limit history
            if len(history) > 10:
                history.pop(0)

        return sentence

    def generate_skeleton(self, Q: SemanticState,
                          params: Parameters,
                          n_sentences: int = 3,
                          bonds_per_sentence: int = 4) -> List[List[Bond]]:
        """Generate a skeleton (multiple sentences).

        Args:
            Q: Starting state
            params: Adaptive parameters
            n_sentences: Number of sentences
            bonds_per_sentence: Bonds per sentence

        Returns:
            List of sentences, each a list of bonds
        """
        self.state.reset(Q)
        skeleton = []
        history = []

        for sent_idx in range(n_sentences):
            sentence = []

            for bond_idx in range(bonds_per_sentence):
                result = self.generate_next(
                    self.state.state,
                    history,
                    params,
                )
                sentence.append(result.bond)
                history.append(result.bond)

                if len(history) > 10:
                    history.pop(0)

            skeleton.append(sentence)

            # Apply boundary jump (if genre params specify)
            # This is simplified - full implementation uses genre params

        return skeleton

    def to_trajectory(self) -> Trajectory:
        """Get current trajectory from state manager."""
        return self.state.to_trajectory()

    def reset(self, Q: Optional[SemanticState] = None):
        """Reset pipeline state."""
        self.state.reset(Q)
