"""Chain Reaction: Resonance Scoring and Winner Selection.

The final stage of semantic selection:
1. Score candidates by resonance with history
2. Apply lasing (exponential amplification above threshold)
3. Select winner via weighted sampling

Principle: Coherent paths amplify, noise dampens.
Like neocortical lateral inhibition + resonance.
"""

from typing import List, Optional, Tuple
import math
import random
import numpy as np

from ..data.models import Bond, SemanticState
from ..config import get_config, ChainConfig
from .physics import coherence


class ChainReaction:
    """Chain reaction: resonance scoring with exponential amplification.

    Mechanism:
        1. Each previous bond "votes" for candidates
        2. Votes accumulate with exponential decay
        3. Above threshold: lasing (quadratic amplification)
        4. Winner selected via weighted sampling

    This creates emergent coherence: paths that resonate with history
    get amplified, random noise gets dampened.
    """

    def __init__(self, config: Optional[ChainConfig] = None):
        self.config = config or get_config().chain

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================

    def select(self, candidates: List[Bond],
               history: List[Bond],
               decay: Optional[float] = None,
               threshold: Optional[float] = None) -> Bond:
        """Select winner via chain reaction.

        Args:
            candidates: Filtered candidates from Dialectic
            history: Recent bond history
            decay: Resonance decay factor
            threshold: Lasing threshold

        Returns:
            Winning bond
        """
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0]

        decay = decay or self.config.decay
        threshold = threshold or self.config.threshold

        # Score all candidates
        scored = []
        for bond in candidates:
            power = self._score(bond, history, decay)
            lased = self._lasing(power, threshold)
            scored.append((bond, lased))

        # Weighted random selection
        total = sum(s for _, s in scored)
        if total <= 0:
            return random.choice(candidates)

        # Sample
        r = random.random() * total
        cumsum = 0
        for bond, score in scored:
            cumsum += score
            if cumsum >= r:
                return bond

        return scored[-1][0]  # Fallback

    def select_deterministic(self, candidates: List[Bond],
                             history: List[Bond]) -> Bond:
        """Select winner deterministically (highest score)."""
        if not candidates:
            raise ValueError("No candidates to select from")

        if len(candidates) == 1:
            return candidates[0]

        # Score and return max
        scored = [(bond, self._score(bond, history)) for bond in candidates]
        return max(scored, key=lambda x: x[1])[0]

    # ========================================================================
    # SCORING
    # ========================================================================

    def _score(self, candidate: Bond, history: List[Bond],
               decay: Optional[float] = None) -> float:
        """Compute resonance score for candidate.

        power = Σ coherence(candidate, prev[i]) × decay^i

        Recent history has more weight (higher decay^i for small i).

        Args:
            candidate: Candidate bond
            history: Bond history (most recent last)
            decay: Decay factor

        Returns:
            Resonance power
        """
        decay = decay or self.config.decay

        if not history:
            return 1.0

        power = 0.0
        for i, prev in enumerate(reversed(history)):
            # State from previous bond
            state = SemanticState(A=prev.A, S=prev.S, tau=prev.tau)
            coh = coherence(state, candidate)

            # Positive coherence only contributes
            if coh > 0:
                power += coh * (decay ** i)

        return max(power, 0.01)  # Minimum score

    def _lasing(self, power: float, threshold: Optional[float] = None) -> float:
        """Apply lasing: exponential amplification above threshold.

        Below threshold: linear
        Above threshold: quadratic (rapid amplification)

        This creates winner-take-all dynamics for strongly resonant candidates.
        """
        threshold = threshold or self.config.threshold

        if power > threshold:
            # Quadratic amplification above threshold
            excess = power - threshold
            return threshold + excess ** 2
        else:
            return power

    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================

    def score_all(self, candidates: List[Bond],
                  history: List[Bond]) -> List[Tuple[Bond, float]]:
        """Score all candidates.

        Returns:
            List of (bond, score) tuples sorted by score descending
        """
        scored = [(bond, self._score(bond, history)) for bond in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def top_k(self, candidates: List[Bond],
              history: List[Bond],
              k: int = 5) -> List[Bond]:
        """Get top K candidates by resonance score."""
        scored = self.score_all(candidates, history)
        return [bond for bond, score in scored[:k]]

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    def analyze_selection(self, candidates: List[Bond],
                          history: List[Bond]) -> dict:
        """Analyze the selection process.

        Returns detailed breakdown of scoring for debugging.
        """
        decay = self.config.decay
        threshold = self.config.threshold

        analysis = {
            'n_candidates': len(candidates),
            'history_length': len(history),
            'decay': decay,
            'threshold': threshold,
            'candidates': [],
        }

        for bond in candidates:
            power = self._score(bond, history, decay)
            lased = self._lasing(power, threshold)

            # Breakdown by history position
            contributions = []
            for i, prev in enumerate(reversed(history)):
                state = SemanticState(A=prev.A, S=prev.S, tau=prev.tau)
                coh = coherence(state, bond)
                contribution = coh * (decay ** i) if coh > 0 else 0
                contributions.append({
                    'position': i,
                    'coherence': coh,
                    'contribution': contribution,
                })

            analysis['candidates'].append({
                'bond': bond.text,
                'power': power,
                'lased': lased,
                'above_threshold': power > threshold,
                'contributions': contributions[:5],  # Top 5
            })

        return analysis


# ============================================================================
# SINGLETON
# ============================================================================

_chain_instance: Optional[ChainReaction] = None


def get_chain() -> ChainReaction:
    """Get singleton ChainReaction instance."""
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = ChainReaction()
    return _chain_instance
