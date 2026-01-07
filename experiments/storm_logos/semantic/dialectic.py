"""Dialectic: Thesis-Antithesis Filtering.

Filters candidates based on dialectical tension:
- THESIS: Current position Q
- ANTITHESIS: Opposite pole in semantic space
- SYNTHESIS: Candidates that HOLD both poles in tension

Principle: Tension > Purity
    coherence 0.80 with tension
    coherence 0.47 with pure direction
"""

from typing import List, Dict, Optional, Tuple
import math

from ..data.models import Bond, SemanticState
from ..config import get_config, DialecticConfig, HealthTarget
from .physics import coherence


class Dialectic:
    """Dialectical engine: filter candidates by tension.

    Implements Hegelian dialectic:
        1. Identify thesis (where we are)
        2. Compute antithesis (what we avoid/deny)
        3. Score candidates by how well they hold both

    Candidates that purely align with thesis are LESS interesting
    than candidates that create productive tension.
    """

    def __init__(self, config: Optional[DialecticConfig] = None):
        self.config = config or get_config().dialectic

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================

    def filter(self, candidates: List[Bond],
               Q: SemanticState,
               tension_weight: Optional[float] = None,
               coherence_threshold: Optional[float] = None) -> List[Bond]:
        """Filter candidates by dialectical tension.

        Args:
            candidates: Unfiltered candidates from Storm
            Q: Current state (thesis)
            tension_weight: Weight for tension vs coherence
            coherence_threshold: Minimum coherence to pass

        Returns:
            Filtered candidates sorted by dialectical score
        """
        tension_weight = tension_weight or self.config.tension_weight
        coherence_threshold = coherence_threshold or self.config.coherence_threshold

        # Compute antithesis
        antithesis = self._compute_antithesis(Q)

        # Score each candidate
        scored = []
        for bond in candidates:
            score = self._dialectical_score(
                bond, Q, antithesis, tension_weight
            )
            coh = coherence(Q, bond)

            # Apply coherence threshold
            if coh >= coherence_threshold:
                scored.append((bond, score, coh))

        # Sort by dialectical score (higher = better)
        scored.sort(key=lambda x: x[1], reverse=True)

        return [bond for bond, score, coh in scored]

    # ========================================================================
    # DIALECTICAL OPERATIONS
    # ========================================================================

    def _compute_antithesis(self, thesis: SemanticState) -> SemanticState:
        """Compute antithesis: the opposite pole.

        Simple negation:
            A_anti = -A_thesis
            S_anti = -S_thesis
            τ_anti = 4.0 - τ_thesis  (flip abstraction)
        """
        return SemanticState(
            A=-thesis.A,
            S=-thesis.S,
            tau=4.0 - thesis.tau
        )

    def _dialectical_score(self, bond: Bond,
                           thesis: SemanticState,
                           antithesis: SemanticState,
                           tension_weight: float) -> float:
        """Score bond by how well it holds dialectical tension.

        High score = bond resonates with BOTH thesis and antithesis
        This creates productive tension rather than pure alignment.

        Score = coh_thesis × (1 - w) + coh_antithesis × w
        """
        coh_thesis = coherence(thesis, bond)
        coh_anti = coherence(antithesis, bond)

        # Combine with tension weight
        # High tension_weight = value antithesis alignment more
        score = coh_thesis * (1 - tension_weight) + coh_anti * tension_weight

        # Bonus for holding both poles (geometric mean)
        if coh_thesis > 0 and coh_anti > 0:
            tension_bonus = math.sqrt(coh_thesis * coh_anti)
            score = score * 0.7 + tension_bonus * 0.3

        return score

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    def analyze(self, Q: SemanticState) -> Dict:
        """Full dialectical analysis of current position.

        Returns thesis, antithesis, synthesis, tension metrics.
        """
        config = get_config()
        health = config.health

        # Thesis = current position
        thesis = {
            'A': Q.A,
            'S': Q.S,
            'tau': Q.tau,
            'description': self._describe_position(Q),
        }

        # Antithesis = opposite
        anti = self._compute_antithesis(Q)
        antithesis = {
            'A': anti.A,
            'S': anti.S,
            'tau': anti.tau,
            'description': self._describe_position(anti),
        }

        # Synthesis = midpoint moved toward health (Aufhebung)
        synthesis_A = (Q.A + anti.A + health.A) / 3
        synthesis_S = max(Q.S, anti.S, health.S)  # Elevate S
        synthesis_tau = health.tau  # Ground in healthy τ

        synth_state = SemanticState(A=synthesis_A, S=synthesis_S, tau=synthesis_tau)
        synthesis = {
            'A': synthesis_A,
            'S': synthesis_S,
            'tau': synthesis_tau,
            'description': self._describe_position(synth_state),
        }

        # Dialectical tension = distance thesis to antithesis
        tension = math.sqrt(
            (thesis['A'] - antithesis['A'])**2 +
            (thesis['S'] - antithesis['S'])**2 +
            (thesis['tau'] - antithesis['tau'])**2
        )

        return {
            'thesis': thesis,
            'antithesis': antithesis,
            'synthesis': synthesis,
            'tension': tension,
            'blocking_defense': self._what_blocks_antithesis(Q, anti),
            'intervention': self._compute_intervention(Q, anti, health),
        }

    def _describe_position(self, state: SemanticState) -> str:
        """Brief description of semantic position."""
        parts = []

        if state.A > 0.3:
            parts.append("affirming")
        elif state.A < -0.3:
            parts.append("negating")
        else:
            parts.append("neutral")

        if state.S > 0.2:
            parts.append("elevated")
        elif state.S < -0.2:
            parts.append("mundane")

        if state.tau < 1.5:
            parts.append("concrete")
        elif state.tau > 3.0:
            parts.append("abstract")

        return ", ".join(parts) if parts else "balanced"

    def _what_blocks_antithesis(self, thesis: SemanticState,
                                antithesis: SemanticState) -> Optional[str]:
        """Identify which defense blocks access to the antithesis."""
        if thesis.A < 0 and antithesis.A > 0:
            return "negation blocks affirmation"

        if thesis.irony > 0.3:
            return "irony blocks sincerity"

        if thesis.tau > 3.0 and antithesis.tau < 2.0:
            return "intellectualization blocks grounding"

        if thesis.S < -0.2 and antithesis.S > 0.2:
            return "devaluation blocks meaning"

        return None

    def _compute_intervention(self, thesis: SemanticState,
                              antithesis: SemanticState,
                              health: HealthTarget) -> Dict:
        """Compute intervention as vector in semantic space.

        No hardcoded phrases. Just physics:
        - Where patient is (thesis)
        - What they avoid (antithesis)
        - Direction toward synthesis
        """
        # Synthesis coordinates
        synthesis_A = (thesis.A + antithesis.A + health.A) / 3
        synthesis_S = max(thesis.S, antithesis.S, health.S)
        synthesis_tau = health.tau

        # Vector from thesis to synthesis
        delta = {
            'dA': synthesis_A - thesis.A,
            'dS': synthesis_S - thesis.S,
            'dtau': synthesis_tau - thesis.tau,
        }

        # Dominant axis
        max_delta = max(abs(delta['dA']), abs(delta['dS']), abs(delta['dtau']))

        if abs(delta['dA']) == max_delta:
            axis = 'A'
            direction = '+affirm' if delta['dA'] > 0 else '+question'
        elif abs(delta['dS']) == max_delta:
            axis = 'S'
            direction = '+elevate' if delta['dS'] > 0 else '+ground'
        else:
            axis = 'τ'
            direction = '+abstract' if delta['dtau'] > 0 else '+concrete'

        return {
            'vector': delta,
            'primary_axis': axis,
            'direction': direction,
            'magnitude': math.sqrt(
                delta['dA']**2 + delta['dS']**2 + delta['dtau']**2
            ),
        }


# ============================================================================
# SINGLETON
# ============================================================================

_dialectic_instance: Optional[Dialectic] = None


def get_dialectic() -> Dialectic:
    """Get singleton Dialectic instance."""
    global _dialectic_instance
    if _dialectic_instance is None:
        _dialectic_instance = Dialectic()
    return _dialectic_instance
