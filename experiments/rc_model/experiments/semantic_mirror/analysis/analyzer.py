"""Semantic Analyzer: Dialectical analysis and diagnosis.

Psychoanalytic analysis:
- Diagnosis: defenses, resistance
- Dialectic: thesis → antithesis → synthesis
- Intervention vectors in semantic space
"""

import math
from typing import Dict, List, Optional

from ..core import SemanticState, ConversationTrajectory, HEALTH, resistance


class SemanticAnalyzer:
    """Analyze conversation for defenses, dialectic, therapeutic direction."""

    # Defense thresholds
    IRONY_THRESHOLD = 0.3
    SARCASM_THRESHOLD = 0.3
    INTELLECTUALIZATION_TAU = 3.0
    NEGATION_A = -0.3
    DEVALUATION_S = -0.3

    def diagnose(self, trajectory: ConversationTrajectory) -> Dict:
        """Psychoanalytic diagnosis: where is patient, what defenses?"""
        curr = trajectory.current
        if not curr:
            return {'status': 'no_data'}

        diagnosis = {
            'position': {'A': curr.A, 'S': curr.S, 'tau': curr.tau},
            'defenses': self._detect_defenses(curr),
            'resistance': resistance(trajectory.velocity, curr),
            'gravity_alignment': self._compute_alignment(trajectory),
            'distance_to_health': curr.distance_to(HEALTH),
        }

        return diagnosis

    def _detect_defenses(self, state: SemanticState) -> List[str]:
        """Detect active psychological defenses."""
        defenses = []

        if state.irony > self.IRONY_THRESHOLD:
            defenses.append('irony_as_distance')
        if state.sarcasm > self.SARCASM_THRESHOLD:
            defenses.append('sarcasm_as_aggression')
        if state.tau > self.INTELLECTUALIZATION_TAU:
            defenses.append('intellectualization')
        if state.A < self.NEGATION_A:
            defenses.append('negation')
        if state.S < self.DEVALUATION_S:
            defenses.append('devaluation')

        return defenses

    def _compute_alignment(self, trajectory: ConversationTrajectory) -> float:
        """Compute alignment with therapeutic direction."""
        curr = trajectory.current
        if not curr:
            return 0.0

        vel = trajectory.velocity
        delta_A = HEALTH.A - curr.A
        delta_S = HEALTH.S - curr.S

        # Alignment = dot product with health direction
        alignment = (
            vel[0] * (1 if delta_A > 0 else -1) +
            vel[1] * (1 if delta_S > 0 else -1)
        )

        return max(-1.0, min(1.0, alignment))

    def dialectic(self, trajectory: ConversationTrajectory) -> Dict:
        """Dialectical Engine: find antithesis and path to synthesis.

        THESIS: patient's current state
        ANTITHESIS: opposite in semantic space
        SYNTHESIS: higher meaning through resolved tension

        Like Hegel: meaning emerges from contradiction.
        Like therapy: growth through confronting opposites.
        """
        curr = trajectory.current
        if not curr:
            return {'status': 'no_data'}

        # THESIS: where patient is now
        thesis = {
            'A': curr.A,
            'S': curr.S,
            'tau': curr.tau,
            'description': self._describe_position(curr),
        }

        # ANTITHESIS: the opposite (what patient avoids/denies)
        anti_state = SemanticState(
            A=-curr.A,
            S=-curr.S,
            tau=4.0 - curr.tau
        )
        antithesis = {
            'A': anti_state.A,
            'S': anti_state.S,
            'tau': anti_state.tau,
            'description': self._describe_position(anti_state),
        }

        # SYNTHESIS: midpoint moved toward health (Aufhebung)
        synthesis = {
            'A': (curr.A + (-curr.A) + HEALTH.A) / 3,
            'S': max(curr.S, -curr.S, HEALTH.S),  # Elevate S
            'tau': HEALTH.tau,  # Ground in healthy τ
        }
        synth_state = SemanticState(
            A=synthesis['A'], S=synthesis['S'], tau=synthesis['tau']
        )
        synthesis['description'] = self._describe_position(synth_state)

        # Dialectical tension
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
            'blocking_defense': self._what_blocks_antithesis(curr, antithesis),
            'intervention': self._compute_intervention(curr, antithesis),
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

    def _what_blocks_antithesis(self, curr: SemanticState,
                                 anti: Dict) -> Optional[str]:
        """Identify which defense blocks access to the antithesis."""
        if curr.A < 0 and anti['A'] > 0:
            return "negation blocks affirmation"

        if curr.irony > 0.3:
            return "irony blocks sincerity"

        if curr.tau > 3.0 and anti['tau'] < 2.0:
            return "intellectualization blocks grounding"

        if curr.S < -0.2 and anti['S'] > 0.2:
            return "devaluation blocks meaning"

        return None

    def _compute_intervention(self, curr: SemanticState, anti: Dict) -> Dict:
        """Compute intervention as vector in semantic space.

        No hardcoded phrases. Just physics:
        - Where patient is (thesis)
        - What they avoid (antithesis)
        - Direction toward synthesis
        """
        # Synthesis coordinates
        synthesis_A = (curr.A + anti['A'] + HEALTH.A) / 3
        synthesis_S = max(curr.S, anti['S'], HEALTH.S)
        synthesis_tau = HEALTH.tau

        # Vector from thesis to synthesis
        delta = {
            'dA': synthesis_A - curr.A,
            'dS': synthesis_S - curr.S,
            'dtau': synthesis_tau - curr.tau,
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
