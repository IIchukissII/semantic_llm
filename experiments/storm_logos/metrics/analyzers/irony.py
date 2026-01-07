"""Irony Analyzer: Detect irony from text and semantic contradictions."""

import re
from typing import List, Optional

from ...data.models import SemanticState
from ...data.postgres import PostgresData, get_data


class IronyAnalyzer:
    """Detect irony from text and semantic signals.

    Irony sources:
    1. Pattern matching (rhetorical markers)
    2. Semantic contradiction (positive words + negative context)
    3. Context mismatch
    """

    IRONY_PATTERNS = [
        r'\b(obviously|clearly|surely|of course)\b',
        r'\b(wonderful|great|fantastic|brilliant)\b.*\b(problem|issue|fail)',
        r'\.{3}',  # Ellipsis
        r'\?$',  # Rhetorical questions
    ]

    SARCASM_AMPLIFIERS = [
        r'\b(so|very|really|absolutely|totally|completely)\b',
        r'\b(always|never|everyone|nobody)\b',
        r'!+',
    ]

    def __init__(self, data: Optional[PostgresData] = None):
        self.data = data or get_data()

    def analyze(self, text: str = None,
                state: SemanticState = None) -> float:
        """Analyze text/state for irony.

        Args:
            text: Text to analyze
            state: Current semantic state

        Returns:
            Irony score (0 to 1)
        """
        score = 0.0

        if text:
            # Pattern matching
            for pattern in self.IRONY_PATTERNS:
                if re.search(pattern, text.lower()):
                    score += 0.15

            # Compute semantic contradiction
            A, S = self._compute_coordinates(text)
            if A > 0.2 and S < -0.1:
                score += 0.2

            # Rhetorical question at end
            if text.strip().endswith('?'):
                score += 0.1

        if state and state.irony > 0:
            # Blend with existing state
            score = max(score, state.irony)

        return min(score, 1.0)

    def analyze_sarcasm(self, text: str) -> float:
        """Detect sarcasm from exaggeration patterns.

        Args:
            text: Text to analyze

        Returns:
            Sarcasm score (0 to 1)
        """
        score = 0.0

        # Amplifiers + high apparent positivity
        amplifier_count = 0
        for pattern in self.SARCASM_AMPLIFIERS:
            amplifier_count += len(re.findall(pattern, text.lower()))

        A, _ = self._compute_coordinates(text)

        if amplifier_count > 0 and A > 0.3:
            score += 0.15 * amplifier_count

        # ALL CAPS sections
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            score += 0.2

        return min(score, 1.0)

    def _compute_coordinates(self, text: str) -> tuple:
        """Compute (A, S) from text words."""
        words = re.findall(r'\b[a-z]+\b', text.lower())

        A_vals, S_vals = [], []
        for word in words:
            coords = self.data.get(word)
            if coords:
                A_vals.append(coords.A)
                S_vals.append(coords.S)

        if not A_vals:
            return (0.0, 0.0)

        return (sum(A_vals) / len(A_vals), sum(S_vals) / len(S_vals))

    def detect_irony_delta(self, prev_state: SemanticState,
                           curr_state: SemanticState) -> float:
        """Detect change in irony level.

        Rising irony = therapist missed patient.

        Args:
            prev_state: Previous state
            curr_state: Current state

        Returns:
            Change in irony (positive = rising)
        """
        return curr_state.irony - prev_state.irony
