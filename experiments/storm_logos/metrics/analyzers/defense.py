"""Defense Analyzer: Detect psychological defense mechanisms."""

import re
from typing import List, Optional

from ...data.models import SemanticState


class DefenseAnalyzer:
    """Detect psychological defense mechanisms.

    Defense mechanisms:
    - Irony as distance (creates emotional distance)
    - Sarcasm as aggression (attacks while denying)
    - Intellectualization (flees to abstraction)
    - Negation (denies positive, stays in negative)
    - Devaluation (reduces meaning/sacred)
    - Minimization (downplays significance)
    - Projection (blames others)
    - Rationalization (over-explains)
    - Humor defense (uses humor to avoid)
    """

    # Thresholds for defense detection
    IRONY_THRESHOLD = 0.3
    SARCASM_THRESHOLD = 0.3
    INTELLECTUALIZATION_TAU = 3.0
    NEGATION_A = -0.3
    DEVALUATION_S = -0.3

    # Pattern-based detection
    MINIMIZATION_PATTERNS = [
        r'\b(just|only|merely|simply|a little|kind of|sort of)\b',
        r'\b(not that bad|could be worse|no big deal)\b',
        r'\b(whatever|anyway|i guess|i suppose)\b',
    ]

    DEFLECTION_PATTERNS = [
        r'\b(anyway|but anyway|moving on|let\'s talk about)\b',
        r'\b(what about|how about|speaking of)\b',
        r'\b(that\'s not important|doesn\'t matter|forget it)\b',
    ]

    PROJECTION_PATTERNS = [
        r'\b(they|he|she|everyone|people)\b.*\b(always|never|fault)\b',
        r'\b(it\'s their|it\'s his|it\'s her|they made me)\b',
        r'\b(because of them|thanks to them|they\'re the ones)\b',
    ]

    RATIONALIZATION_PATTERNS = [
        r'\b(because|since|therefore|that\'s why|the reason)\b',
        r'\b(logically|rationally|objectively|technically)\b',
        r'\b(makes sense|understandable|reasonable)\b',
    ]

    VULNERABILITY_PATTERNS = [
        r'\b(honestly|truthfully|to be honest|the truth is)\b',
        r'\b(i feel|i felt|feeling|scared|afraid|hurt)\b',
        r'\b(vulnerable|raw|exposed|open|admit)\b',
        r'\*[^*]+\*',  # Asterisk actions like *sighs*
    ]

    def analyze(self, state: SemanticState = None,
                text: str = None) -> List[str]:
        """Detect active defenses.

        Args:
            state: Current semantic state
            text: Text to analyze

        Returns:
            List of detected defense names
        """
        defenses = []

        if state:
            defenses.extend(self._detect_from_state(state))

        if text:
            defenses.extend(self._detect_from_text(text))

        return list(set(defenses))  # Deduplicate

    def _detect_from_state(self, state: SemanticState) -> List[str]:
        """Detect defenses from semantic state."""
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

        # Extended markers if available
        if state.minimization > 0.3:
            defenses.append('minimization')

        if state.deflection > 0.3:
            defenses.append('deflection')

        if state.projection > 0.3:
            defenses.append('projection')

        if state.rationalization > 0.3:
            defenses.append('rationalization')

        if state.humor_defense > 0.3:
            defenses.append('humor_defense')

        if state.self_deprecation > 0.3:
            defenses.append('self_deprecation')

        return defenses

    def _detect_from_text(self, text: str) -> List[str]:
        """Detect defenses from text patterns."""
        defenses = []
        text_lower = text.lower()

        if self._pattern_score(text_lower, self.MINIMIZATION_PATTERNS) > 0.3:
            defenses.append('minimization')

        if self._pattern_score(text_lower, self.DEFLECTION_PATTERNS) > 0.3:
            defenses.append('deflection')

        if self._pattern_score(text_lower, self.PROJECTION_PATTERNS) > 0.3:
            defenses.append('projection')

        if self._pattern_score(text_lower, self.RATIONALIZATION_PATTERNS) > 0.3:
            defenses.append('rationalization')

        return defenses

    def _pattern_score(self, text: str, patterns: List[str]) -> float:
        """Score text against patterns."""
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            score += 0.2 * len(matches)
        return min(score, 1.0)

    def detect_vulnerability(self, text: str) -> float:
        """Detect vulnerability (openness, not defense).

        Args:
            text: Text to analyze

        Returns:
            Vulnerability score (0 to 1)
        """
        return self._pattern_score(text.lower(), self.VULNERABILITY_PATTERNS)

    def get_defense_description(self, defense: str) -> str:
        """Get human-readable description of defense."""
        descriptions = {
            'irony_as_distance': 'Using irony to create emotional distance',
            'sarcasm_as_aggression': 'Aggressive expression disguised as humor',
            'intellectualization': 'Fleeing to abstraction to avoid feelings',
            'negation': 'Denying positive possibilities, staying negative',
            'devaluation': 'Reducing meaning and significance',
            'minimization': 'Downplaying the importance of feelings',
            'deflection': 'Changing subject to avoid difficult topics',
            'projection': 'Attributing own feelings to others',
            'rationalization': 'Over-explaining to avoid emotional truth',
            'humor_defense': 'Using humor to deflect from pain',
            'self_deprecation': 'Attacking self before others can',
        }
        return descriptions.get(defense, defense)
