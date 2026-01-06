"""Semantic Detector: Detect position and markers from text.

Analyzes text to determine:
- Position in (A, S, τ) space
- Irony and sarcasm markers
- Primary emotion and intensity
"""

import re
from typing import List, Tuple

from ..core import SemanticState, SemanticData


class SemanticDetector:
    """Detect semantic position and emotional markers from text."""

    # Irony patterns: contradiction between surface and meaning
    IRONY_PATTERNS = [
        r'\b(obviously|clearly|surely|of course)\b',  # False certainty
        r'\b(wonderful|great|fantastic|brilliant)\b.*\b(problem|issue|fail)',
        r'\b(just|only|merely|simply)\b',  # Minimizers
        r'\.{3}',  # Ellipsis (hesitation)
        r'\?$',  # Rhetorical questions
    ]

    # Sarcasm amplifiers: exaggeration markers
    SARCASM_AMPLIFIERS = [
        r'\b(so|very|really|absolutely|totally|completely)\b',
        r'\b(always|never|everyone|nobody)\b',  # Absolutes
        r'!+',  # Exclamation excess
    ]

    # Emotion lexicon (core words)
    EMOTION_WORDS = {
        'joy': ['happy', 'joy', 'love', 'wonderful', 'great', 'beautiful',
                'delight', 'pleasure', 'bliss', 'ecstasy'],
        'sadness': ['sad', 'sorry', 'grief', 'loss', 'miss', 'lonely',
                    'melancholy', 'despair', 'sorrow', 'mourn'],
        'anger': ['angry', 'hate', 'furious', 'rage', 'annoyed',
                  'irritated', 'hostile', 'bitter', 'resentful'],
        'fear': ['afraid', 'scared', 'worry', 'anxious', 'terror',
                 'dread', 'panic', 'nervous', 'uneasy'],
        'surprise': ['surprised', 'shocked', 'unexpected', 'sudden',
                     'astonished', 'amazed', 'startled'],
        'disgust': ['disgusting', 'horrible', 'awful', 'terrible',
                    'revolting', 'repulsive', 'vile'],
    }

    def __init__(self, data: SemanticData):
        self.data = data

    def detect(self, text: str) -> SemanticState:
        """Analyze text and return semantic state with markers."""
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # 1. Base coordinates from known words
        A, S, tau = self._compute_coordinates(words)

        # 2. Detect irony
        irony = self._detect_irony(text, A, S)

        # 3. Detect sarcasm
        sarcasm = self._detect_sarcasm(text, A)

        # 4. Detect emotion
        emotion, intensity = self._detect_emotion(words)

        return SemanticState(
            A=A, S=S, tau=tau,
            irony=irony, sarcasm=sarcasm,
            emotion=emotion, intensity=intensity
        )

    def _compute_coordinates(self, words: List[str]) -> Tuple[float, float, float]:
        """Compute (A, S, τ) from word coordinates."""
        A_vals, S_vals, tau_vals = [], [], []

        for word in words:
            coords = self.data.get(word)
            if coords:
                A_vals.append(coords.A)
                S_vals.append(coords.S)
                tau_vals.append(coords.tau)

        if not A_vals:
            return (0.0, 0.0, 2.5)

        return (
            sum(A_vals) / len(A_vals),
            sum(S_vals) / len(S_vals),
            sum(tau_vals) / len(tau_vals),
        )

    def _detect_irony(self, text: str, A: float, S: float) -> float:
        """Detect irony from semantic contradictions."""
        score = 0.0

        # Pattern matching
        for pattern in self.IRONY_PATTERNS:
            if re.search(pattern, text.lower()):
                score += 0.15

        # Semantic contradiction: positive words + negative context
        if A > 0.2 and S < -0.1:
            score += 0.2

        # Question at end often signals rhetorical/ironic
        if text.strip().endswith('?'):
            score += 0.1

        return min(score, 1.0)

    def _detect_sarcasm(self, text: str, A: float) -> float:
        """Detect sarcasm from exaggeration patterns."""
        score = 0.0

        # Amplifiers + high apparent positivity
        amplifier_count = 0
        for pattern in self.SARCASM_AMPLIFIERS:
            amplifier_count += len(re.findall(pattern, text.lower()))

        if amplifier_count > 0 and A > 0.3:
            score += 0.15 * amplifier_count

        # ALL CAPS sections
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            score += 0.2

        return min(score, 1.0)

    def _detect_emotion(self, words: List[str]) -> Tuple[str, float]:
        """Detect primary emotion and intensity."""
        emotion_scores = {e: 0 for e in self.EMOTION_WORDS}

        for word in words:
            for emotion, lexicon in self.EMOTION_WORDS.items():
                if word in lexicon:
                    emotion_scores[emotion] += 1

        if not any(emotion_scores.values()):
            return ("neutral", 0.0)

        primary = max(emotion_scores, key=emotion_scores.get)
        intensity = min(emotion_scores[primary] / 3.0, 1.0)

        return (primary, intensity)
