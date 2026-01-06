"""Semantic Detector: Detect position and markers from text.

Analyzes text to determine:
- Position in (A, S, τ) space
- Multiple psychological markers (irony, sarcasm, vulnerability, deflection, etc.)
- Primary emotion and intensity
- Defense mechanisms
"""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from ..core import SemanticState, SemanticData


@dataclass
class ExtendedState(SemanticState):
    """Extended semantic state with additional markers."""
    # Defense markers (0-1 each)
    vulnerability: float = 0.0      # Openness, rawness
    deflection: float = 0.0         # Changing subject, avoiding
    self_deprecation: float = 0.0   # Self-criticism, self-attack
    minimization: float = 0.0       # Downplaying, "just", "only"
    projection: float = 0.0         # Blaming others
    rationalization: float = 0.0    # Over-explaining, justifying
    humor_defense: float = 0.0      # Using humor to avoid


class SemanticDetector:
    """Detect semantic position and psychological markers from text."""

    # Irony patterns
    IRONY_PATTERNS = [
        r'\b(obviously|clearly|surely|of course)\b',
        r'\b(wonderful|great|fantastic|brilliant)\b.*\b(problem|issue|fail)',
        r'\.{3}',  # Ellipsis
        r'\?$',  # Rhetorical questions
    ]

    # Sarcasm amplifiers
    SARCASM_AMPLIFIERS = [
        r'\b(so|very|really|absolutely|totally|completely)\b',
        r'\b(always|never|everyone|nobody)\b',
        r'!+',
    ]

    # Minimization patterns
    MINIMIZATION_PATTERNS = [
        r'\b(just|only|merely|simply|a little|kind of|sort of)\b',
        r'\b(not that bad|could be worse|no big deal)\b',
        r'\b(whatever|anyway|i guess|i suppose)\b',
    ]

    # Vulnerability patterns
    VULNERABILITY_PATTERNS = [
        r'\b(honestly|truthfully|to be honest|the truth is)\b',
        r'\b(i feel|i felt|feeling|scared|afraid|hurt)\b',
        r'\b(vulnerable|raw|exposed|open|admit)\b',
        r'\*[^*]+\*',  # Asterisk actions like *sighs*
    ]

    # Self-deprecation patterns
    SELF_DEPRECATION_PATTERNS = [
        r'\b(stupid|idiot|pathetic|worthless|useless)\b.*\b(i|me|my)\b',
        r'\b(i|me)\b.*\b(stupid|idiot|pathetic|worthless|failure)\b',
        r'\b(i can\'t|i couldn\'t|i failed|my fault|i suck)\b',
        r'\b(who am i to|what do i know)\b',
    ]

    # Deflection patterns
    DEFLECTION_PATTERNS = [
        r'\b(anyway|but anyway|moving on|let\'s talk about)\b',
        r'\b(what about|how about|speaking of)\b',
        r'\b(that\'s not important|doesn\'t matter|forget it)\b',
    ]

    # Projection patterns
    PROJECTION_PATTERNS = [
        r'\b(they|he|she|everyone|people)\b.*\b(always|never|fault)\b',
        r'\b(it\'s their|it\'s his|it\'s her|they made me)\b',
        r'\b(because of them|thanks to them|they\'re the ones)\b',
    ]

    # Rationalization patterns
    RATIONALIZATION_PATTERNS = [
        r'\b(because|since|therefore|that\'s why|the reason)\b',
        r'\b(logically|rationally|objectively|technically)\b',
        r'\b(makes sense|understandable|reasonable)\b',
    ]

    # Humor defense patterns
    HUMOR_DEFENSE_PATTERNS = [
        r'\b(haha|lol|lmao|joke|kidding|joking)\b',
        r'\*laughs?\*|\*chuckles?\*|\*snorts?\*',
        r'\b(funny|hilarious|comedy|ridiculous)\b',
    ]

    # Emotion lexicon
    EMOTION_WORDS = {
        'joy': ['happy', 'joy', 'love', 'wonderful', 'great', 'beautiful',
                'delight', 'pleasure', 'bliss', 'grateful', 'hopeful'],
        'sadness': ['sad', 'sorry', 'grief', 'loss', 'miss', 'lonely',
                    'melancholy', 'despair', 'sorrow', 'empty', 'hollow'],
        'anger': ['angry', 'hate', 'furious', 'rage', 'annoyed',
                  'irritated', 'hostile', 'bitter', 'resentful', 'frustrated'],
        'fear': ['afraid', 'scared', 'worry', 'anxious', 'terror',
                 'dread', 'panic', 'nervous', 'uneasy', 'terrified'],
        'shame': ['ashamed', 'embarrassed', 'humiliated', 'guilty',
                  'pathetic', 'worthless', 'failure', 'disgrace'],
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

        # 1. Base coordinates
        A, S, tau = self._compute_coordinates(words)

        # 2. Detect all markers
        irony = self._detect_irony(text, A, S)
        sarcasm = self._detect_sarcasm(text, A)
        emotion, intensity = self._detect_emotion(words)

        # Base state
        state = SemanticState(
            A=A, S=S, tau=tau,
            irony=irony, sarcasm=sarcasm,
            emotion=emotion, intensity=intensity
        )

        return state

    def detect_extended(self, text: str) -> ExtendedState:
        """Full detection with all psychological markers."""
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Base coordinates
        A, S, tau = self._compute_coordinates(words)

        # All markers
        irony = self._detect_irony(text, A, S)
        sarcasm = self._detect_sarcasm(text, A)
        emotion, intensity = self._detect_emotion(words)

        # Extended markers
        vulnerability = self._detect_pattern(text, self.VULNERABILITY_PATTERNS)
        deflection = self._detect_pattern(text, self.DEFLECTION_PATTERNS)
        self_deprecation = self._detect_pattern(text, self.SELF_DEPRECATION_PATTERNS)
        minimization = self._detect_pattern(text, self.MINIMIZATION_PATTERNS)
        projection = self._detect_pattern(text, self.PROJECTION_PATTERNS)
        rationalization = self._detect_pattern(text, self.RATIONALIZATION_PATTERNS)
        humor_defense = self._detect_pattern(text, self.HUMOR_DEFENSE_PATTERNS)

        return ExtendedState(
            A=A, S=S, tau=tau,
            irony=irony, sarcasm=sarcasm,
            emotion=emotion, intensity=intensity,
            vulnerability=vulnerability,
            deflection=deflection,
            self_deprecation=self_deprecation,
            minimization=minimization,
            projection=projection,
            rationalization=rationalization,
            humor_defense=humor_defense,
        )

    def _detect_pattern(self, text: str, patterns: List[str]) -> float:
        """Detect patterns and return score 0-1."""
        score = 0.0
        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            score += 0.2 * len(matches)
        return min(score, 1.0)

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
