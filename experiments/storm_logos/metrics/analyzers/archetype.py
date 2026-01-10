"""Archetype Analyzer: Detect Jungian archetypes in text.

Based on Carl Jung's archetypal theory, identifies patterns of:
- Shadow: repressed, unknown aspects of self
- Anima/Animus: contrasexual psyche
- Self: wholeness and integration
- Mother: nurturing/devouring maternal principle
- Father: authority, order, spiritual principle
- Hero: ego's journey toward individuation
- Trickster: agent of change, boundary-crossing
- Death/Rebirth: transformation through symbolic death

Patterns are loaded from config/archetypes.json - NOT hardcoded.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ...data.models import Bond, DreamState


@dataclass
class ArchetypePattern:
    """Pattern definition for an archetype."""
    name: str
    keywords: List[str]
    phrases: List[str]
    A_range: Tuple[float, float]
    S_range: Tuple[float, float]
    description: str


def _get_config_path() -> Path:
    """Get path to archetypes config file."""
    return Path(__file__).parent.parent.parent / "config" / "archetypes.json"


def load_archetypes_config(config_path: Optional[Path] = None) -> Dict:
    """Load archetype patterns from JSON config.

    Args:
        config_path: Optional custom path. Defaults to config/archetypes.json.

    Returns:
        Dict with 'archetypes' and 'dream_symbols' keys.
    """
    if config_path is None:
        config_path = _get_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Archetype config not found: {config_path}\n"
            "Create config/archetypes.json with archetype patterns."
        )

    with open(config_path) as f:
        return json.load(f)


class ArchetypeAnalyzer:
    """Detect Jungian archetypes in text and symbols.

    Uses keyword matching, phrase detection, and semantic coordinate
    alignment to identify archetypal patterns.

    All patterns are loaded from config/archetypes.json.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize analyzer with patterns from config.

        Args:
            config_path: Optional custom config path.
        """
        config = load_archetypes_config(config_path)

        self.archetypes: Dict[str, ArchetypePattern] = {}
        self.dream_symbols: Dict[str, Tuple[str, str]] = {}
        self._compiled_patterns: Dict[str, List] = {}

        # Load archetypes from config
        for name, data in config.get("archetypes", {}).items():
            self.archetypes[name] = ArchetypePattern(
                name=name,
                keywords=data.get("keywords", []),
                phrases=data.get("phrases", []),
                A_range=tuple(data.get("A_range", [-1.0, 1.0])),
                S_range=tuple(data.get("S_range", [-1.0, 1.0])),
                description=data.get("description", ""),
            )
            # Compile regex patterns
            self._compiled_patterns[name] = [
                re.compile(p, re.IGNORECASE) for p in data.get("phrases", [])
            ]

        # Load dream symbols from config
        for symbol, data in config.get("dream_symbols", {}).items():
            self.dream_symbols[symbol] = (
                data.get("archetype", ""),
                data.get("meaning", ""),
            )

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text for archetypal content.

        Args:
            text: Dream or text to analyze

        Returns:
            Dict mapping archetype names to scores (0-1)
        """
        text_lower = text.lower()
        scores = {}

        for name, pattern in self.archetypes.items():
            score = 0.0

            # Keyword matching
            keyword_matches = sum(1 for kw in pattern.keywords if kw in text_lower)
            score += min(keyword_matches * 0.1, 0.5)  # Cap at 0.5

            # Phrase matching
            phrase_matches = sum(
                1 for p in self._compiled_patterns.get(name, [])
                if p.search(text)
            )
            score += min(phrase_matches * 0.2, 0.5)  # Cap at 0.5

            scores[name] = min(score, 1.0)

        return scores

    def analyze_symbols(self, symbols: List[Bond]) -> Dict[str, float]:
        """Analyze extracted symbols for archetypal content.

        Args:
            symbols: List of bonds extracted from dream

        Returns:
            Dict mapping archetype names to scores
        """
        scores = {name: 0.0 for name in self.archetypes}

        for symbol in symbols:
            text = symbol.text.lower()

            # Check known dream symbols
            for sym_word, (archetype, _) in self.dream_symbols.items():
                if sym_word in text:
                    scores[archetype] += 0.3

            # Check coordinate alignment with archetype ranges
            for name, pattern in self.archetypes.items():
                a_min, a_max = pattern.A_range
                s_min, s_max = pattern.S_range

                # Score based on coordinate fit
                if a_min <= symbol.A <= a_max and s_min <= symbol.S <= s_max:
                    scores[name] += 0.15

                # Keyword match in symbol
                for kw in pattern.keywords:
                    if kw in text:
                        scores[name] += 0.2

        # Normalize scores
        for name in scores:
            scores[name] = min(scores[name], 1.0)

        return scores

    def get_symbol_interpretation(self, symbol: Bond) -> Tuple[str, str]:
        """Get archetype and interpretation for a specific symbol.

        Args:
            symbol: Bond to interpret

        Returns:
            (archetype_name, interpretation_text)
        """
        text = symbol.text.lower()

        # Check known symbols first
        for sym_word, (archetype, interpretation) in self.dream_symbols.items():
            if sym_word in text:
                return (archetype, interpretation)

        # Check keyword matches
        best_match = ("", "")
        best_score = 0

        for name, pattern in self.archetypes.items():
            score = sum(1 for kw in pattern.keywords if kw in text)

            # Add coordinate alignment
            a_min, a_max = pattern.A_range
            s_min, s_max = pattern.S_range
            if a_min <= symbol.A <= a_max and s_min <= symbol.S <= s_max:
                score += 1

            if score > best_score:
                best_score = score
                best_match = (name, pattern.description)

        return best_match if best_score >= 2 else ("", "")

    def create_dream_state(self, text: str, symbols: List[Bond]) -> DreamState:
        """Create a DreamState from text and symbols.

        Args:
            text: Dream narrative
            symbols: Extracted symbols

        Returns:
            DreamState with archetype scores and coordinates
        """
        # Analyze archetypes from both text and symbols
        text_scores = self.analyze_text(text)
        symbol_scores = self.analyze_symbols(symbols)

        # Combine scores (average)
        combined = {}
        for name in self.archetypes:
            combined[name] = (text_scores.get(name, 0) + symbol_scores.get(name, 0)) / 2

        # Calculate average coordinates from symbols
        if symbols:
            avg_A = sum(s.A for s in symbols) / len(symbols)
            avg_S = sum(s.S for s in symbols) / len(symbols)
            avg_tau = sum(s.tau for s in symbols) / len(symbols)
        else:
            avg_A, avg_S, avg_tau = 0.0, 0.0, 2.5

        # Detect dream-specific markers
        text_lower = text.lower()
        transformation = 0.3 if any(w in text_lower for w in
            ["transform", "change", "became", "turned into", "morph"]) else 0.0
        journey = 0.3 if any(w in text_lower for w in
            ["journey", "travel", "walk", "path", "road", "quest"]) else 0.0
        confrontation = 0.3 if any(w in text_lower for w in
            ["face", "confront", "fight", "battle", "challenge"]) else 0.0

        return DreamState(
            A=avg_A,
            S=avg_S,
            tau=avg_tau,
            shadow=combined.get("shadow", 0),
            anima_animus=combined.get("anima_animus", 0),
            self_archetype=combined.get("self", 0),
            mother=combined.get("mother", 0),
            father=combined.get("father", 0),
            hero=combined.get("hero", 0),
            trickster=combined.get("trickster", 0),
            death_rebirth=combined.get("death_rebirth", 0),
            transformation=transformation,
            journey=journey,
            confrontation=confrontation,
        )

    def reload_config(self, config_path: Optional[Path] = None):
        """Reload patterns from config file.

        Useful for hot-reloading after config edits.
        """
        self.__init__(config_path)


# Singleton instance
_analyzer_instance: Optional[ArchetypeAnalyzer] = None


def get_archetype_analyzer(config_path: Optional[Path] = None) -> ArchetypeAnalyzer:
    """Get singleton ArchetypeAnalyzer instance.

    Args:
        config_path: Optional custom config path. Only used on first call.
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ArchetypeAnalyzer(config_path)
    return _analyzer_instance
