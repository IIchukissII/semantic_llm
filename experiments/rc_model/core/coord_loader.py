"""Coordinate loader for RC-Model.

Loads (A, S, τ) coordinates from derived_coordinates.json
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class WordCoord:
    """Semantic coordinates for a word."""
    word: str
    word_type: str  # noun, adj, verb
    A: float        # Affirmation axis
    S: float        # Sacred axis
    tau: float      # Abstraction level (n in the file)
    theta: float    # Phase angle
    r: float        # Magnitude
    variety: int    # Adjective variety (for nouns)

    @property
    def n(self) -> float:
        """Orbital level n (alias for tau)."""
        return self.tau

    def as_tuple(self) -> tuple[float, float, float]:
        """Return (A, S, τ) tuple for RC model."""
        return (self.A, self.S, self.tau)

    def as_quantum(self) -> tuple[float, float, float]:
        """Return (n, θ, r) quantum number tuple."""
        return (self.tau, self.theta, self.r)

    def as_array(self) -> np.ndarray:
        """Return numpy array [A, S, τ]."""
        return np.array([self.A, self.S, self.tau])


class CoordLoader:
    """Load and manage semantic coordinates."""

    DEFAULT_PATH = Path(__file__).parent.parent.parent / "meaning_chain/data/derived_coordinates.json"

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else self.DEFAULT_PATH
        self._coords: dict[str, WordCoord] = {}
        self._nouns: dict[str, WordCoord] = {}
        self._adjs: dict[str, WordCoord] = {}
        self._verbs: dict[str, WordCoord] = {}
        self._loaded = False

    def load(self) -> 'CoordLoader':
        """Load coordinates from JSON file."""
        if self._loaded:
            return self

        with open(self.path, 'r') as f:
            data = json.load(f)

        for word, info in data.get('coordinates', {}).items():
            coord = WordCoord(
                word=word,
                word_type=info.get('word_type', 'unknown'),
                A=info.get('A', 0.0),
                S=info.get('S', 0.0),
                tau=info.get('n', 0.0),  # n = τ (abstraction level)
                theta=info.get('theta', 0.0),
                r=info.get('r', 0.0),
                variety=info.get('variety', 0),
            )
            self._coords[word] = coord

            if coord.word_type == 'noun':
                self._nouns[word] = coord
            elif coord.word_type in ('adj', 'adjective'):
                self._adjs[word] = coord
            elif coord.word_type == 'verb':
                self._verbs[word] = coord

        self._loaded = True
        return self

    def get(self, word: str) -> Optional[WordCoord]:
        """Get coordinates for a word."""
        if not self._loaded:
            self.load()
        return self._coords.get(word.lower())

    def get_tuple(self, word: str) -> Optional[tuple[float, float, float]]:
        """Get (A, S, τ) tuple for a word."""
        coord = self.get(word)
        return coord.as_tuple() if coord else None

    def has(self, word: str) -> bool:
        """Check if word has coordinates."""
        if not self._loaded:
            self.load()
        return word.lower() in self._coords

    @property
    def nouns(self) -> dict[str, WordCoord]:
        """All noun coordinates."""
        if not self._loaded:
            self.load()
        return self._nouns

    @property
    def adjectives(self) -> dict[str, WordCoord]:
        """All adjective coordinates."""
        if not self._loaded:
            self.load()
        return self._adjs

    @property
    def verbs(self) -> dict[str, WordCoord]:
        """All verb coordinates."""
        if not self._loaded:
            self.load()
        return self._verbs

    def stats(self) -> dict:
        """Return statistics about loaded coordinates."""
        if not self._loaded:
            self.load()
        return {
            'total': len(self._coords),
            'nouns': len(self._nouns),
            'adjectives': len(self._adjs),
            'verbs': len(self._verbs),
            'path': str(self.path),
        }

    def coverage(self, words: list[str]) -> dict:
        """Check coverage of a word list."""
        if not self._loaded:
            self.load()

        found = [w for w in words if w.lower() in self._coords]
        missing = [w for w in words if w.lower() not in self._coords]

        return {
            'total': len(words),
            'found': len(found),
            'missing': len(missing),
            'coverage': len(found) / len(words) if words else 0,
            'missing_words': missing[:20],  # first 20 missing
        }


# Singleton instance for convenience
_default_loader: Optional[CoordLoader] = None

def get_coords(word: str) -> Optional[tuple[float, float, float]]:
    """Get coordinates for a word using default loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = CoordLoader().load()
    return _default_loader.get_tuple(word)

def get_loader() -> CoordLoader:
    """Get the default loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = CoordLoader().load()
    return _default_loader
