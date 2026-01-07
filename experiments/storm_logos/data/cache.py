"""Coordinate Cache.

Fast caching layer for coordinate lookups.
"""

from typing import Dict, Optional, Tuple
import json
from pathlib import Path

from .models import WordCoordinates


class CoordinateCache:
    """In-memory cache for word coordinates.

    Provides fast lookups with optional persistence.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self._cache: Dict[str, WordCoordinates] = {}
        self.cache_path = cache_path
        self._hits = 0
        self._misses = 0

        if cache_path and cache_path.exists():
            self.load()

    def get(self, word: str) -> Optional[WordCoordinates]:
        """Get coordinates from cache."""
        coords = self._cache.get(word.lower())
        if coords:
            self._hits += 1
        else:
            self._misses += 1
        return coords

    def set(self, word: str, coords: WordCoordinates):
        """Add coordinates to cache."""
        self._cache[word.lower()] = coords

    def has(self, word: str) -> bool:
        """Check if word is in cache."""
        return word.lower() in self._cache

    def get_coords(self, word: str) -> Optional[Tuple[float, float, float]]:
        """Get (A, S, τ) tuple from cache."""
        coords = self.get(word)
        if coords:
            return (coords.A, coords.S, coords.tau)
        return None

    def bulk_set(self, items: Dict[str, WordCoordinates]):
        """Add multiple items to cache."""
        for word, coords in items.items():
            self._cache[word.lower()] = coords

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Number of items in cache."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': self.size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
        }

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def save(self, path: Optional[Path] = None):
        """Save cache to disk."""
        path = path or self.cache_path
        if not path:
            raise ValueError("No cache path specified")

        data = {
            word: {
                'A': coords.A,
                'S': coords.S,
                'tau': coords.tau,
                'source': coords.source,
            }
            for word, coords in self._cache.items()
        }

        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: Optional[Path] = None):
        """Load cache from disk."""
        path = path or self.cache_path
        if not path or not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        for word, coords in data.items():
            self._cache[word] = WordCoordinates(
                word=word,
                A=coords['A'],
                S=coords['S'],
                tau=coords['tau'],
                source=coords.get('source', 'cache'),
            )
