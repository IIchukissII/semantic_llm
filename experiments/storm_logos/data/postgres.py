"""PostgreSQL Data Layer.

Connection to PostgreSQL for bonds and word coordinates.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psycopg2

from .models import Bond, WordCoordinates
from ..config import get_config, DatabaseConfig


class PostgresData:
    """Central repository for semantic data from PostgreSQL.

    Loads coordinates and bonds, provides clean access.
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None,
                 load_bonds: bool = False):
        self.config = db_config or get_config().db
        self._coordinates: Dict[str, WordCoordinates] = {}
        self._bonds: List[Bond] = []
        self._nouns: Dict[str, WordCoordinates] = {}
        self._adjectives: Dict[str, WordCoordinates] = {}
        self._loaded = False
        self._bonds_loaded = False

        self.load_coordinates()
        if load_bonds:
            self.load_bonds()

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def coordinates(self) -> Dict[str, WordCoordinates]:
        """All word coordinates."""
        return self._coordinates

    @property
    def bonds(self) -> List[Bond]:
        """All loaded bonds."""
        return self._bonds

    @property
    def nouns(self) -> Dict[str, WordCoordinates]:
        """All nouns with coordinates."""
        return self._nouns

    @property
    def adjectives(self) -> Dict[str, WordCoordinates]:
        """All adjectives with coordinates."""
        return self._adjectives

    @property
    def n_coordinates(self) -> int:
        return len(self._coordinates)

    @property
    def n_bonds(self) -> int:
        return len(self._bonds)

    @property
    def n_nouns(self) -> int:
        return len(self._nouns)

    # ========================================================================
    # LOADERS
    # ========================================================================

    def load_coordinates(self) -> int:
        """Load all word coordinates from all sources."""
        if self._loaded:
            return self.n_coordinates

        # 1. Load from JSON (primary source with A, S, τ)
        self._load_from_json()

        # 2. Derive from noun_abstraction (95K nouns with τ)
        self._load_from_abstraction()

        # 3. Separate into nouns and adjectives
        self._categorize_words()

        self._loaded = True
        return self.n_coordinates

    def _load_from_json(self):
        """Load coordinates from derived_coordinates.json."""
        config = get_config()
        coord_path = config.data_path / "derived_coordinates.json"

        try:
            with open(coord_path) as f:
                data = json.load(f)

            for word, coords in data.get('coordinates', {}).items():
                if isinstance(coords, dict):
                    self._coordinates[word] = WordCoordinates(
                        word=word,
                        A=coords.get('A', 0.0),
                        S=coords.get('S', 0.0),
                        tau=coords.get('n', 2.5),
                        source='json'
                    )
                else:
                    self._coordinates[word] = WordCoordinates(
                        word=word, A=coords[0], S=coords[1], tau=coords[2],
                        source='json'
                    )

            print(f"  JSON: {len(self._coordinates):,} words")

        except Exception as e:
            print(f"  Warning (JSON): {e}")

    def _load_from_abstraction(self):
        """Load/derive coordinates from noun_abstraction table."""
        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            cur.execute('''
                SELECT n.noun_text, na.adj_variety
                FROM noun_abstraction na
                JOIN nouns n ON na.noun_id = n.noun_id
            ''')

            derived = 0
            for noun, variety in cur.fetchall():
                word = noun.lower()
                if word in self._coordinates:
                    continue

                # Derive τ from variety
                if variety and variety > 0:
                    tau = 4.0 - 3.0 * math.log(variety + 1) / math.log(1000)
                    tau = max(0.5, min(4.5, tau))
                else:
                    tau = 3.0

                self._coordinates[word] = WordCoordinates(
                    word=word, A=0.0, S=0.0, tau=tau,
                    source='abstraction'
                )
                derived += 1

            conn.close()
            print(f"  Abstraction: {derived:,} nouns derived")

        except Exception as e:
            print(f"  Warning (DB): {e}")

    def _categorize_words(self):
        """Separate coordinates into nouns and adjectives."""
        for word, coords in self._coordinates.items():
            if coords.source == 'abstraction':
                self._nouns[word] = coords
            else:
                # JSON words could be either
                self._nouns[word] = coords
                self._adjectives[word] = coords

    def load_bonds(self, limit: int = 500000) -> int:
        """Load bonds from hyp_bond_vocab."""
        if self._bonds_loaded:
            return self.n_bonds

        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            cur.execute('''
                SELECT bond, total_count
                FROM hyp_bond_vocab
                WHERE total_count >= 3
                ORDER BY total_count DESC
                LIMIT %s
            ''', (limit,))

            for bond_str, variety in cur.fetchall():
                parts = bond_str.split('|')
                if len(parts) != 2:
                    continue

                adj, noun = parts[0].lower(), parts[1].lower()

                # Skip non-English
                if not self._is_english(adj) or not self._is_english(noun):
                    continue

                # Get coordinates from noun
                if noun in self._coordinates:
                    coords = self._coordinates[noun]
                    self._bonds.append(Bond(
                        noun=noun,
                        adj=adj,
                        variety=variety,
                        A=coords.A,
                        S=coords.S,
                        tau=coords.tau,
                    ))

            conn.close()
            self._bonds_loaded = True
            print(f"  Bonds: {len(self._bonds):,} loaded")

        except Exception as e:
            print(f"  Warning (Bonds): {e}")

        return self.n_bonds

    @staticmethod
    def _is_english(word: str) -> bool:
        """Check if word is likely English."""
        if not word:
            return False
        return all(c.isascii() and (c.isalpha() or c in "-'") for c in word)

    # ========================================================================
    # LOOKUPS
    # ========================================================================

    def get(self, word: str) -> Optional[WordCoordinates]:
        """Get coordinates for a word."""
        return self._coordinates.get(word.lower())

    def get_coords(self, word: str) -> Tuple[float, float, float]:
        """Get (A, S, τ) tuple for a word."""
        coords = self.get(word)
        if coords:
            return (coords.A, coords.S, coords.tau)
        return (0.0, 0.0, 2.5)

    def has(self, word: str) -> bool:
        """Check if word has coordinates."""
        return word.lower() in self._coordinates

    def get_bond(self, adj: str, noun: str) -> Optional[Bond]:
        """Get a specific bond by adj+noun."""
        for bond in self._bonds:
            if bond.adj == adj.lower() and bond.noun == noun.lower():
                return bond
        return None

    def get_bonds_for_noun(self, noun: str) -> List[Bond]:
        """Get all bonds containing a noun."""
        noun_lower = noun.lower()
        return [b for b in self._bonds if b.noun == noun_lower]

    def get_neighbors(self, A: float, S: float, tau: float,
                      radius: float = 0.5) -> List[Bond]:
        """Get bonds within radius of (A, S, τ)."""
        neighbors = []
        for bond in self._bonds:
            dist = math.sqrt(
                (bond.A - A)**2 +
                (bond.S - S)**2 +
                (bond.tau - tau)**2
            )
            if dist <= radius:
                neighbors.append(bond)
        return neighbors

    def stats(self) -> Dict:
        """Return statistics about loaded data."""
        return {
            'n_coordinates': self.n_coordinates,
            'n_nouns': self.n_nouns,
            'n_bonds': self.n_bonds,
            'loaded': self._loaded,
            'bonds_loaded': self._bonds_loaded,
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_data_instance: Optional[PostgresData] = None


def get_data(load_bonds: bool = False) -> PostgresData:
    """Get the singleton PostgresData instance."""
    global _data_instance
    if _data_instance is None:
        print("Loading semantic data...")
        _data_instance = PostgresData(load_bonds=load_bonds)
        print(f"Total: {_data_instance.n_coordinates:,} coordinates")
    return _data_instance


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    data = get_data()
    print(f"\nStats: {data.stats()}")

    # Test lookups
    test_words = ['love', 'death', 'coffee', 'existential', 'meeting']
    print("\nSample coordinates:")
    for word in test_words:
        coords = data.get(word)
        if coords:
            print(f"  {word}: A={coords.A:+.2f}, S={coords.S:+.2f}, τ={coords.tau:.2f} ({coords.source})")
        else:
            print(f"  {word}: not found")
