"""PostgreSQL Data Layer.

Connection to PostgreSQL for bonds and word coordinates.

Extended for runtime learning:
    - learned_bonds table for user-learned bonds
    - learned_words table for user-learned word coordinates
    - Methods for storing and retrieving learned content

Learning Flow:
    1. User input → spaCy extracts bonds
    2. Lookup coordinates (existing or estimate)
    3. Store in learned_bonds table
    4. Sync to Neo4j for trajectory

Tables:
    learned_bonds:  id, adj, noun, A, S, tau, source, confidence,
                    created_at, last_used, use_count
    learned_words:  word, A, S, tau, source, confidence,
                    created_at, last_used
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import psycopg2
from psycopg2 import sql

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

    # ========================================================================
    # LEARNING: Tables and Initialization
    # ========================================================================

    def init_learning_tables(self) -> bool:
        """Create tables for learned bonds and words if they don't exist.

        Tables:
            learned_bonds: Stores user-learned adj+noun bonds with coordinates
            learned_words: Stores user-learned word coordinates

        Returns:
            True if tables were created/verified
        """
        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            # Create learned_bonds table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS learned_bonds (
                    id SERIAL PRIMARY KEY,
                    adj VARCHAR(100) NOT NULL,
                    noun VARCHAR(100) NOT NULL,
                    A FLOAT DEFAULT 0.0,
                    S FLOAT DEFAULT 0.0,
                    tau FLOAT DEFAULT 2.5,
                    source VARCHAR(50) DEFAULT 'conversation',
                    confidence FLOAT DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_used TIMESTAMP DEFAULT NOW(),
                    use_count INTEGER DEFAULT 1,
                    UNIQUE(adj, noun)
                )
            ''')

            # Create learned_words table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS learned_words (
                    word VARCHAR(100) PRIMARY KEY,
                    A FLOAT DEFAULT 0.0,
                    S FLOAT DEFAULT 0.0,
                    tau FLOAT DEFAULT 2.5,
                    source VARCHAR(50) DEFAULT 'conversation',
                    confidence FLOAT DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_used TIMESTAMP DEFAULT NOW()
                )
            ''')

            # Create indexes for faster lookups
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_learned_bonds_adj_noun
                ON learned_bonds(adj, noun)
            ''')
            cur.execute('''
                CREATE INDEX IF NOT EXISTS idx_learned_bonds_last_used
                ON learned_bonds(last_used)
            ''')

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error creating learning tables: {e}")
            return False

    # ========================================================================
    # LEARNING: Bond Operations
    # ========================================================================

    def learn_bond(self, adj: str, noun: str,
                   A: float = None, S: float = None, tau: float = None,
                   source: str = 'conversation',
                   confidence: float = 0.5) -> Optional[Bond]:
        """Learn a new bond or reinforce an existing one.

        If bond exists, increments use_count and updates last_used.
        If bond is new, computes coordinates and stores it.

        Args:
            adj: Adjective
            noun: Noun
            A: Affirmation coordinate (computed if None)
            S: Sacred coordinate (computed if None)
            tau: Abstraction level (computed if None)
            source: Source type ('conversation', 'context')
            confidence: Confidence in coordinates [0-1]

        Returns:
            Bond with coordinates, or None on error
        """
        adj = adj.lower().strip()
        noun = noun.lower().strip()

        # Compute coordinates if not provided
        if A is None or S is None or tau is None:
            coords = self._compute_bond_coordinates(adj, noun)
            A = A if A is not None else coords[0]
            S = S if S is not None else coords[1]
            tau = tau if tau is not None else coords[2]

        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            # Upsert: insert or update on conflict
            cur.execute('''
                INSERT INTO learned_bonds (adj, noun, A, S, tau, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (adj, noun) DO UPDATE SET
                    last_used = NOW(),
                    use_count = learned_bonds.use_count + 1,
                    confidence = GREATEST(learned_bonds.confidence, EXCLUDED.confidence)
                RETURNING id, A, S, tau, use_count
            ''', (adj, noun, A, S, tau, source, confidence))

            row = cur.fetchone()
            conn.commit()
            conn.close()

            return Bond(
                adj=adj,
                noun=noun,
                A=row[1],
                S=row[2],
                tau=row[3],
                variety=row[4],  # use_count as variety
            )

        except Exception as e:
            print(f"Error learning bond: {e}")
            return None

    def get_learned_bond(self, adj: str, noun: str) -> Optional[Bond]:
        """Get a learned bond by adj+noun.

        Args:
            adj: Adjective
            noun: Noun

        Returns:
            Bond if found, None otherwise
        """
        adj = adj.lower().strip()
        noun = noun.lower().strip()

        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            cur.execute('''
                SELECT adj, noun, A, S, tau, use_count
                FROM learned_bonds
                WHERE adj = %s AND noun = %s
            ''', (adj, noun))

            row = cur.fetchone()
            conn.close()

            if row:
                return Bond(
                    adj=row[0],
                    noun=row[1],
                    A=row[2],
                    S=row[3],
                    tau=row[4],
                    variety=row[5],
                )
            return None

        except Exception as e:
            print(f"Error getting learned bond: {e}")
            return None

    def get_all_learned_bonds(self, limit: int = 1000,
                               min_use_count: int = 1) -> List[Bond]:
        """Get all learned bonds.

        Args:
            limit: Maximum bonds to return
            min_use_count: Minimum use count filter

        Returns:
            List of learned bonds
        """
        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            cur.execute('''
                SELECT adj, noun, A, S, tau, use_count
                FROM learned_bonds
                WHERE use_count >= %s
                ORDER BY use_count DESC, last_used DESC
                LIMIT %s
            ''', (min_use_count, limit))

            bonds = []
            for row in cur.fetchall():
                bonds.append(Bond(
                    adj=row[0],
                    noun=row[1],
                    A=row[2],
                    S=row[3],
                    tau=row[4],
                    variety=row[5],
                ))

            conn.close()
            return bonds

        except Exception as e:
            print(f"Error getting learned bonds: {e}")
            return []

    def mark_bond_used(self, adj: str, noun: str) -> bool:
        """Mark a learned bond as used (updates last_used and use_count).

        Args:
            adj: Adjective
            noun: Noun

        Returns:
            True if updated
        """
        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            cur.execute('''
                UPDATE learned_bonds
                SET last_used = NOW(), use_count = use_count + 1
                WHERE adj = %s AND noun = %s
            ''', (adj.lower(), noun.lower()))

            updated = cur.rowcount > 0
            conn.commit()
            conn.close()
            return updated

        except Exception as e:
            print(f"Error marking bond used: {e}")
            return False

    # ========================================================================
    # LEARNING: Word Operations
    # ========================================================================

    def learn_word(self, word: str,
                   A: float = 0.0, S: float = 0.0, tau: float = 2.5,
                   source: str = 'conversation',
                   confidence: float = 0.5) -> Optional[WordCoordinates]:
        """Learn coordinates for a new word.

        Args:
            word: The word to learn
            A: Affirmation coordinate
            S: Sacred coordinate
            tau: Abstraction level
            source: Source type
            confidence: Confidence in coordinates

        Returns:
            WordCoordinates if successful
        """
        word = word.lower().strip()

        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            cur.execute('''
                INSERT INTO learned_words (word, A, S, tau, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (word) DO UPDATE SET
                    last_used = NOW(),
                    confidence = GREATEST(learned_words.confidence, EXCLUDED.confidence)
                RETURNING A, S, tau
            ''', (word, A, S, tau, source, confidence))

            row = cur.fetchone()
            conn.commit()
            conn.close()

            coords = WordCoordinates(
                word=word,
                A=row[0],
                S=row[1],
                tau=row[2],
                source='learned'
            )

            # Also add to in-memory cache
            self._coordinates[word] = coords
            return coords

        except Exception as e:
            print(f"Error learning word: {e}")
            return None

    def get_learned_word(self, word: str) -> Optional[WordCoordinates]:
        """Get learned coordinates for a word.

        Args:
            word: The word to lookup

        Returns:
            WordCoordinates if found
        """
        word = word.lower().strip()

        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            cur.execute('''
                SELECT word, A, S, tau
                FROM learned_words
                WHERE word = %s
            ''', (word,))

            row = cur.fetchone()
            conn.close()

            if row:
                return WordCoordinates(
                    word=row[0],
                    A=row[1],
                    S=row[2],
                    tau=row[3],
                    source='learned'
                )
            return None

        except Exception as e:
            print(f"Error getting learned word: {e}")
            return None

    # ========================================================================
    # LEARNING: Coordinate Computation
    # ========================================================================

    def _compute_bond_coordinates(self, adj: str, noun: str) -> Tuple[float, float, float]:
        """Compute coordinates for a bond from its components.

        Strategy:
        1. If both adj and noun have known coordinates, average them
        2. If only one has coordinates, use that with defaults
        3. If neither, try learned words table
        4. Fall back to defaults

        Args:
            adj: Adjective
            noun: Noun

        Returns:
            (A, S, tau) tuple
        """
        adj_coords = self.get(adj) or self.get_learned_word(adj)
        noun_coords = self.get(noun) or self.get_learned_word(noun)

        if adj_coords and noun_coords:
            # Average both
            return (
                (adj_coords.A + noun_coords.A) / 2,
                (adj_coords.S + noun_coords.S) / 2,
                (adj_coords.tau + noun_coords.tau) / 2,
            )
        elif adj_coords:
            return (adj_coords.A, adj_coords.S, adj_coords.tau)
        elif noun_coords:
            return (noun_coords.A, noun_coords.S, noun_coords.tau)
        else:
            # Unknown - use neutral defaults
            return (0.0, 0.0, 2.5)

    def estimate_word_coordinates(self, word: str) -> Tuple[float, float, float]:
        """Estimate coordinates for an unknown word.

        Uses simple heuristics based on word structure.
        Can be extended with embeddings or LLM-based estimation.

        Args:
            word: The word to estimate

        Returns:
            (A, S, tau) tuple
        """
        word = word.lower()

        # Check if already known
        coords = self.get(word)
        if coords:
            return (coords.A, coords.S, coords.tau)

        # Simple heuristics based on word patterns
        A, S, tau = 0.0, 0.0, 2.5

        # Negative prefixes suggest negative A
        if word.startswith(('un', 'dis', 'non', 'anti', 'mis')):
            A -= 0.3

        # Positive suffixes
        if word.endswith(('ful', 'ive', 'ous')):
            A += 0.2

        # Abstract suffixes suggest higher tau
        if word.endswith(('ness', 'ity', 'tion', 'ism', 'ment')):
            tau += 0.5

        # Concrete suffixes suggest lower tau
        if word.endswith(('er', 'or', 'ist', 'ing')):
            tau -= 0.3

        # Sacred-related patterns
        if any(w in word for w in ['god', 'soul', 'spirit', 'holy', 'divine']):
            S += 0.5
            tau += 0.5

        # Clamp values
        A = max(-1.0, min(1.0, A))
        S = max(-1.0, min(1.0, S))
        tau = max(0.5, min(4.5, tau))

        return (A, S, tau)

    # ========================================================================
    # LEARNING: Statistics
    # ========================================================================

    def get_learning_stats(self) -> Dict:
        """Get statistics about learned content.

        Returns:
            Dictionary with counts and stats
        """
        try:
            conn = psycopg2.connect(**self.config.as_dict())
            cur = conn.cursor()

            stats = {}

            # Learned bonds count
            cur.execute('SELECT COUNT(*) FROM learned_bonds')
            stats['n_learned_bonds'] = cur.fetchone()[0]

            # Learned words count
            cur.execute('SELECT COUNT(*) FROM learned_words')
            stats['n_learned_words'] = cur.fetchone()[0]

            # Total uses
            cur.execute('SELECT SUM(use_count) FROM learned_bonds')
            stats['total_bond_uses'] = cur.fetchone()[0] or 0

            # Most used bonds
            cur.execute('''
                SELECT adj, noun, use_count
                FROM learned_bonds
                ORDER BY use_count DESC
                LIMIT 5
            ''')
            stats['top_bonds'] = [
                {'bond': f"{r[0]} {r[1]}", 'uses': r[2]}
                for r in cur.fetchall()
            ]

            # Average confidence
            cur.execute('SELECT AVG(confidence) FROM learned_bonds')
            stats['avg_confidence'] = cur.fetchone()[0] or 0.0

            conn.close()
            return stats

        except Exception as e:
            # Tables might not exist yet
            return {
                'n_learned_bonds': 0,
                'n_learned_words': 0,
                'total_bond_uses': 0,
                'top_bonds': [],
                'avg_confidence': 0.0,
                'error': str(e),
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
