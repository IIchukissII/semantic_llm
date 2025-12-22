#!/usr/bin/env python3
"""
Data Loader for Semantic LLM
============================

Loads semantic data from CSV/JSON files for reproduction.
Falls back to database if files not available.

Usage:
    from core.data_loader import DataLoader

    loader = DataLoader()

    # Load word vectors
    vectors = loader.load_word_vectors()

    # Load entropy statistics
    entropy = loader.load_entropy_stats()

    # Load verb operators
    verbs = loader.load_verb_operators()
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


# =========================================================================
# NounCloud: Nouns as "projections of projections" (clouds of adjectives)
# =========================================================================

@dataclass
class NounCloud:
    """
    Noun represented as a cloud of adjectives (projection of projections).

    Theory: Nouns are not direct 16D vectors but weighted combinations
    of adjective vectors. The entropy of the adjective distribution
    determines the abstraction level (τ).

    τ = 1 + 5 * (1 - h_adj_norm)

    Where:
      h_adj_norm = normalized Shannon entropy of adjective distribution
      High entropy (many adjectives) → low τ → abstract (e.g., "love")
      Low entropy (few adjectives) → high τ → concrete (e.g., "chair")
    """
    word: str
    adj_profile: Dict[str, float]  # adjective -> weight (normalized probability)
    variety: int                    # number of distinct adjectives
    h_adj: float                    # Shannon entropy of adj distribution
    h_adj_norm: float               # Normalized entropy [0, 1]
    tau: float                      # Derived: 1 + 5 * (1 - h_adj_norm)

    # Centroid computed from weighted sum of adjective vectors
    j: np.ndarray = field(default_factory=lambda: np.zeros(5))   # 5D j-space centroid
    i: np.ndarray = field(default_factory=lambda: np.zeros(11))  # 11D i-space centroid

    # Metadata
    is_cloud: bool = True           # False if fell back to direct vectors
    total_count: int = 0            # Total adjective occurrences

    @property
    def vector(self) -> np.ndarray:
        """Full 16D vector (j + i)."""
        return np.concatenate([self.j, self.i])

    @property
    def j_magnitude(self) -> float:
        """Transcendental depth (||j||)."""
        return float(np.linalg.norm(self.j))

    def top_adjectives(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n adjectives by weight."""
        return sorted(self.adj_profile.items(), key=lambda x: -x[1])[:n]

    def __repr__(self):
        top3 = ', '.join(f'{a}:{w:.2f}' for a, w in self.top_adjectives(3))
        return f"NounCloud({self.word}, τ={self.tau:.2f}, variety={self.variety}, [{top3}...])"

# Try to import psycopg2, but don't fail if not available
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class DataLoader:
    """
    Load semantic data from CSV/JSON or database.

    Priority:
        1. CSV/JSON files in data/ directory
        2. PostgreSQL database (if available)
    """

    def __init__(self, data_dir: Optional[Path] = None, db_config: Optional[dict] = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.json_dir = self.data_dir / "json"
        self.csv_dir = self.data_dir / "csv"

        self.db_config = db_config or {
            "dbname": "bonds",
            "user": "bonds",
            "password": "bonds_secret",
            "host": "localhost",
            "port": 5432
        }

        # Cache loaded data
        self._word_vectors = None
        self._entropy_stats = None
        self._verb_operators = None
        self._spin_pairs = None

    def _get_db_connection(self):
        """Get database connection if available."""
        if not HAS_PSYCOPG2:
            return None
        try:
            return psycopg2.connect(**self.db_config)
        except:
            return None

    # =========================================================================
    # Word Vectors
    # =========================================================================

    def load_word_vectors(self, force_reload: bool = False) -> Dict:
        """
        Load 16D word vectors with τ.

        Returns:
            {word: {
                'word_type': int,  # 0=noun, 1=verb, 2=adj
                'tau': float,
                'j': {beauty: float, life: float, ...},
                'i': {truth: float, freedom: float, ...},
                'verb': {beauty: float, ...} or None
            }}
        """
        if self._word_vectors is not None and not force_reload:
            return self._word_vectors

        # Try JSON first
        json_file = self.json_dir / "word_vectors.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            self._word_vectors = data.get("words", {})
            return self._word_vectors

        # Try CSV
        csv_file = self.csv_dir / "word_vectors.csv"
        if csv_file.exists():
            self._word_vectors = self._load_vectors_from_csv(csv_file)
            return self._word_vectors

        # Fall back to database
        self._word_vectors = self._load_vectors_from_db()
        return self._word_vectors

    def _load_vectors_from_csv(self, csv_file: Path) -> Dict:
        """Load word vectors from CSV file."""
        vectors = {}
        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']
        i_dims = ['truth', 'freedom', 'meaning', 'order', 'peace',
                  'power', 'nature', 'time', 'knowledge', 'self', 'society']
        verb_dims = ['beauty', 'life', 'sacred', 'good', 'love', 'truth']

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['word']
                vectors[word] = {
                    'word_type': int(row['word_type']),
                    'tau': float(row['tau']) if row['tau'] else None,
                    'variety': float(row['variety']) if row['variety'] else None,
                    'count': int(row['count']) if row['count'] else 0,
                    'j': {d: float(row[f'j_{d}']) if row.get(f'j_{d}') else 0
                          for d in j_dims},
                    'i': {d: float(row[f'i_{d}']) if row.get(f'i_{d}') else 0
                          for d in i_dims},
                }
                # Add verb vector if present
                if row.get('verb_beauty'):
                    vectors[word]['verb'] = {
                        d: float(row[f'verb_{d}']) if row.get(f'verb_{d}') else 0
                        for d in verb_dims
                    }

        return vectors

    def _load_vectors_from_db(self) -> Dict:
        """Load word vectors from database."""
        conn = self._get_db_connection()
        if conn is None:
            print("Warning: No CSV files found and database unavailable.")
            print("Run 'python scripts/export_data.py --all' to generate CSV files.")
            return {}

        cur = conn.cursor()
        cur.execute("""
            SELECT word, word_type, j, i, tau, variety, verb, count, n_books
            FROM hyp_semantic_index
        """)

        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']
        i_dims = ['truth', 'freedom', 'meaning', 'order', 'peace',
                  'power', 'nature', 'time', 'knowledge', 'self', 'society']
        verb_dims = ['beauty', 'life', 'sacred', 'good', 'love', 'truth']

        vectors = {}
        for word, wtype, j, i, tau, variety, verb, count, n_books in cur.fetchall():
            vectors[word] = {
                'word_type': wtype,
                'tau': tau,
                'variety': variety,
                'count': count,
                'j': dict(zip(j_dims, j)) if j else None,
                'i': dict(zip(i_dims, i)) if i else None,
            }
            if verb:
                vectors[word]['verb'] = dict(zip(verb_dims, verb))

        cur.close()
        conn.close()
        return vectors

    # =========================================================================
    # Entropy Statistics
    # =========================================================================

    def load_entropy_stats(self, force_reload: bool = False) -> Dict:
        """
        Load Shannon entropy statistics for nouns.

        Returns:
            {noun: {
                'h_adj': float,      # Shannon entropy of adjectives
                'h_verb': float,     # Shannon entropy of verbs
                'h_adj_norm': float, # Normalized entropy
                'h_verb_norm': float,
                'delta': float,      # H_adj - H_verb
                'tau_entropy': float # τ from entropy
            }}
        """
        if self._entropy_stats is not None and not force_reload:
            return self._entropy_stats

        # Try JSON first
        json_file = self.json_dir / "entropy_stats.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            self._entropy_stats = data.get("nouns", {})
            return self._entropy_stats

        # Try CSV
        csv_file = self.csv_dir / "entropy_stats.csv"
        if csv_file.exists():
            self._entropy_stats = self._load_entropy_from_csv(csv_file)
            return self._entropy_stats

        # Fall back to computing from database
        self._entropy_stats = self._compute_entropy_from_db()
        return self._entropy_stats

    def _load_entropy_from_csv(self, csv_file: Path) -> Dict:
        """Load entropy stats from CSV file."""
        stats = {}
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                noun = row['noun']
                stats[noun] = {
                    'h_adj': float(row['h_adj']),
                    'h_verb': float(row['h_verb']),
                    'h_adj_norm': float(row['h_adj_norm']),
                    'h_verb_norm': float(row['h_verb_norm']),
                    'delta': float(row['delta']),
                    'tau_entropy': float(row['tau_entropy']),
                    'variety_adj': int(row['variety_adj']),
                    'variety_verb': int(row['variety_verb'])
                }
        return stats

    def _compute_entropy_from_db(self) -> Dict:
        """Compute entropy stats from database."""
        conn = self._get_db_connection()
        if conn is None:
            print("Warning: No CSV files found and database unavailable.")
            return {}

        cur = conn.cursor()

        # Load adjective profiles
        cur.execute('''
            SELECT bond, total_count
            FROM hyp_bond_vocab
            WHERE total_count >= 2
        ''')

        noun_adj = defaultdict(lambda: defaultdict(int))
        for bond, count in cur.fetchall():
            parts = bond.split('|')
            if len(parts) == 2:
                adj, noun = parts
                noun_adj[noun][adj.lower()] += count

        # Load verb profiles
        cur.execute('''
            SELECT verb, object, SUM(total_count) as count
            FROM hyp_svo_triads
            WHERE total_count >= 1
            GROUP BY verb, object
        ''')

        noun_verb = defaultdict(lambda: defaultdict(int))
        for verb, noun, count in cur.fetchall():
            noun_verb[noun][verb] += count

        cur.close()
        conn.close()

        # Compute entropy
        common_nouns = set(noun_adj.keys()) & set(noun_verb.keys())
        stats = {}

        for noun in common_nouns:
            h_adj = self._shannon_entropy(noun_adj[noun])
            h_verb = self._shannon_entropy(noun_verb[noun])
            h_adj_norm = self._normalized_entropy(noun_adj[noun])
            h_verb_norm = self._normalized_entropy(noun_verb[noun])

            stats[noun] = {
                'h_adj': h_adj,
                'h_verb': h_verb,
                'h_adj_norm': h_adj_norm,
                'h_verb_norm': h_verb_norm,
                'delta': h_adj - h_verb,
                'tau_entropy': 1 + 5 * (1 - h_adj_norm),
                'variety_adj': len(noun_adj[noun]),
                'variety_verb': len(noun_verb[noun])
            }

        return stats

    @staticmethod
    def _shannon_entropy(counts: Dict[str, int]) -> float:
        """Compute Shannon entropy."""
        if not counts:
            return 0.0
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy

    @staticmethod
    def _normalized_entropy(counts: Dict[str, int]) -> float:
        """Compute normalized entropy."""
        if not counts or len(counts) <= 1:
            return 0.0
        h = DataLoader._shannon_entropy(counts)
        h_max = np.log2(len(counts))
        return h / h_max if h_max > 0 else 0.0

    # =========================================================================
    # Verb Operators
    # =========================================================================

    def load_verb_operators(self, force_reload: bool = False) -> Dict:
        """
        Load verb 6D transition operators.

        Returns:
            {verb: {
                'vector': {beauty: float, life: float, ...},
                'magnitude': float
            }}
        """
        if self._verb_operators is not None and not force_reload:
            return self._verb_operators

        # Try JSON first
        json_file = self.json_dir / "verb_operators.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)

            raw_verbs = data.get("verbs", {})
            dims = data.get("dimensions", ['beauty', 'life', 'sacred', 'good', 'love'])

            # Handle different formats: "vector" or "impulse"
            self._verb_operators = {}
            for verb, vdata in raw_verbs.items():
                if 'vector' in vdata:
                    vec = vdata['vector']
                elif 'impulse' in vdata:
                    # Convert impulse array to dict
                    vec = dict(zip(dims, vdata['impulse']))
                else:
                    continue

                if isinstance(vec, list):
                    vec = dict(zip(dims, vec))

                self._verb_operators[verb] = {
                    'vector': vec,
                    'magnitude': float(np.linalg.norm(list(vec.values())))
                }

            return self._verb_operators

        # Try CSV
        csv_file = self.csv_dir / "verb_operators.csv"
        if csv_file.exists():
            self._verb_operators = self._load_verbs_from_csv(csv_file)
            return self._verb_operators

        # Extract from word vectors
        vectors = self.load_word_vectors()
        self._verb_operators = {
            w: {'vector': v['verb'], 'magnitude': np.linalg.norm(list(v['verb'].values()))}
            for w, v in vectors.items()
            if v.get('verb') is not None
        }
        return self._verb_operators

    def _load_verbs_from_csv(self, csv_file: Path) -> Dict:
        """Load verb operators from CSV."""
        verbs = {}

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            # Detect dimensions from CSV header (exclude verb, magnitude, etc.)
            exclude = {'verb', 'magnitude', 'spin_magnitude', 'polarity', 'count', 'n_books'}
            verb_dims = [f for f in fieldnames if f not in exclude]

            for row in reader:
                verb = row['verb']
                verbs[verb] = {
                    'vector': {d: float(row[d]) for d in verb_dims if row.get(d)},
                    'magnitude': float(row['magnitude']) if row.get('magnitude') else 0,
                    'count': int(row['count']) if row.get('count') else 0
                }

        return verbs

    # =========================================================================
    # Spin Pairs
    # =========================================================================

    def load_spin_pairs(self, force_reload: bool = False) -> List[Dict]:
        """
        Load prefix spin pairs.

        Returns:
            [{
                'base': str,
                'prefixed': str,
                'j_cosine': float,
                'delta_tau': float,
                'tau_conserved': bool,
                'direction_flipped': bool
            }]
        """
        if self._spin_pairs is not None and not force_reload:
            return self._spin_pairs

        # Try JSON first
        json_file = self.json_dir / "spin_pairs.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            self._spin_pairs = data.get("pairs", [])
            return self._spin_pairs

        # Try CSV
        csv_file = self.csv_dir / "spin_pairs.csv"
        if csv_file.exists():
            self._spin_pairs = []
            with open(csv_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pair = {
                        'base': row['base'],
                        'prefixed': row['prefixed'],
                        'delta_tau': float(row['delta_tau']),
                        'tau_conserved': row['tau_conserved'] == 'True',
                    }
                    # Optional columns (may not be present)
                    if row.get('j_cosine'):
                        pair['j_cosine'] = float(row['j_cosine'])
                    if row.get('direction_flipped'):
                        pair['direction_flipped'] = row['direction_flipped'] == 'True'
                    self._spin_pairs.append(pair)
            return self._spin_pairs

        print("Warning: No spin pairs data found.")
        return []

    # =========================================================================
    # Verb-Object Pairs (for navigation)
    # =========================================================================

    def load_verb_objects(self, force_reload: bool = False) -> Dict[str, List[str]]:
        """
        Load verb-object pairs for navigation.

        Returns:
            {verb: [object1, object2, ...]}
        """
        if hasattr(self, '_verb_objects') and self._verb_objects is not None and not force_reload:
            return self._verb_objects

        csv_file = self.csv_dir / "verb_objects.csv"
        if csv_file.exists():
            self._verb_objects = defaultdict(list)
            with open(csv_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    verb = row['verb']
                    obj = row['object']
                    # Limit objects per verb for memory
                    if len(self._verb_objects[verb]) < 30:
                        self._verb_objects[verb].append(obj)
            self._verb_objects = dict(self._verb_objects)
            return self._verb_objects

        # Fall back to database
        conn = self._get_db_connection()
        if conn is None:
            print("Warning: No verb_objects.csv and database unavailable.")
            return {}

        cur = conn.cursor()
        cur.execute('''
            SELECT s.verb, s.object, SUM(s.total_count) as cnt
            FROM hyp_svo_triads s
            JOIN hyp_semantic_index i ON s.object = i.word
            WHERE s.total_count >= 2
              AND LENGTH(s.verb) >= 2
              AND LENGTH(s.object) >= 2
              AND i.j IS NOT NULL
            GROUP BY s.verb, s.object
            HAVING SUM(s.total_count) >= 3
            ORDER BY s.verb, cnt DESC
        ''')

        self._verb_objects = defaultdict(list)
        for verb, obj, cnt in cur.fetchall():
            if len(self._verb_objects[verb]) < 30:
                self._verb_objects[verb].append(obj)

        cur.close()
        conn.close()
        self._verb_objects = dict(self._verb_objects)
        return self._verb_objects

    def load_svo_patterns(self, force_reload: bool = False) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load subject-specific verb-object patterns.

        Returns:
            {subject: [(verb, object), ...]}
        """
        if hasattr(self, '_svo_patterns') and self._svo_patterns is not None and not force_reload:
            return self._svo_patterns

        csv_file = self.csv_dir / "svo_patterns.csv"
        if csv_file.exists():
            self._svo_patterns = defaultdict(list)
            with open(csv_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subj = row['subject']
                    verb = row['verb']
                    obj = row['object']
                    if len(self._svo_patterns[subj]) < 20:
                        self._svo_patterns[subj].append((verb, obj))
            self._svo_patterns = dict(self._svo_patterns)
            return self._svo_patterns

        # Fall back to database
        conn = self._get_db_connection()
        if conn is None:
            print("Warning: No svo_patterns.csv and database unavailable.")
            return {}

        cur = conn.cursor()
        cur.execute('''
            SELECT subject, verb, object, SUM(total_count) as cnt
            FROM hyp_svo_triads
            WHERE total_count >= 5 AND LENGTH(verb) >= 3 AND LENGTH(object) >= 3
            GROUP BY subject, verb, object
            HAVING SUM(total_count) >= 10
            ORDER BY subject, cnt DESC
        ''')

        self._svo_patterns = defaultdict(list)
        for subj, verb, obj, cnt in cur.fetchall():
            if len(self._svo_patterns[subj]) < 20:
                self._svo_patterns[subj].append((verb, obj))

        cur.close()
        conn.close()
        self._svo_patterns = dict(self._svo_patterns)
        return self._svo_patterns

    # =========================================================================
    # NounCloud: Nouns as Adjective Clouds
    # =========================================================================

    def load_noun_adj_profiles(self, force_reload: bool = False,
                               min_count: int = 2, top_n: int = 100) -> Dict[str, Dict[str, int]]:
        """
        Load noun-adjective profiles (which adjectives describe each noun).

        Returns:
            {noun: {adj: count, ...}}  -- raw counts, not normalized
        """
        if hasattr(self, '_noun_adj_profiles') and self._noun_adj_profiles is not None and not force_reload:
            return self._noun_adj_profiles

        # Try JSON first
        json_file = self.json_dir / "noun_adj_profiles.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            self._noun_adj_profiles = {
                noun: d.get('adj_profile', {})
                for noun, d in data.get('nouns', {}).items()
            }
            return self._noun_adj_profiles

        # Try CSV
        csv_file = self.csv_dir / "bond_statistics.csv"
        if csv_file.exists():
            self._noun_adj_profiles = self._load_adj_profiles_from_csv(csv_file)
            return self._noun_adj_profiles

        # Fall back to database
        self._noun_adj_profiles = self._load_adj_profiles_from_db(min_count, top_n)
        return self._noun_adj_profiles

    def _load_adj_profiles_from_csv(self, csv_file: Path) -> Dict[str, Dict[str, int]]:
        """Load adjective profiles from bond_statistics.csv."""
        profiles = defaultdict(dict)
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                adj = row['adj']
                noun = row['noun']
                count = int(row['count'])
                profiles[noun][adj] = count
        return dict(profiles)

    def _load_adj_profiles_from_db(self, min_count: int = 2, top_n: int = 100) -> Dict[str, Dict[str, int]]:
        """Load adjective profiles from database."""
        conn = self._get_db_connection()
        if conn is None:
            print("Warning: No noun adjective profiles available.")
            return {}

        cur = conn.cursor()
        cur.execute('''
            SELECT bond, total_count
            FROM hyp_bond_vocab
            WHERE total_count >= %s
            ORDER BY total_count DESC
        ''', (min_count,))

        profiles = defaultdict(dict)
        for bond, count in cur.fetchall():
            parts = bond.split('|')
            if len(parts) == 2:
                adj, noun = parts
                adj = adj.lower()
                if len(profiles[noun]) < top_n:
                    profiles[noun][adj] = count

        cur.close()
        conn.close()
        return dict(profiles)

    def load_noun_clouds(self, force_reload: bool = False,
                         min_adjectives: int = 5) -> Dict[str, NounCloud]:
        """
        Load nouns as NounCloud objects (adjective clouds).

        This is the theory-consistent representation:
        - Nouns are weighted combinations of adjective vectors
        - τ is derived from entropy of adjective distribution
        - j/i are centroids computed from adjective vectors

        Args:
            min_adjectives: Minimum adjectives required to create cloud.
                           Nouns with fewer fall back to direct vectors.

        Returns:
            {noun: NounCloud}
        """
        if hasattr(self, '_noun_clouds') and self._noun_clouds is not None and not force_reload:
            return self._noun_clouds

        # Load adjective profiles
        adj_profiles = self.load_noun_adj_profiles(force_reload)

        # Load word vectors (need adjective vectors for centroids)
        vectors = self.load_word_vectors(force_reload)

        # Build adjective vector lookup
        # Note: Adjectives may not be classified as word_type=2, so we use
        # words that appear as adjectives in the adj_profiles
        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']
        i_dims = ['truth', 'freedom', 'meaning', 'order', 'peace',
                  'power', 'nature', 'time', 'knowledge', 'self', 'society']

        # Get all unique adjectives from profiles
        unique_adjs = set()
        for profile in adj_profiles.values():
            unique_adjs.update(profile.keys())

        adj_vectors = {}
        for word, v in vectors.items():
            if word in unique_adjs and v.get('j'):  # word appears as adjective
                adj_vectors[word] = {
                    'j': np.array([v['j'].get(d, 0) for d in j_dims]),
                    'i': np.array([v['i'].get(d, 0) for d in i_dims])
                }

        self._noun_clouds = {}

        for noun, profile in adj_profiles.items():
            variety = len(profile)

            # Skip nouns with too few adjectives (fall back to direct vectors)
            if variety < min_adjectives:
                # Create from direct vector if available
                if noun in vectors and vectors[noun].get('j'):
                    v = vectors[noun]
                    j_arr = np.array([v['j'].get(d, 0) for d in j_dims])
                    i_arr = np.array([v['i'].get(d, 0) for d in i_dims])
                    self._noun_clouds[noun] = NounCloud(
                        word=noun,
                        adj_profile=profile,
                        variety=variety,
                        h_adj=0.0,
                        h_adj_norm=0.0,
                        tau=v.get('tau', 4.0),
                        j=j_arr,
                        i=i_arr,
                        is_cloud=False,  # Marked as fallback
                        total_count=sum(profile.values())
                    )
                continue

            # Compute entropy
            h_adj = self._shannon_entropy(profile)
            h_adj_norm = self._normalized_entropy(profile)
            tau = 1 + 5 * (1 - h_adj_norm)

            # Normalize profile to probabilities
            total = sum(profile.values())
            probs = {adj: count / total for adj, count in profile.items()}

            # Compute centroids as weighted sum of adjective vectors
            j_centroid = np.zeros(5)
            i_centroid = np.zeros(11)
            weight_sum = 0.0

            for adj, weight in probs.items():
                if adj in adj_vectors:
                    j_centroid += weight * adj_vectors[adj]['j']
                    i_centroid += weight * adj_vectors[adj]['i']
                    weight_sum += weight

            # Normalize by weight sum (in case some adjectives not found)
            if weight_sum > 0:
                j_centroid /= weight_sum
                i_centroid /= weight_sum

            self._noun_clouds[noun] = NounCloud(
                word=noun,
                adj_profile=probs,
                variety=variety,
                h_adj=h_adj,
                h_adj_norm=h_adj_norm,
                tau=tau,
                j=j_centroid,
                i=i_centroid,
                is_cloud=True,
                total_count=total
            )

        return self._noun_clouds

    def get_noun_cloud(self, noun: str) -> Optional[NounCloud]:
        """Get NounCloud for a specific noun."""
        clouds = self.load_noun_clouds()
        return clouds.get(noun)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_noun_vector(self, noun: str) -> Optional[np.ndarray]:
        """Get 16D vector for a noun."""
        vectors = self.load_word_vectors()
        if noun not in vectors:
            return None
        v = vectors[noun]
        if v['j'] is None:
            return None
        j = list(v['j'].values())
        i = list(v['i'].values())
        return np.array(j + i)

    def get_noun_tau(self, noun: str) -> Optional[float]:
        """Get τ for a noun."""
        vectors = self.load_word_vectors()
        if noun not in vectors:
            return None
        return vectors[noun].get('tau')

    def get_verb_vector(self, verb: str) -> Optional[np.ndarray]:
        """Get 6D vector for a verb."""
        verbs = self.load_verb_operators()
        if verb not in verbs:
            return None
        return np.array(list(verbs[verb]['vector'].values()))

    def get_j_good(self) -> np.ndarray:
        """Get the 'good' direction in j-space (compass)."""
        # Computed from good-evil, love-hate, beauty-ugly pairs
        return np.array([-0.48, -0.36, -0.17, +0.71, +0.33])


# Convenience function
def load_data(data_dir: Optional[Path] = None) -> DataLoader:
    """Create and return a DataLoader instance."""
    return DataLoader(data_dir)
