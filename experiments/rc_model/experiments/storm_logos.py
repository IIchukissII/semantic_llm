"""Storm-Logos Bond Generation v1.

Cognitive pattern for semantic skeleton generation:
- STORM: Explosion of candidate bonds in radius around Q
- LOGOS: Physics filter (Boltzmann × Zipf × Gravity × Coherence)
- Result: Coherent bond chain for LLM rendering

Master Equation:
    P(bond | Q) ∝ exp(-|Δτ|/kT) × v^(-α(τ)) × exp(-φ/kT) × coherence(Q, bond)

No target-direction. No Bayes/bigrams. Pure semantic physics.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import psycopg2
from scipy.spatial import cKDTree

# Import KT and Q_MAX - try relative first, fallback to definition
try:
    from ..core.semantic_rc_v2 import KT, Q_MAX
except ImportError:
    KT = math.exp(-1/5)  # ≈ 0.819
    Q_MAX = 2.0


# Constants from theory
LAMBDA = 0.5      # Gravity: weight of "falling" (toward concrete)
MU = 0.5          # Gravity: weight of "lift" (toward affirmation)
ALPHA_0 = 2.5     # Zipf baseline
ALPHA_1 = -1.4    # Zipf τ-dependence: α(τ) = α₀ + α₁×τ


@dataclass
class BondCandidate:
    """A candidate bond with coordinates and statistics."""
    noun: str
    adj: Optional[str]
    A: float          # Affirmation coordinate
    S: float          # Sacred coordinate
    tau: float        # Abstraction level
    variety: int      # Count in corpus (for Zipf)

    @property
    def coords(self) -> Tuple[float, float, float]:
        return (self.A, self.S, self.tau)

    @property
    def bond_text(self) -> str:
        if self.adj:
            return f"{self.adj} {self.noun}"
        return self.noun

    def __repr__(self):
        return f"Bond({self.bond_text}, τ={self.tau:.2f}, v={self.variety})"


@dataclass
class SemanticState:
    """Current state Q = (Q_A, Q_S, Q_τ)."""
    A: float = 0.0
    S: float = 0.0
    tau: float = 3.0  # Start mid-abstraction

    def as_array(self) -> np.ndarray:
        return np.array([self.A, self.S, self.tau])

    def copy(self) -> 'SemanticState':
        return SemanticState(A=self.A, S=self.S, tau=self.tau)


@dataclass
class GenreParams:
    """Parameters for genre-specific generation."""
    name: str
    R_storm: float          # Radius for candidate explosion
    coh_threshold: float    # Minimum coherence to pass
    boundary_A_jump: float  # A jump magnitude at boundaries
    boundary_S_jump: float  # S jump magnitude at boundaries
    S_decay: float          # S decay factor at boundaries (1.0 = no decay)
    bonds_per_sentence: int # Target bonds per sentence


# Genre parameter presets
# Note: Smaller R_storm = tighter clusters = better τ stability
GENRE_PARAMS = {
    'dramatic': GenreParams(
        name='dramatic',
        R_storm=0.6,          # Smaller radius for tighter clusters
        coh_threshold=0.2,
        boundary_A_jump=0.5,
        boundary_S_jump=0.4,
        S_decay=1.0,
        bonds_per_sentence=4,
    ),
    'ironic': GenreParams(
        name='ironic',
        R_storm=0.5,          # Even smaller for ironic (more focused)
        coh_threshold=0.3,
        boundary_A_jump=0.4,
        boundary_S_jump=0.0,
        S_decay=0.5,  # Decay toward mundane
        bonds_per_sentence=4,
    ),
    'balanced': GenreParams(
        name='balanced',
        R_storm=0.4,          # Smallest for balanced (most controlled)
        coh_threshold=0.4,
        boundary_A_jump=0.15,
        boundary_S_jump=0.15,
        S_decay=1.0,
        bonds_per_sentence=4,
    ),
}


class BondVocabulary:
    """Vocabulary of bonds with coordinates and spatial index."""

    def __init__(self, db_config: Optional[dict] = None, min_variety: int = 3,
                 english_only: bool = True):
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'bonds',
            'user': 'bonds',
            'password': 'bonds_secret',
        }
        self.min_variety = min_variety
        self.english_only = english_only

        self._bonds: List[BondCandidate] = []
        self._coords: Optional[np.ndarray] = None
        self._tree: Optional[cKDTree] = None
        self._loaded = False

    @staticmethod
    def _is_ascii_word(word: str) -> bool:
        """Check if word contains only ASCII letters."""
        if not word:
            return False
        return all(c.isascii() and (c.isalpha() or c in "-'") for c in word)

    def _is_known_english(self, word: str, coord_dict: dict) -> bool:
        """Check if word exists in our English coordinate dictionary.

        Data-driven filter: words in our corpus are from English texts.
        No hardcoded word lists needed.
        """
        if not word:
            return False
        # Must be ASCII and present in our English-derived coordinates
        return self._is_ascii_word(word) and word.lower() in coord_dict

    def _load_coordinates_json(self) -> dict:
        """Load coordinates from JSON file."""
        import json
        from pathlib import Path

        # Try __file__ first, fallback to absolute path
        try:
            coord_path = Path(__file__).parent.parent.parent / "meaning_chain/data/derived_coordinates.json"
        except NameError:
            coord_path = Path("/home/chukiss/text_project/hypothesis/experiments/semantic_llm/experiments/meaning_chain/data/derived_coordinates.json")
        with open(coord_path) as f:
            data = json.load(f)
        return data.get('coordinates', {})

    def _load_coordinates_db(self, cur) -> dict:
        """Load coordinates from hyp_semantic_words table.

        Returns dict: word -> (A, S, tau)
        """
        cur.execute('''
            SELECT word, j_sum, tau_sum
            FROM hyp_semantic_words
            WHERE word_type = 0  -- nouns only
        ''')

        coords = {}
        for word, j_sum, tau_sum in cur.fetchall():
            if j_sum and len(j_sum) >= 2:
                # j_sum[0] and j_sum[1] approximate A and S
                A = float(j_sum[0])
                S = float(j_sum[1])
                tau = float(tau_sum) if tau_sum else 3.0
                # Normalize tau to reasonable range
                tau = max(0.5, min(tau / 2.0, 5.0))
                coords[word.lower()] = (A, S, tau)

        return coords

    def _load_abstraction_data(self, cur) -> dict:
        """Load abstraction data from noun_abstraction table.

        Derives τ from adj_variety:
        - High variety → concrete → low τ
        - Low variety → abstract → high τ

        Returns dict: noun -> tau
        """
        cur.execute('''
            SELECT n.noun_text, na.adj_variety
            FROM noun_abstraction na
            JOIN nouns n ON na.noun_id = n.noun_id
        ''')

        tau_dict = {}
        max_variety = 1000  # Normalize against this

        for noun, variety in cur.fetchall():
            # Transform variety to tau
            # variety=1 → tau=4.0 (abstract)
            # variety=1000 → tau=1.0 (concrete)
            if variety and variety > 0:
                # Log scale: tau = 4 - 3 * log(variety) / log(max_variety)
                import math
                tau = 4.0 - 3.0 * math.log(variety + 1) / math.log(max_variety)
                tau = max(0.5, min(4.5, tau))
            else:
                tau = 3.0

            tau_dict[noun.lower()] = tau

        return tau_dict

    def load(self, use_full_vocab: bool = True) -> int:
        """Load bonds from database and build spatial index.

        Args:
            use_full_vocab: If True, use hyp_bond_vocab (6M bonds),
                           else use text_activations (~30K bonds)

        Returns:
            Number of bonds loaded
        """
        if self._loaded:
            return len(self._bonds)

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        # Load coordinates from JSON file (most complete A, S, τ)
        coord_dict = self._load_coordinates_json()
        print(f"Loaded {len(coord_dict):,} word coordinates from JSON")

        # Load from DB for additional coverage
        db_coords = self._load_coordinates_db(cur)
        print(f"Loaded {len(db_coords):,} word coordinates from DB")

        # Load abstraction data (τ from adj_variety) - 95K nouns!
        tau_dict = self._load_abstraction_data(cur)
        print(f"Loaded {len(tau_dict):,} abstraction values from noun_abstraction")

        # Merge coordinates (JSON > DB > derived)
        for word, coords in db_coords.items():
            if word not in coord_dict:
                coord_dict[word] = {'A': coords[0], 'S': coords[1], 'n': coords[2]}

        # Add words with only tau (default A=0, S=0)
        derived_count = 0
        for word, tau in tau_dict.items():
            if word not in coord_dict:
                # Use small random A, S for variety
                import random
                A = random.gauss(0, 0.3)
                S = random.gauss(0, 0.3)
                coord_dict[word] = {'A': A, 'S': S, 'n': tau}
                derived_count += 1

        print(f"Derived {derived_count:,} coordinates from abstraction data")
        print(f"Total coordinates: {len(coord_dict):,}")

        bonds = []
        coords_list = []
        skipped = 0

        if use_full_vocab:
            # Use hyp_bond_vocab (6M+ bonds)
            cur.execute('''
                SELECT bond, total_count
                FROM hyp_bond_vocab
                WHERE total_count >= %s
                ORDER BY total_count DESC
                LIMIT 500000
            ''', (self.min_variety,))

            for bond_str, variety in cur.fetchall():
                # Parse bond format: "adj|noun" (confirmed from DB)
                parts = bond_str.split('|')
                if len(parts) != 2:
                    continue

                adj_part, noun_part = parts[0].lower(), parts[1].lower()

                # Get coordinates for noun (also serves as English filter for nouns)
                if noun_part not in coord_dict:
                    skipped += 1
                    continue

                # Filter adjectives: must be ASCII AND known word
                # Check if adj exists in coord_dict (as any word type)
                if self.english_only:
                    if not self._is_ascii_word(adj_part):
                        skipped += 1
                        continue
                    # Stricter: adjective should also be a known English word
                    if adj_part not in coord_dict:
                        skipped += 1
                        continue

                noun, adj = noun_part, adj_part

                word_coords = coord_dict[noun]
                if isinstance(word_coords, dict):
                    A = word_coords.get('A', 0.0)
                    S = word_coords.get('S', 0.0)
                    tau = word_coords.get('n', 3.0)
                else:
                    A, S, tau = word_coords

                bond = BondCandidate(
                    noun=noun,
                    adj=adj,
                    A=float(A),
                    S=float(S),
                    tau=float(tau),
                    variety=int(variety),
                )
                bonds.append(bond)
                coords_list.append([A, S, tau])

        else:
            # Use text_activations (smaller, more curated)
            cur.execute('''
                SELECT
                    n.noun_text,
                    a.adj_text,
                    COUNT(*) as variety
                FROM text_activations ta
                JOIN nouns n ON ta.noun_id = n.noun_id
                JOIN adjectives a ON ta.adj_id = a.adj_id
                GROUP BY n.noun_text, a.adj_text
                HAVING COUNT(*) >= %s
                ORDER BY variety DESC
            ''', (self.min_variety,))

            for noun, adj, variety in cur.fetchall():
                noun_lower = noun.lower()
                adj_lower = adj.lower() if adj else None

                # Noun must be in coord_dict (also serves as English filter)
                if noun_lower not in coord_dict:
                    skipped += 1
                    continue

                # Filter adjectives: must be ASCII AND known word
                if self.english_only and adj_lower:
                    if not self._is_ascii_word(adj_lower):
                        skipped += 1
                        continue
                    if adj_lower not in coord_dict:
                        skipped += 1
                        continue

                word_coords = coord_dict[noun_lower]
                if isinstance(word_coords, dict):
                    A = word_coords.get('A', 0.0)
                    S = word_coords.get('S', 0.0)
                    tau = word_coords.get('n', 3.0)
                else:
                    A, S, tau = word_coords

                bond = BondCandidate(
                    noun=noun_lower,
                    adj=adj_lower,
                    A=float(A),
                    S=float(S),
                    tau=float(tau),
                    variety=int(variety),
                )
                bonds.append(bond)
                coords_list.append([A, S, tau])

        conn.close()

        self._bonds = bonds
        self._coords = np.array(coords_list) if coords_list else np.zeros((0, 3))

        # Build KD-tree for spatial queries
        if len(self._coords) > 0:
            self._tree = cKDTree(self._coords)

        self._loaded = True
        filter_info = "coords/english" if self.english_only else "coords"
        print(f"Loaded {len(bonds):,} bonds with spatial index (skipped {skipped:,} without {filter_info})")
        return len(bonds)

    def query_radius(self, Q: SemanticState, R: float) -> List[BondCandidate]:
        """Get all bonds within radius R of state Q.

        Args:
            Q: Current semantic state
            R: Search radius

        Returns:
            List of candidate bonds
        """
        if not self._loaded or self._tree is None:
            return []

        center = Q.as_array()
        indices = self._tree.query_ball_point(center, R)

        return [self._bonds[i] for i in indices]

    def query_nearest(self, Q: SemanticState, k: int = 50) -> List[BondCandidate]:
        """Get k nearest bonds to state Q.

        Args:
            Q: Current semantic state
            k: Number of neighbors

        Returns:
            List of candidate bonds
        """
        if not self._loaded or self._tree is None:
            return []

        center = Q.as_array()
        k = min(k, len(self._bonds))
        distances, indices = self._tree.query(center, k=k)

        return [self._bonds[i] for i in indices]

    @property
    def stats(self) -> dict:
        """Return vocabulary statistics."""
        if not self._loaded:
            return {'loaded': False}

        varieties = [b.variety for b in self._bonds]
        taus = [b.tau for b in self._bonds]

        return {
            'loaded': True,
            'n_bonds': len(self._bonds),
            'variety_mean': np.mean(varieties),
            'variety_median': np.median(varieties),
            'variety_max': max(varieties),
            'tau_mean': np.mean(taus),
            'tau_std': np.std(taus),
        }


class StormLogosGenerator:
    """Storm-Logos semantic skeleton generator.

    Cognitive pattern:
        STORM: Explosion of candidates in radius around Q
        LOGOS: Physics evaluation (Boltzmann × Zipf × Gravity)
        FILTER: Coherence threshold
        SELECT: Weighted sampling
        UPDATE: RC dynamics
    """

    def __init__(
        self,
        vocabulary: Optional[BondVocabulary] = None,
        kT: float = KT,
        lambda_: float = LAMBDA,
        mu: float = MU,
        alpha_0: float = ALPHA_0,
        alpha_1: float = ALPHA_1,
        dt: float = 0.5,
        decay: float = 0.05,
        Q_max: float = Q_MAX,
    ):
        """Initialize generator.

        Args:
            vocabulary: Bond vocabulary (will create if None)
            kT: Boltzmann temperature
            lambda_: Gravity falling weight
            mu: Gravity lift weight
            alpha_0: Zipf baseline
            alpha_1: Zipf τ-dependence
            dt: RC time step
            decay: RC decay rate
            Q_max: RC saturation limit
        """
        self.vocab = vocabulary or BondVocabulary()
        self.kT = kT
        self.lambda_ = lambda_
        self.mu = mu
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.dt = dt
        self.decay = decay
        self.Q_max = Q_max

    def load_vocabulary(self) -> int:
        """Load vocabulary if not already loaded."""
        return self.vocab.load()

    # ═══════════════════════════════════════════════════════════════
    # PHYSICS FUNCTIONS
    # ═══════════════════════════════════════════════════════════════

    def boltzmann(self, Q: SemanticState, bond: BondCandidate) -> float:
        """Boltzmann factor for abstraction barrier.

        P_boltz = exp(-|Δτ| / kT)
        """
        delta_tau = abs(bond.tau - Q.tau)
        return math.exp(-delta_tau / self.kT)

    def zipf(self, bond: BondCandidate, variety_cap: int = 500) -> float:
        """Zipf factor with variable α.

        P_zipf = v^(-α(τ))
        α(τ) = α₀ + α₁×τ

        HIGH τ (concrete): α < 0 → prefers HIGH variety
        LOW τ (abstract):  α > 0 → prefers LOW variety

        Note: Cap variety to prevent "bad X" domination pattern
        """
        if bond.variety <= 0:
            return 0.0

        # Cap variety to prevent high-frequency bonds dominating
        v = min(bond.variety, variety_cap)

        alpha = self.alpha_0 + self.alpha_1 * bond.tau
        # Clamp alpha to reasonable range to avoid extreme preferences
        alpha = max(-1.0, min(alpha, 2.0))

        # Use log variety for smoother distribution
        log_v = math.log(v + 1)
        return math.exp(-alpha * log_v / 2.5)

    def gravity(self, bond: BondCandidate) -> float:
        """Gravity factor (semantic potential).

        P_grav = exp(-φ / kT)
        φ = λτ - μA

        Prefers concrete (low τ) and affirming (high A)
        """
        phi = self.lambda_ * bond.tau - self.mu * bond.A
        return math.exp(-phi / self.kT)

    def coherence(self, Q: SemanticState, bond: BondCandidate) -> float:
        """Coherence (cosine similarity in A-S plane).

        coherence = cos(θ_Q, θ_bond)
        """
        v1 = np.array([Q.A, Q.S])
        v2 = np.array([bond.A, bond.S])

        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.5  # Neutral if either is near origin

        cos_sim = np.dot(v1, v2) / (norm1 * norm2)
        # Map from [-1, 1] to [0, 1]
        return (cos_sim + 1) / 2

    def score_bond(
        self,
        Q: SemanticState,
        bond: BondCandidate,
        tau_weight: float = 2.5,
    ) -> float:
        """Combined physics score for a bond.

        score = P_boltz^tau_weight × P_zipf × P_grav

        Args:
            Q: Current state
            bond: Candidate bond
            tau_weight: Extra weight on Boltzmann (τ stability)
        """
        p_boltz = self.boltzmann(Q, bond)
        p_zipf = self.zipf(bond)
        p_grav = self.gravity(bond)

        # Weight τ stability more heavily
        return (p_boltz ** tau_weight) * p_zipf * p_grav

    # ═══════════════════════════════════════════════════════════════
    # RC DYNAMICS
    # ═══════════════════════════════════════════════════════════════

    def update_rc(
        self,
        Q: SemanticState,
        bond: BondCandidate,
        tau_inertia: float = 0.3,
    ) -> SemanticState:
        """Update state using RC dynamics.

        dQ_x/dt = (x_w - Q_x) × (1 - |Q_x|/Q_max) - Q_x × decay

        Args:
            Q: Current state
            bond: Selected bond
            tau_inertia: Slower update for τ (0=normal, 1=no update)
        """
        new_Q = Q.copy()

        for attr, target in [('A', bond.A), ('S', bond.S), ('tau', bond.tau)]:
            current = getattr(new_Q, attr)

            # Attraction to input
            attraction = target - current

            # Saturation factor
            saturation = max(0, 1 - abs(current) / self.Q_max)

            # Forgetting
            forgetting = current * self.decay

            # Update - use reduced dt for τ (more inertia)
            dt = self.dt
            if attr == 'tau':
                dt = self.dt * (1 - tau_inertia)

            delta = attraction * saturation * dt - forgetting * dt
            setattr(new_Q, attr, current + delta)

        return new_Q

    def apply_boundary_jump(
        self,
        Q: SemanticState,
        params: GenreParams,
    ) -> SemanticState:
        """Apply genre-specific jump at sentence boundary."""
        new_Q = Q.copy()

        # A jump
        new_Q.A += np.random.randn() * params.boundary_A_jump

        # S jump (with potential decay for ironic)
        new_Q.S = new_Q.S * params.S_decay + np.random.randn() * params.boundary_S_jump

        return new_Q

    # ═══════════════════════════════════════════════════════════════
    # STORM-LOGOS GENERATION
    # ═══════════════════════════════════════════════════════════════

    def generate(
        self,
        genre: str = 'balanced',
        n_sentences: int = 3,
        Q_start: Optional[SemanticState] = None,
        seed: Optional[int] = None,
        fallback_k: int = 100,
        temperature: float = 1.5,
        repetition_penalty: float = 0.1,
        verbose: bool = False,
    ) -> Tuple[List[List[BondCandidate]], List[SemanticState]]:
        """Generate semantic skeleton via Storm-Logos pattern.

        Args:
            genre: 'dramatic', 'ironic', or 'balanced'
            n_sentences: Number of sentences to generate
            Q_start: Initial state (random if None)
            seed: Random seed
            fallback_k: Use k-nearest if radius query returns too few
            temperature: Sampling temperature (higher = more diverse)
            repetition_penalty: Penalty for repeated bonds (0-1)
            verbose: Print progress

        Returns:
            (skeleton, trajectory) where:
                skeleton: List of sentences, each a list of BondCandidates
                trajectory: List of Q states at each step
        """
        if seed is not None:
            np.random.seed(seed)

        if not self.vocab._loaded:
            self.load_vocabulary()

        params = GENRE_PARAMS.get(genre, GENRE_PARAMS['balanced'])

        # Initialize state
        if Q_start is None:
            Q = SemanticState(
                A=np.random.randn() * 0.3,
                S=np.random.randn() * 0.3,
                tau=3.0 + np.random.randn() * 0.5,
            )
        else:
            Q = Q_start.copy()

        skeleton = []
        trajectory = [Q.copy()]
        used_bonds = set()  # Track used bonds for repetition penalty
        used_adjs = {}  # Track adjective usage counts for diversity
        used_nouns = {}  # Track noun usage counts for diversity

        for sent_idx in range(n_sentences):
            sentence = []

            for bond_idx in range(params.bonds_per_sentence):

                # ═══════════════════════════════════
                # STORM: Explosion of candidates
                # ═══════════════════════════════════
                candidates = self.vocab.query_radius(Q, R=params.R_storm)

                # Fallback to k-nearest if too few
                if len(candidates) < 10:
                    candidates = self.vocab.query_nearest(Q, k=fallback_k)

                if len(candidates) == 0:
                    if verbose:
                        print(f"  No candidates at sent={sent_idx}, bond={bond_idx}")
                    break

                # ═══════════════════════════════════
                # LOGOS: Physics evaluation + Coherence filter
                # ═══════════════════════════════════
                scored = []
                for bond in candidates:
                    # Physics score
                    energy = self.score_bond(Q, bond)

                    # Coherence filter
                    coh = self.coherence(Q, bond)

                    if coh >= params.coh_threshold:
                        score = energy * coh

                        # Apply repetition penalty for exact bond
                        if bond.bond_text in used_bonds:
                            score *= repetition_penalty

                        # Apply diversity penalty for repeated adjectives
                        # Prevents "bad X, bad Y, bad Z" patterns
                        adj_count = used_adjs.get(bond.adj, 0)
                        if adj_count > 0:
                            score *= (0.5 ** adj_count)  # Exponential decay

                        # Apply diversity penalty for repeated nouns
                        # Prevents "X breath, Y breath, Z breath" patterns
                        noun_count = used_nouns.get(bond.noun, 0)
                        if noun_count > 0:
                            score *= (0.3 ** noun_count)  # Stronger decay for nouns

                        scored.append((bond, score))

                # Fallback: relax coherence if nothing passes
                if len(scored) == 0:
                    for bond in candidates:
                        energy = self.score_bond(Q, bond)
                        score = energy
                        if bond.bond_text in used_bonds:
                            score *= repetition_penalty
                        # Apply diversity penalties even in fallback
                        adj_count = used_adjs.get(bond.adj, 0)
                        if adj_count > 0:
                            score *= (0.5 ** adj_count)
                        noun_count = used_nouns.get(bond.noun, 0)
                        if noun_count > 0:
                            score *= (0.3 ** noun_count)
                        scored.append((bond, score))

                if len(scored) == 0:
                    break

                # ═══════════════════════════════════
                # SELECTION: Weighted sampling with temperature
                # ═══════════════════════════════════
                bonds, scores = zip(*scored)
                scores = np.array(scores)

                # Apply temperature and normalize to probabilities
                log_scores = np.log(scores + 1e-10)
                log_scores = (log_scores - log_scores.max()) / temperature
                probs = np.exp(log_scores)
                probs = probs / probs.sum()

                # Sample
                idx = np.random.choice(len(bonds), p=probs)
                selected = bonds[idx]
                sentence.append(selected)
                used_bonds.add(selected.bond_text)

                # Track adjective and noun usage for diversity
                if selected.adj:
                    used_adjs[selected.adj] = used_adjs.get(selected.adj, 0) + 1
                if selected.noun:
                    used_nouns[selected.noun] = used_nouns.get(selected.noun, 0) + 1

                if verbose:
                    print(f"  [{sent_idx}:{bond_idx}] {selected.bond_text} "
                          f"(τ={selected.tau:.2f}, v={selected.variety})")

                # ═══════════════════════════════════
                # UPDATE: RC dynamics
                # ═══════════════════════════════════
                Q = self.update_rc(Q, selected)
                trajectory.append(Q.copy())

            if sentence:
                skeleton.append(sentence)

            # ═══════════════════════════════════
            # BOUNDARY: Genre-specific jump
            # ═══════════════════════════════════
            Q = self.apply_boundary_jump(Q, params)
            trajectory.append(Q.copy())

        return skeleton, trajectory

    def skeleton_to_text(self, skeleton: List[List[BondCandidate]]) -> str:
        """Convert skeleton to text representation."""
        lines = []
        for i, sentence in enumerate(skeleton):
            bonds_text = ', '.join(b.bond_text for b in sentence)
            lines.append(f"Sentence {i+1}: {bonds_text}")
        return '\n'.join(lines)

    def skeleton_to_prompt(
        self,
        skeleton: List[List[BondCandidate]],
        genre: str = 'balanced',
    ) -> str:
        """Generate LLM prompt from skeleton.

        Args:
            skeleton: Generated skeleton
            genre: Genre for style instructions

        Returns:
            Prompt for LLM
        """
        all_bonds = [b.bond_text for sent in skeleton for b in sent]

        genre_instructions = {
            'dramatic': (
                "Gothic/Dostoevsky style: high tension, emotional intensity, "
                "philosophical depth. Create atmosphere of dread and wonder."
            ),
            'ironic': (
                "Kafka/Poe style: underlying tension in mundane setting. "
                "Horror or absurdity emerges from ordinary things. "
                "Keep concepts everyday, but inject unease."
            ),
            'balanced': (
                "Austen/Plato style: measured, controlled tone. "
                "Clear reasoning, calm and analytical. "
                "Social observation with wit."
            ),
        }

        instruction = genre_instructions.get(genre, genre_instructions['balanced'])

        prompt = f"""Write a short {genre} paragraph.

Style: {instruction}

Incorporate these semantic concepts naturally:
{', '.join(all_bonds)}

Write 2-3 sentences that feel authentically {genre}:"""

        return prompt


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def compute_skeleton_metrics(
    skeleton: List[List[BondCandidate]],
    trajectory: List[SemanticState],
    genre: str,
) -> dict:
    """Compute quality metrics for generated skeleton.

    Args:
        skeleton: Generated skeleton
        trajectory: State trajectory
        genre: Target genre

    Returns:
        Dict of metrics
    """
    all_bonds = [b for sent in skeleton for b in sent]

    if len(all_bonds) < 2:
        return {'error': 'Too few bonds'}

    # Coherence: mean cosine similarity between consecutive bonds
    coherences = []
    for i in range(len(all_bonds) - 1):
        b1, b2 = all_bonds[i], all_bonds[i+1]
        v1 = np.array([b1.A, b1.S])
        v2 = np.array([b2.A, b2.S])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            coh = np.dot(v1, v2) / (n1 * n2)
            coherences.append(coh)

    # τ stability: autocorrelation
    taus = np.array([b.tau for b in all_bonds])
    if len(taus) > 1:
        tau_autocorr = np.corrcoef(taus[:-1], taus[1:])[0, 1]
    else:
        tau_autocorr = 0.0

    # Genre alignment: distance to cluster center
    from ..experiments.genre_classifier import GenreClassifier
    centers = GenreClassifier.CLUSTER_CENTERS

    # Compute actual (τ, A, S) ratios from trajectory
    Q_array = np.array([[s.A, s.S, s.tau] for s in trajectory])
    if len(Q_array) > 1:
        deltas = np.abs(np.diff(Q_array, axis=0))
        mean_delta = deltas.mean(axis=0)
        # Approximate ratio (simplified)
        vec = np.array([mean_delta[2], mean_delta[0], mean_delta[1]])  # τ, A, S order
        vec = vec / (vec.mean() + 1e-6)  # Normalize
    else:
        vec = np.array([1.0, 1.0, 1.0])

    # Distance to target center
    target_center = centers.get(genre, centers['balanced'])
    genre_distance = np.linalg.norm(vec - target_center)

    # Vocabulary diversity
    unique_nouns = set(b.noun for b in all_bonds)
    diversity = len(unique_nouns) / len(all_bonds)

    # Variety statistics
    varieties = [b.variety for b in all_bonds]

    return {
        'n_bonds': len(all_bonds),
        'n_sentences': len(skeleton),
        'mean_coherence': np.mean(coherences) if coherences else 0.0,
        'tau_autocorr': float(tau_autocorr) if not np.isnan(tau_autocorr) else 0.0,
        'genre_distance': float(genre_distance),
        'vocabulary_diversity': diversity,
        'variety_mean': np.mean(varieties),
        'variety_median': np.median(varieties),
        'tau_mean': float(taus.mean()),
        'tau_std': float(taus.std()),
    }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def run_experiment(n_samples: int = 5, verbose: bool = True):
    """Run Storm-Logos generation experiment.

    Args:
        n_samples: Number of samples per genre
        verbose: Print progress
    """
    print("=" * 70)
    print("STORM-LOGOS BOND GENERATION EXPERIMENT")
    print("=" * 70)

    # Initialize
    generator = StormLogosGenerator()
    n_bonds = generator.load_vocabulary()
    print(f"\nVocabulary: {n_bonds:,} bonds")
    print(f"Stats: {generator.vocab.stats}")
    print()

    results = {}

    for genre in ['dramatic', 'ironic', 'balanced']:
        print(f"\n{'─' * 60}")
        print(f"GENRE: {genre.upper()}")
        print(f"{'─' * 60}")

        genre_metrics = []

        for sample_idx in range(n_samples):
            if verbose:
                print(f"\n[Sample {sample_idx + 1}/{n_samples}]")

            # Generate
            skeleton, trajectory = generator.generate(
                genre=genre,
                n_sentences=3,
                seed=42 + sample_idx,
                verbose=verbose,
            )

            # Compute metrics
            metrics = compute_skeleton_metrics(skeleton, trajectory, genre)
            genre_metrics.append(metrics)

            if verbose:
                print(f"\nSkeleton:")
                print(generator.skeleton_to_text(skeleton))
                print(f"\nMetrics:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.3f}")
                    else:
                        print(f"  {k}: {v}")

        # Aggregate metrics
        agg = {}
        for key in genre_metrics[0].keys():
            if key == 'error':
                continue
            values = [m[key] for m in genre_metrics if key in m]
            if values and isinstance(values[0], (int, float)):
                agg[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                }

        results[genre] = {
            'samples': genre_metrics,
            'aggregate': agg,
        }

        print(f"\n{'─' * 40}")
        print(f"AGGREGATE ({genre}):")
        print(f"{'─' * 40}")
        for k, v in agg.items():
            print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n{:<12} {:>12} {:>12} {:>12} {:>12}".format(
        "Genre", "Coherence", "τ-Autocorr", "Distance", "Diversity"
    ))
    print("-" * 60)

    for genre in ['dramatic', 'ironic', 'balanced']:
        agg = results[genre]['aggregate']
        print("{:<12} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}".format(
            genre,
            agg.get('mean_coherence', {}).get('mean', 0),
            agg.get('tau_autocorr', {}).get('mean', 0),
            agg.get('genre_distance', {}).get('mean', 0),
            agg.get('vocabulary_diversity', {}).get('mean', 0),
        ))

    # Success criteria
    print("\n" + "-" * 60)
    print("SUCCESS CRITERIA:")
    print("-" * 60)

    all_coh = [results[g]['aggregate'].get('mean_coherence', {}).get('mean', 0)
               for g in results]
    all_tau = [results[g]['aggregate'].get('tau_autocorr', {}).get('mean', 0)
               for g in results]
    all_div = [results[g]['aggregate'].get('vocabulary_diversity', {}).get('mean', 0)
               for g in results]

    mean_coh = np.mean(all_coh)
    mean_tau = np.mean(all_tau)
    mean_div = np.mean(all_div)

    print(f"  Mean coherence > 0.5:      {mean_coh:.3f} {'✓' if mean_coh > 0.5 else '✗'}")
    print(f"  τ autocorrelation > 0.7:   {mean_tau:.3f} {'✓' if mean_tau > 0.7 else '✗'}")
    print(f"  Vocabulary diversity > 0.6: {mean_div:.3f} {'✓' if mean_div > 0.6 else '✗'}")

    return results


def generate_and_prompt(genre: str = 'dramatic', seed: int = 42):
    """Generate skeleton and create LLM prompt.

    Args:
        genre: Target genre
        seed: Random seed
    """
    generator = StormLogosGenerator()
    generator.load_vocabulary()

    skeleton, trajectory = generator.generate(
        genre=genre,
        n_sentences=3,
        seed=seed,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("SKELETON:")
    print("=" * 60)
    print(generator.skeleton_to_text(skeleton))

    print("\n" + "=" * 60)
    print("LLM PROMPT:")
    print("=" * 60)
    print(generator.skeleton_to_prompt(skeleton, genre))

    metrics = compute_skeleton_metrics(skeleton, trajectory, genre)
    print("\n" + "=" * 60)
    print("METRICS:")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    return skeleton, trajectory, metrics


# ═══════════════════════════════════════════════════════════════
# OLLAMA LLM GENERATION
# ═══════════════════════════════════════════════════════════════

class OllamaRenderer:
    """Render semantic skeletons to text using Ollama."""

    def __init__(self, model: str = "mistral:7b", timeout: int = 120):
        """Initialize renderer.

        Args:
            model: Ollama model name
            timeout: Request timeout in seconds
        """
        self.model = model
        self.timeout = timeout

    def render(self, prompt: str) -> str:
        """Render prompt to text using Ollama.

        Args:
            prompt: LLM prompt

        Returns:
            Generated text
        """
        import subprocess

        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except Exception as e:
            return f"[ERROR: {e}]"

    def render_skeleton(
        self,
        skeleton: List[List[BondCandidate]],
        genre: str,
        generator: StormLogosGenerator,
    ) -> dict:
        """Render skeleton to full result.

        Args:
            skeleton: Generated skeleton
            genre: Genre name
            generator: Generator instance (for prompt creation)

        Returns:
            Dict with skeleton, prompt, and output
        """
        prompt = generator.skeleton_to_prompt(skeleton, genre)
        output = self.render(prompt)

        return {
            'genre': genre,
            'skeleton': [[b.bond_text for b in sent] for sent in skeleton],
            'skeleton_details': [
                [{'noun': b.noun, 'adj': b.adj, 'A': b.A, 'S': b.S, 'tau': b.tau, 'variety': b.variety}
                 for b in sent]
                for sent in skeleton
            ],
            'prompt': prompt,
            'output': output,
            'model': self.model,
        }


def run_full_experiment(
    n_samples: int = 3,
    save_dir: Optional[str] = None,
    model: str = "mistral:7b",
    verbose: bool = True,
) -> dict:
    """Run full Storm-Logos experiment with LLM rendering.

    Args:
        n_samples: Samples per genre
        save_dir: Directory to save results (creates if needed)
        model: Ollama model name
        verbose: Print progress

    Returns:
        Full results dict
    """
    import json
    from datetime import datetime
    from pathlib import Path

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    print("=" * 70)
    print("STORM-LOGOS FULL EXPERIMENT")
    print("=" * 70)

    # Initialize
    generator = StormLogosGenerator()  # uses KT = e^(-1/5) ≈ 0.82
    generator.load_vocabulary()
    renderer = OllamaRenderer(model=model)

    print(f"\nModel: {model}")
    print(f"Vocabulary: {len(generator.vocab._bonds):,} bonds")

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'n_samples': n_samples,
            'n_bonds': len(generator.vocab._bonds),
            'kT': generator.kT,
        },
        'samples': [],
        'metrics': {},
    }

    for genre in ['dramatic', 'ironic', 'balanced']:
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"GENRE: {genre.upper()}")
            print(f"{'─' * 60}")

        genre_samples = []
        genre_metrics = []

        for i in range(n_samples):
            if verbose:
                print(f"\n[{genre} {i+1}/{n_samples}]")

            # Generate skeleton
            skeleton, trajectory = generator.generate(
                genre=genre,
                n_sentences=3,
                seed=3000 + i * 100 + hash(genre) % 100,
                temperature=0.9,
                repetition_penalty=0.2,
            )

            # Compute metrics
            metrics = compute_skeleton_metrics(skeleton, trajectory, genre)
            genre_metrics.append(metrics)

            # Render with LLM
            result = renderer.render_skeleton(skeleton, genre, generator)
            result['metrics'] = metrics
            result['sample_id'] = i
            genre_samples.append(result)

            if verbose:
                print(f"  Skeleton: {', '.join(result['skeleton'][0][:3])}...")
                print(f"  Metrics: coh={metrics['mean_coherence']:.2f}, "
                      f"div={metrics['vocabulary_diversity']:.2f}")
                output_preview = result['output'][:100].replace('\n', ' ')
                print(f"  Output: {output_preview}...")

        # Aggregate metrics
        results['samples'].extend(genre_samples)
        results['metrics'][genre] = {
            'coherence_mean': np.mean([m['mean_coherence'] for m in genre_metrics]),
            'coherence_std': np.std([m['mean_coherence'] for m in genre_metrics]),
            'diversity_mean': np.mean([m['vocabulary_diversity'] for m in genre_metrics]),
            'diversity_std': np.std([m['vocabulary_diversity'] for m in genre_metrics]),
            'tau_autocorr_mean': np.mean([m['tau_autocorr'] for m in genre_metrics]),
            'tau_autocorr_std': np.std([m['tau_autocorr'] for m in genre_metrics]),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Genre':<12} {'Coherence':>12} {'Diversity':>12} {'τ-Autocorr':>12}")
    print("-" * 50)
    for genre, m in results['metrics'].items():
        print(f"{genre:<12} {m['coherence_mean']:>12.3f} "
              f"{m['diversity_mean']:>12.3f} {m['tau_autocorr_mean']:>12.3f}")

    # Save results
    if save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = save_path / f"storm_logos_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {results_file}")

        # Also save a readable report
        report_file = save_path / f"storm_logos_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write("# Storm-Logos Generation Report\n\n")
            f.write(f"**Date**: {results['metadata']['timestamp']}\n")
            f.write(f"**Model**: {model}\n")
            f.write(f"**Vocabulary**: {results['metadata']['n_bonds']:,} bonds\n\n")

            f.write("## Metrics Summary\n\n")
            f.write("| Genre | Coherence | Diversity | τ-Autocorr |\n")
            f.write("|-------|-----------|-----------|------------|\n")
            for genre, m in results['metrics'].items():
                f.write(f"| {genre} | {m['coherence_mean']:.3f} | "
                       f"{m['diversity_mean']:.3f} | {m['tau_autocorr_mean']:.3f} |\n")

            f.write("\n## Generated Samples\n\n")
            for sample in results['samples']:
                f.write(f"### {sample['genre'].upper()} (Sample {sample['sample_id']+1})\n\n")
                f.write("**Skeleton**:\n")
                for i, sent in enumerate(sample['skeleton']):
                    f.write(f"- Sentence {i+1}: {', '.join(sent)}\n")
                f.write(f"\n**Output**:\n> {sample['output']}\n\n")
                f.write("---\n\n")

        print(f"✓ Report saved to: {report_file}")

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full experiment with LLM
        results_dir = Path(__file__).parent.parent / "results"
        run_full_experiment(n_samples=3, save_dir=str(results_dir), verbose=True)
    else:
        # Quick experiment without LLM
        run_experiment(n_samples=3, verbose=True)
