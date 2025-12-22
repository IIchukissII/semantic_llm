"""
Hybrid Quantum-LLM Architecture
================================

The key insight: Separate THINKING from SPEAKING.

QUANTUM CORE (16D):
  - Navigates semantic space
  - Produces trajectories (what to say)
  - Explainable, auditable, aligned by construction

LLM RENDERER:
  - Translates trajectory to fluent text
  - Just surface form (how to say it)
  - Constrained by trajectory

FEEDBACK LOOP:
  - Re-encodes LLM output to 16D
  - Compares intended vs actual
  - Rejects if fidelity < threshold
"""

import os
import sys
import numpy as np
import psycopg2
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import re

# Import DataLoader and NounCloud for CSV-based data loading
from core.data_loader import DataLoader, NounCloud

# Load API key
ENV_PATH = Path(__file__).parent.parent / "phase3_generation" / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY"):
                key = line.strip().split("=", 1)[1]
                os.environ["ANTHROPIC_API_KEY"] = key

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: anthropic not installed, using template renderer")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed, Ollama unavailable")

DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SemanticState:
    """A point in semantic space."""
    word: str
    j: np.ndarray  # 5D transcendental direction
    tau: float     # abstraction level
    goodness: float  # projection onto j_good

    # NounCloud metadata (projections of projections)
    is_cloud: bool = False  # True if derived from adjective cloud
    variety: int = 0        # Number of adjectives (if is_cloud)
    h_adj_norm: float = 0.0 # Normalized entropy of adjectives

    def __repr__(self):
        cloud_marker = "☁" if self.is_cloud else ""
        return f"|{self.word}{cloud_marker}, τ={self.tau:.2f}, g={self.goodness:+.2f}⟩"


@dataclass
class Transition:
    """A verb transition between states."""
    verb: str
    from_state: SemanticState
    to_state: SemanticState
    delta_g: float
    is_spin: bool = False  # True if this is a spin (prefix) operation

    def __repr__(self):
        op = "SPIN" if self.is_spin else self.verb
        return f"[{op}]: {self.from_state.word} → {self.to_state.word} (Δg={self.delta_g:+.2f})"


@dataclass
class SpinPair:
    """A prefix spin pair (word ↔ prefixed_word)."""
    base: str
    prefixed: str
    prefix: str  # un, dis, in, im
    j_cosine: float  # should be negative (direction flip)
    delta_tau: float  # should be ~0 (τ conserved)


@dataclass
class Trajectory:
    """A path through semantic space."""
    start: SemanticState
    transitions: List[Transition]
    goal: str  # "good" or "evil" or specific word

    @property
    def end(self) -> SemanticState:
        if self.transitions:
            return self.transitions[-1].to_state
        return self.start

    @property
    def total_delta_g(self) -> float:
        return sum(t.delta_g for t in self.transitions)

    def to_sequence(self) -> List[str]:
        """Convert to word sequence: [noun, verb, noun, verb, ...]"""
        seq = [self.start.word]
        for t in self.transitions:
            if t.is_spin:
                # For spin, the "verb" is the prefix (un-, dis-)
                # The sequence shows: word → prefix → prefixed_word
                seq.append(f"[{t.verb}]")  # Mark spin with brackets
            else:
                seq.append(t.verb)
            seq.append(t.to_state.word)
        return seq

    def energy_profile(self) -> List[Tuple[str, float, float]]:
        """
        Return energy profile as [(word, g, tau), ...] for each state.

        This shows the "quantum energy" at each step.
        """
        profile = [(self.start.word, self.start.goodness, self.start.tau)]
        for t in self.transitions:
            profile.append((t.to_state.word, t.to_state.goodness, t.to_state.tau))
        return profile

    def energy_diagram(self) -> str:
        """
        ASCII art energy diagram showing g at each step.
        """
        profile = self.energy_profile()
        lines = []

        # Scale: g ∈ [-2, +2] → columns 0-40
        def g_to_col(g):
            return int((g + 2) / 4 * 40)

        lines.append("Energy (g) Profile:")
        lines.append("─" * 45)
        lines.append("  -2       -1        0        +1       +2")
        lines.append("   ├────────┼────────┼────────┼────────┤")

        for i, (word, g, tau) in enumerate(profile):
            col = g_to_col(g)
            bar = " " * col + "●"
            lines.append(f"   {bar}")
            lines.append(f"   {word[:12]:<12} g={g:+.2f} τ={tau:.1f}")
            if i < len(profile) - 1:
                # Show transition arrow
                next_g = profile[i + 1][1]
                if next_g > g:
                    lines.append(f"   {'':>12} ↑ Δg={next_g - g:+.2f}")
                else:
                    lines.append(f"   {'':>12} ↓ Δg={next_g - g:+.2f}")

        lines.append("─" * 45)
        return "\n".join(lines)


@dataclass
class RenderResult:
    """Result of LLM rendering."""
    text: str
    trajectory: Trajectory
    extracted_words: List[str] = field(default_factory=list)
    fidelity: float = 0.0
    accepted: bool = False


# =============================================================================
# QUANTUM CORE
# =============================================================================

class QuantumCore:
    """
    The thinking engine. Navigates semantic space.

    Includes Semantic Spin operators (prefix transformations).
    Does NOT generate text - only trajectories.

    Uses DataLoader for CSV-based data loading (no database required).
    """

    PREFIXES = ['un', 'dis', 'in', 'im']  # Spin operators

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.loader = data_loader or DataLoader()
        self.states: Dict[str, SemanticState] = {}
        self.verb_objects: Dict[str, List[str]] = {}
        self.subject_verbs: Dict[str, List[Tuple[str, str]]] = {}  # subject → [(verb, object)]
        self.spin_pairs: Dict[str, SpinPair] = {}  # word → SpinPair (bidirectional)
        self.j_good = None
        self._load_space()
        self._load_verbs()
        self._load_spin_pairs()

    def _load_space(self):
        """
        Load semantic space from CSV/database via DataLoader.

        Uses NounCloud (projections of projections) for nouns:
        - j/i centroids computed from weighted adjective vectors
        - τ derived from entropy of adjective distribution
        - Fallback to direct vectors for sparse nouns (< 5 adjectives)
        """
        print("QuantumCore: Loading semantic space...")

        vectors = self.loader.load_word_vectors()
        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

        # Load NounClouds first (to compute goodness from cloud centroids)
        noun_clouds = self.loader.load_noun_clouds()

        # Compute goodness direction from NounCloud centroids (not raw adjective vectors)
        # This ensures consistency: both goodness direction and noun positions
        # are in the same "cloud centroid" space
        special = {}
        for word in ['good', 'evil', 'love', 'hate', 'peace', 'war']:
            if word in noun_clouds:
                special[word] = noun_clouds[word].j

        directions = []
        for pos, neg in [('good', 'evil'), ('love', 'hate'), ('peace', 'war')]:
            if pos in special and neg in special:
                d = special[pos] - special[neg]
                norm = np.linalg.norm(d)
                if norm > 0:
                    directions.append(d / norm)

        if directions:
            self.j_good = np.mean(directions, axis=0)
            norm = np.linalg.norm(self.j_good)
            if norm > 0:
                self.j_good = self.j_good / norm
        else:
            self.j_good = np.array([1, 1, 1, 1, 1]) / np.sqrt(5)
        cloud_count = 0
        fallback_count = 0

        # First, load nouns from NounClouds (theory-consistent)
        for word, cloud in noun_clouds.items():
            j_arr = cloud.j
            goodness = float(np.dot(j_arr, self.j_good))
            self.states[word] = SemanticState(
                word=word,
                j=j_arr,
                tau=cloud.tau,
                goodness=goodness,
                is_cloud=cloud.is_cloud,
                variety=cloud.variety,
                h_adj_norm=cloud.h_adj_norm
            )
            if cloud.is_cloud:
                cloud_count += 1
            else:
                fallback_count += 1

        # Then, add any remaining words from vectors (adjectives, verbs, etc.)
        for word, v in vectors.items():
            if word not in self.states and v.get('j') and v.get('tau') and v['tau'] > 0:
                j_arr = np.array([v['j'].get(d, 0) for d in j_dims])
                goodness = float(np.dot(j_arr, self.j_good))
                self.states[word] = SemanticState(
                    word=word, j=j_arr, tau=v['tau'], goodness=goodness,
                    is_cloud=False  # Not from NounCloud
                )

        print(f"  Loaded {len(self.states)} states")
        print(f"    NounCloud (theory-consistent): {cloud_count}")
        print(f"    Fallback (sparse nouns): {fallback_count}")
        print(f"    Other (adjectives, verbs, etc.): {len(self.states) - cloud_count - fallback_count}")

    def _load_verbs(self):
        """Load verb transitions from CSV/database via DataLoader."""
        print("QuantumCore: Loading verb transitions...")

        # Load verb-object pairs
        raw_verb_objects = self.loader.load_verb_objects()
        for verb, objs in raw_verb_objects.items():
            # Filter to objects that exist in states
            valid_objs = [obj for obj in objs if obj in self.states]
            if valid_objs:
                self.verb_objects[verb] = valid_objs[:30]

        # Keep verbs with at least 1 valid object
        self.verb_objects = {v: objs for v, objs in self.verb_objects.items()
                            if len(objs) >= 1}

        # Load subject-specific patterns
        raw_svo = self.loader.load_svo_patterns()
        for subj, patterns in raw_svo.items():
            if subj in self.states:
                valid_patterns = [(v, o) for v, o in patterns if o in self.states]
                if valid_patterns:
                    self.subject_verbs[subj] = valid_patterns[:20]

        print(f"  Loaded {len(self.verb_objects)} verbs, {len(self.subject_verbs)} subject-specific patterns")

    def _load_spin_pairs(self):
        """
        Load Semantic Spin pairs (prefix transformations).

        Spin operator: z → z̄ (complex conjugation)
        - τ preserved (abstraction level)
        - g flipped (goodness direction)
        """
        print("QuantumCore: Loading spin pairs...")

        # Find prefix pairs in our vocabulary
        for word, state in self.states.items():
            for prefix in self.PREFIXES:
                # Check if prefixed version exists
                prefixed = prefix + word
                if prefixed in self.states:
                    prefixed_state = self.states[prefixed]

                    # Compute j-cosine (should be negative for true spin)
                    j_cos = float(np.dot(state.j, prefixed_state.j) /
                                 (np.linalg.norm(state.j) * np.linalg.norm(prefixed_state.j)))

                    # Compute τ change (should be ~0 for spin conservation)
                    delta_tau = abs(prefixed_state.tau - state.tau)

                    # Only accept pairs with good spin properties
                    # j_cos < 0.5 (direction change) and delta_tau < 0.5 (τ conserved)
                    if j_cos < 0.5 and delta_tau < 0.5:
                        pair = SpinPair(
                            base=word,
                            prefixed=prefixed,
                            prefix=prefix,
                            j_cosine=j_cos,
                            delta_tau=delta_tau
                        )
                        # Bidirectional mapping
                        self.spin_pairs[word] = pair
                        self.spin_pairs[prefixed] = pair

        print(f"  Loaded {len(self.spin_pairs) // 2} spin pairs")

        # Show some examples
        examples = list(set(p.base for p in self.spin_pairs.values()))[:5]
        if examples:
            print("  Examples:")
            for base in examples:
                pair = self.spin_pairs[base]
                print(f"    {base} ↔ {pair.prefixed} (cos={pair.j_cosine:.2f}, Δτ={pair.delta_tau:.2f})")

    def apply_spin(self, state: SemanticState) -> Optional[Tuple[SemanticState, SpinPair]]:
        """
        Apply spin operator to a state.

        Returns (new_state, spin_pair) or None if no spin available.
        """
        if state.word not in self.spin_pairs:
            return None

        pair = self.spin_pairs[state.word]

        # Determine which direction we're going
        if state.word == pair.base:
            new_word = pair.prefixed
        else:
            new_word = pair.base

        new_state = self.states.get(new_word)
        if new_state:
            return (new_state, pair)
        return None

    def get_state(self, word: str) -> Optional[SemanticState]:
        """Get semantic state for a word."""
        return self.states.get(word)

    def get_transitions(self, state: SemanticState, use_subject_context: bool = True,
                        include_spin: bool = True) -> List[Transition]:
        """Get all possible transitions from a state, including spin operations."""
        transitions = []
        seen = set()  # Avoid duplicates

        # SPIN TRANSITIONS (prefix operators) - highest priority for direction flip
        if include_spin:
            spin_result = self.apply_spin(state)
            if spin_result:
                new_state, pair = spin_result
                delta_g = new_state.goodness - state.goodness
                # Use prefix as "verb" for spin transition
                transitions.append(Transition(
                    verb=pair.prefix + "-",  # e.g., "un-", "dis-"
                    from_state=state,
                    to_state=new_state,
                    delta_g=delta_g,
                    is_spin=True
                ))

        # Subject-specific transitions (higher priority)
        if use_subject_context and state.word in self.subject_verbs:
            for verb, obj in self.subject_verbs[state.word]:
                to_state = self.states.get(obj)
                if to_state and to_state.word != state.word:
                    key = (verb, obj)
                    if key not in seen:
                        seen.add(key)
                        delta_g = to_state.goodness - state.goodness
                        transitions.append(Transition(
                            verb=verb, from_state=state, to_state=to_state, delta_g=delta_g
                        ))

        # Global verb-object pairs
        for verb, objects in self.verb_objects.items():
            for obj in objects:
                to_state = self.states.get(obj)
                if to_state and to_state.word != state.word:
                    key = (verb, obj)
                    if key not in seen:
                        seen.add(key)
                        delta_g = to_state.goodness - state.goodness
                        transitions.append(Transition(
                            verb=verb, from_state=state, to_state=to_state, delta_g=delta_g
                        ))
        return transitions

    def navigate(self, start: str, goal: str = "good", steps: int = 3,
                 temperature: float = 0.3, diversity_weight: float = 0.5) -> Optional[Trajectory]:
        """
        Navigate from start toward goal with temperature-based sampling.

        This is the THINKING process - choosing the path.

        Args:
            temperature: Higher = more random, lower = more greedy
            diversity_weight: Higher = prefer subject-specific transitions
        """
        state = self.get_state(start)
        if not state:
            print(f"  Unknown word: {start}")
            return None

        trajectory = Trajectory(start=state, transitions=[], goal=goal)
        visited = {start}

        for step in range(steps):
            all_trans = self.get_transitions(state)
            # Filter visited
            all_trans = [t for t in all_trans if t.to_state.word not in visited]

            if not all_trans:
                break

            # Score transitions
            scores = []
            for t in all_trans:
                # Base score: delta_g (positive = toward good)
                if goal == "good":
                    base_score = t.delta_g
                elif goal == "evil":
                    base_score = -t.delta_g
                else:
                    # Navigate toward specific word
                    target = self.get_state(goal)
                    if target:
                        base_score = -np.linalg.norm(t.to_state.j - target.j)
                    else:
                        base_score = 0

                # Bonus for spin transitions (τ preserved, direction flipped)
                # Spin is very efficient for direction change!
                spin_bonus = 0.5 if t.is_spin else 0

                # Bonus for subject-specific transitions (more natural)
                is_subject_specific = (state.word in self.subject_verbs and
                                      (t.verb, t.to_state.word) in
                                      [(v, o) for v, o in self.subject_verbs[state.word]])
                subject_bonus = diversity_weight if is_subject_specific else 0

                scores.append(base_score + spin_bonus + subject_bonus)

            # Temperature-based selection
            if temperature > 0:
                scores = np.array(scores)
                # Softmax with temperature
                exp_scores = np.exp((scores - np.max(scores)) / temperature)
                probs = exp_scores / exp_scores.sum()
                # Sample based on probabilities
                idx = np.random.choice(len(all_trans), p=probs)
            else:
                idx = np.argmax(scores)

            best = all_trans[idx]
            trajectory.transitions.append(best)
            visited.add(best.to_state.word)
            state = best.to_state

        return trajectory

    def explore_paths(self, start: str, goal: str = "good", steps: int = 3,
                      n_paths: int = 5) -> List[Trajectory]:
        """
        Explore multiple diverse paths from start toward goal.

        Returns n_paths different trajectories for comparison.
        """
        paths = []
        seen_signatures = set()

        for i in range(n_paths * 3):  # Try more to get unique paths
            if len(paths) >= n_paths:
                break

            # Vary temperature for diversity
            temp = 0.2 + 0.3 * (i / n_paths)
            traj = self.navigate(start, goal, steps, temperature=temp)

            if traj and traj.transitions:
                # Create signature from verbs used
                sig = tuple(t.verb for t in traj.transitions)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    paths.append(traj)

        return paths

    def semantic_distance(self, word1: str, word2: str) -> float:
        """Compute semantic distance between words."""
        s1 = self.get_state(word1)
        s2 = self.get_state(word2)
        if s1 is None or s2 is None:
            return float('inf')
        return float(np.linalg.norm(s1.j - s2.j))

    def semantic_cosine(self, word1: str, word2: str) -> float:
        """Compute semantic cosine similarity."""
        s1 = self.get_state(word1)
        s2 = self.get_state(word2)
        if s1 is None or s2 is None:
            return 0.0
        return float(np.dot(s1.j, s2.j) / (np.linalg.norm(s1.j) * np.linalg.norm(s2.j)))


# =============================================================================
# LLM RENDERER
# =============================================================================

class LLMRenderer:
    """
    Translates trajectories to fluent text.

    Does NOT think - only renders what QuantumCore decides.
    """

    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self.client = None
        if HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
            self.client = anthropic.Anthropic()
            print(f"LLMRenderer: Using {model}")
        else:
            print("LLMRenderer: Using template mode (no API)")

    def render(self, trajectory: Trajectory, style: str = "narrative") -> str:
        """
        Render trajectory to text.

        The LLM is CONSTRAINED by the trajectory.
        It can only express what the trajectory contains.
        """
        if self.client is None:
            return self._template_render(trajectory, style)

        return self._llm_render(trajectory, style)

    def _template_render(self, trajectory: Trajectory, style: str) -> str:
        """Simple template-based rendering."""
        seq = trajectory.to_sequence()

        if style == "narrative":
            # Build narrative
            parts = []
            for i in range(0, len(seq) - 2, 2):
                noun1 = seq[i]
                verb = seq[i + 1]
                noun2 = seq[i + 2]
                parts.append(f"The {noun1} {verb}s the {noun2}")
            return ". ".join(parts) + "."

        elif style == "poetic":
            parts = []
            for i in range(0, len(seq) - 2, 2):
                noun1 = seq[i]
                verb = seq[i + 1]
                noun2 = seq[i + 2]
                parts.append(f"From {noun1} to {noun2}, through {verb}ing")
            return "\n".join(parts)

        else:  # simple
            return " → ".join(seq)

    def _llm_render(self, trajectory: Trajectory, style: str) -> str:
        """LLM-based rendering with trajectory constraint."""
        seq = trajectory.to_sequence()

        # Check if there are spin transitions
        has_spin = any(t.is_spin for t in trajectory.transitions)
        spin_note = ""
        if has_spin:
            spin_note = """
NOTE: Items in [brackets] like [un-] or [dis-] are SEMANTIC SPIN operators.
They flip the meaning: happy → [un-] → unhappy. Use the prefixed word naturally."""

        prompt = f"""You are a text renderer. Your ONLY job is to express the following semantic trajectory as fluent text.

TRAJECTORY (you MUST use ALL these words):
{' → '.join(seq)}
{spin_note}

CONSTRAINTS:
1. Use EVERY word in the trajectory (nouns and verbs)
2. Maintain the ORDER of the trajectory
3. The trajectory moves {'toward good/positive' if trajectory.goal == 'good' else 'toward negative/dark'}
4. Style: {style}

OUTPUT ONLY the rendered text. No explanations."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()


# =============================================================================
# OLLAMA RENDERER (Local LLM)
# =============================================================================

class OllamaRenderer:
    """
    Local LLM rendering using Ollama.

    Uses Docker-hosted Ollama for fast, local inference.
    """

    def __init__(self, model: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = False

        if HAS_REQUESTS:
            try:
                # Check if Ollama is running
                resp = requests.get(f"{base_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = [m['name'] for m in resp.json().get('models', [])]
                    if model in models:
                        self.available = True
                        print(f"OllamaRenderer: Using {model} (local)")
                    else:
                        print(f"OllamaRenderer: Model {model} not found. Available: {models}")
                else:
                    print(f"OllamaRenderer: Ollama not responding")
            except Exception as e:
                print(f"OllamaRenderer: Cannot connect to Ollama ({e})")
        else:
            print("OllamaRenderer: requests library not available")

    def render(self, trajectory: Trajectory, style: str = "narrative") -> str:
        """
        Render trajectory to text using local Ollama.
        """
        if not self.available:
            return self._template_render(trajectory, style)

        return self._ollama_render(trajectory, style)

    def _template_render(self, trajectory: Trajectory, style: str) -> str:
        """Fallback template rendering."""
        seq = trajectory.to_sequence()

        if style == "narrative":
            parts = []
            for i in range(0, len(seq) - 2, 2):
                noun1 = seq[i]
                verb = seq[i + 1]
                noun2 = seq[i + 2]
                if verb.startswith('[') and verb.endswith(']'):
                    # Spin transition
                    parts.append(f"The {noun1} becomes {noun2}")
                else:
                    parts.append(f"The {noun1} {verb}s the {noun2}")
            return ". ".join(parts) + "."

        elif style == "poetic":
            parts = []
            for i in range(0, len(seq) - 2, 2):
                noun1 = seq[i]
                verb = seq[i + 1]
                noun2 = seq[i + 2]
                parts.append(f"From {noun1} to {noun2}")
            return "\n".join(parts)

        else:
            return " → ".join(seq)

    def _ollama_render(self, trajectory: Trajectory, style: str) -> str:
        """Render using Ollama API."""
        seq = trajectory.to_sequence()

        # Check for spin transitions
        has_spin = any(t.is_spin for t in trajectory.transitions)
        spin_note = ""
        if has_spin:
            spin_note = "\nNOTE: [un-] [dis-] etc are prefix operators - use the resulting word naturally."

        prompt = f"""Express this semantic path as fluent {style} text:
{' → '.join(seq)}
{spin_note}
Use ALL words in order. Output ONLY the rendered text, nothing else."""

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 150
                    }
                },
                timeout=30
            )

            if resp.status_code == 200:
                return resp.json().get('response', '').strip()
            else:
                print(f"  Ollama error: {resp.status_code}")
                return self._template_render(trajectory, style)

        except Exception as e:
            print(f"  Ollama request failed: {e}")
            return self._template_render(trajectory, style)


# =============================================================================
# FEEDBACK ENCODER
# =============================================================================

class FeedbackEncoder:
    """
    Re-encodes LLM output to verify semantic fidelity.

    Compares intended trajectory vs actual output.
    """

    def __init__(self, core: QuantumCore):
        self.core = core

    def extract_content_words(self, text: str) -> List[str]:
        """Extract nouns and verbs from text."""
        # Simple extraction - in production would use spaCy
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # Filter to words we know
        return [w for w in words if self.core.get_state(w) is not None]

    def compute_fidelity(self, trajectory: Trajectory, text: str) -> Tuple[float, List[str]]:
        """
        Compute semantic fidelity between intended trajectory and rendered text.

        Returns (fidelity_score, extracted_words)
        """
        intended = trajectory.to_sequence()
        intended_nouns = [intended[i] for i in range(0, len(intended), 2)]

        extracted = self.extract_content_words(text)

        if not extracted:
            return 0.0, extracted

        # Method 1: Word overlap
        overlap = len(set(intended_nouns) & set(extracted)) / len(intended_nouns)

        # Method 2: Semantic direction preservation
        if len(extracted) >= 2:
            first = self.core.get_state(extracted[0])
            last = self.core.get_state(extracted[-1])
            intended_first = trajectory.start
            intended_last = trajectory.end

            if first and last:
                # Check if direction is preserved
                intended_delta = intended_last.goodness - intended_first.goodness
                actual_delta = last.goodness - first.goodness

                # Same sign = direction preserved
                if intended_delta * actual_delta > 0:
                    direction_score = 1.0
                elif abs(intended_delta) < 0.1 or abs(actual_delta) < 0.1:
                    direction_score = 0.5  # Neutral
                else:
                    direction_score = 0.0  # Wrong direction!
            else:
                direction_score = 0.5
        else:
            direction_score = 0.5

        # Combined fidelity
        fidelity = 0.6 * overlap + 0.4 * direction_score

        return fidelity, extracted

    def verify(self, trajectory: Trajectory, text: str, threshold: float = 0.5) -> RenderResult:
        """
        Verify that rendered text matches intended trajectory.
        """
        fidelity, extracted = self.compute_fidelity(trajectory, text)

        return RenderResult(
            text=text,
            trajectory=trajectory,
            extracted_words=extracted,
            fidelity=fidelity,
            accepted=fidelity >= threshold
        )


# =============================================================================
# HYBRID SYSTEM
# =============================================================================

class HybridQuantumLLM:
    """
    The complete hybrid system.

    QuantumCore thinks → LLM speaks → Feedback verifies
    """

    def __init__(self, max_retries: int = 3, fidelity_threshold: float = 0.5,
                 renderer: str = "auto"):
        """
        Initialize hybrid system.

        Args:
            renderer: "claude" | "ollama" | "template" | "auto"
                      auto = prefer ollama if available, else claude, else template
        """
        print("=" * 60)
        print("INITIALIZING HYBRID QUANTUM-LLM SYSTEM")
        print("=" * 60)

        self.core = QuantumCore()

        # Select renderer
        if renderer == "ollama":
            self.renderer = OllamaRenderer()
        elif renderer == "claude":
            self.renderer = LLMRenderer()
        elif renderer == "template":
            self.renderer = LLMRenderer()  # Will fall back to template
            self.renderer.client = None  # Force template mode
        elif renderer == "auto":
            # Try Ollama first (local, fast, free)
            ollama = OllamaRenderer()
            if ollama.available:
                self.renderer = ollama
            else:
                self.renderer = LLMRenderer()
        else:
            self.renderer = LLMRenderer()

        self.encoder = FeedbackEncoder(self.core)
        self.max_retries = max_retries
        self.fidelity_threshold = fidelity_threshold

        print("=" * 60)
        print("SYSTEM READY")
        print("=" * 60)

    def generate(self, start: str, goal: str = "good",
                 steps: int = 3, style: str = "narrative") -> Optional[RenderResult]:
        """
        Generate text with semantic navigation.

        1. QuantumCore navigates (thinks)
        2. LLM renders (speaks)
        3. Feedback verifies (checks)
        4. Loop if needed
        """
        print(f"\n{'='*60}")
        print(f"GENERATING: {start} → {goal}")
        print(f"{'='*60}")

        # Step 1: Navigate (THINKING)
        print("\n[1] QUANTUM CORE: Navigating...")
        trajectory = self.core.navigate(start, goal, steps)

        if trajectory is None:
            print("  Failed to create trajectory")
            return None

        print(f"  Trajectory: {' → '.join(trajectory.to_sequence())}")
        print(f"  Total Δg: {trajectory.total_delta_g:+.2f}")

        # Show energy profile at each step
        print("\n  Energy quanta per step:")
        profile = trajectory.energy_profile()
        for i, (word, g, tau) in enumerate(profile):
            if i == 0:
                print(f"    START: {word:<12} g={g:+.2f} τ={tau:.1f}")
            else:
                prev_g = profile[i-1][1]
                prev_tau = profile[i-1][2]
                delta_g = g - prev_g
                delta_tau = tau - prev_tau
                trans = trajectory.transitions[i-1]

                if trans.is_spin:
                    # Spin transition - show τ conservation
                    print(f"    SPIN[{trans.verb}] → {word:<12} g={g:+.2f} τ={tau:.1f} (Δg={delta_g:+.2f}, Δτ={delta_tau:+.2f}) ★")
                else:
                    print(f"    [{trans.verb:<8}] → {word:<12} g={g:+.2f} τ={tau:.1f} (Δg={delta_g:+.2f})")

        # Step 2-3: Render and Verify (with retry loop)
        for attempt in range(self.max_retries):
            print(f"\n[2] LLM RENDERER: Rendering (attempt {attempt + 1})...")
            text = self.renderer.render(trajectory, style)
            print(f"  Output: \"{text}\"")

            print(f"\n[3] FEEDBACK: Verifying...")
            result = self.encoder.verify(trajectory, text, self.fidelity_threshold)
            print(f"  Extracted: {result.extracted_words}")
            print(f"  Fidelity: {result.fidelity:.2f}")
            print(f"  Accepted: {result.accepted}")

            if result.accepted:
                print(f"\n✓ GENERATION SUCCESSFUL")
                return result

            print(f"  Retrying with different rendering...")

        print(f"\n✗ GENERATION FAILED after {self.max_retries} attempts")
        return result  # Return last attempt

    def explain(self, result: RenderResult) -> str:
        """
        Explain the generation process.

        This is AUDITABLE - we can show exactly why each choice was made.
        """
        if result is None:
            return "No result to explain"

        traj = result.trajectory
        lines = [
            "EXPLANATION OF GENERATION",
            "=" * 40,
            f"Goal: Navigate toward '{traj.goal}'",
            f"",
            f"Starting state: {traj.start}",
            f"",
            "Navigation steps:"
        ]

        for i, t in enumerate(traj.transitions):
            lines.append(f"  {i+1}. From {t.from_state.word} (g={t.from_state.goodness:+.2f})")
            lines.append(f"     Action: [{t.verb}]")
            lines.append(f"     To: {t.to_state.word} (g={t.to_state.goodness:+.2f})")
            lines.append(f"     Change: Δg={t.delta_g:+.2f}")
            lines.append("")

        lines.extend([
            f"Final state: {traj.end}",
            f"Total goodness change: {traj.total_delta_g:+.2f}",
            f"",
            f"Rendered text: \"{result.text}\"",
            f"Semantic fidelity: {result.fidelity:.2f}",
            f"Verification: {'PASSED' if result.accepted else 'FAILED'}"
        ])

        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the hybrid system."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", choices=["auto", "ollama", "claude", "template"],
                        default="auto", help="Renderer to use")
    parser.add_argument("--complex", action="store_true", help="Run complex abstraction tests")
    args = parser.parse_args()

    system = HybridQuantumLLM(renderer=args.renderer)

    print("\n" + "=" * 60)
    print("DEMO: HYBRID QUANTUM-LLM GENERATION")
    print("=" * 60)

    # Basic test cases
    tests = [
        ("war", "good", "Transform war to peace"),
        ("hate", "good", "Transform hate to love"),
        ("order", "evil", "Corrupt order (→ disorder?)"),
    ]

    for start, goal, description in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {description}")
        print(f"{'='*60}")

        result = system.generate(start, goal, steps=3, style="narrative")

        if result:
            print(f"\n{system.explain(result)}")

    # Complex abstraction tests
    if args.complex:
        run_complex_tests(system)
    else:
        print("\n[Use --complex flag for extended abstraction tests]")

    # Show spin pairs available
    print("\n" + "=" * 60)
    print("AVAILABLE SPIN PAIRS")
    print("=" * 60)
    spin_examples = list(set(p.base for p in system.core.spin_pairs.values()))[:15]
    print("\nPrefix spin operators (τ conserved, g flipped):")
    for base in sorted(spin_examples):
        pair = system.core.spin_pairs[base]
        base_g = system.core.states[base].goodness
        pref_g = system.core.states[pair.prefixed].goodness
        print(f"  {base:<15} ↔ {pair.prefixed:<15} (Δg={pref_g - base_g:+.2f}, cos={pair.j_cosine:.2f})")


def run_complex_tests(system: HybridQuantumLLM):
    """Run complex abstraction tests with feedback loop."""
    print("\n" + "=" * 70)
    print("COMPLEX ABSTRACTION TESTS")
    print("=" * 70)

    # Abstract concepts at high τ levels
    abstract_concepts = [
        # Philosophical concepts
        ("truth", "good", 5, "Journey from truth toward ultimate good"),
        ("freedom", "good", 5, "Path of freedom to goodness"),
        ("justice", "evil", 5, "Corruption of justice"),
        ("wisdom", "evil", 5, "Perversion of wisdom"),

        # Emotional transformations
        ("fear", "good", 4, "Transmutation of fear"),
        ("anger", "good", 4, "Channeling anger to good"),
        ("joy", "evil", 4, "Joy corrupted"),
        ("hope", "evil", 4, "Hope destroyed"),

        # Existential concepts
        ("existence", "good", 5, "Existence toward meaning"),
        ("chaos", "good", 5, "Order from chaos"),
        ("darkness", "good", 5, "Light from darkness"),
        ("silence", "good", 4, "Silence to understanding"),

        # Social concepts
        ("power", "evil", 5, "Corruption of power"),
        ("authority", "evil", 5, "Authority's dark turn"),
        ("innocence", "evil", 5, "Loss of innocence"),
        ("trust", "evil", 4, "Betrayal of trust"),
    ]

    results = []

    for start, goal, steps, description in abstract_concepts:
        # Check if word exists in semantic space
        if system.core.get_state(start) is None:
            print(f"\n[SKIP] {start} not in semantic space")
            continue

        print(f"\n{'='*70}")
        print(f"COMPLEX TEST: {description}")
        print(f"  Start: {start}, Goal: {goal}, Steps: {steps}")
        print(f"{'='*70}")

        # Generate with longer trajectory
        result = system.generate(start, goal, steps=steps, style="narrative")

        if result:
            # Show energy diagram
            print("\n" + result.trajectory.energy_diagram())

            # Store results for analysis
            results.append({
                "start": start,
                "goal": goal,
                "steps": steps,
                "fidelity": result.fidelity,
                "accepted": result.accepted,
                "text": result.text,
                "total_delta_g": result.trajectory.total_delta_g
            })

    # Summary of complex tests
    print("\n" + "=" * 70)
    print("COMPLEX TEST SUMMARY")
    print("=" * 70)

    accepted = [r for r in results if r["accepted"]]
    rejected = [r for r in results if not r["accepted"]]

    print(f"\nTotal tests: {len(results)}")
    print(f"Accepted: {len(accepted)} ({100*len(accepted)/len(results):.1f}%)" if results else "")
    print(f"Rejected: {len(rejected)}")

    print("\nFidelity distribution:")
    if results:
        fidelities = [r["fidelity"] for r in results]
        print(f"  Mean: {np.mean(fidelities):.3f}")
        print(f"  Min:  {min(fidelities):.3f}")
        print(f"  Max:  {max(fidelities):.3f}")

    print("\nTotal Δg distribution (toward good = positive):")
    if results:
        good_tests = [r for r in results if r["goal"] == "good"]
        evil_tests = [r for r in results if r["goal"] == "evil"]

        if good_tests:
            good_deltas = [r["total_delta_g"] for r in good_tests]
            print(f"  Good trajectories: mean Δg = {np.mean(good_deltas):+.3f}")

        if evil_tests:
            evil_deltas = [r["total_delta_g"] for r in evil_tests]
            print(f"  Evil trajectories: mean Δg = {np.mean(evil_deltas):+.3f}")

    # Show best and worst examples
    if results:
        print("\n--- Best Fidelity Example ---")
        best = max(results, key=lambda r: r["fidelity"])
        print(f"  {best['start']} → {best['goal']}: fidelity={best['fidelity']:.3f}")
        print(f"  \"{best['text']}\"")

        print("\n--- Lowest Fidelity Example ---")
        worst = min(results, key=lambda r: r["fidelity"])
        print(f"  {worst['start']} → {worst['goal']}: fidelity={worst['fidelity']:.3f}")
        print(f"  \"{worst['text']}\"")

    # Multi-path exploration for key concepts
    print("\n" + "=" * 70)
    print("MULTI-PATH EXPLORATION")
    print("=" * 70)

    key_concepts = ["freedom", "truth", "power"]
    for concept in key_concepts:
        if system.core.get_state(concept) is None:
            continue

        print(f"\n--- Multiple paths from '{concept}' toward good ---")
        paths = system.core.explore_paths(concept, "good", steps=4, n_paths=3)

        for i, traj in enumerate(paths):
            seq = traj.to_sequence()
            print(f"  Path {i+1}: {' → '.join(seq)}")
            print(f"           Total Δg: {traj.total_delta_g:+.2f}")

    return results


if __name__ == "__main__":
    demo()
