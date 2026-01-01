"""
Semantic Laser: Coherent Meaning Extraction with Euler Physics + Intent Collapse

Combines laser physics with Euler orbital theory AND intent-driven navigation:

LASER PHYSICS:
1. PUMPING - Excite concepts via Boltzmann-weighted transitions
2. POPULATION INVERSION - Track orbital distribution
3. STIMULATED EMISSION - Coherence requires orbital + j-vector alignment
4. COHERENT OUTPUT - Beam with specific frequency (τ) and polarization (j)

EULER ORBITAL THEORY:
- τ_n = 1 + n/e (quantized abstraction levels)
- kT ≈ 0.82 (natural semantic temperature)
- Veil at τ = e (human/transcendental boundary)
- Boltzmann transitions: P ∝ exp(-Δτ/kT)

INTENT COLLAPSE (NEW):
- Verbs act as operators that collapse navigation
- Pumping prioritizes intent-aligned transitions
- "understand" + "help" → navigate toward what those verbs act upon

The combination: coherent meaning requires concepts that are
at similar energy levels (orbital), aligned in meaning (j-vector),
AND relevant to the user's intent (verb operators).
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from chain_core.intent_collapse import IntentCollapse, IntentOperator

# Euler Constants
E = np.e                    # Euler's number
KT_NATURAL = 0.82           # Natural semantic temperature
VEIL_TAU = E                # The Veil boundary (τ = e ≈ 2.718)
GROUND_STATE_TAU = 1.37     # Ground state (n=1 orbital)


@dataclass
class ExcitedState:
    """A concept in excited state after pumping."""
    word: str
    tau: float
    affirmation: float               # A: Affirmation score (formerly g)
    j: np.ndarray  # 5D j-vector
    sacred: float = 0.0              # S: Sacred score
    visits: int = 1
    sources: Set[str] = field(default_factory=set)  # Which seeds reached this

    # Intent collapse tracking (NEW)
    collapsed_by_intent: bool = False  # True if reached via intent-driven path
    intent_score: float = 0.0          # How aligned with intent [0, 1]

    @property
    def A(self) -> float:
        """Alias for affirmation."""
        return self.affirmation

    @property
    def g(self) -> float:
        """Legacy alias (g ≈ A)."""
        return self.affirmation

    @property
    def energy(self) -> float:
        """Energy level (higher tau = higher energy)."""
        return self.tau

    @property
    def orbital(self) -> int:
        """Orbital number n = (tau - 1) * e."""
        return int(round((self.tau - 1) * np.e))


@dataclass
class CoherentBeam:
    """Output of semantic laser - coherent meaning."""
    concepts: List[str]
    j_centroid: np.ndarray  # Average j-vector (beam direction)
    coherence: float  # How aligned (0-1)
    affirmation: float  # Average A (Affirmation)
    tau_mean: float  # Average abstraction
    tau_spread: float  # Spread in abstraction

    @property
    def A(self) -> float:
        """Alias for affirmation."""
        return self.affirmation

    @property
    def g_polarity(self) -> float:
        """Legacy alias (g ≈ A)."""
        return self.affirmation

    @property
    def intensity(self) -> float:
        """Beam intensity = coherence * number of concepts."""
        return self.coherence * len(self.concepts)


class SemanticLaser:
    """
    Semantic Laser for coherent meaning extraction.

    Unlike random walk that finds high-connectivity nodes,
    this finds semantically COHERENT clusters using j-vector alignment.

    NEW: Intent collapse support - verbs act as operators that
    collapse navigation toward intent-relevant concepts.
    """

    def __init__(self, graph: MeaningGraph = None, temperature: float = KT_NATURAL,
                 intent_strength: float = 0.3):
        """
        Initialize Semantic Laser.

        Args:
            graph: MeaningGraph connection
            temperature: Boltzmann temperature (kT)
            intent_strength: α parameter for intent weighting (0=pure Boltzmann, 1=hard collapse)
                           Recommended: 0.3 (soft guidance, "wind not wall")
        """
        self.graph = graph or MeaningGraph()
        self._j_dims = ['beauty', 'life', 'sacred', 'good', 'love']
        self.kT = temperature  # Semantic temperature for Boltzmann transitions
        self.intent_strength = intent_strength  # α: intent as wind, not wall

        # Intent collapse support
        self.intent_collapse = IntentCollapse(self.graph)
        self.intent_enabled = False
        self.intent_verbs: List[str] = []

        mode = "soft guidance" if 0 < intent_strength < 1 else ("hard collapse" if intent_strength >= 1 else "pure Boltzmann")
        print(f"[SemanticLaser] kT={self.kT:.2f}, α={self.intent_strength:.2f} ({mode})")

    def set_intent(self, verbs: List[str]) -> Dict:
        """
        Set intent operators from user's verbs.

        This enables intent-driven pumping: navigation will prioritize
        transitions aligned with these verbs.

        Args:
            verbs: Verbs from user query (e.g., ['understand', 'help'])

        Returns:
            Stats: {operators, targets, intent_j}
        """
        if not verbs:
            self.intent_enabled = False
            self.intent_verbs = []
            return {'operators': 0, 'targets': 0, 'intent_j': None}

        self.intent_verbs = verbs
        stats = self.intent_collapse.set_intent(verbs)
        self.intent_enabled = stats['operators'] > 0 or stats['targets'] > 0

        if self.intent_enabled:
            print(f"[SemanticLaser] Intent set: {verbs} -> "
                  f"{stats['operators']} operators, {stats['targets']} targets")

        return stats

    def _get_j_vector(self, j_data) -> np.ndarray:
        """
        Convert j-vector data to numpy array.

        Now that j-vectors are synced to Neo4j, we receive them directly
        from concept properties rather than looking up by word.
        """
        if j_data is None:
            return np.zeros(5)
        if isinstance(j_data, np.ndarray):
            return j_data if len(j_data) >= 5 else np.pad(j_data, (0, 5-len(j_data)))
        if isinstance(j_data, list):
            return np.array(j_data[:5] if len(j_data) >= 5 else j_data + [0]*(5-len(j_data)))
        if isinstance(j_data, dict):
            return np.array([j_data.get(d, 0) for d in self._j_dims])
        return np.zeros(5)

    # =========================================================================
    # Phase 1: PUMPING - Broad exploration
    # =========================================================================

    def _boltzmann_weight(self, tau_from: float, tau_to: float) -> float:
        """
        Compute Boltzmann transition probability.

        P ∝ exp(-|Δτ|/kT)

        Transitions to similar τ levels are more probable.
        This respects Euler orbital quantization.
        """
        delta_tau = abs(tau_to - tau_from)
        return np.exp(-delta_tau / self.kT)

    def pump(self, seeds: List[str],
             pump_power: int = 10,  # walks per seed
             pump_depth: int = 5    # steps per walk
             ) -> Dict[str, ExcitedState]:
        """
        Pumping phase: intent-WEIGHTED Boltzmann exploration.

        Formula: P(next) ∝ exp(-|Δτ|/kT) × (1 + α × intent_alignment)

        where α = intent_strength:
          - α = 0:   Pure Boltzmann (cloud)
          - α = 0.3: Soft guidance (wind)
          - α ≥ 2:   Strong collapse (wall)

        Intent is wind, not wall.

        Args:
            seeds: Starting concepts
            pump_power: Number of walks per seed
            pump_depth: Steps per walk

        Returns:
            {word: ExcitedState} - all excited concepts
        """
        excited = {}  # word -> ExcitedState

        # Track statistics
        total_intent_weight = 0.0
        total_weight = 0.0

        for seed in seeds:
            # Get seed properties
            concept = self.graph.get_concept(seed)
            if not concept:
                continue

            seed_tau = concept.get('tau', 2.0)
            seed_state = ExcitedState(
                word=seed,
                tau=seed_tau,
                affirmation=concept.get('g', 0.0),  # g ≈ A
                j=self._get_j_vector(concept.get('j')),
                visits=1,
                sources={seed}
            )
            excited[seed] = seed_state

            # Walks from this seed
            for _ in range(pump_power):
                current = seed
                current_tau = seed_tau

                for step in range(pump_depth):
                    # Get ALL neighbors
                    neighbors = self._get_neighbors_with_tau(current)
                    if not neighbors:
                        break

                    # Compute intent-weighted Boltzmann probabilities
                    words = []
                    taus = []
                    weights = []
                    intent_scores = []

                    for word, tau in neighbors:
                        # Base Boltzmann weight
                        boltz = self._boltzmann_weight(current_tau, tau)

                        # Intent alignment (0 or 1+)
                        intent_align = 0.0
                        if self.intent_enabled:
                            # Check if word is in intent targets
                            if word in self.intent_collapse.intent_targets:
                                intent_align = 1.0
                            # Or check if edge verb matches intent verbs
                            # (this is approximated - full check would need edge data)

                        # Combined weight: P ∝ boltz × (1 + α × intent)
                        combined = boltz * (1.0 + self.intent_strength * intent_align)

                        words.append(word)
                        taus.append(tau)
                        weights.append(combined)
                        intent_scores.append(intent_align)

                        # Track for statistics
                        total_weight += combined
                        total_intent_weight += combined * intent_align

                    # Normalize and sample
                    total = sum(weights)
                    if total <= 0:
                        break

                    probs = [w / total for w in weights]
                    idx = np.random.choice(len(words), p=probs)

                    next_word = words[idx]
                    next_tau = taus[idx]
                    next_intent = intent_scores[idx]

                    # Get full concept
                    next_concept = self.graph.get_concept(next_word)
                    if not next_concept:
                        break

                    # Add to excited states
                    if next_word in excited:
                        excited[next_word].visits += 1
                        excited[next_word].sources.add(seed)
                        if next_intent > 0:
                            excited[next_word].collapsed_by_intent = True
                            excited[next_word].intent_score = max(
                                excited[next_word].intent_score, next_intent
                            )
                    else:
                        excited[next_word] = ExcitedState(
                            word=next_word,
                            tau=next_concept.get('tau', 2.0),
                            affirmation=next_concept.get('g', 0.0),  # g ≈ A
                            j=self._get_j_vector(next_concept.get('j')),
                            visits=1,
                            sources={seed},
                            collapsed_by_intent=(next_intent > 0),
                            intent_score=next_intent
                        )

                    current = next_word
                    current_tau = next_tau

        # Log intent influence (not collapse ratio, but weight contribution)
        if total_weight > 0 and self.intent_enabled:
            influence = total_intent_weight / total_weight
            print(f"[SemanticLaser] Intent influence: {influence:.1%} (α={self.intent_strength:.2f})")

        return excited

    def _get_intent_transition(self, word: str) -> Optional[Dict]:
        """
        Get intent-driven transition from word.

        Uses the graph's get_intent_transitions() which filters by:
        1. Edges where verb matches an intent verb
        2. Edges leading to intent targets (what intent verbs act upon)

        Returns None if no intent-aligned transition found.
        """
        if not self.intent_collapse.intent_verbs:
            return None

        transitions = self.graph.get_intent_transitions(
            word,
            self.intent_collapse.intent_verbs,
            self.intent_collapse.intent_targets,
            limit=5
        )

        if not transitions:
            return None

        # Pick best by intent score (transitions are sorted by score)
        verb, target, score = transitions[0]

        # Get full concept properties
        concept = self.graph.get_concept(target)
        if not concept:
            return None

        return {
            'word': target,
            'tau': concept.get('tau', 2.0),
            'g': concept.get('g', 0.0),
            'j': concept.get('j'),
            'intent_score': score,
            'collapse_verb': verb
        }

    def _get_neighbors_with_tau(self, word: str, limit: int = 30) -> List[Tuple[str, float]]:
        """Get neighbors with their tau values for Boltzmann weighting."""
        if not self.graph.driver:
            return []

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (source:Concept {word: $word})-[r:VIA]->(target:Concept)
                WHERE size(target.word) >= 3
                RETURN target.word as target, target.tau as tau, r.weight as weight
                ORDER BY r.weight DESC
                LIMIT $limit
            """, word=word, limit=limit)

            return [(r["target"], r["tau"] or 2.0) for r in result]

    def _get_neighbors_unfiltered(self, word: str, limit: int = 30) -> List[Tuple[str, str, float]]:
        """Get neighbors without tau filtering."""
        if not self.graph.driver:
            return []

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (source:Concept {word: $word})-[r:VIA]->(target:Concept)
                WHERE size(target.word) >= 3
                RETURN target.word as target, r.verb as verb, r.weight as weight
                ORDER BY r.weight DESC
                LIMIT $limit
            """, word=word, limit=limit)

            return [(r["target"], r["verb"], r["weight"]) for r in result]

    # =========================================================================
    # Phase 2: POPULATION ANALYSIS
    # =========================================================================

    def analyze_population(self, excited: Dict[str, ExcitedState]) -> Dict:
        """
        Analyze the excited population with Euler orbital statistics.

        Returns statistics about the excited states including
        orbital distribution, veil crossings, and intent collapse metrics.
        """
        if not excited:
            return {}

        states = list(excited.values())

        # Tau distribution (energy levels)
        taus = [s.tau for s in states]

        # A distribution (Affirmation)
        affirmations = [s.affirmation for s in states]

        # J-vectors
        js = np.array([s.j for s in states])
        j_mean = js.mean(axis=0)

        # Visit distribution
        visits = [s.visits for s in states]

        # Euler orbital analysis
        orbitals = [s.orbital for s in states]
        orbital_dist = {}
        for n in orbitals:
            orbital_dist[n] = orbital_dist.get(n, 0) + 1

        # Count concepts below/above the Veil
        below_veil = sum(1 for t in taus if t < VEIL_TAU)
        above_veil = len(taus) - below_veil
        human_fraction = below_veil / len(taus) if taus else 0.5

        # Find dominant orbital (population inversion target)
        dominant_orbital = max(orbital_dist.keys(), key=lambda n: orbital_dist[n]) if orbital_dist else 1

        # Intent collapse statistics (NEW)
        intent_collapsed = sum(1 for s in states if s.collapsed_by_intent)
        intent_fraction = intent_collapsed / len(states) if states else 0.0
        avg_intent_score = np.mean([s.intent_score for s in states if s.collapsed_by_intent]) if intent_collapsed > 0 else 0.0

        return {
            'total_excited': len(states),
            'tau_mean': np.mean(taus),
            'tau_std': np.std(taus),
            'tau_min': np.min(taus),
            'tau_max': np.max(taus),
            'affirmation_mean': np.mean(affirmations),
            'affirmation_std': np.std(affirmations),
            'g_mean': np.mean(affirmations),  # Legacy alias
            'g_std': np.std(affirmations),    # Legacy alias
            'j_centroid': j_mean,
            'j_magnitude': np.linalg.norm(j_mean),
            'total_visits': sum(visits),
            'max_visits': max(visits),
            'multi_source': sum(1 for s in states if len(s.sources) > 1),
            # Euler statistics
            'orbital_dist': orbital_dist,
            'dominant_orbital': dominant_orbital,
            'below_veil': below_veil,
            'above_veil': above_veil,
            'human_fraction': human_fraction,
            'mean_orbital': np.mean(orbitals),
            # Intent collapse statistics (NEW)
            'intent_collapsed': intent_collapsed,
            'intent_fraction': intent_fraction,
            'avg_intent_score': avg_intent_score
        }

    # =========================================================================
    # Phase 3: STIMULATED EMISSION - Coherence detection
    # =========================================================================

    def _orbital_coherence(self, tau1: float, tau2: float) -> float:
        """
        Compute orbital coherence between two concepts.

        Concepts at the same orbital level have coherence = 1.
        Coherence decays exponentially with orbital distance.
        """
        n1 = int(round((tau1 - 1) * E))
        n2 = int(round((tau2 - 1) * E))
        delta_n = abs(n1 - n2)
        return np.exp(-delta_n / 2.0)  # Decay with orbital distance

    def stimulated_emission(self, excited: Dict[str, ExcitedState],
                            coherence_threshold: float = 0.3,
                            min_cluster_size: int = 3,
                            orbital_weight: float = 0.3
                            ) -> List[CoherentBeam]:
        """
        Stimulated emission: find coherent clusters.

        True laser coherence requires BOTH:
        1. j-vector alignment (polarization coherence)
        2. Orbital proximity (frequency coherence)

        Combined coherence = (1-w)*j_coherence + w*orbital_coherence

        Args:
            excited: Excited states from pumping
            coherence_threshold: Minimum combined coherence
            min_cluster_size: Minimum concepts per beam
            orbital_weight: Weight for orbital vs j-vector coherence (0-1)

        Returns:
            List of CoherentBeam (sorted by intensity)
        """
        if len(excited) < min_cluster_size:
            return []

        # Filter: prefer multi-source concepts (reached from multiple seeds = more central)
        # and concepts with meaningful j-vectors
        all_states = list(excited.values())

        # Score by: sources * visits * (1 + j_magnitude)
        for s in all_states:
            s.relevance = len(s.sources) * s.visits * (1 + np.linalg.norm(s.j))

        # Sort by relevance and take top concepts
        all_states.sort(key=lambda s: -s.relevance)

        # Take top 50% or minimum needed
        n_keep = max(min_cluster_size * 3, len(all_states) // 2)
        states = all_states[:n_keep]

        if len(states) < min_cluster_size:
            return []

        # Compute j-vector matrix
        js = np.array([s.j for s in states])
        taus = [s.tau for s in states]
        words = [s.word for s in states]

        # Normalize j-vectors
        norms = np.linalg.norm(js, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        js_normalized = js / norms

        # Compute j-vector coherence matrix (cosine similarity)
        j_coherence_matrix = js_normalized @ js_normalized.T

        # Compute orbital coherence matrix
        n_states = len(states)
        orbital_coherence_matrix = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                orbital_coherence_matrix[i, j] = self._orbital_coherence(taus[i], taus[j])

        # Combined coherence: weighted average of j and orbital coherence
        combined_coherence = (1 - orbital_weight) * j_coherence_matrix + orbital_weight * orbital_coherence_matrix

        # Find clusters by greedy coherence
        used = set()
        beams = []

        # Sort by visit count (population) - most excited first
        indices = sorted(range(len(states)), key=lambda i: -states[i].visits)

        for i in indices:
            if i in used:
                continue

            # Find all concepts coherent with this one
            cluster_indices = [i]
            for j in range(len(states)):
                if j != i and j not in used:
                    if combined_coherence[i, j] >= coherence_threshold:
                        cluster_indices.append(j)

            if len(cluster_indices) >= min_cluster_size:
                # Form a beam
                cluster_states = [states[idx] for idx in cluster_indices]
                cluster_words = [words[idx] for idx in cluster_indices]

                # Compute beam properties
                cluster_js = np.array([s.j for s in cluster_states])
                cluster_taus = [s.tau for s in cluster_states]
                cluster_affirmations = [s.affirmation for s in cluster_states]
                cluster_orbitals = [s.orbital for s in cluster_states]

                j_centroid = cluster_js.mean(axis=0)

                # Combined coherence = average pairwise
                n = len(cluster_indices)
                if n > 1:
                    pairwise_sum = sum(
                        combined_coherence[cluster_indices[a], cluster_indices[b]]
                        for a in range(n) for b in range(a+1, n)
                    )
                    coherence = pairwise_sum / (n * (n-1) / 2)
                else:
                    coherence = 1.0

                beam = CoherentBeam(
                    concepts=cluster_words,
                    j_centroid=j_centroid,
                    coherence=coherence,
                    affirmation=np.mean(cluster_affirmations),
                    tau_mean=np.mean(cluster_taus),
                    tau_spread=np.std(cluster_taus)
                )
                # Add orbital info to beam
                beam.orbital_mean = np.mean(cluster_orbitals)
                beam.orbital_spread = np.std(cluster_orbitals)
                beams.append(beam)

                # Mark as used
                used.update(cluster_indices)

        # Sort by intensity
        beams.sort(key=lambda b: -b.intensity)

        return beams

    # =========================================================================
    # Phase 4: COHERENT OUTPUT
    # =========================================================================

    def lase(self, seeds: List[str],
             pump_power: int = 10,
             pump_depth: int = 5,
             coherence_threshold: float = 0.3,
             min_cluster_size: int = 3,
             intent_verbs: List[str] = None
             ) -> Dict:
        """
        Full laser operation: pump -> analyze -> emit -> output.

        NEW: Can take intent_verbs to enable intent-driven navigation.

        Args:
            seeds: Input concepts
            pump_power: Exploration intensity
            pump_depth: Exploration depth
            coherence_threshold: Minimum j-alignment
            min_cluster_size: Minimum beam size
            intent_verbs: Optional verbs for intent collapse (NEW)

        Returns:
            {
                'beams': List[CoherentBeam],
                'population': Dict (statistics),
                'excited': Dict[str, ExcitedState],
                'intent': Dict (intent stats) - NEW
            }
        """
        # Set intent if provided
        intent_stats = None
        if intent_verbs:
            intent_stats = self.set_intent(intent_verbs)

        # Phase 1: Pump (now intent-aware)
        excited = self.pump(seeds, pump_power, pump_depth)

        # Phase 2: Analyze
        population = self.analyze_population(excited)

        # Phase 3: Stimulated emission
        beams = self.stimulated_emission(
            excited, coherence_threshold, min_cluster_size
        )

        # Phase 4: Compute laser metrics (now includes intent)
        metrics = self.compute_laser_metrics(population, beams)

        return {
            'beams': beams,
            'population': population,
            'excited': excited,
            'seeds': seeds,
            'metrics': metrics,
            # Intent information (NEW)
            'intent': {
                'enabled': self.intent_enabled,
                'verbs': self.intent_verbs,
                'stats': intent_stats
            }
        }

    def get_primary_beam(self, result: Dict) -> Optional[CoherentBeam]:
        """Get the most intense coherent beam."""
        beams = result.get('beams', [])
        return beams[0] if beams else None

    def get_beam_themes(self, beam: CoherentBeam) -> List[str]:
        """Interpret j-centroid as themes."""
        themes = []
        for i, dim in enumerate(self._j_dims):
            val = beam.j_centroid[i]
            if val > 0.1:
                themes.append(f"+{dim}")
            elif val < -0.1:
                themes.append(f"-{dim}")
        return themes

    def compute_laser_metrics(self, population: Dict, beams: List[CoherentBeam]) -> Dict:
        """
        Compute combined Euler-Laser metrics with intent collapse.

        Nuclear laser formula:
        Output_coherence = pump_energy × medium_quality × mirror_alignment × intent_focus

        pump_energy      = veil_crossings × (1 - human_fraction)
        medium_quality   = count(τ > e) / total_states
        mirror_alignment = mean(j_coherence across beams)
        intent_focus     = fraction of states reached via intent collapse (NEW)
        """
        total_states = population.get('total_excited', 1)
        above_veil = population.get('above_veil', 0)
        human_fraction = population.get('human_fraction', 0.5)

        # Pump energy: how much excitation reached transcendental levels
        # More veil crossings and less human fraction = more pump energy
        pump_energy = above_veil * (1 - human_fraction) if above_veil > 0 else 0.1

        # Medium quality: fraction of states above the Veil (lasing medium)
        medium_quality = above_veil / total_states if total_states > 0 else 0.0

        # Mirror alignment: average coherence across beams (j-vector alignment)
        if beams:
            mirror_alignment = np.mean([b.coherence for b in beams])
        else:
            mirror_alignment = 0.0

        # Intent focus: fraction of states reached via intent collapse (NEW)
        intent_fraction = population.get('intent_fraction', 0.0)
        intent_focus = 0.5 + 0.5 * intent_fraction  # Range [0.5, 1.0]

        # Output power: combined metric (now includes intent focus)
        output_power = pump_energy * (0.1 + medium_quality) * (0.1 + mirror_alignment) * intent_focus

        # Spectral purity: how narrow is the orbital distribution
        orbital_dist = population.get('orbital_dist', {})
        if orbital_dist:
            total_in_orbitals = sum(orbital_dist.values())
            dominant_count = max(orbital_dist.values())
            spectral_purity = dominant_count / total_in_orbitals if total_in_orbitals > 0 else 0.0
        else:
            spectral_purity = 0.0

        # Lasing threshold: did we achieve coherent output?
        lasing_achieved = len(beams) > 0 and mirror_alignment > 0.5

        return {
            'pump_energy': pump_energy,
            'medium_quality': medium_quality,
            'mirror_alignment': mirror_alignment,
            'output_power': output_power,
            'spectral_purity': spectral_purity,
            'lasing_achieved': lasing_achieved,
            'beam_count': len(beams),
            # Intent metrics (NEW)
            'intent_fraction': intent_fraction,
            'intent_focus': intent_focus
        }

    def close(self):
        """Close graph connection."""
        if self.graph:
            self.graph.close()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate Euler-aware semantic laser."""
    print("=" * 70)
    print("EULER-AWARE SEMANTIC LASER")
    print("=" * 70)
    print()
    print(f"  Euler Constants:")
    print(f"    e = {E:.4f} (orbital spacing = 1/e)")
    print(f"    kT = {KT_NATURAL:.2f} (natural temperature)")
    print(f"    Veil at τ = e ≈ {VEIL_TAU:.2f}")
    print()

    laser = SemanticLaser()

    # Test with dream symbols
    seeds = ['myth', 'hero', 'monster', 'eye', 'portal', 'mountain', 'maze']

    print(f"Seeds: {seeds}")
    print()

    # Lase!
    result = laser.lase(
        seeds,
        pump_power=8,
        pump_depth=4,
        coherence_threshold=0.25,
        min_cluster_size=3
    )

    # Population stats with Euler info
    pop = result['population']
    print(f"POPULATION (Euler Analysis):")
    print(f"  Excited states: {pop['total_excited']}")
    print(f"  τ range: {pop['tau_min']:.2f} - {pop['tau_max']:.2f}")
    print(f"  τ mean: {pop['tau_mean']:.2f} ± {pop['tau_std']:.2f}")
    print(f"  Mean orbital: n = {pop['mean_orbital']:.1f}")
    print(f"  Dominant orbital: n = {pop['dominant_orbital']}")
    print(f"  Human realm: {pop['human_fraction']:.1%} (below Veil)")
    print(f"  A mean: {pop['affirmation_mean']:.2f}")
    print(f"  Multi-source concepts: {pop['multi_source']}")

    # Show orbital distribution
    print(f"\n  Orbital Distribution:")
    for n in sorted(pop['orbital_dist'].keys()):
        count = pop['orbital_dist'][n]
        bar = '█' * min(count, 30)
        veil_marker = " ← VEIL" if n == int(round((VEIL_TAU - 1) * E)) else ""
        print(f"    n={n}: {bar} ({count}){veil_marker}")
    print()

    # Beams with orbital info
    print(f"COHERENT BEAMS: {len(result['beams'])}")
    for i, beam in enumerate(result['beams'][:3]):
        themes = laser.get_beam_themes(beam)
        realm = "human" if beam.tau_mean < VEIL_TAU else "transcendental"
        print(f"\n  Beam {i+1} ({realm}):")
        print(f"    Concepts: {beam.concepts[:8]}")
        print(f"    Coherence: {beam.coherence:.2f}")
        print(f"    Intensity: {beam.intensity:.1f}")
        print(f"    A-polarity: {beam.affirmation:+.2f}")
        print(f"    τ: {beam.tau_mean:.2f} ± {beam.tau_spread:.2f}")
        if hasattr(beam, 'orbital_mean'):
            print(f"    Orbital: n={beam.orbital_mean:.1f} ± {beam.orbital_spread:.1f}")
        print(f"    Themes: {themes}")

    # Laser metrics
    metrics = result['metrics']
    print()
    print("=" * 70)
    print("LASER METRICS")
    print("=" * 70)
    print(f"  Pump energy:       {metrics['pump_energy']:.2f}")
    print(f"  Medium quality:    {metrics['medium_quality']:.2%}")
    print(f"  Mirror alignment:  {metrics['mirror_alignment']:.2f}")
    print(f"  Spectral purity:   {metrics['spectral_purity']:.2%}")
    print(f"  Output power:      {metrics['output_power']:.3f}")
    print(f"  Lasing achieved:   {'YES ✓' if metrics['lasing_achieved'] else 'NO ✗'}")

    laser.close()


if __name__ == "__main__":
    demo()
