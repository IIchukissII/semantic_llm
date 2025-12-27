"""
Semantic Laser: Coherent Meaning Extraction

Inspired by laser physics:
1. PUMPING - Broad exploration, excite many concepts
2. POPULATION - Collect excited states with properties
3. STIMULATED EMISSION - j-vector alignment triggers coherence
4. COHERENT OUTPUT - Aligned concepts form meaning beam

Theory-based approach:
- No tau filtering during exploration
- Use g, tau, j for coherence detection
- Polarization selects aligned meaning
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


@dataclass
class ExcitedState:
    """A concept in excited state after pumping."""
    word: str
    tau: float
    g: float
    j: np.ndarray  # 5D j-vector
    visits: int = 1
    sources: Set[str] = field(default_factory=set)  # Which seeds reached this

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
    g_polarity: float  # Average g (good/evil)
    tau_mean: float  # Average abstraction
    tau_spread: float  # Spread in abstraction

    @property
    def intensity(self) -> float:
        """Beam intensity = coherence * number of concepts."""
        return self.coherence * len(self.concepts)


class SemanticLaser:
    """
    Semantic Laser for coherent meaning extraction.

    Unlike random walk that finds high-connectivity nodes,
    this finds semantically COHERENT clusters using j-vector alignment.
    """

    def __init__(self, graph: MeaningGraph = None):
        self.graph = graph or MeaningGraph()
        self._j_dims = ['beauty', 'life', 'sacred', 'good', 'love']
        print(f"[SemanticLaser] Using j-vectors from Neo4j")

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

    def pump(self, seeds: List[str],
             pump_power: int = 10,  # walks per seed
             pump_depth: int = 5    # steps per walk
             ) -> Dict[str, ExcitedState]:
        """
        Pumping phase: excite concepts through broad exploration.

        No tau filtering - explore everything reachable.

        Args:
            seeds: Starting concepts
            pump_power: Number of random walks per seed
            pump_depth: Steps per walk

        Returns:
            {word: ExcitedState} - all excited concepts
        """
        excited = {}  # word -> ExcitedState

        for seed in seeds:
            # Get seed properties
            concept = self.graph.get_concept(seed)
            if not concept:
                continue

            seed_state = ExcitedState(
                word=seed,
                tau=concept.get('tau', 2.0),
                g=concept.get('g', 0.0),
                j=self._get_j_vector(concept.get('j')),
                visits=1,
                sources={seed}
            )
            excited[seed] = seed_state

            # Random walks from this seed
            for _ in range(pump_power):
                current = seed
                for _ in range(pump_depth):
                    # Get ALL transitions (no tau filter)
                    neighbors = self._get_neighbors_unfiltered(current)
                    if not neighbors:
                        break

                    # Random selection (uniform - we filter later by coherence)
                    next_word = np.random.choice([n[0] for n in neighbors])

                    # Get properties
                    next_concept = self.graph.get_concept(next_word)
                    if not next_concept:
                        continue

                    # Add to excited states
                    if next_word in excited:
                        excited[next_word].visits += 1
                        excited[next_word].sources.add(seed)
                    else:
                        excited[next_word] = ExcitedState(
                            word=next_word,
                            tau=next_concept.get('tau', 2.0),
                            g=next_concept.get('g', 0.0),
                            j=self._get_j_vector(next_concept.get('j')),
                            visits=1,
                            sources={seed}
                        )

                    current = next_word

        return excited

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
        Analyze the excited population.

        Returns statistics about the excited states.
        """
        if not excited:
            return {}

        states = list(excited.values())

        # Tau distribution (energy levels)
        taus = [s.tau for s in states]

        # G distribution (polarity)
        gs = [s.g for s in states]

        # J-vectors
        js = np.array([s.j for s in states])
        j_mean = js.mean(axis=0)

        # Visit distribution
        visits = [s.visits for s in states]

        return {
            'total_excited': len(states),
            'tau_mean': np.mean(taus),
            'tau_std': np.std(taus),
            'tau_min': np.min(taus),
            'tau_max': np.max(taus),
            'g_mean': np.mean(gs),
            'g_std': np.std(gs),
            'j_centroid': j_mean,
            'j_magnitude': np.linalg.norm(j_mean),
            'total_visits': sum(visits),
            'max_visits': max(visits),
            'multi_source': sum(1 for s in states if len(s.sources) > 1)
        }

    # =========================================================================
    # Phase 3: STIMULATED EMISSION - Coherence detection
    # =========================================================================

    def stimulated_emission(self, excited: Dict[str, ExcitedState],
                            coherence_threshold: float = 0.3,
                            min_cluster_size: int = 3
                            ) -> List[CoherentBeam]:
        """
        Stimulated emission: find coherent clusters.

        Concepts with aligned j-vectors form coherent beams.
        Filter out noise (zero j-vectors, single-source, low visits).

        Args:
            excited: Excited states from pumping
            coherence_threshold: Minimum j-vector alignment (cosine similarity)
            min_cluster_size: Minimum concepts per beam

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
        words = [s.word for s in states]

        # Normalize j-vectors
        norms = np.linalg.norm(js, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        js_normalized = js / norms

        # Compute coherence matrix (cosine similarity)
        coherence_matrix = js_normalized @ js_normalized.T

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
                    if coherence_matrix[i, j] >= coherence_threshold:
                        cluster_indices.append(j)

            if len(cluster_indices) >= min_cluster_size:
                # Form a beam
                cluster_states = [states[idx] for idx in cluster_indices]
                cluster_words = [words[idx] for idx in cluster_indices]

                # Compute beam properties
                cluster_js = np.array([s.j for s in cluster_states])
                cluster_taus = [s.tau for s in cluster_states]
                cluster_gs = [s.g for s in cluster_states]

                j_centroid = cluster_js.mean(axis=0)

                # Coherence = average pairwise similarity
                n = len(cluster_indices)
                if n > 1:
                    pairwise_sum = sum(
                        coherence_matrix[cluster_indices[a], cluster_indices[b]]
                        for a in range(n) for b in range(a+1, n)
                    )
                    coherence = pairwise_sum / (n * (n-1) / 2)
                else:
                    coherence = 1.0

                beam = CoherentBeam(
                    concepts=cluster_words,
                    j_centroid=j_centroid,
                    coherence=coherence,
                    g_polarity=np.mean(cluster_gs),
                    tau_mean=np.mean(cluster_taus),
                    tau_spread=np.std(cluster_taus)
                )
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
             min_cluster_size: int = 3
             ) -> Dict:
        """
        Full laser operation: pump -> analyze -> emit -> output.

        Args:
            seeds: Input concepts
            pump_power: Exploration intensity
            pump_depth: Exploration depth
            coherence_threshold: Minimum j-alignment
            min_cluster_size: Minimum beam size

        Returns:
            {
                'beams': List[CoherentBeam],
                'population': Dict (statistics),
                'excited': Dict[str, ExcitedState]
            }
        """
        # Phase 1: Pump
        excited = self.pump(seeds, pump_power, pump_depth)

        # Phase 2: Analyze
        population = self.analyze_population(excited)

        # Phase 3: Stimulated emission
        beams = self.stimulated_emission(
            excited, coherence_threshold, min_cluster_size
        )

        return {
            'beams': beams,
            'population': population,
            'excited': excited,
            'seeds': seeds
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

    def close(self):
        """Close graph connection."""
        if self.graph:
            self.graph.close()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate semantic laser."""
    print("=" * 70)
    print("SEMANTIC LASER")
    print("=" * 70)
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

    # Population stats
    pop = result['population']
    print(f"POPULATION:")
    print(f"  Excited states: {pop['total_excited']}")
    print(f"  τ range: {pop['tau_min']:.2f} - {pop['tau_max']:.2f}")
    print(f"  τ mean: {pop['tau_mean']:.2f} ± {pop['tau_std']:.2f}")
    print(f"  g mean: {pop['g_mean']:.2f}")
    print(f"  Multi-source concepts: {pop['multi_source']}")
    print()

    # Beams
    print(f"COHERENT BEAMS: {len(result['beams'])}")
    for i, beam in enumerate(result['beams'][:3]):
        themes = laser.get_beam_themes(beam)
        print(f"\n  Beam {i+1}:")
        print(f"    Concepts: {beam.concepts[:8]}")
        print(f"    Coherence: {beam.coherence:.2f}")
        print(f"    Intensity: {beam.intensity:.1f}")
        print(f"    g-polarity: {beam.g_polarity:+.2f}")
        print(f"    τ: {beam.tau_mean:.2f} ± {beam.tau_spread:.2f}")
        print(f"    Themes: {themes}")

    laser.close()


if __name__ == "__main__":
    demo()
