#!/usr/bin/env python3
"""
Semantic Optics

Exploring optical analogies in semantic space:

1. REFRACTIVE INDEX n(τ): How "dense" is meaning at each τ-level?
   - Dense regions slow meaning propagation
   - n = c/v where v is semantic velocity

2. REFRACTION: Meaning bends at τ-boundaries
   - Snell's law: n₁ sin(θ₁) = n₂ sin(θ₂)
   - Meaning paths bend toward normal in dense regions

3. LOGOS AS LENS: Focusing through j-good alignment
   - Focal length: how strongly it focuses
   - Aperture: range of j-directions accepted
   - Aberrations: imperfect focusing

4. INTERFERENCE: Multiple meaning paths combine
   - Constructive: paths reinforce
   - Destructive: paths cancel

5. POLARIZATION: j-direction filtering
   - Logos as polarizer
   - j-good as polarization axis

"Meaning refracts through the lens of understanding"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import sys

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from chain_core.storm_logos import Storm, Logos, StormLogosBuilder, StormResult
from core.data_loader import DataLoader


@dataclass
class RefractiveIndex:
    """Refractive index at a τ-level."""
    tau_level: int
    n: float  # refractive index
    density: float  # concept density
    avg_degree: float  # connectivity
    transition_speed: float  # how fast meaning moves


@dataclass
class LensProperties:
    """Optical properties of the Logos lens."""
    focal_length: float  # focusing power
    aperture: float  # acceptance angle (j-spread)
    magnification: float  # coherence amplification
    aberration: float  # deviation from ideal focus
    polarization_efficiency: float  # j-alignment filtering


@dataclass
class InterferencePattern:
    """Interference between meaning paths."""
    concept: str
    n_paths: int
    path_phases: List[float]  # j-alignment of each path
    constructive: bool  # paths reinforce?
    amplitude: float  # combined strength


class SemanticOptics:
    """Explore optical properties of semantic space."""

    def __init__(self):
        self.graph = MeaningGraph()
        self.loader = DataLoader()
        self._j_good = None
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

    def _load_j_good(self):
        if self._j_good is None:
            self._j_good = np.array(self.loader.get_j_good())

    def _query(self, query: str, params: dict = None) -> list:
        if not self.graph.is_connected():
            return []
        with self.graph.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(r) for r in result]

    # =========================================================================
    # REFRACTIVE INDEX
    # =========================================================================

    def measure_refractive_index(self) -> List[RefractiveIndex]:
        """
        Measure refractive index at each τ-level.

        n(τ) = c / v(τ)

        where v(τ) is the "velocity" of meaning propagation:
        - High connectivity = fast propagation = low n
        - Low connectivity = slow propagation = high n
        """
        print("\n--- MEASURING REFRACTIVE INDEX ---")

        query = """
        MATCH (c:Concept)
        WHERE c.tau IS NOT NULL
        WITH c, COUNT { (c)-[:VIA]-() } as degree
        RETURN round(c.tau) as tau_bin,
               count(*) as concept_count,
               avg(degree) as avg_degree
        ORDER BY tau_bin
        """

        records = self._query(query)
        total_concepts = sum(r['concept_count'] for r in records)

        # Reference velocity at τ=3 (middle)
        mid_record = next((r for r in records if r['tau_bin'] == 3), None)
        v_ref = mid_record['avg_degree'] if mid_record else 10.0

        indices = []
        for r in records:
            tau = int(r['tau_bin'])
            density = r['concept_count'] / total_concepts
            avg_degree = r['avg_degree']

            # Velocity proportional to connectivity
            v = avg_degree + 0.1  # avoid division by zero

            # Refractive index inversely proportional to velocity
            # Normalize so n(τ=3) ≈ 1.0
            n = v_ref / v

            indices.append(RefractiveIndex(
                tau_level=tau,
                n=n,
                density=density,
                avg_degree=avg_degree,
                transition_speed=v
            ))

        return indices

    def print_refractive_indices(self, indices: List[RefractiveIndex]):
        """Print refractive index table."""
        print(f"\n{'τ':>4} {'n':>8} {'Density':>10} {'Degree':>10} {'v':>10}")
        print("-" * 45)

        for idx in indices:
            print(f"{idx.tau_level:>4} {idx.n:>8.3f} {idx.density:>10.3%} "
                  f"{idx.avg_degree:>10.1f} {idx.transition_speed:>10.1f}")

        # Interpretation
        low_tau = [i for i in indices if i.tau_level <= 2]
        high_tau = [i for i in indices if i.tau_level >= 5]

        if low_tau and high_tau:
            n_low = np.mean([i.n for i in low_tau])
            n_high = np.mean([i.n for i in high_tau])
            print(f"\nAverage n(τ≤2) = {n_low:.3f} (ground)")
            print(f"Average n(τ≥5) = {n_high:.3f} (sky)")

            if n_low < n_high:
                print("\n→ Ground is 'optically thin' (fast propagation)")
                print("→ Sky is 'optically dense' (slow propagation)")
            else:
                print("\n→ Ground is 'optically dense' (slow propagation)")
                print("→ Sky is 'optically thin' (fast propagation)")

    # =========================================================================
    # REFRACTION (SNELL'S LAW)
    # =========================================================================

    def measure_refraction(self) -> Dict:
        """
        Measure how meaning "bends" at τ-boundaries.

        When crossing from τ₁ to τ₂:
        - If n₂ > n₁: meaning bends toward normal (slows down)
        - If n₂ < n₁: meaning bends away from normal (speeds up)
        """
        print("\n--- MEASURING REFRACTION AT τ-BOUNDARIES ---")

        # Get transitions across τ-boundaries
        query = """
        MATCH (s:Concept)-[r:VIA]->(t:Concept)
        WHERE s.tau IS NOT NULL AND t.tau IS NOT NULL
        WITH round(s.tau) as s_tau, round(t.tau) as t_tau,
             s.j as s_j, t.j as t_j, count(*) as cnt
        WHERE s_tau <> t_tau
        RETURN s_tau, t_tau, cnt
        ORDER BY s_tau, t_tau
        """

        records = self._query(query)

        # Build transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        for r in records:
            transitions[int(r['s_tau'])][int(r['t_tau'])] = r['cnt']

        print("\nTransition matrix (from → to):")
        taus = sorted(set(list(transitions.keys()) +
                         [t for d in transitions.values() for t in d.keys()]))

        header = "    " + " ".join(f"{t:>6}" for t in taus)
        print(header)
        print("-" * len(header))

        for s_tau in taus:
            row = f"{s_tau:>3}:"
            for t_tau in taus:
                cnt = transitions[s_tau][t_tau]
                if cnt > 0:
                    row += f" {cnt:>6}"
                else:
                    row += "      ."
            print(row)

        # Compute "bending" - preference for certain τ-jumps
        up_jumps = sum(cnt for s in transitions for t, cnt in transitions[s].items() if t > s)
        down_jumps = sum(cnt for s in transitions for t, cnt in transitions[s].items() if t < s)

        print(f"\nTotal upward jumps (→ high τ): {up_jumps}")
        print(f"Total downward jumps (→ low τ): {down_jumps}")

        if up_jumps > down_jumps:
            print("\n→ Meaning tends to refract UPWARD (toward sky)")
        else:
            print("\n→ Meaning tends to refract DOWNWARD (toward ground)")

        return {
            "transitions": dict(transitions),
            "up_jumps": up_jumps,
            "down_jumps": down_jumps
        }

    # =========================================================================
    # LOGOS AS LENS
    # =========================================================================

    def measure_lens_properties(self, seeds: List[str],
                                 n_trials: int = 10) -> LensProperties:
        """
        Measure optical properties of the Logos lens.

        The lens focuses chaotic storm thoughts into coherent pattern.
        """
        print("\n--- MEASURING LOGOS LENS PROPERTIES ---")
        self._load_j_good()

        focal_lengths = []
        apertures = []
        magnifications = []
        aberrations = []
        polarization_effs = []

        for trial in range(n_trials):
            builder = StormLogosBuilder(storm_temperature=1.5, n_walks=5)

            try:
                # Generate storm
                storm = builder.storm.generate(seeds, n_walks=5, steps_per_walk=8)

                # Focus through lens
                pattern = builder.logos.focus(storm, intent_j=None)

                # Measure focal length (how concentrated is the focus?)
                # Fewer core concepts = shorter focal length = stronger focusing
                focal_length = len(pattern.core_concepts)
                focal_lengths.append(focal_length)

                # Measure aperture (j-spread of accepted concepts)
                if storm.thoughts:
                    j_vectors = [t.j for t in storm.thoughts if t.j is not None]
                    if j_vectors:
                        j_spread = np.std([np.linalg.norm(j) for j in j_vectors])
                        apertures.append(j_spread)

                # Magnification (coherence amplification)
                # Storm has random coherence, lens creates order
                if storm.thoughts:
                    # Storm "coherence" - random j-alignment
                    storm_js = [t.j for t in storm.thoughts[:20] if t.j is not None]
                    if len(storm_js) >= 2:
                        storm_center = np.mean(storm_js, axis=0)
                        storm_coh = np.mean([
                            np.dot(j / (np.linalg.norm(j) + 1e-6),
                                   storm_center / (np.linalg.norm(storm_center) + 1e-6))
                            for j in storm_js
                        ])
                        magnification = pattern.coherence / max(0.1, storm_coh)
                        magnifications.append(magnification)

                # Aberration (deviation from j-good)
                if np.linalg.norm(pattern.j_center) > 1e-6:
                    j_norm = pattern.j_center / np.linalg.norm(pattern.j_center)
                    aberration = 1 - abs(np.dot(j_norm, self._j_good))
                    aberrations.append(aberration)

                # Polarization efficiency (j-good alignment)
                pol_eff = pattern.g_direction  # g correlates with j-good alignment
                polarization_effs.append(pol_eff)

            finally:
                builder.close()

        return LensProperties(
            focal_length=np.mean(focal_lengths) if focal_lengths else 0,
            aperture=np.mean(apertures) if apertures else 0,
            magnification=np.mean(magnifications) if magnifications else 1,
            aberration=np.mean(aberrations) if aberrations else 0,
            polarization_efficiency=np.mean(polarization_effs) if polarization_effs else 0
        )

    def print_lens_properties(self, props: LensProperties):
        """Print lens properties."""
        print(f"\nLogos Lens Properties:")
        print(f"  Focal length:    {props.focal_length:.2f} concepts")
        print(f"  Aperture:        {props.aperture:.3f} (j-spread)")
        print(f"  Magnification:   {props.magnification:.2f}x (coherence gain)")
        print(f"  Aberration:      {props.aberration:.3f} (j-good deviation)")
        print(f"  Polarization:    {props.polarization_efficiency:+.3f} (g-direction)")

        # Interpretation
        print("\nInterpretation:")
        if props.focal_length < 5:
            print("  • Short focal length = STRONG focusing (concentrated meaning)")
        else:
            print("  • Long focal length = WEAK focusing (diffuse meaning)")

        if props.magnification > 1.5:
            print("  • High magnification = lens AMPLIFIES coherence")
        else:
            print("  • Low magnification = lens MAINTAINS coherence")

        if props.aberration < 0.3:
            print("  • Low aberration = good j-good alignment")
        else:
            print("  • High aberration = focus deviates from j-good")

    # =========================================================================
    # INTERFERENCE
    # =========================================================================

    def measure_interference(self, seeds: List[str]) -> List[InterferencePattern]:
        """
        Measure interference when multiple meaning paths reach same concept.

        Constructive: paths with similar j-direction reinforce
        Destructive: paths with opposite j-direction cancel
        """
        print("\n--- MEASURING INTERFERENCE PATTERNS ---")
        self._load_j_good()

        # Find concepts reachable by multiple paths from seeds
        path_data = defaultdict(list)  # concept -> list of (j, path)

        storm = Storm(temperature=1.5)

        for seed in seeds:
            seed_concept = self.graph.get_concept(seed)
            if not seed_concept:
                continue

            seed_j = np.array(seed_concept.get('j', [0]*5))

            # Multiple walks from this seed
            for walk in range(10):
                current = seed
                current_j = seed_j
                path = [seed]

                for step in range(8):
                    transitions = storm._get_transitions(current)
                    next_word = storm._sample_next(transitions)

                    if not next_word:
                        break

                    next_concept = self.graph.get_concept(next_word)
                    if not next_concept:
                        break

                    next_j = np.array(next_concept.get('j', [0]*5))
                    path.append(next_word)

                    # Record this path reaching this concept
                    path_data[next_word].append({
                        'j': next_j,
                        'path': path.copy(),
                        'source': seed
                    })

                    current = next_word
                    current_j = next_j

        storm.close()

        # Find concepts with multiple paths
        patterns = []
        for concept, paths in path_data.items():
            if len(paths) < 2:
                continue

            # Compute phases (j-alignment with j_good)
            phases = []
            for p in paths:
                j = p['j']
                if np.linalg.norm(j) > 1e-6:
                    phase = np.dot(j / np.linalg.norm(j), self._j_good)
                    phases.append(phase)

            if len(phases) < 2:
                continue

            # Check for constructive/destructive interference
            phase_std = np.std(phases)
            avg_phase = np.mean(phases)

            # Constructive if phases are similar (low std)
            constructive = phase_std < 0.3

            # Amplitude combines based on phase alignment
            if constructive:
                amplitude = len(phases) * abs(avg_phase)  # Reinforce
            else:
                amplitude = abs(sum(phases))  # May cancel

            patterns.append(InterferencePattern(
                concept=concept,
                n_paths=len(paths),
                path_phases=phases,
                constructive=constructive,
                amplitude=amplitude
            ))

        # Sort by number of paths
        patterns.sort(key=lambda p: -p.n_paths)

        return patterns[:20]  # Top 20

    def print_interference_patterns(self, patterns: List[InterferencePattern]):
        """Print interference patterns."""
        print(f"\n{'Concept':<15} {'Paths':>6} {'Type':>12} {'Amplitude':>10}")
        print("-" * 45)

        n_constructive = 0
        n_destructive = 0

        for p in patterns[:15]:
            ptype = "constructive" if p.constructive else "destructive"
            print(f"{p.concept:<15} {p.n_paths:>6} {ptype:>12} {p.amplitude:>10.3f}")

            if p.constructive:
                n_constructive += 1
            else:
                n_destructive += 1

        print(f"\nTotal: {n_constructive} constructive, {n_destructive} destructive")

        if n_constructive > n_destructive:
            print("→ Meaning paths tend to REINFORCE (coherent propagation)")
        else:
            print("→ Meaning paths tend to INTERFERE (divergent meanings)")

    # =========================================================================
    # POLARIZATION
    # =========================================================================

    def measure_polarization(self) -> Dict:
        """
        Measure j-direction distribution (polarization).

        Logos acts as polarizer, selecting j-good aligned concepts.
        """
        print("\n--- MEASURING SEMANTIC POLARIZATION ---")
        self._load_j_good()

        query = """
        MATCH (c:Concept)
        WHERE c.j IS NOT NULL AND c.tau IS NOT NULL
        RETURN c.word as word, c.j as j, c.tau as tau, c.g as g
        LIMIT 5000
        """

        records = self._query(query)

        # Compute j-good alignment for each concept
        alignments = []
        for r in records:
            j = np.array(r['j'])
            if np.linalg.norm(j) > 1e-6:
                j_norm = j / np.linalg.norm(j)
                alignment = np.dot(j_norm, self._j_good)
                alignments.append({
                    'word': r['word'],
                    'alignment': alignment,
                    'tau': r['tau'],
                    'g': r['g']
                })

        # Distribution of alignments
        align_values = [a['alignment'] for a in alignments]

        print(f"\nPolarization distribution (j · j_good):")
        print(f"  Mean alignment:   {np.mean(align_values):+.3f}")
        print(f"  Std alignment:    {np.std(align_values):.3f}")
        print(f"  % aligned (>0):   {sum(1 for a in align_values if a > 0) / len(align_values) * 100:.1f}%")
        print(f"  % anti-aligned:   {sum(1 for a in align_values if a < 0) / len(align_values) * 100:.1f}%")

        # Histogram
        bins = [-1, -0.5, 0, 0.5, 1]
        hist = [sum(1 for a in align_values if bins[i] <= a < bins[i+1])
                for i in range(len(bins)-1)]

        print(f"\n  Histogram of j-good alignment:")
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            bar = "█" * int(hist[i] / max(hist) * 30)
            print(f"    [{lo:+.1f}, {hi:+.1f}): {bar} ({hist[i]})")

        # Polarization by τ-level
        print(f"\n  Polarization by τ-level:")
        tau_align = defaultdict(list)
        for a in alignments:
            tau_bin = int(round(a['tau']))
            tau_align[tau_bin].append(a['alignment'])

        for tau in sorted(tau_align.keys()):
            avg = np.mean(tau_align[tau])
            print(f"    τ={tau}: {avg:+.3f}")

        return {
            'mean_alignment': np.mean(align_values),
            'std_alignment': np.std(align_values),
            'positive_fraction': sum(1 for a in align_values if a > 0) / len(align_values),
            'by_tau': {tau: np.mean(vals) for tau, vals in tau_align.items()}
        }

    # =========================================================================
    # FULL ANALYSIS
    # =========================================================================

    def run_full_analysis(self, seeds: List[str] = None):
        """Run complete optical analysis."""
        if seeds is None:
            seeds = ["love", "truth", "beauty", "wisdom"]

        print("\n" + "=" * 70)
        print("SEMANTIC OPTICS ANALYSIS")
        print("=" * 70)
        print(f"Seeds: {seeds}")

        # 1. Refractive index
        print("\n" + "=" * 50)
        print("1. REFRACTIVE INDEX n(τ)")
        print("=" * 50)
        indices = self.measure_refractive_index()
        self.print_refractive_indices(indices)

        # 2. Refraction
        print("\n" + "=" * 50)
        print("2. REFRACTION AT τ-BOUNDARIES")
        print("=" * 50)
        refraction = self.measure_refraction()

        # 3. Lens properties
        print("\n" + "=" * 50)
        print("3. LOGOS LENS PROPERTIES")
        print("=" * 50)
        lens = self.measure_lens_properties(seeds)
        self.print_lens_properties(lens)

        # 4. Interference
        print("\n" + "=" * 50)
        print("4. INTERFERENCE PATTERNS")
        print("=" * 50)
        patterns = self.measure_interference(seeds)
        self.print_interference_patterns(patterns)

        # 5. Polarization
        print("\n" + "=" * 50)
        print("5. SEMANTIC POLARIZATION")
        print("=" * 50)
        polarization = self.measure_polarization()

        # Summary
        print("\n" + "=" * 70)
        print("OPTICAL SUMMARY")
        print("=" * 70)

        print(f"""
SEMANTIC OPTICS FINDINGS:

1. REFRACTIVE INDEX:
   - Ground (τ≤2) n = {np.mean([i.n for i in indices if i.tau_level <= 2]):.3f}
   - Sky (τ≥5)    n = {np.mean([i.n for i in indices if i.tau_level >= 5]):.3f}
   - Meaning propagates faster where connectivity is high

2. REFRACTION:
   - {refraction['up_jumps']} upward jumps vs {refraction['down_jumps']} downward
   - {"Meaning refracts DOWNWARD (toward ground)" if refraction['down_jumps'] > refraction['up_jumps'] else "Meaning refracts UPWARD"}

3. LOGOS LENS:
   - Focal length: {lens.focal_length:.1f} concepts
   - Magnification: {lens.magnification:.2f}x coherence gain
   - Aberration: {lens.aberration:.3f}

4. INTERFERENCE:
   - {sum(1 for p in patterns if p.constructive)} constructive patterns
   - {sum(1 for p in patterns if not p.constructive)} destructive patterns
   - Meaning paths {"reinforce" if sum(1 for p in patterns if p.constructive) > len(patterns)/2 else "interfere"}

5. POLARIZATION:
   - Mean j-good alignment: {polarization['mean_alignment']:+.3f}
   - {polarization['positive_fraction']*100:.1f}% concepts aligned with j-good
""")

        # Save results
        self._save_results(indices, lens, patterns, polarization, refraction)

    def _save_results(self, indices, lens, patterns, polarization, refraction):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"optics_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "refractive_indices": [
                {"tau": i.tau_level, "n": i.n, "density": i.density}
                for i in indices
            ],
            "lens": {
                "focal_length": lens.focal_length,
                "magnification": lens.magnification,
                "aberration": lens.aberration,
                "polarization_efficiency": lens.polarization_efficiency
            },
            "interference": {
                "n_constructive": sum(1 for p in patterns if p.constructive),
                "n_destructive": sum(1 for p in patterns if not p.constructive)
            },
            "polarization": polarization,
            "refraction": {
                "up_jumps": refraction['up_jumps'],
                "down_jumps": refraction['down_jumps']
            }
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def close(self):
        if self.graph:
            self.graph.close()


def main():
    optics = SemanticOptics()
    try:
        optics.run_full_analysis(["love", "truth", "beauty", "wisdom"])
    finally:
        optics.close()


if __name__ == "__main__":
    main()
