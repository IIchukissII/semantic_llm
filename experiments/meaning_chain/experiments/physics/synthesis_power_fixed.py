#!/usr/bin/env python3
"""
Synthesis-Power Uncertainty: Fixed Version
============================================

Uses Monte Carlo attractors (non-poles) as synthesis concepts
instead of relying on the strict _find_synthesis method.

The key insight:
    All non-pole attractors are potential synthesis concepts -
    they're stable concepts that coexist with the paradox.

Hypothesis:
    C_synthesis × P_poles ≈ constant
    "The harder the paradox, the messier the resolution"

Usage:
    python synthesis_power_fixed.py
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import sys

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from graph.meaning_graph import MeaningGraph

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


@dataclass
class Measurement:
    """Single C×P measurement."""
    question: str
    thesis: str
    antithesis: str

    # Power metrics
    pole_tension: float
    pole_power: float

    # Coherence metrics
    synthesis_concepts: List[str]
    synthesis_coherence: float

    # The product
    c_times_p: float


class SynthesisPowerTest:
    """Fixed synthesis-power uncertainty test."""

    def __init__(self):
        self.graph = MeaningGraph()
        self.measurements: List[Measurement] = []
        self._j_cache: Dict[str, np.ndarray] = {}
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

    def get_j(self, word: str) -> Optional[np.ndarray]:
        """Get j-vector for a concept."""
        if word in self._j_cache:
            return self._j_cache[word]

        concept = self.graph.get_concept(word)
        if not concept or not concept.get('j'):
            return None

        j = np.array(concept['j'])
        if len(j) != 5:
            return None

        self._j_cache[word] = j
        return j

    def cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity."""
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def measure(self, question: str) -> Optional[Measurement]:
        """Measure C and P for a question."""
        try:
            from chain_core.monte_carlo_renderer import MonteCarloRenderer
            from chain_core.paradox_detector import ParadoxDetector
        except ImportError as e:
            print(f"[ERROR] Import failed: {e}")
            return None

        # Get paradox for poles
        detector = ParadoxDetector(n_samples=25)

        try:
            landscape = detector.detect(question)

            if not landscape.strongest:
                return None

            paradox = landscape.strongest
            thesis = paradox.thesis
            antithesis = paradox.antithesis

            # Compute pole tension and power
            j_thesis = self.get_j(thesis)
            j_anti = self.get_j(antithesis)

            if j_thesis is None or j_anti is None:
                return None

            pole_tension = -self.cosine_sim(j_thesis, j_anti)
            pole_power = max(0, pole_tension * paradox.stability * 10)

            # Get ALL attractors from the landscape as synthesis candidates
            # These are non-pole concepts that appear with the paradox
            all_concepts = set()

            # From paradox landscape attractors
            for p in landscape.paradoxes[:10]:
                if p.thesis not in [thesis, antithesis]:
                    all_concepts.add(p.thesis)
                if p.antithesis not in [thesis, antithesis]:
                    all_concepts.add(p.antithesis)

            # Also get from MC sampling
            mc = MonteCarloRenderer(n_samples=20)
            try:
                mc_landscape = mc.sample_landscape(question)
                for word, count in mc_landscape.core_attractors[:15]:
                    if word not in [thesis, antithesis]:
                        all_concepts.add(word)
            finally:
                mc.close()

            # Filter to concepts with valid j-vectors
            synthesis_concepts = []
            for word in all_concepts:
                j = self.get_j(word)
                if j is not None:
                    synthesis_concepts.append(word)

            synthesis_concepts = synthesis_concepts[:10]

            if len(synthesis_concepts) < 2:
                return None

            # Compute synthesis coherence
            j_vectors = [self.get_j(w) for w in synthesis_concepts]
            j_vectors = [j for j in j_vectors if j is not None]

            if len(j_vectors) < 2:
                return None

            sims = []
            for i in range(len(j_vectors)):
                for k in range(i+1, len(j_vectors)):
                    sims.append(self.cosine_sim(j_vectors[i], j_vectors[k]))

            synthesis_coherence = np.mean(sims) if sims else 0.0

            # The product (use absolute values to handle negative coherence)
            c_times_p = abs(synthesis_coherence) * pole_power

            return Measurement(
                question=question,
                thesis=thesis,
                antithesis=antithesis,
                pole_tension=pole_tension,
                pole_power=pole_power,
                synthesis_concepts=synthesis_concepts,
                synthesis_coherence=synthesis_coherence,
                c_times_p=c_times_p
            )

        finally:
            detector.close()

    def run(self) -> Dict:
        """Run the test."""
        print("\n" + "="*70)
        print("SYNTHESIS-POWER UNCERTAINTY (FIXED)")
        print("|C_synthesis| × P_poles")
        print("="*70)

        questions = [
            "What is love?",
            "What is the meaning of life?",
            "What is consciousness?",
            "What is truth?",
            "What is freedom?",
            "What is death?",
            "What is beauty?",
            "What is wisdom?",
            "What is time?",
            "What is reality?",
            "What is happiness?",
            "What is suffering?",
            "What is knowledge?",
            "What is faith?",
            "What is power?",
        ]

        print(f"\nTesting {len(questions)} questions...")
        print("-"*70)

        for q in questions:
            print(f"\n  {q}")
            m = self.measure(q)

            if m:
                self.measurements.append(m)
                print(f"    Paradox: {m.thesis} ↔ {m.antithesis}")
                print(f"    P (pole power):    {m.pole_power:.3f}")
                print(f"    C (synth coh):     {m.synthesis_coherence:.3f}")
                print(f"    |C| × P:           {m.c_times_p:.3f}")
                print(f"    Synthesis ({len(m.synthesis_concepts)}): {m.synthesis_concepts[:4]}")
            else:
                print(f"    [Failed to measure]")

        if len(self.measurements) < 5:
            print("\n[ERROR] Insufficient measurements")
            return {}

        results = self._analyze()
        self._print_results(results)
        self._save_results(results)

        return results

    def _analyze(self) -> Dict:
        """Analyze results."""
        C = np.array([abs(m.synthesis_coherence) for m in self.measurements])
        P = np.array([m.pole_power for m in self.measurements])
        CP = np.array([m.c_times_p for m in self.measurements])

        # Filter zeros
        valid = (C > 0) & (P > 0)
        C_v, P_v, CP_v = C[valid], P[valid], CP[valid]

        if len(C_v) < 3:
            return {'error': 'Too few valid measurements'}

        # Stats
        cp_mean = float(np.mean(CP_v))
        cp_std = float(np.std(CP_v))
        cp_cv = cp_std / cp_mean if cp_mean > 0 else float('inf')

        # Correlation
        corr = float(np.corrcoef(C_v, P_v)[0, 1])

        # Uncertainty check: high P → low C?
        median_p = np.median(P_v)
        c_when_high_p = float(np.mean(C_v[P_v > median_p]))
        c_when_low_p = float(np.mean(C_v[P_v <= median_p]))
        uncertainty_holds = c_when_low_p > c_when_high_p

        return {
            'n_valid': int(np.sum(valid)),
            'n_total': len(self.measurements),
            'C_mean': float(np.mean(C_v)),
            'C_std': float(np.std(C_v)),
            'P_mean': float(np.mean(P_v)),
            'P_std': float(np.std(P_v)),
            'CP_mean': cp_mean,
            'CP_std': cp_std,
            'CP_cv': cp_cv,
            'correlation': corr,
            'C_when_high_P': c_when_high_p,
            'C_when_low_P': c_when_low_p,
            'uncertainty_holds': uncertainty_holds,
            'conservation_holds': cp_cv < 0.5,
            'measurements': [
                {
                    'question': m.question,
                    'thesis': m.thesis,
                    'antithesis': m.antithesis,
                    'P': m.pole_power,
                    'C': m.synthesis_coherence,
                    'CP': m.c_times_p,
                }
                for m in self.measurements
            ]
        }

    def _print_results(self, r: Dict):
        """Print results."""
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        print(f"\nValid measurements: {r['n_valid']}/{r['n_total']}")
        print(f"\n  |C| (synthesis coherence): {r['C_mean']:.3f} ± {r['C_std']:.3f}")
        print(f"  P (pole power):           {r['P_mean']:.3f} ± {r['P_std']:.3f}")
        print(f"  |C| × P:                  {r['CP_mean']:.3f} ± {r['CP_std']:.3f}")
        print(f"  CV(|C|×P):                {r['CP_cv']:.3f}")
        print(f"  Correlation(|C|, P):      {r['correlation']:.3f}")

        print("\n" + "-"*70)
        print("UNCERTAINTY TEST")
        print("-"*70)
        print(f"  |C| when P is HIGH: {r['C_when_high_P']:.3f}")
        print(f"  |C| when P is LOW:  {r['C_when_low_P']:.3f}")

        print("\n" + "="*70)
        print("CONCLUSIONS")
        print("="*70)

        if r['uncertainty_holds']:
            print("\n  ✓ UNCERTAINTY PRINCIPLE HOLDS!")
            print("    High power paradoxes → Lower synthesis coherence")
            print("    'The harder the paradox, the messier the resolution'")
        else:
            print("\n  ✗ Uncertainty principle NOT confirmed")
            print("    No clear relationship between power and coherence")

        if r['conservation_holds']:
            print(f"\n  ✓ CONSERVATION: |C| × P ≈ {r['CP_mean']:.2f} (CV={r['CP_cv']:.2f})")
        else:
            print(f"\n  ○ Partial conservation: CV={r['CP_cv']:.2f} (>0.5)")

        if r['correlation'] < -0.2:
            print(f"\n  ✓ NEGATIVE CORRELATION: r = {r['correlation']:.3f}")
            print("    Supports trade-off between coherence and power")
        elif r['correlation'] < 0:
            print(f"\n  ○ Weak negative correlation: r = {r['correlation']:.3f}")
        else:
            print(f"\n  ✗ No negative correlation: r = {r['correlation']:.3f}")

    def _save_results(self, r: Dict):
        """Save results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"synthesis_power_fixed_{timestamp}.json"

        output = {
            'timestamp': timestamp,
            'title': 'Synthesis-Power Uncertainty (Fixed)',
            'hypothesis': '|C_synthesis| × P_poles ≈ constant',
            **r
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def close(self):
        self.graph.close()


def main():
    test = SynthesisPowerTest()
    try:
        results = test.run()

        if results.get('uncertainty_holds') and results.get('correlation', 0) < 0:
            print("\n[SUCCESS] Uncertainty principle validated!")
            return 0
        else:
            print("\n[PARTIAL] Mixed results")
            return 0  # Still informative
    finally:
        test.close()


if __name__ == "__main__":
    exit(main())
