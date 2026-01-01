#!/usr/bin/env python3
"""
Synthesis-Power Uncertainty Principle
======================================

Tests the REFINED C×P hypothesis:

    C_synthesis × P_poles ≈ constant

Where:
    C_synthesis = coherence among SYNTHESIS concepts (bridge concepts)
    P_poles = power from POLE opposition (thesis ↔ antithesis)

The insight:
    Previous test measured coherence across ALL beam concepts,
    which included both poles — of course it was negative!

    The TRUE uncertainty is:
        Strong paradox (high pole tension) → Hard to synthesize (low synthesis coherence)
        Weak paradox (low pole tension) → Easy to synthesize (high synthesis coherence)

    "The harder the paradox, the messier the resolution"

Usage:
    python synthesis_power_uncertainty.py

Results saved to: results/synthesis_power_YYYYMMDD_HHMMSS.json
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import sys

# Setup paths
_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from graph.meaning_graph import MeaningGraph

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


@dataclass
class UncertaintyMeasurement:
    """Measurement of Synthesis Coherence and Pole Power."""
    question: str

    # Poles (the paradox)
    thesis: str
    antithesis: str
    pole_power: float          # Opposition strength
    pole_tension: float        # Raw tension (-cos)
    pole_stability: float      # How often both appear

    # Synthesis (the bridge)
    synthesis_concepts: List[str]
    synthesis_coherence: float  # Alignment among bridge concepts
    synthesis_quality: float    # How well synthesis bridges poles

    # The product (should be approximately constant)
    cs_times_pp: float

    # For analysis
    n_synthesis: int
    paradox_type: str  # archetypal or conceptual


class SynthesisPowerTester:
    """
    Test the refined uncertainty principle:

        Synthesis_Coherence × Pole_Power ≈ constant
    """

    def __init__(self):
        self.graph = MeaningGraph()
        self.measurements: List[UncertaintyMeasurement] = []
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)
        self._j_cache: Dict[str, np.ndarray] = {}

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
        """Compute cosine similarity."""
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def compute_pole_power(self, thesis: str, antithesis: str,
                           stability: float) -> tuple:
        """
        Compute pole power: opposition strength between thesis and antithesis.

        Returns (power, tension, stability)
        """
        j_thesis = self.get_j(thesis)
        j_anti = self.get_j(antithesis)

        if j_thesis is None or j_anti is None:
            return 0.0, 0.0, stability

        # Tension = negative cosine (opposition)
        tension = -self.cosine_sim(j_thesis, j_anti)

        # Power = tension × stability × 10
        power = max(0, tension * stability * 10)

        return power, tension, stability

    def compute_synthesis_coherence(self, synthesis_concepts: List[str],
                                     thesis: str, antithesis: str) -> tuple:
        """
        Compute synthesis coherence: how well bridge concepts align.

        Also computes synthesis quality: how well synthesis bridges the poles.

        Returns (coherence, quality)
        """
        if not synthesis_concepts:
            return 0.0, 0.0

        # Get j-vectors for synthesis concepts
        synth_js = []
        for word in synthesis_concepts:
            j = self.get_j(word)
            if j is not None:
                synth_js.append((word, j))

        if len(synth_js) < 2:
            return 0.0, 0.0

        # Coherence = average pairwise similarity among synthesis concepts
        sims = []
        for i in range(len(synth_js)):
            for k in range(i+1, len(synth_js)):
                sims.append(self.cosine_sim(synth_js[i][1], synth_js[k][1]))

        coherence = np.mean(sims) if sims else 0.0

        # Quality = how well synthesis is positioned between poles
        j_thesis = self.get_j(thesis)
        j_anti = self.get_j(antithesis)

        if j_thesis is None or j_anti is None:
            return coherence, 0.0

        # Midpoint between poles
        midpoint = (j_thesis + j_anti) / 2
        mid_norm = np.linalg.norm(midpoint)

        if mid_norm < 1e-8:
            # Poles are perfectly opposite - synthesis should be orthogonal
            quality_scores = []
            for _, j in synth_js:
                # How orthogonal is synthesis to both poles?
                orth_thesis = 1 - abs(self.cosine_sim(j, j_thesis))
                orth_anti = 1 - abs(self.cosine_sim(j, j_anti))
                quality_scores.append((orth_thesis + orth_anti) / 2)
            quality = np.mean(quality_scores) if quality_scores else 0.0
        else:
            # Synthesis should be close to midpoint
            midpoint = midpoint / mid_norm
            quality_scores = []
            for _, j in synth_js:
                quality_scores.append(self.cosine_sim(j, midpoint))
            quality = np.mean(quality_scores) if quality_scores else 0.0

        return coherence, quality

    def measure_question(self, question: str) -> Optional[UncertaintyMeasurement]:
        """Measure synthesis coherence and pole power for a question."""
        try:
            from chain_core.paradox_detector import ParadoxDetector
        except ImportError:
            print("[ERROR] ParadoxDetector not available")
            return None

        detector = ParadoxDetector(n_samples=25)

        try:
            landscape = detector.detect(question)

            if not landscape.strongest:
                return None

            paradox = landscape.strongest

            # Compute pole power
            pole_power, tension, stability = self.compute_pole_power(
                paradox.thesis, paradox.antithesis, paradox.stability
            )

            # Get synthesis concepts
            synthesis = paradox.synthesis_concepts[:5]

            # Compute synthesis coherence
            synth_coherence, synth_quality = self.compute_synthesis_coherence(
                synthesis, paradox.thesis, paradox.antithesis
            )

            # The product
            cs_times_pp = synth_coherence * pole_power

            return UncertaintyMeasurement(
                question=question,
                thesis=paradox.thesis,
                antithesis=paradox.antithesis,
                pole_power=pole_power,
                pole_tension=tension,
                pole_stability=stability,
                synthesis_concepts=synthesis,
                synthesis_coherence=synth_coherence,
                synthesis_quality=synth_quality,
                cs_times_pp=cs_times_pp,
                n_synthesis=len(synthesis),
                paradox_type=paradox.paradox_type
            )

        finally:
            detector.close()

    def run_test(self) -> Dict:
        """Run the synthesis-power uncertainty test."""
        print("\n" + "="*70)
        print("SYNTHESIS-POWER UNCERTAINTY PRINCIPLE")
        print("C_synthesis × P_poles ≈ constant")
        print("="*70)

        test_questions = [
            "What is love?",
            "What is the meaning of life?",
            "What is consciousness?",
            "What is truth?",
            "What is freedom?",
            "What is death?",
            "What is beauty?",
            "What is wisdom?",
            "What is time?",
            "What is the nature of reality?",
            "What is happiness?",
            "What is suffering?",
            "What is knowledge?",
            "What is faith?",
            "What is power?",
            "What is justice?",
            "What is art?",
            "What is meaning?",
        ]

        print(f"\nTesting {len(test_questions)} questions...")
        print("-"*70)

        for q in test_questions:
            print(f"\n  {q}")
            measurement = self.measure_question(q)

            if measurement and measurement.n_synthesis >= 2:
                self.measurements.append(measurement)
                print(f"    Paradox: {measurement.thesis} ↔ {measurement.antithesis}")
                print(f"    Pole Power (P):       {measurement.pole_power:.3f}")
                print(f"    Synth Coherence (C):  {measurement.synthesis_coherence:.3f}")
                print(f"    C × P:                {measurement.cs_times_pp:.3f}")
                print(f"    Synthesis: {measurement.synthesis_concepts[:3]}")
            else:
                print(f"    [Insufficient synthesis concepts]")

        if len(self.measurements) < 5:
            print("\n[ERROR] Insufficient measurements")
            return {}

        # Analyze
        results = self._analyze()
        self._print_results(results)
        self._save_results(results)

        return results

    def _analyze(self) -> Dict:
        """Analyze the uncertainty relationship."""
        # Extract arrays
        C = np.array([m.synthesis_coherence for m in self.measurements])
        P = np.array([m.pole_power for m in self.measurements])
        CP = np.array([m.cs_times_pp for m in self.measurements])

        # Filter out zero values for meaningful analysis
        valid_mask = (C != 0) & (P != 0)
        C_valid = C[valid_mask]
        P_valid = P[valid_mask]
        CP_valid = CP[valid_mask]

        # Statistics
        cp_mean = np.mean(CP_valid) if len(CP_valid) > 0 else 0
        cp_std = np.std(CP_valid) if len(CP_valid) > 0 else 0
        cp_cv = cp_std / abs(cp_mean) if abs(cp_mean) > 0.01 else float('inf')

        # Correlation
        if len(C_valid) > 2:
            correlation = float(np.corrcoef(C_valid, P_valid)[0, 1])
        else:
            correlation = 0.0

        # Conservation score (lower CV + negative correlation = better)
        if cp_cv < float('inf'):
            conservation_score = (1 / (1 + cp_cv)) * (1 - correlation) / 2
        else:
            conservation_score = 0.0

        # Check for true uncertainty relationship
        # In true uncertainty: high P → low C, high C → low P
        high_p_mask = P_valid > np.median(P_valid)
        low_p_mask = ~high_p_mask

        c_when_high_p = np.mean(C_valid[high_p_mask]) if np.any(high_p_mask) else 0
        c_when_low_p = np.mean(C_valid[low_p_mask]) if np.any(low_p_mask) else 0

        uncertainty_evidence = c_when_low_p > c_when_high_p

        return {
            'n_measurements': len(self.measurements),
            'n_valid': int(np.sum(valid_mask)),
            'statistics': {
                'C_mean': float(np.mean(C_valid)) if len(C_valid) > 0 else 0,
                'C_std': float(np.std(C_valid)) if len(C_valid) > 0 else 0,
                'P_mean': float(np.mean(P_valid)) if len(P_valid) > 0 else 0,
                'P_std': float(np.std(P_valid)) if len(P_valid) > 0 else 0,
                'CP_mean': float(cp_mean),
                'CP_std': float(cp_std),
                'CP_cv': float(cp_cv) if cp_cv < float('inf') else None,
            },
            'correlation_CP': correlation,
            'conservation_score': float(conservation_score),
            'uncertainty_evidence': {
                'C_when_high_P': float(c_when_high_p),
                'C_when_low_P': float(c_when_low_p),
                'supports_uncertainty': uncertainty_evidence,
            },
            'conclusion': {
                'conservation_holds': cp_cv < 0.5 if cp_cv < float('inf') else False,
                'negative_correlation': correlation < -0.2,
                'uncertainty_validated': uncertainty_evidence and correlation < 0,
            },
            'measurements': [
                {
                    'question': m.question,
                    'thesis': m.thesis,
                    'antithesis': m.antithesis,
                    'pole_power': m.pole_power,
                    'synthesis_coherence': m.synthesis_coherence,
                    'cs_times_pp': m.cs_times_pp,
                    'synthesis': m.synthesis_concepts,
                }
                for m in self.measurements
            ]
        }

    def _print_results(self, results: Dict):
        """Print analysis results."""
        print("\n" + "="*70)
        print("RESULTS: SYNTHESIS-POWER UNCERTAINTY")
        print("="*70)

        stats = results['statistics']
        print(f"\nMeasurements: {results['n_valid']} valid / {results['n_measurements']} total")

        print("\n" + "-"*70)
        print("STATISTICS")
        print("-"*70)
        print(f"  Synthesis Coherence (C): {stats['C_mean']:.3f} ± {stats['C_std']:.3f}")
        print(f"  Pole Power (P):          {stats['P_mean']:.3f} ± {stats['P_std']:.3f}")
        print(f"  C × P:                   {stats['CP_mean']:.3f} ± {stats['CP_std']:.3f}")
        if stats['CP_cv']:
            print(f"  CV(C×P):                 {stats['CP_cv']:.3f}")
        print(f"  Correlation(C, P):       {results['correlation_CP']:.3f}")

        print("\n" + "-"*70)
        print("UNCERTAINTY EVIDENCE")
        print("-"*70)
        ue = results['uncertainty_evidence']
        print(f"  C when P is HIGH: {ue['C_when_high_P']:.3f}")
        print(f"  C when P is LOW:  {ue['C_when_low_P']:.3f}")

        if ue['supports_uncertainty']:
            print(f"  → High power paradoxes have LOWER synthesis coherence ✓")
        else:
            print(f"  → No clear relationship between power and coherence")

        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        conc = results['conclusion']

        if conc['uncertainty_validated']:
            print("\n  ✓ UNCERTAINTY PRINCIPLE VALIDATED!")
            print("    Strong paradoxes (high P) → Weak synthesis (low C)")
            print("    Weak paradoxes (low P) → Strong synthesis (high C)")
            print("\n    'The harder the paradox, the messier the resolution'")
        elif conc['negative_correlation']:
            print("\n  ○ PARTIAL SUPPORT")
            print("    Negative correlation exists but not fully validated")
        else:
            print("\n  ✗ NO CLEAR UNCERTAINTY RELATIONSHIP")
            print("    C and P appear to be independent")

        if conc['conservation_holds']:
            print(f"\n  ✓ C × P ≈ {stats['CP_mean']:.2f} (conserved, CV={stats['CP_cv']:.2f})")

    def _save_results(self, results: Dict):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"synthesis_power_{timestamp}.json"

        output = {
            'timestamp': timestamp,
            'title': 'Synthesis-Power Uncertainty Principle',
            'hypothesis': 'C_synthesis × P_poles ≈ constant',
            'interpretation': 'Strong paradoxes are harder to synthesize cleanly',
            **results
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def close(self):
        self.graph.close()


def main():
    tester = SynthesisPowerTester()

    try:
        results = tester.run_test()

        if results.get('conclusion', {}).get('uncertainty_validated'):
            print("\n[SUCCESS] Synthesis-Power uncertainty validated!")
            return 0
        else:
            print("\n[PARTIAL] Mixed results")
            return 1

    finally:
        tester.close()


if __name__ == "__main__":
    exit(main())
