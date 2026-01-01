#!/usr/bin/env python3
"""
Coherence × Power Conservation: Phase-Shifted Test
===================================================

Tests whether C×P ≈ constant is:
  (a) An artifact of biased j-vectors, OR
  (b) A real semantic uncertainty principle

The Pirate Insight:
    All j-vectors are biased toward global mean ≈ [-0.82, -0.97, ...]
    This makes opposites look similar (cos ≈ 0.99)!

    Solution: Phase shift by centering: i = j - j_mean
    Result: love/hate → -0.22 (now properly opposite)

Hypothesis:
    If C×P conservation is REAL, it should be STRONGER with phase-shifted metrics.
    If it's an ARTIFACT, phase shift will destroy the correlation.

Metrics:
    RAW:
        Coherence_raw = mean(cos(j_i, j_j)) for concepts in beam
        Power_raw = -cos(thesis.j, antithesis.j) × stability

    PHASE-SHIFTED:
        Coherence_shifted = mean(cos(i_i, i_j)) where i = j - j_mean
        Power_shifted = -cos(thesis.i, antithesis.i) × stability

Usage:
    python coherence_power_conservation.py

Results saved to: results/cp_conservation_YYYYMMDD_HHMMSS.json
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import sys

# Setup paths
_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from graph.meaning_graph import MeaningGraph
from core.data_loader import DataLoader

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


@dataclass
class CPMeasurement:
    """Single Coherence × Power measurement."""
    question: str

    # Raw (biased) metrics
    coherence_raw: float
    power_raw: float
    cp_raw: float

    # Phase-shifted (centered) metrics
    coherence_shifted: float
    power_shifted: float
    cp_shifted: float

    # Paradox info
    thesis: str
    antithesis: str
    n_concepts: int


class PhaseShiftedMetrics:
    """
    Compute Coherence and Power with phase-shifted j-vectors.

    The Pirate Insight: subtract global mean to reveal true opposition.
    """

    def __init__(self):
        self.graph = MeaningGraph()
        self.loader = DataLoader()
        self.global_mean: np.ndarray = None
        self._concept_cache: Dict[str, np.ndarray] = {}

    def compute_global_mean(self) -> np.ndarray:
        """Compute mean j-vector across all concepts."""
        if self.global_mean is not None:
            return self.global_mean

        print("Computing global j-vector mean (for phase shift)...")

        word_vectors = self.loader.load_word_vectors()

        all_j = []
        for word, wv in word_vectors.items():
            j_dict = wv.get('j', None)
            if j_dict and isinstance(j_dict, dict):
                j_vec = np.array([j_dict.get(dim, 0) for dim in J_DIMS])
                if np.linalg.norm(j_vec) > 0.1:
                    all_j.append(j_vec)

        if all_j:
            self.global_mean = np.mean(all_j, axis=0)
        else:
            self.global_mean = np.zeros(5)

        print(f"  Global mean: [{', '.join([f'{v:.3f}' for v in self.global_mean])}]")
        return self.global_mean

    def get_concept_j(self, word: str) -> Optional[np.ndarray]:
        """Get raw j-vector for a concept."""
        if word in self._concept_cache:
            return self._concept_cache[word]

        concept = self.graph.get_concept(word)
        if not concept or not concept.get('j'):
            return None

        j = np.array(concept['j'])
        if len(j) != 5:
            return None

        self._concept_cache[word] = j
        return j

    def center(self, j: np.ndarray) -> np.ndarray:
        """Apply phase shift: i = j - j_mean."""
        if self.global_mean is None:
            self.compute_global_mean()
        return j - self.global_mean

    def cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def compute_coherence_raw(self, concepts: List[str]) -> float:
        """Compute coherence using RAW j-vectors."""
        j_vectors = []
        for word in concepts:
            j = self.get_concept_j(word)
            if j is not None:
                j_vectors.append(j)

        if len(j_vectors) < 2:
            return 0.0

        # Average pairwise cosine similarity
        sims = []
        for i in range(len(j_vectors)):
            for k in range(i+1, len(j_vectors)):
                sims.append(self.cosine_sim(j_vectors[i], j_vectors[k]))

        return np.mean(sims) if sims else 0.0

    def compute_coherence_shifted(self, concepts: List[str]) -> float:
        """Compute coherence using PHASE-SHIFTED j-vectors."""
        i_vectors = []
        for word in concepts:
            j = self.get_concept_j(word)
            if j is not None:
                i_vectors.append(self.center(j))

        if len(i_vectors) < 2:
            return 0.0

        # Average pairwise cosine similarity on centered vectors
        sims = []
        for i in range(len(i_vectors)):
            for k in range(i+1, len(i_vectors)):
                sims.append(self.cosine_sim(i_vectors[i], i_vectors[k]))

        return np.mean(sims) if sims else 0.0

    def compute_power_raw(self, thesis: str, antithesis: str,
                          stability: float = 1.0) -> float:
        """Compute power using RAW j-vectors."""
        j1 = self.get_concept_j(thesis)
        j2 = self.get_concept_j(antithesis)

        if j1 is None or j2 is None:
            return 0.0

        # Tension = negative dot product (opposition)
        tension = -self.cosine_sim(j1, j2)

        # Power = tension × stability
        return max(0, tension * stability * 10)

    def compute_power_shifted(self, thesis: str, antithesis: str,
                               stability: float = 1.0) -> float:
        """Compute power using PHASE-SHIFTED j-vectors."""
        j1 = self.get_concept_j(thesis)
        j2 = self.get_concept_j(antithesis)

        if j1 is None or j2 is None:
            return 0.0

        # Center both vectors
        i1 = self.center(j1)
        i2 = self.center(j2)

        # Tension = negative dot product on centered vectors
        tension = -self.cosine_sim(i1, i2)

        # Power = tension × stability
        return max(0, tension * stability * 10)

    def close(self):
        self.graph.close()


class CPConservationTester:
    """
    Test whether Coherence × Power is conserved (uncertainty principle).

    Compares:
        1. Raw metrics (potentially biased)
        2. Phase-shifted metrics (properly centered)
    """

    def __init__(self):
        self.metrics = PhaseShiftedMetrics()
        self.measurements: List[CPMeasurement] = []
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

    def measure_question(self, question: str) -> Optional[CPMeasurement]:
        """Measure C and P for a question using both raw and shifted metrics."""
        try:
            from chain_core.paradox_detector import ParadoxDetector
        except ImportError:
            print("[ERROR] ParadoxDetector not available")
            return None

        detector = ParadoxDetector(n_samples=20)

        try:
            landscape = detector.detect(question)

            if not landscape.strongest:
                return None

            paradox = landscape.strongest

            # Get all concepts from attractors
            concepts = [word for word, _ in landscape.paradoxes[0].synthesis_concepts[:10]] if landscape.paradoxes else []
            concepts.extend([paradox.thesis, paradox.antithesis])

            # Get concepts from MC core attractors
            if hasattr(landscape, 'core_attractors'):
                concepts.extend([w for w, _ in getattr(landscape, 'core_attractors', [])[:10]])

            # Get more concepts from paradoxes
            for p in landscape.paradoxes[:5]:
                if p.thesis not in concepts:
                    concepts.append(p.thesis)
                if p.antithesis not in concepts:
                    concepts.append(p.antithesis)

            concepts = list(set(concepts))[:15]

            # Compute RAW metrics
            coherence_raw = self.metrics.compute_coherence_raw(concepts)
            power_raw = self.metrics.compute_power_raw(
                paradox.thesis, paradox.antithesis, paradox.stability
            )

            # Compute PHASE-SHIFTED metrics
            coherence_shifted = self.metrics.compute_coherence_shifted(concepts)
            power_shifted = self.metrics.compute_power_shifted(
                paradox.thesis, paradox.antithesis, paradox.stability
            )

            return CPMeasurement(
                question=question,
                coherence_raw=coherence_raw,
                power_raw=power_raw,
                cp_raw=coherence_raw * power_raw,
                coherence_shifted=coherence_shifted,
                power_shifted=power_shifted,
                cp_shifted=coherence_shifted * power_shifted,
                thesis=paradox.thesis,
                antithesis=paradox.antithesis,
                n_concepts=len(concepts)
            )

        finally:
            detector.close()

    def run_test(self) -> Dict:
        """Run the C×P conservation test."""
        print("\n" + "="*70)
        print("COHERENCE × POWER CONSERVATION TEST")
        print("Phase-Shifted vs Raw Metrics")
        print("="*70)

        # Compute global mean first
        self.metrics.compute_global_mean()

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
        ]

        print(f"\nTesting {len(test_questions)} questions...")
        print("-"*70)

        for q in test_questions:
            print(f"\n  {q}")
            measurement = self.measure_question(q)

            if measurement:
                self.measurements.append(measurement)
                print(f"    Paradox: {measurement.thesis} ↔ {measurement.antithesis}")
                print(f"    RAW:     C={measurement.coherence_raw:.3f}, "
                      f"P={measurement.power_raw:.3f}, C×P={measurement.cp_raw:.3f}")
                print(f"    SHIFTED: C={measurement.coherence_shifted:.3f}, "
                      f"P={measurement.power_shifted:.3f}, C×P={measurement.cp_shifted:.3f}")
            else:
                print(f"    [No paradox found]")

        if len(self.measurements) < 3:
            print("\n[ERROR] Insufficient measurements")
            return {}

        # Analyze results
        results = self._analyze()
        self._print_results(results)
        self._save_results(results)

        return results

    def _analyze(self) -> Dict:
        """Analyze C×P conservation for both raw and shifted metrics."""
        # Extract values
        c_raw = np.array([m.coherence_raw for m in self.measurements])
        p_raw = np.array([m.power_raw for m in self.measurements])
        cp_raw = np.array([m.cp_raw for m in self.measurements])

        c_shifted = np.array([m.coherence_shifted for m in self.measurements])
        p_shifted = np.array([m.power_shifted for m in self.measurements])
        cp_shifted = np.array([m.cp_shifted for m in self.measurements])

        # Compute statistics
        def compute_stats(c, p, cp):
            return {
                'c_mean': float(np.mean(c)),
                'c_std': float(np.std(c)),
                'p_mean': float(np.mean(p)),
                'p_std': float(np.std(p)),
                'cp_mean': float(np.mean(cp)),
                'cp_std': float(np.std(cp)),
                'cp_cv': float(np.std(cp) / np.mean(cp)) if np.mean(cp) > 0 else float('inf'),
                'correlation_cp': float(np.corrcoef(c, p)[0, 1]) if len(c) > 2 else 0,
            }

        raw_stats = compute_stats(c_raw, p_raw, cp_raw)
        shifted_stats = compute_stats(c_shifted, p_shifted, cp_shifted)

        # Determine which is better
        # Lower CV = more conserved
        # Stronger negative correlation = more uncertainty-like

        raw_conservation_score = (1 / (1 + raw_stats['cp_cv'])) * (1 - raw_stats['correlation_cp']) / 2
        shifted_conservation_score = (1 / (1 + shifted_stats['cp_cv'])) * (1 - shifted_stats['correlation_cp']) / 2

        return {
            'n_measurements': len(self.measurements),
            'raw': raw_stats,
            'shifted': shifted_stats,
            'raw_conservation_score': float(raw_conservation_score),
            'shifted_conservation_score': float(shifted_conservation_score),
            'phase_shift_improves': shifted_conservation_score > raw_conservation_score,
            'measurements': [
                {
                    'question': m.question,
                    'thesis': m.thesis,
                    'antithesis': m.antithesis,
                    'coherence_raw': m.coherence_raw,
                    'power_raw': m.power_raw,
                    'cp_raw': m.cp_raw,
                    'coherence_shifted': m.coherence_shifted,
                    'power_shifted': m.power_shifted,
                    'cp_shifted': m.cp_shifted,
                }
                for m in self.measurements
            ]
        }

    def _print_results(self, results: Dict):
        """Print analysis results."""
        print("\n" + "="*70)
        print("RESULTS: C×P CONSERVATION ANALYSIS")
        print("="*70)

        print(f"\nMeasurements: {results['n_measurements']}")

        print("\n" + "-"*70)
        print("RAW METRICS (potentially biased)")
        print("-"*70)
        raw = results['raw']
        print(f"  Coherence:  {raw['c_mean']:.3f} ± {raw['c_std']:.3f}")
        print(f"  Power:      {raw['p_mean']:.3f} ± {raw['p_std']:.3f}")
        print(f"  C × P:      {raw['cp_mean']:.3f} ± {raw['cp_std']:.3f}")
        print(f"  CV(C×P):    {raw['cp_cv']:.3f}")
        print(f"  Corr(C,P):  {raw['correlation_cp']:.3f}")

        print("\n" + "-"*70)
        print("PHASE-SHIFTED METRICS (i = j - j_mean)")
        print("-"*70)
        shifted = results['shifted']
        print(f"  Coherence:  {shifted['c_mean']:.3f} ± {shifted['c_std']:.3f}")
        print(f"  Power:      {shifted['p_mean']:.3f} ± {shifted['p_std']:.3f}")
        print(f"  C × P:      {shifted['cp_mean']:.3f} ± {shifted['cp_std']:.3f}")
        print(f"  CV(C×P):    {shifted['cp_cv']:.3f}")
        print(f"  Corr(C,P):  {shifted['correlation_cp']:.3f}")

        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)

        if results['phase_shift_improves']:
            print("\n  ✓ PHASE SHIFT IMPROVES C×P CONSERVATION!")
            print(f"    Raw score:     {results['raw_conservation_score']:.3f}")
            print(f"    Shifted score: {results['shifted_conservation_score']:.3f}")
            print("\n  This suggests C×P conservation is a REAL semantic property,")
            print("  not an artifact of biased j-vectors.")
        else:
            print("\n  ○ Phase shift does not improve conservation")
            print(f"    Raw score:     {results['raw_conservation_score']:.3f}")
            print(f"    Shifted score: {results['shifted_conservation_score']:.3f}")

        # Interpret correlation
        if shifted['correlation_cp'] < -0.3:
            print("\n  ✓ NEGATIVE CORRELATION in shifted metrics!")
            print("  → Supports Heisenberg-like uncertainty principle")
            print("  → High Coherence ⟺ Low Power (trade-off exists)")
        elif shifted['correlation_cp'] < 0:
            print("\n  ○ Weak negative correlation in shifted metrics")
            print("  → Partial support for uncertainty principle")
        else:
            print("\n  ✗ No negative correlation in shifted metrics")
            print("  → C and P may be independent, not complementary")

    def _save_results(self, results: Dict):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"cp_conservation_{timestamp}.json"

        output = {
            'timestamp': timestamp,
            'title': 'Coherence × Power Conservation Test',
            'hypothesis': 'C × P ≈ constant (semantic uncertainty principle)',
            'method': 'Compare raw vs phase-shifted (centered) j-vector metrics',
            'global_mean': self.metrics.global_mean.tolist() if self.metrics.global_mean is not None else None,
            **results
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def close(self):
        self.metrics.close()


def main():
    tester = CPConservationTester()

    try:
        results = tester.run_test()

        if results.get('phase_shift_improves'):
            print("\n[SUCCESS] C×P conservation validated with phase shift!")
            return 0
        else:
            print("\n[PARTIAL] Mixed results")
            return 1

    finally:
        tester.close()


if __name__ == "__main__":
    exit(main())
