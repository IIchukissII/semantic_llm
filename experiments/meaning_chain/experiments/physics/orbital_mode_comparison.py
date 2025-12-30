#!/usr/bin/env python3
"""
Orbital Mode Comparison Experiment
==================================

Systematically compares auto vs ground vs transcend modes
across different query types.

Hypothesis:
- Auto mode should have highest resonance (matches natural frequency)
- Ground mode should work best for practical queries
- Transcend mode should work best for philosophical queries
- Forcing wrong mode should reduce resonance quality

Metrics:
- Resonance quality (how well we hit target orbital)
- Coherence (j-vector alignment of beam)
- τ mean (actual abstraction level achieved)
- Beam concepts (qualitative difference)

Usage:
    python orbital_mode_comparison.py
"""

import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add paths
_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from chain_core.decomposer import Decomposer
from chain_core.orbital import (
    ResonantLaser, ResonantResult,
    E, VEIL_TAU, orbital_to_tau, tau_to_orbital
)


@dataclass
class QueryTest:
    """A test query with expected characteristics."""
    query: str
    category: str  # "practical", "balanced", "philosophical"
    expected_natural_orbital: int  # Expected natural orbital


# Test queries across different abstraction levels
TEST_QUERIES = [
    # Practical queries (should naturally resonate at low orbitals)
    QueryTest("How do I fix my sleep schedule?", "practical", 1),
    QueryTest("What should I eat for breakfast?", "practical", 1),
    QueryTest("How do I organize my desk?", "practical", 1),

    # Balanced queries (mid-level abstraction)
    QueryTest("What is the meaning of love?", "balanced", 2),
    QueryTest("How do I find purpose in work?", "balanced", 2),
    QueryTest("Why do relationships matter?", "balanced", 2),

    # Philosophical queries (should naturally resonate at high orbitals)
    QueryTest("What is consciousness?", "philosophical", 3),
    QueryTest("Does free will exist?", "philosophical", 4),
    QueryTest("What is the nature of reality?", "philosophical", 4),
]


@dataclass
class ModeResult:
    """Result from running a query in a specific mode."""
    mode: str
    detected_orbital: int
    target_orbital: int
    resonance_quality: float
    coherence: float
    tau_mean: float
    beam_concepts: List[str]
    lasing_achieved: bool


@dataclass
class QueryResult:
    """Complete results for a query across all modes."""
    query: str
    category: str
    expected_orbital: int
    decomposition: Dict
    auto_result: ModeResult
    ground_result: ModeResult
    transcend_result: ModeResult


class OrbitalModeComparison:
    """
    Systematically compares orbital modes.
    """

    def __init__(self):
        self.graph = MeaningGraph()
        if not self.graph.is_connected():
            raise RuntimeError("Neo4j not connected")

        self.laser = ResonantLaser(self.graph)
        self.decomposer = Decomposer()
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

    def _decompose(self, query: str) -> Dict:
        """Decompose query."""
        result = self.decomposer.decompose(query)
        return {'nouns': result.nouns, 'verbs': result.verbs}

    def _run_mode(self, nouns: List[str], verbs: List[str], mode: str) -> ModeResult:
        """Run laser in specific mode."""
        if mode == "auto":
            result = self.laser.lase_resonant(nouns, verbs, tuning="soft")
        elif mode == "ground":
            result = self.laser.lase_grounded(nouns, verbs)
        elif mode == "transcend":
            result = self.laser.lase_transcendent(nouns, verbs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return ModeResult(
            mode=mode,
            detected_orbital=result.detected_orbital,
            target_orbital=result.target_orbital,
            resonance_quality=result.resonance_quality,
            coherence=result.coherence,
            tau_mean=result.population.get('tau_mean', 0),
            beam_concepts=result.concepts[:6],
            lasing_achieved=result.metrics.get('lasing_achieved', False)
        )

    def run_query(self, test: QueryTest) -> QueryResult:
        """Run a single query through all modes."""
        decomp = self._decompose(test.query)
        nouns = decomp['nouns'] or ['meaning']
        verbs = decomp['verbs'] or []

        auto = self._run_mode(nouns, verbs, "auto")
        ground = self._run_mode(nouns, verbs, "ground")
        transcend = self._run_mode(nouns, verbs, "transcend")

        return QueryResult(
            query=test.query,
            category=test.category,
            expected_orbital=test.expected_natural_orbital,
            decomposition=decomp,
            auto_result=auto,
            ground_result=ground,
            transcend_result=transcend
        )

    def run_all(self) -> Dict:
        """Run all test queries."""
        print("\n" + "=" * 70)
        print("ORBITAL MODE COMPARISON EXPERIMENT")
        print("=" * 70)
        print(f"\nModes: auto, ground (n=2), transcend (n=6)")
        print(f"Queries: {len(TEST_QUERIES)} across 3 categories")
        print(f"Veil at τ = e ≈ {VEIL_TAU:.2f}")
        print()

        results = []
        category_stats = {
            'practical': {'auto': [], 'ground': [], 'transcend': []},
            'balanced': {'auto': [], 'ground': [], 'transcend': []},
            'philosophical': {'auto': [], 'ground': [], 'transcend': []}
        }

        for i, test in enumerate(TEST_QUERIES):
            print(f"\n[{i+1}/{len(TEST_QUERIES)}] {test.category.upper()}: {test.query[:50]}...")

            qr = self.run_query(test)
            results.append(qr)

            # Print comparison table
            print(f"\n  {'Mode':<12} {'Detected':<10} {'Target':<10} {'Resonance':<12} {'Coherence':<10} {'τ mean':<10}")
            print(f"  {'-'*64}")

            for mode_result in [qr.auto_result, qr.ground_result, qr.transcend_result]:
                print(f"  {mode_result.mode:<12} n={mode_result.detected_orbital:<8} n={mode_result.target_orbital:<8} "
                      f"{mode_result.resonance_quality:>6.0%}       {mode_result.coherence:>6.2f}      {mode_result.tau_mean:>6.2f}")

                # Collect stats
                category_stats[test.category][mode_result.mode].append({
                    'resonance': mode_result.resonance_quality,
                    'coherence': mode_result.coherence,
                    'tau_mean': mode_result.tau_mean
                })

            print(f"\n  Beam concepts:")
            print(f"    auto:      {qr.auto_result.beam_concepts}")
            print(f"    ground:    {qr.ground_result.beam_concepts}")
            print(f"    transcend: {qr.transcend_result.beam_concepts}")

        # Aggregate analysis
        self._print_aggregate_analysis(category_stats)

        # Save results
        output = self._save_results(results, category_stats)

        return output

    def _print_aggregate_analysis(self, category_stats: Dict):
        """Print aggregate analysis."""
        print("\n" + "=" * 70)
        print("AGGREGATE ANALYSIS")
        print("=" * 70)

        print("\n## Average Resonance Quality by Category × Mode")
        print(f"\n{'Category':<15} {'Auto':<12} {'Ground':<12} {'Transcend':<12} {'Best Mode':<12}")
        print("-" * 63)

        for category in ['practical', 'balanced', 'philosophical']:
            auto_res = np.mean([s['resonance'] for s in category_stats[category]['auto']])
            ground_res = np.mean([s['resonance'] for s in category_stats[category]['ground']])
            trans_res = np.mean([s['resonance'] for s in category_stats[category]['transcend']])

            best = 'auto' if auto_res >= ground_res and auto_res >= trans_res else \
                   ('ground' if ground_res >= trans_res else 'transcend')

            print(f"{category:<15} {auto_res:>6.0%}       {ground_res:>6.0%}       {trans_res:>6.0%}       {best}")

        print("\n## Average Coherence by Category × Mode")
        print(f"\n{'Category':<15} {'Auto':<12} {'Ground':<12} {'Transcend':<12}")
        print("-" * 51)

        for category in ['practical', 'balanced', 'philosophical']:
            auto_coh = np.mean([s['coherence'] for s in category_stats[category]['auto']])
            ground_coh = np.mean([s['coherence'] for s in category_stats[category]['ground']])
            trans_coh = np.mean([s['coherence'] for s in category_stats[category]['transcend']])

            print(f"{category:<15} {auto_coh:>6.2f}       {ground_coh:>6.2f}       {trans_coh:>6.2f}")

        print("\n## Average τ by Category × Mode")
        print(f"\n{'Category':<15} {'Auto':<12} {'Ground':<12} {'Transcend':<12}")
        print("-" * 51)

        for category in ['practical', 'balanced', 'philosophical']:
            auto_tau = np.mean([s['tau_mean'] for s in category_stats[category]['auto']])
            ground_tau = np.mean([s['tau_mean'] for s in category_stats[category]['ground']])
            trans_tau = np.mean([s['tau_mean'] for s in category_stats[category]['transcend']])

            print(f"{category:<15} {auto_tau:>6.2f}       {ground_tau:>6.2f}       {trans_tau:>6.2f}")

        # Key findings
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        # Compare auto resonance across categories
        auto_by_cat = {cat: np.mean([s['resonance'] for s in category_stats[cat]['auto']])
                       for cat in ['practical', 'balanced', 'philosophical']}

        print(f"\n1. AUTO MODE RESONANCE:")
        for cat, res in sorted(auto_by_cat.items(), key=lambda x: -x[1]):
            print(f"   {cat}: {res:.0%}")

        # Compare forced modes
        print(f"\n2. FORCING EFFECT (resonance penalty for wrong mode):")

        # Practical queries: ground should be best
        practical_auto = np.mean([s['resonance'] for s in category_stats['practical']['auto']])
        practical_ground = np.mean([s['resonance'] for s in category_stats['practical']['ground']])
        practical_trans = np.mean([s['resonance'] for s in category_stats['practical']['transcend']])
        print(f"   Practical queries:")
        print(f"     auto={practical_auto:.0%}, ground={practical_ground:.0%}, transcend={practical_trans:.0%}")
        print(f"     Penalty for transcend: {practical_auto - practical_trans:+.0%}")

        # Philosophical queries: transcend should be best
        phil_auto = np.mean([s['resonance'] for s in category_stats['philosophical']['auto']])
        phil_ground = np.mean([s['resonance'] for s in category_stats['philosophical']['ground']])
        phil_trans = np.mean([s['resonance'] for s in category_stats['philosophical']['transcend']])
        print(f"   Philosophical queries:")
        print(f"     auto={phil_auto:.0%}, ground={phil_ground:.0%}, transcend={phil_trans:.0%}")
        print(f"     Penalty for ground: {phil_auto - phil_ground:+.0%}")

    def _save_results(self, results: List[QueryResult], category_stats: Dict) -> Dict:
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"mode_comparison_{timestamp}.json"

        def result_to_dict(qr: QueryResult) -> Dict:
            return {
                'query': qr.query,
                'category': qr.category,
                'expected_orbital': qr.expected_orbital,
                'decomposition': qr.decomposition,
                'modes': {
                    'auto': {
                        'detected_orbital': qr.auto_result.detected_orbital,
                        'target_orbital': qr.auto_result.target_orbital,
                        'resonance_quality': qr.auto_result.resonance_quality,
                        'coherence': qr.auto_result.coherence,
                        'tau_mean': qr.auto_result.tau_mean,
                        'beam_concepts': qr.auto_result.beam_concepts
                    },
                    'ground': {
                        'detected_orbital': qr.ground_result.detected_orbital,
                        'target_orbital': qr.ground_result.target_orbital,
                        'resonance_quality': qr.ground_result.resonance_quality,
                        'coherence': qr.ground_result.coherence,
                        'tau_mean': qr.ground_result.tau_mean,
                        'beam_concepts': qr.ground_result.beam_concepts
                    },
                    'transcend': {
                        'detected_orbital': qr.transcend_result.detected_orbital,
                        'target_orbital': qr.transcend_result.target_orbital,
                        'resonance_quality': qr.transcend_result.resonance_quality,
                        'coherence': qr.transcend_result.coherence,
                        'tau_mean': qr.transcend_result.tau_mean,
                        'beam_concepts': qr.transcend_result.beam_concepts
                    }
                }
            }

        # Compute aggregate stats
        agg_stats = {}
        for category in ['practical', 'balanced', 'philosophical']:
            agg_stats[category] = {}
            for mode in ['auto', 'ground', 'transcend']:
                agg_stats[category][mode] = {
                    'avg_resonance': np.mean([s['resonance'] for s in category_stats[category][mode]]),
                    'avg_coherence': np.mean([s['coherence'] for s in category_stats[category][mode]]),
                    'avg_tau': np.mean([s['tau_mean'] for s in category_stats[category][mode]])
                }

        output = {
            'timestamp': timestamp,
            'experiment': 'Orbital Mode Comparison',
            'modes_tested': ['auto', 'ground', 'transcend'],
            'categories': ['practical', 'balanced', 'philosophical'],
            'n_queries': len(results),
            'results': [result_to_dict(r) for r in results],
            'aggregate_stats': agg_stats
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

        return output

    def close(self):
        self.laser.close()
        self.graph.close()


def main():
    experiment = OrbitalModeComparison()
    try:
        experiment.run_all()
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
