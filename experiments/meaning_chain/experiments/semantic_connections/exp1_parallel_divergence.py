#!/usr/bin/env python3
"""
Experiment 1: Parallel Path Divergence
=======================================

HYPOTHESIS:
    Parallel paths capture multiple aspects of a concept.
    Coverage(parallel) > Coverage(series)

EXPERIMENT:
    1. From seed concept, navigate in parallel (3 paths)
    2. From same seed, navigate in series (single chain)
    3. Compare coverage (unique concepts / total)

PREDICTION:
    Parallel paths will have higher coverage (>1.5x) than series.

Usage:
    python exp1_parallel_divergence.py

Results saved to: results/parallel_divergence_YYYYMMDD_HHMMSS.json
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from connection_types import SeriesConnection, ParallelConnection, ConnectionResult


@dataclass
class DivergenceResult:
    """Result of divergence comparison"""
    seed: str
    series_coverage: float
    parallel_coverage: float
    coverage_ratio: float  # parallel / series
    series_concepts: int
    parallel_concepts: int
    series_unique: int
    parallel_unique: int
    hypothesis_confirmed: bool  # ratio > 1.5


@dataclass
class ExperimentResult:
    """Complete experiment result"""
    timestamp: str
    hypothesis: str
    seeds: List[str]
    results: List[Dict]
    summary: Dict


def run_divergence_test(
    graph,
    seed: str,
    series_depth: int = 5,
    parallel_paths: int = 3,
    parallel_depth: int = 3
) -> DivergenceResult:
    """
    Compare parallel vs series divergence for a seed concept

    Args:
        graph: MeaningGraph instance
        seed: Starting concept
        series_depth: Steps in series chain
        parallel_paths: Number of parallel paths
        parallel_depth: Steps per parallel path

    Returns:
        DivergenceResult with comparison metrics
    """
    # Series navigation
    series = SeriesConnection(graph)
    series_result = series.connect(seed, depth=series_depth)

    # Parallel navigation
    parallel = ParallelConnection(graph, n_paths=parallel_paths)
    parallel_result = parallel.connect(seed, depth=parallel_depth)

    # Compute coverage ratio
    if series_result.coverage > 0:
        coverage_ratio = parallel_result.coverage / series_result.coverage
    else:
        coverage_ratio = float('inf') if parallel_result.coverage > 0 else 1.0

    return DivergenceResult(
        seed=seed,
        series_coverage=series_result.coverage,
        parallel_coverage=parallel_result.coverage,
        coverage_ratio=coverage_ratio,
        series_concepts=len(series_result.concepts),
        parallel_concepts=len(parallel_result.concepts),
        series_unique=len(set(series_result.concepts)),
        parallel_unique=len(set(parallel_result.concepts)),
        hypothesis_confirmed=coverage_ratio > 1.5,
    )


def run_experiment(graph, seeds: List[str] = None) -> ExperimentResult:
    """
    Run the full divergence experiment

    Args:
        graph: MeaningGraph instance
        seeds: List of seed concepts to test

    Returns:
        ExperimentResult with all findings
    """
    if seeds is None:
        # Default test seeds covering different τ levels
        seeds = [
            "wisdom",      # Abstract
            "love",        # Abstract
            "knowledge",   # Medium
            "feeling",     # Medium
            "word",        # Concrete
            "hand",        # Concrete
        ]

    results = []
    for seed in seeds:
        try:
            result = run_divergence_test(graph, seed)
            results.append(asdict(result))
            print(f"  {seed}: coverage ratio = {result.coverage_ratio:.2f} "
                  f"({'CONFIRMED' if result.hypothesis_confirmed else 'not confirmed'})")
        except Exception as e:
            print(f"  {seed}: ERROR - {e}")
            results.append({
                "seed": seed,
                "error": str(e),
            })

    # Summary statistics
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_ratio = sum(r["coverage_ratio"] for r in valid_results) / len(valid_results)
        confirmed_count = sum(1 for r in valid_results if r["hypothesis_confirmed"])
        confirmation_rate = confirmed_count / len(valid_results)
    else:
        avg_ratio = 0.0
        confirmed_count = 0
        confirmation_rate = 0.0

    summary = {
        "total_seeds": len(seeds),
        "successful_tests": len(valid_results),
        "avg_coverage_ratio": avg_ratio,
        "confirmed_count": confirmed_count,
        "confirmation_rate": confirmation_rate,
        "hypothesis_status": "CONFIRMED" if confirmation_rate > 0.7 else "NOT CONFIRMED",
    }

    return ExperimentResult(
        timestamp=datetime.now().isoformat(),
        hypothesis="Parallel paths have >1.5x coverage vs series",
        seeds=seeds,
        results=results,
        summary=summary,
    )


def save_results(result: ExperimentResult, output_dir: Path):
    """Save experiment results to JSON"""
    output_dir.mkdir(exist_ok=True)
    filename = f"parallel_divergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    """Run experiment with mock or real graph"""
    print("=" * 60)
    print("EXPERIMENT 1: Parallel Path Divergence")
    print("=" * 60)
    print("\nHypothesis: Parallel paths capture multiple aspects of a concept")
    print("            Coverage(parallel) > 1.5 × Coverage(series)")
    print()

    # Try to import real graph, fall back to mock
    try:
        # Add the parent path for imports
        import sys
        from pathlib import Path
        parent = Path(__file__).parent.parent.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))

        from graph.meaning_graph import MeaningGraph
        graph = MeaningGraph()
        if graph.is_connected():
            print("Using real MeaningGraph...")
        else:
            raise Exception("Graph not connected")
    except Exception as e:
        print(f"Could not load real graph: {e}")
        print("Using mock graph for testing...")

        # Mock graph for testing
        class MockGraph:
            def __init__(self):
                self.concepts = {
                    "wisdom": {"tau": 2.5, "j": [0.5, 0.3, 0.2, 0.4, 0.6]},
                    "knowledge": {"tau": 2.0, "j": [0.4, 0.2, 0.1, 0.5, 0.3]},
                    "truth": {"tau": 2.8, "j": [0.3, 0.4, 0.3, 0.6, 0.4]},
                    "experience": {"tau": 1.8, "j": [0.6, 0.5, 0.2, 0.3, 0.5]},
                    "insight": {"tau": 2.3, "j": [0.5, 0.4, 0.3, 0.5, 0.4]},
                    "humility": {"tau": 2.2, "j": [0.2, 0.3, 0.4, 0.7, 0.6]},
                    "acceptance": {"tau": 2.0, "j": [0.3, 0.5, 0.3, 0.6, 0.7]},
                    "love": {"tau": 2.6, "j": [0.8, 0.9, 0.5, 0.7, 0.9]},
                    "feeling": {"tau": 1.7, "j": [0.6, 0.7, 0.3, 0.4, 0.6]},
                    "word": {"tau": 1.4, "j": [0.3, 0.2, 0.1, 0.3, 0.2]},
                    "hand": {"tau": 1.2, "j": [0.2, 0.3, 0.1, 0.2, 0.3]},
                    "thing": {"tau": 1.3, "j": [0.2, 0.2, 0.1, 0.3, 0.2]},
                }
                self.neighbors = {
                    "wisdom": ["knowledge", "truth", "experience", "insight", "humility"],
                    "knowledge": ["truth", "insight", "wisdom", "word"],
                    "truth": ["wisdom", "knowledge", "insight"],
                    "experience": ["wisdom", "feeling", "insight"],
                    "insight": ["wisdom", "truth", "knowledge", "experience"],
                    "humility": ["wisdom", "acceptance", "love"],
                    "acceptance": ["humility", "love", "feeling"],
                    "love": ["humility", "acceptance", "feeling", "wisdom"],
                    "feeling": ["experience", "love", "acceptance"],
                    "word": ["knowledge", "thing", "hand"],
                    "hand": ["thing", "word"],
                    "thing": ["word", "hand"],
                }

            def get_concept(self, word):
                return self.concepts.get(word, {"tau": 1.5, "j": [0.3, 0.3, 0.3, 0.3, 0.3]})

            def get_neighbors(self, word):
                return self.neighbors.get(word, [])

        graph = MockGraph()

    print("\nRunning tests...")
    result = run_experiment(graph)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.summary['successful_tests']}/{result.summary['total_seeds']}")
    print(f"Average coverage ratio: {result.summary['avg_coverage_ratio']:.2f}")
    print(f"Confirmation rate: {result.summary['confirmation_rate']:.1%}")
    print(f"Hypothesis: {result.summary['hypothesis_status']}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    save_results(result, output_dir)

    return result


if __name__ == "__main__":
    main()
