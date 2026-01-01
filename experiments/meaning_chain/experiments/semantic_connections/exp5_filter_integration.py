#!/usr/bin/env python3
"""
Experiment 5: Filter Integration Validation
============================================

Tests the τ-level filter integration into the navigator:

    GROUNDED: Low-pass (τ < 1.74) - concrete concepts
    DEEP: High-pass (τ > 2.1) - abstract concepts
    WISDOM: Band-pass (1.3 < τ < 1.6) - wisdom zone
    FILTERED: Verb-inferred filter

Experiments:
1. Test filter routing from verbs
2. Test FILTERED goal with different query types
3. Compare FILTERED vs direct goals
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

# Add paths
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _EXPERIMENT_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))


def exp5_1_filter_routing():
    """
    Test 1: Filter routing from verbs.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5.1: Filter Routing from Verbs")
    print("=" * 60)

    from chain_core.navigator import SemanticNavigator

    nav = SemanticNavigator()

    # Test verb → filter inference
    test_cases = [
        (["find", "get"], "grounded", "low-pass"),
        (["understand", "contemplate"], "deep", "high-pass"),
        (["know", "feel"], "wisdom", "band-pass"),
        (["explore"], "balanced", "band-pass"),
    ]

    results = []

    try:
        for verbs, expected_goal, expected_filter_type in test_cases:
            inferred = nav._infer_filter_goal_from_verbs(verbs)

            match = (inferred == expected_goal)
            status = "✓" if match else "✗"

            print(f"  {status} Verbs {verbs}: inferred={inferred}, expected={expected_goal}")

            results.append({
                "verbs": verbs,
                "inferred": inferred,
                "expected": expected_goal,
                "match": match,
            })

    finally:
        nav.close()

    passed = sum(1 for r in results if r["match"])
    print(f"\n  Filter Routing: {passed}/{len(results)} tests passed")

    return results, passed == len(results)


def exp5_2_filtered_navigation():
    """
    Test 2: FILTERED goal with different query types.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5.2: FILTERED Goal Navigation")
    print("=" * 60)

    from chain_core.navigator import SemanticNavigator

    nav = SemanticNavigator()

    test_queries = [
        ("Find me a practical solution", "grounded", 1.74),
        ("Help me understand the nature of consciousness", "deep", 2.1),
        ("I want to know about wisdom", "wisdom", 1.45),
    ]

    results = []

    try:
        for query, expected_filter, target_tau in test_queries:
            print(f"\n  Query: {query}")

            result = nav.navigate(query, goal="filtered")

            print(f"    Strategy: {result.strategy}")
            print(f"    Concepts: {result.concepts[:5]}")
            print(f"    τ_mean: {result.quality.tau_mean:.2f} (target: {target_tau})")
            print(f"    Quality: R={result.quality.resonance:.2f}, "
                  f"C={result.quality.coherence:.2f}, "
                  f"S={result.quality.stability:.2f}")

            # Check if filter was correctly inferred
            filter_match = expected_filter in result.strategy

            # Check if τ is in expected range
            if expected_filter == "grounded":
                tau_match = result.quality.tau_mean < 2.0
            elif expected_filter == "deep":
                tau_match = result.quality.tau_mean > 1.8
            else:  # wisdom
                tau_match = 1.2 < result.quality.tau_mean < 2.0

            status = "✓" if (filter_match and tau_match) else "✗"
            print(f"    {status} Filter: {filter_match}, τ range: {tau_match}")

            results.append({
                "query": query,
                "expected_filter": expected_filter,
                "strategy": result.strategy,
                "tau_mean": result.quality.tau_mean,
                "filter_match": filter_match,
                "tau_match": tau_match,
            })

    finally:
        nav.close()

    passed = sum(1 for r in results if r["filter_match"] and r["tau_match"])
    print(f"\n  FILTERED Navigation: {passed}/{len(results)} tests passed")

    return results, passed >= len(results) * 0.6


def exp5_3_filtered_vs_direct():
    """
    Test 3: Compare FILTERED with direct goal navigation.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5.3: FILTERED vs Direct Goals")
    print("=" * 60)

    from chain_core.navigator import SemanticNavigator

    nav = SemanticNavigator()

    query = "What is wisdom?"

    goals = ["filtered", "grounded", "deep", "wisdom", "resonant"]

    results = []

    try:
        for goal in goals:
            result = nav.navigate(query, goal=goal)

            print(f"\n  {goal.upper()}:")
            print(f"    Strategy: {result.strategy}")
            print(f"    Concepts: {result.concepts[:5]}")
            print(f"    τ_mean: {result.quality.tau_mean:.2f}")
            print(f"    R={result.quality.resonance:.2f}, "
                  f"C={result.quality.coherence:.2f}, "
                  f"S={result.quality.stability:.2f}")

            results.append({
                "goal": goal,
                "strategy": result.strategy,
                "concepts": result.concepts[:5],
                "tau_mean": result.quality.tau_mean,
                "resonance": result.quality.resonance,
                "coherence": result.quality.coherence,
                "stability": result.quality.stability,
            })

    finally:
        nav.close()

    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Goal':<12} {'τ_mean':<8} {'R':<6} {'C':<6} {'S':<6}")
    print("-" * 40)
    for r in results:
        print(f"{r['goal']:<12} {r['tau_mean']:<8.2f} "
              f"{r['resonance']:<6.2f} {r['coherence']:<6.2f} {r['stability']:<6.2f}")

    return results, True


def run_all_experiments():
    """Run all filter integration experiments."""
    print("=" * 70)
    print("EXPERIMENT 5: FILTER INTEGRATION VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {}
    all_passed = True

    # Test 1: Filter routing
    results, passed = exp5_1_filter_routing()
    all_results["exp5_1_filter_routing"] = results
    all_passed = all_passed and passed

    # Test 2: FILTERED navigation
    results, passed = exp5_2_filtered_navigation()
    all_results["exp5_2_filtered_navigation"] = results
    all_passed = all_passed and passed

    # Test 3: FILTERED vs direct
    results, passed = exp5_3_filtered_vs_direct()
    all_results["exp5_3_filtered_comparison"] = results

    # Save results
    output_dir = _EXPERIMENT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp5_filter_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"All experiments completed: {'✓ PASS' if all_passed else '✗ PARTIAL'}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_all_experiments()
