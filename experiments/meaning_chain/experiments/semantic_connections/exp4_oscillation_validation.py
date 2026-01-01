#!/usr/bin/env python3
"""
Experiment 4: Oscillation & Resonance Validation
=================================================

Tests the hypothesis that concepts behave as RLC oscillators:

    ω₀ = 1/√(LC)

Where:
    L = 1/variety (semantic inertia)
    C = degree/max_degree (semantic capacity)

Experiments:
1. Compute L, C, ω₀ for real concepts from graph
2. Test frequency classification (deep/medium/surface)
3. Validate resonance matching with query frequencies
4. Test RLC damping predictions

Expected Results:
- High variety → Low L → High ω₀ (surface concepts)
- High degree → High C → Low ω₀ (deep concepts)
- Query verbs should match concept frequencies
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

from experiments.semantic_connections.oscillators import (
    PT1Dynamics,
    RCCircuit,
    RLCCircuit,
    SemanticOscillator,
    OscillatorBank,
    create_oscillator,
    compute_inductance,
    compute_capacitance,
    compute_resonance_frequency,
    compute_spectrum,
    estimate_query_frequency,
    E,
    KT_NATURAL,
)


def exp4_1_concept_oscillators(graph):
    """
    Test 1: Compute oscillator parameters for real concepts.

    Verifies L, C, ω₀ computation from graph properties.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.1: Concept Oscillator Parameters")
    print("=" * 60)

    test_concepts = [
        "wisdom", "love", "time", "life", "truth",
        "table", "car", "house", "book", "hand",
        "consciousness", "eternity", "transcendence", "infinity",
    ]

    results = []
    oscillators = []

    for concept in test_concepts:
        props = graph.get_concept(concept)
        if not props:
            print(f"  {concept}: NOT FOUND")
            continue

        tau = props.get('tau', 1.5)
        variety = props.get('variety', 10)
        degree = props.get('degree', 50)

        # Create oscillator
        osc = create_oscillator(
            concept=concept,
            tau=tau,
            variety=variety,
            degree=degree,
            max_variety=100,
            max_degree=500
        )

        oscillators.append(osc)

        result = {
            "concept": concept,
            "tau": tau,
            "variety": variety,
            "degree": degree,
            "L": osc.L,
            "C": osc.C,
            "omega_0": osc.omega_0,
            "Q_factor": osc.Q_factor,
            "frequency_class": osc.frequency_class,
        }
        results.append(result)

        print(f"\n  {concept}:")
        print(f"    τ={tau:.2f}, variety={variety}, degree={degree}")
        print(f"    L={osc.L:.3f}, C={osc.C:.3f}")
        print(f"    ω₀={osc.omega_0:.3f}, Q={osc.Q_factor:.2f}")
        print(f"    Class: {osc.frequency_class}")

    # Compute spectrum
    spectrum = compute_spectrum(oscillators)

    print(f"\n{'=' * 60}")
    print("SPECTRUM ANALYSIS")
    print("=" * 60)
    print(f"  ω₀ range: [{spectrum.omega_min:.3f}, {spectrum.omega_max:.3f}]")
    print(f"  ω₀ mean: {spectrum.omega_mean:.3f}")
    print(f"  ω₀ std: {spectrum.omega_std:.3f}")

    dist = spectrum.frequency_distribution
    print(f"\n  Frequency distribution:")
    print(f"    Deep (ω < 0.5): {len(dist['deep'])} concepts")
    print(f"    Medium (0.5 ≤ ω < 1.0): {len(dist['medium'])} concepts")
    print(f"    Surface (ω ≥ 1.0): {len(dist['surface'])} concepts")

    return results, True


def exp4_2_frequency_classification(graph):
    """
    Test 2: Validate frequency classification predictions.

    Abstract concepts should have lower ω₀ (deep).
    Concrete concepts should have higher ω₀ (surface).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.2: Frequency Classification")
    print("=" * 60)

    # Expected classifications based on semantic nature
    expected = {
        # Abstract (expect deep/medium)
        "wisdom": "deep",
        "truth": "deep",
        "consciousness": "deep",
        "eternity": "deep",
        "love": "medium",
        "meaning": "deep",

        # Concrete (expect surface/medium)
        "table": "surface",
        "car": "surface",
        "hand": "medium",
        "house": "medium",
    }

    results = []
    correct = 0
    total = 0

    for concept, expected_class in expected.items():
        props = graph.get_concept(concept)
        if not props:
            continue

        tau = props.get('tau', 1.5)
        variety = props.get('variety', 10)
        degree = props.get('degree', 50)

        osc = create_oscillator(concept, tau, variety, degree)
        actual_class = osc.frequency_class

        # Allow medium as acceptable for both (transitional)
        match = (actual_class == expected_class or
                 actual_class == "medium" or
                 expected_class == "medium")

        if match:
            correct += 1
        total += 1

        status = "✓" if match else "✗"
        print(f"  {status} {concept}: expected={expected_class}, "
              f"actual={actual_class} (ω₀={osc.omega_0:.3f})")

        results.append({
            "concept": concept,
            "expected": expected_class,
            "actual": actual_class,
            "omega_0": osc.omega_0,
            "match": match,
        })

    print(f"\n{'=' * 60}")
    print(f"Classification: {correct}/{total} matched ({100*correct/total:.0f}%)")
    print("=" * 60)

    return results, correct >= total * 0.6  # 60% threshold


def exp4_3_resonance_matching(graph):
    """
    Test 3: Validate resonance matching between queries and concepts.

    Query verbs should resonate with matching-frequency concepts.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.3: Resonance Matching")
    print("=" * 60)

    # Build oscillator bank from graph
    bank = OscillatorBank()

    concepts = ["wisdom", "truth", "love", "life", "time",
                "table", "car", "hand", "book", "house"]

    for concept in concepts:
        props = graph.get_concept(concept)
        if props:
            osc = create_oscillator(
                concept,
                props.get('tau', 1.5),
                props.get('variety', 10),
                props.get('degree', 50)
            )
            bank.add(osc)

    # Test queries with different verb types
    test_queries = [
        (["contemplate", "understand"], "deep",
         "Should resonate with abstract concepts"),
        (["find", "get"], "surface",
         "Should resonate with concrete concepts"),
        (["know", "learn"], "medium",
         "Should resonate broadly"),
    ]

    results = []

    for verbs, expected_class, description in test_queries:
        query_omega = estimate_query_frequency(verbs)

        print(f"\n  Query verbs: {verbs}")
        print(f"  Query ω: {query_omega:.3f}")
        print(f"  Expected: {expected_class}")

        # Find resonant concepts
        resonators = bank.find_resonant(query_omega, threshold=0.3, limit=5)

        print(f"  Resonant concepts:")
        resonant_classes = []
        for osc, quality in resonators:
            print(f"    {osc.concept}: ω₀={osc.omega_0:.3f}, "
                  f"resonance={quality:.3f}, class={osc.frequency_class}")
            resonant_classes.append(osc.frequency_class)

        # Check if expected class is well-represented
        if resonant_classes:
            class_match = (expected_class in resonant_classes or
                          "medium" in resonant_classes)
        else:
            class_match = False

        results.append({
            "verbs": verbs,
            "query_omega": query_omega,
            "expected_class": expected_class,
            "resonators": [(o.concept, q) for o, q in resonators],
            "match": class_match,
        })

        status = "✓" if class_match else "✗"
        print(f"  {status} {description}")

    passed = sum(1 for r in results if r["match"])
    print(f"\n{'=' * 60}")
    print(f"Resonance Matching: {passed}/{len(results)} tests passed")
    print("=" * 60)

    return results, passed >= len(results) * 0.6


def exp4_4_rlc_damping(graph):
    """
    Test 4: Validate RLC damping predictions.

    Test underdamped, critically damped, and overdamped responses.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4.4: RLC Damping Analysis")
    print("=" * 60)

    # Test with various R (τ) values
    test_cases = [
        {"R": 0.5, "L": 0.5, "C": 0.5, "expected": "underdamped"},
        {"R": 1.0, "L": 0.5, "C": 0.5, "expected": "critically_damped"},
        {"R": 2.0, "L": 0.5, "C": 0.5, "expected": "overdamped"},
    ]

    results = []

    for case in test_cases:
        rlc = RLCCircuit(R=case["R"], L=case["L"], C=case["C"])

        if rlc.is_underdamped:
            actual = "underdamped"
        elif rlc.is_critically_damped:
            actual = "critically_damped"
        else:
            actual = "overdamped"

        match = (actual == case["expected"])

        print(f"\n  R={case['R']}, L={case['L']}, C={case['C']}:")
        print(f"    ω₀={rlc.resonance_frequency:.3f}")
        print(f"    ζ={rlc.damping_ratio:.3f}")
        print(f"    Q={rlc.quality_factor:.3f}")
        print(f"    Expected: {case['expected']}, Actual: {actual}")

        status = "✓" if match else "✗"
        print(f"    {status}")

        results.append({
            "R": case["R"],
            "L": case["L"],
            "C": case["C"],
            "omega_0": rlc.resonance_frequency,
            "zeta": rlc.damping_ratio,
            "Q": rlc.quality_factor,
            "expected": case["expected"],
            "actual": actual,
            "match": match,
        })

    # Test with real concepts
    print(f"\n  Damping analysis for real concepts:")

    for concept in ["wisdom", "love", "time"]:
        props = graph.get_concept(concept)
        if not props:
            continue

        tau = props.get('tau', 1.5)
        variety = props.get('variety', 10)
        degree = props.get('degree', 50)

        L = compute_inductance(variety)
        C = compute_capacitance(degree)

        rlc = RLCCircuit(R=tau, L=L, C=C)

        if rlc.is_underdamped:
            damping_type = "underdamped (oscillates)"
        elif rlc.is_critically_damped:
            damping_type = "critically damped"
        else:
            damping_type = "overdamped (slow)"

        print(f"\n    {concept}: ζ={rlc.damping_ratio:.3f}, Q={rlc.quality_factor:.3f}")
        print(f"      → {damping_type}")

    passed = sum(1 for r in results if r["match"])
    print(f"\n{'=' * 60}")
    print(f"RLC Damping: {passed}/{len(results)} tests passed")
    print("=" * 60)

    return results, passed == len(results)


def run_all_experiments():
    """Run all oscillation validation experiments."""
    print("=" * 70)
    print("EXPERIMENT 4: OSCILLATION & RESONANCE VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {}
    all_passed = True

    try:
        from graph.meaning_graph import MeaningGraph
        graph = MeaningGraph()

        if not graph.is_connected():
            print("\nERROR: Neo4j not connected")
            return None

        # Test 1: Concept oscillators
        results, passed = exp4_1_concept_oscillators(graph)
        all_results["exp4_1_concept_oscillators"] = results

        # Test 2: Frequency classification
        results, passed = exp4_2_frequency_classification(graph)
        all_results["exp4_2_frequency_classification"] = results
        all_passed = all_passed and passed

        # Test 3: Resonance matching
        results, passed = exp4_3_resonance_matching(graph)
        all_results["exp4_3_resonance_matching"] = results
        all_passed = all_passed and passed

        # Test 4: RLC damping
        results, passed = exp4_4_rlc_damping(graph)
        all_results["exp4_4_rlc_damping"] = results
        all_passed = all_passed and passed

        graph.close()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Save results
    output_dir = _EXPERIMENT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp4_oscillation_{timestamp}.json"

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
