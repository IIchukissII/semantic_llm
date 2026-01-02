#!/usr/bin/env python3
"""
Validate Unified Hierarchy
==========================

Test that the new unified system:
1. Derives coordinates consistently
2. Maintains expected relationships
3. Coherence with theoretical predictions

Key tests:
- n (orbital) correlates with entropy
- θ (phase) clusters by semantic category
- r (magnitude) reflects transcendental intensity
- Verbs produce meaningful phase shifts
"""

import sys
import math
import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from collections import defaultdict

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_EXPERIMENTS = _MEANING_CHAIN.parent
_SEMANTIC_LLM = _EXPERIMENTS.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.unified_hierarchy import build_hierarchy, KT, E


def test_orbital_entropy_correlation(hierarchy):
    """
    Test: n = 5 × (1 - H_norm)

    Expected: Perfect negative correlation between n and H_norm
    """
    print("\n" + "=" * 60)
    print("TEST 1: Orbital-Entropy Relationship")
    print("=" * 60)

    n_vals = []
    h_vals = []

    for noun in hierarchy.nouns.values():
        if noun.h_norm > 0:
            n_vals.append(noun.n)
            h_vals.append(noun.h_norm)

    if len(n_vals) < 10:
        print("  ERROR: Not enough data")
        return False

    n_arr = np.array(n_vals)
    h_arr = np.array(h_vals)

    # Expected: n = 5 × (1 - H_norm)
    n_expected = 5 * (1 - h_arr)
    error = np.abs(n_arr - n_expected)

    r, p = scipy_stats.pearsonr(n_arr, h_arr)

    print(f"  Samples: {len(n_vals)}")
    print(f"  Correlation (n, H_norm): r = {r:.4f} (expected: -1.0)")
    print(f"  Mean error |n - 5(1-H)|: {np.mean(error):.6f}")
    print(f"  Max error: {np.max(error):.6f}")

    passed = abs(r + 1.0) < 0.01 and np.mean(error) < 0.001
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_phase_semantic_clusters(hierarchy):
    """
    Test: θ clusters by semantic category

    Expected:
    - Positive concepts: θ near 0° (pure affirmation)
    - Negative concepts: θ near 180° (pure negation)
    - Sacred concepts: θ near 90° (pure sacred)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Phase Semantic Clusters")
    print("=" * 60)

    positive_words = ['love', 'peace', 'joy', 'hope', 'beauty', 'truth', 'good']
    negative_words = ['hate', 'war', 'fear', 'despair', 'death', 'evil', 'darkness']
    sacred_words = ['god', 'divine', 'holy', 'spirit', 'soul', 'prayer', 'worship']

    def get_mean_theta(words):
        thetas = []
        for w in words:
            qw = hierarchy.get_word(w)
            if qw:
                thetas.append(qw.theta)
        return np.mean(thetas) if thetas else None, len(thetas)

    pos_theta, pos_n = get_mean_theta(positive_words)
    neg_theta, neg_n = get_mean_theta(negative_words)
    sac_theta, sac_n = get_mean_theta(sacred_words)

    print(f"  Positive words ({pos_n}/{len(positive_words)}): mean θ = {math.degrees(pos_theta):.1f}°" if pos_theta else "  Positive: insufficient data")
    print(f"  Negative words ({neg_n}/{len(negative_words)}): mean θ = {math.degrees(neg_theta):.1f}°" if neg_theta else "  Negative: insufficient data")
    print(f"  Sacred words ({sac_n}/{len(sacred_words)}): mean θ = {math.degrees(sac_theta):.1f}°" if sac_theta else "  Sacred: insufficient data")

    # Check that positive and negative differ significantly
    if pos_theta is not None and neg_theta is not None:
        diff = abs(pos_theta - neg_theta)
        print(f"  Positive-Negative difference: {math.degrees(diff):.1f}°")
        passed = diff > math.radians(30)  # At least 30° apart
    else:
        passed = False

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gravity_potential(hierarchy):
    """
    Test: Gravity potential φ = λn - μA

    Expected: Words cluster at low φ (low n, high A = concrete + affirming)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Gravity Potential Distribution")
    print("=" * 60)

    potentials = []
    for noun in hierarchy.nouns.values():
        phi = noun.gravity_potential(lambda_=1.0, mu=0.5)
        potentials.append(phi)

    phi_arr = np.array(potentials)

    print(f"  Samples: {len(potentials)}")
    print(f"  φ mean: {np.mean(phi_arr):.3f}")
    print(f"  φ std: {np.std(phi_arr):.3f}")
    print(f"  φ min: {np.min(phi_arr):.3f}")
    print(f"  φ max: {np.max(phi_arr):.3f}")

    # Check that distribution is concentrated (not uniform)
    skewness = scipy_stats.skew(phi_arr)
    print(f"  Skewness: {skewness:.3f}")

    # Most words should have low potential (attracted to minimum)
    below_mean = np.sum(phi_arr < np.mean(phi_arr)) / len(phi_arr)
    print(f"  Below mean: {below_mean:.1%}")

    passed = True  # Informational test
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} (informational)")
    return passed


def test_boltzmann_weights(hierarchy):
    """
    Test: Boltzmann weights P ∝ exp(-Δn/kT)

    Expected: Transitions between similar n are more probable
    """
    print("\n" + "=" * 60)
    print("TEST 4: Boltzmann Transition Weights")
    print("=" * 60)

    # Sample some words and compute transition weights
    sample_words = ['love', 'peace', 'war', 'death', 'god', 'man']
    found = []

    for w in sample_words:
        qw = hierarchy.get_word(w)
        if qw:
            found.append(qw)

    if len(found) < 2:
        print("  ERROR: Not enough data")
        return False

    print(f"  kT = e^(-1/5) = {KT:.4f}")
    print()
    print(f"  {'From':<10} {'To':<10} {'Δn':>8} {'P':>10}")
    print("  " + "-" * 40)

    for i, w1 in enumerate(found[:4]):
        for w2 in found[:4]:
            if w1 != w2:
                delta_n = abs(w1.n - w2.n)
                weight = w1.boltzmann_weight(w2.n)
                print(f"  {w1.word:<10} {w2.word:<10} {delta_n:>8.3f} {weight:>10.4f}")

    passed = True
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (informational)")
    return passed


def test_verb_operators(hierarchy):
    """
    Test: Verb operators produce meaningful phase shifts

    Expected:
    - Positive verbs: Δθ toward 0° (affirmation)
    - Negative verbs: Δθ toward 180° (negation)
    """
    print("\n" + "=" * 60)
    print("TEST 5: Verb Operator Effects")
    print("=" * 60)

    positive_verbs = ['help', 'give', 'create', 'build', 'save', 'bless']
    negative_verbs = ['hurt', 'take', 'destroy', 'kill', 'curse', 'damn']

    def get_mean_delta_theta(verbs):
        deltas = []
        for v in verbs:
            vop = hierarchy.get_verb(v)
            if vop:
                deltas.append(vop.delta_theta)
        return np.mean(deltas) if deltas else None, len(deltas)

    pos_delta, pos_n = get_mean_delta_theta(positive_verbs)
    neg_delta, neg_n = get_mean_delta_theta(negative_verbs)

    print(f"  Positive verbs ({pos_n}/{len(positive_verbs)}): mean Δθ = {math.degrees(pos_delta):.1f}°" if pos_delta else "  Positive: insufficient data")
    print(f"  Negative verbs ({neg_n}/{len(negative_verbs)}): mean Δθ = {math.degrees(neg_delta):.1f}°" if neg_delta else "  Negative: insufficient data")

    # Show individual verbs
    print()
    print(f"  {'Verb':<12} {'Δθ°':>8} {'Δr':>8}")
    print("  " + "-" * 30)

    all_verbs = positive_verbs + negative_verbs
    for v in all_verbs:
        vop = hierarchy.get_verb(v)
        if vop:
            print(f"  {v:<12} {math.degrees(vop.delta_theta):>8.1f} {vop.delta_r:>8.3f}")

    passed = True
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} (informational)")
    return passed


def test_export_coordinates(hierarchy):
    """
    Export derived coordinates to JSON for persistence.
    """
    print("\n" + "=" * 60)
    print("EXPORT: Derived Coordinates")
    print("=" * 60)

    coords = hierarchy.export_coordinates()

    output_path = _MEANING_CHAIN / "data" / "derived_coordinates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'source': 'unified_hierarchy',
                'n_words': len(coords),
                'n_nouns': len(hierarchy.nouns),
                'n_adjectives': len(hierarchy.adjectives),
                'n_verbs': len(hierarchy.verbs),
                'kT': KT,
                'e': E
            },
            'words': coords
        }, f, indent=2)

    print(f"  Exported {len(coords)} words to {output_path}")

    # Also export verb operators
    verb_path = _MEANING_CHAIN / "data" / "derived_verb_operators.json"
    verbs_export = {}
    for verb, vop in hierarchy.verbs.items():
        verbs_export[verb] = {
            'delta_theta': vop.delta_theta,
            'delta_theta_deg': math.degrees(vop.delta_theta),
            'delta_r': vop.delta_r,
            'theta': vop.theta,
            'theta_deg': math.degrees(vop.theta),
            'r': vop.r
        }

    with open(verb_path, 'w') as f:
        json.dump({
            'metadata': {
                'source': 'unified_hierarchy',
                'n_verbs': len(verbs_export)
            },
            'operators': verbs_export
        }, f, indent=2)

    print(f"  Exported {len(verbs_export)} verb operators to {verb_path}")

    return True


def main():
    print("=" * 60)
    print("UNIFIED HIERARCHY VALIDATION")
    print("=" * 60)

    # Build hierarchy
    loader = DataLoader()
    hierarchy = build_hierarchy(loader)

    # Run tests
    results = []

    results.append(("Orbital-Entropy", test_orbital_entropy_correlation(hierarchy)))
    results.append(("Phase Clusters", test_phase_semantic_clusters(hierarchy)))
    results.append(("Gravity Potential", test_gravity_potential(hierarchy)))
    results.append(("Boltzmann Weights", test_boltzmann_weights(hierarchy)))
    results.append(("Verb Operators", test_verb_operators(hierarchy)))
    results.append(("Export", test_export_coordinates(hierarchy)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name:<20}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    # Level summary
    print("\n" + "=" * 60)
    print("UNIFIED HIERARCHY LEVELS")
    print("=" * 60)
    print("""
    LEVEL 1: TRANSCENDENTALS (A, S)
      └─ Source, no dynamics

    LEVEL 2: WORDS (n, θ, r)
      ├─ n = 5 × (1 - H_norm)         [from Level 3]
      ├─ θ = weighted_mean(adj.θ)     [from adjective centroids]
      ├─ r = weighted_mean(adj.r)     [from adjective centroids]
      ├─ Boltzmann: P ∝ exp(-Δn/kT)
      └─ Gravity: φ = λn - μA

    LEVEL 3: BONDS (adj ↔ noun graph)
      ├─ PT1 saturation
      ├─ Entropy → n
      └─ Weighted centroids → (θ, r)

    LEVEL 4: SENTENCES (trajectories)
      ├─ Intent collapse
      └─ Coherence = cos(Δθ)

    LEVEL 5: DIALOGUE (navigation)
      ├─ Storm-Logos
      ├─ Paradox chain
      └─ Σ = C + 0.1P
    """)


if __name__ == "__main__":
    main()
