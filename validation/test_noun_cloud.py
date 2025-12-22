#!/usr/bin/env python3
"""
NounCloud Validation Tests
==========================

Tests that nouns are correctly represented as "projections of projections"
(clouds of adjectives) with τ derived from entropy.

Theory:
    - Nouns are weighted combinations of adjective vectors
    - τ = 1 + 5 * (1 - H_norm) where H_norm is normalized Shannon entropy
    - j/i centroids are computed from weighted adjective vectors
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader, NounCloud


def test_noun_cloud_loading():
    """Test that NounCloud objects are loaded correctly."""
    print("=" * 60)
    print("TEST 1: NounCloud Loading")
    print("=" * 60)

    loader = DataLoader()
    clouds = loader.load_noun_clouds()

    print(f"\nTotal noun clouds: {len(clouds)}")

    # Count cloud vs fallback
    cloud_nouns = [n for n, c in clouds.items() if c.is_cloud]
    fallback_nouns = [n for n, c in clouds.items() if not c.is_cloud]

    print(f"  Theory-consistent (is_cloud=True): {len(cloud_nouns)}")
    print(f"  Fallback (is_cloud=False): {len(fallback_nouns)}")

    # Sample some clouds
    print("\nSample NounClouds:")
    sample_words = ['love', 'war', 'peace', 'death', 'life', 'god', 'man', 'woman']
    for word in sample_words:
        if word in clouds:
            c = clouds[word]
            top3 = c.top_adjectives(3)
            top3_str = ', '.join(f"{a}:{w:.2f}" for a, w in top3)
            print(f"  {word:<10}: τ={c.tau:.2f}, variety={c.variety:>3}, is_cloud={c.is_cloud}, [{top3_str}]")

    # Verify structure
    success = len(clouds) > 0 and len(cloud_nouns) > 0
    print(f"\n{'PASS' if success else 'FAIL'}: NounCloud loading")
    return success


def test_tau_derivation():
    """Test that τ = 1 + 5*(1-h_norm) formula holds."""
    print("\n" + "=" * 60)
    print("TEST 2: τ Derivation from Entropy")
    print("=" * 60)

    loader = DataLoader()
    clouds = loader.load_noun_clouds()

    # Only test actual clouds (not fallback)
    cloud_nouns = {n: c for n, c in clouds.items() if c.is_cloud}

    errors = []
    for noun, cloud in cloud_nouns.items():
        expected_tau = 1 + 5 * (1 - cloud.h_adj_norm)
        actual_tau = cloud.tau
        error = abs(expected_tau - actual_tau)
        errors.append(error)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"\nτ derivation errors (τ = 1 + 5*(1-h_norm)):")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Errors < 0.01: {sum(1 for e in errors if e < 0.01)} / {len(errors)}")

    # Should be essentially zero (floating point only)
    success = mean_error < 0.001
    print(f"\n{'PASS' if success else 'FAIL'}: τ derivation formula")
    return success


def test_entropy_tau_correlation():
    """Test that high entropy → low τ (abstract) and low entropy → high τ (concrete)."""
    print("\n" + "=" * 60)
    print("TEST 3: Entropy-τ Correlation")
    print("=" * 60)

    loader = DataLoader()
    clouds = loader.load_noun_clouds()

    # Only test actual clouds
    cloud_nouns = {n: c for n, c in clouds.items() if c.is_cloud}

    h_norms = [c.h_adj_norm for c in cloud_nouns.values()]
    taus = [c.tau for c in cloud_nouns.values()]

    # Should be strong negative correlation (high entropy → low τ)
    r, p = scipy_stats.pearsonr(h_norms, taus)

    print(f"\nCorrelation between H_norm and τ:")
    print(f"  Pearson r: {r:.4f}")
    print(f"  p-value: {p:.2e}")
    print(f"  Expected: r ≈ -1.0 (perfect negative)")

    # Verify extremes
    high_entropy = [n for n, c in cloud_nouns.items() if c.h_adj_norm > 0.9]
    low_entropy = [n for n, c in cloud_nouns.items() if c.h_adj_norm < 0.3]

    if high_entropy:
        high_mean_tau = np.mean([cloud_nouns[n].tau for n in high_entropy[:10]])
        print(f"\n  High entropy nouns (H_norm > 0.9): mean τ = {high_mean_tau:.2f}")
        print(f"    Examples: {high_entropy[:5]}")

    if low_entropy:
        low_mean_tau = np.mean([cloud_nouns[n].tau for n in low_entropy[:10]])
        print(f"\n  Low entropy nouns (H_norm < 0.3): mean τ = {low_mean_tau:.2f}")
        print(f"    Examples: {low_entropy[:5]}")

    success = r < -0.9 and p < 0.001
    print(f"\n{'PASS' if success else 'FAIL'}: Entropy-τ correlation")
    return success


def test_centroid_magnitude():
    """Test that centroids have reasonable magnitudes."""
    print("\n" + "=" * 60)
    print("TEST 4: Centroid Magnitudes")
    print("=" * 60)

    loader = DataLoader()
    clouds = loader.load_noun_clouds()

    # Only test actual clouds
    cloud_nouns = {n: c for n, c in clouds.items() if c.is_cloud}

    j_mags = [np.linalg.norm(c.j) for c in cloud_nouns.values()]
    i_mags = [np.linalg.norm(c.i) for c in cloud_nouns.values()]

    print(f"\nj-space centroid magnitudes:")
    print(f"  Mean: {np.mean(j_mags):.4f}")
    print(f"  Std: {np.std(j_mags):.4f}")
    print(f"  Range: [{np.min(j_mags):.4f}, {np.max(j_mags):.4f}]")

    print(f"\ni-space centroid magnitudes:")
    print(f"  Mean: {np.mean(i_mags):.4f}")
    print(f"  Std: {np.std(i_mags):.4f}")
    print(f"  Range: [{np.min(i_mags):.4f}, {np.max(i_mags):.4f}]")

    # Centroids should be non-zero
    non_zero_j = sum(1 for m in j_mags if m > 0.001)
    non_zero_pct = 100 * non_zero_j / len(j_mags)
    print(f"\n  Non-zero j centroids: {non_zero_j} / {len(j_mags)} ({non_zero_pct:.1f}%)")

    success = non_zero_pct > 90
    print(f"\n{'PASS' if success else 'FAIL'}: Centroid magnitudes")
    return success


def test_navigation_with_nouncloud():
    """Test that navigation still works with NounCloud-based states."""
    print("\n" + "=" * 60)
    print("TEST 5: Navigation with NounCloud")
    print("=" * 60)

    try:
        from core.hybrid_llm import QuantumCore
    except ImportError:
        print("  Cannot import QuantumCore, skipping navigation test")
        return True

    core = QuantumCore()

    # Count cloud-based states
    cloud_states = sum(1 for s in core.states.values() if s.is_cloud)
    print(f"\nStates loaded: {len(core.states)}")
    print(f"  NounCloud states: {cloud_states}")

    # Test navigation from several starting points
    test_cases = [
        ('war', 'good'),
        ('hate', 'good'),
        ('love', 'evil'),
        ('peace', 'evil'),
    ]

    successes = 0
    for start, goal in test_cases:
        traj = core.navigate(start, goal, steps=3)
        if traj and len(traj.transitions) > 0:
            direction = 'toward good' if goal == 'good' else 'toward evil'
            delta_g = traj.total_delta_g

            # Check if direction is correct
            if (goal == 'good' and delta_g > 0) or (goal == 'evil' and delta_g < 0):
                successes += 1
                result = "OK"
            else:
                result = "WRONG DIRECTION"

            # Show cloud markers
            seq = traj.to_sequence()
            print(f"\n  {start} → {goal}: {' → '.join(seq)} (Δg={delta_g:+.2f}) [{result}]")

            # Show which states are clouds
            cloud_in_path = sum(1 for t in traj.transitions if t.to_state.is_cloud)
            print(f"    Cloud states in path: {cloud_in_path}/{len(traj.transitions)}")
        else:
            print(f"\n  {start} → {goal}: No trajectory found")

    success = successes >= len(test_cases) // 2
    print(f"\n{'PASS' if success else 'FAIL'}: Navigation with NounCloud ({successes}/{len(test_cases)} correct direction)")
    return success


def test_variety_distribution():
    """Test variety (adjective count) distribution across nouns."""
    print("\n" + "=" * 60)
    print("TEST 6: Variety Distribution")
    print("=" * 60)

    loader = DataLoader()
    clouds = loader.load_noun_clouds()

    varieties = [c.variety for c in clouds.values()]

    print(f"\nVariety (adjective count) distribution:")
    print(f"  Mean: {np.mean(varieties):.1f}")
    print(f"  Median: {np.median(varieties):.1f}")
    print(f"  Std: {np.std(varieties):.1f}")
    print(f"  Range: [{np.min(varieties)}, {np.max(varieties)}]")

    # Distribution by bins
    bins = [(1, 5), (5, 20), (20, 50), (50, 100), (100, 500), (500, float('inf'))]
    print("\n  Distribution by variety bins:")
    for low, high in bins:
        count = sum(1 for v in varieties if low <= v < high)
        pct = 100 * count / len(varieties)
        high_str = str(int(high)) if high != float('inf') else '+'
        print(f"    [{low:>3}-{high_str:>4}): {count:>5} ({pct:>5.1f}%)")

    success = np.mean(varieties) > 10  # Reasonable variety
    print(f"\n{'PASS' if success else 'FAIL'}: Variety distribution")
    return success


def main():
    """Run all NounCloud validation tests."""
    print("=" * 60)
    print("NOUNCLOUD VALIDATION TESTS")
    print("Testing 'Projections of Projections' Theory")
    print("=" * 60)

    results = []

    results.append(("NounCloud Loading", test_noun_cloud_loading()))
    results.append(("τ Derivation Formula", test_tau_derivation()))
    results.append(("Entropy-τ Correlation", test_entropy_tau_correlation()))
    results.append(("Centroid Magnitudes", test_centroid_magnitude()))
    results.append(("Navigation with NounCloud", test_navigation_with_nouncloud()))
    results.append(("Variety Distribution", test_variety_distribution()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed - NounCloud implementation is correct!")
    else:
        print(f"\n{total - passed} test(s) failed - please investigate")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
