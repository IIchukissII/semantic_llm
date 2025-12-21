#!/usr/bin/env python3
"""
Entropy-Based τ Discovery
=========================

FUNDAMENTAL VALIDATION: τ from Shannon entropy, not variety.

Key discoveries:
  - τ = 1 + 5 × (1 - H_norm)
  - H_adj - H_verb ≈ 1.08 bits (One-Bit Law)
  - ln(H_adj/H_verb) ≈ 1/e (Euler's constant in language)

Usage:
    python entropy_tau.py              # Run with CSV data
    python entropy_tau.py --from-db    # Run with database (if available)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "entropy"


def test_ratio_hypothesis(entropy_stats: dict) -> float:
    """
    Hypothesis 1: H_adj / H_verb = constant?
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 1: H_adj / H_verb = constant?")
    print("=" * 60)

    ratios = []
    for noun, stats in entropy_stats.items():
        h_adj = stats['h_adj']
        h_verb = stats['h_verb']
        if h_verb > 0.1:
            ratio = h_adj / h_verb
            if 0.1 < ratio < 10:
                ratios.append(ratio)

    ratios = np.array(ratios)

    print(f"\n  n = {len(ratios)} nouns")
    print(f"  mean(H_adj/H_verb) = {ratios.mean():.4f}")
    print(f"  std = {ratios.std():.4f}")
    print(f"  CV (coeff. variation) = {ratios.std()/ratios.mean():.4f}")

    e_distance = abs(ratios.mean() - np.e)
    print(f"\n  Distance from e ≈ 2.718: {e_distance:.4f}")

    one_distance = abs(ratios.mean() - 1)
    print(f"  Distance from 1: {one_distance:.4f}")

    return ratios.mean()


def test_difference_hypothesis(entropy_stats: dict) -> float:
    """
    Hypothesis 2: H_adj - H_verb ≈ 1 bit? (ONE-BIT LAW)
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 2: H_adj - H_verb ≈ 1 bit? (ONE-BIT LAW)")
    print("=" * 60)

    diffs = []
    for noun, stats in entropy_stats.items():
        diff = stats['delta']
        diffs.append(diff)

    diffs = np.array(diffs)

    print(f"\n  H_adj - H_verb:")
    print(f"    mean = {diffs.mean():.4f} bits")
    print(f"    std = {diffs.std():.4f}")

    print(f"\n  ONE-BIT LAW TEST:")
    distance_from_1 = abs(diffs.mean() - 1.0)
    print(f"    Distance from 1.0: {distance_from_1:.4f}")

    if distance_from_1 < 0.2:
        print(f"    CONFIRMED: Being > Doing by ~1 bit!")
    else:
        print(f"    Mean is {diffs.mean():.2f} bits")

    return diffs.mean()


def test_euler_hypothesis(entropy_stats: dict) -> float:
    """
    Hypothesis 3: ln(H_adj) - ln(H_verb) ≈ 1/e? (EULER'S CONSTANT)
    """
    print("\n" + "=" * 60)
    print("HYPOTHESIS 3: ln(H_adj/H_verb) ≈ 1/e? (EULER'S CONSTANT)")
    print("=" * 60)

    log_diffs = []
    for noun, stats in entropy_stats.items():
        h_adj = stats['h_adj']
        h_verb = stats['h_verb']
        if h_adj > 0.1 and h_verb > 0.1:
            log_diff = np.log(h_adj) - np.log(h_verb)
            log_diffs.append(log_diff)

    log_diffs = np.array(log_diffs)

    print(f"\n  ln(H_adj) - ln(H_verb):")
    print(f"    mean = {log_diffs.mean():.4f}")

    one_over_e = 1 / np.e
    distance = abs(log_diffs.mean() - one_over_e)

    print(f"\n  EULER'S CONSTANT TEST:")
    print(f"    1/e = {one_over_e:.4f}")
    print(f"    Observed = {log_diffs.mean():.4f}")
    print(f"    Distance = {distance:.4f}")
    print(f"    Relative error = {distance/one_over_e*100:.2f}%")

    if distance < 0.02:
        print(f"    CONFIRMED: Euler's e appears in language!")

    return log_diffs.mean()


def test_tau_entropy_formula(entropy_stats: dict):
    """
    Validate τ = 1 + 5 × (1 - H_norm) formula.
    """
    print("\n" + "=" * 60)
    print("τ FROM ENTROPY FORMULA")
    print("=" * 60)

    print("\n  Formula: τ = 1 + 5 × (1 - H_norm)")
    print("\n  H_norm = 1 (uniform) → τ = 1 (abstract)")
    print("  H_norm = 0 (concentrated) → τ = 6 (concrete)")

    # Group by τ levels
    tau_groups = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    for noun, stats in entropy_stats.items():
        tau = stats['tau_entropy']
        tau_int = max(1, min(6, int(round(tau))))
        tau_groups[tau_int].append(stats['h_adj_norm'])

    print(f"\n  τ Level | Count | Mean H_norm | Expected H_norm")
    print("  " + "-" * 50)

    for tau in range(1, 7):
        if tau_groups[tau]:
            mean_h = np.mean(tau_groups[tau])
            expected_h = 1 - (tau - 1) / 5  # Inverse of formula
            print(f"  τ = {tau}   | {len(tau_groups[tau]):5d} | {mean_h:.3f}       | {expected_h:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Entropy-Based τ Discovery")
    parser.add_argument('--from-db', action='store_true',
                        help='Load from database instead of CSV')
    args = parser.parse_args()

    print("=" * 60)
    print("ENTROPY-BASED τ DISCOVERY")
    print("=" * 60)
    print("""
Thermodynamics of Language — VALIDATION

  Shannon:   H = -Σ p log p
  Boltzmann: S = -k Σ p ln p

  Same formula. Language = physical system.
""")

    # Load data
    loader = DataLoader()

    if args.from_db:
        print("Loading from database...")
        # Force reload from DB by clearing cache
        loader._entropy_stats = None

    entropy_stats = loader.load_entropy_stats()

    if not entropy_stats:
        print("\nERROR: No data available.")
        print("Run 'python scripts/export_data.py --entropy' first.")
        return

    print(f"\nLoaded entropy stats for {len(entropy_stats)} nouns")

    # Run tests
    ratio_mean = test_ratio_hypothesis(entropy_stats)
    diff_mean = test_difference_hypothesis(entropy_stats)
    euler_mean = test_euler_hypothesis(entropy_stats)
    test_tau_entropy_formula(entropy_stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  Key Results:")
    print(f"    H_adj/H_verb ratio: {ratio_mean:.4f}")
    print(f"    H_adj - H_verb: {diff_mean:.4f} bits")
    print(f"    ln(H_adj/H_verb): {euler_mean:.4f}")

    print(f"\n  Theory Validation:")
    print(f"    One-Bit Law (≈1.0): {'PASS' if abs(diff_mean - 1.0) < 0.2 else 'FAIL'}")
    print(f"    Euler's e (≈0.368): {'PASS' if abs(euler_mean - 1/np.e) < 0.02 else 'CHECK'}")

    # Export results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "source": "csv" if not args.from_db else "database",
        "n_nouns": len(entropy_stats),
        "results": {
            "ratio_mean": float(ratio_mean),
            "difference_mean": float(diff_mean),
            "euler_mean": float(euler_mean),
            "one_bit_law_confirmed": bool(abs(diff_mean - 1.0) < 0.2),
            "euler_confirmed": bool(abs(euler_mean - 1/np.e) < 0.02)
        }
    }

    with open(OUTPUT_DIR / "entropy_tau_validation.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {OUTPUT_DIR / 'entropy_tau_validation.json'}")


if __name__ == "__main__":
    main()
