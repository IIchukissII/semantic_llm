#!/usr/bin/env python3
"""
Test Supercritical Mode: Chain Reaction Comparison
===================================================

Compares the Navigator in:
- STABLE mode (α=0.3): Controlled, subcritical
- SUPERCRITICAL mode (α=0.2): Chain reaction, λ > 1

Tests chain reaction dynamics by running multiple exchanges
and measuring power amplification.

Usage:
    python test_supercritical.py
"""

import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json

# Setup paths
_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from chain_core.navigator import SemanticNavigator, NavigationGoal


def test_chain_reaction(nav: SemanticNavigator, query: str,
                        goal: str, n_exchanges: int = 4):
    """
    Simulate chain reaction over multiple exchanges.

    Returns power at each exchange and chain coefficients.
    """
    print(f"\n{'='*60}")
    print(f"Goal: {goal.upper()} | Query: {query}")
    print("="*60)

    powers = []
    coherences = []

    for i in range(n_exchanges):
        result = nav.navigate(query, goal=goal)

        power = result.quality.power
        coherence = result.quality.coherence

        powers.append(power)
        coherences.append(coherence)

        print(f"\n  Exchange {i+1}:")
        print(f"    Power:     {power:.2f}")
        print(f"    Coherence: {coherence:.2f}")
        print(f"    Concepts:  {result.concepts[:5]}")

        if result.thesis and result.antithesis:
            print(f"    Paradox:   {result.thesis} ↔ {result.antithesis}")

    # Compute chain coefficients
    lambdas = []
    for i in range(1, len(powers)):
        if powers[i-1] > 0:
            lam = powers[i] / powers[i-1]
            lambdas.append(lam)

    avg_lambda = np.mean(lambdas) if lambdas else 0

    # Determine phase
    if avg_lambda > 1:
        phase = "SUPERCRITICAL (λ > 1)"
    elif avg_lambda > 0.9:
        phase = "CRITICAL (λ ≈ 1)"
    else:
        phase = "SUBCRITICAL (λ < 1)"

    print(f"\n  Chain Analysis:")
    print(f"    λ per exchange: {[f'{l:.2f}' for l in lambdas]}")
    print(f"    Average λ:      {avg_lambda:.3f}")
    print(f"    Phase:          {phase}")
    print(f"    Power trend:    {powers[0]:.1f} → {powers[-1]:.1f} "
          f"({(powers[-1]/powers[0]-1)*100:+.0f}%)" if powers[0] > 0 else "")

    return {
        'goal': goal,
        'powers': powers,
        'coherences': coherences,
        'lambdas': lambdas,
        'avg_lambda': avg_lambda,
        'phase': phase,
        'amplification': powers[-1] / powers[0] if powers[0] > 0 else 0
    }


def compare_modes():
    """Compare stable vs supercritical modes."""
    print("\n" + "="*70)
    print("SUPERCRITICAL MODE TEST: Chain Reaction Comparison")
    print("="*70)
    print("\nComparing:")
    print("  • STABLE mode (α=0.3): Controlled navigation")
    print("  • SUPERCRITICAL mode (α=0.2): Chain reaction zone")
    print()

    nav = SemanticNavigator()

    test_queries = [
        "What is the meaning of love?",
        "What is consciousness?",
        "What is the relationship between life and death?",
    ]

    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }

    try:
        for query in test_queries:
            print(f"\n{'#'*70}")
            print(f"# QUERY: {query}")
            print("#"*70)

            # Test STABLE mode (subcritical)
            stable_result = test_chain_reaction(nav, query, "stable", n_exchanges=4)

            # Test SUPERCRITICAL mode
            super_result = test_chain_reaction(nav, query, "supercritical", n_exchanges=4)

            # Compare
            print(f"\n{'─'*60}")
            print("COMPARISON:")
            print("─"*60)
            print(f"  Mode           | Avg λ    | Phase           | Amplification")
            print(f"  ─────────────────────────────────────────────────────────")
            print(f"  STABLE (α=0.3) | {stable_result['avg_lambda']:.3f}   | "
                  f"{stable_result['phase']:15} | {stable_result['amplification']:.1f}x")
            print(f"  SUPER  (α=0.2) | {super_result['avg_lambda']:.3f}   | "
                  f"{super_result['phase']:15} | {super_result['amplification']:.1f}x")

            # Determine winner
            if super_result['avg_lambda'] > stable_result['avg_lambda']:
                print(f"\n  → SUPERCRITICAL shows {(super_result['avg_lambda']/stable_result['avg_lambda']-1)*100:.0f}% higher λ")

            results['tests'].append({
                'query': query,
                'stable': stable_result,
                'supercritical': super_result
            })

        # Summary
        print("\n" + "="*70)
        print("SUMMARY: SUPERCRITICAL MODE")
        print("="*70)

        stable_lambdas = [t['stable']['avg_lambda'] for t in results['tests']]
        super_lambdas = [t['supercritical']['avg_lambda'] for t in results['tests']]

        print(f"\nAverage chain coefficient across all tests:")
        print(f"  STABLE (α=0.3):       λ = {np.mean(stable_lambdas):.3f}")
        print(f"  SUPERCRITICAL (α=0.2): λ = {np.mean(super_lambdas):.3f}")

        if np.mean(super_lambdas) > np.mean(stable_lambdas):
            improvement = (np.mean(super_lambdas) / np.mean(stable_lambdas) - 1) * 100
            print(f"\n  ✓ SUPERCRITICAL mode shows {improvement:.0f}% higher chain coefficient!")

        super_supercrit = sum(1 for l in super_lambdas if l > 1)
        stable_supercrit = sum(1 for l in stable_lambdas if l > 1)

        print(f"\nSupercritical phases achieved:")
        print(f"  STABLE:       {stable_supercrit}/{len(stable_lambdas)}")
        print(f"  SUPERCRITICAL: {super_supercrit}/{len(super_lambdas)}")

        # Save results
        results_file = _PHYSICS_DIR / "results" / f"supercritical_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

    finally:
        nav.close()

    return results


def test_single():
    """Quick single test for verification."""
    print("\n" + "="*70)
    print("QUICK TEST: Supercritical vs Stable")
    print("="*70)

    nav = SemanticNavigator()

    query = "What is love?"

    try:
        print(f"\nQuery: {query}")
        print("-" * 50)

        # Stable
        print("\n[STABLE MODE - α=0.3]")
        result_stable = nav.navigate(query, goal="stable")
        print(f"  Strategy:  {result_stable.strategy}")
        print(f"  Power:     {result_stable.quality.power:.2f}")
        print(f"  Coherence: {result_stable.quality.coherence:.2f}")
        print(f"  Concepts:  {result_stable.concepts[:5]}")

        # Supercritical
        print("\n[SUPERCRITICAL MODE - α=0.2]")
        result_super = nav.navigate(query, goal="supercritical")
        print(f"  Strategy:  {result_super.strategy}")
        print(f"  Power:     {result_super.quality.power:.2f}")
        print(f"  Coherence: {result_super.quality.coherence:.2f}")
        print(f"  Concepts:  {result_super.concepts[:5]}")

        if result_super.thesis:
            print(f"  Paradox:   {result_super.thesis} ↔ {result_super.antithesis}")

        # Compare
        print("\n[COMPARISON]")
        power_diff = result_super.quality.power - result_stable.quality.power
        print(f"  Power difference: {power_diff:+.2f}")
        print(f"  Supercritical {'wins' if power_diff > 0 else 'loses'} on power!")

    finally:
        nav.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick single test')
    args = parser.parse_args()

    if args.quick:
        test_single()
    else:
        compare_modes()
