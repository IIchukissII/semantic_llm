#!/usr/bin/env python3
"""
Test SemanticNavigator
======================

Clean test using the hierarchy's built-in navigation.
No hardcoded stopwords or similarity functions.
"""

import sys
import math
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_EXPERIMENTS = _MEANING_CHAIN.parent
_SEMANTIC_LLM = _EXPERIMENTS.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.unified_hierarchy import build_hierarchy, SemanticNavigator


def show_navigation(nav: SemanticNavigator, query: str):
    """Display navigation results."""
    result = nav.navigate(query)

    print("=" * 70)
    print(f"QUERY: \"{result['query']}\"")
    print("=" * 70)

    print(f"\nContent: nouns={result['nouns']}, verbs={result['verbs']}")

    # Seeds
    if result['seeds']:
        print(f"\n{'─'*70}")
        print("SEED COORDINATES")
        print(f"{'─'*70}")
        print(f"{'Word':<12} │ {'n':>5} │ DERIVED (usage) │ RAW (semantic)")
        print(f"{'':12} │ {'':>5} │ {'θ°':>7} {'r':>6} │ {'θ°':>7} {'r':>6}")
        print("-" * 60)
        for word, coords in result['seeds'].items():
            print(f"{word:<12} │ {coords['n']:>5.2f} │ "
                  f"{coords['theta_derived_deg']:>7.1f} {coords['r_derived']:>6.3f} │ "
                  f"{coords['theta_raw_deg']:>7.1f} {coords['r_raw']:>6.3f}")

    # Neighbors
    if result['neighbors']:
        print(f"\n{'─'*70}")
        print("NEIGHBORS (orbital-constrained)")
        print(f"{'─'*70}")

        for seed, systems in result['neighbors'].items():
            print(f"\n▶ From '{seed}':")

            print(f"\n  SEMANTIC (similar meaning, RAW θ):")
            for word, sim, n in systems['semantic'][:6]:
                print(f"    {word:<15} sim={sim:.3f}  n={n:.2f}")

            print(f"\n  USAGE (similar context, DERIVED θ):")
            for word, sim, n in systems['usage'][:6]:
                print(f"    {word:<15} sim={sim:.3f}  n={n:.2f}")

    # Transformations
    if result['transformations']:
        print(f"\n{'─'*70}")
        print("VERB TRANSFORMATIONS")
        print(f"{'─'*70}")
        for key, data in result['transformations'].items():
            print(f"\n  {key}:")
            print(f"    Before: θ = {data['before']['theta_deg']:>7.1f}°, r = {data['before']['r']:.3f}")
            print(f"    After:  θ = {data['after']['theta_deg']:>7.1f}°, r = {data['after']['r']:.3f}")
            print(f"    Shift:  Δθ = {data['shift']['delta_theta_deg']:>+7.1f}°, Δr = {data['shift']['delta_r']:+.3f}")

    # Path coherence
    if result['path_coherence']:
        print(f"\n{'─'*70}")
        print("PATH COHERENCE")
        print(f"{'─'*70}")
        pc = result['path_coherence']
        print(f"  DERIVED: {pc['derived']['average']:+.3f}")
        print(f"  RAW:     {pc['raw']['average']:+.3f}")


def compare_opposites(nav: SemanticNavigator):
    """Compare DERIVED vs RAW for semantic opposites."""
    print("\n" + "=" * 70)
    print("DERIVED vs RAW: Semantic Opposites")
    print("=" * 70)

    pairs = [
        ('love', 'hate'),
        ('good', 'evil'),
        ('peace', 'war'),
        ('life', 'death'),
        ('truth', 'lie'),
    ]

    print(f"\n{'Pair':<15} │ DERIVED sim │ RAW sim │ Interpretation")
    print("-" * 70)

    for w1, w2 in pairs:
        qw1 = nav.hierarchy.get_word(w1)
        qw2 = nav.hierarchy.get_word(w2)

        if qw1 and qw2:
            sim_der = nav.similarity(qw1, qw2, use_raw=False, include_orbital=False)
            sim_raw = nav.similarity(qw1, qw2, use_raw=True, include_orbital=False)

            if sim_der > 0.3 and sim_raw < 0:
                interp = "Similar usage, opposite meaning ✓"
            elif sim_der > 0.3 and sim_raw > 0.3:
                interp = "Similar in both"
            elif sim_der < 0 and sim_raw < 0:
                interp = "Different in both"
            else:
                interp = "Mixed"

            print(f"{w1}/{w2:<10} │ {sim_der:>+10.3f} │ {sim_raw:>+6.3f} │ {interp}")
        else:
            missing = w1 if not qw1 else w2
            print(f"{w1}/{w2:<10} │ {'--':>10} │ {'--':>6} │ '{missing}' not found")


def main():
    print("Loading Unified Hierarchy...")
    loader = DataLoader()
    hierarchy = build_hierarchy(loader)
    nav = SemanticNavigator(hierarchy)

    # Compare opposites (key validation)
    compare_opposites(nav)

    # Test queries
    queries = [
        "What is love?",
        "How to find peace?",
        "The meaning of life",
        "Truth and beauty",
        "Create something new",
    ]

    for query in queries:
        print("\n")
        show_navigation(nav, query)

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter a query (or 'quit' to exit):")

    while True:
        try:
            query = input("\n> ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if query:
                show_navigation(nav, query)
        except (EOFError, KeyboardInterrupt):
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
