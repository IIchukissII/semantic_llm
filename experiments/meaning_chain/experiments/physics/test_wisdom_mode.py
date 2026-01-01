#!/usr/bin/env python3
"""
WISDOM Mode Test Suite
======================

Compares WISDOM vs POWERFUL navigation modes across multiple questions.

WISDOM mode targets the theoretical optimum:
    C_opt = 0.615, P_opt = 6.15
    Meaning_max = C × P = 3.78

POWERFUL mode maximizes raw paradox power.

Results saved to: results/wisdom_test_YYYYMMDD_HHMMSS.json

Usage:
    python test_wisdom_mode.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent

sys.path.insert(0, str(_MEANING_CHAIN))

from chain_core.navigator import SemanticNavigator, NavigationGoal
import numpy as np


def compute_synthesis_coherence(graph, concepts: list, poles: set) -> float:
    """Compute average pairwise coherence among non-pole concepts (consistent metric)."""
    synthesis = [c for c in concepts if c not in poles][:10]

    if len(synthesis) < 2:
        return 0.0

    j_vectors = []
    for word in synthesis:
        concept = graph.get_concept(word)
        if concept and concept.get('j'):
            j = np.array(concept['j'])
            if len(j) == 5:
                j_vectors.append(j)

    if len(j_vectors) < 2:
        return 0.0

    sims = []
    for i in range(len(j_vectors)):
        for k in range(i + 1, len(j_vectors)):
            v1, v2 = j_vectors[i], j_vectors[k]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                sims.append(float(np.dot(v1, v2) / (n1 * n2)))

    return abs(float(np.mean(sims))) if sims else 0.0


def test_wisdom_vs_powerful():
    """Compare WISDOM and POWERFUL modes on various questions."""

    print("=" * 70)
    print("WISDOM vs POWERFUL Mode Comparison")
    print("=" * 70)
    print("\nTheoretical Optimum:")
    print("  C_opt = 0.615, P_opt = 6.15")
    print("  Meaning_max = 3.78")
    print("  Σ = C + 0.1P = 1.22 (semantic budget)")
    print()

    questions = [
        # Philosophical
        "What is wisdom?",
        "What is love?",
        "What is the meaning of life?",
        "What is consciousness?",
        "What is truth?",
        "What is beauty?",
        "What is freedom?",
        "What is death?",

        # Existential
        "What is happiness?",
        "What is suffering?",
        "What is faith?",
        "What is hope?",

        # Abstract
        "What is time?",
        "What is reality?",
        "What is power?",
        "What is knowledge?",

        # Relational
        "What is friendship?",
        "What is justice?",
        "What is art?",
        "What is creativity?",
    ]

    nav = SemanticNavigator()
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'title': 'WISDOM vs POWERFUL Mode Comparison',
        'optimal': {
            'C': 0.615,
            'P': 6.15,
            'meaning': 3.78,
            'sigma': 1.22
        },
        'comparisons': [],
        'summary': {}
    }

    wisdom_total_meaning = 0
    powerful_total_meaning = 0
    wisdom_total_eff = 0
    powerful_total_eff = 0
    wisdom_closer_to_optimal = 0

    try:
        for q in questions:
            print(f"\n{'─' * 60}")
            print(f"  {q}")
            print(f"{'─' * 60}")

            # Get graph for consistent coherence calculation
            graph = nav._init_graph()

            # POWERFUL mode
            powerful = nav.navigate(q, 'powerful')
            p_poles = {powerful.thesis, powerful.antithesis} if powerful.thesis else set()
            p_C = compute_synthesis_coherence(graph, powerful.concepts, p_poles)
            p_P = powerful.quality.power
            p_meaning = p_C * p_P
            p_sigma = p_C + 0.1 * p_P
            p_eff = p_meaning / 3.78 if p_meaning > 0 else 0

            # WISDOM mode
            wisdom = nav.navigate(q, 'wisdom')
            w_poles = {wisdom.thesis, wisdom.antithesis} if wisdom.thesis else set()
            w_C = compute_synthesis_coherence(graph, wisdom.concepts, w_poles)
            w_P = wisdom.quality.power
            w_meaning = w_C * w_P
            w_sigma = w_C + 0.1 * w_P
            w_eff = w_meaning / 3.78 if w_meaning > 0 else 0

            # Distance from optimal C=0.1P ratio
            p_ratio = p_C / (0.1 * p_P) if p_P > 0 else 0
            w_ratio = w_C / (0.1 * w_P) if w_P > 0 else 0
            p_balance = abs(1 - p_ratio)
            w_balance = abs(1 - w_ratio)

            # Which is closer to optimal?
            closer = "WISDOM" if w_balance < p_balance else "POWERFUL"
            if w_balance < p_balance:
                wisdom_closer_to_optimal += 1

            print(f"\n  POWERFUL:")
            print(f"    C={p_C:.3f}, P={p_P:.2f}")
            print(f"    Meaning={p_meaning:.3f}, Σ={p_sigma:.2f}")
            print(f"    Efficiency={p_eff:.1%}, Balance={p_ratio:.2f}")
            print(f"    Paradox: {powerful.thesis} ↔ {powerful.antithesis}")

            print(f"\n  WISDOM:")
            print(f"    C={w_C:.3f}, P={w_P:.2f}")
            print(f"    Meaning={w_meaning:.3f}, Σ={w_sigma:.2f}")
            print(f"    Efficiency={w_eff:.1%}, Balance={w_ratio:.2f}")
            print(f"    Paradox: {wisdom.thesis} ↔ {wisdom.antithesis}")
            print(f"    Synthesis: {wisdom.synthesis[:3]}")

            print(f"\n  → Closer to optimal balance: {closer}")

            # Accumulate
            wisdom_total_meaning += w_meaning
            powerful_total_meaning += p_meaning
            wisdom_total_eff += w_eff
            powerful_total_eff += p_eff

            # Store result
            results['comparisons'].append({
                'question': q,
                'powerful': {
                    'C': p_C,
                    'P': p_P,
                    'meaning': p_meaning,
                    'sigma': p_sigma,
                    'efficiency': p_eff,
                    'balance_ratio': p_ratio,
                    'thesis': powerful.thesis,
                    'antithesis': powerful.antithesis,
                    'concepts': powerful.concepts[:5]
                },
                'wisdom': {
                    'C': w_C,
                    'P': w_P,
                    'meaning': w_meaning,
                    'sigma': w_sigma,
                    'efficiency': w_eff,
                    'balance_ratio': w_ratio,
                    'thesis': wisdom.thesis,
                    'antithesis': wisdom.antithesis,
                    'synthesis': wisdom.synthesis[:5],
                    'concepts': wisdom.concepts[:5]
                },
                'closer_to_optimal': closer
            })

        # Summary
        n = len(questions)
        results['summary'] = {
            'n_questions': n,
            'wisdom_avg_meaning': wisdom_total_meaning / n,
            'powerful_avg_meaning': powerful_total_meaning / n,
            'wisdom_avg_efficiency': wisdom_total_eff / n,
            'powerful_avg_efficiency': powerful_total_eff / n,
            'wisdom_closer_to_optimal': wisdom_closer_to_optimal,
            'powerful_closer_to_optimal': n - wisdom_closer_to_optimal,
            'wisdom_balance_win_rate': wisdom_closer_to_optimal / n
        }

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nQuestions tested: {n}")
        print(f"\n  POWERFUL avg meaning: {powerful_total_meaning/n:.3f}")
        print(f"  WISDOM avg meaning:   {wisdom_total_meaning/n:.3f}")
        print(f"\n  POWERFUL avg efficiency: {powerful_total_eff/n:.1%}")
        print(f"  WISDOM avg efficiency:   {wisdom_total_eff/n:.1%}")
        print(f"\n  Closer to C=0.1P balance:")
        print(f"    WISDOM:   {wisdom_closer_to_optimal}/{n} ({100*wisdom_closer_to_optimal/n:.0f}%)")
        print(f"    POWERFUL: {n-wisdom_closer_to_optimal}/{n} ({100*(n-wisdom_closer_to_optimal)/n:.0f}%)")

        # Save results
        results_dir = _PHYSICS_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / f"wisdom_test_{results['timestamp']}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

        return results

    finally:
        nav.close()


if __name__ == "__main__":
    test_wisdom_vs_powerful()
