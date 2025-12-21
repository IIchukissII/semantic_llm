#!/usr/bin/env python3
"""
Noun-Adj-Verb Correlation Analysis
===================================

User insight: "observe if noun-verb correlate with noun-adj"

The ground is NOUNS:
- Adjectives describe nouns (noun-adj bonds)
- Verbs act on nouns (noun-verb bonds)
- If these correlate, we understand the adjective-verb connection

Questions:
1. Do nouns with rich adj profiles have rich verb profiles?
2. Do the TYPES of adjectives correlate with TYPES of verbs?
3. Is there a shared semantic structure?
"""

import json
import numpy as np
import psycopg2
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}

OUTPUT_DIR = Path(__file__).parent

# j-space dimensions
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']

# Polarity adjectives
ADJ_POSITIVE = {
    'beauty': {'beautiful', 'pretty', 'lovely', 'gorgeous', 'elegant'},
    'life': {'alive', 'living', 'vital', 'vibrant', 'healthy'},
    'sacred': {'sacred', 'holy', 'divine', 'spiritual', 'blessed'},
    'good': {'good', 'great', 'noble', 'kind', 'worthy'},
    'love': {'loving', 'warm', 'tender', 'caring', 'gentle'},
}

ADJ_NEGATIVE = {
    'beauty': {'ugly', 'hideous', 'grotesque', 'awful'},
    'life': {'dead', 'dying', 'lifeless', 'sick'},
    'sacred': {'profane', 'unholy', 'sinful', 'cursed'},
    'good': {'bad', 'evil', 'cruel', 'wicked'},
    'love': {'hateful', 'cold', 'hostile', 'bitter'},
}


def load_noun_adj_profile() -> Dict[str, Dict[str, int]]:
    """Load noun -> adjective counts."""
    print("Loading noun-adj bonds...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute('''
        SELECT bond, total_count
        FROM hyp_bond_vocab
        WHERE total_count >= 2
    ''')

    noun_adj = defaultdict(lambda: defaultdict(int))
    for bond, count in cur.fetchall():
        parts = bond.split('|')
        if len(parts) == 2:
            adj, noun = parts
            noun_adj[noun][adj.lower()] += count

    conn.close()
    print(f"  Loaded adj profile for {len(noun_adj)} nouns")
    return noun_adj


def load_noun_verb_profile() -> Dict[str, Dict[str, int]]:
    """Load noun -> verb counts (nouns as objects of verbs)."""
    print("Loading noun-verb bonds...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute('''
        SELECT verb, object, SUM(total_count) as count
        FROM hyp_svo_triads
        WHERE total_count >= 1
        GROUP BY verb, object
    ''')

    noun_verb = defaultdict(lambda: defaultdict(int))
    for verb, noun, count in cur.fetchall():
        noun_verb[noun][verb] += count

    conn.close()
    print(f"  Loaded verb profile for {len(noun_verb)} nouns")
    return noun_verb


def compute_noun_adj_polarity(adj_counts: Dict[str, int]) -> np.ndarray:
    """Compute noun's adjective polarity on j-space dimensions."""
    polarity = np.zeros(len(J_DIMS))
    total = sum(adj_counts.values())

    if total == 0:
        return polarity

    for i, dim in enumerate(J_DIMS):
        pos = sum(adj_counts.get(adj, 0) for adj in ADJ_POSITIVE[dim])
        neg = sum(adj_counts.get(adj, 0) for adj in ADJ_NEGATIVE[dim])
        polarity[i] = (pos - neg) / total

    return polarity


def compute_variety(profile: Dict[str, int]) -> int:
    """Compute variety (number of unique items)."""
    return len(profile)


def compute_variety_correlation(
    noun_adj: Dict[str, Dict[str, int]],
    noun_verb: Dict[str, Dict[str, int]]
) -> Tuple[float, float, int]:
    """
    Compute correlation between adjective variety and verb variety.

    Q: Do nouns with many adjectives also have many verbs?
    """
    print("\nComputing adj-verb variety correlation...")

    common_nouns = set(noun_adj.keys()) & set(noun_verb.keys())
    print(f"  Common nouns: {len(common_nouns)}")

    if len(common_nouns) < 100:
        return 0, 0, 0

    adj_varieties = []
    verb_varieties = []

    for noun in common_nouns:
        adj_var = compute_variety(noun_adj[noun])
        verb_var = compute_variety(noun_verb[noun])
        adj_varieties.append(adj_var)
        verb_varieties.append(verb_var)

    pearson_r, pearson_p = pearsonr(adj_varieties, verb_varieties)
    spearman_r, spearman_p = spearmanr(adj_varieties, verb_varieties)

    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")

    return pearson_r, spearman_r, len(common_nouns)


def compute_polarity_by_verb(
    noun_adj: Dict[str, Dict[str, int]],
    noun_verb: Dict[str, Dict[str, int]],
    top_verbs: int = 50
) -> Dict[str, np.ndarray]:
    """
    For each verb, compute the average adjective polarity of its objects.

    This gives verb -> adjective space mapping.
    """
    print(f"\nComputing verb polarity from object adjectives (top {top_verbs})...")

    # Count verb frequencies
    verb_counts = defaultdict(int)
    for noun, verbs in noun_verb.items():
        for verb, count in verbs.items():
            verb_counts[verb] += count

    # Get top verbs
    top = sorted(verb_counts.keys(), key=lambda v: -verb_counts[v])[:top_verbs]

    # For each verb, aggregate polarity of its object nouns
    verb_polarity = {}

    for verb in top:
        polarities = []
        weights = []

        for noun, verbs in noun_verb.items():
            if verb in verbs and noun in noun_adj:
                pol = compute_noun_adj_polarity(noun_adj[noun])
                weight = verbs[verb]
                polarities.append(pol)
                weights.append(weight)

        if len(polarities) >= 10:
            polarities = np.array(polarities)
            weights = np.array(weights)
            weighted_pol = np.average(polarities, weights=weights, axis=0)
            verb_polarity[verb] = weighted_pol

    print(f"  Computed polarity for {len(verb_polarity)} verbs")
    return verb_polarity


def validate_verb_opposites(verb_polarity: Dict[str, np.ndarray]):
    """Validate verb opposites using adjective-derived polarity."""
    print("\n" + "=" * 60)
    print("VALIDATION: Verb Opposites (from noun-adj correlation)")
    print("=" * 60)

    known_pairs = [
        ("love", "hate"),
        ("help", "harm"),
        ("create", "destroy"),
        ("build", "break"),
        ("save", "kill"),
        ("give", "take"),
    ]

    print(f"\n{'Pair':<20} {'Pol Cos':>10} {'Status':>10}")
    print("-" * 45)

    correct = 0
    total = 0

    for v1, v2 in known_pairs:
        if v1 in verb_polarity and v2 in verb_polarity:
            p1 = verb_polarity[v1]
            p2 = verb_polarity[v2]
            norm1, norm2 = np.linalg.norm(p1), np.linalg.norm(p2)

            if norm1 > 1e-8 and norm2 > 1e-8:
                cos = np.dot(p1, p2) / (norm1 * norm2)
                status = "ok" if cos < 0 else "WRONG"
                if cos < 0:
                    correct += 1
                total += 1

                print(f"{v1}/{v2:<13} {cos:>10.3f} {status:>10}")
            else:
                print(f"{v1}/{v2:<13} {'zero':>10}")
        else:
            missing = [v for v in [v1, v2] if v not in verb_polarity]
            print(f"{v1}/{v2:<13} {'N/A':>10} missing: {missing}")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.0f}%")


def show_verb_polarity_extremes(verb_polarity: Dict[str, np.ndarray]):
    """Show verbs at polarity extremes."""
    print("\n" + "=" * 60)
    print("VERB POLARITY EXTREMES (from noun-adj correlation)")
    print("=" * 60)

    # By total polarity
    sorted_by_total = sorted(
        verb_polarity.items(),
        key=lambda x: np.sum(x[1])
    )

    print("\nMost negative (dark verbs):")
    for verb, pol in sorted_by_total[:10]:
        print(f"  {verb:<15} sum={np.sum(pol):>8.5f}  pol={pol.round(5)}")

    print("\nMost positive (bright verbs):")
    for verb, pol in sorted_by_total[-10:][::-1]:
        print(f"  {verb:<15} sum={np.sum(pol):>8.5f}  pol={pol.round(5)}")

    # By each dimension
    print("\n" + "-" * 60)
    for i, dim in enumerate(J_DIMS):
        sorted_by_dim = sorted(
            verb_polarity.items(),
            key=lambda x: x[1][i]
        )
        neg = sorted_by_dim[:3]
        pos = sorted_by_dim[-3:][::-1]

        print(f"\n{dim.upper()}:")
        print(f"  Low:  {[(v, round(p[i]*1e4, 2)) for v, p in neg]}")
        print(f"  High: {[(v, round(p[i]*1e4, 2)) for v, p in pos]}")


def main():
    print("=" * 60)
    print("NOUN-ADJ-VERB CORRELATION ANALYSIS")
    print("=" * 60)
    print("""
Ground: NOUNS connect both adj and verb spaces
  - Adjectives describe nouns (noun-adj bonds)
  - Verbs act on nouns (noun-verb bonds)
  - Correlation reveals shared semantic structure
""")

    # Load data
    noun_adj = load_noun_adj_profile()
    noun_verb = load_noun_verb_profile()

    # Q1: Do nouns with many adjectives also have many verbs?
    pearson_r, spearman_r, n = compute_variety_correlation(noun_adj, noun_verb)

    print(f"\n" + "=" * 60)
    print(f"RESULT: Noun variety correlation")
    print(f"  Pearson r = {pearson_r:.4f}")
    print(f"  Spearman r = {spearman_r:.4f}")
    print(f"  n = {n} common nouns")

    if pearson_r > 0.5:
        print("  STRONG correlation: adj-rich nouns are also verb-rich")
    elif pearson_r > 0.3:
        print("  MODERATE correlation")
    else:
        print("  WEAK correlation")

    # Q2: Can we derive verb polarity from noun-adj?
    verb_polarity = compute_polarity_by_verb(noun_adj, noun_verb, top_verbs=200)

    # Validate
    validate_verb_opposites(verb_polarity)

    # Show extremes
    show_verb_polarity_extremes(verb_polarity)

    # Export
    output = {
        "generated_at": datetime.now().isoformat(),
        "method": "verb polarity from noun-adj correlation",
        "variety_correlation": {
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "n_common_nouns": n
        },
        "verb_polarity": {
            verb: pol.tolist() for verb, pol in verb_polarity.items()
        }
    }

    with open(OUTPUT_DIR / "noun_adj_verb_correlation.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to noun_adj_verb_correlation.json")


if __name__ == "__main__":
    main()
