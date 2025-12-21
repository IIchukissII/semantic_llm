#!/usr/bin/env python3
"""
Shannon Entropy Correlation: Noun-Adj-Verb
===========================================

User insight: "we should look on entropy. it must not be linear.
              perhaps somelike shannon."

Previous: variety = count (linear)
Better: Shannon entropy H = -Σ p·log₂(p)

Shannon entropy captures DISTRIBUTION:
- 10 adjectives, equal distribution → high H
- 10 adjectives, one dominant → low H

Connection to Boltzmann: S = -k Σ p ln p
"""

import json
import numpy as np
import psycopg2
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple
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


def shannon_entropy(counts: Dict[str, int]) -> float:
    """
    Compute Shannon entropy: H = -Σ p·log₂(p)

    Returns entropy in bits.
    """
    if not counts:
        return 0.0

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy


def normalized_entropy(counts: Dict[str, int]) -> float:
    """
    Normalized entropy: H / H_max = H / log₂(n)

    Returns value in [0, 1].
    """
    if not counts or len(counts) <= 1:
        return 0.0

    h = shannon_entropy(counts)
    h_max = np.log2(len(counts))

    return h / h_max if h_max > 0 else 0.0


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
    """Load noun -> verb counts."""
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


def compute_entropy_correlation(
    noun_adj: Dict[str, Dict[str, int]],
    noun_verb: Dict[str, Dict[str, int]]
) -> Tuple[float, float, int]:
    """Compute correlation of Shannon entropies."""
    print("\nComputing Shannon entropy correlation...")

    common_nouns = set(noun_adj.keys()) & set(noun_verb.keys())
    print(f"  Common nouns: {len(common_nouns)}")

    if len(common_nouns) < 100:
        return 0, 0, 0

    adj_entropy = []
    verb_entropy = []
    adj_norm_entropy = []
    verb_norm_entropy = []

    for noun in common_nouns:
        h_adj = shannon_entropy(noun_adj[noun])
        h_verb = shannon_entropy(noun_verb[noun])
        hn_adj = normalized_entropy(noun_adj[noun])
        hn_verb = normalized_entropy(noun_verb[noun])

        adj_entropy.append(h_adj)
        verb_entropy.append(h_verb)
        adj_norm_entropy.append(hn_adj)
        verb_norm_entropy.append(hn_verb)

    # Raw entropy correlation
    pearson_raw, p_raw = pearsonr(adj_entropy, verb_entropy)
    spearman_raw, _ = spearmanr(adj_entropy, verb_entropy)

    # Normalized entropy correlation
    pearson_norm, p_norm = pearsonr(adj_norm_entropy, verb_norm_entropy)
    spearman_norm, _ = spearmanr(adj_norm_entropy, verb_norm_entropy)

    print(f"\n  Raw Shannon entropy:")
    print(f"    Pearson r: {pearson_raw:.4f} (p={p_raw:.2e})")
    print(f"    Spearman r: {spearman_raw:.4f}")

    print(f"\n  Normalized entropy (H/H_max):")
    print(f"    Pearson r: {pearson_norm:.4f} (p={p_norm:.2e})")
    print(f"    Spearman r: {spearman_norm:.4f}")

    return pearson_raw, pearson_norm, len(common_nouns)


def analyze_entropy_distribution(
    noun_adj: Dict[str, Dict[str, int]],
    noun_verb: Dict[str, Dict[str, int]]
):
    """Analyze entropy distributions."""
    print("\n" + "=" * 60)
    print("ENTROPY DISTRIBUTION ANALYSIS")
    print("=" * 60)

    common_nouns = list(set(noun_adj.keys()) & set(noun_verb.keys()))

    # Compute entropies
    adj_h = [(noun, shannon_entropy(noun_adj[noun])) for noun in common_nouns]
    verb_h = [(noun, shannon_entropy(noun_verb[noun])) for noun in common_nouns]

    adj_h.sort(key=lambda x: -x[1])
    verb_h.sort(key=lambda x: -x[1])

    print("\nTop 10 nouns by ADJ entropy (most diverse adjective usage):")
    for noun, h in adj_h[:10]:
        n_adj = len(noun_adj[noun])
        print(f"  {noun:<20} H={h:.3f} bits  n={n_adj}")

    print("\nTop 10 nouns by VERB entropy (most diverse verb usage):")
    for noun, h in verb_h[:10]:
        n_verb = len(noun_verb[noun])
        print(f"  {noun:<20} H={h:.3f} bits  n={n_verb}")

    # Compare variety vs entropy for same nouns
    print("\n" + "-" * 60)
    print("VARIETY vs ENTROPY comparison (showing why entropy matters):")

    # Find nouns with high variety but low entropy (dominated by few items)
    variety_entropy = []
    for noun in common_nouns[:5000]:  # Sample
        adj_counts = noun_adj[noun]
        n = len(adj_counts)
        h = shannon_entropy(adj_counts)
        h_max = np.log2(n) if n > 1 else 1
        uniformity = h / h_max if h_max > 0 else 0
        variety_entropy.append((noun, n, h, uniformity))

    # High variety, low uniformity = few dominant adjectives
    variety_entropy.sort(key=lambda x: (x[1] > 20, -x[3]))  # variety > 20, low uniformity

    print("\nNouns with HIGH variety but LOW uniformity (concentrated distribution):")
    for noun, n, h, u in variety_entropy[:10]:
        top_adj = sorted(noun_adj[noun].items(), key=lambda x: -x[1])[:3]
        print(f"  {noun:<15} n={n:>3} H={h:.2f} u={u:.2f}  top: {top_adj}")


def compute_verb_entropy_polarity(
    noun_adj: Dict[str, Dict[str, int]],
    noun_verb: Dict[str, Dict[str, int]],
    top_verbs: int = 100
) -> Dict[str, Tuple[float, float]]:
    """
    For each verb, compute the entropy distribution of its objects.

    Returns (mean_adj_entropy, std_adj_entropy) for each verb.
    """
    print(f"\nComputing verb entropy signatures (top {top_verbs})...")

    # Invert: verb -> list of noun entropies
    verb_obj_entropies = defaultdict(list)

    for noun, verbs in noun_verb.items():
        if noun in noun_adj:
            h_adj = shannon_entropy(noun_adj[noun])
            for verb, count in verbs.items():
                for _ in range(min(count, 10)):  # Weight by count, capped
                    verb_obj_entropies[verb].append(h_adj)

    # Compute stats per verb
    verb_stats = {}
    for verb, entropies in verb_obj_entropies.items():
        if len(entropies) >= 20:
            verb_stats[verb] = (np.mean(entropies), np.std(entropies))

    # Sort by mean entropy
    sorted_verbs = sorted(verb_stats.items(), key=lambda x: x[1][0])

    print("\nVerbs by mean object entropy:")
    print("\nLOW entropy objects (concrete, focused):")
    for verb, (mean, std) in sorted_verbs[:10]:
        print(f"  {verb:<15} mean_H={mean:.3f} std={std:.3f}")

    print("\nHIGH entropy objects (abstract, diverse):")
    for verb, (mean, std) in sorted_verbs[-10:][::-1]:
        print(f"  {verb:<15} mean_H={mean:.3f} std={std:.3f}")

    return verb_stats


def main():
    print("=" * 60)
    print("SHANNON ENTROPY CORRELATION ANALYSIS")
    print("=" * 60)
    print("""
Shannon entropy: H = -Σ p·log₂(p)

This captures DISTRIBUTION, not just count:
- Equal distribution → high entropy
- Concentrated distribution → low entropy

Connection to Boltzmann: S = -k Σ p ln p
""")

    # Load data
    noun_adj = load_noun_adj_profile()
    noun_verb = load_noun_verb_profile()

    # Entropy correlation
    r_raw, r_norm, n = compute_entropy_correlation(noun_adj, noun_verb)

    print(f"\n" + "=" * 60)
    print(f"RESULT: Shannon Entropy Correlation")
    print(f"  Raw entropy Pearson r = {r_raw:.4f}")
    print(f"  Normalized entropy Pearson r = {r_norm:.4f}")
    print(f"  n = {n} common nouns")

    if r_raw > 0.5:
        print("\n  STRONG correlation: adj-diverse nouns are also verb-diverse")
        print("  The semantic structure IS shared (non-linear confirmation)")

    # Analyze distributions
    analyze_entropy_distribution(noun_adj, noun_verb)

    # Verb entropy signatures
    verb_stats = compute_verb_entropy_polarity(noun_adj, noun_verb, top_verbs=200)

    # Export
    output = {
        "generated_at": datetime.now().isoformat(),
        "method": "Shannon entropy correlation",
        "raw_entropy_pearson_r": r_raw,
        "normalized_entropy_pearson_r": r_norm,
        "n_common_nouns": n,
        "verb_entropy_stats": {
            verb: {"mean_H": mean, "std_H": std}
            for verb, (mean, std) in verb_stats.items()
        }
    }

    with open(OUTPUT_DIR / "shannon_entropy_correlation.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to shannon_entropy_correlation.json")


if __name__ == "__main__":
    main()
