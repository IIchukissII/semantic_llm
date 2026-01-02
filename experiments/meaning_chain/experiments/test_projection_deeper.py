#!/usr/bin/env python3
"""
DEEPER INVESTIGATION: Why does the projection hypothesis fail?

The first test showed r² = 0.000 - essentially NO correlation.
This means either:

1. The theory is wrong
2. The data sources are inconsistent
3. There's a transformation we're missing

Let's investigate:
- What are the stored noun j-vectors actually?
- What are the adjective j-vectors?
- Are they from the same source?
"""

import sys
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_EXPERIMENTS = _MEANING_CHAIN.parent
_SEMANTIC_LLM = _EXPERIMENTS.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.j_space import PC1_AFFIRMATION, PC2_SACRED

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


def investigate():
    """Deep investigation of data sources."""

    print("=" * 70)
    print("INVESTIGATING DATA SOURCES")
    print("=" * 70)
    print()

    loader = DataLoader()
    word_vectors = loader.load_word_vectors()
    adj_profiles = loader.load_noun_adj_profiles()

    # Get unique adjectives
    unique_adjs = set()
    for profile in adj_profiles.values():
        unique_adjs.update(profile.keys())

    # Separate by word type
    nouns_with_j = []
    adjs_with_j = []
    verbs_with_j = []
    other_with_j = []

    for word, data in word_vectors.items():
        if not data.get('j'):
            continue

        wtype = data.get('word_type')
        j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])

        if wtype == 'noun' or wtype == 0:
            nouns_with_j.append((word, j_5d, data))
        elif wtype == 'verb' or wtype == 1:
            verbs_with_j.append((word, j_5d, data))
        elif wtype == 'adjective' or wtype == 2:
            adjs_with_j.append((word, j_5d, data))
        else:
            other_with_j.append((word, j_5d, data))

    print("WORD TYPE DISTRIBUTION (with j-vectors):")
    print("-" * 40)
    print(f"  Nouns:      {len(nouns_with_j)}")
    print(f"  Verbs:      {len(verbs_with_j)}")
    print(f"  Adjectives: {len(adjs_with_j)}")
    print(f"  Other:      {len(other_with_j)}")
    print(f"  TOTAL:      {len(word_vectors)}")
    print()

    # Check if adjectives from profiles are in word_vectors
    adjs_in_wv = sum(1 for a in unique_adjs if a in word_vectors)
    adjs_with_j_count = sum(1 for a in unique_adjs if a in word_vectors and word_vectors[a].get('j'))
    print("ADJECTIVE PROFILE COVERAGE:")
    print("-" * 40)
    print(f"  Unique adjectives in profiles: {len(unique_adjs)}")
    print(f"  Found in word_vectors:         {adjs_in_wv} ({100*adjs_in_wv/len(unique_adjs):.1f}%)")
    print(f"  With j-vectors:                {adjs_with_j_count} ({100*adjs_with_j_count/len(unique_adjs):.1f}%)")
    print()

    # Look at j-vector distributions
    if nouns_with_j:
        noun_j = np.array([x[1] for x in nouns_with_j[:1000]])  # Sample
        print("NOUN J-VECTOR STATISTICS (sample 1000):")
        print("-" * 40)
        for i, dim in enumerate(J_DIMS):
            print(f"  {dim:10s}: mean={np.mean(noun_j[:, i]):+.4f}  std={np.std(noun_j[:, i]):.4f}")
        print()

    if adjs_with_j:
        adj_j = np.array([x[1] for x in adjs_with_j[:1000]])  # Sample
        print("ADJECTIVE J-VECTOR STATISTICS (sample 1000):")
        print("-" * 40)
        for i, dim in enumerate(J_DIMS):
            print(f"  {dim:10s}: mean={np.mean(adj_j[:, i]):+.4f}  std={np.std(adj_j[:, i]):.4f}")
        print()

    # Check a specific example
    print("=" * 70)
    print("SPECIFIC EXAMPLE: 'woman'")
    print("=" * 70)
    print()

    test_noun = 'woman'
    if test_noun in word_vectors and word_vectors[test_noun].get('j'):
        data = word_vectors[test_noun]
        j_stored = np.array([data['j'].get(d, 0) for d in J_DIMS])
        A_stored = float(np.dot(j_stored, PC1_AFFIRMATION))
        S_stored = float(np.dot(j_stored, PC2_SACRED))

        print(f"STORED j-vector for '{test_noun}':")
        for i, dim in enumerate(J_DIMS):
            print(f"  {dim:10s}: {j_stored[i]:+.4f}")
        print(f"  → A = {A_stored:+.4f}, S = {S_stored:+.4f}")
        print()

    if test_noun in adj_profiles:
        profile = adj_profiles[test_noun]
        print(f"ADJECTIVE PROFILE for '{test_noun}':")
        sorted_adjs = sorted(profile.items(), key=lambda x: -x[1])[:10]
        total = sum(profile.values())

        print(f"  Total count: {total}, Variety: {len(profile)}")
        print("  Top 10 adjectives:")
        for adj, count in sorted_adjs:
            weight = count / total if total > 0 else 0
            if adj in word_vectors and word_vectors[adj].get('j'):
                adj_j = np.array([word_vectors[adj]['j'].get(d, 0) for d in J_DIMS])
                adj_A = float(np.dot(adj_j, PC1_AFFIRMATION))
                adj_S = float(np.dot(adj_j, PC2_SACRED))
                print(f"    {adj:15s}: count={count:.4f}  weight={weight:.3f}  A={adj_A:+.3f}  S={adj_S:+.3f}")
            else:
                print(f"    {adj:15s}: count={count:.4f}  weight={weight:.3f}  (no j-vector)")
        print()

        # Compute derived
        j_derived = np.zeros(5)
        weight_sum = 0.0
        for adj, count in profile.items():
            if adj in word_vectors and word_vectors[adj].get('j'):
                weight = count / total
                adj_j = np.array([word_vectors[adj]['j'].get(d, 0) for d in J_DIMS])
                j_derived += weight * adj_j
                weight_sum += weight

        if weight_sum > 0:
            j_derived /= weight_sum
            A_derived = float(np.dot(j_derived, PC1_AFFIRMATION))
            S_derived = float(np.dot(j_derived, PC2_SACRED))

            print(f"DERIVED j-vector (weighted average):")
            for i, dim in enumerate(J_DIMS):
                print(f"  {dim:10s}: {j_derived[i]:+.4f}")
            print(f"  → A = {A_derived:+.4f}, S = {S_derived:+.4f}")
            print()

            print("COMPARISON:")
            print("-" * 40)
            print(f"  A: stored = {A_stored:+.4f}, derived = {A_derived:+.4f}, diff = {A_stored - A_derived:+.4f}")
            print(f"  S: stored = {S_stored:+.4f}, derived = {S_derived:+.4f}, diff = {S_stored - S_derived:+.4f}")

    # Check if stored noun j-vectors were computed from BONDS at all
    print()
    print("=" * 70)
    print("CHECKING DATA ORIGIN")
    print("=" * 70)
    print()

    # Sample 5 nouns and check their j-vectors vs adjective derivation
    sample_nouns = ['love', 'god', 'man', 'death', 'life']
    print("Sampling common nouns to see pattern:")
    print()

    for noun in sample_nouns:
        if noun not in word_vectors or not word_vectors[noun].get('j'):
            print(f"  {noun}: no j-vector")
            continue
        if noun not in adj_profiles:
            print(f"  {noun}: no adjective profile")
            continue

        data = word_vectors[noun]
        profile = adj_profiles[noun]

        j_stored = np.array([data['j'].get(d, 0) for d in J_DIMS])
        A_stored = float(np.dot(j_stored, PC1_AFFIRMATION))
        S_stored = float(np.dot(j_stored, PC2_SACRED))

        # Derived
        total = sum(profile.values())
        j_derived = np.zeros(5)
        weight_sum = 0.0
        for adj, count in profile.items():
            if adj in word_vectors and word_vectors[adj].get('j'):
                weight = count / total
                adj_j = np.array([word_vectors[adj]['j'].get(d, 0) for d in J_DIMS])
                j_derived += weight * adj_j
                weight_sum += weight

        if weight_sum > 0:
            j_derived /= weight_sum
            A_derived = float(np.dot(j_derived, PC1_AFFIRMATION))
            S_derived = float(np.dot(j_derived, PC2_SACRED))

            print(f"  {noun:10s}: stored(A={A_stored:+.3f}, S={S_stored:+.3f})  derived(A={A_derived:+.3f}, S={S_derived:+.3f})  diff(A={A_stored-A_derived:+.3f}, S={S_stored-S_derived:+.3f})")
        else:
            print(f"  {noun:10s}: stored(A={A_stored:+.3f}, S={S_stored:+.3f})  derived: no adjectives found")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
    The stored noun j-vectors and the adjective-derived j-vectors
    appear to come from DIFFERENT SOURCES or use different methods.

    Possible explanations:
    1. Stored j-vectors were computed from Word2Vec or similar embeddings
    2. Derived j-vectors use only bond-space adjective statistics
    3. The theory assumes these should match, but they don't

    NEXT STEPS:
    - Trace how stored j-vectors were originally computed
    - Check if bonds were even used in the original computation
    - The projection hierarchy may need a different formulation
    """)


if __name__ == "__main__":
    investigate()
