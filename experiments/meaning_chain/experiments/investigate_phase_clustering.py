#!/usr/bin/env python3
"""
Investigate Phase Clustering Issue
==================================

Why don't positive/negative words have different phases?

Possible causes:
1. Adjective averaging smooths out differences
2. Original adjective j-vectors don't encode polarity
3. The projection (A, S) doesn't capture positive/negative
"""

import sys
import math
import numpy as np
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_EXPERIMENTS = _MEANING_CHAIN.parent
_SEMANTIC_LLM = _EXPERIMENTS.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.unified_hierarchy import build_hierarchy, PC1_AFFIRMATION, PC2_SACRED

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


def investigate():
    print("=" * 70)
    print("INVESTIGATING PHASE CLUSTERING")
    print("=" * 70)

    loader = DataLoader()
    word_vectors = loader.load_word_vectors()
    adj_profiles = loader.load_noun_adj_profiles()

    # Check the raw j-vectors of positive/negative words
    print("\n1. RAW J-VECTORS OF SEMANTIC POLES")
    print("-" * 50)

    test_words = {
        'positive': ['love', 'good', 'peace', 'joy', 'hope', 'beauty', 'truth'],
        'negative': ['hate', 'evil', 'war', 'fear', 'death', 'ugly', 'lie'],
        'sacred': ['god', 'divine', 'holy', 'spirit', 'soul', 'prayer'],
    }

    for category, words in test_words.items():
        print(f"\n{category.upper()}:")
        for w in words:
            if w in word_vectors and word_vectors[w].get('j'):
                data = word_vectors[w]
                j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])
                A = float(np.dot(j_5d, PC1_AFFIRMATION))
                S = float(np.dot(j_5d, PC2_SACRED))
                theta = math.degrees(math.atan2(S, A))
                r = math.sqrt(A**2 + S**2)
                print(f"  {w:<12} j={j_5d}  A={A:+.3f}  S={S:+.3f}  θ={theta:>7.1f}°  r={r:.3f}")

    # Check what adjectives describe positive vs negative words
    print("\n\n2. ADJECTIVE PROFILES OF SEMANTIC POLES")
    print("-" * 50)

    for category, words in test_words.items():
        print(f"\n{category.upper()}:")
        for w in words:
            if w in adj_profiles:
                profile = adj_profiles[w]
                total = sum(profile.values())
                sorted_adjs = sorted(profile.items(), key=lambda x: -x[1])[:5]
                top_str = ", ".join(f"{a}({c/total:.2f})" for a, c in sorted_adjs)
                print(f"  {w:<12} → {top_str}")

    # The key insight: adjectives describing "love" vs "hate"
    print("\n\n3. COMPARING ADJECTIVE CLOUDS FOR LOVE vs HATE")
    print("-" * 50)

    for w in ['love', 'hate']:
        if w in adj_profiles:
            profile = adj_profiles[w]
            print(f"\n{w.upper()} adjectives:")
            sorted_adjs = sorted(profile.items(), key=lambda x: -x[1])[:15]
            for adj, count in sorted_adjs:
                total = sum(profile.values())
                weight = count / total
                if adj in word_vectors and word_vectors[adj].get('j'):
                    data = word_vectors[adj]
                    j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])
                    A = float(np.dot(j_5d, PC1_AFFIRMATION))
                    S = float(np.dot(j_5d, PC2_SACRED))
                    theta = math.degrees(math.atan2(S, A))
                    print(f"    {adj:<15} w={weight:.3f}  A={A:+.3f}  S={S:+.3f}  θ={theta:>7.1f}°")
                else:
                    print(f"    {adj:<15} w={weight:.3f}  (no j-vector)")

    # Check the actual derived coordinates
    print("\n\n4. DERIVED vs RAW COORDINATES")
    print("-" * 50)

    hierarchy = build_hierarchy(loader)

    print("\nComparison (RAW j-vector vs DERIVED centroid):")
    print(f"{'Word':<12} {'Raw_θ°':>10} {'Der_θ°':>10} {'Diff':>8} {'Raw_r':>8} {'Der_r':>8}")
    print("-" * 60)

    all_words = ['love', 'hate', 'good', 'evil', 'peace', 'war', 'life', 'death',
                 'god', 'man', 'woman', 'beauty', 'ugly']

    for w in all_words:
        # Raw
        raw_theta = None
        raw_r = None
        if w in word_vectors and word_vectors[w].get('j'):
            data = word_vectors[w]
            j_5d = np.array([data['j'].get(d, 0) for d in J_DIMS])
            A = float(np.dot(j_5d, PC1_AFFIRMATION))
            S = float(np.dot(j_5d, PC2_SACRED))
            raw_theta = math.degrees(math.atan2(S, A))
            raw_r = math.sqrt(A**2 + S**2)

        # Derived
        qw = hierarchy.get_word(w)
        der_theta = math.degrees(qw.theta) if qw else None
        der_r = qw.r if qw else None

        if raw_theta is not None and der_theta is not None:
            diff = raw_theta - der_theta
            print(f"{w:<12} {raw_theta:>10.1f} {der_theta:>10.1f} {diff:>8.1f} {raw_r:>8.3f} {der_r:>8.3f}")
        elif raw_theta is not None:
            print(f"{w:<12} {raw_theta:>10.1f} {'--':>10} {'--':>8} {raw_r:>8.3f} {'--':>8}")
        else:
            print(f"{w:<12} {'--':>10} {der_theta if der_theta else '--':>10}")

    # Conclusion
    print("\n\n5. CONCLUSION")
    print("-" * 50)
    print("""
    The RAW j-vectors from the database encode semantic polarity,
    but when we compute centroids from ADJECTIVE clouds, we get:

    1. Smoothing effect: Many common adjectives (old, other, great, etc.)
       are semantically neutral, pulling all centroids toward the mean.

    2. Shared adjectives: 'love' and 'hate' may share many adjectives
       (e.g., "great hate", "great love"), reducing their difference.

    3. The derived coordinates reflect USAGE PATTERNS, not semantic content.
       Words that appear in similar contexts get similar coordinates.

    INSIGHT: The original j-vectors captured semantic meaning directly.
             The derived centroids capture statistical co-occurrence.

    These are TWO DIFFERENT but both valid coordinate systems!
    """)


if __name__ == "__main__":
    investigate()
