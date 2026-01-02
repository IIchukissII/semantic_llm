#!/usr/bin/env python3
"""
FALSIFIABLE EXPERIMENT #2: Semantic Similarity

(A, S, τ) should capture SEMANTIC STRUCTURE, not sentiment.

Test: Do semantically similar words have similar coordinates?

HYPOTHESIS:
    If (A, S, τ) = real semantic structure:
    → Similar words should be close in (A, S, τ) space
    → Dissimilar words should be far apart
    → Distance should correlate with human similarity judgments
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple
from scipy import stats

# Add paths
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM = _THIS_FILE.parent.parent.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.semantic_coords import BottleneckEncoder


# ============================================================================
# SEMANTIC SIMILARITY PAIRS (human judgments)
# ============================================================================

# Similar pairs (should be close)
SIMILAR_PAIRS = [
    # Concrete
    ("king", "queen"),
    ("man", "woman"),
    ("boy", "girl"),
    ("father", "mother"),
    ("brother", "sister"),
    ("sun", "moon"),
    ("day", "night"),
    ("light", "darkness"),
    ("fire", "water"),
    ("earth", "sky"),

    # Abstract
    ("love", "hate"),       # opposites but same domain
    ("life", "death"),      # opposites but same domain
    ("truth", "lie"),
    ("good", "evil"),
    ("peace", "war"),
    ("joy", "sorrow"),
    ("hope", "despair"),
    ("freedom", "slavery"),
    ("wisdom", "folly"),
    ("justice", "injustice"),
]

# Dissimilar pairs (should be far)
DISSIMILAR_PAIRS = [
    ("king", "stone"),
    ("love", "table"),
    ("truth", "chair"),
    ("life", "paper"),
    ("freedom", "spoon"),
    ("wisdom", "shoe"),
    ("peace", "hammer"),
    ("justice", "window"),
    ("hope", "floor"),
    ("joy", "wall"),
]

# Category coherence test
CATEGORIES = {
    "emotions": ["love", "fear", "anger", "joy", "sorrow", "hope", "despair"],
    "abstracts": ["truth", "freedom", "justice", "wisdom", "beauty", "meaning", "order"],
    "nature": ["sun", "moon", "earth", "sky", "fire", "water", "light"],
    "family": ["father", "mother", "brother", "sister", "son", "daughter", "child"],
    "time": ["day", "night", "morning", "evening", "year", "time", "moment"],
    "body": ["heart", "mind", "soul", "spirit", "body", "blood", "life"],
}


def run_experiment():
    """Run semantic similarity experiment."""

    print("=" * 70)
    print("FALSIFIABLE EXPERIMENT: Semantic Similarity")
    print("=" * 70)
    print()

    encoder = BottleneckEncoder()
    print(f"Loaded: {encoder.n_words} words")
    print()

    # ========================================
    # TEST 1: Similar vs Dissimilar Distance
    # ========================================

    print("TEST 1: Similar pairs vs Dissimilar pairs")
    print("-" * 50)

    similar_distances = []
    for w1, w2 in SIMILAR_PAIRS:
        c1 = encoder.encode_word(w1)
        c2 = encoder.encode_word(w2)
        if c1 and c2:
            dist = c1.distance(c2)
            similar_distances.append(dist)
            print(f"  {w1:12} ↔ {w2:12}: {dist:.3f}")

    print()
    dissimilar_distances = []
    for w1, w2 in DISSIMILAR_PAIRS:
        c1 = encoder.encode_word(w1)
        c2 = encoder.encode_word(w2)
        if c1 and c2:
            dist = c1.distance(c2)
            dissimilar_distances.append(dist)
            print(f"  {w1:12} ↔ {w2:12}: {dist:.3f}")

    print()
    similar_mean = np.mean(similar_distances)
    dissimilar_mean = np.mean(dissimilar_distances)

    print(f"  Similar pairs mean distance:    {similar_mean:.3f}")
    print(f"  Dissimilar pairs mean distance: {dissimilar_mean:.3f}")
    print(f"  Ratio (dissimilar/similar):     {dissimilar_mean/similar_mean:.2f}x")
    print()

    # Statistical test
    t_stat, p_value = stats.ttest_ind(similar_distances, dissimilar_distances)
    print(f"  t-test: t={t_stat:.2f}, p={p_value:.4f}")

    if dissimilar_mean > similar_mean and p_value < 0.05:
        print("  ✓ PASSED: Dissimilar words are significantly farther apart")
        test1_passed = True
    else:
        print("  ✗ FAILED: No significant difference")
        test1_passed = False

    print()

    # ========================================
    # TEST 2: Category Coherence
    # ========================================

    print("TEST 2: Category Coherence (intra-class vs inter-class)")
    print("-" * 50)

    intra_distances = []
    inter_distances = []

    for cat_name, words in CATEGORIES.items():
        # Get coordinates
        coords = []
        for w in words:
            c = encoder.encode_word(w)
            if c:
                coords.append((w, c))

        if len(coords) < 3:
            continue

        # Intra-class distances
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = coords[i][1].distance(coords[j][1])
                intra_distances.append(dist)

        # Compute category centroid
        A_mean = np.mean([c[1].A for c in coords])
        S_mean = np.mean([c[1].S for c in coords])
        tau_mean = np.mean([c[1].tau for c in coords])

        print(f"  {cat_name:12}: {len(coords)} words, centroid=(A={A_mean:.2f}, S={S_mean:.2f}, τ={tau_mean:.2f})")

    # Inter-class: sample pairs from different categories
    cat_list = list(CATEGORIES.keys())
    for i in range(len(cat_list)):
        for j in range(i+1, len(cat_list)):
            words1 = CATEGORIES[cat_list[i]]
            words2 = CATEGORIES[cat_list[j]]

            for w1 in words1[:3]:
                for w2 in words2[:3]:
                    c1 = encoder.encode_word(w1)
                    c2 = encoder.encode_word(w2)
                    if c1 and c2:
                        inter_distances.append(c1.distance(c2))

    print()
    intra_mean = np.mean(intra_distances)
    inter_mean = np.mean(inter_distances)

    print(f"  Intra-category mean distance: {intra_mean:.3f}")
    print(f"  Inter-category mean distance: {inter_mean:.3f}")
    print(f"  Ratio (inter/intra):          {inter_mean/intra_mean:.2f}x")
    print()

    t_stat, p_value = stats.ttest_ind(intra_distances, inter_distances)
    print(f"  t-test: t={t_stat:.2f}, p={p_value:.4f}")

    if inter_mean > intra_mean and p_value < 0.05:
        print("  ✓ PASSED: Categories are coherent")
        test2_passed = True
    else:
        print("  ✗ FAILED: Categories not coherent")
        test2_passed = False

    print()

    # ========================================
    # TEST 3: Nearest Neighbors Quality
    # ========================================

    print("TEST 3: Nearest Neighbor Quality")
    print("-" * 50)

    test_words = ["truth", "love", "king", "life", "freedom", "wisdom"]

    for word in test_words:
        neighbors = encoder.nearest_word(word, k=5)
        print(f"  {word}: {', '.join(neighbors)}")

    print()
    print("  (Manual inspection: do neighbors make semantic sense?)")
    print()

    # ========================================
    # TEST 4: Dimensional Analysis
    # ========================================

    print("TEST 4: What Each Dimension Captures")
    print("-" * 50)

    # Find extremes on each dimension
    all_words = encoder.words[:5000]  # Sample
    coords_data = []
    for w in all_words:
        c = encoder.encode_word(w)
        if c:
            coords_data.append((w, c.A, c.S, c.tau))

    coords_data.sort(key=lambda x: x[1])  # Sort by A
    print(f"  Lowest A:  {', '.join([x[0] for x in coords_data[:5]])}")
    print(f"  Highest A: {', '.join([x[0] for x in coords_data[-5:]])}")

    coords_data.sort(key=lambda x: x[2])  # Sort by S
    print(f"  Lowest S:  {', '.join([x[0] for x in coords_data[:5]])}")
    print(f"  Highest S: {', '.join([x[0] for x in coords_data[-5:]])}")

    coords_data.sort(key=lambda x: x[3])  # Sort by τ
    print(f"  Lowest τ:  {', '.join([x[0] for x in coords_data[:5]])}")
    print(f"  Highest τ: {', '.join([x[0] for x in coords_data[-5:]])}")

    print()

    # ========================================
    # SUMMARY
    # ========================================

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print(f"  Test 1 (Similar vs Dissimilar): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  Test 2 (Category Coherence):    {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print()

    if test1_passed and test2_passed:
        print("  CONCLUSION: (A, S, τ) captures semantic structure!")
        print()
        print("  The coordinates correctly place:")
        print("    - Similar words close together")
        print("    - Dissimilar words far apart")
        print("    - Categories as coherent clusters")
        print()
        print("  This supports the hypothesis that (A, S, τ)")
        print("  is a valid semantic representation.")
    else:
        print("  CONCLUSION: Mixed results")
        print()
        print("  (A, S, τ) may capture some structure but not all.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
