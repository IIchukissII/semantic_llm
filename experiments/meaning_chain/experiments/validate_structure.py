#!/usr/bin/env python3
"""
FALSIFIABLE EXPERIMENT: Is (A, S, τ) the Real Structure of Meaning?

HYPOTHESIS:
    If (A, S, τ) captures the REAL structure of meaning,
    then a model using only 3 numbers should perform comparably to BERT (768D).

    If (A, S, τ) is just lossy compression,
    performance will degrade significantly.

ANALOGY:
    Quantum orbitals (n, l, m) work because they're the real structure.
    If they were just a convenient model, predictions wouldn't match.

EXPERIMENT:
    Task: Sentiment Classification

    BERT (768D):    accuracy X%
    (A, S, τ) 3D:   accuracy Y%

    If Y ≈ X → Structure is real
    If Y << X → Something essential is lost

This is not optimization. This is EXPERIMENT.
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import random

# Add paths
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM = _THIS_FILE.parent.parent.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.semantic_coords import BottleneckEncoder, SemanticCoord


# ============================================================================
# SENTIMENT DATASET
# ============================================================================

# Simple labeled sentences for sentiment (positive/negative)
# We use simple examples to focus on the structure, not the data
SENTIMENT_DATA = [
    # Positive
    ("I love this beautiful day", 1),
    ("What a wonderful experience", 1),
    ("This is amazing and great", 1),
    ("I feel happy and blessed", 1),
    ("Life is beautiful", 1),
    ("Joy and happiness everywhere", 1),
    ("Excellent work well done", 1),
    ("Perfect solution to the problem", 1),
    ("I am grateful for everything", 1),
    ("This makes me so happy", 1),
    ("Wonderful news today", 1),
    ("I appreciate your kindness", 1),
    ("Such a delightful surprise", 1),
    ("Everything is going great", 1),
    ("I am thrilled with the results", 1),
    ("This brings me peace and joy", 1),
    ("A truly magnificent achievement", 1),
    ("I cherish these moments", 1),
    ("What a blessing this is", 1),
    ("Pure happiness and love", 1),
    ("Success and prosperity await", 1),
    ("I am proud of this accomplishment", 1),
    ("This fills my heart with joy", 1),
    ("A beautiful gift of life", 1),
    ("I treasure every moment", 1),
    ("Wonderful friends and family", 1),
    ("This is truly inspiring", 1),
    ("I feel so alive and free", 1),
    ("What a fantastic opportunity", 1),
    ("I am deeply grateful", 1),

    # Negative
    ("I hate this terrible day", 0),
    ("What a horrible experience", 0),
    ("This is awful and bad", 0),
    ("I feel sad and miserable", 0),
    ("Life is painful", 0),
    ("Sorrow and suffering everywhere", 0),
    ("Terrible work poorly done", 0),
    ("Worst solution to the problem", 0),
    ("I am resentful of everything", 0),
    ("This makes me so angry", 0),
    ("Horrible news today", 0),
    ("I despise your cruelty", 0),
    ("Such a dreadful disaster", 0),
    ("Everything is going wrong", 0),
    ("I am devastated by the results", 0),
    ("This brings me pain and sorrow", 0),
    ("A truly catastrophic failure", 0),
    ("I regret these moments", 0),
    ("What a curse this is", 0),
    ("Pure misery and hate", 0),
    ("Failure and poverty await", 0),
    ("I am ashamed of this disaster", 0),
    ("This breaks my heart with pain", 0),
    ("A terrible burden of life", 0),
    ("I waste every moment", 0),
    ("Terrible enemies and strangers", 0),
    ("This is truly depressing", 0),
    ("I feel so dead and trapped", 0),
    ("What a disastrous situation", 0),
    ("I am deeply resentful", 0),
]


def encode_sentence(encoder: BottleneckEncoder, text: str) -> np.ndarray:
    """
    Encode sentence to (A, S, τ) feature vector.

    Strategy: Compute centroid of all word coordinates.
    Returns: [A_mean, S_mean, τ_mean, A_std, S_std, τ_std]
    """
    words = text.lower().split()
    coords = []

    for word in words:
        # Clean word
        word = ''.join(c for c in word if c.isalpha())
        if not word:
            continue

        coord = encoder.encode_word(word)
        if coord:
            coords.append([coord.A, coord.S, coord.tau])

    if not coords:
        return np.zeros(6)

    coords = np.array(coords)

    # Features: mean and std of (A, S, τ)
    mean = coords.mean(axis=0)
    std = coords.std(axis=0) if len(coords) > 1 else np.zeros(3)

    return np.concatenate([mean, std])


def simple_classifier(features: np.ndarray, weights: np.ndarray, bias: float) -> int:
    """Simple linear classifier."""
    score = np.dot(features, weights) + bias
    return 1 if score > 0 else 0


def train_classifier(X: np.ndarray, y: np.ndarray,
                     lr: float = 0.1, epochs: int = 100) -> Tuple[np.ndarray, float]:
    """Train simple logistic regression."""
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        for i in range(len(X)):
            score = np.dot(X[i], weights) + bias
            pred = 1 / (1 + np.exp(-score))  # sigmoid
            error = y[i] - pred

            weights += lr * error * X[i]
            bias += lr * error

    return weights, bias


def evaluate_accuracy(X: np.ndarray, y: np.ndarray,
                      weights: np.ndarray, bias: float) -> float:
    """Evaluate classifier accuracy."""
    correct = 0
    for i in range(len(X)):
        pred = simple_classifier(X[i], weights, bias)
        if pred == y[i]:
            correct += 1
    return correct / len(X)


def run_experiment():
    """Run the falsifiable experiment."""

    print("=" * 70)
    print("FALSIFIABLE EXPERIMENT: Is (A, S, τ) the Real Structure?")
    print("=" * 70)
    print()

    print("HYPOTHESIS:")
    print("  If (A, S, τ) = real structure → 3D model ≈ 768D model")
    print("  If (A, S, τ) = lossy projection → 3D model << 768D model")
    print()

    # Load encoder
    encoder = BottleneckEncoder()
    print(f"Loaded: {encoder.n_words} words, {encoder.n_verbs} verbs")
    print()

    # Encode all sentences
    print("Encoding sentences with (A, S, τ)...")
    X = []
    y = []

    for text, label in SENTIMENT_DATA:
        features = encode_sentence(encoder, text)
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset: {len(X)} sentences, {X.shape[1]} features")
    print(f"Features: [A_mean, S_mean, τ_mean, A_std, S_std, τ_std]")
    print()

    # Cross-validation (5-fold)
    print("Running 5-fold cross-validation...")
    print()

    n_folds = 5
    fold_size = len(X) // n_folds
    accuracies = []

    indices = list(range(len(X)))
    random.seed(42)
    random.shuffle(indices)

    for fold in range(n_folds):
        # Split
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_idx = indices[test_start:test_end]
        train_idx = indices[:test_start] + indices[test_end:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Train
        weights, bias = train_classifier(X_train, y_train, lr=0.5, epochs=200)

        # Evaluate
        acc = evaluate_accuracy(X_test, y_test, weights, bias)
        accuracies.append(acc)
        print(f"  Fold {fold+1}: {acc*100:.1f}%")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print()
    print("-" * 70)
    print(f"(A, S, τ) 3D MODEL: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    print("-" * 70)
    print()

    # Analyze what the model learned
    print("ANALYSIS: What does (A, S, τ) capture?")
    print()

    # Compare positive vs negative word coordinates
    pos_coords = []
    neg_coords = []

    for text, label in SENTIMENT_DATA:
        words = text.lower().split()
        for word in words:
            word = ''.join(c for c in word if c.isalpha())
            coord = encoder.encode_word(word)
            if coord:
                if label == 1:
                    pos_coords.append([coord.A, coord.S, coord.tau])
                else:
                    neg_coords.append([coord.A, coord.S, coord.tau])

    pos_coords = np.array(pos_coords)
    neg_coords = np.array(neg_coords)

    print("  Positive words:")
    print(f"    A (affirmation): {pos_coords[:,0].mean():+.3f} ± {pos_coords[:,0].std():.3f}")
    print(f"    S (sacred):      {pos_coords[:,1].mean():+.3f} ± {pos_coords[:,1].std():.3f}")
    print(f"    τ (abstraction): {pos_coords[:,2].mean():+.3f} ± {pos_coords[:,2].std():.3f}")
    print()
    print("  Negative words:")
    print(f"    A (affirmation): {neg_coords[:,0].mean():+.3f} ± {neg_coords[:,0].std():.3f}")
    print(f"    S (sacred):      {neg_coords[:,1].mean():+.3f} ± {neg_coords[:,1].std():.3f}")
    print(f"    τ (abstraction): {neg_coords[:,2].mean():+.3f} ± {neg_coords[:,2].std():.3f}")
    print()

    # Delta
    delta_A = pos_coords[:,0].mean() - neg_coords[:,0].mean()
    delta_S = pos_coords[:,1].mean() - neg_coords[:,1].mean()
    delta_tau = pos_coords[:,2].mean() - neg_coords[:,2].mean()

    print("  Δ (positive - negative):")
    print(f"    ΔA: {delta_A:+.3f}  {'← SENTIMENT IS IN A!' if abs(delta_A) > 0.3 else ''}")
    print(f"    ΔS: {delta_S:+.3f}")
    print(f"    Δτ: {delta_tau:+.3f}")
    print()

    # Baseline comparisons
    print("=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print()
    print("  Random baseline:          50.0%")
    print(f"  (A, S, τ) 3D model:       {mean_acc*100:.1f}%")
    print("  ─────────────────────────────────")
    print("  BERT-base (768D):         ~93%  (literature)")
    print("  Fine-tuned BERT:          ~95%  (literature)")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if mean_acc > 0.85:
        print("  RESULT: Y ≈ X")
        print()
        print("  The 3D model achieves comparable performance to BERT.")
        print("  This suggests (A, S, τ) captures the REAL semantic structure.")
        print()
        print("  Like quantum numbers (n, l, m) — if they work, they're real.")

    elif mean_acc > 0.70:
        print("  RESULT: Y < X but substantial")
        print()
        print("  The 3D model captures significant structure but not everything.")
        print("  (A, S, τ) may be a good approximation but not complete.")
        print()
        print("  Possible interpretations:")
        print("    1. (A, S, τ) is core structure, syntax adds the rest")
        print("    2. More features needed (verb operators?)")
        print("    3. Aggregation method loses information")

    else:
        print("  RESULT: Y << X")
        print()
        print("  The 3D model significantly underperforms.")
        print("  (A, S, τ) may be lossy projection, not the real structure.")
        print()
        print("  The hypothesis is FALSIFIED.")

    print()
    print("=" * 70)

    return mean_acc


if __name__ == "__main__":
    run_experiment()
