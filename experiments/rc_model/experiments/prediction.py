"""Experiment 7.2: Next Bond Prediction

Tests whether the RC model has predictive power beyond unigram baseline.

Experiments:
1. Next bond prediction - perplexity vs unigram
2. Within vs Between sentences - perp_within < perp_between?
3. Attractors - do trajectory endpoints cluster?

Physics:
    P(next | Q) ∝ exp(-|Δn|/kT) × exp(-Δθ²/2σ²)

Perplexity:
    PPL = exp(mean(-log P(correct)))
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional
from collections import Counter
import random

from ..core.bond_extractor import BondExtractor, TextBonds, Bond
from ..core.semantic_rc import SemanticRC, Trajectory, KT
from ..core.coord_loader import CoordLoader, get_loader


@dataclass
class PredictionResult:
    """Result of prediction experiment."""
    # Rank-based metrics (more meaningful for large vocab)
    mean_rank_model: float       # Mean rank of correct word (lower is better)
    mean_rank_random: float      # Expected random rank (vocab_size / 2)
    mrr_model: float             # Mean reciprocal rank
    hits_at_10: float            # % correct in top 10
    hits_at_100: float           # % correct in top 100
    hits_at_1000: float          # % correct in top 1000

    # Within/between sentence (rank-based)
    mean_rank_within: float
    mean_rank_between: float

    # Counts
    n_predictions: int
    n_within: int
    n_between: int
    vocab_size: int

    # Raw data
    ranks: list[int]
    is_within: list[bool]


class BondPredictor:
    """Predicts next bond using RC model physics."""

    def __init__(
        self,
        rc_model: Optional[SemanticRC] = None,
        loader: Optional[CoordLoader] = None,
        sigma_theta: float = 0.3,  # Width for angular Gaussian (tighter)
        sigma_r: float = 0.2,      # Width for magnitude Gaussian (tighter)
    ):
        self.rc = rc_model or SemanticRC()
        self.loader = loader or get_loader()
        self.sigma_theta = sigma_theta
        self.sigma_r = sigma_r

        # Build vocabulary of all words with coordinates
        self._vocab = list(self.loader._coords.keys())
        self._vocab_coords = {
            w: (c.n, c.theta, c.r)
            for w, c in self.loader._coords.items()
        }

        # Precompute coordinate arrays for vectorized operations
        self._n_array = np.array([self._vocab_coords[w][0] for w in self._vocab])
        self._theta_array = np.array([self._vocab_coords[w][1] for w in self._vocab])
        self._r_array = np.array([self._vocab_coords[w][2] for w in self._vocab])
        self._word_to_idx = {w: i for i, w in enumerate(self._vocab)}

        # Unigram frequencies (uniform if not provided)
        self._unigram_probs = np.ones(len(self._vocab)) / len(self._vocab)
        self._unigram_dict = {w: 1.0 / len(self._vocab) for w in self._vocab}

    def set_unigram_probs(self, word_counts: dict[str, int]):
        """Set unigram probabilities from word counts."""
        total = sum(word_counts.values())
        for i, w in enumerate(self._vocab):
            self._unigram_probs[i] = word_counts.get(w, 1) / total
            self._unigram_dict[w] = self._unigram_probs[i]

    def _angular_diff_vec(self, theta_array: np.ndarray, theta_q: float) -> np.ndarray:
        """Vectorized angular difference wrapped to [-π, π]."""
        diff = theta_array - theta_q
        diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
        return diff

    def compute_all_probs(self) -> np.ndarray:
        """Compute probability distribution over entire vocabulary (vectorized).

        P(word | Q) ∝ exp(-|Δn|/kT) × exp(-Δθ²/2σ²) × exp(-Δr²/2σ²)
        """
        Q = self.rc.Q
        Q_n, Q_theta, Q_r = Q[0], Q[1], Q[2]

        # Boltzmann factor for n
        delta_n = np.abs(self._n_array - Q_n)
        log_p_n = -delta_n / self.rc.kT

        # Gaussian for theta
        delta_theta = self._angular_diff_vec(self._theta_array, Q_theta)
        log_p_theta = -delta_theta**2 / (2 * self.sigma_theta**2)

        # Gaussian for r
        delta_r = np.abs(self._r_array - Q_r)
        log_p_r = -delta_r**2 / (2 * self.sigma_r**2)

        # Log probability (unnormalized)
        log_scores = log_p_n + log_p_theta + log_p_r

        # Normalize using log-sum-exp for numerical stability
        log_norm = np.max(log_scores) + np.log(np.sum(np.exp(log_scores - np.max(log_scores))))
        log_probs = log_scores - log_norm

        return np.exp(log_probs)

    def get_prob(self, word: str) -> float:
        """Get model probability for a specific word."""
        word_lower = word.lower()
        if word_lower not in self._word_to_idx:
            return 1e-10

        probs = self.compute_all_probs()
        idx = self._word_to_idx[word_lower]
        return probs[idx]

    def get_rank(self, word: str) -> int:
        """Get rank of word in probability distribution (1 = most likely).

        Returns vocab_size if word not found.
        """
        word_lower = word.lower()
        if word_lower not in self._word_to_idx:
            return len(self._vocab)

        probs = self.compute_all_probs()
        idx = self._word_to_idx[word_lower]
        word_prob = probs[idx]

        # Count how many words have higher probability
        rank = int(np.sum(probs > word_prob)) + 1
        return rank

    def get_unigram_prob(self, word: str) -> float:
        """Get unigram probability for a word."""
        return self._unigram_dict.get(word.lower(), 1e-10)


def run_prediction_experiment(
    texts: list[str],
    extractor: Optional[BondExtractor] = None,
    predictor: Optional[BondPredictor] = None,
    seed: int = 42,
) -> PredictionResult:
    """Run prediction experiment on multiple texts.

    Uses PREVIOUS word's coordinates to predict NEXT word.
    This tests semantic coherence directly without RC smoothing.

    Args:
        texts: List of texts to analyze
        extractor: Bond extractor (creates default if None)
        predictor: Bond predictor (creates default if None)
        seed: Random seed

    Returns:
        PredictionResult with all metrics
    """
    random.seed(seed)
    np.random.seed(seed)

    if extractor is None:
        extractor = BondExtractor()
    if predictor is None:
        predictor = BondPredictor()

    # Collect word counts for unigram baseline
    word_counts: Counter = Counter()

    for text in texts:
        bonds = extractor.extract(text)
        for sent in bonds.sentences:
            for bond in sent.bonds:
                if predictor.loader.has(bond.noun):
                    word_counts[bond.noun.lower()] += 1

    # Set unigram probs
    predictor.set_unigram_probs(dict(word_counts))

    # Run prediction using raw coordinates (not RC state)
    ranks = []
    is_within = []
    vocab_size = len(predictor._vocab)

    for text in texts:
        bonds = extractor.extract(text)

        prev_sent_idx = -1
        prev_coord = None  # Previous word's coordinates

        for sent_idx, sent in enumerate(bonds.sentences):
            for bond in sent.bonds:
                if not predictor.loader.has(bond.noun):
                    continue

                # Get current word's coordinates
                coord = predictor.loader.get(bond.noun)
                curr_coord = (coord.n, coord.theta, coord.r)

                # Only predict if we have previous context
                if prev_coord is not None:
                    # Set RC state to previous word's coordinates
                    predictor.rc.Q = np.array(prev_coord)

                    # Get rank of correct word
                    rank = predictor.get_rank(bond.noun)
                    ranks.append(rank)

                    # Within or between sentence?
                    is_within.append(sent_idx == prev_sent_idx)

                prev_coord = curr_coord
                prev_sent_idx = sent_idx

    # Compute rank-based metrics
    ranks_arr = np.array(ranks)
    is_within_arr = np.array(is_within)

    # Mean rank (lower is better)
    mean_rank = np.mean(ranks_arr) if len(ranks_arr) > 0 else vocab_size / 2
    random_expected_rank = vocab_size / 2

    # Mean reciprocal rank
    mrr = np.mean(1.0 / ranks_arr) if len(ranks_arr) > 0 else 0

    # Hits@K
    hits_10 = np.mean(ranks_arr <= 10) if len(ranks_arr) > 0 else 0
    hits_100 = np.mean(ranks_arr <= 100) if len(ranks_arr) > 0 else 0
    hits_1000 = np.mean(ranks_arr <= 1000) if len(ranks_arr) > 0 else 0

    # Within/between sentence ranks
    ranks_within = ranks_arr[is_within_arr] if np.any(is_within_arr) else np.array([vocab_size/2])
    ranks_between = ranks_arr[~is_within_arr] if np.any(~is_within_arr) else np.array([vocab_size/2])

    mean_rank_within = np.mean(ranks_within)
    mean_rank_between = np.mean(ranks_between)

    return PredictionResult(
        mean_rank_model=mean_rank,
        mean_rank_random=random_expected_rank,
        mrr_model=mrr,
        hits_at_10=hits_10,
        hits_at_100=hits_100,
        hits_at_1000=hits_1000,
        mean_rank_within=mean_rank_within,
        mean_rank_between=mean_rank_between,
        n_predictions=len(ranks),
        n_within=int(np.sum(is_within_arr)),
        n_between=int(np.sum(~is_within_arr)),
        vocab_size=vocab_size,
        ranks=ranks,
        is_within=is_within,
    )


def print_prediction_report(result: PredictionResult):
    """Print prediction experiment results."""
    print("=" * 60)
    print("PREDICTION EXPERIMENT: Next Bond Prediction")
    print("=" * 60)

    print(f"\nPredictions made: {result.n_predictions}")
    print(f"  Within-sentence: {result.n_within}")
    print(f"  Between-sentence: {result.n_between}")
    print(f"  Vocabulary size: {result.vocab_size}")

    print("\n" + "-" * 40)
    print("MEAN RANK (lower is better)")
    print("-" * 40)
    print(f"  RC Model:     {result.mean_rank_model:.1f}")
    print(f"  Random:       {result.mean_rank_random:.1f}")

    improvement = (result.mean_rank_random - result.mean_rank_model) / result.mean_rank_random * 100
    print(f"  Improvement:  {improvement:+.1f}%")

    if result.mean_rank_model < result.mean_rank_random:
        print(f"  ✓ Model beats random ({result.mean_rank_random / result.mean_rank_model:.1f}x better)")
    else:
        print(f"  ✗ Model worse than random")

    print("\n" + "-" * 40)
    print("HITS@K (higher is better)")
    print("-" * 40)
    random_hits_10 = 10 / result.vocab_size
    random_hits_100 = 100 / result.vocab_size
    random_hits_1000 = 1000 / result.vocab_size

    print(f"  Hits@10:   {result.hits_at_10:.1%} (random: {random_hits_10:.2%})")
    print(f"  Hits@100:  {result.hits_at_100:.1%} (random: {random_hits_100:.2%})")
    print(f"  Hits@1000: {result.hits_at_1000:.1%} (random: {random_hits_1000:.2%})")
    print(f"  MRR:       {result.mrr_model:.4f}")

    print("\n" + "-" * 40)
    print("WITHIN vs BETWEEN SENTENCES (Mean Rank)")
    print("-" * 40)
    print(f"  Within-sentence:  {result.mean_rank_within:.1f}")
    print(f"  Between-sentence: {result.mean_rank_between:.1f}")

    if result.mean_rank_within < result.mean_rank_between:
        ratio = result.mean_rank_between / result.mean_rank_within
        print(f"  ✓ Within < Between ({ratio:.2f}x easier)")
    else:
        ratio = result.mean_rank_within / result.mean_rank_between
        print(f"  ✗ Between < Within ({ratio:.2f}x harder)")

    print("=" * 60)


# Quick test
def quick_test():
    """Run quick prediction test."""
    texts = [
        """The old man walked slowly through the dark forest.
        He carried a heavy burden on his tired shoulders.
        The ancient trees whispered mysterious secrets.""",

        """The young woman stood bravely before the tall gate.
        Her bright eyes searched the empty courtyard.
        A cold wind stirred the fallen leaves.""",
    ]

    print("Running prediction experiment...")
    result = run_prediction_experiment(texts)
    print_prediction_report(result)
    return result


if __name__ == "__main__":
    quick_test()
