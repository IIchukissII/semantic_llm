"""Experiment 7.1: Coherent vs Random (Sanity Check)

If the model can't distinguish coherent text from noise, everything else is meaningless.

Protocol:
1. Take text T (article, book chapter, etc.)
2. Extract structure: [[bonds], [bonds], ...]
3. Create shuffled versions:
   - shuffle within sentences
   - shuffle between sentences
   - full shuffle (all bonds)
4. Run all through RC-model
5. Compare trajectories

Expected:
    original:        low σ_within, structured jumps, high autocorr
    shuffled_within: high σ_within, same jumps
    shuffled_between: low σ_within, chaotic jumps
    full_shuffle:    everything chaotic

Success criteria:
    - σ_within ratio (orig/shuf) < 0.5
    - autocorr(orig) > 0.5
    - p-value < 0.01
    - Cohen's d > 0.8
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import Optional
import copy

from ..core.bond_extractor import BondExtractor, TextBonds, SentenceBonds, Bond
from ..core.semantic_rc import SemanticRC, Trajectory
from ..core.metrics import compute_metrics, TrajectoryMetrics, compare_metrics, paired_ttest
from ..core.cardiogram import plot_cardiogram, plot_comparison


@dataclass
class ShuffledResult:
    """Result of processing original and shuffled versions."""
    original: Trajectory
    shuffled_within: Trajectory
    shuffled_between: Trajectory
    full_shuffle: Trajectory
    random_vocab: Trajectory  # Random words from full vocabulary

    original_metrics: TrajectoryMetrics
    shuffled_within_metrics: TrajectoryMetrics
    shuffled_between_metrics: TrajectoryMetrics
    full_shuffle_metrics: TrajectoryMetrics
    random_vocab_metrics: TrajectoryMetrics


def shuffle_bonds(text_bonds: TextBonds, mode: str = 'full') -> TextBonds:
    """Shuffle bonds in different ways.

    Args:
        text_bonds: Original extracted bonds
        mode: Shuffle mode:
            - 'within': shuffle bonds within each sentence
            - 'between': shuffle sentences (keep internal order)
            - 'full': shuffle all bonds globally

    Returns:
        New TextBonds with shuffled content
    """
    if mode == 'within':
        # Shuffle bonds within each sentence
        new_sentences = []
        for sent in text_bonds.sentences:
            shuffled_bonds = list(sent.bonds)
            random.shuffle(shuffled_bonds)
            new_sentences.append(SentenceBonds(
                bonds=shuffled_bonds,
                text=sent.text,
            ))
        return TextBonds(sentences=new_sentences)

    elif mode == 'between':
        # Shuffle sentence order (keep internal structure)
        new_sentences = list(text_bonds.sentences)
        random.shuffle(new_sentences)
        return TextBonds(sentences=new_sentences)

    elif mode == 'full':
        # Full shuffle: all bonds mixed, random sentence boundaries
        all_bonds = [b for s in text_bonds.sentences for b in s.bonds]
        random.shuffle(all_bonds)

        # Recreate sentences with similar lengths
        orig_lengths = [len(s) for s in text_bonds.sentences]
        random.shuffle(orig_lengths)

        new_sentences = []
        idx = 0
        for length in orig_lengths:
            end = min(idx + length, len(all_bonds))
            if end > idx:
                new_sentences.append(SentenceBonds(
                    bonds=all_bonds[idx:end],
                    text="[shuffled]",
                ))
            idx = end
            if idx >= len(all_bonds):
                break

        return TextBonds(sentences=new_sentences)

    else:
        raise ValueError(f"Unknown shuffle mode: {mode}")


def create_random_vocab_bonds(text_bonds: TextBonds, loader) -> TextBonds:
    """Create bonds with random words from full vocabulary."""
    all_words = list(loader._coords.keys())

    random_sentences = []
    for sent in text_bonds.sentences:
        random_bonds = []
        for bond in sent.bonds:
            noun = random.choice(all_words)
            adj = random.choice(all_words) if bond.adj else None
            random_bonds.append(Bond(
                noun=noun,
                adj=adj,
                noun_pos='NOUN',
                adj_pos='ADJ' if adj else None,
            ))
        random_sentences.append(SentenceBonds(bonds=random_bonds, text='[random]'))

    return TextBonds(sentences=random_sentences)


def run_sanity_check(
    text: str,
    extractor: Optional[BondExtractor] = None,
    rc_model: Optional[SemanticRC] = None,
    seed: int = 42,
) -> ShuffledResult:
    """Run the coherent vs random sanity check.

    Args:
        text: Input text to analyze
        extractor: Bond extractor (creates default if None)
        rc_model: RC model (creates default if None)
        seed: Random seed for reproducibility

    Returns:
        ShuffledResult with all trajectories and metrics
    """
    random.seed(seed)
    np.random.seed(seed)

    if extractor is None:
        extractor = BondExtractor()
    if rc_model is None:
        rc_model = SemanticRC()

    # Extract bonds
    text_bonds = extractor.extract(text)

    # Create shuffled versions
    shuffled_within = shuffle_bonds(text_bonds, 'within')
    shuffled_between = shuffle_bonds(text_bonds, 'between')
    full_shuffle = shuffle_bonds(text_bonds, 'full')
    random_vocab = create_random_vocab_bonds(text_bonds, rc_model.loader)

    # Process all versions
    traj_original = rc_model.process_text(text_bonds)

    rc_model.reset()
    traj_shuffled_within = rc_model.process_text(shuffled_within)

    rc_model.reset()
    traj_shuffled_between = rc_model.process_text(shuffled_between)

    rc_model.reset()
    traj_full_shuffle = rc_model.process_text(full_shuffle)

    rc_model.reset()
    traj_random_vocab = rc_model.process_text(random_vocab)

    # Compute metrics
    metrics_original = compute_metrics(traj_original)
    metrics_shuffled_within = compute_metrics(traj_shuffled_within)
    metrics_shuffled_between = compute_metrics(traj_shuffled_between)
    metrics_full_shuffle = compute_metrics(traj_full_shuffle)
    metrics_random_vocab = compute_metrics(traj_random_vocab)

    return ShuffledResult(
        original=traj_original,
        shuffled_within=traj_shuffled_within,
        shuffled_between=traj_shuffled_between,
        full_shuffle=traj_full_shuffle,
        random_vocab=traj_random_vocab,
        original_metrics=metrics_original,
        shuffled_within_metrics=metrics_shuffled_within,
        shuffled_between_metrics=metrics_shuffled_between,
        full_shuffle_metrics=metrics_full_shuffle,
        random_vocab_metrics=metrics_random_vocab,
    )


def evaluate_result(result: ShuffledResult) -> dict:
    """Evaluate sanity check result against success criteria.

    Success criteria (from THEORY.md):
        - raw_coherence ratio (orig/random_vocab) > 2.0
        - raw_bond_strength ratio > 2.0
        - raw_coherence > 0.3 (above random baseline ~0.2)

    Args:
        result: ShuffledResult from run_sanity_check

    Returns:
        Dict with evaluation results
    """
    orig = result.original_metrics
    rand = result.random_vocab_metrics
    full = result.full_shuffle_metrics

    # Compute ratios
    def safe_ratio(a, b):
        if b == 0:
            return float('inf') if a > 0 else 1.0
        return a / b

    # Primary: compare with random vocabulary (the real test)
    coherence_ratio_rand = safe_ratio(orig.raw_coherence, rand.raw_coherence)

    # For bond strength: use difference when random is near zero or negative
    # (ratio doesn't work well when denominator is near zero or negative)
    strength_diff = orig.raw_bond_strength - rand.raw_bond_strength
    if rand.raw_bond_strength > 0.05:
        strength_ratio_rand = safe_ratio(orig.raw_bond_strength, rand.raw_bond_strength)
    else:
        # When random is near zero or negative, use difference-based metric
        strength_ratio_rand = strength_diff * 10  # Scale for comparison

    # Secondary: compare with shuffled (within-vocabulary test)
    coherence_ratio_shuf = safe_ratio(orig.raw_coherence, full.raw_coherence)

    # Bond strength pass criterion:
    # - Original should be positive (coherent text has positive bonds)
    # - Original should be significantly better than random
    pass_strength = (orig.raw_bond_strength > 0.1) and (strength_diff > 0.2)

    return {
        # Primary metrics: vs random vocabulary
        'raw_coherence_orig': orig.raw_coherence,
        'raw_coherence_random': rand.raw_coherence,
        'coherence_ratio_random': coherence_ratio_rand,
        'raw_strength_orig': orig.raw_bond_strength,
        'raw_strength_random': rand.raw_bond_strength,
        'strength_ratio_random': strength_ratio_rand,
        'strength_diff': strength_diff,

        # Secondary: vs shuffled (same vocabulary)
        'raw_coherence_shuf': full.raw_coherence,
        'coherence_ratio_shuf': coherence_ratio_shuf,

        # Success criteria checks
        'pass_coherence_vs_random': coherence_ratio_rand > 2.0,
        'pass_strength_vs_random': pass_strength,
        'pass_coherence_above_baseline': orig.raw_coherence > 0.3,

        # Legacy metrics
        'sigma_within_ratio': safe_ratio(orig.sigma_within, full.sigma_within),
        'autocorr_orig': orig.autocorr_lag1,

        # Full data
        'original': orig.as_dict(),
        'random_vocab': rand.as_dict(),
        'full_shuffle': full.as_dict(),
    }


def print_report(result: ShuffledResult):
    """Print a human-readable report of the sanity check."""
    eval_result = evaluate_result(result)

    print("=" * 60)
    print("SANITY CHECK: Coherent vs Random (THEORY.md physics)")
    print("=" * 60)

    print(f"\nBonds processed: {result.original_metrics.n_bonds}")
    print(f"Sentences: {result.original_metrics.n_sentences}")
    print(f"Coverage: {result.original_metrics.coverage:.1%}")

    print("\n" + "-" * 40)
    print("RAW COHERENCE C = cos(Δθ)")
    print("-" * 40)
    print(f"  Original text:    {eval_result['raw_coherence_orig']:.4f}")
    print(f"  Random vocab:     {eval_result['raw_coherence_random']:.4f}")
    print(f"  Ratio (vs rand):  {eval_result['coherence_ratio_random']:.2f}x")
    print(f"  Pass (>2.0x):     {'✓' if eval_result['pass_coherence_vs_random'] else '✗'}")
    print(f"  Pass (>0.3):      {'✓' if eval_result['pass_coherence_above_baseline'] else '✗'}")

    print("\n" + "-" * 40)
    print("BOND STRENGTH S = cos(Δθ) × exp(-|Δn|/kT)")
    print("-" * 40)
    print(f"  Original text:    {eval_result['raw_strength_orig']:.4f}")
    print(f"  Random vocab:     {eval_result['raw_strength_random']:.4f}")
    print(f"  Difference:       {eval_result['strength_diff']:+.4f}")
    print(f"  Pass (orig>0.1 & diff>0.2): {'✓' if eval_result['pass_strength_vs_random'] else '✗'}")

    print("\n" + "-" * 40)
    print("WITHIN-VOCABULARY COMPARISON")
    print("-" * 40)
    print(f"  Shuffled (same vocab): {eval_result['raw_coherence_shuf']:.4f}")
    print(f"  Ratio (vs shuf):       {eval_result['coherence_ratio_shuf']:.2f}x")

    print("\n" + "=" * 60)
    all_pass = (eval_result['pass_coherence_vs_random'] and
                eval_result['pass_strength_vs_random'] and
                eval_result['pass_coherence_above_baseline'])
    print(f"OVERALL: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    print("=" * 60)


# Quick test function
def quick_test():
    """Run a quick test with sample text."""
    sample_text = """
    The old man walked slowly through the dark forest.
    He carried a heavy burden on his tired shoulders.
    The ancient trees whispered mysterious secrets.
    A cold wind blew through the empty branches.
    The lonely traveler continued his difficult journey.
    Hope remained strong in his weary heart.
    """

    print("Running sanity check on sample text...")
    print(f"Text length: {len(sample_text)} chars\n")

    result = run_sanity_check(sample_text)
    print_report(result)

    return result


if __name__ == "__main__":
    quick_test()
