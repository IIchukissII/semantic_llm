#!/usr/bin/env python3
"""
Test THE_MAP theory prediction: variety → τ relationship.

Theory predicts:
- Abstract nouns (love, truth, beauty) → high variety → low τ
- Concrete nouns (chair, table, book) → low variety → high τ

We test this by:
1. Loading real τ data from semantic_vectors
2. Looking at attention entropy for different abstraction levels
3. Verifying the predicted relationship holds
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from semantic_bottleneck_v2 import SemanticBottleneckV2

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "semantic_vectors"


def load_data():
    """Load semantic data."""
    # Load nouns
    with open(DATA_DIR / "noun_vectors_16d.json") as f:
        noun_data = json.load(f)

    # Load τ levels
    with open(DATA_DIR / "tau_levels.json") as f:
        tau_data = json.load(f)

    return noun_data, tau_data


def analyze_tau_distribution(noun_data, tau_data):
    """Analyze the distribution of τ values."""
    print("=" * 70)
    print("TAU DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Get tau values
    if isinstance(tau_data, dict) and 'noun_tau' in tau_data:
        noun_tau = tau_data.get('noun_tau', tau_data)
    elif isinstance(tau_data, dict) and 'nouns' in tau_data:
        noun_tau = tau_data.get('nouns', tau_data)
    else:
        noun_tau = tau_data

    # Convert to list
    tau_values = []
    words_by_tau = {}

    for word, tau in noun_tau.items():
        if isinstance(tau, (int, float)):
            tau_values.append(tau)
            tau_int = int(round(tau))
            if tau_int not in words_by_tau:
                words_by_tau[tau_int] = []
            words_by_tau[tau_int].append(word)

    print(f"\nTotal nouns with τ: {len(tau_values)}")

    if tau_values:
        print(f"τ range: {min(tau_values):.2f} - {max(tau_values):.2f}")
        print(f"τ mean: {np.mean(tau_values):.2f}")
        print(f"τ std: {np.std(tau_values):.2f}")

        print("\nDistribution by τ level:")
        for tau_level in sorted(words_by_tau.keys()):
            words = words_by_tau[tau_level]
            examples = ', '.join(words[:5])
            print(f"  τ={tau_level}: {len(words)} words (e.g., {examples})")

    return noun_tau, words_by_tau


def test_theory_prediction(model, word2idx, words_by_tau):
    """
    Test THE_MAP prediction: variety ∝ 1/τ

    Abstract nouns should have high attention entropy (many adjectives describe them)
    Concrete nouns should have low attention entropy (few adjectives)
    """
    print("\n" + "=" * 70)
    print("THEORY TEST: VARIETY → TAU RELATIONSHIP")
    print("=" * 70)
    print("\nPrediction: High variety (entropy) → Low τ")
    print("           Low variety (entropy) → High τ\n")

    device = next(model.parameters()).device

    results_by_tau = {}

    for tau_level in sorted(words_by_tau.keys()):
        words = words_by_tau[tau_level]

        # Get words in vocabulary
        valid_words = [w for w in words if w in word2idx][:50]  # Sample up to 50

        if not valid_words:
            continue

        # Compute variety (entropy) for these words
        entropies = []

        for word in valid_words:
            word_id = torch.tensor([[word2idx[word]]], dtype=torch.long, device=device)
            word_type = torch.zeros_like(word_id)

            with torch.no_grad():
                semantic = model.encode(word_id, word_type)

            # Get variety (already normalized entropy)
            variety = semantic['variety'].item()
            entropies.append(variety)

        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)

        results_by_tau[tau_level] = {
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'n_words': len(valid_words)
        }

        print(f"τ={tau_level}: variety={mean_entropy:.4f} ± {std_entropy:.4f} (n={len(valid_words)})")

    # Check if correlation is negative (as theory predicts)
    tau_levels = list(results_by_tau.keys())
    entropies = [results_by_tau[t]['mean_entropy'] for t in tau_levels]

    if len(tau_levels) >= 2:
        correlation = np.corrcoef(tau_levels, entropies)[0, 1]
        print(f"\nCorrelation(τ, variety): {correlation:.4f}")
        print(f"Theory predicts: NEGATIVE correlation")
        print(f"Result: {'CONFIRMED ✓' if correlation < -0.3 else 'WEAK ⚠' if correlation < 0 else 'VIOLATED ✗'}")

    return results_by_tau


def main():
    print("=" * 70)
    print("THE_MAP THEORY TEST: VARIETY → TAU")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    noun_data, tau_data = load_data()

    # Analyze τ distribution
    noun_tau, words_by_tau = analyze_tau_distribution(noun_data, tau_data)

    if not words_by_tau:
        print("\nNo valid τ data found. Checking raw data structure...")
        print(f"tau_data type: {type(tau_data)}")
        if isinstance(tau_data, dict):
            print(f"tau_data keys: {list(tau_data.keys())[:10]}")
        return

    # Create vocabulary
    print("\nCreating vocabulary...")
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word in noun_data.get('vectors', noun_data).keys():
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    print(f"  Vocabulary size: {len(word2idx)}")

    # Create untrained model (to test initial structure)
    print("\nCreating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SemanticBottleneckV2(
        vocab_size=len(word2idx),
        embed_dim=128,
        hidden_dim=128,
        n_basis_adjectives=100
    ).to(device)

    # Set basis adjectives
    basis_ids = torch.arange(min(100, len(word2idx)), device=device)
    model.noun_encoder.set_basis_adjectives(basis_ids)

    # Test with UNTRAINED model (random initialization)
    print("\n[UNTRAINED MODEL - Random baseline]")
    test_theory_prediction(model, word2idx, words_by_tau)

    # Load trained model if available
    models_dir = Path(__file__).parent / "models_v2"
    checkpoint_path = models_dir / "semantic_bottleneck_v2_best.pt"

    if checkpoint_path.exists():
        print("\n" + "=" * 70)
        print("[TRAINED MODEL]")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model_trained = SemanticBottleneckV2(
                vocab_size=checkpoint['vocab_size'],
                embed_dim=checkpoint['embed_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                n_basis_adjectives=checkpoint['n_basis_adjectives']
            ).to(device)

            # Set basis adjectives
            basis_ids = torch.arange(checkpoint['n_basis_adjectives'], device=device)
            model_trained.noun_encoder.set_basis_adjectives(basis_ids)

            model_trained.load_state_dict(checkpoint['model_state_dict'])
            model_trained.eval()

            # Load trained vocabulary
            vocab_path = models_dir / "vocabulary.json"
            with open(vocab_path) as f:
                vocab_data = json.load(f)
            word2idx_trained = vocab_data['word2idx']

            test_theory_prediction(model_trained, word2idx_trained, words_by_tau)
        except Exception as e:
            print(f"Error loading trained model: {e}")
    else:
        print(f"\nNo trained model found at {checkpoint_path}")
        print("Run training first to test with learned representations.")


if __name__ == "__main__":
    main()
