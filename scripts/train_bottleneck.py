#!/usr/bin/env python3
"""
Training script for Semantic Bottleneck V3 (Entropy-Based).

KEY CHANGES from V2:
1. τ computed from ENTROPY, not variety
2. Add thermodynamic losses:
   - One-bit law: H_adj - H_verb ≈ 1
   - Euler law: ln(H_adj) - ln(H_verb) ≈ 1/e
3. Being/Doing balance per τ level

Based on discoveries:
  ln(H_adj / H_verb) = 1/e = 0.3679 (±0.006)
  H_adj - H_verb = 1.08 bits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys
from collections import Counter, defaultdict
import random
import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from semantic_bottleneck_v2 import (
    SemanticBottleneckV2, AdjectiveEmbedding, NounAsAdjectiveCloud,
    VerbProjection, TauFromVariety, ContrastiveLoss, OrthogonalityLoss,
    PT1SaturationLoss
)

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SEMANTIC_DIR = DATA_DIR / "semantic_vectors"
BONDS_DIR = DATA_DIR / "bonds"
MODELS_DIR = Path(__file__).parent / "models_v3"

# Database config
DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}

# Thermodynamic constants discovered
ONE_OVER_E = 1 / np.e  # 0.3679
ONE_BIT = 1.0  # H_adj - H_verb average


def shannon_entropy(counts: dict) -> float:
    """Compute Shannon entropy H = -Σ p log₂ p"""
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


def normalized_entropy(counts: dict) -> float:
    """Compute H / H_max = H / log₂(n)"""
    if not counts or len(counts) <= 1:
        return 0.0

    h = shannon_entropy(counts)
    h_max = np.log2(len(counts))
    return h / h_max if h_max > 0 else 0.0


def compute_tau_from_entropy(h_norm: float) -> float:
    """
    Compute τ from normalized entropy.

    τ = 1 + 5 × (1 - H_norm)

    High entropy (H_norm → 1) → τ → 1 (abstract)
    Low entropy (H_norm → 0) → τ → 6 (concrete)
    """
    return 1.0 + 5.0 * (1.0 - h_norm)


class EntropyTauLoss(nn.Module):
    """
    Loss that enforces τ = f(entropy), not f(variety).

    τ_pred should match τ_entropy = 1 + 5 × (1 - H_norm)
    """

    def forward(self, tau_pred: torch.Tensor, h_adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau_pred: Predicted τ [batch]
            h_adj_norm: Normalized adjective entropy [batch]
        """
        tau_target = 1.0 + 5.0 * (1.0 - h_adj_norm)
        return F.mse_loss(tau_pred, tau_target)


class OneBitLaw(nn.Module):
    """
    Loss that enforces: H_adj - H_verb ≈ 1 bit.

    Being > Doing by 1 bit on average.
    """

    def __init__(self, target: float = 1.0):
        super().__init__()
        self.target = target

    def forward(self, h_adj: torch.Tensor, h_verb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_adj: Adjective entropy [batch]
            h_verb: Verb entropy [batch]
        """
        delta = h_adj - h_verb
        return F.mse_loss(delta, torch.full_like(delta, self.target))


class EulerLaw(nn.Module):
    """
    Loss that enforces: ln(H_adj) - ln(H_verb) ≈ 1/e.

    The logarithmic ratio equals Euler's constant inverse.
    """

    def __init__(self, target: float = ONE_OVER_E):
        super().__init__()
        self.target = target

    def forward(self, h_adj: torch.Tensor, h_verb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_adj: Adjective entropy [batch] (must be > 0)
            h_verb: Verb entropy [batch] (must be > 0)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        log_diff = torch.log(h_adj + eps) - torch.log(h_verb + eps)
        return F.mse_loss(log_diff, torch.full_like(log_diff, self.target))


class BeingDoingBalance(nn.Module):
    """
    Loss that enforces correct Being/Doing ratio per τ level.

    From data:
        τ=1: H_adj/H_verb = 1.79
        τ=3: H_adj/H_verb = 1.17
        τ=6: H_adj/H_verb ≈ 0

    Linear model: ratio ≈ 2.0 - 0.3 × τ
    """

    def forward(self, h_adj: torch.Tensor, h_verb: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_adj: Adjective entropy [batch]
            h_verb: Verb entropy [batch]
            tau: τ level [batch]
        """
        # Expected ratio: linear function of τ
        # At τ=1: ratio ≈ 1.8, at τ=6: ratio ≈ 0
        target_ratio = torch.clamp(2.0 - 0.35 * tau, min=0.0)

        # Actual ratio (with safeguard)
        actual_ratio = h_adj / (h_verb + 1e-6)
        actual_ratio = torch.clamp(actual_ratio, max=5.0)  # Cap for stability

        return F.mse_loss(actual_ratio, target_ratio)


class SemanticVocabularyV3:
    """Vocabulary with entropy-based τ."""

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}

        # Semantic data
        self.noun_vectors = {}  # word -> 16D
        self.verb_vectors = {}  # word -> 6D

        # Entropy-based data (NEW)
        self.noun_entropy = {}      # word -> (H_adj, H_adj_norm, variety)
        self.noun_verb_entropy = {} # word -> (H_verb, H_verb_norm, variety)
        self.noun_tau_entropy = {}  # word -> τ from entropy

        # Word types
        self.nouns = set()
        self.verbs = set()
        self.adjectives = set()

    def load_semantic_data(self):
        """Load pre-computed semantic vectors."""
        print("Loading semantic data...")

        # Load noun vectors
        if (SEMANTIC_DIR / "noun_vectors_16d.json").exists():
            with open(SEMANTIC_DIR / "noun_vectors_16d.json") as f:
                data = json.load(f)
                self.dimensions = data['dimensions']
                for word, vec in data['vectors'].items():
                    self.noun_vectors[word] = np.array(vec, dtype=np.float32)
                    self.nouns.add(word)

        # Load verb vectors
        if (SEMANTIC_DIR / "verb_vectors_6d.json").exists():
            with open(SEMANTIC_DIR / "verb_vectors_6d.json") as f:
                data = json.load(f)
                self.verb_dimensions = data['dimensions']
                for word, vec in data['vectors'].items():
                    self.verb_vectors[word] = np.array(vec, dtype=np.float32)
                    self.verbs.add(word)

        print(f"  Loaded {len(self.noun_vectors)} nouns")
        print(f"  Loaded {len(self.verb_vectors)} verbs")

    def compute_entropy_from_db(self):
        """Compute entropy and entropy-based τ from database."""
        print("\nComputing entropy from database...")

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Load noun-adj profiles
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

        # Load noun-verb profiles
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

        # Compute entropy for each noun
        for noun in self.nouns:
            if noun in noun_adj:
                adj_counts = dict(noun_adj[noun])
                h_adj = shannon_entropy(adj_counts)
                h_adj_norm = normalized_entropy(adj_counts)
                variety = len(adj_counts)

                self.noun_entropy[noun] = (h_adj, h_adj_norm, variety)

                # Compute τ from entropy
                tau = compute_tau_from_entropy(h_adj_norm)
                self.noun_tau_entropy[noun] = tau

            if noun in noun_verb:
                verb_counts = dict(noun_verb[noun])
                h_verb = shannon_entropy(verb_counts)
                h_verb_norm = normalized_entropy(verb_counts)
                variety_v = len(verb_counts)

                self.noun_verb_entropy[noun] = (h_verb, h_verb_norm, variety_v)

        print(f"  Computed entropy for {len(self.noun_entropy)} nouns")
        print(f"  Computed verb entropy for {len(self.noun_verb_entropy)} nouns")

    def build_vocabulary(self, min_freq: int = 2):
        """Build vocabulary from semantic words."""
        for word in self.noun_vectors:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        for word in self.verb_vectors:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"  Vocabulary size: {len(self.word2idx)}")

    def get_noun_target(self, word: str) -> tuple:
        """Get target vector, τ (from entropy), and entropies."""
        vec = self.noun_vectors.get(word)

        # Entropy-based τ
        tau = self.noun_tau_entropy.get(word, 3.0)

        # Entropies
        h_adj, h_adj_norm, _ = self.noun_entropy.get(word, (0.0, 0.5, 0))
        h_verb, h_verb_norm, _ = self.noun_verb_entropy.get(word, (0.0, 0.5, 0))

        return vec, tau, h_adj, h_verb, h_adj_norm

    def get_verb_target(self, word: str) -> np.ndarray:
        """Get target 6D vector for a verb."""
        return self.verb_vectors.get(word)


class SemanticWordDatasetV3(Dataset):
    """Dataset with entropy data."""

    def __init__(self, vocab: SemanticVocabularyV3, word_type: str = 'noun'):
        self.vocab = vocab
        self.word_type = word_type

        if word_type == 'noun':
            # Only include nouns with entropy data
            self.words = [w for w in vocab.noun_vectors.keys()
                          if w in vocab.word2idx and w in vocab.noun_entropy]
        elif word_type == 'verb':
            self.words = [w for w in vocab.verb_vectors.keys() if w in vocab.word2idx]
        else:
            raise ValueError(f"Unknown word type: {word_type}")

        print(f"  {len(self.words)} {word_type}s in dataset")

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        word_idx = self.vocab.word2idx[word]

        if self.word_type == 'noun':
            vec, tau, h_adj, h_verb, h_adj_norm = self.vocab.get_noun_target(word)
            return word_idx, vec, tau, h_adj, h_verb, h_adj_norm
        else:
            target = self.vocab.get_verb_target(word)
            return word_idx, target


def collate_nouns_v3(batch):
    """Collate noun batch with entropy data."""
    word_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
    targets = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32)
    taus = torch.tensor([x[2] for x in batch], dtype=torch.float32)
    h_adjs = torch.tensor([x[3] for x in batch], dtype=torch.float32)
    h_verbs = torch.tensor([x[4] for x in batch], dtype=torch.float32)
    h_adj_norms = torch.tensor([x[5] for x in batch], dtype=torch.float32)
    return word_ids, targets, taus, h_adjs, h_verbs, h_adj_norms


def collate_verbs_v3(batch):
    """Collate verb batch."""
    word_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
    targets = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32)
    return word_ids, targets


def train_v3(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.009,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    n_basis_adjectives: int = 100,
    ortho_weight: float = 4.7,
    entropy_tau_weight: float = 1.0,  # NEW: weight for entropy-τ loss
    one_bit_weight: float = 0.5,      # NEW: weight for 1-bit law
    euler_weight: float = 0.5,        # NEW: weight for 1/e law
    optimizer_type: str = 'SGD',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train the V3 semantic bottleneck with thermodynamic losses."""

    print("=" * 70)
    print("SEMANTIC BOTTLENECK V3 TRAINING (Entropy-Based)")
    print("=" * 70)
    print(f"\nThermodynamic constants:")
    print(f"  1/e = {ONE_OVER_E:.4f}")
    print(f"  1-bit law = {ONE_BIT:.4f}")

    # Load vocabulary and semantic data
    vocab = SemanticVocabularyV3()
    vocab.load_semantic_data()
    vocab.compute_entropy_from_db()  # NEW: compute entropy
    vocab.build_vocabulary()

    # Create datasets
    print("\nCreating datasets...")
    noun_dataset = SemanticWordDatasetV3(vocab, 'noun')
    verb_dataset = SemanticWordDatasetV3(vocab, 'verb')

    noun_loader = DataLoader(
        noun_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_nouns_v3
    )
    verb_loader = DataLoader(
        verb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_verbs_v3
    )

    # Create model
    print("\nCreating model...")
    model = SemanticBottleneckV2(
        vocab_size=len(vocab.word2idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_basis_adjectives=n_basis_adjectives
    ).to(device)

    # Set basis adjectives
    basis_ids = torch.tensor(
        [vocab.word2idx[w] for w in list(vocab.noun_vectors.keys())[:n_basis_adjectives]],
        dtype=torch.long
    )
    model.noun_encoder.set_basis_adjectives(basis_ids)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Loss functions
    mse_loss = nn.MSELoss()
    contrastive_loss = ContrastiveLoss(margin=0.5)
    ortho_loss = OrthogonalityLoss()

    # NEW: Thermodynamic losses
    entropy_tau_loss = EntropyTauLoss()
    one_bit_loss = OneBitLaw(target=ONE_BIT)
    euler_loss = EulerLaw(target=ONE_OVER_E)

    # Optimizer
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print("\nTraining...")
    MODELS_DIR.mkdir(exist_ok=True)

    history = {
        'noun_loss': [], 'verb_loss': [], 'tau_loss': [],
        'contrastive_loss': [], 'ortho_loss': [],
        'entropy_tau_loss': [], 'one_bit_loss': [], 'euler_loss': [],
        'variety_mean': [], 'total_loss': []
    }

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: [] for k in history}

        # Train on nouns
        pbar = tqdm(noun_loader, desc=f"Epoch {epoch+1}/{epochs} [Nouns]")
        for word_ids, targets, taus, h_adjs, h_verbs, h_adj_norms in pbar:
            word_ids = word_ids.to(device)
            targets = targets.to(device)
            taus = taus.to(device)
            h_adjs = h_adjs.to(device)
            h_verbs = h_verbs.to(device)
            h_adj_norms = h_adj_norms.to(device)

            optimizer.zero_grad()

            # Forward pass
            word_ids_seq = word_ids.unsqueeze(1)
            word_types = torch.zeros_like(word_ids_seq)

            semantic = model.encode(word_ids_seq, word_types)

            j = semantic['j']
            i = semantic['i']
            pred_semantic = torch.cat([j, i], dim=-1)
            pred_tau = semantic['tau']
            variety = semantic['variety']

            # Standard losses
            noun_loss = mse_loss(pred_semantic, targets)
            ortho = ortho_loss(j, i)

            # NEW: Entropy-based τ loss (instead of variety-based)
            entropy_tau = entropy_tau_loss(pred_tau, h_adj_norms)

            # NEW: Thermodynamic losses
            valid_entropy = (h_adjs > 0.1) & (h_verbs > 0.1)
            if valid_entropy.sum() > 0:
                one_bit = one_bit_loss(h_adjs[valid_entropy], h_verbs[valid_entropy])
                euler = euler_loss(h_adjs[valid_entropy], h_verbs[valid_entropy])
            else:
                one_bit = torch.tensor(0.0, device=device)
                euler = torch.tensor(0.0, device=device)

            # Total loss
            loss = (noun_loss +
                    entropy_tau_weight * entropy_tau +
                    ortho_weight * ortho +
                    one_bit_weight * one_bit +
                    euler_weight * euler)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Record losses
            epoch_losses['noun_loss'].append(noun_loss.item())
            epoch_losses['tau_loss'].append(entropy_tau.item())
            epoch_losses['ortho_loss'].append(ortho.item())
            epoch_losses['entropy_tau_loss'].append(entropy_tau.item())
            epoch_losses['one_bit_loss'].append(one_bit.item())
            epoch_losses['euler_loss'].append(euler.item())
            epoch_losses['variety_mean'].append(variety.mean().item())

            pbar.set_postfix({
                'noun': f"{noun_loss.item():.4f}",
                'e-tau': f"{entropy_tau.item():.4f}",
                '1bit': f"{one_bit.item():.4f}",
                'euler': f"{euler.item():.4f}"
            })

        # Train on verbs
        pbar = tqdm(verb_loader, desc=f"Epoch {epoch+1}/{epochs} [Verbs]")
        for word_ids, targets in pbar:
            word_ids = word_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            word_ids_seq = word_ids.unsqueeze(1)
            verb_proj = model.verb_projection(word_ids_seq).squeeze(1)
            verb_loss = mse_loss(verb_proj, targets)

            verb_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['verb_loss'].append(verb_loss.item())

        # Epoch summary
        scheduler.step()

        mean_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        total = sum(mean_losses.values())
        mean_losses['total_loss'] = total

        for k, v in mean_losses.items():
            history[k].append(v)

        print(f"\nEpoch {epoch+1}: " +
              f"noun={mean_losses['noun_loss']:.4f}, " +
              f"verb={mean_losses['verb_loss']:.4f}, " +
              f"e-tau={mean_losses['entropy_tau_loss']:.4f}, " +
              f"1bit={mean_losses['one_bit_loss']:.4f}, " +
              f"euler={mean_losses['euler_loss']:.4f}, " +
              f"ortho={mean_losses['ortho_loss']:.4f}")

        # Save best model
        if total < best_loss:
            best_loss = total
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab.word2idx),
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'n_basis_adjectives': n_basis_adjectives,
                'history': history,
                'best_loss': best_loss,
                'version': 'v3_entropy'
            }, MODELS_DIR / "semantic_bottleneck_v3_best.pt")

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'vocab_size': len(vocab.word2idx),
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'n_basis_adjectives': n_basis_adjectives,
        'history': history,
        'version': 'v3_entropy'
    }, MODELS_DIR / "semantic_bottleneck_v3_final.pt")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE (V3 Entropy-Based)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {MODELS_DIR}")
    print("=" * 70)

    return model, vocab, history


if __name__ == "__main__":
    train_v3(
        epochs=50,
        batch_size=128,
        lr=0.009,
        embed_dim=128,
        hidden_dim=128,
        n_basis_adjectives=100,
        ortho_weight=4.7,
        entropy_tau_weight=1.0,
        one_bit_weight=0.5,
        euler_weight=0.5,
        optimizer_type='SGD'
    )
