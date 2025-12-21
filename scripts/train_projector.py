#!/usr/bin/env python3
"""
Train SemanticProjector to map embeddings -> 16D semantic space.

Uses pre-computed word vectors from our corpus as supervision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Try to use sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Note: pip install sentence-transformers for better embeddings")

from semantic_core import SemanticProjector, SemanticVocabulary, J_DIMS, I_DIMS


class WordEmbeddingDataset(Dataset):
    """Dataset of word embeddings with 16D target coords."""

    def __init__(self, vocab: SemanticVocabulary, embedder=None, max_words: int = 10000):
        self.vocab = vocab
        self.words = []
        self.embeddings = []
        self.targets = []  # 16D coords
        self.taus = []

        # Use simple embeddings if no embedder provided
        if embedder is None:
            print("Using random embeddings (for structure test)")
            embed_dim = 768
            for word in list(vocab.vectors.keys())[:max_words]:
                coords = vocab.get_coords(word)
                if coords:
                    self.words.append(word)
                    self.embeddings.append(np.random.randn(embed_dim).astype(np.float32))
                    self.targets.append(coords.vector.astype(np.float32))
                    self.taus.append(coords.tau)
        else:
            print(f"Computing embeddings for {min(len(vocab.vectors), max_words)} words...")
            words_list = list(vocab.vectors.keys())[:max_words]

            # Batch compute embeddings
            batch_size = 256
            for i in tqdm(range(0, len(words_list), batch_size)):
                batch_words = words_list[i:i+batch_size]
                batch_emb = embedder.encode(batch_words, convert_to_numpy=True)

                for word, emb in zip(batch_words, batch_emb):
                    coords = vocab.get_coords(word)
                    if coords:
                        self.words.append(word)
                        self.embeddings.append(emb.astype(np.float32))
                        self.targets.append(coords.vector.astype(np.float32))
                        self.taus.append(coords.tau)

        self.embeddings = np.array(self.embeddings)
        self.targets = np.array(self.targets)
        self.taus = np.array(self.taus, dtype=np.float32)

        print(f"Dataset: {len(self.words)} words, embed_dim={self.embeddings.shape[1]}")

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx]),
            torch.tensor(self.targets[idx]),
            torch.tensor(self.taus[idx])
        )


def train_projector(
    n_words: int = 5000,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    use_gpu: bool = True,
    use_embedder: bool = True
):
    """Train the semantic projector."""

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    vocab = SemanticVocabulary()
    print(f"Loaded vocabulary with {len(vocab.vectors)} words")

    # Get embedder
    embedder = None
    embed_dim = 768

    if use_embedder and HAS_SENTENCE_TRANSFORMERS:
        print("Loading sentence-transformers model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 384D, fast
        embed_dim = 384
    elif use_embedder:
        print("sentence-transformers not available, using random embeddings")

    # Create dataset
    dataset = WordEmbeddingDataset(vocab, embedder, max_words=n_words)
    embed_dim = dataset.embeddings.shape[1]

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = SemanticProjector(embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss weights
    coord_weight = 1.0
    tau_weight = 0.5

    print(f"\nTraining on {train_size} words, validating on {val_size}")
    print(f"Embed dim: {embed_dim}, Output: 16D + tau")

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        for emb, target, tau_target in train_loader:
            emb = emb.to(device)
            target = target.to(device)
            tau_target = tau_target.to(device)

            optimizer.zero_grad()

            # Forward
            tau_pred, i_pred, j_pred = model(emb)
            pred = torch.cat([j_pred, i_pred], dim=-1)

            # Loss
            coord_loss = nn.functional.mse_loss(pred, target)
            tau_loss = nn.functional.mse_loss(tau_pred, tau_target)
            loss = coord_weight * coord_loss + tau_weight * tau_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for emb, target, tau_target in val_loader:
                emb = emb.to(device)
                target = target.to(device)
                tau_target = tau_target.to(device)

                tau_pred, i_pred, j_pred = model(emb)
                pred = torch.cat([j_pred, i_pred], dim=-1)

                coord_loss = nn.functional.mse_loss(pred, target)
                tau_loss = nn.functional.mse_loss(tau_pred, tau_target)
                loss = coord_weight * coord_loss + tau_weight * tau_loss

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Save model
    save_path = Path(__file__).parent / "models"
    save_path.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'embed_dim': embed_dim,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, save_path / "semantic_projector.pt")

    print(f"\nModel saved to {save_path / 'semantic_projector.pt'}")

    # Plot training
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Semantic Projector Training')
    plt.legend()
    plt.savefig(save_path / 'training_curve.png')
    plt.close()

    return model, dataset


def test_projector(model, dataset, n_samples: int = 10):
    """Test the trained projector."""
    print("\n" + "=" * 60)
    print("PROJECTOR TEST")
    print("=" * 60)

    model.eval()
    device = next(model.parameters()).device

    # Sample some words
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    print(f"\n{'Word':<15} {'True tau':<10} {'Pred tau':<10} {'Coord dist':<12}")
    print("-" * 50)

    total_dist = 0
    total_tau_err = 0

    for idx in indices:
        emb, target, tau_true = dataset[idx]
        emb = emb.unsqueeze(0).to(device)

        with torch.no_grad():
            tau_pred, i_pred, j_pred = model(emb)
            pred = torch.cat([j_pred, i_pred], dim=-1)

        # Compute errors
        coord_dist = torch.norm(pred - target.to(device)).item()
        tau_err = abs(tau_pred.item() - tau_true.item())

        word = dataset.words[idx]
        print(f"{word:<15} {tau_true.item():<10.2f} {tau_pred.item():<10.2f} {coord_dist:<12.4f}")

        total_dist += coord_dist
        total_tau_err += tau_err

    print("-" * 50)
    print(f"{'Average:':<15} {'':<10} {'':<10} {total_dist/n_samples:<12.4f}")
    print(f"Avg tau error: {total_tau_err/n_samples:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC PROJECTOR TRAINING")
    print("=" * 60)

    # Train with real embeddings if available
    model, dataset = train_projector(
        n_words=5000,
        epochs=50,
        batch_size=128,
        use_embedder=True
    )

    # Test
    test_projector(model, dataset)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
