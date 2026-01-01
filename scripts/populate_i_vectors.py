#!/usr/bin/env python3
"""
Populate i-vectors (11D surface space) for all words.

The i-space dimensions are:
  truth, freedom, meaning, order, peace, power, nature, time, knowledge, self, society

Method:
  1. Load word embeddings (sentence-transformers)
  2. Get embeddings for seed words (the 11 dimension names)
  3. Project all words onto these 11 dimensions using cosine similarity
  4. Update the CSV with new i-vectors

This creates orthogonal i-vectors to the existing j-vectors (5D transcendentals).
"""

import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
import sys

# Try to use sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    print("Install sentence-transformers: pip install sentence-transformers")
    HAS_SENTENCE_TRANSFORMERS = False
    sys.exit(1)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_FILE = DATA_DIR / "csv" / "word_vectors.csv"
OUTPUT_FILE = DATA_DIR / "csv" / "word_vectors_with_i.csv"

# I-space dimensions (surface)
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']


def load_embedder():
    """Load sentence-transformer model."""
    print("Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def compute_dimension_embeddings(embedder):
    """Get embeddings for each i-dimension concept."""
    print("Computing dimension embeddings...")

    # Use extended context for richer embeddings
    dim_contexts = {
        'truth': 'truth honesty reality fact genuine authentic',
        'freedom': 'freedom liberty independence autonomy free',
        'meaning': 'meaning purpose significance sense understanding',
        'order': 'order structure system organization arrangement',
        'peace': 'peace calm serenity harmony tranquility',
        'power': 'power strength force energy control authority',
        'nature': 'nature natural environment world earth life',
        'time': 'time temporal moment duration age period',
        'knowledge': 'knowledge wisdom understanding information learning',
        'self': 'self identity individual person ego soul',
        'society': 'society community people group culture civilization',
    }

    dim_embeddings = {}
    for dim in I_DIMS:
        context = dim_contexts.get(dim, dim)
        emb = embedder.encode(context, normalize_embeddings=True)
        dim_embeddings[dim] = emb

    return dim_embeddings


def project_word(word_emb, dim_embeddings):
    """Project word embedding onto i-space dimensions."""
    i_vector = []
    for dim in I_DIMS:
        # Cosine similarity (embeddings are already normalized)
        sim = np.dot(word_emb, dim_embeddings[dim])
        i_vector.append(float(sim))
    return i_vector


def populate_i_vectors():
    """Main function to populate i-vectors."""
    if not CSV_FILE.exists():
        print(f"Error: {CSV_FILE} not found")
        return

    embedder = load_embedder()
    dim_embeddings = compute_dimension_embeddings(embedder)

    # Load existing data
    print(f"Loading {CSV_FILE}...")
    rows = []
    fieldnames = None

    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} words")

    # Get all words to embed
    words = [row['word'] for row in rows]

    # Compute embeddings in batches
    print("Computing word embeddings...")
    batch_size = 256
    word_embeddings = {}

    for i in tqdm(range(0, len(words), batch_size)):
        batch_words = words[i:i+batch_size]
        batch_emb = embedder.encode(batch_words, normalize_embeddings=True)
        for w, emb in zip(batch_words, batch_emb):
            word_embeddings[w] = emb

    # Project to i-space
    print("Projecting to i-space...")
    for row in tqdm(rows):
        word = row['word']
        if word in word_embeddings:
            i_vec = project_word(word_embeddings[word], dim_embeddings)
            for dim, val in zip(I_DIMS, i_vec):
                row[f'i_{dim}'] = f"{val:.6f}"

    # Write output
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done!")

    # Show sample
    print("\nSample i-vectors:")
    samples = ['truth', 'love', 'chair', 'freedom', 'god']
    for word in samples:
        row = next((r for r in rows if r['word'] == word), None)
        if row:
            i_vals = [float(row[f'i_{d}']) for d in I_DIMS]
            i_norm = np.linalg.norm(i_vals)
            print(f"  {word}: ||i|| = {i_norm:.3f}")
            # Top 3 dimensions
            sorted_dims = sorted(zip(I_DIMS, i_vals), key=lambda x: -x[1])[:3]
            print(f"    Top: {', '.join(f'{d}={v:.2f}' for d, v in sorted_dims)}")


if __name__ == '__main__':
    populate_i_vectors()
