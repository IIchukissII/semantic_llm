#!/usr/bin/env python3
"""
Generate Semantic Bottleneck Data Files

Computes and saves:
1. semantic_coordinates.json: {word: [A, S, τ]} for all words
2. verb_operators_2d.json: {verb: [ΔA, ΔS]} for all verbs

Run once to generate pre-computed data for O(1) lookup.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent path for imports
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM = _THIS_FILE.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.data_loader import DataLoader

# Principal Component Vectors (validated in exp10b)
PC1_AFFIRMATION = np.array([-0.448, -0.519, -0.118, -0.480, -0.534])
PC2_SACRED = np.array([-0.513, +0.128, -0.732, +0.420, +0.090])
DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


def generate_word_coordinates(loader: DataLoader) -> dict:
    """Generate (A, S, τ) coordinates for all words."""

    words = loader.load_word_vectors()

    coordinates = {}
    skipped = 0

    for word, data in words.items():
        # Skip words without j-vector
        if not data.get('j'):
            skipped += 1
            continue

        # Extract j-vector
        j_vec = np.array([data['j'].get(d, 0) for d in DIMS])

        # Project to (A, S)
        A = float(np.dot(j_vec, PC1_AFFIRMATION))
        S = float(np.dot(j_vec, PC2_SACRED))

        # Get τ (abstraction level)
        tau = float(data.get('tau', 3.0))

        # Store as [A, S, τ] array for compact JSON
        coordinates[word] = [round(A, 4), round(S, 4), round(tau, 2)]

    print(f"[Generate] Processed {len(coordinates)} words, skipped {skipped}")
    return coordinates


def generate_verb_operators(loader: DataLoader) -> tuple:
    """Generate (ΔA, ΔS) operators for all verbs."""

    verbs = loader.load_verb_operators()

    # Compute mean verb vector
    verb_vectors = []
    for data in verbs.values():
        vec = np.array([data['vector'].get(d, 0) for d in DIMS])
        verb_vectors.append(vec)

    mean_verb = np.mean(verb_vectors, axis=0)

    operators = {}

    for verb, data in verbs.items():
        # Get verb vector
        vec = np.array([data['vector'].get(d, 0) for d in DIMS])

        # Compute delta from mean
        delta = vec - mean_verb

        # Project to (ΔA, ΔS)
        dA = float(np.dot(delta, PC1_AFFIRMATION))
        dS = float(np.dot(delta, PC2_SACRED))

        # Store as [ΔA, ΔS] array
        operators[verb] = [round(dA, 4), round(dS, 4)]

    print(f"[Generate] Processed {len(operators)} verbs")
    return operators, mean_verb.tolist()


def save_coordinates(coordinates: dict, output_path: Path):
    """Save word coordinates to JSON."""

    data = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "description": "Semantic coordinates (A, S, τ) for each word",
        "pc1_affirmation": PC1_AFFIRMATION.tolist(),
        "pc2_sacred": PC2_SACRED.tolist(),
        "dimensions": DIMS,
        "n_words": len(coordinates),
        "words": coordinates
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Generate] Saved {output_path.name}: {len(coordinates)} words, {size_mb:.2f} MB")


def save_operators(operators: dict, mean_verb: list, output_path: Path):
    """Save verb operators to JSON."""

    data = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "description": "Verb operators (ΔA, ΔS) relative to mean verb",
        "mean_verb_5d": mean_verb,
        "pc1_affirmation": PC1_AFFIRMATION.tolist(),
        "pc2_sacred": PC2_SACRED.tolist(),
        "n_verbs": len(operators),
        "operators": operators
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"[Generate] Saved {output_path.name}: {len(operators)} verbs, {size_kb:.1f} KB")


def main():
    """Generate all bottleneck data files."""

    print("=" * 60)
    print("SEMANTIC BOTTLENECK DATA GENERATION")
    print("=" * 60)
    print()

    # Output directory
    output_dir = _SEMANTIC_LLM / "data" / "json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source data
    loader = DataLoader()

    # Generate word coordinates
    print("Generating word coordinates...")
    coordinates = generate_word_coordinates(loader)
    coord_path = output_dir / "semantic_coordinates.json"
    save_coordinates(coordinates, coord_path)
    print()

    # Generate verb operators
    print("Generating verb operators...")
    operators, mean_verb = generate_verb_operators(loader)
    ops_path = output_dir / "verb_operators_2d.json"
    save_operators(operators, mean_verb, ops_path)
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Word coordinates: {len(coordinates)} words → {coord_path.name}")
    print(f"Verb operators:   {len(operators)} verbs → {ops_path.name}")
    print()

    # Sample outputs
    print("Sample word coordinates:")
    for word in ['truth', 'love', 'death', 'god', 'chair']:
        if word in coordinates:
            A, S, tau = coordinates[word]
            print(f"  {word:12} → (A={A:+.3f}, S={S:+.3f}, τ={tau:.2f})")

    print()
    print("Sample verb operators:")
    for verb in ['love', 'hate', 'create', 'destroy', 'think']:
        if verb in operators:
            dA, dS = operators[verb]
            print(f"  {verb:12} → (ΔA={dA:+.4f}, ΔS={dS:+.4f})")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
