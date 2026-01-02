#!/usr/bin/env python3
"""
Semantic Bottleneck: (A, S, τ) Encoding

Demonstrates that meaning can be encoded in just 3 numbers per word:
- A = Affirmation (PC1, 83.3% variance)
- S = Sacred (PC2, 11.7% variance)
- τ = Abstraction level

Verbs are 2D operators (ΔA, ΔS) that transform word coordinates.

Compression ratio: 100-250x vs standard word embeddings.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent paths
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM = _THIS_FILE.parent.parent.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.data_loader import DataLoader

# Principal Component Vectors (from exp10b)
PC1_AFFIRMATION = np.array([-0.448, -0.519, -0.118, -0.480, -0.534])
PC2_SACRED = np.array([-0.513, +0.128, -0.732, +0.420, +0.090])
DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


class SemanticBottleneck:
    """
    Encodes words as (A, S, τ) and verbs as (ΔA, ΔS).

    This is the minimal semantic representation - 3 numbers per word.
    """

    def __init__(self):
        self.loader = DataLoader()
        self.words = None
        self.verbs = None
        self.mean_verb = None

    def _load(self):
        if self.words is not None:
            return

        self.words = self.loader.load_word_vectors()
        self.verbs = self.loader.load_verb_operators()

        # Compute mean verb vector for relative operators
        verb_vectors = []
        for data in self.verbs.values():
            vec = np.array([data['vector'].get(d, 0) for d in DIMS])
            verb_vectors.append(vec)
        self.mean_verb = np.mean(verb_vectors, axis=0)

        print(f"[Bottleneck] Loaded {len(self.words)} words, {len(self.verbs)} verbs")

    def encode_word(self, word: str) -> tuple:
        """
        Encode a word as (A, S, τ).

        Returns:
            (A, S, τ) or None if word not found
        """
        self._load()

        if word not in self.words:
            return None

        data = self.words[word]
        if not data.get('j'):
            return None

        j_vec = np.array([data['j'].get(d, 0) for d in DIMS])
        A = np.dot(j_vec, PC1_AFFIRMATION)
        S = np.dot(j_vec, PC2_SACRED)
        tau = data.get('tau', 3.0)

        return (A, S, tau)

    def encode_verb(self, verb: str) -> tuple:
        """
        Encode a verb as (ΔA, ΔS) operator.

        Returns:
            (ΔA, ΔS) or None if verb not found
        """
        self._load()

        if verb not in self.verbs:
            return None

        vec = np.array([self.verbs[verb]['vector'].get(d, 0) for d in DIMS])
        delta = vec - self.mean_verb

        dA = np.dot(delta, PC1_AFFIRMATION)
        dS = np.dot(delta, PC2_SACRED)

        return (dA, dS)

    def apply_verb(self, word_coords: tuple, verb: str) -> tuple:
        """
        Apply verb operator to word coordinates.

        verb(word) = (A + ΔA, S + ΔS, τ)
        """
        verb_op = self.encode_verb(verb)
        if verb_op is None:
            return word_coords

        A, S, tau = word_coords
        dA, dS = verb_op

        return (A + dA, S + dS, tau)

    def encode_sentence(self, nouns: list, verbs: list) -> list:
        """
        Encode a sentence as chain of transformations.

        Returns list of (word, A, S, τ) tuples showing trajectory.
        """
        trajectory = []

        for noun in nouns:
            coords = self.encode_word(noun)
            if coords:
                trajectory.append((noun, *coords))

                # Apply each verb as transformation
                current = coords
                for verb in verbs:
                    new_coords = self.apply_verb(current, verb)
                    trajectory.append((f"{verb}({noun})", *new_coords))
                    current = new_coords

        return trajectory

    def distance(self, word1: str, word2: str) -> float:
        """
        Semantic distance in (A, S, τ) space.
        """
        c1 = self.encode_word(word1)
        c2 = self.encode_word(word2)

        if c1 is None or c2 is None:
            return float('inf')

        # Euclidean distance in 3D
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

    def find_nearest(self, coords: tuple, n: int = 5) -> list:
        """
        Find n nearest words to given coordinates.
        """
        self._load()

        distances = []
        for word, data in self.words.items():
            if not data.get('j'):
                continue
            c = self.encode_word(word)
            if c:
                dist = np.sqrt((coords[0]-c[0])**2 + (coords[1]-c[1])**2 + (coords[2]-c[2])**2)
                distances.append((word, dist, c))

        distances.sort(key=lambda x: x[1])
        return distances[:n]


def main():
    """Demonstrate the semantic bottleneck."""

    bottleneck = SemanticBottleneck()

    print("=" * 60)
    print("SEMANTIC BOTTLENECK: (A, S, τ) ENCODING")
    print("=" * 60)
    print()

    # 1. Show word encodings
    print("1. WORD ENCODINGS (3 numbers per word)")
    print("-" * 40)
    print(f"{'Word':<15} {'A':>8} {'S':>8} {'τ':>8}")
    print("-" * 40)

    test_words = ['truth', 'love', 'death', 'god', 'evil', 'beauty',
                  'chair', 'water', 'infinity', 'wisdom']
    for word in test_words:
        coords = bottleneck.encode_word(word)
        if coords:
            print(f"{word:<15} {coords[0]:>8.3f} {coords[1]:>8.3f} {coords[2]:>8.2f}")

    print()

    # 2. Show verb operators
    print("2. VERB OPERATORS (2 numbers per verb)")
    print("-" * 40)
    print(f"{'Verb':<12} {'ΔA':>8} {'ΔS':>8} {'Direction':<15}")
    print("-" * 40)

    test_verbs = ['love', 'hate', 'create', 'destroy', 'think', 'feel',
                  'live', 'die', 'give', 'take', 'help', 'hurt']
    for verb in test_verbs:
        op = bottleneck.encode_verb(verb)
        if op:
            direction = ""
            if abs(op[0]) > abs(op[1]):
                direction = "A+" if op[0] > 0 else "A-"
            else:
                direction = "S+" if op[1] > 0 else "S-"
            print(f"{verb:<12} {op[0]:>8.4f} {op[1]:>8.4f} {direction:<15}")

    print()

    # 3. Apply verb transformations
    print("3. VERB TRANSFORMATIONS")
    print("-" * 40)

    word = "truth"
    coords = bottleneck.encode_word(word)
    print(f"Start: {word} = (A={coords[0]:.3f}, S={coords[1]:.3f}, τ={coords[2]:.2f})")
    print()

    for verb in ['love', 'destroy', 'create', 'think']:
        new_coords = bottleneck.apply_verb(coords, verb)
        print(f"  + {verb}: (A={new_coords[0]:.3f}, S={new_coords[1]:.3f}, τ={new_coords[2]:.2f})")

        # Find what word this is closest to
        nearest = bottleneck.find_nearest(new_coords, n=3)
        neighbors = ", ".join([f"{w}({d:.2f})" for w, d, _ in nearest])
        print(f"    → nearest: {neighbors}")

    print()

    # 4. Compression ratio
    print("4. COMPRESSION RATIO")
    print("-" * 40)

    n_words = 16000
    n_verbs = 500

    standard_dim = 300  # Word2Vec
    bert_dim = 768      # BERT

    our_word_dim = 3    # (A, S, τ)
    our_verb_dim = 2    # (ΔA, ΔS)

    standard_size = n_words * standard_dim
    bert_size = n_words * bert_dim
    our_size = n_words * our_word_dim + n_verbs * our_verb_dim

    print(f"Vocabulary: {n_words} words, {n_verbs} verbs")
    print()
    print(f"Word2Vec (300d):   {standard_size:,} floats")
    print(f"BERT (768d):       {bert_size:,} floats")
    print(f"(A,S,τ) + (ΔA,ΔS): {our_size:,} floats")
    print()
    print(f"Compression vs Word2Vec: {standard_size/our_size:.0f}x")
    print(f"Compression vs BERT:     {bert_size/our_size:.0f}x")

    print()

    # 5. Semantic trajectory
    print("5. SEMANTIC TRAJECTORY: 'love transforms truth'")
    print("-" * 40)

    trajectory = bottleneck.encode_sentence(['truth'], ['love', 'think', 'feel'])
    for step in trajectory:
        print(f"  {step[0]:<20} → (A={step[1]:.3f}, S={step[2]:.3f}, τ={step[3]:.2f})")

    print()
    print("=" * 60)
    print("CONCLUSION: Meaning = 3 numbers. Verbs = 2 numbers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
