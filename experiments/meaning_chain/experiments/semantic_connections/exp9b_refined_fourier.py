#!/usr/bin/env python3
"""
Experiment 9b: Refined Fourier Navigation
==========================================

Improvements over exp9:
1. Cosine similarity instead of Euclidean distance
2. Proper j/i weighting based on theory
3. Use validated τ constants for verb classification
4. Better semantic coherence in navigation
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _EXPERIMENT_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']

# Validated constants from THEORY.md
TAU_NOUN = 1.888
TAU_ADJ = 2.123
TAU_VERB = 2.259
TAU_LIFT = 3.977
TAU_GROUND = 2.080
TAU_NEUTRAL = 1.812
EXPANSION_SLOPE = 0.30
EXPANSION_INTERCEPT = 0.70


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


class RefinedSemanticFourier:
    """
    Refined Fourier semantics with proper weighting.
    """

    def __init__(self, vectors: Dict):
        self.vectors = vectors
        self._build_index()

    def _build_index(self):
        """Build vector index for fast lookup."""
        self.j_matrix = []
        self.i_matrix = []
        self.tau_list = []
        self.word_list = []
        self.word_types = []

        for word, data in self.vectors.items():
            if not data.get('j') or not data.get('i') or not data.get('tau'):
                continue

            j = np.array([data['j'].get(d, 0) for d in J_DIMS])
            i = np.array([data['i'].get(d, 0) for d in I_DIMS])

            self.j_matrix.append(j)
            self.i_matrix.append(i)
            self.tau_list.append(data['tau'])
            self.word_list.append(word)
            self.word_types.append(data.get('word_type', 'unknown'))

        self.j_matrix = np.array(self.j_matrix)
        self.i_matrix = np.array(self.i_matrix)
        self.tau_list = np.array(self.tau_list)

        # Compute norms
        self.j_norms = np.linalg.norm(self.j_matrix, axis=1)
        self.i_norms = np.linalg.norm(self.i_matrix, axis=1)

    def get_word_index(self, word: str) -> Optional[int]:
        """Get index for a word."""
        try:
            return self.word_list.index(word)
        except ValueError:
            return None

    def semantic_similarity(self, word1: str, word2: str,
                           j_weight: float = 0.6, tau_weight: float = 0.2) -> Optional[float]:
        """
        Compute semantic similarity using Fourier components.

        j_weight: Weight for j-space similarity (transcendental)
        tau_weight: Weight for τ similarity
        i_weight: 1 - j_weight - tau_weight (surface)
        """
        idx1 = self.get_word_index(word1)
        idx2 = self.get_word_index(word2)

        if idx1 is None or idx2 is None:
            return None

        # j-space similarity (cosine)
        j_sim = cosine_similarity(self.j_matrix[idx1], self.j_matrix[idx2])

        # i-space similarity (cosine)
        i_sim = cosine_similarity(self.i_matrix[idx1], self.i_matrix[idx2])

        # τ similarity (Gaussian)
        tau_diff = abs(self.tau_list[idx1] - self.tau_list[idx2])
        tau_sim = np.exp(-tau_diff / 0.5)

        i_weight = 1.0 - j_weight - tau_weight

        return j_weight * j_sim + i_weight * i_sim + tau_weight * tau_sim

    def find_similar(self, word: str, top_k: int = 10,
                    tau_filter: Tuple[float, float] = None,
                    word_type_filter: str = None) -> List[Tuple[str, float]]:
        """
        Find similar words using refined similarity.
        """
        idx = self.get_word_index(word)
        if idx is None:
            return []

        j_query = self.j_matrix[idx]
        i_query = self.i_matrix[idx]
        tau_query = self.tau_list[idx]

        similarities = []

        for i in range(len(self.word_list)):
            if i == idx:
                continue

            # Optional filters
            if tau_filter:
                if not (tau_filter[0] <= self.tau_list[i] <= tau_filter[1]):
                    continue

            if word_type_filter:
                if self.word_types[i] != word_type_filter:
                    continue

            # Compute similarity
            j_sim = cosine_similarity(j_query, self.j_matrix[i])
            i_sim = cosine_similarity(i_query, self.i_matrix[i])
            tau_sim = np.exp(-abs(tau_query - self.tau_list[i]) / 0.5)

            # Weighted combination
            sim = 0.5 * j_sim + 0.3 * i_sim + 0.2 * tau_sim
            similarities.append((self.word_list[i], sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    def project_to_tau(self, word: str, target_tau: float,
                      top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Project word to different τ-level, find nearest words there.
        """
        idx = self.get_word_index(word)
        if idx is None:
            return []

        j_query = self.j_matrix[idx]
        i_query = self.i_matrix[idx]

        # Find words near target_tau with similar direction
        tau_range = (target_tau - 0.3, target_tau + 0.3)
        candidates = []

        for i in range(len(self.word_list)):
            if not (tau_range[0] <= self.tau_list[i] <= tau_range[1]):
                continue

            # Similarity based on direction (j and i separately)
            j_sim = cosine_similarity(j_query, self.j_matrix[i])
            i_sim = cosine_similarity(i_query, self.i_matrix[i])

            # Weight j more for transcendental projection
            sim = 0.7 * j_sim + 0.3 * i_sim
            candidates.append((self.word_list[i], sim, self.tau_list[i]))

        candidates.sort(key=lambda x: -x[1])
        return [(w, s) for w, s, t in candidates[:top_k]]

    def blend_words(self, word1: str, word2: str, alpha: float = 0.5,
                   top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Blend two words in semantic space.
        """
        idx1 = self.get_word_index(word1)
        idx2 = self.get_word_index(word2)

        if idx1 is None or idx2 is None:
            return []

        # Interpolate vectors
        j_blend = (1 - alpha) * self.j_matrix[idx1] + alpha * self.j_matrix[idx2]
        i_blend = (1 - alpha) * self.i_matrix[idx1] + alpha * self.i_matrix[idx2]
        tau_blend = (1 - alpha) * self.tau_list[idx1] + alpha * self.tau_list[idx2]

        # Find nearest to blend
        candidates = []
        for i in range(len(self.word_list)):
            j_sim = cosine_similarity(j_blend, self.j_matrix[i])
            i_sim = cosine_similarity(i_blend, self.i_matrix[i])
            tau_sim = np.exp(-abs(tau_blend - self.tau_list[i]) / 0.5)

            sim = 0.5 * j_sim + 0.3 * i_sim + 0.2 * tau_sim
            candidates.append((self.word_list[i], sim))

        candidates.sort(key=lambda x: -x[1])

        # Exclude source words
        result = [(w, s) for w, s in candidates if w not in (word1, word2)]
        return result[:top_k]


class RefinedProjectionNavigator:
    """
    Navigate projection hierarchy with validated constants.
    """

    def __init__(self, vectors: Dict, fourier: RefinedSemanticFourier):
        self.vectors = vectors
        self.fourier = fourier
        self._classify_verbs()

    def _classify_verbs(self):
        """Classify verbs using validated τ thresholds."""
        self.lifting_verbs = []
        self.grounding_verbs = []
        self.neutral_verbs = []

        for i, word in enumerate(self.fourier.word_list):
            if self.fourier.word_types[i] != 'verb':
                continue

            tau = self.fourier.tau_list[i]

            # Use validated thresholds
            if tau > 3.0:  # Close to TAU_LIFT = 3.977
                self.lifting_verbs.append((word, tau))
            elif tau < 1.9:  # Close to TAU_NEUTRAL = 1.812
                self.neutral_verbs.append((word, tau))
            else:
                self.grounding_verbs.append((word, tau))

        # Sort by τ
        self.lifting_verbs.sort(key=lambda x: -x[1])
        self.grounding_verbs.sort(key=lambda x: x[1])
        self.neutral_verbs.sort(key=lambda x: x[1])

    def navigate_up(self, word: str, steps: int = 3) -> List[Dict]:
        """Navigate from concrete toward abstract (increasing τ)."""
        path = []
        current = word

        for step in range(steps):
            idx = self.fourier.get_word_index(current)
            if idx is None:
                break

            current_tau = self.fourier.tau_list[idx]
            target_tau = min(6.0, current_tau + 0.5)

            # Project to higher τ
            candidates = self.fourier.project_to_tau(current, target_tau, top_k=10)

            # Find best that's not current
            next_word = None
            for w, sim in candidates:
                if w != current:
                    next_word = w
                    next_sim = sim
                    break

            path.append({
                'step': step,
                'word': current,
                'tau': current_tau,
                'target_tau': target_tau,
                'next_word': next_word,
                'similarity': next_sim if next_word else 0,
            })

            if next_word:
                current = next_word
            else:
                break

        return path

    def navigate_down(self, word: str, steps: int = 3) -> List[Dict]:
        """Navigate from abstract toward concrete (decreasing τ)."""
        path = []
        current = word

        for step in range(steps):
            idx = self.fourier.get_word_index(current)
            if idx is None:
                break

            current_tau = self.fourier.tau_list[idx]
            target_tau = max(1.0, current_tau - 0.5)

            # Project to lower τ
            candidates = self.fourier.project_to_tau(current, target_tau, top_k=10)

            next_word = None
            for w, sim in candidates:
                if w != current:
                    next_word = w
                    next_sim = sim
                    break

            path.append({
                'step': step,
                'word': current,
                'tau': current_tau,
                'target_tau': target_tau,
                'next_word': next_word,
                'similarity': next_sim if next_word else 0,
            })

            if next_word:
                current = next_word
            else:
                break

        return path

    def apply_verb_operator(self, noun: str, verb: str) -> Dict:
        """Apply verb as operator to noun."""
        noun_idx = self.fourier.get_word_index(noun)
        verb_idx = self.fourier.get_word_index(verb)

        if noun_idx is None or verb_idx is None:
            return {'error': 'Word not found'}

        noun_tau = self.fourier.tau_list[noun_idx]
        verb_tau = self.fourier.tau_list[verb_idx]

        # Determine operation type and target τ
        if verb_tau > 3.0:
            direction = "lifting"
            delta = (verb_tau - TAU_VERB) * 0.3
            target_tau = min(6.0, noun_tau + delta)
        elif verb_tau < 1.9:
            direction = "grounding"
            delta = (TAU_VERB - verb_tau) * 0.3
            target_tau = max(1.0, noun_tau - delta)
        else:
            direction = "neutral"
            target_tau = noun_tau * 0.8 + verb_tau * 0.2

        # Find result
        candidates = self.fourier.project_to_tau(noun, target_tau, top_k=5)

        return {
            'noun': noun,
            'verb': verb,
            'noun_tau': noun_tau,
            'verb_tau': verb_tau,
            'target_tau': target_tau,
            'direction': direction,
            'result': candidates,
        }


def run_refined_tests():
    """Run refined Fourier tests."""
    print("=" * 70)
    print("EXPERIMENT 9b: REFINED FOURIER NAVIGATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    loader = DataLoader()
    vectors = loader.load_word_vectors()

    fourier = RefinedSemanticFourier(vectors)
    nav = RefinedProjectionNavigator(vectors, fourier)

    # Test 1: Semantic similarity
    print("\n" + "=" * 60)
    print("TEST 1: Semantic Similarity (Cosine-based)")
    print("=" * 60)

    test_pairs = [
        ('love', 'hate'),
        ('love', 'beauty'),
        ('truth', 'wisdom'),
        ('chair', 'table'),
        ('god', 'sacred'),
    ]

    for w1, w2 in test_pairs:
        sim = fourier.semantic_similarity(w1, w2)
        if sim is not None:
            print(f"  {w1} ↔ {w2}: {sim:.3f}")

    # Test 2: Find similar with filters
    print("\n" + "=" * 60)
    print("TEST 2: Find Similar (with τ and type filters)")
    print("=" * 60)

    test_words = ['beauty', 'truth', 'love', 'wisdom']

    for word in test_words:
        similar = fourier.find_similar(word, top_k=5)
        words_str = ', '.join([f"{w}({s:.2f})" for w, s in similar])
        print(f"\n  Similar to '{word}': {words_str}")

        # With noun filter
        similar_nouns = fourier.find_similar(word, top_k=5, word_type_filter='noun')
        words_str = ', '.join([f"{w}({s:.2f})" for w, s in similar_nouns])
        print(f"  Similar nouns: {words_str}")

    # Test 3: Blending
    print("\n" + "=" * 60)
    print("TEST 3: Semantic Blending")
    print("=" * 60)

    blends = [
        ('love', 'wisdom'),
        ('beauty', 'truth'),
        ('sacred', 'power'),
        ('life', 'meaning'),
    ]

    for w1, w2 in blends:
        result = fourier.blend_words(w1, w2, alpha=0.5, top_k=5)
        words_str = ', '.join([f"{w}({s:.2f})" for w, s in result])
        print(f"\n  {w1} + {w2} → {words_str}")

    # Test 4: Projection navigation
    print("\n" + "=" * 60)
    print("TEST 4: Projection Navigation")
    print("=" * 60)

    print("\n  ASCENDING (concrete → abstract):")
    for start in ['chair', 'flower', 'stone', 'water']:
        path = nav.navigate_up(start, steps=4)
        if path:
            chain = ' → '.join([f"{p['word']}({p['tau']:.1f})" for p in path])
            if path[-1]['next_word']:
                chain += f" → {path[-1]['next_word']}"
            print(f"    {chain}")

    print("\n  DESCENDING (abstract → concrete):")
    for start in ['truth', 'beauty', 'wisdom', 'love']:
        path = nav.navigate_down(start, steps=4)
        if path:
            chain = ' → '.join([f"{p['word']}({p['tau']:.1f})" for p in path])
            if path[-1]['next_word']:
                chain += f" → {path[-1]['next_word']}"
            print(f"    {chain}")

    # Test 5: Verb operators
    print("\n" + "=" * 60)
    print("TEST 5: Verb Operators")
    print("=" * 60)

    print(f"\n  Lifting verbs (top 5): {[v for v, t in nav.lifting_verbs[:5]]}")
    print(f"  Grounding verbs (top 5): {[v for v, t in nav.grounding_verbs[:5]]}")

    verb_tests = [
        ('idea', 'transcend'),
        ('truth', 'understand'),
        ('love', 'feel'),
        ('beauty', 'see'),
    ]

    print("\n  Verb applications:")
    for noun, verb in verb_tests:
        result = nav.apply_verb_operator(noun, verb)
        if 'error' not in result:
            top_results = ', '.join([w for w, s in result['result'][:3]])
            print(f"    {verb}({noun}): τ {result['noun_tau']:.2f}→{result['target_tau']:.2f} "
                  f"[{result['direction']}] → {top_results}")

    # Test 6: Full projection chain
    print("\n" + "=" * 60)
    print("TEST 6: Full Projection Chain (j-space → adj → noun)")
    print("=" * 60)

    for j_dim in J_DIMS:
        print(f"\n  {j_dim.upper()} chain:")

        # Start from transcendental
        idx = fourier.get_word_index(j_dim)
        if idx is None:
            continue

        tau_trans = fourier.tau_list[idx]
        print(f"    [Transcendental] {j_dim} (τ={tau_trans:.2f})")

        # Find adjectives with high similarity
        adj_candidates = fourier.find_similar(j_dim, top_k=5, word_type_filter='adjective')
        if adj_candidates:
            adj, adj_sim = adj_candidates[0]
            adj_idx = fourier.get_word_index(adj)
            adj_tau = fourier.tau_list[adj_idx] if adj_idx else 0
            print(f"    [Adjective] {adj} (τ={adj_tau:.2f}, sim={adj_sim:.2f})")

            # Find nouns with high similarity to adjective
            noun_candidates = fourier.find_similar(adj, top_k=5, word_type_filter='noun')
            if noun_candidates:
                noun, noun_sim = noun_candidates[0]
                noun_idx = fourier.get_word_index(noun)
                noun_tau = fourier.tau_list[noun_idx] if noun_idx else 0
                print(f"    [Noun] {noun} (τ={noun_tau:.2f}, sim={noun_sim:.2f})")

    print("\n" + "=" * 70)
    print("REFINED FOURIER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_refined_tests()
