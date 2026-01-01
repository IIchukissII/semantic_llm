#!/usr/bin/env python3
"""
Experiment 9: Projection Hierarchy Navigation + Fourier Semantics
==================================================================

Tests:
1. Navigation through projection levels (j-space → adj → noun)
2. Verb operators moving between levels
3. Fourier-based semantic encoding/decoding
4. Semantic spectrum analysis

Theory:
- Word = Σ wⱼ × transcendental_j + Σ wᵢ × surface_i
- j-space (5D) = low-frequency basis (deep, universal)
- i-space (11D) = high-frequency basis (surface, specific)
- τ = "frequency" dimension
- ||j|| ∝ τ (amplitude scales with frequency)
- ||i|| = const (phase invariant)
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Add paths
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


class SemanticFourier:
    """
    Fourier-like encoding/decoding for semantic space.

    Analogy:
        Signal:    f(t) = Σ aₙ × exp(i × ωₙ × t)
        Semantic:  word = Σ wⱼ × j_basis + Σ wᵢ × i_basis

    j-dimensions = low-frequency (transcendental, deep)
    i-dimensions = high-frequency (surface, specific)
    τ = frequency parameter
    """

    def __init__(self, vectors: Dict):
        self.vectors = vectors
        self._compute_basis_stats()

    def _compute_basis_stats(self):
        """Compute mean and std for normalization."""
        j_vecs = []
        i_vecs = []
        taus = []

        for word, data in self.vectors.items():
            if data.get('j') and data.get('i') and data.get('tau'):
                j = np.array([data['j'].get(d, 0) for d in J_DIMS])
                i = np.array([data['i'].get(d, 0) for d in I_DIMS])
                j_vecs.append(j)
                i_vecs.append(i)
                taus.append(data['tau'])

        self.j_mean = np.mean(j_vecs, axis=0)
        self.j_std = np.std(j_vecs, axis=0)
        self.i_mean = np.mean(i_vecs, axis=0)
        self.i_std = np.std(i_vecs, axis=0)
        self.tau_mean = np.mean(taus)
        self.tau_std = np.std(taus)

    def encode(self, word: str) -> Optional[Dict]:
        """
        Encode word into Fourier-like spectrum.

        Returns:
            {
                'word': str,
                'tau': float (frequency),
                'j_spectrum': np.array (low-freq amplitudes),
                'i_spectrum': np.array (high-freq amplitudes),
                'j_power': float (total low-freq power),
                'i_power': float (total high-freq power),
                'total_power': float
            }
        """
        if word not in self.vectors:
            return None

        data = self.vectors[word]
        if not data.get('j') or not data.get('i') or not data.get('tau'):
            return None

        j = np.array([data['j'].get(d, 0) for d in J_DIMS])
        i = np.array([data['i'].get(d, 0) for d in I_DIMS])
        tau = data['tau']

        # Normalize to spectrum (centered, scaled)
        j_spectrum = (j - self.j_mean) / (self.j_std + 1e-8)
        i_spectrum = (i - self.i_mean) / (self.i_std + 1e-8)

        # Power spectral density
        j_power = np.sum(j_spectrum ** 2)
        i_power = np.sum(i_spectrum ** 2)

        return {
            'word': word,
            'tau': tau,
            'j_spectrum': j_spectrum,
            'i_spectrum': i_spectrum,
            'j_power': j_power,
            'i_power': i_power,
            'total_power': j_power + i_power,
            'j_raw': j,
            'i_raw': i,
        }

    def decode(self, j_spectrum: np.ndarray, i_spectrum: np.ndarray,
               tau_hint: float = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Decode spectrum back to words (find nearest neighbors).

        Args:
            j_spectrum: Low-frequency components
            i_spectrum: High-frequency components
            tau_hint: Optional τ constraint
            top_k: Number of nearest words

        Returns:
            List of (word, distance) tuples
        """
        # Denormalize
        j_target = j_spectrum * (self.j_std + 1e-8) + self.j_mean
        i_target = i_spectrum * (self.i_std + 1e-8) + self.i_mean

        distances = []

        for word, data in self.vectors.items():
            if not data.get('j') or not data.get('i'):
                continue

            j = np.array([data['j'].get(d, 0) for d in J_DIMS])
            i = np.array([data['i'].get(d, 0) for d in I_DIMS])

            # Combined distance
            d_j = np.linalg.norm(j - j_target)
            d_i = np.linalg.norm(i - i_target)

            # Optional τ constraint
            if tau_hint is not None and data.get('tau'):
                d_tau = abs(data['tau'] - tau_hint) * 0.5
            else:
                d_tau = 0

            total_d = d_j + d_i + d_tau
            distances.append((word, total_d))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def blend(self, word1: str, word2: str, alpha: float = 0.5) -> Optional[Dict]:
        """
        Blend two words in Fourier space.

        Returns the blended spectrum and nearest words.
        """
        enc1 = self.encode(word1)
        enc2 = self.encode(word2)

        if enc1 is None or enc2 is None:
            return None

        # Linear interpolation in spectrum space
        j_blend = (1 - alpha) * enc1['j_spectrum'] + alpha * enc2['j_spectrum']
        i_blend = (1 - alpha) * enc1['i_spectrum'] + alpha * enc2['i_spectrum']
        tau_blend = (1 - alpha) * enc1['tau'] + alpha * enc2['tau']

        # Decode
        nearest = self.decode(j_blend, i_blend, tau_hint=tau_blend, top_k=10)

        return {
            'word1': word1,
            'word2': word2,
            'alpha': alpha,
            'j_blend': j_blend,
            'i_blend': i_blend,
            'tau_blend': tau_blend,
            'nearest': nearest,
        }

    def project_to_level(self, word: str, target_tau: float) -> Optional[Dict]:
        """
        Project word to different τ-level (change frequency, preserve direction).

        This is the key operation for projection hierarchy navigation.
        """
        enc = self.encode(word)
        if enc is None:
            return None

        # Keep direction (normalized spectrum), change τ
        j_direction = enc['j_spectrum'] / (np.linalg.norm(enc['j_spectrum']) + 1e-8)
        i_direction = enc['i_spectrum'] / (np.linalg.norm(enc['i_spectrum']) + 1e-8)

        # Scale j by τ (expansion law: ||j|| ∝ τ)
        # At target_tau, expected ||j|| = 0.30 × target_tau + 0.70
        expected_j_norm = 0.30 * target_tau + 0.70
        current_j_norm = np.linalg.norm(enc['j_raw'])
        scale_factor = expected_j_norm / (current_j_norm + 1e-8)

        j_projected = j_direction * scale_factor * np.linalg.norm(enc['j_spectrum'])

        # i stays invariant (surface invariance)
        i_projected = enc['i_spectrum']

        # Decode at target τ
        nearest = self.decode(j_projected, i_projected, tau_hint=target_tau, top_k=10)

        return {
            'source_word': word,
            'source_tau': enc['tau'],
            'target_tau': target_tau,
            'nearest': nearest,
        }

    def spectrum_analysis(self, word: str) -> Optional[Dict]:
        """
        Full spectrum analysis of a word.
        """
        enc = self.encode(word)
        if enc is None:
            return None

        # Dominant j-dimension
        j_abs = np.abs(enc['j_spectrum'])
        dominant_j_idx = np.argmax(j_abs)
        dominant_j = J_DIMS[dominant_j_idx]

        # Dominant i-dimension
        i_abs = np.abs(enc['i_spectrum'])
        dominant_i_idx = np.argmax(i_abs)
        dominant_i = I_DIMS[dominant_i_idx]

        # Power ratio
        power_ratio = enc['j_power'] / (enc['i_power'] + 1e-8)

        return {
            'word': word,
            'tau': enc['tau'],
            'j_power': enc['j_power'],
            'i_power': enc['i_power'],
            'power_ratio': power_ratio,
            'dominant_j': (dominant_j, enc['j_spectrum'][dominant_j_idx]),
            'dominant_i': (dominant_i, enc['i_spectrum'][dominant_i_idx]),
            'j_spectrum': dict(zip(J_DIMS, enc['j_spectrum'])),
            'i_spectrum': dict(zip(I_DIMS, enc['i_spectrum'])),
        }


class ProjectionNavigator:
    """
    Navigate through projection hierarchy using verbs as operators.
    """

    def __init__(self, vectors: Dict, fourier: SemanticFourier):
        self.vectors = vectors
        self.fourier = fourier
        self._classify_words()

    def _classify_words(self):
        """Classify words by type."""
        self.nouns = {}
        self.adjectives = {}
        self.verbs = {}

        for word, data in self.vectors.items():
            wtype = data.get('word_type', '')
            if wtype == 'noun':
                self.nouns[word] = data
            elif wtype == 'adjective':
                self.adjectives[word] = data
            elif wtype == 'verb':
                self.verbs[word] = data

        # Classify verbs by operator type
        self.lifting_verbs = []
        self.grounding_verbs = []
        self.neutral_verbs = []

        for verb, data in self.verbs.items():
            tau = data.get('tau', 2.0)
            if tau > 3.0:
                self.lifting_verbs.append(verb)
            elif tau < 1.9:
                self.neutral_verbs.append(verb)
            else:
                self.grounding_verbs.append(verb)

    def apply_verb(self, noun: str, verb: str) -> Optional[Dict]:
        """
        Apply verb operator to noun → move in τ-space.

        verb(noun) → result at different τ-level
        """
        if noun not in self.vectors or verb not in self.vectors:
            return None

        noun_data = self.vectors[noun]
        verb_data = self.vectors[verb]

        noun_tau = noun_data.get('tau', 1.5)
        verb_tau = verb_data.get('tau', 2.0)

        # Determine τ shift based on verb type
        if verb in self.lifting_verbs:
            target_tau = min(6.0, noun_tau + (verb_tau - 2.0) * 0.5)
            direction = "lifting"
        elif verb in self.grounding_verbs:
            target_tau = max(1.0, noun_tau - (2.0 - verb_tau) * 0.5)
            direction = "grounding"
        else:
            target_tau = noun_tau * 0.7 + verb_tau * 0.3
            direction = "neutral"

        # Project to target τ
        projection = self.fourier.project_to_level(noun, target_tau)

        return {
            'noun': noun,
            'verb': verb,
            'noun_tau': noun_tau,
            'verb_tau': verb_tau,
            'target_tau': target_tau,
            'direction': direction,
            'projection': projection,
        }

    def navigate_hierarchy(self, start_word: str, direction: str = "up",
                          steps: int = 3) -> List[Dict]:
        """
        Navigate through projection hierarchy.

        direction: "up" (toward j-space) or "down" (toward concrete)
        """
        path = []
        current = start_word

        for step in range(steps):
            current_data = self.vectors.get(current)
            if not current_data:
                break

            current_tau = current_data.get('tau', 1.5)

            # Determine target τ
            if direction == "up":
                target_tau = min(6.0, current_tau + 0.5)
                verbs = self.lifting_verbs[:3] if self.lifting_verbs else ['understand']
            else:
                target_tau = max(1.0, current_tau - 0.5)
                verbs = self.grounding_verbs[:3] if self.grounding_verbs else ['get']

            # Project to target
            projection = self.fourier.project_to_level(current, target_tau)

            if projection and projection['nearest']:
                # Find best next word (not current)
                next_word = None
                for word, dist in projection['nearest']:
                    if word != current:
                        next_word = word
                        break

                path.append({
                    'step': step,
                    'word': current,
                    'tau': current_tau,
                    'target_tau': target_tau,
                    'next_word': next_word,
                    'candidates': projection['nearest'][:5],
                })

                if next_word:
                    current = next_word
                else:
                    break
            else:
                break

        return path


def exp9_1_fourier_encoding(loader):
    """Test Fourier encoding/decoding."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9.1: Fourier Semantic Encoding")
    print("=" * 60)

    vectors = loader.load_word_vectors()
    fourier = SemanticFourier(vectors)

    # Test words across τ-levels
    test_words = ['chair', 'beauty', 'love', 'understand', 'truth', 'god', 'table', 'wisdom']

    print("\n  Spectrum Analysis:")
    for word in test_words:
        analysis = fourier.spectrum_analysis(word)
        if analysis:
            print(f"\n  {word} (τ={analysis['tau']:.2f}):")
            print(f"    j-power: {analysis['j_power']:.3f}, i-power: {analysis['i_power']:.3f}")
            print(f"    Power ratio (j/i): {analysis['power_ratio']:.3f}")
            print(f"    Dominant j: {analysis['dominant_j'][0]} ({analysis['dominant_j'][1]:.2f})")
            print(f"    Dominant i: {analysis['dominant_i'][0]} ({analysis['dominant_i'][1]:.2f})")

    return {'test_words': test_words}


def exp9_2_fourier_blending(loader):
    """Test blending words in Fourier space."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9.2: Fourier Blending")
    print("=" * 60)

    vectors = loader.load_word_vectors()
    fourier = SemanticFourier(vectors)

    # Test blends
    blends = [
        ('love', 'knowledge', 0.5),
        ('beauty', 'truth', 0.5),
        ('life', 'death', 0.5),
        ('sacred', 'power', 0.5),
        ('wisdom', 'strength', 0.5),
    ]

    results = []
    print("\n  Semantic Blending (α=0.5):")

    for word1, word2, alpha in blends:
        blend = fourier.blend(word1, word2, alpha)
        if blend:
            nearest_words = [w for w, d in blend['nearest'][:5]]
            print(f"\n  {word1} + {word2} → {nearest_words}")
            print(f"    τ_blend: {blend['tau_blend']:.2f}")
            results.append({
                'word1': word1,
                'word2': word2,
                'nearest': nearest_words,
                'tau_blend': blend['tau_blend'],
            })

    return {'blends': results}


def exp9_3_projection_navigation(loader):
    """Test projection through τ-levels."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9.3: Projection Navigation")
    print("=" * 60)

    vectors = loader.load_word_vectors()
    fourier = SemanticFourier(vectors)

    # Test projection from concrete to abstract
    test_words = ['chair', 'flower', 'woman', 'heart']
    target_taus = [1.5, 2.0, 2.5, 3.0]

    print("\n  Projecting concrete words to higher τ:")

    for word in test_words:
        enc = fourier.encode(word)
        if not enc:
            continue

        print(f"\n  {word} (τ={enc['tau']:.2f}):")
        for target in target_taus:
            if target > enc['tau']:
                proj = fourier.project_to_level(word, target)
                if proj and proj['nearest']:
                    nearest = [w for w, d in proj['nearest'][:3]]
                    print(f"    → τ={target:.1f}: {nearest}")

    return {}


def exp9_4_verb_operators(loader):
    """Test verbs as operators between levels."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9.4: Verb Operators")
    print("=" * 60)

    vectors = loader.load_word_vectors()
    fourier = SemanticFourier(vectors)
    nav = ProjectionNavigator(vectors, fourier)

    print(f"\n  Classified verbs:")
    print(f"    Lifting: {len(nav.lifting_verbs)} (e.g., {nav.lifting_verbs[:3]})")
    print(f"    Grounding: {len(nav.grounding_verbs)} (e.g., {nav.grounding_verbs[:3]})")
    print(f"    Neutral: {len(nav.neutral_verbs)} (e.g., {nav.neutral_verbs[:3]})")

    # Test verb application
    test_cases = [
        ('idea', 'transcend'),
        ('idea', 'understand'),
        ('idea', 'get'),
        ('truth', 'find'),
        ('love', 'feel'),
    ]

    print("\n  Verb application (verb(noun) → τ shift):")
    for noun, verb in test_cases:
        result = nav.apply_verb(noun, verb)
        if result and result['projection']:
            nearest = [w for w, d in result['projection']['nearest'][:3]]
            print(f"\n  {verb}({noun}):")
            print(f"    τ: {result['noun_tau']:.2f} → {result['target_tau']:.2f} ({result['direction']})")
            print(f"    Result: {nearest}")

    return {}


def exp9_5_hierarchy_paths(loader):
    """Test navigation through hierarchy."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9.5: Hierarchy Path Navigation")
    print("=" * 60)

    vectors = loader.load_word_vectors()
    fourier = SemanticFourier(vectors)
    nav = ProjectionNavigator(vectors, fourier)

    # Navigate up from concrete
    start_words = ['chair', 'flower', 'stone']

    print("\n  Ascending paths (concrete → abstract):")
    for start in start_words:
        path = nav.navigate_hierarchy(start, direction="up", steps=4)
        if path:
            print(f"\n  {start}:")
            for step in path:
                print(f"    [{step['step']}] {step['word']} (τ={step['tau']:.2f}) → {step['next_word']}")

    # Navigate down from abstract
    start_words = ['truth', 'beauty', 'wisdom']

    print("\n  Descending paths (abstract → concrete):")
    for start in start_words:
        path = nav.navigate_hierarchy(start, direction="down", steps=4)
        if path:
            print(f"\n  {start}:")
            for step in path:
                print(f"    [{step['step']}] {step['word']} (τ={step['tau']:.2f}) → {step['next_word']}")

    return {}


def exp9_6_spectrum_patterns(loader):
    """Analyze spectrum patterns by word type."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 9.6: Spectrum Patterns by Word Type")
    print("=" * 60)

    vectors = loader.load_word_vectors()
    fourier = SemanticFourier(vectors)

    # Collect spectra by type
    type_spectra = defaultdict(list)

    for word, data in vectors.items():
        wtype = data.get('word_type', 'unknown')
        if wtype not in ('noun', 'verb', 'adjective'):
            continue

        enc = fourier.encode(word)
        if enc:
            type_spectra[wtype].append({
                'j_power': enc['j_power'],
                'i_power': enc['i_power'],
                'ratio': enc['j_power'] / (enc['i_power'] + 1e-8),
            })

    print("\n  Power spectrum by word type:")
    for wtype in ['noun', 'adjective', 'verb']:
        if wtype in type_spectra:
            specs = type_spectra[wtype]
            j_powers = [s['j_power'] for s in specs]
            i_powers = [s['i_power'] for s in specs]
            ratios = [s['ratio'] for s in specs]

            print(f"\n  {wtype.upper()} (n={len(specs)}):")
            print(f"    j-power: mean={np.mean(j_powers):.3f}, std={np.std(j_powers):.3f}")
            print(f"    i-power: mean={np.mean(i_powers):.3f}, std={np.std(i_powers):.3f}")
            print(f"    ratio (j/i): mean={np.mean(ratios):.3f}, std={np.std(ratios):.3f}")

    return {}


def run_all_experiments():
    """Run all experiments."""
    print("=" * 70)
    print("EXPERIMENT 9: PROJECTION HIERARCHY + FOURIER SEMANTICS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {}

    try:
        loader = DataLoader()

        # Test 1: Fourier encoding
        result = exp9_1_fourier_encoding(loader)
        all_results["fourier_encoding"] = result

        # Test 2: Blending
        result = exp9_2_fourier_blending(loader)
        all_results["blending"] = result

        # Test 3: Projection navigation
        result = exp9_3_projection_navigation(loader)
        all_results["projection"] = result

        # Test 4: Verb operators
        result = exp9_4_verb_operators(loader)
        all_results["verb_operators"] = result

        # Test 5: Hierarchy paths
        result = exp9_5_hierarchy_paths(loader)
        all_results["hierarchy_paths"] = result

        # Test 6: Spectrum patterns
        result = exp9_6_spectrum_patterns(loader)
        all_results["spectrum_patterns"] = result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: FOURIER SEMANTICS")
    print("=" * 70)
    print("""
  Key findings:

  1. FOURIER ENCODING:
     - Words encoded as (τ, j_spectrum, i_spectrum)
     - j = low-frequency (transcendental depth)
     - i = high-frequency (surface properties)

  2. SEMANTIC BLENDING:
     - Linear interpolation in spectrum space
     - word1 + word2 → nearest neighbors

  3. PROJECTION NAVIGATION:
     - Move words between τ-levels
     - Preserve direction, change amplitude

  4. VERB OPERATORS:
     - Lifting verbs: τ ↑ (transcend, understand)
     - Grounding verbs: τ ↓ (get, make)
     - verb(noun) → noun at different τ

  5. HIERARCHY PATHS:
     - Navigate concrete → abstract (ascending)
     - Navigate abstract → concrete (descending)
    """)
    print("=" * 70)

    # Save results
    output_dir = _EXPERIMENT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp9_projection_fourier_{timestamp}.json"

    # Convert numpy to lists for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(all_results), f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
