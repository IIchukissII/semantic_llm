"""
Experiment 10: Phase + Shannon Word Decomposition
==================================================

Using ONLY our theoretical framework:
- j-vector (5 transcendentals): beauty, life, sacred, good, love
- i-vector (11 surface): truth, freedom, meaning, order, peace, power, nature, time, knowledge, self, society
- τ (abstractness level)
- Phase shift: centered j-vector (j - j_mean)
- Shannon entropy: information content of decomposition

Goal: Understand how words decompose into "semantic frequencies"
      and what their information content tells us.
"""

import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Set up paths
_THIS_DIR = Path(__file__).resolve().parent
_MEANING_CHAIN = _THIS_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.data_loader import DataLoader

# Constants from theory
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']


def get_j_vector(word_data: dict) -> np.ndarray:
    """Extract j-vector (5D transcendental) from word data"""
    j_dict = word_data.get('j', {})
    return np.array([j_dict.get(dim, 0.0) for dim in J_DIMS])


def get_i_vector(word_data: dict) -> np.ndarray:
    """Extract i-vector (11D surface) from word data"""
    i_dict = word_data.get('i', {})
    return np.array([i_dict.get(dim, 0.0) for dim in I_DIMS])


def shannon_entropy(probs: np.ndarray) -> float:
    """
    Calculate Shannon entropy: H = -Σ pᵢ log₂(pᵢ)

    Input should be probability distribution (sums to 1, all >= 0)
    """
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 1e-10]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def normalize_to_distribution(vec: np.ndarray) -> np.ndarray:
    """
    Convert vector to probability distribution.
    Use absolute values, then normalize to sum to 1.
    """
    abs_vec = np.abs(vec)
    total = np.sum(abs_vec)
    if total < 1e-10:
        return np.zeros_like(vec)
    return abs_vec / total


def phase_shift(vec: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    """
    Apply phase shift: centered = vec - mean
    This makes opposite concepts have negative values.
    """
    return vec - mean_vec


def analyze_word(word: str, word_data: dict, j_mean: np.ndarray, i_mean: np.ndarray) -> dict:
    """
    Complete analysis of a word's decomposition.
    """
    # Get raw vectors
    j = get_j_vector(word_data)
    i = get_i_vector(word_data)
    tau = word_data.get('tau', 2.5)

    # Apply phase shift (centering)
    j_shifted = phase_shift(j, j_mean)
    i_shifted = phase_shift(i, i_mean)

    # Convert to probability distributions for entropy
    j_prob = normalize_to_distribution(j)
    i_prob = normalize_to_distribution(i)
    j_shifted_prob = normalize_to_distribution(j_shifted)
    i_shifted_prob = normalize_to_distribution(i_shifted)

    # Calculate Shannon entropy
    H_j = shannon_entropy(j_prob)
    H_i = shannon_entropy(i_prob)
    H_j_shifted = shannon_entropy(j_shifted_prob)
    H_i_shifted = shannon_entropy(i_shifted_prob)

    # Maximum possible entropy (uniform distribution)
    H_j_max = np.log2(5)   # 5 dimensions → max 2.32 bits
    H_i_max = np.log2(11)  # 11 dimensions → max 3.46 bits

    # Normalized entropy (0 = focused, 1 = diffuse)
    H_j_norm = H_j / H_j_max if H_j_max > 0 else 0
    H_i_norm = H_i / H_i_max if H_i_max > 0 else 0

    # Magnitude (energy)
    mag_j = np.linalg.norm(j)
    mag_i = np.linalg.norm(i)
    mag_j_shifted = np.linalg.norm(j_shifted)
    mag_i_shifted = np.linalg.norm(i_shifted)

    # Dominant dimension
    j_dominant = J_DIMS[np.argmax(np.abs(j))] if mag_j > 0.01 else 'none'
    i_dominant = I_DIMS[np.argmax(np.abs(i))] if mag_i > 0.01 else 'none'

    # Phase angle (using shifted vectors)
    # Angle from origin in j-space
    phase_angle = np.arctan2(j_shifted[1], j_shifted[0]) if mag_j_shifted > 0.01 else 0

    return {
        'word': word,
        'tau': tau,
        # Raw vectors
        'j': j,
        'i': i,
        # Shifted vectors
        'j_shifted': j_shifted,
        'i_shifted': i_shifted,
        # Entropy (bits)
        'H_j': H_j,
        'H_i': H_i,
        'H_j_shifted': H_j_shifted,
        'H_i_shifted': H_i_shifted,
        # Normalized entropy
        'H_j_norm': H_j_norm,
        'H_i_norm': H_i_norm,
        # Magnitude
        'mag_j': mag_j,
        'mag_i': mag_i,
        'mag_j_shifted': mag_j_shifted,
        'mag_i_shifted': mag_i_shifted,
        # Dominant
        'j_dominant': j_dominant,
        'i_dominant': i_dominant,
        # Phase
        'phase_angle': phase_angle,
        # Total information
        'H_total': H_j + H_i,
        'H_total_shifted': H_j_shifted + H_i_shifted,
    }


def run_experiment():
    """Run Phase + Shannon decomposition experiment"""
    print("=" * 70)
    print("EXPERIMENT 10: PHASE + SHANNON WORD DECOMPOSITION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    loader = DataLoader()
    word_vectors = loader.load_word_vectors()
    print(f"Loaded {len(word_vectors)} word vectors")

    # ================================================================
    # STEP 1: Calculate mean vectors (for phase shift)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Calculate Mean Vectors (Phase Reference)")
    print("=" * 60)

    all_j = []
    all_i = []

    for word, data in word_vectors.items():
        j = get_j_vector(data)
        i = get_i_vector(data)
        if np.linalg.norm(j) > 0.01:  # Only non-zero vectors
            all_j.append(j)
        if np.linalg.norm(i) > 0.01:
            all_i.append(i)

    j_mean = np.mean(all_j, axis=0)
    i_mean = np.mean(all_i, axis=0)

    print(f"\nj_mean (phase reference for 5 transcendentals):")
    for dim, val in zip(J_DIMS, j_mean):
        print(f"  {dim:10}: {val:+.4f}")

    print(f"\ni_mean (phase reference for 11 surface dims):")
    for dim, val in zip(I_DIMS, i_mean):
        print(f"  {dim:10}: {val:+.4f}")

    # ================================================================
    # STEP 2: Analyze key words
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Analyze Key Words")
    print("=" * 60)

    test_words = [
        # Transcendentals
        'beauty', 'life', 'sacred', 'good', 'love',
        # Surface concepts
        'truth', 'freedom', 'meaning', 'power', 'knowledge',
        # Concrete
        'chair', 'table', 'water', 'stone', 'dog',
        # Abstract
        'wisdom', 'justice', 'faith', 'hope', 'dream',
        # Opposites
        'evil', 'death', 'hate', 'fear', 'chaos',
    ]

    results = []
    for word in test_words:
        if word in word_vectors:
            analysis = analyze_word(word, word_vectors[word], j_mean, i_mean)
            results.append(analysis)

    # Print detailed analysis
    for r in results:
        print(f"\n  {r['word'].upper()} (τ={r['tau']:.2f})")
        print(f"    j-vector: [{', '.join([f'{v:+.2f}' for v in r['j']])}]")
        print(f"    j-shifted: [{', '.join([f'{v:+.2f}' for v in r['j_shifted']])}]")
        print(f"    Dominant j: {r['j_dominant']}, Dominant i: {r['i_dominant']}")
        print(f"    H(j)={r['H_j']:.2f} bits, H(i)={r['H_i']:.2f} bits")
        print(f"    H(j) norm={r['H_j_norm']:.2f}, H(i) norm={r['H_i_norm']:.2f}")
        print(f"    |j|={r['mag_j']:.2f}, |j_shifted|={r['mag_j_shifted']:.2f}")
        print(f"    Phase angle: {np.degrees(r['phase_angle']):.1f}°")

    # ================================================================
    # STEP 3: Entropy Statistics
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Entropy Statistics Across All Words")
    print("=" * 60)

    all_results = []
    for word, data in word_vectors.items():
        try:
            analysis = analyze_word(word, data, j_mean, i_mean)
            if analysis['mag_j'] > 0.01:  # Only words with j-vectors
                all_results.append(analysis)
        except:
            pass

    print(f"\nAnalyzed {len(all_results)} words with valid j-vectors")

    # Statistics
    H_j_values = [r['H_j'] for r in all_results]
    H_i_values = [r['H_i'] for r in all_results]
    H_j_shifted_values = [r['H_j_shifted'] for r in all_results]
    tau_values = [r['tau'] for r in all_results]
    mag_j_values = [r['mag_j'] for r in all_results]

    print(f"\n  H(j) - Transcendental Entropy:")
    print(f"    Mean: {np.mean(H_j_values):.3f} bits")
    print(f"    Std:  {np.std(H_j_values):.3f} bits")
    print(f"    Max:  {np.max(H_j_values):.3f} bits (max possible: {np.log2(5):.3f})")
    print(f"    Min:  {np.min(H_j_values):.3f} bits")

    print(f"\n  H(i) - Surface Entropy:")
    print(f"    Mean: {np.mean(H_i_values):.3f} bits")
    print(f"    Std:  {np.std(H_i_values):.3f} bits")
    print(f"    Max:  {np.max(H_i_values):.3f} bits (max possible: {np.log2(11):.3f})")

    print(f"\n  H(j_shifted) - Phase-Shifted Entropy:")
    print(f"    Mean: {np.mean(H_j_shifted_values):.3f} bits")
    print(f"    Std:  {np.std(H_j_shifted_values):.3f} bits")

    # ================================================================
    # STEP 4: Entropy vs τ Correlation
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Entropy vs τ (Abstractness) Correlation")
    print("=" * 60)

    # Correlation
    corr_H_j_tau = np.corrcoef(H_j_values, tau_values)[0, 1]
    corr_H_i_tau = np.corrcoef(H_i_values, tau_values)[0, 1]
    corr_mag_tau = np.corrcoef(mag_j_values, tau_values)[0, 1]

    print(f"\n  Correlation H(j) vs τ: r = {corr_H_j_tau:.4f}")
    print(f"  Correlation H(i) vs τ: r = {corr_H_i_tau:.4f}")
    print(f"  Correlation |j| vs τ:  r = {corr_mag_tau:.4f}")

    # ================================================================
    # STEP 5: Focused vs Diffuse Words
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Focused (Low Entropy) vs Diffuse (High Entropy)")
    print("=" * 60)

    # Sort by H(j)
    sorted_by_entropy = sorted(all_results, key=lambda x: x['H_j'])

    print("\n  MOST FOCUSED (lowest H(j) - concentrated meaning):")
    for r in sorted_by_entropy[:10]:
        print(f"    {r['word']:15} H(j)={r['H_j']:.2f} τ={r['tau']:.2f} dominant={r['j_dominant']}")

    print("\n  MOST DIFFUSE (highest H(j) - spread meaning):")
    for r in sorted_by_entropy[-10:]:
        print(f"    {r['word']:15} H(j)={r['H_j']:.2f} τ={r['tau']:.2f} dominant={r['j_dominant']}")

    # ================================================================
    # STEP 6: Phase Analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Phase Angle Distribution")
    print("=" * 60)

    phase_angles = [r['phase_angle'] for r in all_results]
    phase_degrees = [np.degrees(p) for p in phase_angles]

    print(f"\n  Phase angle statistics:")
    print(f"    Mean: {np.mean(phase_degrees):.1f}°")
    print(f"    Std:  {np.std(phase_degrees):.1f}°")

    # Group by quadrant
    q1 = len([p for p in phase_degrees if 0 <= p < 90])
    q2 = len([p for p in phase_degrees if 90 <= p < 180])
    q3 = len([p for p in phase_degrees if -180 <= p < -90])
    q4 = len([p for p in phase_degrees if -90 <= p < 0])

    print(f"\n  Quadrant distribution:")
    print(f"    Q1 (0-90°):    {q1:5} ({100*q1/len(phase_degrees):.1f}%)")
    print(f"    Q2 (90-180°):  {q2:5} ({100*q2/len(phase_degrees):.1f}%)")
    print(f"    Q3 (-180,-90): {q3:5} ({100*q3/len(phase_degrees):.1f}%)")
    print(f"    Q4 (-90-0°):   {q4:5} ({100*q4/len(phase_degrees):.1f}%)")

    # ================================================================
    # STEP 7: Information Content Formula
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Total Information Content")
    print("=" * 60)

    # Total information = H(j) + H(i) + log2(τ-levels)
    # But we can also weight by magnitude

    total_info = []
    for r in all_results:
        # Information = entropy × magnitude (weighted information)
        info = (r['H_j'] * r['mag_j'] + r['H_i'] * r['mag_i'])
        total_info.append((r['word'], info, r['tau'], r['H_j'], r['mag_j']))

    total_info.sort(key=lambda x: -x[1])

    print("\n  HIGHEST Information Content (H × |vec|):")
    for word, info, tau, h_j, mag_j in total_info[:15]:
        print(f"    {word:15} I={info:.3f} τ={tau:.2f} H(j)={h_j:.2f} |j|={mag_j:.2f}")

    print("\n  LOWEST Information Content:")
    for word, info, tau, h_j, mag_j in total_info[-10:]:
        print(f"    {word:15} I={info:.3f} τ={tau:.2f} H(j)={h_j:.2f} |j|={mag_j:.2f}")

    # ================================================================
    # STEP 8: Decomposition Formula
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Word Decomposition Formula")
    print("=" * 60)

    print("""
    WORD DECOMPOSITION (from our theory):
    ═════════════════════════════════════

    Word = j⃗ + i⃗ + τ

    Where:
      j⃗ = [beauty, life, sacred, good, love]   (5D transcendental)
      i⃗ = [truth, freedom, ..., society]        (11D surface)
      τ = abstractness level (orbital n = (τ-1)×e)

    PHASE SHIFT (centering):
    ════════════════════════

      j⃗_shifted = j⃗ - j⃗_mean
      i⃗_shifted = i⃗ - i⃗_mean

    This makes opposite concepts have NEGATIVE values.

    SHANNON ENTROPY:
    ════════════════

      H(j⃗) = -Σ pᵢ log₂(pᵢ)     where pᵢ = |jᵢ| / Σ|jᵢ|

      Low H  → Focused (one dominant transcendental)
      High H → Diffuse (spread across transcendentals)

    INFORMATION CONTENT:
    ════════════════════

      I(word) = H(j⃗) × ||j⃗|| + H(i⃗) × ||i⃗||

      Entropy × Magnitude = Total information

    PHASE ANGLE:
    ════════════

      θ = atan2(j_life, j_beauty)

      Position in j-space relative to beauty-life plane.
    """)

    # ================================================================
    # STEP 9: Key Findings
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 9: Key Findings")
    print("=" * 60)

    print(f"""
    1. ENTROPY STATISTICS:
       - Mean H(j) = {np.mean(H_j_values):.3f} bits (max {np.log2(5):.3f})
       - Mean H(i) = {np.mean(H_i_values):.3f} bits (max {np.log2(11):.3f})
       - Words use ~{100*np.mean(H_j_values)/np.log2(5):.0f}% of j-space information capacity

    2. ENTROPY vs τ CORRELATION:
       - r(H(j), τ) = {corr_H_j_tau:.4f}
       - r(|j|, τ) = {corr_mag_tau:.4f}
       - {'Positive: abstract words more diffuse' if corr_H_j_tau > 0 else 'Negative: abstract words more focused'}

    3. PHASE DISTRIBUTION:
       - Mean phase: {np.mean(phase_degrees):.1f}°
       - Phase spread: {np.std(phase_degrees):.1f}°
       - Words distributed across all quadrants

    4. INFORMATION FORMULA:
       I(word) = H × ||vec||
       Combines entropy (spread) with magnitude (energy)
    """)

    print("\n" + "=" * 70)
    print("EXPERIMENT 10 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_experiment()
