"""
Experiment 10b: Dialectical Quadrant Clustering
================================================

Exploring WHY words cluster in Q1 (0-90°) and Q3 (-180 to -90°).

This bimodal distribution suggests dialectical structure in semantic space:
- Q1 and Q3 are OPPOSITE quadrants
- 81% of words fall in these two opposing regions
- Only 19% in Q2 and Q4

Questions:
1. What characterizes Q1 vs Q3 words?
2. Are semantic opposites in opposite quadrants?
3. What is the underlying dialectical principle?
"""

import numpy as np
from datetime import datetime
from collections import defaultdict
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MEANING_CHAIN = _THIS_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.data_loader import DataLoader

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']


def get_j_vector(word_data: dict) -> np.ndarray:
    j_dict = word_data.get('j', {})
    return np.array([j_dict.get(dim, 0.0) for dim in J_DIMS])


def get_quadrant(j_vec: np.ndarray, j_mean: np.ndarray) -> tuple:
    """
    Get quadrant based on phase angle in beauty-life plane.
    Returns (quadrant_number, angle_degrees)
    """
    j_shifted = j_vec - j_mean
    # Phase angle using beauty (x) and life (y)
    angle = np.arctan2(j_shifted[1], j_shifted[0])  # life, beauty
    angle_deg = np.degrees(angle)

    if 0 <= angle_deg < 90:
        return 1, angle_deg
    elif 90 <= angle_deg < 180:
        return 2, angle_deg
    elif -180 <= angle_deg < -90:
        return 3, angle_deg
    else:  # -90 to 0
        return 4, angle_deg


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 10b: DIALECTICAL QUADRANT CLUSTERING")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    loader = DataLoader()
    word_vectors = loader.load_word_vectors()
    print(f"Loaded {len(word_vectors)} word vectors")

    # Calculate j_mean
    all_j = []
    for word, data in word_vectors.items():
        j = get_j_vector(data)
        if np.linalg.norm(j) > 0.01:
            all_j.append(j)
    j_mean = np.mean(all_j, axis=0)

    print(f"\nj_mean = [{', '.join([f'{v:+.3f}' for v in j_mean])}]")
    print(f"        [{', '.join(J_DIMS)}]")

    # ================================================================
    # Classify all words by quadrant
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Classify Words by Quadrant")
    print("=" * 60)

    quadrants = {1: [], 2: [], 3: [], 4: []}

    for word, data in word_vectors.items():
        j = get_j_vector(data)
        if np.linalg.norm(j) < 0.01:
            continue

        q, angle = get_quadrant(j, j_mean)
        tau = data.get('tau', 2.5)
        word_type = data.get('word_type', 'unknown')

        quadrants[q].append({
            'word': word,
            'j': j,
            'j_shifted': j - j_mean,
            'angle': angle,
            'tau': tau,
            'word_type': word_type,
            'mag': np.linalg.norm(j - j_mean)
        })

    total = sum(len(q) for q in quadrants.values())
    for q in [1, 2, 3, 4]:
        pct = 100 * len(quadrants[q]) / total
        print(f"  Q{q}: {len(quadrants[q]):5} words ({pct:.1f}%)")

    # ================================================================
    # Analyze Q1 characteristics
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Q1 Characteristics (beauty+, life+)")
    print("=" * 60)

    q1 = quadrants[1]
    q1_sorted = sorted(q1, key=lambda x: -x['mag'])

    print(f"\n  Q1 top words (by magnitude):")
    for w in q1_sorted[:20]:
        j_str = ', '.join([f'{v:+.2f}' for v in w['j_shifted']])
        print(f"    {w['word']:15} τ={w['tau']:.2f} angle={w['angle']:+6.1f}° [{j_str}]")

    # Q1 averages
    q1_tau_mean = np.mean([w['tau'] for w in q1])
    q1_j_mean = np.mean([w['j_shifted'] for w in q1], axis=0)

    print(f"\n  Q1 averages:")
    print(f"    Mean τ: {q1_tau_mean:.2f}")
    print(f"    Mean j_shifted: [{', '.join([f'{v:+.3f}' for v in q1_j_mean])}]")

    # Dominant dimension in Q1
    q1_dominant = J_DIMS[np.argmax(q1_j_mean)]
    print(f"    Dominant dimension: {q1_dominant}")

    # Word types in Q1
    q1_types = defaultdict(int)
    for w in q1:
        q1_types[w['word_type']] += 1
    print(f"    Word types: {dict(q1_types)}")

    # ================================================================
    # Analyze Q3 characteristics
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Q3 Characteristics (beauty-, life-)")
    print("=" * 60)

    q3 = quadrants[3]
    q3_sorted = sorted(q3, key=lambda x: -x['mag'])

    print(f"\n  Q3 top words (by magnitude):")
    for w in q3_sorted[:20]:
        j_str = ', '.join([f'{v:+.2f}' for v in w['j_shifted']])
        print(f"    {w['word']:15} τ={w['tau']:.2f} angle={w['angle']:+6.1f}° [{j_str}]")

    # Q3 averages
    q3_tau_mean = np.mean([w['tau'] for w in q3])
    q3_j_mean = np.mean([w['j_shifted'] for w in q3], axis=0)

    print(f"\n  Q3 averages:")
    print(f"    Mean τ: {q3_tau_mean:.2f}")
    print(f"    Mean j_shifted: [{', '.join([f'{v:+.3f}' for v in q3_j_mean])}]")

    q3_dominant = J_DIMS[np.argmax(np.abs(q3_j_mean))]
    print(f"    Dominant dimension (by magnitude): {q3_dominant}")

    # Word types in Q3
    q3_types = defaultdict(int)
    for w in q3:
        q3_types[w['word_type']] += 1
    print(f"    Word types: {dict(q3_types)}")

    # ================================================================
    # Compare Q1 vs Q3 directly
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Q1 vs Q3 Direct Comparison")
    print("=" * 60)

    print(f"""
    DIMENSION        Q1 mean      Q3 mean      Difference
    ─────────────────────────────────────────────────────""")
    for i, dim in enumerate(J_DIMS):
        diff = q1_j_mean[i] - q3_j_mean[i]
        sign = "+" if diff > 0 else ""
        print(f"    {dim:12}  {q1_j_mean[i]:+.4f}      {q3_j_mean[i]:+.4f}      {sign}{diff:.4f}")

    print(f"""

    τ (abstractness) Q1: {q1_tau_mean:.2f}  Q3: {q3_tau_mean:.2f}  Diff: {q1_tau_mean - q3_tau_mean:+.2f}
    """)

    # ================================================================
    # Check if semantic opposites are in opposite quadrants
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Semantic Opposites Test")
    print("=" * 60)

    opposite_pairs = [
        ('love', 'hate'),
        ('life', 'death'),
        ('good', 'evil'),
        ('beauty', 'ugly'),
        ('truth', 'lie'),
        ('hope', 'despair'),
        ('peace', 'war'),
        ('light', 'dark'),
        ('joy', 'sorrow'),
        ('wisdom', 'folly'),
        ('order', 'chaos'),
        ('freedom', 'slavery'),
        ('sacred', 'profane'),
        ('create', 'destroy'),
        ('begin', 'end'),
    ]

    print(f"\n  Testing if opposites fall in opposite quadrants:")
    print(f"  {'Pair':<20} {'Word1':>10} {'Q1':>4} {'Word2':>10} {'Q2':>4} {'Opposite?':>10}")
    print(f"  {'-'*60}")

    opposite_count = 0
    same_count = 0

    for w1, w2 in opposite_pairs:
        if w1 not in word_vectors or w2 not in word_vectors:
            continue

        j1 = get_j_vector(word_vectors[w1])
        j2 = get_j_vector(word_vectors[w2])

        if np.linalg.norm(j1) < 0.01 or np.linalg.norm(j2) < 0.01:
            continue

        q1_num, angle1 = get_quadrant(j1, j_mean)
        q2_num, angle2 = get_quadrant(j2, j_mean)

        # Check if opposite quadrants (1↔3 or 2↔4)
        is_opposite = (q1_num == 1 and q2_num == 3) or \
                      (q1_num == 3 and q2_num == 1) or \
                      (q1_num == 2 and q2_num == 4) or \
                      (q1_num == 4 and q2_num == 2)

        if is_opposite:
            opposite_count += 1
            result = "✓ YES"
        elif q1_num == q2_num:
            same_count += 1
            result = "✗ SAME"
        else:
            result = "~ adjacent"

        print(f"  {w1+'/'+w2:<20} {w1:>10} Q{q1_num}   {w2:>10} Q{q2_num}   {result:>10}")

    total_pairs = opposite_count + same_count + (len([p for p in opposite_pairs if p[0] in word_vectors and p[1] in word_vectors]) - opposite_count - same_count)
    print(f"\n  Summary: {opposite_count} opposite, {same_count} same quadrant")

    # ================================================================
    # Analyze the sparse quadrants Q2 and Q4
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Sparse Quadrants Q2 and Q4 Analysis")
    print("=" * 60)

    q2 = quadrants[2]
    q4 = quadrants[4]

    print(f"\n  Q2 (beauty-, life+) - {len(q2)} words:")
    q2_sorted = sorted(q2, key=lambda x: -x['mag'])[:15]
    for w in q2_sorted:
        print(f"    {w['word']:15} τ={w['tau']:.2f} angle={w['angle']:+6.1f}°")

    print(f"\n  Q4 (beauty+, life-) - {len(q4)} words:")
    q4_sorted = sorted(q4, key=lambda x: -x['mag'])[:15]
    for w in q4_sorted:
        print(f"    {w['word']:15} τ={w['tau']:.2f} angle={w['angle']:+6.1f}°")

    # What makes Q2 and Q4 different?
    if len(q2) > 0:
        q2_tau_mean = np.mean([w['tau'] for w in q2])
        q2_j_mean = np.mean([w['j_shifted'] for w in q2], axis=0)
        print(f"\n  Q2 mean τ: {q2_tau_mean:.2f}")
        print(f"  Q2 mean j: [{', '.join([f'{v:+.3f}' for v in q2_j_mean])}]")

    if len(q4) > 0:
        q4_tau_mean = np.mean([w['tau'] for w in q4])
        q4_j_mean = np.mean([w['j_shifted'] for w in q4], axis=0)
        print(f"\n  Q4 mean τ: {q4_tau_mean:.2f}")
        print(f"  Q4 mean j: [{', '.join([f'{v:+.3f}' for v in q4_j_mean])}]")

    # ================================================================
    # The Dialectical Principle
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 7: The Dialectical Principle")
    print("=" * 60)

    # Calculate the main axis of variation
    all_j_shifted = np.array([w['j_shifted'] for q in quadrants.values() for w in q])

    # Covariance matrix
    cov_matrix = np.cov(all_j_shifted.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\n  Principal Components of j-space:")
    print(f"  (Eigenvalues show variance explained)")
    total_var = np.sum(eigenvalues)
    for i in range(5):
        pct = 100 * eigenvalues[i] / total_var
        vec = eigenvectors[:, i]
        vec_str = ', '.join([f'{v:+.3f}' for v in vec])
        print(f"    PC{i+1}: λ={eigenvalues[i]:.4f} ({pct:.1f}%) [{vec_str}]")
        # Interpret the component
        dominant_idx = np.argmax(np.abs(vec))
        dominant_dim = J_DIMS[dominant_idx]
        print(f"         Dominant: {dominant_dim} ({vec[dominant_idx]:+.3f})")

    # ================================================================
    # The beauty-life correlation
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Beauty-Life Correlation (The Diagonal)")
    print("=" * 60)

    beauty_vals = all_j_shifted[:, 0]
    life_vals = all_j_shifted[:, 1]

    corr_beauty_life = np.corrcoef(beauty_vals, life_vals)[0, 1]
    print(f"\n  Correlation(beauty, life) = {corr_beauty_life:.4f}")

    # Check all pairwise correlations
    print(f"\n  All pairwise correlations in j-space:")
    print(f"  {'':12}", end='')
    for dim in J_DIMS:
        print(f"{dim:>8}", end='')
    print()

    for i, dim1 in enumerate(J_DIMS):
        print(f"  {dim1:12}", end='')
        for j, dim2 in enumerate(J_DIMS):
            corr = np.corrcoef(all_j_shifted[:, i], all_j_shifted[:, j])[0, 1]
            print(f"{corr:+.3f}   ", end='')
        print()

    # ================================================================
    # Interpretation
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 9: Interpretation - Why Q1 and Q3?")
    print("=" * 60)

    print("""
    THE DIALECTICAL STRUCTURE OF SEMANTIC SPACE
    ════════════════════════════════════════════

    Finding: 81% of words cluster in Q1 (beauty+, life+) or Q3 (beauty-, life-)

    This means beauty and life are CORRELATED:
    - High beauty → tends to have high life
    - Low beauty → tends to have low life

    THE DIAGONAL AXIS:

                        life (+)
                          │
                     Q2   │   Q1 (41%)
                     ░░░░░│●●●●●●●●●●
                          │  ●●●●●●●●  ← POSITIVE POLE
              beauty (-)──┼──────────beauty (+)
                 ●●●●●●●● │   ░░░░░
       NEGATIVE → ●●●●●●●●│     Q4
         POLE     Q3 (41%)│
                          │
                        life (-)

    The main variation is along the DIAGONAL (Q1↔Q3), not the axes.

    This suggests a single underlying dimension:

        THESIS (Q1)              ANTITHESIS (Q3)
        ───────────              ──────────────
        beauty+, life+           beauty-, life-
        "vital beauty"           "dead ugliness"
        creation                 destruction
        affirmation              negation

    Q2 and Q4 are sparse because they represent CONTRADICTIONS:
    - Q2: beauty-, life+ = "ugly vitality" (rare concept)
    - Q4: beauty+, life- = "beautiful death" (rare, poetic)

    THE DIALECTICAL PRINCIPLE:
    ══════════════════════════

    Semantic space is organized around a SINGLE axis of affirmation/negation.
    Beauty and life are not independent - they form a unified dimension.

    This is Hegel's thesis/antithesis made visible in data.
    """)

    print("\n" + "=" * 70)
    print("EXPERIMENT 10b COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_experiment()
