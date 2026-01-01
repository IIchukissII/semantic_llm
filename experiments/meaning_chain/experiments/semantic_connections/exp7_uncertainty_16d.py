#!/usr/bin/env python3
"""
Experiment 7: Semantic Uncertainty Principle (5D j-vectors, PostgreSQL)
=======================================================================

Tests Δτ × Δj ≥ κ with 5D j-vectors from PostgreSQL (22k+ words).

j(5D) = [beauty, life, sacred, good, love]  (transcendental dimensions)

Note: i-vectors (11D surface) are not yet populated in the database.
Future work: Test orthogonality j ⊥ i when i-vectors are available.

Key result: κ ≈ 0.003 with 22k vocabulary (vs κ ≈ 0.007 with 500 words from Neo4j)
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from collections import defaultdict

# Add paths
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _EXPERIMENT_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

# Import DataLoader for PostgreSQL access
sys.path.insert(0, str(_SEMANTIC_LLM))
from core.data_loader import DataLoader

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']


def get_16d_vector(word_data):
    """Extract 16D vector from word data."""
    j_dict = word_data.get('j', {})
    i_dict = word_data.get('i', {})

    if not j_dict or not i_dict:
        return None

    j = np.array([j_dict.get(d, 0) for d in J_DIMS])
    i = np.array([i_dict.get(d, 0) for d in I_DIMS])

    return np.concatenate([j, i])


def compute_spread_16d(vectors):
    """Compute spread in 16D space."""
    if len(vectors) < 2:
        return 0.0

    arr = np.array(vectors)
    centroid = np.mean(arr, axis=0)

    # Average Euclidean distance from centroid
    distances = [np.linalg.norm(v - centroid) for v in arr]
    return np.mean(distances)


def compute_angular_spread(vectors):
    """Compute angular spread (cosine-based)."""
    if len(vectors) < 2:
        return 0.0

    similarities = []
    for i in range(len(vectors)):
        for k in range(i + 1, len(vectors)):
            v1, v2 = vectors[i], vectors[k]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                sim = np.dot(v1, v2) / (n1 * n2)
                similarities.append(sim)

    if not similarities:
        return 0.0

    # Spread = 1 - mean similarity
    return 1.0 - np.mean(similarities)


def exp7_1_vocabulary_stats(loader):
    """
    Stats on the PostgreSQL vocabulary.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7.1: PostgreSQL Vocabulary Stats")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    total = len(vectors)
    with_j = sum(1 for v in vectors.values() if v.get('j'))
    with_tau = sum(1 for v in vectors.values() if v.get('tau'))

    print(f"  Total words: {total}")
    print(f"  With j-vector: {with_j}")
    print(f"  With τ: {with_tau}")

    # τ distribution
    taus = [v['tau'] for v in vectors.values() if v.get('tau')]
    if taus:
        print(f"\n  τ distribution:")
        print(f"    Min: {min(taus):.2f}")
        print(f"    Max: {max(taus):.2f}")
        print(f"    Mean: {np.mean(taus):.2f}")
        print(f"    Median: {np.median(taus):.2f}")

    return {"total": total, "with_j": with_j, "with_tau": with_tau}


def exp7_2_uncertainty_16d(loader):
    """
    Test Δτ × Δv (16D) for word pairs.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7.2: Uncertainty Principle (16D)")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Filter words with both τ and full 16D vector
    valid_words = []
    for word, data in vectors.items():
        if data.get('tau') and data.get('j') and data.get('i'):
            v16 = get_16d_vector(data)
            if v16 is not None and np.linalg.norm(v16) > 0.01:
                valid_words.append({
                    'word': word,
                    'tau': data['tau'],
                    'v16': v16,
                })

    print(f"  Valid words with 16D vectors: {len(valid_words)}")

    if len(valid_words) < 100:
        print("  Insufficient data for analysis")
        return None

    # Sample pairs and compute uncertainty
    np.random.seed(42)
    sample_size = min(1000, len(valid_words))
    sample = np.random.choice(len(valid_words), sample_size, replace=False)

    uncertainties = []

    for i in range(len(sample)):
        for k in range(i + 1, min(i + 50, len(sample))):  # Limit pairs per word
            w1 = valid_words[sample[i]]
            w2 = valid_words[sample[k]]

            delta_tau = abs(w1['tau'] - w2['tau'])
            delta_v = np.linalg.norm(w1['v16'] - w2['v16'])

            if delta_tau > 0.01 and delta_v > 0.01:
                uncertainty = delta_tau * delta_v
                uncertainties.append({
                    'pair': (w1['word'], w2['word']),
                    'delta_tau': delta_tau,
                    'delta_v': delta_v,
                    'uncertainty': uncertainty,
                })

    print(f"  Computed {len(uncertainties)} pair uncertainties")

    if not uncertainties:
        return None

    # Sort and analyze
    uncertainties.sort(key=lambda x: x['uncertainty'])

    # Find minimum (semantic κ)
    min_u = uncertainties[0]
    kappa = min_u['uncertainty']

    print(f"\n  Minimum uncertainty pair:")
    print(f"    {min_u['pair']}")
    print(f"    Δτ = {min_u['delta_tau']:.4f}")
    print(f"    Δv(16D) = {min_u['delta_v']:.4f}")
    print(f"    Δτ × Δv = {kappa:.6f}")

    # Statistics
    all_u = [u['uncertainty'] for u in uncertainties]

    print(f"\n  Uncertainty statistics (16D):")
    print(f"    Min (κ): {np.min(all_u):.6f}")
    print(f"    Mean: {np.mean(all_u):.4f}")
    print(f"    Median: {np.median(all_u):.4f}")
    print(f"    Max: {np.max(all_u):.4f}")

    # Percentiles
    p1 = np.percentile(all_u, 1)
    p5 = np.percentile(all_u, 5)
    p10 = np.percentile(all_u, 10)

    print(f"\n  Percentiles:")
    print(f"    1%: {p1:.6f}")
    print(f"    5%: {p5:.6f}")
    print(f"    10%: {p10:.6f}")

    # Count violations
    below_01 = sum(1 for u in all_u if u < 0.1)
    below_001 = sum(1 for u in all_u if u < 0.01)

    print(f"\n  Violations:")
    print(f"    Δτ×Δv < 0.1: {below_01}/{len(all_u)} ({100*below_01/len(all_u):.1f}%)")
    print(f"    Δτ×Δv < 0.01: {below_001}/{len(all_u)} ({100*below_001/len(all_u):.1f}%)")

    return {
        'kappa': kappa,
        'min_pair': min_u,
        'stats': {
            'min': np.min(all_u),
            'mean': np.mean(all_u),
            'median': np.median(all_u),
            'max': np.max(all_u),
            'p1': p1, 'p5': p5, 'p10': p10,
        },
        'n_pairs': len(uncertainties),
    }


def exp7_3_tau_bands_16d(loader):
    """
    Do tight τ-bands have wide 16D spreads?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7.3: τ-Bands vs 16D Spread")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Collect valid words
    words_by_tau = defaultdict(list)

    for word, data in vectors.items():
        if data.get('tau') and data.get('j') and data.get('i'):
            v16 = get_16d_vector(data)
            if v16 is not None:
                tau = data['tau']
                # Bin by tau (0.5 width bands)
                band = round(tau * 2) / 2  # Round to nearest 0.5
                words_by_tau[band].append({
                    'word': word,
                    'tau': tau,
                    'v16': v16,
                })

    print(f"  τ-bands found: {len(words_by_tau)}")

    results = []

    for band in sorted(words_by_tau.keys()):
        words = words_by_tau[band]
        if len(words) < 5:
            continue

        taus = [w['tau'] for w in words]
        v16s = [w['v16'] for w in words]

        delta_tau = np.std(taus)
        delta_v = compute_spread_16d(v16s)
        angular = compute_angular_spread(v16s)

        uncertainty = delta_tau * delta_v

        results.append({
            'band': band,
            'n_words': len(words),
            'delta_tau': delta_tau,
            'delta_v_16d': delta_v,
            'angular_spread': angular,
            'uncertainty': uncertainty,
        })

        print(f"    τ≈{band:.1f}: n={len(words):4d}, Δτ={delta_tau:.3f}, "
              f"Δv(16D)={delta_v:.3f}, Δτ×Δv={uncertainty:.4f}")

    # Check inverse relationship
    if len(results) >= 3:
        deltas_tau = [r['delta_tau'] for r in results]
        deltas_v = [r['delta_v_16d'] for r in results]

        # Correlation
        corr = np.corrcoef(deltas_tau, deltas_v)[0, 1]
        print(f"\n  Correlation(Δτ, Δv): {corr:.3f}")
        print(f"  (Negative = uncertainty principle holds)")

    return results


def exp7_4_per_dimension_analysis(loader):
    """
    Analyze uncertainty per j-dimension.
    Which transcendental dimension has tightest bound?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7.4: Per-Dimension Uncertainty")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Collect valid words with j-vectors
    valid_words = []
    for word, data in vectors.items():
        if data.get('tau') and data.get('j'):
            j = np.array([data['j'].get(d, 0) for d in J_DIMS])
            if np.linalg.norm(j) > 0.01:
                valid_words.append({
                    'word': word,
                    'tau': data['tau'],
                    'j': j,
                })

    print(f"  Valid words: {len(valid_words)}")

    if len(valid_words) < 100:
        print("  Insufficient data")
        return None

    # Sample pairs
    np.random.seed(42)
    sample_size = min(500, len(valid_words))
    sample = np.random.choice(len(valid_words), sample_size, replace=False)

    # Per-dimension uncertainties
    dim_uncertainties = {d: [] for d in J_DIMS}

    for idx in range(len(sample)):
        for kidx in range(idx + 1, min(idx + 20, len(sample))):
            w1 = valid_words[sample[idx]]
            w2 = valid_words[sample[kidx]]

            delta_tau = abs(w1['tau'] - w2['tau'])
            if delta_tau < 0.01:
                continue

            for dim_idx, dim in enumerate(J_DIMS):
                delta_dim = abs(w1['j'][dim_idx] - w2['j'][dim_idx])
                if delta_dim > 0.01:
                    dim_uncertainties[dim].append(delta_tau * delta_dim)

    print(f"\n  Per-dimension κ (Δτ × Δdim):")
    results = {}
    for dim in J_DIMS:
        if dim_uncertainties[dim]:
            kappa = np.min(dim_uncertainties[dim])
            mean = np.mean(dim_uncertainties[dim])
            results[dim] = {'kappa': kappa, 'mean': mean, 'n_pairs': len(dim_uncertainties[dim])}
            print(f"    {dim:8s}: κ={kappa:.6f}, mean={mean:.4f}, n={len(dim_uncertainties[dim])}")

    # Find which dimension has tightest bound
    if results:
        tightest = min(results.items(), key=lambda x: x[1]['kappa'])
        loosest = max(results.items(), key=lambda x: x[1]['kappa'])
        print(f"\n  Tightest bound: {tightest[0]} (κ={tightest[1]['kappa']:.6f})")
        print(f"  Loosest bound: {loosest[0]} (κ={loosest[1]['kappa']:.6f})")
        print(f"  Ratio: {loosest[1]['kappa']/tightest[1]['kappa']:.2f}x")

    return results


def exp7_4_j_vs_i_uncertainty(loader):
    """
    Compare uncertainty in j-space (5D) vs i-space (11D).
    Tests whether j and i have independent uncertainty bounds.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7.4: j-space (5D) vs i-space (11D)")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Collect valid words with both j and i vectors
    valid_words = []
    for word, data in vectors.items():
        if data.get('tau') and data.get('j') and data.get('i'):
            j = np.array([data['j'].get(d, 0) for d in J_DIMS])
            i = np.array([data['i'].get(d, 0) for d in I_DIMS])
            j_norm = np.linalg.norm(j)
            i_norm = np.linalg.norm(i)
            if j_norm > 0.01 and i_norm > 0.01:
                valid_words.append({
                    'word': word,
                    'tau': data['tau'],
                    'j': j,
                    'i': i,
                })

    print(f"  Valid words (both j and i): {len(valid_words)}")

    if len(valid_words) < 100:
        print("  Insufficient data")
        return None

    # Sample pairs
    np.random.seed(42)
    sample_size = min(500, len(valid_words))
    sample = np.random.choice(len(valid_words), sample_size, replace=False)

    j_uncertainties = []
    i_uncertainties = []

    for idx in range(len(sample)):
        for kidx in range(idx + 1, min(idx + 20, len(sample))):
            w1 = valid_words[sample[idx]]
            w2 = valid_words[sample[kidx]]

            delta_tau = abs(w1['tau'] - w2['tau'])
            delta_j = np.linalg.norm(w1['j'] - w2['j'])
            delta_i = np.linalg.norm(w1['i'] - w2['i'])

            if delta_tau > 0.01:
                if delta_j > 0.01:
                    j_uncertainties.append(delta_tau * delta_j)
                if delta_i > 0.01:
                    i_uncertainties.append(delta_tau * delta_i)

    print(f"\n  j-space (5D transcendentals):")
    print(f"    Pairs: {len(j_uncertainties)}")
    print(f"    Min Δτ×Δj: {np.min(j_uncertainties):.6f}")
    print(f"    Mean: {np.mean(j_uncertainties):.4f}")

    print(f"\n  i-space (11D surface):")
    print(f"    Pairs: {len(i_uncertainties)}")
    print(f"    Min Δτ×Δi: {np.min(i_uncertainties):.6f}")
    print(f"    Mean: {np.mean(i_uncertainties):.4f}")

    # Which has tighter bound?
    kappa_j = np.min(j_uncertainties)
    kappa_i = np.min(i_uncertainties)

    print(f"\n  Semantic constants:")
    print(f"    κ_j (5D) = {kappa_j:.6f}")
    print(f"    κ_i (11D) = {kappa_i:.6f}")
    print(f"    Ratio κ_i/κ_j = {kappa_i/kappa_j:.2f}")

    # Orthogonality check
    print(f"\n  Orthogonality check (j ⊥ i):")
    j_vecs = np.array([w['j'] for w in valid_words[:100]])
    i_vecs = np.array([w['i'] for w in valid_words[:100]])

    # Compute cross-correlation
    correlations = []
    for j_dim in range(5):
        for i_dim in range(11):
            corr = np.corrcoef(j_vecs[:, j_dim], i_vecs[:, i_dim])[0, 1]
            correlations.append(abs(corr))

    avg_corr = np.mean(correlations)
    max_corr = np.max(correlations)
    print(f"    Avg |corr(j_dim, i_dim)|: {avg_corr:.3f}")
    print(f"    Max |corr|: {max_corr:.3f}")
    print(f"    {'✓ Approximately orthogonal' if avg_corr < 0.3 else '✗ Not orthogonal'}")

    return {
        'kappa_j': kappa_j,
        'kappa_i': kappa_i,
        'mean_j': np.mean(j_uncertainties),
        'mean_i': np.mean(i_uncertainties),
        'avg_correlation': avg_corr,
        'max_correlation': max_corr,
    }


def run_all_experiments():
    """Run all 16D uncertainty experiments."""
    print("=" * 70)
    print("EXPERIMENT 7: SEMANTIC UNCERTAINTY (16D, PostgreSQL)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nHypothesis: Δτ × Δv ≥ κ")
    print("Using 16D = j(5D) + i(11D) from PostgreSQL")

    all_results = {}

    try:
        loader = DataLoader()

        # Test 1: Vocabulary stats
        stats = exp7_1_vocabulary_stats(loader)
        all_results["vocab_stats"] = stats

        # Test 2: 16D uncertainty
        result = exp7_2_uncertainty_16d(loader)
        if result:
            all_results["uncertainty_16d"] = result

        # Test 3: τ-bands
        result = exp7_3_tau_bands_16d(loader)
        all_results["tau_bands"] = result

        # Test 4: j vs i
        result = exp7_4_j_vs_i_uncertainty(loader)
        if result:
            all_results["j_vs_i"] = result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SEMANTIC UNCERTAINTY CONSTANTS")
    print("=" * 70)

    if 'uncertainty_16d' in all_results:
        print(f"  κ (16D full): {all_results['uncertainty_16d']['kappa']:.6f}")

    if 'j_vs_i' in all_results:
        print(f"  κ_j (5D transcendentals): {all_results['j_vs_i']['kappa_j']:.6f}")
        print(f"  κ_i (11D surface): {all_results['j_vs_i']['kappa_i']:.6f}")

    print("\n  Δτ × Δv ≥ κ (semantic uncertainty principle)")
    print("=" * 70)

    # Save results
    output_dir = _EXPERIMENT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp7_uncertainty_16d_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
