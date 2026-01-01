#!/usr/bin/env python3
"""
Experiment 8: Projection Hierarchy
===================================

Tests the projection hierarchy hypothesis:
  j-space (5D) → Adjectives → Nouns

Key predictions:
1. Adjectives have higher τ than nouns (more abstract)
2. Adjectives have larger ||j|| than nouns (closer to transcendentals)
3. j[dim] flows: transcendental → adjective → noun (diminishing)
4. Verbs act as operators (not in hierarchy, but transform τ)
5. Semantic holes exist in j-space grid
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

from core.data_loader import DataLoader

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']


def get_j_vector(word_data):
    """Extract j-vector from word data."""
    j_dict = word_data.get('j', {})
    if not j_dict:
        return None
    return np.array([j_dict.get(d, 0) for d in J_DIMS])


def get_i_vector(word_data):
    """Extract i-vector from word data."""
    i_dict = word_data.get('i', {})
    if not i_dict:
        return None
    return np.array([i_dict.get(d, 0) for d in I_DIMS])


def exp8_1_word_type_distribution(loader):
    """
    Check word type distribution and basic stats.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8.1: Word Type Distribution")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    type_counts = defaultdict(int)
    type_tau = defaultdict(list)
    type_j_norm = defaultdict(list)
    type_i_norm = defaultdict(list)

    for word, data in vectors.items():
        wtype = data.get('word_type', 'unknown')
        if isinstance(wtype, (int, float)):
            wtype = 'unknown'
        type_counts[wtype] += 1

        if data.get('tau'):
            type_tau[wtype].append(data['tau'])

        j = get_j_vector(data)
        if j is not None:
            type_j_norm[wtype].append(np.linalg.norm(j))

        i = get_i_vector(data)
        if i is not None:
            type_i_norm[wtype].append(np.linalg.norm(i))

    print(f"\n  Word type counts:")
    for wtype in ['noun', 'verb', 'adjective', 'adverb', 'unknown']:
        if wtype in type_counts:
            print(f"    {wtype:12s}: {type_counts[wtype]:6d}")

    print(f"\n  τ by word type:")
    for wtype in ['noun', 'verb', 'adjective', 'adverb']:
        if wtype in type_tau and type_tau[wtype]:
            taus = type_tau[wtype]
            print(f"    {wtype:12s}: τ_mean={np.mean(taus):.2f}, "
                  f"τ_median={np.median(taus):.2f}, "
                  f"τ_std={np.std(taus):.2f}")

    print(f"\n  ||j|| by word type:")
    for wtype in ['noun', 'verb', 'adjective', 'adverb']:
        if wtype in type_j_norm and type_j_norm[wtype]:
            norms = type_j_norm[wtype]
            print(f"    {wtype:12s}: ||j||_mean={np.mean(norms):.3f}, "
                  f"||j||_std={np.std(norms):.3f}")

    print(f"\n  ||i|| by word type:")
    for wtype in ['noun', 'verb', 'adjective', 'adverb']:
        if wtype in type_i_norm and type_i_norm[wtype]:
            norms = type_i_norm[wtype]
            print(f"    {wtype:12s}: ||i||_mean={np.mean(norms):.3f}, "
                  f"||i||_std={np.std(norms):.3f}")

    return {
        'counts': dict(type_counts),
        'tau_means': {k: np.mean(v) for k, v in type_tau.items() if v},
        'j_norm_means': {k: np.mean(v) for k, v in type_j_norm.items() if v},
    }


def exp8_2_projection_hierarchy(loader):
    """
    Test: τ(adjectives) > τ(nouns)
    Test: ||j||(adjectives) > ||j||(nouns)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8.2: Projection Hierarchy (τ and ||j||)")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Collect by type
    nouns = []
    adjectives = []
    verbs = []

    for word, data in vectors.items():
        wtype = data.get('word_type', '')
        if not data.get('tau') or not data.get('j'):
            continue

        j = get_j_vector(data)
        if j is None:
            continue

        entry = {
            'word': word,
            'tau': data['tau'],
            'j': j,
            'j_norm': np.linalg.norm(j),
        }

        if wtype == 'noun':
            nouns.append(entry)
        elif wtype == 'adjective':
            adjectives.append(entry)
        elif wtype == 'verb':
            verbs.append(entry)

    print(f"\n  Counts: nouns={len(nouns)}, adjectives={len(adjectives)}, verbs={len(verbs)}")

    # Compare τ
    noun_tau = np.mean([n['tau'] for n in nouns])
    adj_tau = np.mean([a['tau'] for a in adjectives])
    verb_tau = np.mean([v['tau'] for v in verbs])

    print(f"\n  τ comparison:")
    print(f"    Nouns:      τ_mean = {noun_tau:.3f}")
    print(f"    Adjectives: τ_mean = {adj_tau:.3f}")
    print(f"    Verbs:      τ_mean = {verb_tau:.3f}")

    if adj_tau > noun_tau:
        print(f"    ✓ Adjectives more abstract (τ_adj > τ_noun by {adj_tau - noun_tau:.3f})")
    else:
        print(f"    ✗ Unexpected: τ_adj ≤ τ_noun")

    # Compare ||j||
    noun_j = np.mean([n['j_norm'] for n in nouns])
    adj_j = np.mean([a['j_norm'] for a in adjectives])
    verb_j = np.mean([v['j_norm'] for v in verbs])

    print(f"\n  ||j|| comparison:")
    print(f"    Nouns:      ||j||_mean = {noun_j:.3f}")
    print(f"    Adjectives: ||j||_mean = {adj_j:.3f}")
    print(f"    Verbs:      ||j||_mean = {verb_j:.3f}")

    if adj_j > noun_j:
        print(f"    ✓ Adjectives closer to j-space (||j||_adj > ||j||_noun by {adj_j - noun_j:.3f})")
    else:
        print(f"    ✗ Unexpected: ||j||_adj ≤ ||j||_noun")

    # Statistical test (t-test)
    from scipy import stats
    t_tau, p_tau = stats.ttest_ind([a['tau'] for a in adjectives], [n['tau'] for n in nouns])
    t_j, p_j = stats.ttest_ind([a['j_norm'] for a in adjectives], [n['j_norm'] for n in nouns])

    print(f"\n  Statistical tests:")
    print(f"    τ: t={t_tau:.2f}, p={p_tau:.4f} {'(significant)' if p_tau < 0.05 else '(not significant)'}")
    print(f"    ||j||: t={t_j:.2f}, p={p_j:.4f} {'(significant)' if p_j < 0.05 else '(not significant)'}")

    return {
        'noun_tau': noun_tau,
        'adj_tau': adj_tau,
        'verb_tau': verb_tau,
        'noun_j': noun_j,
        'adj_j': adj_j,
        'verb_j': verb_j,
        'p_tau': p_tau,
        'p_j': p_j,
    }


def exp8_3_transcendental_projection(loader):
    """
    Test: j[dim] flows from transcendental → adjective → noun
    For each j-dimension, find the related adjective and nouns.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8.3: Transcendental → Adjective → Noun Flow")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Define projection chains
    chains = {
        'beauty': {
            'transcendental': 'beauty',
            'adjectives': ['beautiful', 'gorgeous', 'pretty', 'lovely', 'attractive', 'elegant'],
            'nouns': ['flower', 'painting', 'sunset', 'woman', 'art', 'rose'],
        },
        'life': {
            'transcendental': 'life',
            'adjectives': ['alive', 'living', 'lively', 'vital', 'vibrant', 'animated'],
            'nouns': ['animal', 'plant', 'creature', 'organism', 'baby', 'heart'],
        },
        'sacred': {
            'transcendental': 'sacred',
            'adjectives': ['divine', 'holy', 'sacred', 'spiritual', 'blessed', 'heavenly'],
            'nouns': ['temple', 'church', 'altar', 'prayer', 'god', 'ritual'],
        },
        'good': {
            'transcendental': 'good',
            'adjectives': ['good', 'kind', 'generous', 'virtuous', 'noble', 'benevolent'],
            'nouns': ['kindness', 'charity', 'gift', 'help', 'deed', 'hero'],
        },
        'love': {
            'transcendental': 'love',
            'adjectives': ['loving', 'beloved', 'affectionate', 'caring', 'tender', 'devoted'],
            'nouns': ['mother', 'lover', 'heart', 'kiss', 'embrace', 'family'],
        },
    }

    results = {}

    for dim_idx, dim in enumerate(J_DIMS):
        chain = chains[dim]
        print(f"\n  {dim.upper()} chain:")

        # Get j[dim] for transcendental (should be 1.0 by definition)
        trans_word = chain['transcendental']
        if trans_word in vectors:
            trans_j = get_j_vector(vectors[trans_word])
            if trans_j is not None:
                print(f"    Transcendental '{trans_word}': j[{dim}] = {trans_j[dim_idx]:.3f}")

        # Get j[dim] for adjectives
        adj_values = []
        for adj in chain['adjectives']:
            if adj in vectors:
                j = get_j_vector(vectors[adj])
                if j is not None:
                    adj_values.append(j[dim_idx])
                    print(f"    Adjective '{adj}': j[{dim}] = {j[dim_idx]:.3f}")
        adj_mean = np.mean(adj_values) if adj_values else 0

        # Get j[dim] for nouns
        noun_values = []
        for noun in chain['nouns']:
            if noun in vectors:
                j = get_j_vector(vectors[noun])
                if j is not None:
                    noun_values.append(j[dim_idx])
                    print(f"    Noun '{noun}': j[{dim}] = {j[dim_idx]:.3f}")
        noun_mean = np.mean(noun_values) if noun_values else 0

        # Check flow
        print(f"\n    Flow: {dim} (1.0) → adj ({adj_mean:.3f}) → noun ({noun_mean:.3f})")
        if adj_mean > noun_mean:
            print(f"    ✓ Diminishing flow confirmed")
        else:
            print(f"    ? No clear flow (adj ≤ noun)")

        results[dim] = {
            'adj_mean': adj_mean,
            'noun_mean': noun_mean,
            'flow_confirmed': adj_mean > noun_mean,
        }

    return results


def exp8_4_verbs_as_operators(loader):
    """
    Test: Verbs have different τ based on their operator type.
    Lifting verbs: high τ
    Grounding verbs: low τ
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8.4: Verbs as Operators")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Define operator categories
    lifting_verbs = ['transcend', 'understand', 'contemplate', 'realize', 'enlighten',
                     'ascend', 'elevate', 'inspire', 'awaken', 'transform']
    grounding_verbs = ['get', 'take', 'make', 'use', 'put', 'give', 'find', 'hold',
                       'see', 'touch', 'grab', 'build', 'create', 'fix']
    neutral_verbs = ['know', 'feel', 'think', 'believe', 'learn', 'remember',
                     'hope', 'dream', 'imagine', 'wish']

    def analyze_verbs(verb_list, category):
        taus = []
        j_norms = []
        found = []
        for verb in verb_list:
            if verb in vectors:
                data = vectors[verb]
                if data.get('tau'):
                    taus.append(data['tau'])
                    j = get_j_vector(data)
                    if j is not None:
                        j_norms.append(np.linalg.norm(j))
                    found.append(verb)
        return taus, j_norms, found

    print(f"\n  Lifting operators (τ ↑):")
    lift_tau, lift_j, lift_found = analyze_verbs(lifting_verbs, 'lifting')
    if lift_tau:
        print(f"    Found: {', '.join(lift_found)}")
        print(f"    τ_mean = {np.mean(lift_tau):.3f}, ||j||_mean = {np.mean(lift_j):.3f}")

    print(f"\n  Grounding operators (τ ↓):")
    ground_tau, ground_j, ground_found = analyze_verbs(grounding_verbs, 'grounding')
    if ground_tau:
        print(f"    Found: {', '.join(ground_found)}")
        print(f"    τ_mean = {np.mean(ground_tau):.3f}, ||j||_mean = {np.mean(ground_j):.3f}")

    print(f"\n  Neutral operators:")
    neutral_tau, neutral_j, neutral_found = analyze_verbs(neutral_verbs, 'neutral')
    if neutral_tau:
        print(f"    Found: {', '.join(neutral_found)}")
        print(f"    τ_mean = {np.mean(neutral_tau):.3f}, ||j||_mean = {np.mean(neutral_j):.3f}")

    # Check hypothesis
    if lift_tau and ground_tau:
        lift_mean = np.mean(lift_tau)
        ground_mean = np.mean(ground_tau)
        print(f"\n  Operator hypothesis:")
        if lift_mean > ground_mean:
            print(f"    ✓ Lifting verbs higher τ ({lift_mean:.3f} > {ground_mean:.3f})")
        else:
            print(f"    ✗ Unexpected: lifting τ ≤ grounding τ")

    return {
        'lifting_tau': np.mean(lift_tau) if lift_tau else None,
        'grounding_tau': np.mean(ground_tau) if ground_tau else None,
        'neutral_tau': np.mean(neutral_tau) if neutral_tau else None,
    }


def exp8_5_semantic_holes(loader):
    """
    Find sparse regions in j-space (Mendeleev-style).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8.5: Semantic Holes (Mendeleev Grid)")
    print("=" * 60)

    vectors = loader.load_word_vectors()

    # Create 2D grid for beauty × life
    n_bins = 10
    grid = [[[] for _ in range(n_bins)] for _ in range(n_bins)]

    for word, data in vectors.items():
        j = get_j_vector(data)
        if j is None:
            continue

        # Normalize to [0, 1] range (assuming j values are in [-1, 1])
        beauty = (j[0] + 1) / 2  # j[beauty]
        life = (j[1] + 1) / 2    # j[life]

        # Clamp to valid range
        beauty = max(0, min(0.999, beauty))
        life = max(0, min(0.999, life))

        # Bin
        b_idx = int(beauty * n_bins)
        l_idx = int(life * n_bins)

        grid[l_idx][b_idx].append(word)

    # Count empty cells
    empty_cells = 0
    sparse_cells = 0
    total_words = 0

    print(f"\n  beauty × life grid ({n_bins}×{n_bins}):")
    print(f"\n  life↑")

    for l_idx in range(n_bins - 1, -1, -1):
        row = ""
        for b_idx in range(n_bins):
            count = len(grid[l_idx][b_idx])
            total_words += count
            if count == 0:
                empty_cells += 1
                row += "   ."
            elif count < 10:
                sparse_cells += 1
                row += f"  {count:2d}"
            else:
                row += f" {count:3d}"
        print(f"  {l_idx/n_bins:.1f} │{row}")

    print(f"      └" + "─" * (n_bins * 4 + 1) + "→ beauty")
    print(f"        " + "".join(f" {i/n_bins:.1f}" for i in range(n_bins)))

    print(f"\n  Statistics:")
    print(f"    Total cells: {n_bins * n_bins}")
    print(f"    Empty cells: {empty_cells} ({100 * empty_cells / (n_bins * n_bins):.1f}%)")
    print(f"    Sparse cells (<10): {sparse_cells} ({100 * sparse_cells / (n_bins * n_bins):.1f}%)")
    print(f"    Total words in grid: {total_words}")

    # Find example holes
    print(f"\n  Example semantic holes (empty regions):")
    for l_idx in range(n_bins):
        for b_idx in range(n_bins):
            if len(grid[l_idx][b_idx]) == 0:
                beauty_range = f"{b_idx/n_bins:.1f}-{(b_idx+1)/n_bins:.1f}"
                life_range = f"{l_idx/n_bins:.1f}-{(l_idx+1)/n_bins:.1f}"
                print(f"    beauty=[{beauty_range}], life=[{life_range}]")
                if empty_cells > 5:
                    break
        if empty_cells > 5:
            print(f"    ... and {empty_cells - 5} more")
            break

    return {
        'empty_cells': empty_cells,
        'sparse_cells': sparse_cells,
        'total_cells': n_bins * n_bins,
    }


def run_all_experiments():
    """Run all projection hierarchy experiments."""
    print("=" * 70)
    print("EXPERIMENT 8: PROJECTION HIERARCHY")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nHypothesis: j-space (5D) → Adjectives → Nouns")
    print("Verbs as operators between levels")

    all_results = {}

    try:
        loader = DataLoader()

        # Test 1: Word type distribution
        result = exp8_1_word_type_distribution(loader)
        all_results["word_types"] = result

        # Test 2: Projection hierarchy (τ and ||j||)
        result = exp8_2_projection_hierarchy(loader)
        all_results["hierarchy"] = result

        # Test 3: Transcendental flow
        result = exp8_3_transcendental_projection(loader)
        all_results["transcendental_flow"] = result

        # Test 4: Verbs as operators
        result = exp8_4_verbs_as_operators(loader)
        all_results["verbs_operators"] = result

        # Test 5: Semantic holes
        result = exp8_5_semantic_holes(loader)
        all_results["semantic_holes"] = result

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PROJECTION HIERARCHY")
    print("=" * 70)

    if 'hierarchy' in all_results:
        h = all_results['hierarchy']
        print(f"\n  τ by word type:")
        print(f"    Nouns:      {h['noun_tau']:.3f}")
        print(f"    Adjectives: {h['adj_tau']:.3f}")
        print(f"    Verbs:      {h['verb_tau']:.3f}")

        print(f"\n  ||j|| by word type:")
        print(f"    Nouns:      {h['noun_j']:.3f}")
        print(f"    Adjectives: {h['adj_j']:.3f}")
        print(f"    Verbs:      {h['verb_j']:.3f}")

        if h['adj_tau'] > h['noun_tau'] and h['adj_j'] > h['noun_j']:
            print(f"\n  ✓ PROJECTION HIERARCHY CONFIRMED")
            print(f"    Adjectives are between j-space and nouns")
        else:
            print(f"\n  ? Partial confirmation (check individual tests)")

    if 'verbs_operators' in all_results:
        v = all_results['verbs_operators']
        if v.get('lifting_tau') and v.get('grounding_tau'):
            if v['lifting_tau'] > v['grounding_tau']:
                print(f"\n  ✓ VERBS AS OPERATORS CONFIRMED")
                print(f"    Lifting: τ={v['lifting_tau']:.3f}")
                print(f"    Grounding: τ={v['grounding_tau']:.3f}")

    if 'semantic_holes' in all_results:
        s = all_results['semantic_holes']
        print(f"\n  Semantic Holes:")
        print(f"    Empty: {s['empty_cells']}/{s['total_cells']} "
              f"({100*s['empty_cells']/s['total_cells']:.1f}%)")

    print("\n" + "=" * 70)

    # Save results
    output_dir = _EXPERIMENT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp8_projection_hierarchy_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
