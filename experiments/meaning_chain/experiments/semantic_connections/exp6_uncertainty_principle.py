#!/usr/bin/env python3
"""
Experiment 6: Semantic Uncertainty Principle
=============================================

Tests the hypothesis:

    Δτ × Δj ≥ κ  (semantic uncertainty)

Analogous to:
    Δt × Δω ≥ 1/2  (Fourier)
    Δx × Δp ≥ ℏ/2  (Quantum)

The idea: precision in τ (abstraction level) trades off with
precision in j (semantic direction).

Experiments:
1. Sample concepts, compute Δτ and Δj for neighborhoods
2. Test if tight τ-bands have wide j-spreads
3. Compute the uncertainty product Δτ × Δj
4. Find the minimum (the semantic "ℏ")
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


def compute_j_spread(j_vectors):
    """
    Compute spread (uncertainty) in j-vector space.

    Δj = average distance from centroid
    """
    if len(j_vectors) < 2:
        return 0.0

    j_array = np.array(j_vectors)
    centroid = np.mean(j_array, axis=0)

    # Average distance from centroid
    distances = [np.linalg.norm(j - centroid) for j in j_array]
    return np.mean(distances)


def compute_j_variance(j_vectors):
    """
    Compute variance in j-vector directions.

    Uses angular spread (cosine similarity variance).
    """
    if len(j_vectors) < 2:
        return 0.0

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(len(j_vectors)):
        for k in range(i + 1, len(j_vectors)):
            v1, v2 = j_vectors[i], j_vectors[k]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                sim = np.dot(v1, v2) / (n1 * n2)
                similarities.append(sim)

    if not similarities:
        return 0.0

    # Variance in similarity (lower = more spread)
    # Convert to spread: 1 - mean_similarity
    return 1.0 - np.mean(similarities)


def exp6_1_concept_neighborhoods(graph):
    """
    Test 1: Compute Δτ × Δj for concept neighborhoods.

    For each concept, look at its neighbors and compute:
    - Δτ = std of τ values in neighborhood
    - Δj = spread of j-vectors in neighborhood
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6.1: Concept Neighborhood Uncertainty")
    print("=" * 60)

    test_concepts = [
        "wisdom", "love", "time", "life", "truth",
        "knowledge", "beauty", "death", "god", "soul",
        "mind", "heart", "world", "nature", "spirit",
        "man", "woman", "child", "father", "mother",
    ]

    results = []
    uncertainty_products = []

    for concept in test_concepts:
        props = graph.get_concept(concept)
        if not props:
            continue

        concept_tau = props.get('tau', 1.5)
        concept_j = props.get('j', [0, 0, 0, 0, 0])
        if isinstance(concept_j, list):
            concept_j = np.array(concept_j)

        # Get neighbors
        transitions = graph.get_all_transitions(concept, limit=20)

        neighbor_taus = [concept_tau]
        neighbor_js = [concept_j]

        for verb, target, weight in transitions:
            target_props = graph.get_concept(target)
            if target_props:
                neighbor_taus.append(target_props.get('tau', 1.5))
                j = target_props.get('j', [0, 0, 0, 0, 0])
                if isinstance(j, list):
                    j = np.array(j)
                neighbor_js.append(j)

        if len(neighbor_taus) < 3:
            continue

        # Compute uncertainties
        delta_tau = np.std(neighbor_taus)
        delta_j = compute_j_spread(neighbor_js)
        j_variance = compute_j_variance(neighbor_js)

        # Uncertainty product
        uncertainty = delta_tau * delta_j
        uncertainty_products.append(uncertainty)

        results.append({
            "concept": concept,
            "tau": concept_tau,
            "n_neighbors": len(neighbor_taus) - 1,
            "delta_tau": delta_tau,
            "delta_j": delta_j,
            "j_variance": j_variance,
            "uncertainty": uncertainty,
        })

        print(f"\n  {concept}:")
        print(f"    τ = {concept_tau:.2f}, neighbors = {len(neighbor_taus)-1}")
        print(f"    Δτ = {delta_tau:.4f}")
        print(f"    Δj = {delta_j:.4f}")
        print(f"    Δτ × Δj = {uncertainty:.6f}")

    # Find minimum uncertainty (semantic "ℏ")
    if uncertainty_products:
        min_uncertainty = min(uncertainty_products)
        mean_uncertainty = np.mean(uncertainty_products)

        print(f"\n{'=' * 60}")
        print("UNCERTAINTY ANALYSIS")
        print("=" * 60)
        print(f"  Minimum Δτ × Δj = {min_uncertainty:.6f} (semantic κ?)")
        print(f"  Mean Δτ × Δj = {mean_uncertainty:.6f}")
        print(f"  Max Δτ × Δj = {max(uncertainty_products):.6f}")

    return results, min_uncertainty if uncertainty_products else 0


def exp6_2_tau_bands(graph):
    """
    Test 2: Do tight τ-bands have wide j-spreads?

    Group concepts by τ level, compute j-spread within each band.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6.2: τ-Bands vs j-Spread")
    print("=" * 60)

    # Sample many concepts
    sample_concepts = [
        "wisdom", "love", "time", "life", "truth", "knowledge",
        "beauty", "death", "god", "soul", "mind", "heart",
        "world", "nature", "spirit", "man", "woman", "child",
        "father", "mother", "king", "war", "peace", "hope",
        "fear", "joy", "pain", "light", "darkness", "fire",
        "water", "earth", "sky", "sun", "moon", "star",
        "tree", "flower", "bird", "fish", "stone", "gold",
    ]

    # Collect τ and j for each concept
    concept_data = []

    for concept in sample_concepts:
        props = graph.get_concept(concept)
        if not props:
            continue

        tau = props.get('tau', 1.5)
        j = props.get('j', [0, 0, 0, 0, 0])
        if isinstance(j, list):
            j = np.array(j)

        concept_data.append({
            "concept": concept,
            "tau": tau,
            "j": j,
        })

    # Define τ-bands
    bands = [
        ("ground", 1.0, 1.5),
        ("everyday", 1.5, 2.0),
        ("meaningful", 2.0, 2.5),
        ("transcendent", 2.5, 4.0),
    ]

    results = []

    print(f"\n  Sampled {len(concept_data)} concepts")
    print(f"\n  τ-band analysis:")

    for band_name, tau_min, tau_max in bands:
        # Filter concepts in this band
        band_concepts = [c for c in concept_data
                        if tau_min <= c["tau"] < tau_max]

        if len(band_concepts) < 2:
            print(f"    {band_name}: insufficient data")
            continue

        # Compute τ spread (should be small for tight bands)
        taus = [c["tau"] for c in band_concepts]
        delta_tau = np.std(taus)

        # Compute j spread
        js = [c["j"] for c in band_concepts]
        delta_j = compute_j_spread(js)
        j_var = compute_j_variance(js)

        uncertainty = delta_tau * delta_j

        results.append({
            "band": band_name,
            "tau_range": (tau_min, tau_max),
            "n_concepts": len(band_concepts),
            "delta_tau": delta_tau,
            "delta_j": delta_j,
            "j_variance": j_var,
            "uncertainty": uncertainty,
        })

        print(f"    {band_name} [{tau_min:.1f}, {tau_max:.1f}): "
              f"n={len(band_concepts)}, Δτ={delta_tau:.3f}, Δj={delta_j:.3f}, "
              f"Δτ×Δj={uncertainty:.4f}")

    # Check if tighter bands have wider j-spread
    if len(results) >= 2:
        # Sort by delta_tau (ascending = tighter)
        sorted_results = sorted(results, key=lambda x: x["delta_tau"])

        print(f"\n  Checking uncertainty principle:")
        print(f"  (Tighter τ should have wider j)")

        for r in sorted_results:
            print(f"    Δτ={r['delta_tau']:.3f} → Δj={r['delta_j']:.3f} "
                  f"(product={r['uncertainty']:.4f})")

    return results


def exp6_3_navigation_uncertainty(graph):
    """
    Test 3: Navigation results and uncertainty.

    Compare GROUNDED (tight τ) vs DEEP (wide τ) navigation
    and measure j-spread in results.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6.3: Navigation Uncertainty")
    print("=" * 60)

    from chain_core.navigator import SemanticNavigator

    nav = SemanticNavigator()

    test_queries = [
        "What is wisdom?",
        "What is love?",
        "What is truth?",
    ]

    goals = ["grounded", "deep", "resonant", "filtered"]

    results = []

    try:
        for query in test_queries:
            print(f"\n  Query: {query}")

            query_results = []

            for goal in goals:
                result = nav.navigate(query, goal=goal)

                # Get j-vectors for result concepts
                j_vectors = []
                taus = []

                for concept in result.concepts[:8]:
                    props = graph.get_concept(concept)
                    if props:
                        taus.append(props.get('tau', 1.5))
                        j = props.get('j', [0, 0, 0, 0, 0])
                        if isinstance(j, list):
                            j = np.array(j)
                        j_vectors.append(j)

                if len(j_vectors) < 2:
                    continue

                delta_tau = np.std(taus)
                delta_j = compute_j_spread(j_vectors)
                uncertainty = delta_tau * delta_j

                query_results.append({
                    "goal": goal,
                    "tau_mean": result.quality.tau_mean,
                    "delta_tau": delta_tau,
                    "delta_j": delta_j,
                    "uncertainty": uncertainty,
                })

                print(f"    {goal:12s}: Δτ={delta_tau:.3f}, Δj={delta_j:.3f}, "
                      f"Δτ×Δj={uncertainty:.4f}")

            results.append({
                "query": query,
                "goals": query_results,
            })

    finally:
        nav.close()

    # Analyze: does GROUNDED (low τ) have higher j-spread?
    print(f"\n{'=' * 60}")
    print("UNCERTAINTY PRINCIPLE CHECK")
    print("=" * 60)

    grounded_uncertainties = []
    deep_uncertainties = []

    for qr in results:
        for gr in qr["goals"]:
            if gr["goal"] == "grounded":
                grounded_uncertainties.append(gr)
            elif gr["goal"] == "deep":
                deep_uncertainties.append(gr)

    if grounded_uncertainties and deep_uncertainties:
        avg_grounded = np.mean([g["uncertainty"] for g in grounded_uncertainties])
        avg_deep = np.mean([g["uncertainty"] for g in deep_uncertainties])

        print(f"  GROUNDED avg uncertainty: {avg_grounded:.4f}")
        print(f"  DEEP avg uncertainty: {avg_deep:.4f}")

        print(f"\n  GROUNDED (tight τ): Δτ={np.mean([g['delta_tau'] for g in grounded_uncertainties]):.3f}, "
              f"Δj={np.mean([g['delta_j'] for g in grounded_uncertainties]):.3f}")
        print(f"  DEEP (wide τ): Δτ={np.mean([g['delta_tau'] for g in deep_uncertainties]):.3f}, "
              f"Δj={np.mean([g['delta_j'] for g in deep_uncertainties]):.3f}")

    return results


def exp6_4_find_semantic_constant(graph):
    """
    Test 4: Find the semantic uncertainty constant κ.

    Sample many concept pairs and find minimum Δτ × Δj.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6.4: Semantic Constant κ")
    print("=" * 60)

    # Get a broad sample of concepts
    sample_words = [
        "wisdom", "love", "time", "life", "truth", "knowledge",
        "beauty", "death", "god", "soul", "mind", "heart", "fear",
        "hope", "joy", "pain", "light", "fire", "water", "earth",
        "man", "woman", "king", "war", "peace", "word", "name",
        "eye", "hand", "day", "night", "year", "way", "world",
    ]

    all_concepts = []

    for word in sample_words:
        props = graph.get_concept(word)
        if props:
            tau = props.get('tau', 1.5)
            j = props.get('j', [0, 0, 0, 0, 0])
            if isinstance(j, list):
                j = np.array(j)
            all_concepts.append({"word": word, "tau": tau, "j": j})

    print(f"  Sampled {len(all_concepts)} concepts")

    # Compute pairwise uncertainties
    pair_uncertainties = []

    for i in range(len(all_concepts)):
        for k in range(i + 1, len(all_concepts)):
            c1, c2 = all_concepts[i], all_concepts[k]

            delta_tau = abs(c1["tau"] - c2["tau"])
            delta_j = np.linalg.norm(c1["j"] - c2["j"])

            if delta_tau > 0.01 and delta_j > 0.01:  # Avoid trivial pairs
                uncertainty = delta_tau * delta_j
                pair_uncertainties.append({
                    "pair": (c1["word"], c2["word"]),
                    "delta_tau": delta_tau,
                    "delta_j": delta_j,
                    "uncertainty": uncertainty,
                })

    if pair_uncertainties:
        # Sort by uncertainty
        pair_uncertainties.sort(key=lambda x: x["uncertainty"])

        # Find minimum
        min_pair = pair_uncertainties[0]

        print(f"\n  Minimum uncertainty pair:")
        print(f"    {min_pair['pair']}")
        print(f"    Δτ = {min_pair['delta_tau']:.4f}")
        print(f"    Δj = {min_pair['delta_j']:.4f}")
        print(f"    Δτ × Δj = {min_pair['uncertainty']:.6f}")

        # Statistics
        uncertainties = [p["uncertainty"] for p in pair_uncertainties]

        print(f"\n  Uncertainty statistics:")
        print(f"    Min:    κ = {min(uncertainties):.6f}")
        print(f"    Mean:   {np.mean(uncertainties):.6f}")
        print(f"    Median: {np.median(uncertainties):.6f}")
        print(f"    Max:    {max(uncertainties):.6f}")

        # Check distribution
        below_threshold = sum(1 for u in uncertainties if u < 0.01)
        print(f"\n  Pairs with Δτ×Δj < 0.01: {below_threshold}/{len(uncertainties)}")

        # The semantic constant
        kappa = min(uncertainties)
        print(f"\n{'=' * 60}")
        print(f"  SEMANTIC CONSTANT κ ≈ {kappa:.6f}")
        print(f"  (Δτ × Δj ≥ κ)")
        print("=" * 60)

        return pair_uncertainties, kappa

    return [], 0.0


def run_all_experiments():
    """Run all uncertainty principle experiments."""
    print("=" * 70)
    print("EXPERIMENT 6: SEMANTIC UNCERTAINTY PRINCIPLE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nHypothesis: Δτ × Δj ≥ κ")
    print("(Precision in τ trades off with precision in j)")

    all_results = {}

    try:
        from graph.meaning_graph import MeaningGraph
        graph = MeaningGraph()

        if not graph.is_connected():
            print("\nERROR: Neo4j not connected")
            return None

        # Test 1: Concept neighborhoods
        results, min_u = exp6_1_concept_neighborhoods(graph)
        all_results["exp6_1_neighborhoods"] = results

        # Test 2: τ-bands vs j-spread
        results = exp6_2_tau_bands(graph)
        all_results["exp6_2_tau_bands"] = results

        # Test 3: Navigation uncertainty
        results = exp6_3_navigation_uncertainty(graph)
        all_results["exp6_3_navigation"] = results

        # Test 4: Find semantic constant
        results, kappa = exp6_4_find_semantic_constant(graph)
        all_results["exp6_4_semantic_constant"] = {"kappa": kappa}

        graph.close()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Save results
    output_dir = _EXPERIMENT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp6_uncertainty_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
