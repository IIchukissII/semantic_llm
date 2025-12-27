#!/usr/bin/env python3
"""
Comprehensive Experiments: Understanding Semantic Direction

Tests how verb properties (i, Δτ, Δg) affect navigation and meaning.
"""

import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
from collections import defaultdict

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from chain_core.intent_collapse import IntentCollapse
from chain_core.semantic_laser import SemanticLaser, KT_NATURAL


def experiment_1_grounding_vs_ascending(graph):
    """
    Experiment 1: Do grounding verbs (Δτ < 0) lead to more concrete concepts?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Grounding vs Ascending Verbs")
    print("=" * 70)
    print("Hypothesis: Grounding verbs (love, find, give) should lead to")
    print("            lower-τ (more concrete) concepts than ascending verbs.")

    # Get grounding and ascending verbs
    with graph.driver.session() as session:
        # Grounding verbs (Δτ < -0.1)
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.delta_tau < -0.1 AND v.transition_count > 20
            RETURN v.verb as verb, v.delta_tau as dt
            ORDER BY v.delta_tau ASC
            LIMIT 5
        """)
        grounding = [(r['verb'], r['dt']) for r in result]

        # Ascending verbs (Δτ > 0.2)
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.delta_tau > 0.2 AND v.transition_count > 10
            RETURN v.verb as verb, v.delta_tau as dt
            ORDER BY v.delta_tau DESC
            LIMIT 5
        """)
        ascending = [(r['verb'], r['dt']) for r in result]

    print(f"\nGrounding verbs: {[(v, f'{dt:.2f}') for v, dt in grounding]}")
    print(f"Ascending verbs: {[(v, f'{dt:.2f}') for v, dt in ascending]}")

    # Test navigation with each type
    collapse = IntentCollapse(graph)
    seeds = ['meaning', 'truth', 'life']

    results = {'grounding': [], 'ascending': []}

    for verb_type, verbs in [('grounding', grounding), ('ascending', ascending)]:
        verb_list = [v for v, _ in verbs]
        collapse.set_intent(verb_list)

        nav_result = collapse.navigate(seeds, n_walks=5, steps_per_walk=6)

        avg_tau = np.mean([s.tau for s in nav_result.states]) if nav_result.states else 0
        concepts = list(set(s.word for s in nav_result.states))[:10]

        results[verb_type] = {
            'verbs': verb_list,
            'avg_tau': avg_tau,
            'concepts': concepts,
            'collapse_ratio': nav_result.collapse_ratio
        }

        print(f"\n{verb_type.upper()}:")
        print(f"  Avg τ of reached concepts: {avg_tau:.3f}")
        print(f"  Concepts: {concepts[:8]}")

    # Analysis
    tau_diff = results['ascending']['avg_tau'] - results['grounding']['avg_tau']
    print(f"\n→ Δτ difference: {tau_diff:+.3f}")
    if tau_diff > 0.1:
        print("✓ CONFIRMED: Ascending verbs lead to higher-τ (more abstract) concepts")
    elif tau_diff < -0.1:
        print("✗ REVERSED: Grounding verbs unexpectedly lead to higher-τ concepts")
    else:
        print("~ INCONCLUSIVE: No significant difference")

    return results


def experiment_2_life_vs_knowledge(graph):
    """
    Experiment 2: Life-affirming (i[life]>0) vs Knowledge-seeking (i[love]<0)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Life-Affirming vs Knowledge-Seeking Verbs")
    print("=" * 70)
    print("Hypothesis: Life-affirming verbs should lead to concepts with higher g")
    print("            (more 'good'), knowledge verbs to more neutral concepts.")

    with graph.driver.session() as session:
        # Life-affirming (i[1] > 0.1, life dimension)
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.i[1] > 0.1 AND v.transition_count > 10
            RETURN v.verb as verb, v.i[1] as life_score
            ORDER BY life_score DESC
            LIMIT 5
        """)
        life_verbs = [(r['verb'], r['life_score']) for r in result]

        # Knowledge-seeking (i[4] < -0.1, love dimension negative)
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.i[4] < -0.1 AND v.transition_count > 10
            RETURN v.verb as verb, v.i[4] as love_score
            ORDER BY love_score ASC
            LIMIT 5
        """)
        knowledge_verbs = [(r['verb'], r['love_score']) for r in result]

    print(f"\nLife-affirming verbs: {[(v, f'{s:.2f}') for v, s in life_verbs]}")
    print(f"Knowledge-seeking verbs: {[(v, f'{s:.2f}') for v, s in knowledge_verbs]}")

    collapse = IntentCollapse(graph)
    seeds = ['wisdom', 'soul', 'heart']

    results = {}
    for verb_type, verbs in [('life', life_verbs), ('knowledge', knowledge_verbs)]:
        verb_list = [v for v, _ in verbs]
        collapse.set_intent(verb_list)

        nav_result = collapse.navigate(seeds, n_walks=5, steps_per_walk=6)

        avg_g = np.mean([s.g for s in nav_result.states]) if nav_result.states else 0
        concepts = list(set(s.word for s in nav_result.states))[:10]

        results[verb_type] = {
            'verbs': verb_list,
            'avg_g': avg_g,
            'concepts': concepts
        }

        print(f"\n{verb_type.upper()}:")
        print(f"  Avg g of reached concepts: {avg_g:+.3f}")
        print(f"  Concepts: {concepts[:8]}")

    g_diff = results['life']['avg_g'] - results['knowledge']['avg_g']
    print(f"\n→ Δg difference: {g_diff:+.3f}")
    if g_diff > 0.1:
        print("✓ CONFIRMED: Life-affirming verbs lead to more 'good' concepts")
    else:
        print("~ Results need deeper analysis")

    return results


def experiment_3_orthogonal_verbs(graph):
    """
    Experiment 3: Do orthogonal verbs lead to different semantic regions?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Orthogonal Verb Pairs")
    print("=" * 70)
    print("Hypothesis: Orthogonal verbs (create/destroy, rise/fall) should")
    print("            lead to NON-OVERLAPPING concept sets.")

    pairs = [
        ('create', 'destroy'),
        ('rise', 'fall'),
        ('find', 'lose'),
        ('begin', 'end')
    ]

    collapse = IntentCollapse(graph)
    seeds = ['life', 'meaning', 'world']

    results = {}
    for v1, v2 in pairs:
        concepts = {}
        for verb in [v1, v2]:
            collapse.set_intent([verb])
            nav_result = collapse.navigate(seeds, n_walks=5, steps_per_walk=6)
            concepts[verb] = set(s.word for s in nav_result.states)

        overlap = concepts[v1] & concepts[v2]
        union = concepts[v1] | concepts[v2]
        jaccard = len(overlap) / len(union) if union else 0

        results[f"{v1}/{v2}"] = {
            v1: list(concepts[v1])[:8],
            v2: list(concepts[v2])[:8],
            'overlap': list(overlap)[:5],
            'jaccard': jaccard
        }

        print(f"\n{v1.upper()} / {v2.upper()}:")
        print(f"  {v1}: {list(concepts[v1])[:6]}")
        print(f"  {v2}: {list(concepts[v2])[:6]}")
        print(f"  Overlap: {list(overlap)[:5]}")
        print(f"  Jaccard similarity: {jaccard:.2%}")

        if jaccard < 0.3:
            print(f"  ✓ LOW OVERLAP - verbs lead to different regions")
        else:
            print(f"  ~ HIGH OVERLAP - verbs share semantic territory")

    return results


def experiment_4_moral_dimension(graph):
    """
    Experiment 4: Explore moral push (Δg) effects
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Moral Dimension (Δg)")
    print("=" * 70)
    print("Hypothesis: Verbs with Δg > 0 should lead to 'good' concepts,")
    print("            Δg < 0 should lead to morally complex concepts.")

    with graph.driver.session() as session:
        # Good-pushing verbs
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.delta_g > 0.2 AND v.transition_count > 10
            RETURN v.verb as verb, v.delta_g as dg
            ORDER BY dg DESC
            LIMIT 5
        """)
        good_verbs = [(r['verb'], r['dg']) for r in result]

        # Evil-pushing verbs
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.delta_g < -0.3 AND v.transition_count > 10
            RETURN v.verb as verb, v.delta_g as dg
            ORDER BY dg ASC
            LIMIT 5
        """)
        dark_verbs = [(r['verb'], r['dg']) for r in result]

    print(f"\nGood-pushing verbs: {[(v, f'{dg:+.2f}') for v, dg in good_verbs]}")
    print(f"Dark-pushing verbs: {[(v, f'{dg:+.2f}') for v, dg in dark_verbs]}")

    collapse = IntentCollapse(graph)
    seeds = ['power', 'desire', 'fate']

    results = {}
    for verb_type, verbs in [('good', good_verbs), ('dark', dark_verbs)]:
        verb_list = [v for v, _ in verbs]
        collapse.set_intent(verb_list)

        nav_result = collapse.navigate(seeds, n_walks=5, steps_per_walk=6)

        avg_g = np.mean([s.g for s in nav_result.states]) if nav_result.states else 0
        concepts = list(set(s.word for s in nav_result.states))[:10]

        results[verb_type] = {
            'verbs': verb_list,
            'avg_g': avg_g,
            'concepts': concepts
        }

        print(f"\n{verb_type.upper()} VERBS:")
        print(f"  Avg g: {avg_g:+.3f}")
        print(f"  Concepts: {concepts[:8]}")

    return results


def experiment_5_i_vector_clustering(graph):
    """
    Experiment 5: Do verbs cluster by i-vector similarity?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Verb Clustering by i-Vector")
    print("=" * 70)
    print("Question: Do semantically similar verbs cluster together?")

    dims = ['beauty', 'life', 'sacred', 'good', 'love']

    with graph.driver.session() as session:
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.i IS NOT NULL AND v.transition_count > 20
            RETURN v.verb as verb, v.i as i
        """)

        verbs_data = [(r['verb'], np.array(r['i'])) for r in result]

    # Find dominant dimension for each verb
    clusters = defaultdict(list)
    for verb, i in verbs_data:
        abs_i = np.abs(i)
        dom_idx = np.argmax(abs_i)
        sign = '+' if i[dom_idx] > 0 else '-'
        cluster = f"{sign}{dims[dom_idx]}"
        clusters[cluster].append((verb, i[dom_idx]))

    print("\nVerb clusters by dominant i-dimension:")
    for cluster, verbs in sorted(clusters.items()):
        verbs_sorted = sorted(verbs, key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"\n  {cluster}:")
        for v, score in verbs_sorted:
            print(f"    {v}: {score:+.3f}")

    return dict(clusters)


def experiment_6_semantic_laser_comparison(graph):
    """
    Experiment 6: Compare laser output with different verb intents
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Semantic Laser with Different Intents")
    print("=" * 70)

    laser = SemanticLaser(graph)
    seeds = ['dream', 'hope', 'fear']

    intent_sets = [
        (['love', 'embrace', 'accept'], "Acceptance"),
        (['understand', 'learn', 'know'], "Knowledge"),
        (['create', 'build', 'make'], "Creation"),
        (['destroy', 'break', 'end'], "Destruction"),
    ]

    results = {}
    for verbs, name in intent_sets:
        result = laser.lase(
            seeds=seeds,
            pump_power=10,
            pump_depth=5,
            coherence_threshold=0.3,
            intent_verbs=verbs
        )

        beams = result['beams']
        pop = result['population']
        metrics = result.get('metrics', {})

        primary = beams[0] if beams else None

        results[name] = {
            'verbs': verbs,
            'beams': len(beams),
            'coherence': primary.coherence if primary else 0,
            'concepts': primary.concepts[:6] if primary else [],
            'g_polarity': primary.g_polarity if primary else 0,
            'tau_mean': pop['tau_mean'],
            'lasing': metrics.get('lasing_achieved', False)
        }

        print(f"\n{name.upper()} intent ({verbs}):")
        print(f"  Beams: {len(beams)}, Coherence: {results[name]['coherence']:.2f}")
        print(f"  Lasing: {'✓' if results[name]['lasing'] else '✗'}")
        print(f"  Concepts: {results[name]['concepts']}")
        print(f"  g-polarity: {results[name]['g_polarity']:+.2f}, τ: {results[name]['tau_mean']:.2f}")

    laser.close()
    return results


def save_results(all_results):
    """Save all experiment results to JSON."""
    output_dir = _THIS_FILE.parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"experiments_{timestamp}.json"

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, set):
            return list(obj)
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert(obj)

    with open(filename, 'w') as f:
        json.dump(deep_convert(all_results), f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


def main():
    print("=" * 70)
    print("COMPREHENSIVE EXPERIMENTS: Understanding Semantic Direction")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    graph = MeaningGraph()

    if not graph.is_connected():
        print("Not connected to Neo4j")
        return

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {}
    }

    # Run all experiments
    all_results['experiments']['1_grounding_ascending'] = experiment_1_grounding_vs_ascending(graph)
    all_results['experiments']['2_life_knowledge'] = experiment_2_life_vs_knowledge(graph)
    all_results['experiments']['3_orthogonal'] = experiment_3_orthogonal_verbs(graph)
    all_results['experiments']['4_moral'] = experiment_4_moral_dimension(graph)
    all_results['experiments']['5_clustering'] = experiment_5_i_vector_clustering(graph)
    all_results['experiments']['6_laser'] = experiment_6_semantic_laser_comparison(graph)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("""
Key Findings to Look For:
1. Grounding vs Ascending: Does Δτ predict concept abstraction level?
2. Life vs Knowledge: Does i-vector dimension predict semantic flavor?
3. Orthogonal Verbs: Do they lead to different semantic regions?
4. Moral Dimension: Does Δg predict concept morality?
5. Clustering: Do similar verbs cluster by i-vector?
6. Laser: How does intent affect coherent beam output?
""")

    save_results(all_results)
    graph.close()

    print("\nExperiments complete!")


if __name__ == "__main__":
    main()
