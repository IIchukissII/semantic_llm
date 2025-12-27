#!/usr/bin/env python3
"""
Store Complete Verb Properties in Neo4j

Computes and stores the dual structure of verbs (parallel to nouns):

  NOUN                    VERB
  ----                    ----
  τ (tau)                 Δτ (delta_tau) - ascend/descend effect
  g (good/evil)           Δg (delta_g) - moral push direction
  j (5D position)         i (i_vector) - intrinsic action type (centered j)
                          j (j_vector) - effect direction (raw)

The "phase shift" insight: i_vector = j_vector - global_mean
This makes orthogonal verbs (create/destroy) truly perpendicular.
"""

import numpy as np
from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph


def compute_global_j_mean(graph: MeaningGraph) -> np.ndarray:
    """Compute the global mean j-vector across all VerbOperators."""
    with graph.driver.session() as session:
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.j IS NOT NULL
            RETURN v.j as j
        """)

        all_j = []
        for record in result:
            j = record['j']
            if j and len(j) >= 5:
                all_j.append(np.array(j[:5]))

        if all_j:
            return np.mean(all_j, axis=0)
        return np.zeros(5)


def compute_verb_transition_stats(graph: MeaningGraph, verb: str) -> dict:
    """Compute Δτ and Δg from VIA transitions for a verb."""
    with graph.driver.session() as session:
        result = session.run("""
            MATCH (src:Concept)-[r:VIA {verb: $verb}]->(tgt:Concept)
            WHERE src.tau IS NOT NULL AND tgt.tau IS NOT NULL
            RETURN avg(tgt.tau - src.tau) as delta_tau,
                   avg(tgt.g - src.g) as delta_g,
                   avg(src.tau) as avg_src_tau,
                   avg(tgt.tau) as avg_tgt_tau,
                   count(*) as edge_count
        """, verb=verb)

        record = result.single()
        if record and record['edge_count'] > 0:
            return {
                'delta_tau': record['delta_tau'] or 0.0,
                'delta_g': record['delta_g'] or 0.0,
                'avg_src_tau': record['avg_src_tau'] or 0.0,
                'avg_tgt_tau': record['avg_tgt_tau'] or 0.0,
                'edge_count': record['edge_count']
            }
        return None


def get_all_verb_operators(graph: MeaningGraph) -> list:
    """Get all VerbOperator verbs."""
    with graph.driver.session() as session:
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.j IS NOT NULL
            RETURN v.verb as verb, v.j as j
        """)
        return [(r['verb'], np.array(r['j'][:5])) for r in result if r['j'] and len(r['j']) >= 5]


def store_verb_properties(graph: MeaningGraph, verb: str,
                          i_vector: list, delta_tau: float, delta_g: float,
                          edge_count: int):
    """Store computed properties on VerbOperator node."""
    with graph.driver.session() as session:
        session.run("""
            MATCH (v:VerbOperator {verb: $verb})
            SET v.i = $i_vector,
                v.delta_tau = $delta_tau,
                v.delta_g = $delta_g,
                v.transition_count = $edge_count,
                v.phase_shifted = true
        """, verb=verb, i_vector=i_vector, delta_tau=delta_tau,
             delta_g=delta_g, edge_count=edge_count)


def store_global_mean(graph: MeaningGraph, global_mean: list):
    """Store global mean as a special node for reference."""
    with graph.driver.session() as session:
        session.run("""
            MERGE (g:GlobalMean {name: 'verb_j_mean'})
            SET g.j_mean = $mean,
                g.updated_at = datetime()
        """, mean=global_mean)


def main():
    print("=" * 70)
    print("STORING VERB DUAL PROPERTIES IN NEO4J")
    print("=" * 70)

    graph = MeaningGraph()

    if not graph.is_connected():
        print("Not connected to Neo4j")
        return

    # Step 1: Compute global mean
    print("\n[1/4] Computing global j-vector mean...")
    global_mean = compute_global_j_mean(graph)
    print(f"  Global mean: {global_mean}")

    # Store global mean
    store_global_mean(graph, global_mean.tolist())
    print("  Stored as GlobalMean node")

    # Step 2: Get all VerbOperators
    print("\n[2/4] Getting all VerbOperators...")
    verbs = get_all_verb_operators(graph)
    print(f"  Found {len(verbs)} VerbOperators with j-vectors")

    # Step 3: Compute and store properties
    print("\n[3/4] Computing verb properties...")

    updated = 0
    skipped = 0
    dims = ['beauty', 'life', 'sacred', 'good', 'love']

    for verb, j_raw in verbs:
        # Compute i-vector (phase-shifted j)
        i_vector = (j_raw - global_mean).tolist()

        # Get transition stats
        stats = compute_verb_transition_stats(graph, verb)

        if stats:
            store_verb_properties(
                graph, verb,
                i_vector=i_vector,
                delta_tau=stats['delta_tau'],
                delta_g=stats['delta_g'],
                edge_count=stats['edge_count']
            )
            updated += 1
        else:
            # No transitions, store i-vector only with zero deltas
            store_verb_properties(
                graph, verb,
                i_vector=i_vector,
                delta_tau=0.0,
                delta_g=0.0,
                edge_count=0
            )
            skipped += 1

    print(f"  Updated: {updated} verbs with full properties")
    print(f"  Skipped: {skipped} verbs (no transitions, i-vector only)")

    # Step 4: Verify
    print("\n[4/4] Verifying stored properties...")

    with graph.driver.session() as session:
        result = session.run("""
            MATCH (v:VerbOperator)
            WHERE v.i IS NOT NULL
            RETURN v.verb as verb, v.i as i, v.delta_tau as dt, v.delta_g as dg,
                   v.transition_count as tc
            ORDER BY v.transition_count DESC
            LIMIT 10
        """)

        print(f"\n  {'Verb':<15} {'i-dominant':<12} {'Δτ':>8} {'Δg':>8} {'edges':>8}")
        print("  " + "-" * 55)

        for r in result:
            verb = r['verb']
            i = np.array(r['i'])
            dt = r['dt'] or 0
            dg = r['dg'] or 0
            tc = r['tc'] or 0

            # Find dominant i dimension
            abs_i = np.abs(i)
            dom_idx = np.argmax(abs_i)
            sign = '+' if i[dom_idx] > 0 else '-'

            print(f"  {verb:<15} {sign}{dims[dom_idx]:<11} {dt:>+8.3f} {dg:>+8.3f} {tc:>8}")

    # Summary
    print("\n" + "=" * 70)
    print("SCHEMA UPDATE COMPLETE")
    print("=" * 70)
    print("""
VerbOperator nodes now have:
  - j: raw j-vector (5D)
  - i: centered j-vector (phase-shifted, 5D)
  - delta_tau: abstraction effect (↑ ascend / ↓ descend)
  - delta_g: moral push direction
  - transition_count: number of VIA edges
  - phase_shifted: true (flag indicating new properties)

GlobalMean node stores:
  - j_mean: the global mean for phase shifting
""")

    graph.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
