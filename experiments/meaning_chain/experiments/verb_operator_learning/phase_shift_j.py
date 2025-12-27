#!/usr/bin/env python3
"""
Phase-Shifted J-Vector Similarity

The Pirate Insight:
    All VerbOperator j-vectors are biased toward [-0.82, -0.97, -0.92, -0.80, -0.95]
    This makes opposite verbs have 99% cosine similarity (useless!)

    Solution: "Shift the phase" by centering - subtract the global mean.

    Mathematical effect:
        cos(θ) → cos(θ - φ) where φ is the bias angle
        This is like using sin when cos is biased toward 1

    Result:
        love/hate: 0.99 → -0.22 (NOW OPPOSITE!)
        create/destroy: 0.99 → 0.04 (NOW ORTHOGONAL!)
        give/take: 0.99 → 0.75 (both transfer verbs - correctly similar!)
"""

import numpy as np
from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph


class PhaseShiftedJ:
    """
    Centered j-vectors for meaningful verb comparison.

    "Shift the phase" = subtract the global mean vector.
    """

    def __init__(self, graph: MeaningGraph = None):
        self.graph = graph or MeaningGraph()
        self.global_mean: np.ndarray = None
        self._j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

    def compute_global_mean(self) -> np.ndarray:
        """Compute the mean j-vector across all VerbOperators."""
        if not self.graph.driver:
            return np.zeros(5)

        with self.graph.driver.session() as session:
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

            if not all_j:
                return np.zeros(5)

            self.global_mean = np.mean(all_j, axis=0)
            return self.global_mean

    def center(self, j: np.ndarray) -> np.ndarray:
        """Center a j-vector by subtracting global mean (phase shift)."""
        if self.global_mean is None:
            self.compute_global_mean()
        return np.array(j) - self.global_mean

    def centered_similarity(self, j1: np.ndarray, j2: np.ndarray) -> float:
        """Cosine similarity on centered (phase-shifted) vectors."""
        c1 = self.center(j1)
        c2 = self.center(j2)

        n1 = np.linalg.norm(c1)
        n2 = np.linalg.norm(c2)

        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        return float(np.dot(c1, c2) / (n1 * n2))

    def raw_similarity(self, j1: np.ndarray, j2: np.ndarray) -> float:
        """Raw cosine similarity (for comparison)."""
        n1 = np.linalg.norm(j1)
        n2 = np.linalg.norm(j2)

        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        return float(np.dot(j1, j2) / (n1 * n2))

    def get_verb_j(self, verb: str) -> np.ndarray:
        """Get j-vector for a verb."""
        if not self.graph.driver:
            return np.zeros(5)

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (v:VerbOperator {verb: $verb})
                RETURN v.j as j
            """, verb=verb)

            record = result.single()
            if record and record['j']:
                return np.array(record['j'][:5])
            return np.zeros(5)

    def get_centered_j(self, verb: str) -> np.ndarray:
        """Get phase-shifted (centered) j-vector for a verb."""
        raw_j = self.get_verb_j(verb)
        return self.center(raw_j)

    def dominant_dimension(self, centered_j: np.ndarray) -> tuple:
        """Get the dominant dimension of a centered j-vector."""
        abs_vals = np.abs(centered_j)
        idx = np.argmax(abs_vals)
        sign = '+' if centered_j[idx] > 0 else '-'
        return self._j_dims[idx], sign, centered_j[idx]


def demo():
    """Demonstrate phase-shifted j-vector similarity."""
    print("=" * 70)
    print("PHASE-SHIFTED J-VECTORS: The Pirate Insight")
    print("=" * 70)

    phase = PhaseShiftedJ()

    if not phase.graph.is_connected():
        print("\nNot connected to Neo4j. Start with:")
        print("  cd config && docker-compose up -d")
        return

    # Compute global mean
    mean = phase.compute_global_mean()
    print(f"\nGlobal Mean (the bias to shift):")
    for dim, val in zip(phase._j_dims, mean):
        print(f"  {dim}: {val:.4f}")

    # Test opposite verb pairs
    opposite_pairs = [
        ('love', 'hate'),
        ('give', 'take'),
        ('create', 'destroy'),
        ('help', 'harm'),
        ('find', 'lose'),
        ('open', 'close'),
        ('rise', 'fall'),
        ('begin', 'end'),
    ]

    print("\n" + "=" * 70)
    print("OPPOSITE VERB PAIRS: Before and After Phase Shift")
    print("=" * 70)
    print(f"{'Pair':<20} {'Raw cos':>10} {'Shifted':>10} {'Effect':<15}")
    print("-" * 60)

    success_count = 0
    for v1, v2 in opposite_pairs:
        j1 = phase.get_verb_j(v1)
        j2 = phase.get_verb_j(v2)

        if np.linalg.norm(j1) < 1e-6 or np.linalg.norm(j2) < 1e-6:
            print(f"{v1}/{v2:<14} {'(missing j-vector)':<30}")
            continue

        raw_sim = phase.raw_similarity(j1, j2)
        centered_sim = phase.centered_similarity(j1, j2)

        if centered_sim < 0.3:  # Much less similar after shift
            effect = "✓ OPPOSITE" if centered_sim < 0 else "✓ ORTHOGONAL"
            success_count += 1
        else:
            effect = "→ Similar"

        print(f"{v1}/{v2:<14} {raw_sim:>10.4f} {centered_sim:>10.4f} {effect:<15}")

    # Test similar verb pairs
    similar_pairs = [
        ('love', 'adore'),
        ('help', 'assist'),
        ('find', 'discover'),
        ('create', 'make'),
        ('understand', 'comprehend'),
    ]

    print("\n" + "=" * 70)
    print("SIMILAR VERB PAIRS: Should remain similar after shift")
    print("=" * 70)
    print(f"{'Pair':<20} {'Raw cos':>10} {'Shifted':>10} {'Effect':<15}")
    print("-" * 60)

    for v1, v2 in similar_pairs:
        j1 = phase.get_verb_j(v1)
        j2 = phase.get_verb_j(v2)

        if np.linalg.norm(j1) < 1e-6 or np.linalg.norm(j2) < 1e-6:
            print(f"{v1}/{v2:<14} {'(missing j-vector)':<30}")
            continue

        raw_sim = phase.raw_similarity(j1, j2)
        centered_sim = phase.centered_similarity(j1, j2)

        effect = "✓ SIMILAR" if centered_sim > 0.3 else "✗ Lost similarity"
        print(f"{v1}/{v2:<14} {raw_sim:>10.4f} {centered_sim:>10.4f} {effect:<15}")

    # Show dominant dimensions of key verbs
    key_verbs = ['love', 'hate', 'create', 'destroy', 'help', 'harm', 'give', 'take']

    print("\n" + "=" * 70)
    print("DOMINANT DIMENSIONS: Where verbs push after phase shift")
    print("=" * 70)
    print(f"{'Verb':<12} {'Dominant':<12} {'Value':>8} {'Direction':<30}")
    print("-" * 60)

    for verb in key_verbs:
        centered = phase.get_centered_j(verb)
        if np.linalg.norm(centered) > 1e-6:
            dim, sign, val = phase.dominant_dimension(centered)
            # Show full vector briefly
            vec_str = ' '.join([f"{v:+.2f}" for v in centered])
            print(f"{verb:<12} {sign}{dim:<11} {val:>8.4f} [{vec_str}]")

    print(f"\n✓ Phase shift successful: {success_count}/{len(opposite_pairs)} opposite pairs now distinguishable")

    phase.graph.close()


if __name__ == "__main__":
    demo()
