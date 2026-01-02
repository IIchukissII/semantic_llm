#!/usr/bin/env python3
"""
DEMO: Quantum Regional Navigation

The 12-bit encoding is for REGIONAL FILTERING, not similarity search.

Use cases:
1. Filter by orbital level (n): abstract vs concrete concepts
2. Filter by direction (θ): affirmation vs negation axis
3. Filter by intensity (r): strong vs weak transcendental character
4. Navigate between regions by shifting (Δn, Δθ, Δr)

This is how Neo4j would use it: fast range queries on compact properties.
"""

import sys
from pathlib import Path
import json
import math
from collections import defaultdict

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.semantic_quantum import QuantumEncoder, QuantumWord


def load_all_words():
    """Load all quantized words."""
    encoder = QuantumEncoder()
    encoder._load()

    words = []
    for word, bits in encoder._quantized.items():
        qw = QuantumWord.from_bits(bits, word)
        words.append(qw)

    return words


def demo_orbital_filter():
    """Demo: Filter words by abstraction level."""

    print("=" * 70)
    print("DEMO 1: Orbital Filter (n = abstraction level)")
    print("=" * 70)
    print()

    words = load_all_words()

    # Count by orbital
    by_orbital = defaultdict(list)
    for qw in words:
        by_orbital[qw.n].append(qw)

    print("Query: Find most ABSTRACT concepts (n ≤ 2, human realm)")
    print("-" * 50)

    abstract_words = []
    for n in range(0, 3):
        for qw in by_orbital[n]:
            # Filter for high intensity (transcendentally significant)
            if qw.r >= 1.0:
                abstract_words.append(qw)

    # Sort by intensity
    abstract_words.sort(key=lambda x: -x.r)

    print(f"Found {len(abstract_words)} abstract concepts with r≥1.0")
    print()
    print("Top 20 by intensity:")
    for qw in abstract_words[:20]:
        print(f"  {qw.word:<15} n={qw.n} θ={qw.theta_deg:+4.0f}° r={qw.r:.2f}")

    print()

    print("Query: Find most CONCRETE concepts (n ≥ 6, transcendental realm)")
    print("-" * 50)

    concrete_words = []
    for n in range(6, 16):
        for qw in by_orbital.get(n, []):
            if qw.r >= 0.5:
                concrete_words.append(qw)

    concrete_words.sort(key=lambda x: -x.r)

    print(f"Found {len(concrete_words)} concrete concepts with r≥0.5")
    print()
    print("Top 20 by intensity:")
    for qw in concrete_words[:20]:
        print(f"  {qw.word:<15} n={qw.n} θ={qw.theta_deg:+4.0f}° r={qw.r:.2f}")

    print()


def demo_direction_filter():
    """Demo: Filter words by semantic direction."""

    print("=" * 70)
    print("DEMO 2: Direction Filter (θ = semantic axis)")
    print("=" * 70)
    print()

    words = load_all_words()

    # Define direction ranges
    directions = {
        "AFFIRMING (θ≈0°)": (-30, 30),
        "SACRED (θ≈90°)": (60, 120),
        "NEGATING (θ≈180°)": (150, 180),  # Also -180 to -150
        "PROFANE (θ≈-90°)": (-120, -60),
    }

    for name, (theta_min, theta_max) in directions.items():
        print(f"Query: {name}")
        print("-" * 50)

        filtered = []
        for qw in words:
            theta = qw.theta_deg

            # Handle wrap-around for negating
            if name == "NEGATING (θ≈180°)":
                if (theta >= 150 or theta <= -150) and qw.r >= 0.8:
                    filtered.append(qw)
            else:
                if theta_min <= theta <= theta_max and qw.r >= 0.8:
                    filtered.append(qw)

        filtered.sort(key=lambda x: -x.r)

        print(f"Found {len(filtered)} words with r≥0.8")
        print("Top 10:")
        for qw in filtered[:10]:
            print(f"  {qw.word:<15} θ={qw.theta_deg:+4.0f}° r={qw.r:.2f} n={qw.n}")
        print()


def demo_intensity_filter():
    """Demo: Filter by transcendental intensity."""

    print("=" * 70)
    print("DEMO 3: Intensity Filter (r = emanation strength)")
    print("=" * 70)
    print()

    words = load_all_words()

    # Intensity bands
    bands = [
        ("WEAK (r<0.5)", 0, 0.5),
        ("MODERATE (0.5≤r<1)", 0.5, 1.0),
        ("STRONG (1≤r<2)", 1.0, 2.0),
        ("VERY STRONG (r≥2)", 2.0, 5.0),
    ]

    for name, r_min, r_max in bands:
        filtered = [qw for qw in words if r_min <= qw.r < r_max]

        # Sample by n level
        sample_by_n = defaultdict(list)
        for qw in filtered:
            sample_by_n[qw.n].append(qw)

        print(f"{name}: {len(filtered)} words")

        for n in sorted(sample_by_n.keys())[:5]:
            sample = sorted(sample_by_n[n], key=lambda x: -x.r)[:3]
            words_str = ", ".join([qw.word for qw in sample])
            print(f"  n={n}: {words_str}")

        print()


def demo_region_query():
    """Demo: Combined region query (like Neo4j Cypher)."""

    print("=" * 70)
    print("DEMO 4: Combined Region Query")
    print("=" * 70)
    print()

    words = load_all_words()

    queries = [
        {
            "name": "Sacred abstractions",
            "desc": "Abstract (n≤3), sacred (θ>45°), strong (r≥0.8)",
            "n": (0, 3), "theta": (45, 135), "r": (0.8, 5)
        },
        {
            "name": "Profane concrete",
            "desc": "Concrete (n≥4), profane (θ<-30°), moderate (r≥0.5)",
            "n": (4, 15), "theta": (-180, -30), "r": (0.5, 5)
        },
        {
            "name": "Affirming human realm",
            "desc": "Human (n≤4), affirming (|θ|<45°), any intensity",
            "n": (0, 4), "theta": (-45, 45), "r": (0, 5)
        },
        {
            "name": "Extreme transcendentals",
            "desc": "Any level, very strong (r≥2)",
            "n": (0, 15), "theta": (-180, 180), "r": (2, 5)
        },
    ]

    for q in queries:
        print(f"Query: {q['name']}")
        print(f"  {q['desc']}")
        print("-" * 50)

        n_min, n_max = q['n']
        theta_min, theta_max = q['theta']
        r_min, r_max = q['r']

        filtered = []
        for qw in words:
            if not (n_min <= qw.n <= n_max):
                continue
            if not (theta_min <= qw.theta_deg <= theta_max):
                continue
            if not (r_min <= qw.r <= r_max):
                continue
            filtered.append(qw)

        filtered.sort(key=lambda x: -x.r)

        print(f"Found: {len(filtered)} words")
        print("Sample (by r):")
        for qw in filtered[:8]:
            print(f"  {qw.word:<15} n={qw.n} θ={qw.theta_deg:+4.0f}° r={qw.r:.2f}")

        print()


def demo_navigation_shift():
    """Demo: Navigate by shifting coordinates."""

    print("=" * 70)
    print("DEMO 5: Navigation by Shift (Δn, Δθ)")
    print("=" * 70)
    print()

    encoder = QuantumEncoder()
    words = load_all_words()

    # Build index for fast lookup
    by_coords = defaultdict(list)
    for qw in words:
        by_coords[(qw.n, qw.theta_idx, qw.r_idx)].append(qw.word)

    start_word = "truth"
    qw = encoder.encode(start_word)

    print(f"Starting point: '{start_word}'")
    print(f"  Coordinates: n={qw.n}, θ_idx={qw.theta_idx}, r_idx={qw.r_idx}")
    print(f"  Meaning: τ≈{qw.tau:.2f}, θ≈{qw.theta_deg:.0f}°, r≈{qw.r:.2f}")
    print()

    # Navigation operations
    shifts = [
        ("More abstract", -1, 0, 0),
        ("More concrete", +1, 0, 0),
        ("More sacred", 0, +2, 0),
        ("More profane", 0, -2, 0),
        ("Higher intensity", 0, 0, +2),
        ("Lower intensity", 0, 0, -2),
        ("Opposite direction", 0, +8, 0),
    ]

    print("Navigation options:")
    print("-" * 60)

    for name, dn, dtheta, dr in shifts:
        new_n = max(0, min(15, qw.n + dn))
        new_theta = (qw.theta_idx + dtheta) % 16
        new_r = max(0, min(15, qw.r_idx + dr))

        neighbors = by_coords.get((new_n, new_theta, new_r), [])

        if neighbors:
            sample = neighbors[:5]
            print(f"  {name}: → ({new_n}, {new_theta}, {new_r})")
            print(f"    Words: {', '.join(sample)}")
        else:
            # Find closest non-empty bin
            closest = None
            min_dist = 100
            for (n, t, r), ws in by_coords.items():
                d = abs(n - new_n) + abs((t - new_theta) % 16) + abs(r - new_r)
                if d < min_dist:
                    min_dist = d
                    closest = ws

            if closest:
                print(f"  {name}: → ({new_n}, {new_theta}, {new_r}) [nearest bin]")
                print(f"    Words: {', '.join(closest[:5])}")

    print()


def demo_cypher_simulation():
    """Demo: Simulate Neo4j Cypher queries."""

    print("=" * 70)
    print("DEMO 6: Neo4j Cypher Query Simulation")
    print("=" * 70)
    print()

    words = load_all_words()

    print("Neo4j Schema:")
    print("  (:Word {text, n, theta_idx, r_idx, hex})")
    print()

    cypher_queries = [
        (
            "Find abstract sacred concepts",
            "MATCH (w:Word) WHERE w.n <= 2 AND w.theta_idx >= 10 RETURN w.text LIMIT 10",
            lambda qw: qw.n <= 2 and qw.theta_idx >= 10
        ),
        (
            "Find high-intensity words",
            "MATCH (w:Word) WHERE w.r_idx >= 12 RETURN w.text LIMIT 10",
            lambda qw: qw.r_idx >= 12
        ),
        (
            "Navigate from truth's region",
            "MATCH (w:Word) WHERE w.n = 4 AND w.theta_idx BETWEEN 5 AND 9 RETURN w.text LIMIT 10",
            lambda qw: qw.n == 4 and 5 <= qw.theta_idx <= 9
        ),
        (
            "Find opposite pole from affirming",
            "MATCH (w:Word) WHERE w.theta_idx >= 14 OR w.theta_idx <= 2 RETURN w.text LIMIT 10",
            lambda qw: qw.theta_idx >= 14 or qw.theta_idx <= 2
        ),
    ]

    for name, cypher, predicate in cypher_queries:
        print(f"Query: {name}")
        print(f"Cypher: {cypher}")
        print()

        results = [qw for qw in words if predicate(qw)]
        results.sort(key=lambda x: -x.r)

        print(f"Results ({len(results)} matches):")
        for qw in results[:8]:
            print(f"  {qw.word:<15} ({qw.to_hex()})")
        print()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " QUANTUM REGIONAL NAVIGATION ".center(68) + "║")
    print("║" + " 12-bit encoding for Neo4j range queries ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    demo_orbital_filter()
    demo_direction_filter()
    demo_intensity_filter()
    demo_region_query()
    demo_navigation_shift()
    demo_cypher_simulation()

    print("=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print()
    print("  The 12-bit (n, θ, r) encoding is NOT for nearest-neighbor search.")
    print()
    print("  It's for REGIONAL QUERIES in a semantic graph database:")
    print("    • Filter by abstraction level (n)")
    print("    • Filter by semantic direction (θ)")
    print("    • Filter by transcendental intensity (r)")
    print("    • Navigate by shifting coordinates (Δn, Δθ, Δr)")
    print()
    print("  Neo4j benefits:")
    print("    • 3 bytes per node (compact)")
    print("    • Fast range queries (indexed)")
    print("    • Interpretable dimensions")
    print()


if __name__ == "__main__":
    main()
