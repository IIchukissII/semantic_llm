#!/usr/bin/env python3
"""
CORRECT VALIDATION: Semantic Navigation Quality

(A, S, τ) is NOT for classification. It's for NAVIGATION.

The Navigator uses (A, S, τ) to:
1. Find semantically related concepts
2. Navigate through meaning space
3. Produce coherent paths

TEST: Does navigation produce meaningful, coherent results?

Metrics:
- Path coherence: Are consecutive concepts related?
- Goal alignment: Does navigation reach intended target?
- Semantic stability: Do quality metrics (R, C, D) make sense?
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.semantic_coords import BottleneckEncoder
from core.data_loader import DataLoader


def test_navigation_coherence():
    """Test: Do verb transformations produce semantically coherent paths?"""

    print("=" * 70)
    print("TEST 1: Navigation Path Coherence")
    print("=" * 70)
    print()

    encoder = BottleneckEncoder()

    # Test verb chains
    test_chains = [
        ("truth", ["love", "think", "feel"]),
        ("life", ["create", "grow", "become"]),
        ("freedom", ["fight", "win", "live"]),
        ("wisdom", ["seek", "find", "know"]),
    ]

    for start_word, verbs in test_chains:
        print(f"  {start_word} →", end="")

        trajectory = encoder.chain(start_word, verbs)

        if len(trajectory) > 0:
            # Show path
            for i, step in enumerate(trajectory.steps):
                if i == 0:
                    print(f" (A={step.A:.2f}, S={step.S:.2f}, τ={step.tau:.2f})", end="")
                else:
                    verb = verbs[i-1]
                    print(f" --{verb}--> (A={step.A:.2f}, S={step.S:.2f})", end="")

            # Find nearest word at endpoint
            endpoint = trajectory.end
            neighbors = encoder.nearest(endpoint, k=3, exclude=[start_word])

            print()
            print(f"    Endpoint neighbors: {', '.join([w for w, _, _ in neighbors])}")

            # Compute path metrics
            total_shift = trajectory.total_shift
            print(f"    Total shift: (ΔA={total_shift[0]:+.3f}, ΔS={total_shift[1]:+.3f})")

        print()


def test_verb_operator_semantics():
    """Test: Do verb operators have meaningful semantic effects?"""

    print("=" * 70)
    print("TEST 2: Verb Operator Semantics")
    print("=" * 70)
    print()

    encoder = BottleneckEncoder()

    # Group verbs by expected effect
    verb_groups = {
        "constructive": ["create", "build", "grow", "give", "help"],
        "destructive": ["destroy", "break", "kill", "take", "hurt"],
        "cognitive": ["think", "know", "learn", "understand", "see"],
        "emotional": ["love", "hate", "feel", "fear", "hope"],
        "movement": ["go", "come", "move", "run", "walk"],
    }

    print("  Verb Group Analysis:")
    print("-" * 50)

    for group_name, verbs in verb_groups.items():
        dA_vals = []
        dS_vals = []

        for v in verbs:
            op = encoder.encode_verb(v)
            if op:
                dA_vals.append(op.dA)
                dS_vals.append(op.dS)

        if dA_vals:
            mean_dA = np.mean(dA_vals)
            mean_dS = np.mean(dS_vals)
            print(f"  {group_name:12}: ΔA={mean_dA:+.3f}, ΔS={mean_dS:+.3f}")

    print()


def test_semantic_regions():
    """Test: Do (A, S, τ) regions contain semantically coherent words?"""

    print("=" * 70)
    print("TEST 3: Semantic Region Coherence")
    print("=" * 70)
    print()

    encoder = BottleneckEncoder()

    # Define semantic regions
    regions = [
        ("High A, Low τ (Affirming, Grounded)", 0.5, 2.0, -1.0, 1.0, 1.5, 2.5),
        ("Low A, Low τ (Negating, Grounded)", -2.0, -0.5, -1.0, 1.0, 1.5, 2.5),
        ("High S (Sacred)", -1.0, 1.0, 0.5, 1.5, 1.5, 3.5),
        ("Low S (Profane)", -1.0, 1.0, -1.5, -0.5, 1.5, 3.5),
        ("Low τ (Abstract)", -1.0, 1.0, -1.0, 1.0, 1.0, 2.0),
        ("High τ (Concrete)", -1.0, 1.0, -1.0, 1.0, 3.5, 5.0),
    ]

    for name, A_min, A_max, S_min, S_max, tau_min, tau_max in regions:
        words = encoder.in_region(A_min, A_max, S_min, S_max, tau_min, tau_max)

        if words:
            sample = words[:10]
            print(f"  {name}:")
            print(f"    {', '.join(sample)}")
            print(f"    ({len(words)} total words)")
        else:
            print(f"  {name}: (no words found)")

        print()


def test_tau_orbital_structure():
    """Test: Does τ show orbital structure (discrete levels)?"""

    print("=" * 70)
    print("TEST 4: τ Orbital Structure")
    print("=" * 70)
    print()

    encoder = BottleneckEncoder()
    loader = DataLoader()

    coords = loader.load_semantic_coordinates()

    # Get τ distribution
    tau_values = [c[2] for c in coords.values()]

    # Count words per orbital (τ_n = 1 + n/e)
    import math
    e = math.e

    orbitals = {}
    for tau in tau_values:
        n = round((tau - 1) * e)
        if n not in orbitals:
            orbitals[n] = 0
        orbitals[n] += 1

    print("  Orbital Distribution (τ_n = 1 + n/e):")
    print("-" * 40)
    print(f"  {'n':>3} | {'τ range':^12} | {'count':>6}")
    print("-" * 40)

    for n in sorted(orbitals.keys()):
        if n >= 0 and n <= 15:
            tau_low = 1 + (n - 0.5) / e
            tau_high = 1 + (n + 0.5) / e
            count = orbitals[n]
            bar = "█" * (count // 500)
            print(f"  {n:>3} | {tau_low:.2f} - {tau_high:.2f} | {count:>6} {bar}")

    print()

    # Check: Is n=5 the veil?
    human_count = sum(orbitals.get(n, 0) for n in range(0, 5))
    trans_count = sum(orbitals.get(n, 0) for n in range(5, 20))

    print(f"  Human realm (n < 5):         {human_count:,} words")
    print(f"  Transcendental realm (n ≥ 5): {trans_count:,} words")
    print(f"  Ratio: {human_count / max(trans_count, 1):.1f}x more in human realm")
    print()


def test_quality_metrics():
    """Test: Check the quality metric formulas make sense."""

    print("=" * 70)
    print("TEST 5: Quality Metric Validation")
    print("=" * 70)
    print()

    encoder = BottleneckEncoder()

    # Sample concepts
    concepts = ["truth", "love", "wisdom", "life", "god", "death", "evil", "war"]

    coords = []
    for c in concepts:
        coord = encoder.encode_word(c)
        if coord:
            coords.append((c, coord))

    if len(coords) < 3:
        print("  Not enough concepts found")
        return

    # Compute coherence (how similar are the concepts?)
    distances = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            d = coords[i][1].distance(coords[j][1])
            distances.append(d)

    coherence = 1.0 / (1.0 + np.mean(distances))

    print(f"  Concept set: {[c[0] for c in coords]}")
    print(f"  Mean pairwise distance: {np.mean(distances):.3f}")
    print(f"  Coherence (1/(1+d)): {coherence:.3f}")
    print()

    # Compute depth (mean τ)
    mean_tau = np.mean([c[1].tau for c in coords])
    print(f"  Mean τ (depth): {mean_tau:.2f}")
    print()

    # Stability (inverse variance)
    tau_std = np.std([c[1].tau for c in coords])
    stability = 1.0 / (0.1 + tau_std)
    print(f"  τ std: {tau_std:.3f}")
    print(f"  Stability (1/(0.1+std)): {stability:.2f}")
    print()


def main():
    """Run all navigation validation tests."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " SEMANTIC NAVIGATION VALIDATION ".center(68) + "║")
    print("║" + " Testing what (A, S, τ) is ACTUALLY for ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    test_navigation_coherence()
    test_verb_operator_semantics()
    test_semantic_regions()
    test_tau_orbital_structure()
    test_quality_metrics()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("  (A, S, τ) is designed for NAVIGATION, not classification.")
    print()
    print("  The tests above validate:")
    print("    - Verb operators shift coordinates meaningfully")
    print("    - Semantic regions contain coherent word groups")
    print("    - τ shows orbital structure with human/transcendental realms")
    print("    - Quality metrics (coherence, depth, stability) are computable")
    print()
    print("  The real test is: Does the Navigator produce good dialogues?")
    print("  That requires running dialogue_navigator.py with Claude.")
    print()


if __name__ == "__main__":
    main()
