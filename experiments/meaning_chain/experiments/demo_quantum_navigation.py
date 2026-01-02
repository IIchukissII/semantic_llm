#!/usr/bin/env python3
"""
DEMO: Quantum Navigation with 12-bit Encoding

Tests semantic navigation using quantized (n, θ, r) coordinates.

This demo validates:
1. Quantized coordinates preserve navigation capability
2. Nearest-neighbor search works with 12-bit encoding
3. Semantic paths remain coherent after quantization
4. Reconstruction error is acceptable for navigation
"""

import sys
from pathlib import Path
import math
import numpy as np

# Add paths
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.semantic_quantum import QuantumEncoder, QuantumWord
from core.semantic_coords import BottleneckEncoder


def demo_quantum_encoding():
    """Demo 1: Basic quantum encoding and decoding."""

    print("=" * 70)
    print("DEMO 1: Quantum Encoding")
    print("=" * 70)
    print()

    qenc = QuantumEncoder()

    # Test words across semantic space
    test_words = [
        "truth", "love", "god", "death", "wisdom",
        "life", "peace", "war", "freedom", "evil",
        "light", "darkness", "hope", "despair", "beauty"
    ]

    print(f"{'Word':<12} {'Hex':>5} {'n':>3} {'θ°':>6} {'r':>5} │ {'A':>6} {'S':>6} {'τ':>5}")
    print("-" * 70)

    for word in test_words:
        qw = qenc.encode(word)
        if qw:
            print(f"{word:<12} {qw.to_hex():>5} {qw.n:>3} {qw.theta_deg:>+6.0f} {qw.r:>5.2f} │ "
                  f"{qw.A:>+6.2f} {qw.S:>+6.2f} {qw.tau:>5.2f}")

    print()


def demo_quadrant_analysis():
    """Demo 2: Semantic quadrants via θ."""

    print("=" * 70)
    print("DEMO 2: Semantic Quadrants (θ Analysis)")
    print("=" * 70)
    print()

    qenc = QuantumEncoder()

    # Group words by quadrant
    quadrants = {
        "Q1 (Affirming+Sacred)": [],      # θ: 0° to 90°
        "Q2 (Negating+Sacred)": [],       # θ: 90° to 180°
        "Q3 (Negating+Profane)": [],      # θ: -180° to -90°
        "Q4 (Affirming+Profane)": [],     # θ: -90° to 0°
    }

    # Sample words
    sample_words = [
        "god", "holy", "divine", "sacred", "heaven",  # Q1?
        "evil", "demon", "curse", "hell", "sin",      # Q2?
        "death", "pain", "suffer", "wound", "decay",  # Q3?
        "life", "birth", "grow", "flesh", "earth",    # Q4?
        "truth", "love", "wisdom", "beauty", "peace", # Mixed
        "war", "hate", "fear", "anger", "despair",    # Mixed
    ]

    for word in sample_words:
        qw = qenc.encode(word)
        if qw:
            theta = qw.theta_deg
            if 0 <= theta < 90:
                quadrants["Q1 (Affirming+Sacred)"].append((word, theta, qw.r))
            elif 90 <= theta <= 180:
                quadrants["Q2 (Negating+Sacred)"].append((word, theta, qw.r))
            elif -180 <= theta < -90:
                quadrants["Q3 (Negating+Profane)"].append((word, theta, qw.r))
            else:  # -90 <= theta < 0
                quadrants["Q4 (Affirming+Profane)"].append((word, theta, qw.r))

    for quadrant, words in quadrants.items():
        print(f"{quadrant}:")
        if words:
            for word, theta, r in sorted(words, key=lambda x: -x[2])[:5]:
                print(f"    {word:<12} θ={theta:>+6.0f}° r={r:.2f}")
        else:
            print("    (none)")
        print()


def demo_orbital_levels():
    """Demo 3: Navigation across orbital levels (n)."""

    print("=" * 70)
    print("DEMO 3: Orbital Levels (n = Abstraction)")
    print("=" * 70)
    print()

    qenc = QuantumEncoder()

    # Group by orbital
    orbitals = {}

    # Get all words
    qenc._load()
    for word, bits in list(qenc._quantized.items())[:5000]:
        qw = QuantumWord.from_bits(bits, word)
        if qw.n not in orbitals:
            orbitals[qw.n] = []
        orbitals[qw.n].append((word, qw.r))

    print("Orbital distribution:")
    print()

    for n in sorted(orbitals.keys()):
        if n > 10:
            continue
        words = orbitals[n]
        # Sort by intensity (r)
        top_words = sorted(words, key=lambda x: -x[1])[:5]
        word_list = ", ".join([w[0] for w in top_words])
        tau = 1 + n / math.e
        realm = "human" if n < 5 else "transcendental"
        print(f"  n={n} (τ≈{tau:.2f}, {realm}): {word_list} [{len(words)} words]")

    print()


def demo_quantum_navigation():
    """Demo 4: Navigation using quantized coordinates."""

    print("=" * 70)
    print("DEMO 4: Quantum Navigation")
    print("=" * 70)
    print()

    qenc = QuantumEncoder()
    benc = BottleneckEncoder()

    # Navigate from concept to concept
    start_words = ["truth", "love", "freedom", "wisdom"]

    for start_word in start_words:
        qw_start = qenc.encode(start_word)
        if not qw_start:
            continue

        print(f"Navigation from '{start_word}' ({qw_start.to_hex()}):")
        print(f"  Starting: n={qw_start.n}, θ={qw_start.theta_deg:+.0f}°, r={qw_start.r:.2f}")
        print()

        # Find nearest in quantum space
        neighbors = qenc.nearest(qw_start, k=6)

        print("  Nearest neighbors (quantum distance):")
        for word, dist, qw in neighbors:
            if word == start_word:
                continue
            # Show transformation needed
            dn = qw.n - qw_start.n
            dtheta = qw.theta_deg - qw_start.theta_deg
            dr = qw.r - qw_start.r
            print(f"    {word:<15} d={dist:.3f}  Δn={dn:+d} Δθ={dtheta:+5.0f}° Δr={dr:+.2f}")

        print()


def demo_path_coherence():
    """Demo 5: Test path coherence with quantized vs full coordinates."""

    print("=" * 70)
    print("DEMO 5: Quantization Error Analysis")
    print("=" * 70)
    print()

    qenc = QuantumEncoder()
    benc = BottleneckEncoder()

    # Compare quantized vs full precision
    test_words = ["truth", "love", "god", "death", "wisdom", "freedom", "peace", "war"]

    print("Reconstruction error (quantized → A, S, τ):")
    print()
    print(f"{'Word':<12} {'|ΔA|':>6} {'|ΔS|':>6} {'|Δτ|':>6} │ {'Total':>6}")
    print("-" * 50)

    errors = []
    for word in test_words:
        # Get full precision
        full = benc.encode_word(word)
        if not full:
            continue

        # Get quantized and reconstruct
        qw = qenc.encode(word)
        if not qw:
            continue

        # Compute errors
        dA = abs(qw.A - full.A)
        dS = abs(qw.S - full.S)
        dtau = abs(qw.tau - full.tau)
        total = math.sqrt(dA**2 + dS**2 + dtau**2)

        errors.append((dA, dS, dtau, total))
        print(f"{word:<12} {dA:>6.3f} {dS:>6.3f} {dtau:>6.3f} │ {total:>6.3f}")

    print("-" * 50)

    if errors:
        mean_errors = np.mean(errors, axis=0)
        print(f"{'Mean':<12} {mean_errors[0]:>6.3f} {mean_errors[1]:>6.3f} "
              f"{mean_errors[2]:>6.3f} │ {mean_errors[3]:>6.3f}")

    print()

    # Navigation error comparison
    print("Navigation comparison (nearest neighbor consistency):")
    print()

    consistent = 0
    total = 0

    for word in test_words:
        full = benc.encode_word(word)
        qw = qenc.encode(word)
        if not full or not qw:
            continue

        # Get neighbors from both
        full_neighbors = set([w for w, _, _ in benc.nearest(full, k=5, exclude=[word])])
        quant_neighbors = set([w for w, _, _ in qenc.nearest(qw, k=5) if w != word][:4])

        overlap = len(full_neighbors & quant_neighbors)
        consistent += overlap
        total += len(quant_neighbors)

        print(f"  {word}: {overlap}/{len(quant_neighbors)} neighbors match")

    if total > 0:
        print()
        print(f"  Overall consistency: {consistent}/{total} ({100*consistent/total:.0f}%)")

    print()


def demo_semantic_compass():
    """Demo 6: Semantic compass - navigation by θ direction."""

    print("=" * 70)
    print("DEMO 6: Semantic Compass")
    print("=" * 70)
    print()

    qenc = QuantumEncoder()

    # Cardinal directions
    directions = [
        (0, "EAST", "Affirmation (A+)"),
        (90, "NORTH", "Sacred (S+)"),
        (180, "WEST", "Negation (A-)"),
        (-90, "SOUTH", "Profane (S-)"),
    ]

    print("Semantic directions and exemplar words:")
    print()

    qenc._load()

    for target_theta, name, meaning in directions:
        print(f"  {name} ({target_theta:+4d}°) - {meaning}:")

        # Find words closest to this direction with high r
        candidates = []
        for word, bits in qenc._quantized.items():
            qw = QuantumWord.from_bits(bits, word)

            # Angular distance
            dtheta = abs(qw.theta_deg - target_theta)
            if dtheta > 180:
                dtheta = 360 - dtheta

            # Only consider words with strong projection (r > 0.5)
            if qw.r > 0.5 and dtheta < 30:
                candidates.append((word, dtheta, qw.r))

        # Sort by r (intensity), take top 5
        candidates.sort(key=lambda x: -x[2])
        for word, dtheta, r in candidates[:5]:
            print(f"      {word:<15} θ={dtheta:+.0f}° off, r={r:.2f}")

        print()


def main():
    """Run all navigation demos."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " QUANTUM NAVIGATION DEMO ".center(68) + "║")
    print("║" + " Testing 12-bit (n, θ, r) encoding in navigation ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    demo_quantum_encoding()
    demo_quadrant_analysis()
    demo_orbital_levels()
    demo_quantum_navigation()
    demo_path_coherence()
    demo_semantic_compass()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("  The 12-bit quantized encoding supports semantic navigation:")
    print()
    print("    n (orbital)  → Navigate abstraction levels (human ↔ transcendental)")
    print("    θ (phase)    → Navigate semantic directions (affirmation ↔ negation)")
    print("    r (intensity)→ Filter by transcendental strength")
    print()
    print("  Navigation operators:")
    print("    Δn: Shift abstraction level")
    print("    Δθ: Rotate semantic direction")
    print("    Δr: Amplify/dampen intensity")
    print()
    print("  Use case: Neo4j graph navigation with 3-byte node properties")
    print()


if __name__ == "__main__":
    main()
