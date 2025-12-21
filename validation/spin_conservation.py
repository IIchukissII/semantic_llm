#!/usr/bin/env python3
"""
Spin Prefix Analysis
====================

Hypothesis: Prefixes (un-, dis-, im-) act like quantum spin operators.

Physics analogy:
  Spin = intrinsic property
  Doesn't change position (τ), only direction
  Discrete: +½, -½

Predictions:
  1. τ conserved: τ(happy) ≈ τ(unhappy)
  2. Direction flip: cos(happy, unhappy) ≈ -1
  3. Discrete values: spin clusters into 2-3 values
  4. Algebra: un- + un- forbidden
  5. Complex conjugation: z_unhappy = z̄_happy
"""

import numpy as np
import psycopg2
from collections import defaultdict
import json
from pathlib import Path

DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}

OUTPUT_DIR = Path(__file__).parent


def load_semantic_index():
    """Load semantic vectors from database."""
    print("Loading semantic index...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT word, j, i, tau, tau_entropy, h_norm
        FROM hyp_semantic_index
        WHERE j IS NOT NULL
    """)

    index = {}
    for row in cur.fetchall():
        word, j, i, tau, tau_entropy, h_norm = row
        if j is not None:
            index[word] = {
                'j': np.array(j),
                'i': np.array(i) if i else None,
                'tau': tau_entropy if tau_entropy else tau,
                'h_norm': h_norm if h_norm else 0
            }

    conn.close()
    print(f"  Loaded {len(index)} words")
    return index


def cosine_similarity(v1, v2):
    """Cosine similarity."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def test_tau_conservation(index):
    """Test 1: Does τ stay the same under prefix transformation?"""
    print("\n" + "=" * 70)
    print("TEST 1: τ CONSERVATION")
    print("=" * 70)
    print("Hypothesis: Prefix doesn't change energy level (τ)")

    # Pairs to test
    pairs = [
        # un- prefix
        ('happy', 'unhappy'),
        ('fair', 'unfair'),
        ('certain', 'uncertain'),
        ('known', 'unknown'),
        ('clear', 'unclear'),
        ('safe', 'unsafe'),
        ('like', 'unlike'),
        ('able', 'unable'),
        # dis- prefix
        ('order', 'disorder'),
        ('connect', 'disconnect'),
        ('appear', 'disappear'),
        ('agree', 'disagree'),
        ('trust', 'distrust'),
        ('comfort', 'discomfort'),
        # im-/in- prefix
        ('possible', 'impossible'),
        ('perfect', 'imperfect'),
        ('complete', 'incomplete'),
        ('visible', 'invisible'),
        ('direct', 'indirect'),
        ('correct', 'incorrect'),
        # de- prefix
        ('compose', 'decompose'),
        ('code', 'decode'),
        # re- prefix (not negation - reconstruction)
        ('build', 'rebuild'),
        ('make', 'remake'),
        ('do', 'redo'),
    ]

    results = []
    print(f"\n  {'Base':<15} {'Prefixed':<15} {'τ_base':>8} {'τ_pref':>8} {'Δτ':>8}")
    print("  " + "-" * 60)

    for base, prefixed in pairs:
        if base in index and prefixed in index:
            tau_base = index[base]['tau']
            tau_pref = index[prefixed]['tau']
            delta_tau = tau_pref - tau_base

            results.append({
                'base': base,
                'prefixed': prefixed,
                'tau_base': tau_base,
                'tau_pref': tau_pref,
                'delta_tau': delta_tau
            })

            conserved = "✓" if abs(delta_tau) < 0.5 else "✗"
            print(f"  {base:<15} {prefixed:<15} {tau_base:>8.2f} {tau_pref:>8.2f} {delta_tau:>+8.2f} {conserved}")

    if results:
        delta_taus = [r['delta_tau'] for r in results]
        print(f"\n  Mean |Δτ| = {np.mean(np.abs(delta_taus)):.3f}")
        print(f"  Std Δτ = {np.std(delta_taus):.3f}")
        conserved_pct = 100 * sum(1 for d in delta_taus if abs(d) < 0.5) / len(delta_taus)
        print(f"  τ conserved (|Δτ| < 0.5): {conserved_pct:.1f}%")

    return results


def test_direction_flip(index):
    """Test 2: Does prefix flip the direction in semantic space?"""
    print("\n" + "=" * 70)
    print("TEST 2: DIRECTION FLIP (Spin Inversion)")
    print("=" * 70)
    print("Hypothesis: cos(base, prefixed) ≈ -1 for negating prefixes")

    pairs = [
        ('happy', 'unhappy', 'un-'),
        ('fair', 'unfair', 'un-'),
        ('order', 'disorder', 'dis-'),
        ('connect', 'disconnect', 'dis-'),
        ('possible', 'impossible', 'im-'),
        ('complete', 'incomplete', 'in-'),
        ('visible', 'invisible', 'in-'),
        ('appear', 'disappear', 'dis-'),
        ('trust', 'distrust', 'dis-'),
        ('known', 'unknown', 'un-'),
    ]

    results = []
    print(f"\n  {'Base':<12} {'Prefixed':<12} {'j-cos':>8} {'16D-cos':>8} {'Flip?'}")
    print("  " + "-" * 55)

    for base, prefixed, prefix in pairs:
        if base in index and prefixed in index:
            j_base = index[base]['j']
            j_pref = index[prefixed]['j']

            j_cos = cosine_similarity(j_base, j_pref)

            # Full 16D
            if index[base]['i'] is not None and index[prefixed]['i'] is not None:
                full_base = np.concatenate([j_base, index[base]['i']])
                full_pref = np.concatenate([j_pref, index[prefixed]['i']])
                full_cos = cosine_similarity(full_base, full_pref)
            else:
                full_cos = j_cos

            results.append({
                'base': base,
                'prefixed': prefixed,
                'prefix': prefix,
                'j_cos': j_cos,
                'full_cos': full_cos
            })

            flip = "✓ FLIP" if j_cos < -0.3 else ("~ partial" if j_cos < 0.3 else "✗")
            print(f"  {base:<12} {prefixed:<12} {j_cos:>+8.3f} {full_cos:>+8.3f} {flip}")

    if results:
        j_coss = [r['j_cos'] for r in results]
        print(f"\n  Mean j-cosine = {np.mean(j_coss):.3f}")
        print(f"  Flips (cos < -0.3): {sum(1 for c in j_coss if c < -0.3)}/{len(j_coss)}")

    return results


def test_spin_discreteness(index):
    """Test 3: Is spin discrete like in physics (+½, -½)?"""
    print("\n" + "=" * 70)
    print("TEST 3: SPIN DISCRETENESS")
    print("=" * 70)
    print("Hypothesis: Spin values cluster into discrete levels")

    # Find all prefix pairs in vocabulary
    prefixes = {
        'un': [],
        'dis': [],
        'im': [],
        'in': [],
        'de': [],
        're': [],
    }

    for word in index:
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                base = word[len(prefix):]
                if base in index:
                    prefixes[prefix].append((base, word))

    print(f"\n  Prefix pairs found:")
    for prefix, pairs in prefixes.items():
        print(f"    {prefix}-: {len(pairs)} pairs")

    # Compute spin values for each prefix
    print(f"\n  Spin values by prefix:")

    all_spins = {}
    for prefix, pairs in prefixes.items():
        if len(pairs) < 3:
            continue

        spins = []
        for base, prefixed in pairs[:20]:  # Sample
            j_base = index[base]['j']
            j_pref = index[prefixed]['j']

            # "Spin" = projection onto flip direction
            spin = cosine_similarity(j_base, j_pref)
            spins.append(spin)

        all_spins[prefix] = spins

        if spins:
            print(f"\n    {prefix}-: n={len(spins)}")
            print(f"      mean = {np.mean(spins):+.3f}")
            print(f"      std  = {np.std(spins):.3f}")

            # Check discreteness: do values cluster?
            hist, bins = np.histogram(spins, bins=5, range=(-1, 1))
            print(f"      distribution: {list(hist)}")

    return all_spins


def test_complex_conjugation(index):
    """Test 5: Is prefix = complex conjugation (z → z̄)?"""
    print("\n" + "=" * 70)
    print("TEST 5: COMPLEX CONJUGATION")
    print("=" * 70)
    print("Hypothesis: z_prefixed = z̄_base (conjugation)")
    print("           τ same, sentiment flipped")

    pairs = [
        ('happy', 'unhappy'),
        ('fair', 'unfair'),
        ('order', 'disorder'),
        ('trust', 'distrust'),
        ('possible', 'impossible'),
        ('complete', 'incomplete'),
    ]

    print(f"\n  Formula: z = τ + i·s where s = sentiment (j·j_good)")
    print(f"\n  {'Base':<12} {'Prefixed':<12} {'τ_base':>7} {'τ_pref':>7} {'s_base':>7} {'s_pref':>7} {'Conj?'}")
    print("  " + "-" * 65)

    # j_good direction (normalized)
    j_good = np.array([1, 1, 1, 1, 1]) / np.sqrt(5)

    results = []
    for base, prefixed in pairs:
        if base in index and prefixed in index:
            tau_base = index[base]['tau']
            tau_pref = index[prefixed]['tau']

            # Sentiment = projection onto good
            s_base = np.dot(index[base]['j'], j_good)
            s_pref = np.dot(index[prefixed]['j'], j_good)

            # Check conjugation: τ same, s flipped
            tau_same = abs(tau_pref - tau_base) < 0.5
            s_flipped = (s_base * s_pref) < 0  # opposite signs

            is_conjugate = tau_same and s_flipped

            results.append({
                'base': base,
                'prefixed': prefixed,
                'tau_base': tau_base,
                'tau_pref': tau_pref,
                's_base': s_base,
                's_pref': s_pref,
                'is_conjugate': is_conjugate
            })

            conj = "✓ z̄" if is_conjugate else ("τ≠" if not tau_same else "s≠")
            print(f"  {base:<12} {prefixed:<12} {tau_base:>7.2f} {tau_pref:>7.2f} "
                  f"{s_base:>+7.2f} {s_pref:>+7.2f} {conj}")

    if results:
        conj_pct = 100 * sum(1 for r in results if r['is_conjugate']) / len(results)
        print(f"\n  Conjugation confirmed: {conj_pct:.1f}%")

    return results


def test_prefix_algebra(index):
    """Test 4: Is there prefix algebra (un- + un- forbidden)?"""
    print("\n" + "=" * 70)
    print("TEST 4: PREFIX ALGEBRA")
    print("=" * 70)
    print("Hypothesis: Double negation rare/forbidden (like spin rules)")

    # Check for double prefixes
    double_prefixes = [
        ('un', 'un'),   # ununhappy?
        ('un', 'dis'),  # undisturbed?
        ('dis', 'un'),  # disunited?
        ('re', 're'),   # rereread?
        ('de', 're'),   # derestrict?
        ('un', 're'),   # unreread?
    ]

    print(f"\n  Double prefix patterns:")

    for p1, p2 in double_prefixes:
        pattern = p1 + p2
        found = [w for w in index if w.startswith(pattern) and len(w) > len(pattern) + 2]
        print(f"    {p1}-{p2}-: {len(found)} words")
        if found[:3]:
            print(f"      examples: {found[:3]}")

    # Single vs double
    single_counts = {}
    for prefix in ['un', 'dis', 'im', 'in', 're', 'de']:
        single_counts[prefix] = sum(1 for w in index if w.startswith(prefix))

    print(f"\n  Single prefix counts:")
    for p, c in sorted(single_counts.items(), key=lambda x: -x[1]):
        print(f"    {p}-: {c}")


def main():
    print("=" * 70)
    print("SPIN PREFIX ANALYSIS")
    print("=" * 70)
    print("""
Physics analogy:
  Spin = intrinsic property, discrete (+½, -½)
  Changes state direction, not position

Hypothesis:
  Prefix (un-, dis-, im-) = spin operator
  τ conserved (energy level same)
  Direction flipped (spin inversion)
  z_prefixed = z̄_base (complex conjugation)
""")

    # Load data
    index = load_semantic_index()

    # Run tests
    tau_results = test_tau_conservation(index)
    flip_results = test_direction_flip(index)
    spin_results = test_spin_discreteness(index)
    conj_results = test_complex_conjugation(index)
    test_prefix_algebra(index)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if tau_results:
        delta_taus = [abs(r['delta_tau']) for r in tau_results]
        print(f"\n  τ Conservation: mean |Δτ| = {np.mean(delta_taus):.3f}")

    if flip_results:
        j_coss = [r['j_cos'] for r in flip_results]
        print(f"  Direction Flip: mean cos = {np.mean(j_coss):.3f}")

    if conj_results:
        conj_pct = 100 * sum(1 for r in conj_results if r['is_conjugate']) / len(conj_results)
        print(f"  Complex Conjugation: {conj_pct:.1f}% confirmed")

    # Export
    output = {
        "tau_conservation": tau_results,
        "direction_flip": flip_results,
        "complex_conjugation": conj_results
    }

    with open(OUTPUT_DIR / "spin_prefix_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\n  Exported to spin_prefix_results.json")


if __name__ == "__main__":
    main()
