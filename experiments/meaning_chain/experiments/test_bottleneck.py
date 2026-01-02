#!/usr/bin/env python3
"""
Integration Test: Semantic Bottleneck

Tests the (A, S, τ) encoding system:
1. Word encoding: word → (A, S, τ)
2. Verb encoding: verb → (ΔA, ΔS)
3. Transformations: verb(word) → (A', S', τ)
4. Nearest-neighbor search
5. Analogy solving
6. Performance benchmarks
"""

import time
import sys
from pathlib import Path

# Add paths
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM = _THIS_FILE.parent.parent.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.semantic_coords import BottleneckEncoder, SemanticCoord, VerbOperator
from core.data_loader import DataLoader


def test_word_encoding():
    """Test word → (A, S, τ) encoding."""
    print("=" * 60)
    print("TEST 1: Word Encoding")
    print("=" * 60)

    encoder = BottleneckEncoder()

    # Test words
    test_words = [
        ("truth", 0.865, -0.175, 2.40),
        ("love", 0.071, -0.198, 2.48),
        ("death", -1.464, 0.200, 2.40),
        ("god", 1.307, 0.231, 2.29),
    ]

    all_passed = True
    for word, exp_A, exp_S, exp_tau in test_words:
        coord = encoder.encode_word(word)
        if coord is None:
            print(f"  ✗ {word}: NOT FOUND")
            all_passed = False
            continue

        # Check values (tolerance 0.01)
        ok_A = abs(coord.A - exp_A) < 0.01
        ok_S = abs(coord.S - exp_S) < 0.01
        ok_tau = abs(coord.tau - exp_tau) < 0.01

        if ok_A and ok_S and ok_tau:
            print(f"  ✓ {word}: (A={coord.A:.3f}, S={coord.S:.3f}, τ={coord.tau:.2f})")
        else:
            print(f"  ✗ {word}: (A={coord.A:.3f}, S={coord.S:.3f}, τ={coord.tau:.2f})")
            print(f"       Expected: (A={exp_A:.3f}, S={exp_S:.3f}, τ={exp_tau:.2f})")
            all_passed = False

    print()
    return all_passed


def test_verb_encoding():
    """Test verb → (ΔA, ΔS) encoding."""
    print("=" * 60)
    print("TEST 2: Verb Encoding")
    print("=" * 60)

    encoder = BottleneckEncoder()

    # Test verbs
    test_verbs = [
        ("love", -0.1716, 0.1645),
        ("live", 0.2071, -0.0360),
        ("help", -0.2879, 0.1285),
    ]

    all_passed = True
    for verb, exp_dA, exp_dS in test_verbs:
        op = encoder.encode_verb(verb)
        if op is None:
            print(f"  ✗ {verb}: NOT FOUND")
            all_passed = False
            continue

        ok_dA = abs(op.dA - exp_dA) < 0.001
        ok_dS = abs(op.dS - exp_dS) < 0.001

        if ok_dA and ok_dS:
            print(f"  ✓ {verb}: (ΔA={op.dA:+.4f}, ΔS={op.dS:+.4f})")
        else:
            print(f"  ✗ {verb}: (ΔA={op.dA:+.4f}, ΔS={op.dS:+.4f})")
            print(f"       Expected: (ΔA={exp_dA:+.4f}, ΔS={exp_dS:+.4f})")
            all_passed = False

    print()
    return all_passed


def test_transformations():
    """Test verb(word) transformations."""
    print("=" * 60)
    print("TEST 3: Verb Transformations")
    print("=" * 60)

    encoder = BottleneckEncoder()

    truth = encoder.encode_word("truth")
    love_op = encoder.encode_verb("love")

    # Apply transformation
    result = truth + love_op

    print(f"  truth = (A={truth.A:.3f}, S={truth.S:.3f}, τ={truth.tau:.2f})")
    print(f"  love  = (ΔA={love_op.dA:+.4f}, ΔS={love_op.dS:+.4f})")
    print(f"  love(truth) = (A={result.A:.3f}, S={result.S:.3f}, τ={result.tau:.2f})")

    # Verify: A' = A + ΔA, S' = S + ΔS, τ unchanged
    ok_A = abs(result.A - (truth.A + love_op.dA)) < 0.0001
    ok_S = abs(result.S - (truth.S + love_op.dS)) < 0.0001
    ok_tau = abs(result.tau - truth.tau) < 0.0001

    if ok_A and ok_S and ok_tau:
        print(f"  ✓ Transformation correct")
    else:
        print(f"  ✗ Transformation error")

    print()
    return ok_A and ok_S and ok_tau


def test_nearest_neighbor():
    """Test nearest-neighbor search."""
    print("=" * 60)
    print("TEST 4: Nearest-Neighbor Search")
    print("=" * 60)

    encoder = BottleneckEncoder()

    truth = encoder.encode_word("truth")
    neighbors = encoder.nearest(truth, k=5, exclude=["truth"])

    print(f"  Nearest to 'truth' (A={truth.A:.3f}, S={truth.S:.3f}, τ={truth.tau:.2f}):")
    for word, dist, coord in neighbors:
        print(f"    {word:<15} dist={dist:.4f}")

    # Verify we got 5 results
    passed = len(neighbors) == 5
    if passed:
        print(f"  ✓ Found 5 neighbors")
    else:
        print(f"  ✗ Expected 5 neighbors, got {len(neighbors)}")

    print()
    return passed


def test_trajectory():
    """Test semantic trajectory."""
    print("=" * 60)
    print("TEST 5: Semantic Trajectory")
    print("=" * 60)

    encoder = BottleneckEncoder()

    trajectory = encoder.chain("truth", ["love", "think", "feel"])

    print(f"  Trajectory: truth → love → think → feel")
    for step in trajectory:
        print(f"    {step.word:<20} (A={step.A:.3f}, S={step.S:.3f}, τ={step.tau:.2f})")

    dA, dS = trajectory.total_shift
    print(f"  Total shift: (ΔA={dA:+.4f}, ΔS={dS:+.4f})")
    print(f"  Path length: {trajectory.path_length:.4f}")

    passed = len(trajectory) == 4
    if passed:
        print(f"  ✓ Trajectory has 4 steps")
    else:
        print(f"  ✗ Expected 4 steps, got {len(trajectory)}")

    print()
    return passed


def test_analogy():
    """Test analogy solving."""
    print("=" * 60)
    print("TEST 6: Analogy Solving")
    print("=" * 60)

    encoder = BottleneckEncoder()

    # man:woman :: king:?
    results = encoder.analogy("king", "man", "woman", k=5)

    print(f"  king - man + woman = ?")
    print(f"  Top 5 results: {results}")

    passed = len(results) > 0
    if passed:
        print(f"  ✓ Found analogy results")
    else:
        print(f"  ✗ No analogy results")

    print()
    return passed


def test_performance():
    """Test encoding performance."""
    print("=" * 60)
    print("TEST 7: Performance Benchmark")
    print("=" * 60)

    encoder = BottleneckEncoder()

    # Word encoding
    n_words = 1000
    start = time.time()
    for i in range(n_words):
        encoder.encode_word("truth")
    elapsed = time.time() - start
    word_per_sec = n_words / elapsed

    print(f"  Word encoding: {word_per_sec:.0f} words/sec ({elapsed*1000/n_words:.3f} ms/word)")

    # Nearest neighbor
    n_queries = 100
    coord = encoder.encode_word("truth")
    start = time.time()
    for i in range(n_queries):
        encoder.nearest(coord, k=5)
    elapsed = time.time() - start
    query_per_sec = n_queries / elapsed

    print(f"  Nearest neighbor: {query_per_sec:.0f} queries/sec ({elapsed*1000/n_queries:.3f} ms/query)")

    # Verify performance targets
    passed = word_per_sec > 10000 and query_per_sec > 10
    if passed:
        print(f"  ✓ Performance targets met")
    else:
        print(f"  ✗ Performance below target")

    print()
    return passed


def test_data_loader():
    """Test DataLoader integration."""
    print("=" * 60)
    print("TEST 8: DataLoader Integration")
    print("=" * 60)

    loader = DataLoader()

    # Load coordinates
    coords = loader.load_semantic_coordinates()
    print(f"  Loaded {len(coords)} word coordinates")

    # Load verb operators
    ops = loader.load_verb_operators_2d()
    print(f"  Loaded {len(ops)} verb operators")

    # Test convenience methods
    truth_coord = loader.get_semantic_coord("truth")
    love_op = loader.get_verb_operator_2d("love")

    print(f"  truth: {truth_coord}")
    print(f"  love:  {love_op}")

    passed = len(coords) > 20000 and len(ops) > 400
    if passed:
        print(f"  ✓ DataLoader integration works")
    else:
        print(f"  ✗ DataLoader data missing")

    print()
    return passed


def test_statistics():
    """Test vocabulary statistics."""
    print("=" * 60)
    print("TEST 9: Vocabulary Statistics")
    print("=" * 60)

    encoder = BottleneckEncoder()

    stats = encoder.stats()

    print(f"  Words: {stats['n_words']}")
    print(f"  Verbs: {stats['n_verbs']}")
    print(f"  A range: [{stats['A_range'][0]:.2f}, {stats['A_range'][1]:.2f}]")
    print(f"  S range: [{stats['S_range'][0]:.2f}, {stats['S_range'][1]:.2f}]")
    print(f"  τ range: [{stats['tau_range'][0]:.2f}, {stats['tau_range'][1]:.2f}]")

    passed = stats['n_words'] > 20000 and stats['n_verbs'] > 400
    if passed:
        print(f"  ✓ Statistics correct")
    else:
        print(f"  ✗ Statistics incorrect")

    print()
    return passed


def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " SEMANTIC BOTTLENECK INTEGRATION TEST ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    tests = [
        ("Word Encoding", test_word_encoding),
        ("Verb Encoding", test_verb_encoding),
        ("Transformations", test_transformations),
        ("Nearest Neighbor", test_nearest_neighbor),
        ("Trajectory", test_trajectory),
        ("Analogy", test_analogy),
        ("Performance", test_performance),
        ("DataLoader", test_data_loader),
        ("Statistics", test_statistics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    print()
    print(f"  {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print()
        print("  🎉 ALL TESTS PASSED!")
        print()
    else:
        print()
        print("  ⚠️  Some tests failed")
        print()

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
