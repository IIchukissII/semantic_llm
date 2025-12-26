#!/usr/bin/env python3
"""
Test Learning Implementation for Meaning Chain

Tests:
1. EntropyCalculator - entropy and τ computation
2. Neo4jLearningStore - storing observations in Neo4j
3. BookProcessor with learning - processing books and learning concepts
4. ConversationLearner with learning - learning from conversations

Usage:
    python tests/test_learning.py
"""

import sys
from pathlib import Path

# Add parent to path
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.learning import (
    EntropyCalculator,
    VectorCalculator,
    ConceptLearner,
    Neo4jLearningStore,
    LearningGraphAdapter
)
from graph.meaning_graph import MeaningGraph


def test_entropy_calculator():
    """Test entropy and τ computation."""
    print("=" * 60)
    print("Test: EntropyCalculator")
    print("=" * 60)

    calc = EntropyCalculator()

    # Test 1: Uniform distribution (max entropy)
    uniform = {"a": 1, "b": 1, "c": 1, "d": 1}
    h_uniform = calc.shannon_entropy(uniform)
    h_norm_uniform = calc.normalized_entropy(uniform)
    tau_uniform = calc.tau_from_entropy(h_norm_uniform)

    print(f"\n1. Uniform distribution {uniform}")
    print(f"   H = {h_uniform:.3f} bits")
    print(f"   H_norm = {h_norm_uniform:.3f}")
    print(f"   τ = {tau_uniform:.2f} (should be ~1, abstract)")

    assert 0.9 <= h_norm_uniform <= 1.0, "Uniform should have high normalized entropy"
    assert 1.0 <= tau_uniform <= 1.5, "Uniform should have low τ (abstract)"

    # Test 2: Concentrated distribution (low entropy)
    concentrated = {"dominant": 100, "rare": 1}
    h_conc = calc.shannon_entropy(concentrated)
    h_norm_conc = calc.normalized_entropy(concentrated)
    tau_conc = calc.tau_from_entropy(h_norm_conc)

    print(f"\n2. Concentrated distribution {concentrated}")
    print(f"   H = {h_conc:.3f} bits")
    print(f"   H_norm = {h_norm_conc:.3f}")
    print(f"   τ = {tau_conc:.2f} (should be ~6, concrete)")

    assert 0 <= h_norm_conc <= 0.3, "Concentrated should have low normalized entropy"
    assert 5.0 <= tau_conc <= 6.0, "Concentrated should have high τ (concrete)"

    # Test 3: Middle distribution (more skewed)
    middle = {"dominant": 20, "common": 5, "rare": 1}
    h_mid = calc.shannon_entropy(middle)
    h_norm_mid = calc.normalized_entropy(middle)
    tau_mid = calc.tau_from_entropy(h_norm_mid)

    print(f"\n3. Skewed distribution {middle}")
    print(f"   H = {h_mid:.3f} bits")
    print(f"   H_norm = {h_norm_mid:.3f}")
    print(f"   τ = {tau_mid:.2f}")

    # Just verify it's between the extremes
    assert h_norm_conc < h_norm_mid < h_norm_uniform, "Middle should be between extremes"
    assert tau_uniform < tau_mid < tau_conc, "τ should be between extremes"

    # Test 4: Confidence from observations
    conf_1 = calc.confidence_from_observations(1)
    conf_10 = calc.confidence_from_observations(10)
    conf_100 = calc.confidence_from_observations(100)
    conf_1000 = calc.confidence_from_observations(1000)

    print(f"\n4. Confidence from observations:")
    print(f"   n=1:    conf = {conf_1:.3f}")
    print(f"   n=10:   conf = {conf_10:.3f}")
    print(f"   n=100:  conf = {conf_100:.3f}")
    print(f"   n=1000: conf = {conf_1000:.3f}")

    assert conf_1 < conf_10 < conf_100 < conf_1000, "Confidence should increase with observations"
    assert conf_1 >= 0.1, "Confidence should never be below 0.1"
    assert conf_1000 <= 1.0, "Confidence should never exceed 1.0"

    print("\n✓ EntropyCalculator tests passed!")
    return True


def test_concept_learner():
    """Test in-memory ConceptLearner."""
    print("\n" + "=" * 60)
    print("Test: ConceptLearner (in-memory)")
    print("=" * 60)

    learner = ConceptLearner()

    # Learn a concept through observations
    print("\n1. Learning 'serendipity' from adjectives...")

    learner.observe_adjective("serendipity", "happy", count=5, source="book")
    learner.observe_adjective("serendipity", "unexpected", count=3, source="book")
    learner.observe_adjective("serendipity", "fortunate", count=2, source="book")

    state = learner.get_concept_state("serendipity")
    print(f"   After 3 adj types (10 observations):")
    print(f"   - variety: {state.variety}")
    print(f"   - H: {state.h_adj:.3f}, H_norm: {state.h_adj_norm:.3f}")
    print(f"   - τ: {state.tau:.2f}")
    print(f"   - confidence: {state.confidence:.3f}")

    assert state.variety == 3, "Should have 3 adjective types"
    assert state.total_count == 10, "Should have 10 total observations"

    # Save values before next observations (state is modified in place)
    h_adj_before = state.h_adj
    variety_before = state.variety

    # Add more adjectives with similar weights to increase uniformity
    print("\n2. Adding more adjectives with similar weights...")

    learner.observe_adjective("serendipity", "wonderful", count=4, source="conversation")
    learner.observe_adjective("serendipity", "rare", count=3, source="conversation")
    learner.observe_adjective("serendipity", "magical", count=3, source="conversation")

    state2 = learner.get_concept_state("serendipity")
    print(f"   After 6 adj types ({state2.total_count} observations):")
    print(f"   - variety: {state2.variety}")
    print(f"   - H: {state2.h_adj:.3f}, H_norm: {state2.h_adj_norm:.3f}")
    print(f"   - τ: {state2.tau:.2f}")
    print(f"   - confidence: {state2.confidence:.3f}")

    assert state2.variety == 6, "Should have 6 adjective types"
    assert state2.variety > variety_before, "Variety should increase"
    assert state2.h_adj > h_adj_before, "Raw entropy should increase with variety"

    # Test batch observation
    print("\n3. Batch observation for 'ephemeral'...")

    learner.observe_batch("ephemeral", {
        "brief": 10,
        "fleeting": 8,
        "transient": 5,
        "momentary": 3
    }, source="corpus")

    state3 = learner.get_concept_state("ephemeral")
    print(f"   - variety: {state3.variety}")
    print(f"   - τ: {state3.tau:.2f}")

    assert state3.variety == 4, "Should have 4 adjective types"

    # Test stats
    print("\n4. Learning stats:")
    stats = learner.get_stats()
    print(f"   - Total concepts: {stats['total_concepts']}")
    print(f"   - Total observations: {stats['total_observations']}")
    print(f"   - Sources: {stats['sources']}")

    assert stats['total_concepts'] == 2, "Should have 2 concepts"

    print("\n✓ ConceptLearner tests passed!")
    return True


def test_neo4j_learning():
    """Test Neo4j learning store (requires running Neo4j)."""
    print("\n" + "=" * 60)
    print("Test: Neo4jLearningStore")
    print("=" * 60)

    # Connect to graph
    graph = MeaningGraph()

    if not graph.is_connected():
        print("⚠ Neo4j not connected. Skipping Neo4j tests.")
        print("  Start with: cd config && docker-compose up -d")
        return False

    print("\n1. Setting up learning schema...")
    store = Neo4jLearningStore(graph.driver)
    store.setup_schema()

    # Test observation
    print("\n2. Testing adjective observation...")

    # Use a unique test word to avoid conflicts
    import time
    test_word = f"testconcept_{int(time.time())}"

    store.observe_adjective(test_word, "beautiful", count=5, source="test")
    store.observe_adjective(test_word, "mysterious", count=3, source="test")
    store.observe_adjective(test_word, "profound", count=2, source="test")

    # Get distribution
    dist = store.get_adj_distribution(test_word)
    print(f"   Distribution for '{test_word}': {dist}")

    assert len(dist) == 3, "Should have 3 adjectives"
    assert dist.get("beautiful") == 5, "beautiful should have count 5"

    # Update concept parameters
    print("\n3. Updating concept parameters from distribution...")

    result = store.update_concept_from_distribution(test_word)
    print(f"   τ = {result['tau']:.2f}")
    print(f"   g = {result['g']:.3f}")
    print(f"   variety = {result['variety']}")
    print(f"   confidence = {result['confidence']:.3f}")

    assert result['variety'] == 3, "Should have variety 3"
    assert 1.0 <= result['tau'] <= 6.0, "τ should be in valid range"

    # Clean up test data
    print("\n4. Cleaning up test data...")
    with graph.driver.session() as session:
        session.run("""
            MATCH (c:Concept {word: $word})
            OPTIONAL MATCH (c)-[r:DESCRIBED_BY]->(a:Adjective)
            DELETE r
            DETACH DELETE c
        """, word=test_word)

    print("\n✓ Neo4jLearningStore tests passed!")
    graph.close()
    return True


def test_learning_stats():
    """Test learning statistics from Neo4j."""
    print("\n" + "=" * 60)
    print("Test: Learning Statistics")
    print("=" * 60)

    graph = MeaningGraph()

    if not graph.is_connected():
        print("⚠ Neo4j not connected. Skipping.")
        return False

    store = Neo4jLearningStore(graph.driver)
    stats = store.get_stats()

    print(f"\n  Learned concepts: {stats.get('learned_concepts', 0)}")
    print(f"  Adjectives: {stats.get('adjectives', 0)}")
    print(f"  Observation edges: {stats.get('observation_edges', 0)}")
    print(f"  Total observations: {stats.get('total_observations', 0)}")

    if stats.get('avg_tau'):
        print(f"  Avg τ: {stats['avg_tau']:.2f}")
    if stats.get('avg_confidence'):
        print(f"  Avg confidence: {stats['avg_confidence']:.3f}")

    print("\n✓ Stats retrieved successfully!")
    graph.close()
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Meaning Chain Learning Tests")
    print("=" * 60)

    results = {
        "EntropyCalculator": test_entropy_calculator(),
        "ConceptLearner": test_concept_learner(),
        "Neo4jLearningStore": test_neo4j_learning(),
        "LearningStats": test_learning_stats()
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED/SKIPPED"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed or were skipped."))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
