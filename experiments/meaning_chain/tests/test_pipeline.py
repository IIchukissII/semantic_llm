#!/usr/bin/env python3
"""
Test the meaning chain pipeline.

Tests:
1. Decomposition: sentence -> concepts
2. Tree building: concepts -> meaning tree
3. Rendering: tree -> LLM prompt
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.data_loader import DataLoader


def test_decomposer():
    """Test sentence decomposition."""
    print("\n" + "="*60)
    print("TEST: Decomposer")
    print("="*60)

    from experiments.meaning_chain.core.decomposer import Decomposer

    decomposer = Decomposer()

    test_cases = [
        "What is the meaning of love?",
        "How does happiness relate to life?",
        "Tell me about truth and beauty",
        "Why do people seek freedom?",
    ]

    for query in test_cases:
        print(f"\nQuery: {query}")
        result = decomposer.decompose(query)
        print(f"  Nouns: {result.nouns}")
        print(f"  Verbs: {result.verbs}")
        print(f"  Unknown: {result.unknown_words}")

    print("\n✓ Decomposer test complete")
    return True


def test_tree_builder():
    """Test meaning tree construction."""
    print("\n" + "="*60)
    print("TEST: TreeBuilder")
    print("="*60)

    from experiments.meaning_chain.core.tree_builder import TreeBuilder, TreeBuilderConfig

    builder = TreeBuilder(config=TreeBuilderConfig(max_depth=3, max_children=3))

    test_roots = [
        ["love"],
        ["happiness", "life"],
        ["truth", "freedom"],
    ]

    for roots in test_roots:
        print(f"\nRoots: {roots}")
        tree = builder.build_tree(roots)

        print(f"  Total nodes: {tree.total_nodes}")
        print(f"  Paths: {len(tree.get_paths())}")

        # Show first few paths
        for path in tree.get_paths()[:3]:
            print(f"    Path: {' -> '.join(path)}")

        # Show transitions
        print(f"  Transitions: {len(tree.get_transitions())}")
        for t in tree.get_transitions()[:3]:
            print(f"    {t[0]} --({t[1]})--> {t[2]}")

    print("\n✓ TreeBuilder test complete")
    return True


def test_tree_structure():
    """Test tree pretty printing."""
    print("\n" + "="*60)
    print("TEST: Tree Structure")
    print("="*60)

    from experiments.meaning_chain.core.tree_builder import TreeBuilder, TreeBuilderConfig

    builder = TreeBuilder(config=TreeBuilderConfig(max_depth=2, max_children=2))

    tree = builder.build_tree(["love"], "What is love?")
    print(tree.pretty_print())

    print("\n✓ Tree structure test complete")
    return True


def test_serializer():
    """Test tree serialization for prompts."""
    print("\n" + "="*60)
    print("TEST: TreeSerializer")
    print("="*60)

    from experiments.meaning_chain.core.tree_builder import TreeBuilder, TreeBuilderConfig
    from experiments.meaning_chain.core.renderer import TreeSerializer

    builder = TreeBuilder(config=TreeBuilderConfig(max_depth=2, max_children=2))
    tree = builder.build_tree(["happiness"], "What brings happiness?")

    serializer = TreeSerializer()

    print("\n--- Paths Text ---")
    print(serializer.to_paths_text(tree))

    print("\n--- Transitions Text ---")
    print(serializer.to_transitions_text(tree))

    print("\n--- Concepts List ---")
    print(serializer.to_concepts_list(tree))

    print("\n✓ Serializer test complete")
    return True


def test_full_pipeline():
    """Test complete pipeline without LLM."""
    print("\n" + "="*60)
    print("TEST: Full Pipeline (without LLM)")
    print("="*60)

    from experiments.meaning_chain.core.tree_builder import MeaningChainPipeline

    pipeline = MeaningChainPipeline()

    query = "What is the connection between love and happiness?"
    print(f"\nQuery: {query}")

    tree = pipeline.process(query)

    print(f"\nTree Summary:")
    print(f"  Roots: {tree.root_count}")
    print(f"  Total Nodes: {tree.total_nodes}")
    print(f"  Paths: {len(tree.get_paths())}")

    print(f"\nTree Structure:")
    print(tree.pretty_print())

    print("\n✓ Full pipeline test complete")
    return True


def test_feedback():
    """Test feedback analyzer."""
    print("\n" + "="*60)
    print("TEST: Feedback Analyzer")
    print("="*60)

    from experiments.meaning_chain.core.tree_builder import TreeBuilder, TreeBuilderConfig
    from experiments.meaning_chain.core.feedback import FeedbackAnalyzer

    builder = TreeBuilder(config=TreeBuilderConfig(max_depth=2, max_children=2))
    analyzer = FeedbackAnalyzer()

    tree = builder.build_tree(["love", "happiness"], "love and happiness")

    # Test good response (includes concepts)
    good_response = """Love and happiness are deeply connected.
    When we experience love, happiness naturally follows.
    Joy emerges from the bond between these concepts."""

    # Test poor response (missing concepts)
    poor_response = """The weather is nice today.
    I enjoy watching movies on weekends."""

    print("\n--- Good Response ---")
    result = analyzer.analyze(good_response, tree)
    print(f"  Alignment: {result.alignment_score:.0%}")
    print(f"  Coverage: {result.concept_coverage:.0%}")
    print(f"  Accepted: {result.accepted}")

    print("\n--- Poor Response ---")
    result = analyzer.analyze(poor_response, tree)
    print(f"  Alignment: {result.alignment_score:.0%}")
    print(f"  Coverage: {result.concept_coverage:.0%}")
    print(f"  Accepted: {result.accepted}")
    print(f"  Suggestions: {result.suggestions}")

    print("\n✓ Feedback test complete")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  MEANING CHAIN PIPELINE TESTS")
    print("="*60)

    tests = [
        test_decomposer,
        test_tree_builder,
        test_tree_structure,
        test_serializer,
        test_full_pipeline,
        test_feedback,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
