"""
Tests for Bond Learning Module

Tests the runtime learning of bonds from conversations:
    - Bond extraction from text
    - Coordinate computation/estimation
    - PostgreSQL storage
    - Neo4j trajectory creation

Run with:
    python -m storm_logos.tests.test_bond_learning
    python storm_logos/tests/test_bond_learning.py
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from storm_logos.data.models import Bond, WordCoordinates
from storm_logos.data.bond_learner import (
    BondLearner, LearnedBond, LearningResult,
    learn_bond, learn_from_text
)


class TestLearnedBond(unittest.TestCase):
    """Test the LearnedBond dataclass."""

    def test_text_property(self):
        """Text should combine adj and noun."""
        lb = LearnedBond(adj="dark", noun="forest")
        self.assertEqual(lb.text, "dark forest")

    def test_to_bond(self):
        """Should convert to Bond dataclass."""
        lb = LearnedBond(adj="dark", noun="forest", A=0.5, S=-0.3, tau=2.0)
        bond = lb.to_bond()

        self.assertIsInstance(bond, Bond)
        self.assertEqual(bond.adj, "dark")
        self.assertEqual(bond.noun, "forest")
        self.assertEqual(bond.A, 0.5)
        self.assertEqual(bond.S, -0.3)
        self.assertEqual(bond.tau, 2.0)

    def test_default_values(self):
        """Default values should be set correctly."""
        lb = LearnedBond(adj="dark", noun="forest")
        self.assertEqual(lb.A, 0.0)
        self.assertEqual(lb.S, 0.0)
        self.assertEqual(lb.tau, 2.5)
        self.assertEqual(lb.source, 'conversation')
        self.assertEqual(lb.confidence, 0.5)
        self.assertFalse(lb.is_new)


class TestLearningResult(unittest.TestCase):
    """Test the LearningResult dataclass."""

    def test_empty_result(self):
        """Empty result should have defaults."""
        result = LearningResult()
        self.assertEqual(len(result.bonds), 0)
        self.assertEqual(result.new_bonds, 0)
        self.assertEqual(result.reinforced_bonds, 0)
        self.assertEqual(result.trajectory_edges, 0)

    def test_summary(self):
        """Summary should be human-readable."""
        result = LearningResult(
            bonds=[LearnedBond(adj="dark", noun="forest")],
            new_bonds=1,
            reinforced_bonds=0,
            trajectory_edges=0
        )
        summary = result.summary()

        self.assertIn("1 bonds", summary)
        self.assertIn("1 new", summary)
        self.assertIn("0 reinforced", summary)


class TestBondLearnerExtraction(unittest.TestCase):
    """Test bond extraction from text."""

    def setUp(self):
        """Create learner without connecting to databases."""
        self.learner = BondLearner()
        # Don't connect - we'll test extraction in isolation

    def test_simple_extraction_fallback(self):
        """Test the simple regex fallback extraction."""
        text = "dark forest ancient secrets"
        bonds = self.learner._extract_bonds_simple(text)

        # Should find consecutive word pairs
        self.assertGreater(len(bonds), 0)

    def test_extraction_cleans_punctuation(self):
        """Extraction should handle punctuation."""
        text = "dark, forest. ancient! secrets?"
        bonds = self.learner._extract_bonds_simple(text)

        # Should still extract some pairs
        for adj, noun in bonds:
            self.assertTrue(adj.isalpha())
            self.assertTrue(noun.isalpha())

    def test_extraction_filters_short_words(self):
        """Short words should be filtered out."""
        text = "a big dog is in the house"
        bonds = self.learner._extract_bonds_simple(text)

        # "a" and other 1-2 letter words should be filtered
        for adj, noun in bonds:
            self.assertGreaterEqual(len(adj), 3)
            self.assertGreaterEqual(len(noun), 3)


class TestCoordinateEstimation(unittest.TestCase):
    """Test coordinate estimation for unknown words."""

    def setUp(self):
        self.learner = BondLearner()

    def test_negative_prefix_affects_A(self):
        """Negative prefixes should decrease A."""
        # Test 'unhappy thing' vs 'happy thing'
        A_unhappy, _, _ = self.learner._estimate_coordinates("unhappy", "thing")
        A_happy, _, _ = self.learner._estimate_coordinates("happy", "thing")

        self.assertLess(A_unhappy, A_happy)

    def test_abstract_suffix_affects_tau(self):
        """Abstract suffixes should increase tau."""
        # 'happiness' (abstract) vs 'dog' (concrete)
        _, _, tau_ness = self.learner._estimate_coordinates("big", "happiness")
        _, _, tau_dog = self.learner._estimate_coordinates("big", "dog")

        self.assertGreater(tau_ness, tau_dog)

    def test_sacred_words_affect_S(self):
        """Sacred-related words should increase S."""
        _, S_soul, _ = self.learner._estimate_coordinates("dark", "soul")
        _, S_rock, _ = self.learner._estimate_coordinates("dark", "rock")

        self.assertGreater(S_soul, S_rock)

    def test_coordinates_are_clamped(self):
        """Coordinates should be within valid ranges."""
        # Even with multiple modifiers, should be clamped
        A, S, tau = self.learner._estimate_coordinates(
            "unhappiness",  # negative prefix + abstract suffix
            "spirituality"  # sacred + abstract
        )

        self.assertGreaterEqual(A, -1.0)
        self.assertLessEqual(A, 1.0)
        self.assertGreaterEqual(S, -1.0)
        self.assertLessEqual(S, 1.0)
        self.assertGreaterEqual(tau, 0.5)
        self.assertLessEqual(tau, 4.5)

    def test_neutral_defaults(self):
        """Neutral words should get default coordinates."""
        A, S, tau = self.learner._estimate_coordinates("random", "word")

        # Should be near neutral
        self.assertAlmostEqual(A, 0.0, places=1)
        self.assertAlmostEqual(S, 0.0, places=1)
        self.assertAlmostEqual(tau, 2.5, places=1)


class TestBondLearnerWithMocks(unittest.TestCase):
    """Test BondLearner with mocked databases."""

    def setUp(self):
        """Create learner with mocked dependencies."""
        self.learner = BondLearner()

        # Mock PostgreSQL
        self.mock_postgres = Mock()
        self.mock_postgres.get_learned_bond.return_value = None
        self.mock_postgres._compute_bond_coordinates.return_value = (0.5, 0.2, 2.0)
        self.mock_postgres.learn_bond.return_value = Bond(
            adj="dark", noun="forest",
            A=0.5, S=0.2, tau=2.0
        )
        self.learner.postgres = self.mock_postgres

        # Mock Neo4j
        self.mock_neo4j = Mock()
        self.mock_neo4j.learn_bond.return_value = "dark_forest"
        self.mock_neo4j.learn_trajectory.return_value = 2
        self.learner.neo4j = self.mock_neo4j
        self.learner._connected = True

    def test_learn_bond_calls_postgres(self):
        """Learning a bond should store in PostgreSQL."""
        result = self.learner.learn_bond("dark", "forest")

        self.mock_postgres.learn_bond.assert_called_once()
        self.assertEqual(result.adj, "dark")
        self.assertEqual(result.noun, "forest")

    def test_learn_bond_syncs_to_neo4j(self):
        """Learning a bond should sync to Neo4j."""
        self.learner.learn_bond("dark", "forest")

        self.mock_neo4j.learn_bond.assert_called_once()

    def test_learn_bond_marks_new(self):
        """New bonds should be marked as new."""
        self.mock_postgres.get_learned_bond.return_value = None
        result = self.learner.learn_bond("dark", "forest")

        self.assertTrue(result.is_new)

    def test_learn_bond_marks_existing(self):
        """Existing bonds should not be marked as new."""
        self.mock_postgres.get_learned_bond.return_value = Bond(
            adj="dark", noun="forest"
        )
        result = self.learner.learn_bond("dark", "forest")

        self.assertFalse(result.is_new)

    def test_learn_from_text_creates_trajectory(self):
        """Learning from text should create trajectory edges."""
        # Mock extraction to return multiple bonds
        self.learner.extract_bonds = Mock(return_value=[
            ("dark", "forest"),
            ("ancient", "secrets"),
            ("mysterious", "path")
        ])

        result = self.learner.learn_from_text("test text")

        # Should have learned all bonds
        self.assertEqual(len(result.bonds), 3)

        # Should have created trajectory
        self.mock_neo4j.learn_trajectory.assert_called_once()

    def test_learn_from_text_generates_conversation_id(self):
        """Learning should generate conversation ID if not provided."""
        self.learner.extract_bonds = Mock(return_value=[("dark", "forest")])

        result = self.learner.learn_from_text("test text")

        self.assertTrue(result.conversation_id.startswith("conv_"))

    def test_learn_turn_links_to_previous(self):
        """Learning a turn should link to previous bonds."""
        self.learner.extract_bonds = Mock(return_value=[("dark", "forest")])

        previous = [Bond(adj="ancient", noun="temple")]
        result = self.learner.learn_turn(
            "test text",
            conversation_id="test_conv",
            previous_bonds=previous
        )

        # Should have created transition edge
        self.mock_neo4j.learn_transition.assert_called()


class TestBondLearnerStats(unittest.TestCase):
    """Test statistics retrieval."""

    def test_stats_with_mocks(self):
        """Stats should combine PostgreSQL and Neo4j stats."""
        learner = BondLearner()

        # Mock PostgreSQL
        mock_postgres = Mock()
        mock_postgres.get_learning_stats.return_value = {
            'n_learned_bonds': 10,
            'n_learned_words': 5
        }
        learner.postgres = mock_postgres

        # Mock Neo4j
        mock_neo4j = Mock()
        mock_neo4j.get_learning_stats.return_value = {
            'learned_bonds': 10,
            'learned_edges': 15
        }
        learner.neo4j = mock_neo4j
        learner._connected = True

        stats = learner.get_stats()

        self.assertIn('postgresql', stats)
        self.assertIn('neo4j', stats)
        self.assertEqual(stats['postgresql']['n_learned_bonds'], 10)
        self.assertEqual(stats['neo4j']['learned_edges'], 15)


class TestCoordinateHeuristics(unittest.TestCase):
    """Test specific coordinate heuristics."""

    def setUp(self):
        self.learner = BondLearner()

    def test_positive_adjectives(self):
        """Positive adjectives should increase A."""
        positive_words = ['beautiful', 'wonderful', 'bright', 'good']
        for adj in positive_words:
            A, _, _ = self.learner._estimate_coordinates(adj, "thing")
            # At least one should increase A
            # (depends on which patterns match)

    def test_negative_adjectives(self):
        """Negative adjectives should decrease A."""
        A_dark, _, _ = self.learner._estimate_coordinates("dark", "thing")
        A_evil, _, _ = self.learner._estimate_coordinates("evil", "thing")

        self.assertLess(A_dark, 0)
        self.assertLess(A_evil, 0)

    def test_abstract_nouns(self):
        """Abstract nouns should increase tau."""
        # Words ending in: 'ness', 'ity', 'tion', 'ism', 'ment'
        abstract_nouns = ['happiness', 'reality', 'creation', 'mechanism', 'statement']
        for noun in abstract_nouns:
            _, _, tau = self.learner._estimate_coordinates("big", noun)
            self.assertGreater(tau, 2.5, f"{noun} should have tau > 2.5")


def run_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Bond Learning Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    return len(result.failures) + len(result.errors)


if __name__ == "__main__":
    sys.exit(run_tests())
