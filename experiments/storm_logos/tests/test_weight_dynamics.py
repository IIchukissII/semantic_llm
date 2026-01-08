"""
Tests for Weight Dynamics Module

Tests the learning and forgetting dynamics:
    dw/dt = lambda * (w_target - w)

Run with:
    python -m storm_logos.tests.test_weight_dynamics
    python storm_logos/tests/test_weight_dynamics.py
"""

import unittest
import math
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from storm_logos.data.weight_dynamics import (
    # Constants
    W_MIN, W_MAX, LAMBDA_LEARN, LAMBDA_FORGET,
    LEARNING_INCREMENT, DORMANCY_THRESHOLD,
    WEIGHT_CORPUS, WEIGHT_CONVERSATION, WEIGHT_CONTEXT,
    # Functions
    decay_weight, learn_weight, learn_weight_simple,
    time_to_dormancy, half_life, analyze_weight,
    weight_source_name, compute_decay_batch, decay_statistics,
    get_dynamics_info
)


class TestConstants(unittest.TestCase):
    """Test that constants have expected values."""

    def test_weight_bounds(self):
        """Weight bounds should be 0.1 to 1.0."""
        self.assertEqual(W_MIN, 0.1)
        self.assertEqual(W_MAX, 1.0)

    def test_rate_constants(self):
        """Learning should be faster than forgetting."""
        self.assertEqual(LAMBDA_LEARN, 0.3)
        self.assertEqual(LAMBDA_FORGET, 0.05)
        self.assertGreater(LAMBDA_LEARN, LAMBDA_FORGET)  # 6x faster

    def test_dormancy_threshold(self):
        """Dormancy threshold should be 0.2."""
        self.assertEqual(DORMANCY_THRESHOLD, 0.2)

    def test_weight_sources(self):
        """Weight sources should follow hierarchy."""
        self.assertEqual(WEIGHT_CORPUS, 1.0)
        self.assertEqual(WEIGHT_CONVERSATION, 0.2)
        self.assertEqual(WEIGHT_CONTEXT, 0.1)
        self.assertGreater(WEIGHT_CORPUS, WEIGHT_CONVERSATION)
        self.assertGreater(WEIGHT_CONVERSATION, WEIGHT_CONTEXT)


class TestDecayWeight(unittest.TestCase):
    """Test the decay_weight function."""

    def test_no_decay_at_zero_days(self):
        """No decay should occur at 0 days."""
        self.assertEqual(decay_weight(1.0, 0), 1.0)
        self.assertEqual(decay_weight(0.5, 0), 0.5)

    def test_decay_approaches_w_min(self):
        """Weight should approach w_min over time."""
        w = decay_weight(1.0, 100)
        self.assertGreater(w, W_MIN)  # Never goes below floor
        self.assertLess(w, 0.2)       # But gets close

    def test_decay_never_below_w_min(self):
        """Weight should never go below w_min."""
        w = decay_weight(1.0, 1000)  # Very long time
        self.assertGreaterEqual(w, W_MIN)

    def test_decay_formula(self):
        """Test the exponential decay formula."""
        days = 1.0
        w_before = 1.0
        expected = W_MIN + (w_before - W_MIN) * math.exp(-LAMBDA_FORGET * days)
        actual = decay_weight(w_before, days)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_decay_at_one_day(self):
        """After 1 day, weight should decrease slightly."""
        w = decay_weight(1.0, 1)
        self.assertGreater(w, 0.95)
        self.assertLess(w, 1.0)

    def test_decay_at_one_week(self):
        """After 7 days, weight should decrease more."""
        w = decay_weight(1.0, 7)
        self.assertGreater(w, 0.7)
        self.assertLess(w, 0.85)

    def test_half_life_decay(self):
        """Weight should halve (toward floor) at half-life."""
        t_half = half_life(LAMBDA_FORGET)  # ~13.86 days
        w_start = 1.0
        w_after = decay_weight(w_start, t_half)
        # Should be at midpoint between start and floor
        expected_midpoint = W_MIN + (w_start - W_MIN) / 2
        self.assertAlmostEqual(w_after, expected_midpoint, places=2)


class TestLearnWeight(unittest.TestCase):
    """Test the learn_weight function."""

    def test_learning_increases_weight(self):
        """Learning should increase weight."""
        w_before = 0.2
        w_after = learn_weight(w_before)
        self.assertGreater(w_after, w_before)

    def test_learning_never_exceeds_w_max(self):
        """Weight should never exceed w_max."""
        w = learn_weight(0.99)
        self.assertLessEqual(w, W_MAX)

    def test_learning_from_conversation(self):
        """Learning from conversation weight should increase."""
        w = learn_weight(WEIGHT_CONVERSATION)
        self.assertGreater(w, WEIGHT_CONVERSATION)
        self.assertLess(w, W_MAX)

    def test_learning_formula(self):
        """Test the exponential learning formula."""
        w_before = 0.5
        expected = W_MAX - (W_MAX - w_before) * math.exp(-LAMBDA_LEARN)
        actual = learn_weight(w_before)
        self.assertAlmostEqual(actual, expected, places=10)


class TestLearnWeightSimple(unittest.TestCase):
    """Test the simple additive learning."""

    def test_simple_increment(self):
        """Weight should increase by increment."""
        w_before = 0.5
        w_after = learn_weight_simple(w_before)
        self.assertAlmostEqual(w_after, w_before + LEARNING_INCREMENT, places=10)

    def test_simple_cap_at_w_max(self):
        """Weight should cap at w_max."""
        w = learn_weight_simple(0.98)
        self.assertEqual(w, W_MAX)

    def test_multiple_reinforcements(self):
        """Multiple reinforcements should accumulate."""
        w = WEIGHT_CONVERSATION
        for _ in range(5):
            w = learn_weight_simple(w)
        expected = WEIGHT_CONVERSATION + 5 * LEARNING_INCREMENT
        self.assertAlmostEqual(w, expected, places=10)


class TestTimeToDormancy(unittest.TestCase):
    """Test the time_to_dormancy function."""

    def test_already_dormant(self):
        """Already dormant weights return 0 days."""
        self.assertEqual(time_to_dormancy(0.2), 0.0)
        self.assertEqual(time_to_dormancy(0.1), 0.0)

    def test_saturated_weight(self):
        """Full weight should take longest."""
        days = time_to_dormancy(1.0)
        self.assertGreater(days, 30)  # Should take many days

    def test_higher_weight_takes_longer(self):
        """Higher weights should take longer to decay."""
        days_high = time_to_dormancy(1.0)
        days_mid = time_to_dormancy(0.5)
        days_low = time_to_dormancy(0.3)
        self.assertGreater(days_high, days_mid)
        self.assertGreater(days_mid, days_low)


class TestHalfLife(unittest.TestCase):
    """Test the half_life function."""

    def test_forgetting_half_life(self):
        """Forgetting half-life should be ~13.86 days."""
        t_half = half_life(LAMBDA_FORGET)
        expected = math.log(2) / LAMBDA_FORGET
        self.assertAlmostEqual(t_half, expected, places=2)
        self.assertAlmostEqual(t_half, 13.86, places=1)

    def test_learning_half_life(self):
        """Learning half-life should be ~2.31 events."""
        t_half = half_life(LAMBDA_LEARN)
        expected = math.log(2) / LAMBDA_LEARN
        self.assertAlmostEqual(t_half, expected, places=2)


class TestAnalyzeWeight(unittest.TestCase):
    """Test the analyze_weight function."""

    def test_dormant_analysis(self):
        """Low weights should be marked dormant."""
        state = analyze_weight(0.15)
        self.assertTrue(state.is_dormant)
        self.assertFalse(state.is_active)

    def test_active_analysis(self):
        """Higher weights should be marked active."""
        state = analyze_weight(0.5)
        self.assertTrue(state.is_active)
        self.assertFalse(state.is_dormant)

    def test_saturated_analysis(self):
        """Full weights should be marked saturated."""
        state = analyze_weight(1.0)
        self.assertTrue(state.is_saturated)

    def test_relative_strength(self):
        """Relative strength should be normalized."""
        state_min = analyze_weight(W_MIN)
        state_max = analyze_weight(W_MAX)
        self.assertAlmostEqual(state_min.relative_strength, 0.0, places=2)
        self.assertAlmostEqual(state_max.relative_strength, 1.0, places=2)


class TestWeightSourceName(unittest.TestCase):
    """Test the weight_source_name function."""

    def test_corpus_source(self):
        """High weights should be corpus."""
        self.assertEqual(weight_source_name(1.0), "corpus")
        self.assertEqual(weight_source_name(0.95), "corpus")

    def test_conversation_source(self):
        """Low weights should be conversation."""
        self.assertEqual(weight_source_name(0.2), "conversation")

    def test_context_source(self):
        """Very low weights should be context."""
        self.assertEqual(weight_source_name(0.1), "context")


class TestBatchOperations(unittest.TestCase):
    """Test batch decay operations."""

    def test_compute_decay_batch(self):
        """Batch decay should work on lists."""
        weights = [1.0, 0.8, 0.5, 0.3]
        decayed = compute_decay_batch(weights, 1.0)
        self.assertEqual(len(decayed), len(weights))
        for w_before, w_after in zip(weights, decayed):
            self.assertLessEqual(w_after, w_before)
            self.assertGreaterEqual(w_after, W_MIN)

    def test_decay_statistics(self):
        """Statistics should be computed correctly."""
        before = [1.0, 0.8, 0.5, 0.3]
        after = compute_decay_batch(before, 10.0)
        stats = decay_statistics(before, after)

        self.assertEqual(stats["count"], 4)
        self.assertGreater(stats["total_decay"], 0)
        self.assertGreater(stats["avg_weight_before"], stats["avg_weight_after"])


class TestDynamicsInfo(unittest.TestCase):
    """Test the get_dynamics_info function."""

    def test_info_structure(self):
        """Info should have expected structure."""
        info = get_dynamics_info()
        self.assertIn("parameters", info)
        self.assertIn("weight_sources", info)
        self.assertIn("half_lives", info)
        self.assertIn("key_insight", info)

    def test_info_parameters(self):
        """Parameters should match constants."""
        info = get_dynamics_info()
        self.assertEqual(info["parameters"]["w_min"], W_MIN)
        self.assertEqual(info["parameters"]["w_max"], W_MAX)
        self.assertEqual(info["parameters"]["lambda_forget"], LAMBDA_FORGET)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_negative_days(self):
        """Negative days should return unchanged weight."""
        self.assertEqual(decay_weight(1.0, -1), 1.0)

    def test_very_long_decay(self):
        """Very long decay should approach but not go below floor."""
        w = decay_weight(1.0, 10000)
        self.assertGreaterEqual(w, W_MIN)
        self.assertLess(abs(w - W_MIN), 0.001)

    def test_w_min_decay(self):
        """Decaying from w_min should stay at w_min."""
        w = decay_weight(W_MIN, 100)
        self.assertEqual(w, W_MIN)

    def test_learning_at_w_max(self):
        """Learning at w_max should stay at w_max."""
        w = learn_weight(W_MAX)
        self.assertEqual(w, W_MAX)

    def test_simple_learning_at_w_max(self):
        """Simple learning at w_max should stay at w_max."""
        w = learn_weight_simple(W_MAX)
        self.assertEqual(w, W_MAX)


class TestCapacitorAnalogy(unittest.TestCase):
    """
    Test that the dynamics behave like a capacitor.

    Learning  = Charging (voltage rises toward max)
    Forgetting = Discharging (voltage falls toward baseline)
    """

    def test_charging_curve(self):
        """Learning should follow exponential charging curve."""
        w = WEIGHT_CONVERSATION
        weights = [w]
        for _ in range(20):
            w = learn_weight(w)
            weights.append(w)

        # Should approach W_MAX asymptotically
        for i in range(len(weights) - 1):
            delta = weights[i + 1] - weights[i]
            remaining = W_MAX - weights[i]
            # Each step should be proportional to remaining distance
            if remaining > 0.01:
                ratio = delta / remaining
                self.assertGreater(ratio, 0.2)
                self.assertLess(ratio, 0.3)  # ~26% per step

    def test_discharging_curve(self):
        """Forgetting should follow exponential discharging curve."""
        w = 1.0
        weights = [w]
        for _ in range(30):  # 30 days
            w = decay_weight(w, 1.0)
            weights.append(w)

        # Should approach W_MIN asymptotically
        for i in range(len(weights) - 1):
            delta = weights[i] - weights[i + 1]
            excess = weights[i] - W_MIN
            # Each step should be proportional to excess above floor
            if excess > 0.01:
                ratio = delta / excess
                self.assertGreater(ratio, 0.04)
                self.assertLess(ratio, 0.06)  # ~4.9% per day


def run_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Weight Dynamics Tests")
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
