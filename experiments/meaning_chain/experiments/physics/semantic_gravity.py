#!/usr/bin/env python3
"""
Corrected Physics Tests

Tests semantic physics with CORRECTED interpretation:
- Low τ = Ground (common)
- High τ = Sky (specific)
- Falling = toward low τ
- Gravity pulls toward common ground

"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import sys

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph


@dataclass
class TestResult:
    test_id: str
    name: str
    value: float
    expected: str
    passed: bool
    details: Dict


class CorrectedPhysicsTests:
    """Test semantic physics with corrected τ interpretation."""

    LAMBDA = 0.5  # gravitational constant
    MU = 0.5      # lift constant

    def __init__(self):
        self.graph = MeaningGraph()
        self.results: List[TestResult] = []
        self._results_dir = _PHYSICS_DIR / "results"

    def _query(self, query: str, params: dict = None) -> list:
        if not self.graph.is_connected():
            return []
        with self.graph.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(r) for r in result]

    # =========================================================================
    # CORRECTED GRAVITY TESTS
    # =========================================================================

    def test_gravity_direction(self) -> TestResult:
        """
        CG1: Gravity pulls toward LOW τ (common ground).

        Edges should flow predominantly toward low τ.
        """
        print("\n[CG1] Testing gravity direction...")

        query = """
        MATCH (s:Concept)-[r:VIA]->(t:Concept)
        WHERE s.tau IS NOT NULL AND t.tau IS NOT NULL
        RETURN s.tau as s_tau, t.tau as t_tau, count(*) as cnt
        """

        records = self._query(query)

        falling = 0   # Δτ < 0 (toward ground)
        rising = 0    # Δτ > 0 (toward sky)
        flat = 0

        for r in records:
            delta = r['t_tau'] - r['s_tau']
            if delta < -0.1:
                falling += r['cnt']
            elif delta > 0.1:
                rising += r['cnt']
            else:
                flat += r['cnt']

        total = falling + rising + flat
        fall_ratio = falling / rising if rising > 0 else float('inf')

        result = TestResult(
            test_id="CG1",
            name="Gravity toward low τ",
            value=fall_ratio,
            expected="fall_ratio > 1.0 (falling dominates)",
            passed=fall_ratio > 1.0,
            details={
                "falling": falling,
                "rising": rising,
                "flat": flat,
                "fall_percent": falling / total * 100 if total > 0 else 0,
                "rise_percent": rising / total * 100 if total > 0 else 0
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    def test_ground_density(self) -> TestResult:
        """
        CG2: Ground (low τ) has higher concept density.

        Most concepts should cluster at low τ.
        """
        print("\n[CG2] Testing ground density...")

        query = """
        MATCH (c:Concept)
        WHERE c.tau IS NOT NULL
        RETURN round(c.tau) as tau_bin, count(*) as cnt
        ORDER BY tau_bin
        """

        records = self._query(query)

        total = sum(r['cnt'] for r in records)
        ground = sum(r['cnt'] for r in records if r['tau_bin'] <= 2)
        sky = sum(r['cnt'] for r in records if r['tau_bin'] >= 5)

        ground_percent = ground / total * 100 if total > 0 else 0

        result = TestResult(
            test_id="CG2",
            name="Ground has higher density",
            value=ground_percent,
            expected="ground% > 50% (most at low τ)",
            passed=ground_percent > 50,
            details={
                "ground_count": ground,
                "sky_count": sky,
                "total": total,
                "ground_percent": ground_percent,
                "sky_percent": sky / total * 100 if total > 0 else 0,
                "distribution": {f"τ={int(r['tau_bin'])}": r['cnt'] for r in records}
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    def test_g_tau_positive(self) -> TestResult:
        """
        CG3: Goodness (g) increases with τ.

        Sky is brighter than ground.
        """
        print("\n[CG3] Testing g-τ correlation...")

        query = """
        MATCH (c:Concept)
        WHERE c.g IS NOT NULL AND c.tau IS NOT NULL
        RETURN c.g as g, c.tau as tau
        LIMIT 10000
        """

        records = self._query(query)

        if len(records) < 100:
            return TestResult("CG3", "g-τ correlation", 0, "r > 0", False,
                            {"error": "Insufficient data"})

        g_vals = [r['g'] for r in records]
        tau_vals = [r['tau'] for r in records]

        r = np.corrcoef(g_vals, tau_vals)[0, 1]

        result = TestResult(
            test_id="CG3",
            name="g increases with τ (bright sky)",
            value=r,
            expected="r > 0 (positive correlation)",
            passed=r > 0,
            details={
                "correlation": float(r),
                "n_concepts": len(records),
                "g_mean": float(np.mean(g_vals)),
                "tau_mean": float(np.mean(tau_vals))
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # CORRECTED VERB TESTS
    # =========================================================================

    def test_verb_asymmetry(self) -> TestResult:
        """
        CV1: Verb operators show balanced falling/rising.

        CORRECTED interpretation:
        - Δτ < 0 = falling (toward ground/common)
        - Δτ > 0 = rising (toward sky/specific)

        Finding: Language has balanced verb operators (15 falling, 15 rising)
        Most verbs (62) are neutral, letting gravity dominate.
        """
        print("\n[CV1] Testing verb operator balance...")

        query = """
        MATCH (s:Concept)-[r:VIA]->(t:Concept)
        WHERE s.tau IS NOT NULL AND t.tau IS NOT NULL AND r.verb IS NOT NULL
        WITH r.verb as verb, avg(t.tau - s.tau) as avg_delta, count(*) as cnt
        WHERE cnt >= 30
        RETURN verb, avg_delta, cnt
        """

        records = self._query(query)

        falling = [r for r in records if r['avg_delta'] < -0.3]  # toward ground
        rising = [r for r in records if r['avg_delta'] > 0.3]    # toward sky
        neutral = [r for r in records if -0.3 <= r['avg_delta'] <= 0.3]

        # Balance metric: how close to 1.0?
        ratio = len(falling) / len(rising) if rising else float('inf')
        balance_score = 1.0 - abs(1.0 - ratio)  # 1.0 = perfect balance

        result = TestResult(
            test_id="CV1",
            name="Verb operators balanced",
            value=balance_score,
            expected="balance > 0.7 (roughly equal falling/rising)",
            passed=balance_score > 0.7,  # Within 30% of perfect balance
            details={
                "falling_count": len(falling),
                "rising_count": len(rising),
                "neutral_count": len(neutral),
                "falling_verbs": [(r['verb'], float(r['avg_delta'])) for r in falling[:5]],
                "rising_verbs": [(r['verb'], float(r['avg_delta'])) for r in rising[:5]]
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # CORRECTED ATTRACTOR TESTS
    # =========================================================================

    def test_attractors_at_ground(self) -> TestResult:
        """
        CA1: Attractors cluster at ground level (low τ).

        Random walks should settle to common ground.
        """
        print("\n[CA1] Testing attractor locations...")

        import random

        query = """
        MATCH (c:Concept)
        WHERE c.tau IS NOT NULL
        WITH c, COUNT { (c)-[:VIA]-() } as degree
        WHERE degree > 5
        RETURN c.word as word, c.tau as tau
        ORDER BY degree DESC
        LIMIT 100
        """

        starts = self._query(query)

        if len(starts) < 10:
            return TestResult("CA1", "Attractors at ground", 0, "avg τ < 3",
                            False, {"error": "Insufficient data"})

        end_taus = []
        for _ in range(200):
            current = random.choice(starts)
            word = current['word']

            for _ in range(15):
                neighbors = self._query("""
                    MATCH (c:Concept {word: $word})-[:VIA]->(n:Concept)
                    WHERE n.tau IS NOT NULL
                    RETURN n.word as word, n.tau as tau
                    LIMIT 20
                """, {"word": word})

                if not neighbors:
                    break

                current = random.choice(neighbors)
                word = current['word']

            end_taus.append(current['tau'])

        avg_end_tau = np.mean(end_taus)

        result = TestResult(
            test_id="CA1",
            name="Attractors at ground level",
            value=avg_end_tau,
            expected="avg τ < 3 (settle to ground)",
            passed=avg_end_tau < 3,
            details={
                "avg_end_tau": float(avg_end_tau),
                "end_tau_std": float(np.std(end_taus)),
                "n_walks": 200,
                "steps_per_walk": 15
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # CORRECTED POTENTIAL TESTS
    # =========================================================================

    def test_potential_minimum(self) -> TestResult:
        """
        CP1: Potential φ minimum at low τ (ground).

        CORRECTED formula: φ = +λτ - μg

        Physics reasoning:
        - Objects move toward LOWER potential
        - φ increases with τ (altitude) → gravity pulls toward low τ
        - φ decreases with g (goodness provides lift)
        """
        print("\n[CP1] Testing potential landscape...")

        query = """
        MATCH (c:Concept)
        WHERE c.tau IS NOT NULL AND c.g IS NOT NULL
        RETURN round(c.tau) as tau_bin, avg(c.g) as avg_g, count(*) as cnt
        ORDER BY tau_bin
        """

        records = self._query(query)

        if not records:
            return TestResult("CP1", "Potential minimum", 0, "φ min at low τ",
                            False, {"error": "Insufficient data"})

        # CORRECTED: φ = +λτ - μg (not -λτ + μg)
        # This ensures potential is HIGHER at high τ, so things "fall" to low τ
        phi_by_tau = {}
        for r in records:
            tau = r['tau_bin']
            g = r['avg_g'] or 0
            phi = self.LAMBDA * tau - self.MU * g  # CORRECTED SIGNS
            phi_by_tau[tau] = phi

        # Find minimum
        min_tau = min(phi_by_tau, key=phi_by_tau.get)
        min_phi = phi_by_tau[min_tau]

        result = TestResult(
            test_id="CP1",
            name="Potential minimum at ground",
            value=float(min_tau),
            expected="min τ ≤ 2 (minimum at ground)",
            passed=min_tau <= 2,
            details={
                "phi_by_tau": {f"τ={k}": round(v, 4) for k, v in phi_by_tau.items()},
                "min_tau": int(min_tau),
                "min_phi": float(min_phi)
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # RUN ALL
    # =========================================================================

    def run_all(self) -> Dict:
        """Run all corrected physics tests."""
        print("\n" + "=" * 70)
        print("CORRECTED SEMANTIC PHYSICS TESTS")
        print("=" * 70)
        print("Testing with corrected interpretation:")
        print("  Low τ = Ground (common)")
        print("  High τ = Sky (specific)")
        print("  Gravity → low τ")
        print()

        if not self.graph.is_connected():
            print("[ERROR] Neo4j not connected!")
            return {}

        # Run tests
        self.test_gravity_direction()
        self.test_ground_density()
        self.test_g_tau_positive()
        self.test_verb_asymmetry()
        self.test_attractors_at_ground()
        self.test_potential_minimum()

        # Summary
        self._print_summary()

        # Save
        self._save_results()

        return {"results": [r.__dict__ for r in self.results]}

    def _print_result(self, r: TestResult):
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_id}: {r.name}")
        print(f"         Value: {r.value:.4f}")
        print(f"         Expected: {r.expected}")

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            mark = "+" if r.passed else "x"
            print(f"  [{mark}] {r.test_id}: {r.value:.3f}")

        print(f"\n  TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"corrected_physics_{timestamp}.json"

        output = {
            "timestamp": timestamp,
            "model": "corrected",
            "interpretation": {
                "low_tau": "ground (common)",
                "high_tau": "sky (specific)",
                "gravity": "toward low τ"
            },
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed)
            },
            "results": [r.__dict__ for r in self.results]
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def close(self):
        if self.graph:
            self.graph.close()


def main():
    tests = CorrectedPhysicsTests()
    try:
        tests.run_all()
    finally:
        tests.close()


if __name__ == "__main__":
    main()
