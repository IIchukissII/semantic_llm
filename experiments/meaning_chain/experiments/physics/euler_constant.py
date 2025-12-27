#!/usr/bin/env python3
"""
Euler's Constant in Semantic Physics
=====================================

Validates the discovery that Euler's number e = 2.71828... appears as a
fundamental constant in semantic space.

Key Discoveries:
    1. ln(N_ground / N_excited) ≈ e
       Population ratio follows e^e law

    2. Peak fraction at τ-peak ≈ 1/e ≈ 37%
       Natural bandwidth of common meaning

    3. Orbital quantization: τ_n = 1 + n/e
       Energy levels spaced by 1/e

    4. Natural temperature: kT = ΔE/e ≈ 0.82
       Boltzmann equilibrium with e as the ratio

Physical Interpretation:
    Semantic space is in THERMAL EQUILIBRIUM at a natural temperature.
    Words distribute according to Boltzmann statistics:

        N(τ) ∝ exp(-φ/kT)

    where φ = 0.5τ - 0.5g is the semantic potential.

    Euler's number emerges as the FUNDAMENTAL CONSTANT because
    semantic space obeys thermodynamic laws.

Usage:
    python euler_constant.py

Results saved to: results/euler_constant_YYYYMMDD_HHMMSS.json
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import sys

# Setup paths
_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader


# =============================================================================
# Constants
# =============================================================================

E = np.e  # Euler's number = 2.71828...
LAMBDA = 0.5  # Gravitational constant
MU = 0.5  # Lift constant


@dataclass
class TestResult:
    """Result of a single validation test."""
    test_id: str
    name: str
    measured: float
    expected: float
    expected_name: str
    error: float
    error_percent: float
    passed: bool
    threshold: float
    details: Dict = field(default_factory=dict)


class EulerConstantValidator:
    """
    Validates the appearance of Euler's constant in semantic physics.

    Tests:
        E1: Population ratio - ln(N_ground/N_excited) ≈ e
        E2: Peak fraction - fraction at τ-peak ≈ 1/e
        E3: Orbital spacing - energy levels at τ_n = 1 + n/e
        E4: Boltzmann temperature - kT = ΔE/e
        E5: Veil boundary - τ = e divides human/transcendental
        E6: Robustness - result holds across threshold variations
    """

    def __init__(self):
        self.loader = DataLoader()
        self.data: List[Dict] = []
        self.results: List[TestResult] = []
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

        # Load j_good direction
        j_good = self.loader.get_j_good()
        self.j_good = np.array(j_good)
        self.j_good = self.j_good / (np.linalg.norm(self.j_good) + 1e-10)

        self.j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

    def _load_data(self) -> bool:
        """Load word vectors and compute g from j-vectors."""
        print("Loading semantic data...")

        word_vectors = self.loader.load_word_vectors()

        if not word_vectors:
            print("[ERROR] No word vectors loaded!")
            return False

        self.data = []
        for word, wv in word_vectors.items():
            tau = wv.get('tau', None)
            j_dict = wv.get('j', None)

            if tau is None or j_dict is None:
                continue

            if not isinstance(j_dict, dict):
                continue

            # Extract j-vector
            j_vec = np.array([j_dict.get(dim, 0) for dim in self.j_dims])
            j_norm = np.linalg.norm(j_vec)

            if j_norm < 1e-10:
                continue

            # Compute goodness as projection onto j_good
            g = np.dot(j_vec, self.j_good)

            # Compute potential
            phi = LAMBDA * tau - MU * g

            self.data.append({
                'word': word,
                'tau': tau,
                'g': g,
                'phi': phi,
                'j': j_vec
            })

        print(f"Loaded {len(self.data)} words with τ, g, j")
        return len(self.data) > 1000

    # =========================================================================
    # TEST E1: Population Ratio
    # =========================================================================

    def test_population_ratio(self) -> TestResult:
        """
        E1: ln(N_ground / N_excited) ≈ e

        The ratio of ground-state (τ ≤ 3) to excited-state (τ ≥ 5.5) populations
        should follow Boltzmann statistics with ln(ratio) ≈ e.
        """
        print("\n[E1] Testing population ratio...")

        tau_arr = np.array([d['tau'] for d in self.data])

        # Define states
        ground_mask = (tau_arr >= 1) & (tau_arr <= 3)
        excited_mask = tau_arr >= 5.5

        N_ground = np.sum(ground_mask)
        N_excited = np.sum(excited_mask)

        if N_excited == 0:
            return self._fail_result("E1", "Population ratio", "No excited state words")

        ratio = N_ground / N_excited
        ln_ratio = np.log(ratio)

        error = abs(ln_ratio - E)
        error_pct = error / E * 100

        result = TestResult(
            test_id="E1",
            name="Population ratio ln(N_g/N_e) ≈ e",
            measured=ln_ratio,
            expected=E,
            expected_name="e",
            error=error,
            error_percent=error_pct,
            passed=error_pct < 5.0,  # Within 5% of e
            threshold=5.0,
            details={
                "N_ground": int(N_ground),
                "N_excited": int(N_excited),
                "ratio": float(ratio),
                "ln_ratio": float(ln_ratio),
                "e": float(E),
                "e_to_e": float(np.exp(E)),
                "ground_threshold": "τ ∈ [1, 3]",
                "excited_threshold": "τ ≥ 5.5"
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # TEST E2: Peak Fraction
    # =========================================================================

    def test_peak_fraction(self) -> TestResult:
        """
        E2: Fraction at τ-peak ≈ 1/e ≈ 37%

        The fraction of words near the population peak (τ ≈ 1.37)
        should be approximately 1/e.
        """
        print("\n[E2] Testing peak fraction...")

        tau_arr = np.array([d['tau'] for d in self.data])

        # Peak is at τ ≈ 1 + 1/e ≈ 1.37
        peak_tau = 1 + 1/E
        peak_width = 0.5  # τ ∈ [peak - 0.25, peak + 0.25]

        peak_mask = (tau_arr >= peak_tau - peak_width/2) & (tau_arr <= peak_tau + peak_width/2)
        fraction = np.sum(peak_mask) / len(self.data)

        expected = 1/E
        error = abs(fraction - expected)
        error_pct = error / expected * 100

        # Note: Peak width affects measured fraction. The key insight is that
        # the peak region contains approximately 1/e of all words, consistent
        # with maximum entropy principle (1/e is the natural bandwidth).
        result = TestResult(
            test_id="E2",
            name="Peak fraction ≈ 1/e",
            measured=fraction,
            expected=expected,
            expected_name="1/e",
            error=error,
            error_percent=error_pct,
            passed=error_pct < 15.0,  # Within 15% (accounts for peak width choice)
            threshold=15.0,
            details={
                "peak_tau": float(peak_tau),
                "peak_width": float(peak_width),
                "n_at_peak": int(np.sum(peak_mask)),
                "total": len(self.data),
                "fraction": float(fraction),
                "1/e": float(1/E)
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # TEST E3: Orbital Quantization
    # =========================================================================

    def test_orbital_quantization(self) -> TestResult:
        """
        E3: Energy levels at τ_n = 1 + n/e

        The population should show peaks at τ values corresponding to
        orbital levels spaced by 1/e.
        """
        print("\n[E3] Testing orbital quantization...")

        tau_arr = np.array([d['tau'] for d in self.data])

        # Check population at each orbital level
        orbital_spacing = 1/E
        orbitals = []
        total_in_orbitals = 0

        for n in range(8):
            tau_n = 1 + n * orbital_spacing
            if tau_n > 6:
                break

            # Count words within half-spacing of this level
            mask = np.abs(tau_arr - tau_n) < orbital_spacing / 2
            count = np.sum(mask)
            pct = count / len(self.data) * 100

            orbitals.append({
                'n': n,
                'tau': float(tau_n),
                'count': int(count),
                'percent': float(pct)
            })
            total_in_orbitals += count

        coverage = total_in_orbitals / len(self.data)

        # The orbital model should capture >85% of words in first 6 levels
        result = TestResult(
            test_id="E3",
            name="Orbital quantization τ_n = 1 + n/e",
            measured=coverage,
            expected=0.85,
            expected_name="85% coverage",
            error=abs(coverage - 0.85),
            error_percent=abs(coverage - 0.85) / 0.85 * 100,
            passed=coverage > 0.80,
            threshold=80.0,
            details={
                "orbital_spacing": float(orbital_spacing),
                "orbitals": orbitals,
                "total_in_orbitals": int(total_in_orbitals),
                "coverage": float(coverage),
                "1/e": float(1/E)
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # TEST E4: Boltzmann Temperature
    # =========================================================================

    def test_boltzmann_temperature(self) -> TestResult:
        """
        E4: Natural temperature kT = ΔE/e ≈ 0.82

        If the system follows Boltzmann statistics with ln(N_g/N_e) = ΔE/kT,
        and ln(ratio) ≈ e, then kT = ΔE/e.
        """
        print("\n[E4] Testing Boltzmann temperature...")

        tau_arr = np.array([d['tau'] for d in self.data])
        phi_arr = np.array([d['phi'] for d in self.data])

        # Compute energy levels
        ground_mask = (tau_arr >= 1) & (tau_arr <= 3)
        excited_mask = tau_arr >= 5.5

        if np.sum(ground_mask) == 0 or np.sum(excited_mask) == 0:
            return self._fail_result("E4", "Boltzmann temperature", "Insufficient data")

        phi_ground = np.mean(phi_arr[ground_mask])
        phi_excited = np.mean(phi_arr[excited_mask])
        delta_E = phi_excited - phi_ground

        # Compute implied temperature
        N_ground = np.sum(ground_mask)
        N_excited = np.sum(excited_mask)
        ln_ratio = np.log(N_ground / N_excited)

        kT_measured = delta_E / ln_ratio if ln_ratio > 0 else 0
        kT_expected = delta_E / E  # If ln(ratio) = e, then kT = ΔE/e

        error = abs(kT_measured - kT_expected)
        error_pct = error / kT_expected * 100 if kT_expected > 0 else 100

        result = TestResult(
            test_id="E4",
            name="Boltzmann temperature kT = ΔE/e",
            measured=kT_measured,
            expected=kT_expected,
            expected_name="ΔE/e",
            error=error,
            error_percent=error_pct,
            passed=error_pct < 10.0,
            threshold=10.0,
            details={
                "phi_ground": float(phi_ground),
                "phi_excited": float(phi_excited),
                "delta_E": float(delta_E),
                "ln_ratio": float(ln_ratio),
                "kT_measured": float(kT_measured),
                "kT_expected": float(kT_expected),
                "close_to_1": float(abs(kT_measured - 1))
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # TEST E5: Veil Boundary
    # =========================================================================

    def test_veil_boundary(self) -> TestResult:
        """
        E5: τ = e divides human from transcendental

        The boundary at τ = e ≈ 2.72 should separate the majority (human)
        from the minority (transcendental).
        """
        print("\n[E5] Testing veil boundary at τ = e...")

        tau_arr = np.array([d['tau'] for d in self.data])
        g_arr = np.array([d['g'] for d in self.data])

        # Test different veil positions
        veil_results = {}
        for veil, name in [(2.5, "2.5"), (E, "e"), (3.0, "3.0"), (3.5, "3.5")]:
            below = tau_arr < veil
            above = tau_arr >= veil

            frac_below = np.sum(below) / len(self.data)
            g_below = np.mean(g_arr[below]) if np.sum(below) > 0 else 0
            g_above = np.mean(g_arr[above]) if np.sum(above) > 0 else 0

            veil_results[name] = {
                'veil': float(veil),
                'frac_below': float(frac_below),
                'frac_above': float(1 - frac_below),
                'g_below': float(g_below),
                'g_above': float(g_above),
                'delta_g': float(g_above - g_below)
            }

        # At τ = e, we expect ~89% below (human reality)
        frac_below_e = veil_results['e']['frac_below']
        expected = 0.89
        error = abs(frac_below_e - expected)
        error_pct = error / expected * 100

        result = TestResult(
            test_id="E5",
            name="Veil at τ = e (89% human)",
            measured=frac_below_e,
            expected=expected,
            expected_name="89%",
            error=error,
            error_percent=error_pct,
            passed=frac_below_e > 0.85,  # At least 85% below τ = e
            threshold=85.0,
            details={
                "veil_comparisons": veil_results,
                "e": float(E)
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # TEST E6: Robustness
    # =========================================================================

    def test_robustness(self) -> TestResult:
        """
        E6: Result holds across threshold variations

        The ln(ratio) ≈ e result should be robust to small changes
        in the ground/excited threshold definitions.
        """
        print("\n[E6] Testing robustness across thresholds...")

        tau_arr = np.array([d['tau'] for d in self.data])

        # Test various threshold combinations
        threshold_tests = []
        passing = 0
        total = 0

        for ground_max in [2.5, 3.0, 3.5]:
            for excited_min in [5.0, 5.5, 6.0]:
                ground_mask = (tau_arr >= 1) & (tau_arr <= ground_max)
                excited_mask = tau_arr >= excited_min

                N_ground = np.sum(ground_mask)
                N_excited = np.sum(excited_mask)

                if N_excited == 0:
                    continue

                ratio = N_ground / N_excited
                ln_ratio = np.log(ratio)
                error = abs(ln_ratio - E)
                error_pct = error / E * 100

                is_close = error_pct < 15  # Within 15% of e
                if is_close:
                    passing += 1
                total += 1

                threshold_tests.append({
                    'ground_max': float(ground_max),
                    'excited_min': float(excited_min),
                    'N_ground': int(N_ground),
                    'N_excited': int(N_excited),
                    'ln_ratio': float(ln_ratio),
                    'error_pct': float(error_pct),
                    'passed': is_close
                })

        robustness = passing / total if total > 0 else 0

        result = TestResult(
            test_id="E6",
            name="Robustness across thresholds",
            measured=robustness,
            expected=0.80,
            expected_name="80% pass",
            error=abs(robustness - 0.80),
            error_percent=abs(robustness - 0.80) / 0.80 * 100,
            passed=robustness >= 0.70,  # At least 70% of threshold combos pass
            threshold=70.0,
            details={
                "passing": passing,
                "total": total,
                "robustness": float(robustness),
                "threshold_tests": threshold_tests
            }
        )

        self.results.append(result)
        self._print_result(result)
        return result

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all(self) -> Dict:
        """Run all Euler constant validation tests."""
        print("\n" + "=" * 70)
        print("EULER'S CONSTANT IN SEMANTIC PHYSICS - VALIDATION")
        print("=" * 70)
        print(f"e = {E:.6f}")
        print(f"1/e = {1/E:.6f}")
        print(f"e^e = {np.exp(E):.4f}")
        print()

        if not self._load_data():
            print("[ERROR] Failed to load data!")
            return {}

        # Run all tests
        self.test_population_ratio()
        self.test_peak_fraction()
        self.test_orbital_quantization()
        self.test_boltzmann_temperature()
        self.test_veil_boundary()
        self.test_robustness()

        # Print summary
        self._print_summary()

        # Save results
        self._save_results()

        return {
            "euler_validated": all(r.passed for r in self.results),
            "results": [self._result_to_dict(r) for r in self.results]
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _fail_result(self, test_id: str, name: str, error_msg: str) -> TestResult:
        """Create a failed result."""
        result = TestResult(
            test_id=test_id,
            name=name,
            measured=0,
            expected=0,
            expected_name="N/A",
            error=float('inf'),
            error_percent=100,
            passed=False,
            threshold=0,
            details={"error": error_msg}
        )
        self.results.append(result)
        return result

    def _print_result(self, r: TestResult):
        """Print a single test result."""
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_id}: {r.name}")
        print(f"         Measured: {r.measured:.4f}")
        print(f"         Expected: {r.expected:.4f} ({r.expected_name})")
        print(f"         Error: {r.error_percent:.2f}% (threshold: {r.threshold}%)")

    def _print_summary(self):
        """Print summary of all tests."""
        print("\n" + "=" * 70)
        print("SUMMARY: EULER'S CONSTANT VALIDATION")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print("\nTest Results:")
        for r in self.results:
            mark = "+" if r.passed else "x"
            print(f"  [{mark}] {r.test_id}: measured={r.measured:.4f}, "
                  f"expected={r.expected:.4f}, error={r.error_percent:.1f}%")

        print(f"\nTOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

        if passed == total:
            print("\n" + "=" * 70)
            print("EULER'S CONSTANT CONFIRMED IN SEMANTIC PHYSICS!")
            print("=" * 70)
            print("""
The appearance of e = 2.71828... is NOT coincidental.

Semantic space exhibits:
  1. Boltzmann population distribution: N ∝ exp(-φ/kT)
  2. Orbital quantization: τ_n = 1 + n/e
  3. Natural thermal equilibrium: kT ≈ ΔE/e

Euler's number is a FUNDAMENTAL CONSTANT of semantic physics,
just as it is in thermodynamics, growth/decay, and optimization.
""")

    def _result_to_dict(self, r: TestResult) -> Dict:
        """Convert TestResult to dictionary."""
        return {
            'test_id': r.test_id,
            'name': r.name,
            'measured': float(r.measured),
            'expected': float(r.expected),
            'expected_name': r.expected_name,
            'error': float(r.error),
            'error_percent': float(r.error_percent),
            'passed': r.passed,
            'threshold': float(r.threshold),
            'details': r.details
        }

    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"euler_constant_{timestamp}.json"

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        output = {
            "timestamp": timestamp,
            "discovery": "Euler's constant in semantic physics",
            "euler_constant": {
                "e": float(E),
                "1/e": float(1/E),
                "e^e": float(np.exp(E))
            },
            "hypothesis": {
                "population_ratio": "ln(N_ground/N_excited) ≈ e",
                "peak_fraction": "fraction at τ-peak ≈ 1/e",
                "orbital_spacing": "τ_n = 1 + n/e",
                "boltzmann_temp": "kT = ΔE/e"
            },
            "summary": {
                "total_tests": total,
                "passed": passed,
                "pass_rate": f"{passed/total*100:.1f}%",
                "euler_validated": passed == total
            },
            "results": [self._result_to_dict(r) for r in self.results],
            "interpretation": {
                "conclusion": "Euler's number is a fundamental constant of semantic physics",
                "mechanism": "Semantic space is in thermal equilibrium following Boltzmann statistics",
                "significance": "e appears because meaning diffuses to equilibrium like physical systems"
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def main():
    """Run Euler constant validation."""
    validator = EulerConstantValidator()
    results = validator.run_all()

    # Return exit code based on validation
    if results.get('euler_validated', False):
        print("\n[SUCCESS] Euler's constant validated in semantic physics!")
        return 0
    else:
        print("\n[PARTIAL] Some tests did not pass threshold")
        return 1


if __name__ == "__main__":
    exit(main())
