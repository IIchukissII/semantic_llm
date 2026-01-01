#!/usr/bin/env python3
"""
Experiment 2: PT1 Dynamics Validation
======================================

HYPOTHESIS:
    Semantic saturation follows PT1 (first-order lag) dynamics.
    V(t) = V_max × (1 - e^(-t/τ))

EXPERIMENT:
    1. Validate cascade fractions (63.21% rule)
    2. Test RLC circuit oscillator properties
    3. Compare resonance predictions

PREDICTION:
    - At t=τ: saturation = 63.21%
    - At t=3τ: saturation = 95.02%
    - At t=5τ: saturation = 99.33%

Usage:
    python exp2_pt1_dynamics.py

Results saved to: results/pt1_dynamics_YYYYMMDD_HHMMSS.json
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oscillators import (
    PT1Dynamics,
    RCCircuit,
    RLCCircuit,
    SemanticOscillator,
    compute_inductance,
    compute_capacitance,
    create_oscillator,
    E,
    KT_NATURAL,
)


@dataclass
class PT1ValidationResult:
    """Result of PT1 validation"""
    test_name: str
    expected: float
    measured: float
    error_percent: float
    passed: bool


@dataclass
class OscillatorTestResult:
    """Result of oscillator test"""
    concept: str
    tau: float
    variety: int
    degree: int
    omega_0: float
    Q_factor: float
    frequency_class: str
    damping_state: str


def test_pt1_saturation_fractions():
    """Test the 63.21% cascade rule"""
    print("\n--- PT1 Saturation Fractions ---")

    pt1 = PT1Dynamics(V_max=1.0, tau=1.0)

    # Expected fractions at each time constant
    expected = {
        1: 1 - 1/E,           # 63.21%
        2: 1 - 1/E**2,        # 86.47%
        3: 1 - 1/E**3,        # 95.02%
        4: 1 - 1/E**4,        # 98.17%
        5: 1 - 1/E**5,        # 99.33%
    }

    results = []
    for t, exp in expected.items():
        measured = pt1.saturation_fraction(t)
        error = abs(measured - exp) / exp * 100
        passed = error < 0.1  # < 0.1% error

        result = PT1ValidationResult(
            test_name=f"Saturation at t={t}τ",
            expected=exp,
            measured=measured,
            error_percent=error,
            passed=passed,
        )
        results.append(asdict(result))

        status = "✓" if passed else "✗"
        print(f"  t={t}τ: expected={exp:.4f}, measured={measured:.4f}, "
              f"error={error:.4f}% {status}")

    return results


def test_cascade_fractions():
    """Test cascade capture fractions"""
    print("\n--- Cascade Capture Fractions ---")

    pt1 = PT1Dynamics()
    fractions = pt1.cascade_fractions(5)

    # Each step captures 63.21% of remaining
    expected_incremental = [
        1 - 1/E,           # 63.21%
        (1/E) * (1 - 1/E), # 23.25%
        (1/E)**2 * (1 - 1/E), # 8.56%
        (1/E)**3 * (1 - 1/E), # 3.15%
        (1/E)**4 * (1 - 1/E), # 1.16%
    ]

    results = []
    for i, (cum_frac, inc_exp) in enumerate(zip(fractions, expected_incremental)):
        # Cumulative fraction
        exp_cum = 1 - (1/E)**(i+1)
        error = abs(cum_frac - exp_cum) / exp_cum * 100
        passed = error < 0.1

        result = PT1ValidationResult(
            test_name=f"Cascade step {i+1}",
            expected=exp_cum,
            measured=cum_frac,
            error_percent=error,
            passed=passed,
        )
        results.append(asdict(result))

        print(f"  Step {i+1}: cumulative={cum_frac:.4f} ({cum_frac*100:.1f}%), "
              f"incremental≈{inc_exp*100:.1f}%")

    return results


def test_rc_circuit():
    """Test RC circuit properties"""
    print("\n--- RC Circuit Properties ---")

    # R = τ (abstraction), C = connectivity
    rc = RCCircuit(R=2.0, C=0.5)

    results = []

    # Time constant
    tau_expected = 1.0  # R × C = 2.0 × 0.5
    tau_measured = rc.time_constant
    error = abs(tau_measured - tau_expected) / tau_expected * 100
    results.append(asdict(PT1ValidationResult(
        test_name="RC time constant",
        expected=tau_expected,
        measured=tau_measured,
        error_percent=error,
        passed=error < 0.1,
    )))
    print(f"  τ = RC = {tau_measured:.4f} (expected {tau_expected})")

    # Cutoff frequency
    omega_c_expected = 1.0  # 1/(RC)
    omega_c_measured = rc.cutoff_frequency()
    error = abs(omega_c_measured - omega_c_expected) / omega_c_expected * 100
    results.append(asdict(PT1ValidationResult(
        test_name="Cutoff frequency",
        expected=omega_c_expected,
        measured=omega_c_measured,
        error_percent=error,
        passed=error < 0.1,
    )))
    print(f"  ω_c = 1/τ = {omega_c_measured:.4f} (expected {omega_c_expected})")

    # Gain at cutoff (should be 1/√2 ≈ 0.707)
    gain_expected = 1 / np.sqrt(2)
    gain_measured = rc.gain_at(omega_c_measured)
    error = abs(gain_measured - gain_expected) / gain_expected * 100
    results.append(asdict(PT1ValidationResult(
        test_name="Gain at cutoff",
        expected=gain_expected,
        measured=gain_measured,
        error_percent=error,
        passed=error < 1.0,
    )))
    print(f"  |H(ω_c)| = {gain_measured:.4f} (expected {gain_expected:.4f})")

    return results


def test_rlc_resonance():
    """Test RLC circuit resonance"""
    print("\n--- RLC Circuit Resonance ---")

    # R = τ = 2.0, L = 0.5, C = 0.5
    rlc = RLCCircuit(R=0.5, L=0.5, C=0.5)

    results = []

    # Resonance frequency
    omega_0_expected = 1 / np.sqrt(0.5 * 0.5)  # = 2.0
    omega_0_measured = rlc.resonance_frequency
    error = abs(omega_0_measured - omega_0_expected) / omega_0_expected * 100
    results.append(asdict(PT1ValidationResult(
        test_name="Resonance frequency",
        expected=omega_0_expected,
        measured=omega_0_measured,
        error_percent=error,
        passed=error < 0.1,
    )))
    print(f"  ω₀ = 1/√(LC) = {omega_0_measured:.4f} (expected {omega_0_expected:.4f})")

    # Quality factor
    Q_expected = np.sqrt(0.5 / 0.5) / 0.5  # = 2.0
    Q_measured = rlc.quality_factor
    error = abs(Q_measured - Q_expected) / Q_expected * 100
    results.append(asdict(PT1ValidationResult(
        test_name="Quality factor",
        expected=Q_expected,
        measured=Q_measured,
        error_percent=error,
        passed=error < 0.1,
    )))
    print(f"  Q = √(L/C)/R = {Q_measured:.4f} (expected {Q_expected:.4f})")

    # Damping ratio
    zeta_expected = 0.5 / (2 * np.sqrt(0.5 / 0.5))  # = 0.25
    zeta_measured = rlc.damping_ratio
    error = abs(zeta_measured - zeta_expected) / zeta_expected * 100
    results.append(asdict(PT1ValidationResult(
        test_name="Damping ratio",
        expected=zeta_expected,
        measured=zeta_measured,
        error_percent=error,
        passed=error < 0.1,
    )))
    print(f"  ζ = R/(2√(L/C)) = {zeta_measured:.4f} (expected {zeta_expected:.4f})")

    # Check damping state
    damping_state = "underdamped" if rlc.is_underdamped else ("critically" if rlc.is_critically_damped else "overdamped")
    print(f"  Damping state: {damping_state} (ζ < 1: oscillates)")

    return results


def test_semantic_oscillators():
    """Test semantic oscillator properties"""
    print("\n--- Semantic Oscillator Properties ---")

    # Test concepts with different properties
    test_concepts = [
        {"name": "wisdom", "tau": 2.5, "variety": 80, "degree": 300},
        {"name": "love", "tau": 2.6, "variety": 90, "degree": 400},
        {"name": "table", "tau": 1.3, "variety": 20, "degree": 50},
        {"name": "god", "tau": 3.0, "variety": 95, "degree": 350},
    ]

    results = []
    for concept in test_concepts:
        osc = create_oscillator(
            concept=concept["name"],
            tau=concept["tau"],
            variety=concept["variety"],
            degree=concept["degree"],
        )

        # Create RLC equivalent
        rlc = RLCCircuit(R=osc.tau, L=osc.L, C=osc.C)

        result = OscillatorTestResult(
            concept=concept["name"],
            tau=concept["tau"],
            variety=concept["variety"],
            degree=concept["degree"],
            omega_0=osc.omega_0,
            Q_factor=osc.Q_factor,
            frequency_class=osc.frequency_class,
            damping_state="underdamped" if rlc.is_underdamped else "overdamped",
        )
        results.append(asdict(result))

        print(f"  {concept['name']}: τ={osc.tau:.1f}, ω₀={osc.omega_0:.2f}, "
              f"Q={osc.Q_factor:.2f}, class={osc.frequency_class}")

    return results


def test_euler_temperature():
    """Test Euler temperature law kT = e^(-1/5)"""
    print("\n--- Euler Temperature Law ---")

    # kT = e^(-1/5) from H1 hypothesis
    kT_expected = np.exp(-1/5)  # ≈ 0.8187
    kT_measured = KT_NATURAL

    error = abs(kT_measured - kT_expected) / kT_expected * 100

    result = PT1ValidationResult(
        test_name="kT = e^(-1/5)",
        expected=kT_expected,
        measured=kT_measured,
        error_percent=error,
        passed=error < 0.1,
    )

    print(f"  kT = e^(-1/5) = {kT_measured:.6f} (expected {kT_expected:.6f})")
    print(f"  Error: {error:.4f}%")

    return [asdict(result)]


def run_experiment():
    """Run all PT1 dynamics tests"""
    print("=" * 60)
    print("EXPERIMENT 2: PT1 Dynamics Validation")
    print("=" * 60)

    all_results = {
        "pt1_saturation": test_pt1_saturation_fractions(),
        "cascade_fractions": test_cascade_fractions(),
        "rc_circuit": test_rc_circuit(),
        "rlc_resonance": test_rlc_resonance(),
        "semantic_oscillators": test_semantic_oscillators(),
        "euler_temperature": test_euler_temperature(),
    }

    # Count passes
    total_tests = 0
    passed_tests = 0
    for category, results in all_results.items():
        for result in results:
            if "passed" in result:
                total_tests += 1
                if result["passed"]:
                    passed_tests += 1

    # Summary
    summary = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "status": "VALIDATED" if passed_tests == total_tests else "PARTIAL",
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Status: {summary['status']}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "PT1 Dynamics Validation",
        "results": all_results,
        "summary": summary,
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    filename = f"pt1_dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")

    return output


if __name__ == "__main__":
    run_experiment()
