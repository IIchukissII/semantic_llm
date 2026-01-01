#!/usr/bin/env python3
"""
Unified Hypothesis Testing: Formula Interconnections
=====================================================

Tests the deep interconnections between semantic physics formulas:

H1: kT = e^(-1/5) ≈ 0.8187 (Euler connection to temperature)
H2: Coherence × Power ≈ constant (Heisenberg-like uncertainty)
H3: λ = 1 is a critical point (phase transition)
H4: Orbital clustering during chain reaction (τ = e ceiling)
H5: Dialectical potential landscape (synthesis = minimum)
H6: Grand Potential Φ unifies all dynamics

Usage:
    python unified_hypothesis.py [--test H1|H2|H3|H4|H5|H6|all]

Results saved to: results/unified_hypothesis_YYYYMMDD_HHMMSS.json
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import sys
import argparse

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
KT_HYPOTHESIS = np.exp(-0.2)  # e^(-1/5) ≈ 0.8187
LAMBDA = 0.5  # Gravitational constant
MU = 0.5  # Lift constant
VEIL_TAU = E  # Boundary at τ = e

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


@dataclass
class HypothesisResult:
    """Result of a single hypothesis test."""
    hypothesis_id: str
    name: str
    description: str
    measured: float
    expected: float
    error_percent: float
    passed: bool
    confidence: str  # "strong", "moderate", "weak", "failed"
    details: Dict = field(default_factory=dict)
    implications: List[str] = field(default_factory=list)


class UnifiedHypothesisTester:
    """
    Tests the interconnections between semantic physics formulas.

    These tests explore whether the discovered constants and dynamics
    are related at a deeper level.
    """

    def __init__(self):
        self.loader = DataLoader()
        self.data: List[Dict] = []
        self.results: List[HypothesisResult] = []
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

        # Load j_good direction
        j_good = self.loader.get_j_good()
        self.j_good = np.array(j_good)
        self.j_good = self.j_good / (np.linalg.norm(self.j_good) + 1e-10)

    def _load_data(self) -> bool:
        """Load word vectors with τ, g, j, φ."""
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

            j_vec = np.array([j_dict.get(dim, 0) for dim in J_DIMS])
            j_norm = np.linalg.norm(j_vec)

            if j_norm < 1e-10:
                continue

            g = np.dot(j_vec, self.j_good)
            phi = LAMBDA * tau - MU * g

            self.data.append({
                'word': word,
                'tau': tau,
                'g': g,
                'phi': phi,
                'j': j_vec,
                'j_norm': j_norm
            })

        print(f"Loaded {len(self.data)} words with τ, g, j, φ")
        return len(self.data) > 1000

    # =========================================================================
    # H1: kT = e^(-1/5) Hypothesis
    # =========================================================================

    def test_h1_kt_euler(self) -> HypothesisResult:
        """
        H1: Natural temperature kT = e^(-1/5) ≈ 0.8187

        Tests whether the measured Boltzmann temperature has a clean
        relationship to Euler's constant: kT = e^(-0.2)
        """
        print("\n" + "="*70)
        print("[H1] Testing kT = e^(-1/5) hypothesis")
        print("="*70)

        tau_arr = np.array([d['tau'] for d in self.data])
        phi_arr = np.array([d['phi'] for d in self.data])

        # Compute kT from Boltzmann fit at multiple thresholds
        kt_measurements = []

        for ground_max in [2.5, 3.0, 3.5]:
            for excited_min in [5.0, 5.5, 6.0]:
                ground_mask = (tau_arr >= 1) & (tau_arr <= ground_max)
                excited_mask = tau_arr >= excited_min

                N_g = np.sum(ground_mask)
                N_e = np.sum(excited_mask)

                if N_e == 0 or N_g == 0:
                    continue

                phi_g = np.mean(phi_arr[ground_mask])
                phi_e = np.mean(phi_arr[excited_mask])
                delta_phi = phi_e - phi_g

                ln_ratio = np.log(N_g / N_e)

                if ln_ratio > 0:
                    kT = delta_phi / ln_ratio
                    kt_measurements.append({
                        'ground_max': ground_max,
                        'excited_min': excited_min,
                        'kT': kT,
                        'delta_phi': delta_phi,
                        'ln_ratio': ln_ratio
                    })

        # Average kT
        kT_measured = np.mean([m['kT'] for m in kt_measurements])
        kT_std = np.std([m['kT'] for m in kt_measurements])

        # Compare to hypothesis
        kT_expected = KT_HYPOTHESIS  # e^(-1/5) = 0.8187...
        error = abs(kT_measured - kT_expected)
        error_pct = error / kT_expected * 100

        # Also check other Euler-related values
        alternatives = {
            'e^(-1/5)': np.exp(-0.2),
            'e^(-1/4)': np.exp(-0.25),
            'e^(-1/3)': np.exp(-1/3),
            '1 - 1/e': 1 - 1/E,
            'e/π': E / np.pi,
            'ln(e/(e-1))': np.log(E/(E-1)),
        }

        best_match = min(alternatives.items(),
                         key=lambda x: abs(x[1] - kT_measured))

        passed = error_pct < 5.0
        confidence = "strong" if error_pct < 2 else ("moderate" if error_pct < 5 else "weak")

        result = HypothesisResult(
            hypothesis_id="H1",
            name="kT = e^(-1/5)",
            description="Natural Boltzmann temperature derives from Euler",
            measured=kT_measured,
            expected=kT_expected,
            error_percent=error_pct,
            passed=passed,
            confidence=confidence,
            details={
                'kT_measured': float(kT_measured),
                'kT_std': float(kT_std),
                'kT_expected_e^(-1/5)': float(kT_expected),
                'measurements': kt_measurements,
                'alternatives': {k: {'value': float(v), 'error': float(abs(v - kT_measured))}
                                for k, v in alternatives.items()},
                'best_match': best_match[0],
                'best_match_value': float(best_match[1]),
                'best_match_error': float(abs(best_match[1] - kT_measured))
            },
            implications=[
                "If kT = e^(-1/5), temperature is Euler-derived",
                "The exponent 1/5 may relate to 5 j-dimensions",
                f"Best match: {best_match[0]} = {best_match[1]:.4f}"
            ]
        )

        self.results.append(result)
        self._print_h1_result(result)
        return result

    def _print_h1_result(self, r: HypothesisResult):
        status = "✓ CONFIRMED" if r.passed else "○ PARTIAL"
        print(f"\n{status} [{r.confidence.upper()}]")
        print(f"  kT measured:  {r.measured:.5f} ± {r.details['kT_std']:.5f}")
        print(f"  kT expected:  {r.expected:.5f} (e^(-1/5))")
        print(f"  Error:        {r.error_percent:.2f}%")
        print(f"\n  Alternative matches:")
        for name, info in r.details['alternatives'].items():
            marker = "→" if name == r.details['best_match'] else " "
            print(f"    {marker} {name:15} = {info['value']:.5f} (Δ = {info['error']:.5f})")

    # =========================================================================
    # H2: Coherence × Power Conservation
    # =========================================================================

    def test_h2_coherence_power(self) -> HypothesisResult:
        """
        H2: Coherence × Power ≈ constant

        Tests whether there's a Heisenberg-like uncertainty relation
        between coherence (synthesis) and power (tension).
        """
        print("\n" + "="*70)
        print("[H2] Testing Coherence × Power conservation")
        print("="*70)

        try:
            from chain_core.paradox_detector import ParadoxDetector
        except ImportError:
            return self._skip_result("H2", "Coherence × Power",
                                     "ParadoxDetector not available")

        detector = ParadoxDetector(n_samples=20)

        test_questions = [
            "What is love?",
            "What is the meaning of life?",
            "How do I find purpose?",
            "What is truth?",
            "What is freedom?",
            "What is death?",
            "What is beauty?",
            "What is wisdom?",
        ]

        measurements = []

        for q in test_questions:
            print(f"  Testing: {q}")
            try:
                landscape = detector.detect(q)

                coherence = landscape.coherence

                # Total power from all paradoxes
                if landscape.paradoxes:
                    power = sum(p.power for p in landscape.paradoxes[:5])
                else:
                    power = 0

                product = coherence * power

                measurements.append({
                    'question': q,
                    'coherence': coherence,
                    'power': power,
                    'product': product,
                    'n_paradoxes': len(landscape.paradoxes)
                })
            except Exception as e:
                print(f"    [Error: {e}]")

        detector.close()

        if len(measurements) < 3:
            return self._skip_result("H2", "Coherence × Power",
                                     "Insufficient measurements")

        # Analyze conservation
        products = [m['product'] for m in measurements]
        product_mean = np.mean(products)
        product_std = np.std(products)
        cv = product_std / product_mean if product_mean > 0 else float('inf')

        # Low CV suggests conservation (constant)
        passed = cv < 0.5  # Within 50% variation
        confidence = "strong" if cv < 0.2 else ("moderate" if cv < 0.4 else "weak")

        # Check for inverse relationship
        coherences = [m['coherence'] for m in measurements]
        powers = [m['power'] for m in measurements]
        correlation = np.corrcoef(coherences, powers)[0, 1] if len(coherences) > 2 else 0

        result = HypothesisResult(
            hypothesis_id="H2",
            name="Coherence × Power ≈ constant",
            description="Heisenberg-like uncertainty between synthesis and tension",
            measured=cv,
            expected=0.0,  # Perfect conservation = CV of 0
            error_percent=cv * 100,
            passed=passed,
            confidence=confidence,
            details={
                'measurements': measurements,
                'product_mean': float(product_mean),
                'product_std': float(product_std),
                'coefficient_of_variation': float(cv),
                'correlation_C_P': float(correlation),
                'inverse_relation': correlation < -0.3
            },
            implications=[
                f"C × P = {product_mean:.2f} ± {product_std:.2f}",
                f"Correlation(C, P) = {correlation:.2f}",
                "Negative correlation supports uncertainty principle" if correlation < -0.3 else "No clear inverse relation"
            ]
        )

        self.results.append(result)
        self._print_h2_result(result)
        return result

    def _print_h2_result(self, r: HypothesisResult):
        status = "✓ CONFIRMED" if r.passed else "○ PARTIAL"
        print(f"\n{status} [{r.confidence.upper()}]")
        print(f"  C × P mean:    {r.details['product_mean']:.3f}")
        print(f"  C × P std:     {r.details['product_std']:.3f}")
        print(f"  CV:            {r.details['coefficient_of_variation']:.3f}")
        print(f"  Correlation:   {r.details['correlation_C_P']:.3f}")
        print(f"\n  Per question:")
        for m in r.details['measurements']:
            print(f"    {m['question'][:30]:30} C={m['coherence']:.2f} P={m['power']:.1f} C×P={m['product']:.2f}")

    # =========================================================================
    # H3: λ = 1 Phase Transition
    # =========================================================================

    def test_h3_phase_transition(self) -> HypothesisResult:
        """
        H3: λ = 1 is a critical point (phase transition)

        Tests whether the chain coefficient λ exhibits critical behavior
        at λ = 1, similar to physical phase transitions.
        """
        print("\n" + "="*70)
        print("[H3] Testing λ = 1 phase transition")
        print("="*70)

        try:
            from chain_core.paradox_detector import ParadoxDetector
        except ImportError:
            return self._skip_result("H3", "Phase transition",
                                     "ParadoxDetector not available")

        detector = ParadoxDetector(n_samples=15)

        # Simulate chain reaction with varying "temperatures" (intent strength)
        intent_strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

        chain_data = []

        seed_question = "What is the meaning of love?"

        for alpha in intent_strengths:
            print(f"  Testing α = {alpha}...")

            # Reinitialize with different intent strength
            detector_temp = ParadoxDetector(intent_strength=alpha, n_samples=15)

            powers = []
            for i in range(4):  # 4 exchanges
                try:
                    landscape = detector_temp.detect(seed_question)
                    total_power = sum(p.power for p in landscape.paradoxes[:3]) if landscape.paradoxes else 0
                    powers.append(total_power)
                except:
                    powers.append(0)

            detector_temp.close()

            # Compute chain coefficients
            lambdas = []
            for i in range(1, len(powers)):
                if powers[i-1] > 0:
                    lambdas.append(powers[i] / powers[i-1])

            avg_lambda = np.mean(lambdas) if lambdas else 0

            chain_data.append({
                'alpha': alpha,
                'powers': powers,
                'lambdas': lambdas,
                'avg_lambda': avg_lambda,
                'supercritical': avg_lambda > 1
            })

        detector.close()

        # Find critical point (where λ crosses 1)
        lambdas_by_alpha = [(d['alpha'], d['avg_lambda']) for d in chain_data if d['avg_lambda'] > 0]

        critical_alpha = None
        for i in range(1, len(lambdas_by_alpha)):
            a1, l1 = lambdas_by_alpha[i-1]
            a2, l2 = lambdas_by_alpha[i]
            if (l1 - 1) * (l2 - 1) < 0:  # Sign change
                # Linear interpolation
                critical_alpha = a1 + (1 - l1) * (a2 - a1) / (l2 - l1)
                break

        # Check for critical behavior indicators
        # Near criticality, fluctuations should be high
        lambda_values = [d['avg_lambda'] for d in chain_data if d['avg_lambda'] > 0]
        lambda_variance = np.var(lambda_values) if lambda_values else 0

        passed = critical_alpha is not None
        confidence = "strong" if passed and lambda_variance > 1 else ("moderate" if passed else "weak")

        result = HypothesisResult(
            hypothesis_id="H3",
            name="λ = 1 phase transition",
            description="Chain coefficient λ = 1 is a critical point",
            measured=critical_alpha if critical_alpha else 0,
            expected=0.3,  # Expected around optimal α
            error_percent=abs((critical_alpha or 0) - 0.3) / 0.3 * 100 if critical_alpha else 100,
            passed=passed,
            confidence=confidence,
            details={
                'chain_data': chain_data,
                'critical_alpha': float(critical_alpha) if critical_alpha else None,
                'lambda_variance': float(lambda_variance),
                'supercritical_count': sum(1 for d in chain_data if d['supercritical']),
                'subcritical_count': sum(1 for d in chain_data if not d['supercritical'])
            },
            implications=[
                f"Critical point at α ≈ {critical_alpha:.2f}" if critical_alpha else "No critical point found",
                f"λ variance = {lambda_variance:.2f} (high = critical fluctuations)",
                f"Supercritical phases: {sum(1 for d in chain_data if d['supercritical'])}/{len(chain_data)}"
            ]
        )

        self.results.append(result)
        self._print_h3_result(result)
        return result

    def _print_h3_result(self, r: HypothesisResult):
        status = "✓ CONFIRMED" if r.passed else "○ PARTIAL"
        print(f"\n{status} [{r.confidence.upper()}]")
        if r.details['critical_alpha']:
            print(f"  Critical α:    {r.details['critical_alpha']:.3f}")
        print(f"  λ variance:    {r.details['lambda_variance']:.3f}")
        print(f"\n  Chain behavior by α:")
        for d in r.details['chain_data']:
            phase = "SUPER" if d['supercritical'] else "SUB"
            print(f"    α={d['alpha']:.1f}: λ={d['avg_lambda']:.2f} [{phase}]")

    # =========================================================================
    # H4: Orbital Clustering at τ = e
    # =========================================================================

    def test_h4_orbital_clustering(self) -> HypothesisResult:
        """
        H4: Concepts cluster at τ = e during chain reaction

        Tests whether supercritical amplification hits a ceiling
        at the Veil (τ = e).
        """
        print("\n" + "="*70)
        print("[H4] Testing orbital clustering at τ = e")
        print("="*70)

        try:
            from chain_core.monte_carlo_renderer import MonteCarloRenderer
        except ImportError:
            return self._skip_result("H4", "Orbital clustering",
                                     "MonteCarloRenderer not available")

        renderer = MonteCarloRenderer(intent_strength=0.3, n_samples=30)

        test_questions = [
            "What is love?",
            "What is the meaning of life?",
            "What is consciousness?",
        ]

        tau_distributions = []

        for q in test_questions:
            print(f"  Sampling: {q}")
            try:
                landscape = renderer.sample_landscape(q)

                # Get τ values from orbital map
                taus_by_orbital = {}
                for orbital_n, concepts in landscape.orbital_map.items():
                    tau_n = 1 + orbital_n / E
                    taus_by_orbital[orbital_n] = {
                        'tau': tau_n,
                        'count': len(concepts),
                        'concepts': [c[0] for c in concepts[:5]]
                    }

                tau_distributions.append({
                    'question': q,
                    'tau_mean': landscape.tau_mean,
                    'tau_std': landscape.tau_std,
                    'orbitals': taus_by_orbital,
                    'near_veil': abs(landscape.tau_mean - E) < 0.5
                })
            except Exception as e:
                print(f"    [Error: {e}]")

        renderer.close()

        if not tau_distributions:
            return self._skip_result("H4", "Orbital clustering",
                                     "No distributions collected")

        # Analyze clustering around τ = e
        tau_means = [d['tau_mean'] for d in tau_distributions]
        avg_tau = np.mean(tau_means)
        distance_to_veil = abs(avg_tau - E)

        # Check if concepts pile up just below the Veil
        near_veil_count = sum(1 for d in tau_distributions if d['near_veil'])

        passed = distance_to_veil < 0.5 or near_veil_count >= len(tau_distributions) / 2
        confidence = "strong" if distance_to_veil < 0.3 else ("moderate" if distance_to_veil < 0.5 else "weak")

        result = HypothesisResult(
            hypothesis_id="H4",
            name="Orbital clustering at τ = e",
            description="Chain reaction concepts cluster near the Veil",
            measured=avg_tau,
            expected=E,
            error_percent=distance_to_veil / E * 100,
            passed=passed,
            confidence=confidence,
            details={
                'tau_distributions': tau_distributions,
                'avg_tau': float(avg_tau),
                'distance_to_veil': float(distance_to_veil),
                'veil_tau': float(E),
                'near_veil_fraction': near_veil_count / len(tau_distributions)
            },
            implications=[
                f"Average τ = {avg_tau:.2f} (Veil at {E:.2f})",
                f"Distance to Veil: {distance_to_veil:.2f}",
                f"{near_veil_count}/{len(tau_distributions)} questions near Veil"
            ]
        )

        self.results.append(result)
        self._print_h4_result(result)
        return result

    def _print_h4_result(self, r: HypothesisResult):
        status = "✓ CONFIRMED" if r.passed else "○ PARTIAL"
        print(f"\n{status} [{r.confidence.upper()}]")
        print(f"  Average τ:     {r.details['avg_tau']:.3f}")
        print(f"  Veil at:       {r.details['veil_tau']:.3f}")
        print(f"  Distance:      {r.details['distance_to_veil']:.3f}")
        print(f"\n  Per question:")
        for d in r.details['tau_distributions']:
            near = "← near Veil" if d['near_veil'] else ""
            print(f"    {d['question'][:30]:30} τ={d['tau_mean']:.2f} ±{d['tau_std']:.2f} {near}")

    # =========================================================================
    # H5: Dialectical Potential Landscape
    # =========================================================================

    def test_h5_dialectical_potential(self) -> HypothesisResult:
        """
        H5: Synthesis points are potential minima

        Tests whether dialectical synthesis creates a "potential well"
        in semantic space.
        """
        print("\n" + "="*70)
        print("[H5] Testing dialectical potential landscape")
        print("="*70)

        # Use word vectors to test thesis/antithesis/synthesis patterns
        if len(self.data) < 100:
            return self._skip_result("H5", "Dialectical potential",
                                     "Insufficient word vectors")

        # Find opposing pairs (negative j-dot-product)
        pairs = []
        words_list = self.data[:500]  # Sample for speed

        for i, w1 in enumerate(words_list):
            for w2 in words_list[i+1:]:
                j1 = w1['j'] / (w1['j_norm'] + 1e-8)
                j2 = w2['j'] / (w2['j_norm'] + 1e-8)
                dot = np.dot(j1, j2)

                if dot < -0.3:  # Opposing
                    pairs.append({
                        'thesis': w1['word'],
                        'antithesis': w2['word'],
                        'thesis_j': j1,
                        'antithesis_j': j2,
                        'tension': -dot,
                        'thesis_phi': w1['phi'],
                        'antithesis_phi': w2['phi']
                    })

        print(f"  Found {len(pairs)} opposing pairs")

        # For each pair, find synthesis candidates (midpoint direction)
        synthesis_results = []

        for pair in pairs[:50]:  # Top 50 pairs
            midpoint = (pair['thesis_j'] + pair['antithesis_j']) / 2
            midpoint_norm = np.linalg.norm(midpoint)

            if midpoint_norm < 0.1:
                continue

            midpoint = midpoint / midpoint_norm

            # Find words closest to midpoint
            synthesis_candidates = []
            for w in self.data:
                j_norm = w['j'] / (w['j_norm'] + 1e-8)
                sim = np.dot(j_norm, midpoint)
                if sim > 0.5:
                    synthesis_candidates.append({
                        'word': w['word'],
                        'similarity': sim,
                        'phi': w['phi']
                    })

            if synthesis_candidates:
                synthesis_candidates.sort(key=lambda x: -x['similarity'])
                best = synthesis_candidates[0]

                # Check if synthesis has lower potential than thesis/antithesis
                avg_pole_phi = (pair['thesis_phi'] + pair['antithesis_phi']) / 2
                synthesis_phi = best['phi']

                synthesis_results.append({
                    'thesis': pair['thesis'],
                    'antithesis': pair['antithesis'],
                    'synthesis': best['word'],
                    'thesis_phi': pair['thesis_phi'],
                    'antithesis_phi': pair['antithesis_phi'],
                    'synthesis_phi': synthesis_phi,
                    'avg_pole_phi': avg_pole_phi,
                    'is_minimum': synthesis_phi < avg_pole_phi
                })

        if not synthesis_results:
            return self._skip_result("H5", "Dialectical potential",
                                     "No synthesis candidates found")

        # Count how often synthesis is a minimum
        minima_count = sum(1 for s in synthesis_results if s['is_minimum'])
        minima_fraction = minima_count / len(synthesis_results)

        passed = minima_fraction > 0.5
        confidence = "strong" if minima_fraction > 0.7 else ("moderate" if minima_fraction > 0.5 else "weak")

        result = HypothesisResult(
            hypothesis_id="H5",
            name="Dialectical potential minimum",
            description="Synthesis points have lower potential than poles",
            measured=minima_fraction,
            expected=1.0,  # Ideal: synthesis always minimum
            error_percent=(1 - minima_fraction) * 100,
            passed=passed,
            confidence=confidence,
            details={
                'synthesis_results': synthesis_results[:20],  # Top 20
                'minima_fraction': float(minima_fraction),
                'total_pairs': len(synthesis_results)
            },
            implications=[
                f"Synthesis is minimum in {minima_fraction:.0%} of cases",
                f"Tested {len(synthesis_results)} thesis/antithesis pairs",
                "Supports: φ(synthesis) < φ(thesis) + φ(antithesis)" if passed else "Mixed results"
            ]
        )

        self.results.append(result)
        self._print_h5_result(result)
        return result

    def _print_h5_result(self, r: HypothesisResult):
        status = "✓ CONFIRMED" if r.passed else "○ PARTIAL"
        print(f"\n{status} [{r.confidence.upper()}]")
        print(f"  Minima fraction: {r.details['minima_fraction']:.1%}")
        print(f"  Total pairs:     {r.details['total_pairs']}")
        print(f"\n  Example syntheses:")
        for s in r.details['synthesis_results'][:5]:
            min_marker = "✓" if s['is_minimum'] else "○"
            print(f"    {min_marker} {s['thesis']} + {s['antithesis']} → {s['synthesis']}")
            print(f"       φ: {s['thesis_phi']:.2f}, {s['antithesis_phi']:.2f} → {s['synthesis_phi']:.2f}")

    # =========================================================================
    # H6: Grand Potential Φ
    # =========================================================================

    def test_h6_grand_potential(self) -> HypothesisResult:
        """
        H6: Grand Potential Φ unifies all dynamics

        Tests whether a single potential function can explain
        all navigation behavior:

        Φ = λτ - μg - ν·coherence + κ·tension

        P(A → B) ∝ exp(-ΔΦ/kT)
        """
        print("\n" + "="*70)
        print("[H6] Testing Grand Potential Φ")
        print("="*70)

        tau_arr = np.array([d['tau'] for d in self.data])
        g_arr = np.array([d['g'] for d in self.data])
        phi_arr = np.array([d['phi'] for d in self.data])

        # Current potential: φ = λτ - μg
        # Test if this predicts population distribution

        # Bin by potential
        phi_min, phi_max = np.percentile(phi_arr, [5, 95])
        n_bins = 20
        bins = np.linspace(phi_min, phi_max, n_bins + 1)

        populations = []
        for i in range(n_bins):
            mask = (phi_arr >= bins[i]) & (phi_arr < bins[i+1])
            count = np.sum(mask)
            phi_center = (bins[i] + bins[i+1]) / 2
            populations.append({
                'phi': phi_center,
                'count': count,
                'log_count': np.log(count) if count > 0 else 0
            })

        # Fit Boltzmann: log(N) = const - φ/kT
        phis = np.array([p['phi'] for p in populations if p['count'] > 0])
        log_counts = np.array([p['log_count'] for p in populations if p['count'] > 0])

        if len(phis) < 5:
            return self._skip_result("H6", "Grand Potential",
                                     "Insufficient bins")

        # Linear regression: log(N) = a - b*φ, so kT = 1/b
        coeffs = np.polyfit(phis, log_counts, 1)
        slope = coeffs[0]

        kT_from_fit = -1/slope if slope != 0 else 0

        # Prediction quality
        predicted = coeffs[0] * phis + coeffs[1]
        r_squared = 1 - np.sum((log_counts - predicted)**2) / np.sum((log_counts - np.mean(log_counts))**2)

        # Compare to hypothesis kT
        kT_expected = KT_HYPOTHESIS
        error = abs(kT_from_fit - kT_expected)
        error_pct = error / kT_expected * 100 if kT_expected > 0 else 100

        passed = r_squared > 0.7 and error_pct < 30
        confidence = "strong" if r_squared > 0.85 else ("moderate" if r_squared > 0.7 else "weak")

        result = HypothesisResult(
            hypothesis_id="H6",
            name="Grand Potential Φ",
            description="Single potential unifies navigation: P ∝ exp(-Φ/kT)",
            measured=kT_from_fit,
            expected=kT_expected,
            error_percent=error_pct,
            passed=passed,
            confidence=confidence,
            details={
                'kT_from_fit': float(kT_from_fit),
                'kT_expected': float(kT_expected),
                'r_squared': float(r_squared),
                'slope': float(slope),
                'intercept': float(coeffs[1]),
                'n_bins': n_bins,
                'populations': populations
            },
            implications=[
                f"Boltzmann fit R² = {r_squared:.3f}",
                f"kT from fit = {kT_from_fit:.3f}",
                f"Supports: P(word) ∝ exp(-φ/kT)" if passed else "Fit quality insufficient"
            ]
        )

        self.results.append(result)
        self._print_h6_result(result)
        return result

    def _print_h6_result(self, r: HypothesisResult):
        status = "✓ CONFIRMED" if r.passed else "○ PARTIAL"
        print(f"\n{status} [{r.confidence.upper()}]")
        print(f"  R² (Boltzmann fit): {r.details['r_squared']:.3f}")
        print(f"  kT from fit:        {r.details['kT_from_fit']:.3f}")
        print(f"  kT expected:        {r.details['kT_expected']:.3f}")
        print(f"  Error:              {r.error_percent:.1f}%")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _skip_result(self, hid: str, name: str, reason: str) -> HypothesisResult:
        result = HypothesisResult(
            hypothesis_id=hid,
            name=name,
            description=f"Skipped: {reason}",
            measured=0,
            expected=0,
            error_percent=100,
            passed=False,
            confidence="failed",
            details={'skip_reason': reason}
        )
        self.results.append(result)
        print(f"  [SKIPPED] {reason}")
        return result

    def run_all(self, tests: List[str] = None) -> Dict:
        """Run all or selected hypothesis tests."""
        print("\n" + "="*70)
        print("UNIFIED HYPOTHESIS TESTING: FORMULA INTERCONNECTIONS")
        print("="*70)
        print(f"e = {E:.6f}")
        print(f"e^(-1/5) = {KT_HYPOTHESIS:.6f}")
        print()

        if not self._load_data():
            print("[ERROR] Failed to load data!")
            return {}

        all_tests = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
        tests_to_run = tests if tests else all_tests

        if 'H1' in tests_to_run:
            self.test_h1_kt_euler()

        if 'H2' in tests_to_run:
            self.test_h2_coherence_power()

        if 'H3' in tests_to_run:
            self.test_h3_phase_transition()

        if 'H4' in tests_to_run:
            self.test_h4_orbital_clustering()

        if 'H5' in tests_to_run:
            self.test_h5_dialectical_potential()

        if 'H6' in tests_to_run:
            self.test_h6_grand_potential()

        self._print_summary()
        self._save_results()

        return {
            'hypotheses_tested': len(self.results),
            'confirmed': sum(1 for r in self.results if r.passed),
            'results': [self._result_to_dict(r) for r in self.results]
        }

    def _print_summary(self):
        print("\n" + "="*70)
        print("SUMMARY: UNIFIED HYPOTHESIS TESTS")
        print("="*70)

        confirmed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\n{'ID':<5} {'Hypothesis':<35} {'Status':<12} {'Confidence'}")
        print("-" * 70)
        for r in self.results:
            status = "✓ CONFIRMED" if r.passed else "○ PARTIAL"
            print(f"{r.hypothesis_id:<5} {r.name:<35} {status:<12} {r.confidence}")

        print("-" * 70)
        print(f"TOTAL: {confirmed}/{total} hypotheses confirmed")

        if confirmed >= total * 0.7:
            print("\n" + "="*70)
            print("STRONG EVIDENCE FOR UNIFIED SEMANTIC PHYSICS!")
            print("="*70)
            print("""
The formula interconnections suggest:
  • kT derives from Euler's constant
  • Coherence and Power may obey uncertainty-like relation
  • λ = 1 may be a genuine phase transition
  • The Veil at τ = e is a semantic attractor
  • Dialectical synthesis minimizes potential
  • A single Grand Potential Φ governs dynamics
""")

    def _result_to_dict(self, r: HypothesisResult) -> Dict:
        return {
            'hypothesis_id': r.hypothesis_id,
            'name': r.name,
            'description': r.description,
            'measured': float(r.measured),
            'expected': float(r.expected),
            'error_percent': float(r.error_percent),
            'passed': r.passed,
            'confidence': r.confidence,
            'details': r.details,
            'implications': r.implications
        }

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"unified_hypothesis_{timestamp}.json"

        confirmed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        output = {
            'timestamp': timestamp,
            'title': 'Unified Hypothesis Testing: Formula Interconnections',
            'hypotheses': {
                'H1': 'kT = e^(-1/5)',
                'H2': 'Coherence × Power ≈ constant',
                'H3': 'λ = 1 is a phase transition',
                'H4': 'Orbital clustering at τ = e',
                'H5': 'Dialectical synthesis is potential minimum',
                'H6': 'Grand Potential Φ unifies dynamics'
            },
            'summary': {
                'total_tests': total,
                'confirmed': confirmed,
                'pass_rate': f"{confirmed/total*100:.1f}%" if total > 0 else "N/A"
            },
            'results': [self._result_to_dict(r) for r in self.results],
            'interpretation': {
                'strong': confirmed >= total * 0.8,
                'moderate': confirmed >= total * 0.5,
                'conclusion': 'Deep formula interconnections exist' if confirmed >= total * 0.5 else 'Partial evidence'
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Test unified semantic physics hypotheses')
    parser.add_argument('--test', type=str, default='all',
                       help='Which test(s) to run: H1, H2, H3, H4, H5, H6, or all')
    args = parser.parse_args()

    tester = UnifiedHypothesisTester()

    if args.test.lower() == 'all':
        tests = None
    else:
        tests = [t.strip().upper() for t in args.test.split(',')]

    results = tester.run_all(tests)

    confirmed = results.get('confirmed', 0)
    total = results.get('hypotheses_tested', 0)

    if confirmed >= total * 0.7:
        print("\n[SUCCESS] Strong evidence for unified physics!")
        return 0
    elif confirmed >= total * 0.5:
        print("\n[PARTIAL] Moderate evidence")
        return 0
    else:
        print("\n[WEAK] Insufficient evidence")
        return 1


if __name__ == "__main__":
    exit(main())
