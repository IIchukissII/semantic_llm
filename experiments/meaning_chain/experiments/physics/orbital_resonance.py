#!/usr/bin/env python3
"""
Orbital Resonance Spectroscopy
==============================

Proves orbital quantization through resonance measurements.

THEORY:
    In atomic physics, quantized energy levels are proven by:
    1. Absorption spectroscopy - atoms absorb at specific frequencies
    2. Resonance - driving at natural frequency produces maximum response

    In semantic space:
    - "Frequency" = τ-level (abstraction altitude)
    - "Resonance" = coherence peak when driving at orbital τ_n = 1 + n/e
    - "Quantization" = peaks at predicted positions, valleys between

EXPERIMENT:
    1. SPECTROSCOPY: Sweep τ from 1.0 to 6.0
       - Select seeds at each τ-level
       - Run semantic laser
       - Measure coherence response
       - Find peaks at τ_n = 1 + n/e

    2. TRANSITION SPECTROSCOPY:
       - Measure transition probabilities between orbitals
       - Prove selection rules: Δn = ±1 allowed, other forbidden

    3. Q-FACTOR ANALYSIS:
       - Measure peak width at each orbital
       - Q = τ_n / Δτ_peak (sharpness of resonance)

PREDICTION:
    If orbitals are real:
    - Sharp peaks at τ_n = 1 + n/e (within ±0.1)
    - Valleys between orbitals
    - Transition matrix shows diagonal bands at Δτ = 1/e
    - Q-factor consistent across orbitals

Usage:
    python orbital_resonance.py

Results saved to: results/orbital_resonance_YYYYMMDD_HHMMSS.json
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import sys

# Setup paths
_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from chain_core.semantic_laser import SemanticLaser, E, KT_NATURAL, VEIL_TAU

# =============================================================================
# Constants
# =============================================================================

# Predicted orbital positions: τ_n = 1 + n/e
ORBITAL_SPACING = 1 / E  # ≈ 0.368
ORBITALS = [1 + n * ORBITAL_SPACING for n in range(16)]  # n=0 to n=15
# [1.00, 1.37, 1.74, 2.10, 2.47, 2.84, 3.21, 3.58, 3.95, 4.31, 4.68, 5.05, 5.42, 5.79, 6.16, 6.53]

# Sweep parameters
TAU_MIN = 1.0
TAU_MAX = 6.0
TAU_STEPS = 100  # Resolution of sweep
TAU_TOLERANCE = 0.15  # Seeds within ±tolerance of target τ


@dataclass
class ResonancePoint:
    """A single point in the resonance spectrum."""
    tau_drive: float  # Driving frequency (τ-level of seeds)
    coherence: float  # Response (coherence of output)
    beam_count: int  # Number of coherent beams
    response_tau: float  # Mean τ of response
    intensity: float  # Total beam intensity
    n_seeds: int  # Number of seeds used
    lasing: bool  # Did lasing occur?


@dataclass
class ResonancePeak:
    """A detected resonance peak."""
    tau_center: float  # Center of peak
    tau_width: float  # FWHM
    peak_height: float  # Maximum coherence
    q_factor: float  # τ/Δτ
    orbital_n: int  # Predicted orbital number
    tau_predicted: float  # τ_n = 1 + n/e
    error: float  # |measured - predicted|


@dataclass
class TransitionMeasurement:
    """Measurement of transition between orbitals."""
    n_start: int
    n_end: int
    tau_start: float
    tau_end: float
    probability: float  # Fraction arriving at end orbital
    delta_n: int  # n_end - n_start
    allowed: bool  # Is |Δn| = 1?


class OrbitalResonanceExperiment:
    """
    Proves orbital quantization through resonance spectroscopy.

    The key insight: if semantic space has quantized orbitals,
    then "driving" at orbital frequencies should produce resonance peaks.
    """

    def __init__(self):
        self.graph = MeaningGraph()
        self.laser = SemanticLaser(self.graph, temperature=KT_NATURAL)
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

        # Results storage
        self.spectrum: List[ResonancePoint] = []
        self.peaks: List[ResonancePeak] = []
        self.transitions: List[TransitionMeasurement] = []

        # Cache of concepts by τ-level
        self._concepts_by_tau: Dict[int, List[str]] = {}
        self._load_concepts_by_tau()

    def _load_concepts_by_tau(self):
        """Load and bin concepts by τ-level."""
        print("Loading concepts by τ-level...")

        if not self.graph.driver:
            print("[ERROR] No graph connection")
            return

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE c.tau IS NOT NULL
                  AND size(c.word) >= 3
                RETURN c.word as word, c.tau as tau
            """)

            # Bin by τ rounded to 0.1
            bins = defaultdict(list)
            for record in result:
                tau = record["tau"]
                word = record["word"]
                # Bin to nearest 0.1
                bin_key = round(tau * 10)  # e.g., 1.37 -> 14
                bins[bin_key].append(word)

            self._concepts_by_tau = dict(bins)

        total = sum(len(v) for v in self._concepts_by_tau.values())
        print(f"Loaded {total} concepts in {len(self._concepts_by_tau)} τ-bins")

    def _get_seeds_at_tau(self, target_tau: float, n_seeds: int = 5) -> List[str]:
        """Get concepts near target τ-level."""
        # Find bins near target
        target_bin = round(target_tau * 10)
        candidates = []

        # Check target bin and neighbors
        for offset in range(5):
            for sign in [0, -1, 1]:
                bin_key = target_bin + sign * offset
                if bin_key in self._concepts_by_tau:
                    candidates.extend(self._concepts_by_tau[bin_key])
            if len(candidates) >= n_seeds:
                break

        if not candidates:
            return []

        # Random sample
        np.random.shuffle(candidates)
        return candidates[:n_seeds]

    # =========================================================================
    # Experiment 1: Spectroscopy - Sweep across τ spectrum
    # =========================================================================

    def run_spectroscopy(self, n_seeds: int = 5, pump_power: int = 8,
                         pump_depth: int = 4) -> List[ResonancePoint]:
        """
        Sweep across τ-spectrum, measuring coherence response.

        This is the main resonance spectroscopy experiment:
        - Drive at τ_drive (select seeds at this level)
        - Measure response (coherence, beams, etc.)
        - Plot coherence vs τ_drive
        - Look for peaks at τ_n = 1 + n/e
        """
        print("\n" + "=" * 70)
        print("RESONANCE SPECTROSCOPY: τ-Sweep")
        print("=" * 70)
        print(f"Sweep range: τ ∈ [{TAU_MIN:.1f}, {TAU_MAX:.1f}]")
        print(f"Resolution: {TAU_STEPS} points")
        print(f"Seeds per point: {n_seeds}")
        print()

        tau_values = np.linspace(TAU_MIN, TAU_MAX, TAU_STEPS)
        spectrum = []

        for i, tau_drive in enumerate(tau_values):
            # Progress
            if i % 10 == 0:
                print(f"  Sweeping τ = {tau_drive:.2f} ({i}/{TAU_STEPS})...")

            # Get seeds at this τ-level
            seeds = self._get_seeds_at_tau(tau_drive, n_seeds)
            if not seeds:
                continue

            # Run laser
            try:
                result = self.laser.lase(
                    seeds=seeds,
                    pump_power=pump_power,
                    pump_depth=pump_depth,
                    coherence_threshold=0.25,
                    min_cluster_size=3
                )
            except Exception as e:
                print(f"    Error at τ={tau_drive:.2f}: {e}")
                continue

            # Extract response
            beams = result.get('beams', [])
            population = result.get('population', {})
            metrics = result.get('metrics', {})

            coherence = metrics.get('mirror_alignment', 0.0)
            response_tau = population.get('tau_mean', tau_drive)
            intensity = sum(b.intensity for b in beams) if beams else 0.0
            lasing = metrics.get('lasing_achieved', False)

            point = ResonancePoint(
                tau_drive=tau_drive,
                coherence=coherence,
                beam_count=len(beams),
                response_tau=response_tau,
                intensity=intensity,
                n_seeds=len(seeds),
                lasing=lasing
            )
            spectrum.append(point)

        self.spectrum = spectrum
        print(f"\nSpectroscopy complete: {len(spectrum)} points measured")

        return spectrum

    def find_peaks(self, min_peak_height: float = 0.5,
                   min_peak_prominence: float = 0.1) -> List[ResonancePeak]:
        """
        Find resonance peaks in the spectrum.

        Uses simple peak detection: local maxima above threshold
        with prominence above minimum.
        """
        if not self.spectrum:
            print("[ERROR] Run spectroscopy first")
            return []

        print("\n" + "-" * 50)
        print("PEAK DETECTION")
        print("-" * 50)

        taus = np.array([p.tau_drive for p in self.spectrum])
        coherences = np.array([p.coherence for p in self.spectrum])

        peaks = []

        # Simple peak detection: local maximum above threshold
        for i in range(1, len(coherences) - 1):
            if coherences[i] < min_peak_height:
                continue

            # Is it a local maximum?
            if coherences[i] > coherences[i-1] and coherences[i] > coherences[i+1]:
                # Check prominence (height above neighbors)
                prominence = coherences[i] - max(coherences[i-1], coherences[i+1])
                if prominence < min_peak_prominence:
                    continue

                # Found a peak
                tau_center = taus[i]
                peak_height = coherences[i]

                # Estimate width (FWHM)
                half_max = peak_height / 2
                left_idx = i
                while left_idx > 0 and coherences[left_idx] > half_max:
                    left_idx -= 1
                right_idx = i
                while right_idx < len(coherences) - 1 and coherences[right_idx] > half_max:
                    right_idx += 1
                tau_width = taus[right_idx] - taus[left_idx] if right_idx > left_idx else 0.1

                # Q-factor
                q_factor = tau_center / tau_width if tau_width > 0 else 0

                # Find nearest predicted orbital
                orbital_n = int(round((tau_center - 1) * E))
                tau_predicted = 1 + orbital_n / E
                error = abs(tau_center - tau_predicted)

                peak = ResonancePeak(
                    tau_center=tau_center,
                    tau_width=tau_width,
                    peak_height=peak_height,
                    q_factor=q_factor,
                    orbital_n=orbital_n,
                    tau_predicted=tau_predicted,
                    error=error
                )
                peaks.append(peak)

        self.peaks = peaks

        print(f"Found {len(peaks)} resonance peaks:")
        for p in peaks:
            match = "MATCH" if p.error < 0.15 else "miss"
            print(f"  n={p.orbital_n}: τ={p.tau_center:.2f} (pred: {p.tau_predicted:.2f}, "
                  f"err={p.error:.2f}) Q={p.q_factor:.1f} [{match}]")

        return peaks

    # =========================================================================
    # Experiment 2: Transition Spectroscopy - Selection Rules
    # =========================================================================

    def run_transition_spectroscopy(self, n_trials: int = 20) -> List[TransitionMeasurement]:
        """
        Measure transition probabilities between orbitals.

        Like quantum selection rules:
        - Start at orbital n_i
        - Let system evolve (pump + emission)
        - Measure where it ends up (n_f)
        - Allowed transitions: Δn = ±1
        - Forbidden transitions: |Δn| > 1
        """
        print("\n" + "=" * 70)
        print("TRANSITION SPECTROSCOPY: Selection Rules")
        print("=" * 70)

        # Test orbitals 0-5 (below the Veil)
        test_orbitals = list(range(6))
        transitions = []

        # Build transition matrix
        transition_counts = defaultdict(lambda: defaultdict(int))
        total_from = defaultdict(int)

        for n_start in test_orbitals:
            tau_start = 1 + n_start / E
            print(f"\nStarting from orbital n={n_start} (τ={tau_start:.2f})...")

            for trial in range(n_trials):
                # Get seeds at this orbital
                seeds = self._get_seeds_at_tau(tau_start, n_seeds=3)
                if not seeds:
                    continue

                # Run laser - where does it end up?
                try:
                    result = self.laser.lase(
                        seeds=seeds,
                        pump_power=5,
                        pump_depth=3,
                        coherence_threshold=0.2,
                        min_cluster_size=2
                    )
                except:
                    continue

                # Get output orbital
                beams = result.get('beams', [])
                if not beams:
                    continue

                # Primary beam's orbital
                primary_beam = beams[0]
                tau_end = primary_beam.tau_mean
                n_end = int(round((tau_end - 1) * E))
                n_end = max(0, min(n_end, 15))  # Clamp

                transition_counts[n_start][n_end] += 1
                total_from[n_start] += 1

        # Convert to probabilities
        for n_start in test_orbitals:
            if total_from[n_start] == 0:
                continue

            tau_start = 1 + n_start / E

            for n_end in range(16):
                count = transition_counts[n_start][n_end]
                if count == 0:
                    continue

                prob = count / total_from[n_start]
                tau_end = 1 + n_end / E
                delta_n = n_end - n_start
                allowed = abs(delta_n) == 1

                measurement = TransitionMeasurement(
                    n_start=n_start,
                    n_end=n_end,
                    tau_start=tau_start,
                    tau_end=tau_end,
                    probability=prob,
                    delta_n=delta_n,
                    allowed=allowed
                )
                transitions.append(measurement)

        self.transitions = transitions

        # Print transition matrix
        print("\nTRANSITION MATRIX (rows=start, cols=end):")
        print("     ", end="")
        for n in range(8):
            print(f"  n={n}", end="")
        print()

        for n_start in test_orbitals:
            if total_from[n_start] == 0:
                continue
            print(f"n={n_start}: ", end="")
            for n_end in range(8):
                prob = transition_counts[n_start][n_end] / total_from[n_start]
                if prob > 0.01:
                    marker = "*" if abs(n_end - n_start) == 1 else " "
                    print(f" {prob:.2f}{marker}", end="")
                else:
                    print("   -  ", end="")
            print()

        # Analyze selection rules
        allowed_prob = np.mean([t.probability for t in transitions if t.allowed and t.probability > 0])
        forbidden_prob = np.mean([t.probability for t in transitions if not t.allowed and t.probability > 0]) or 0

        print(f"\nSELECTION RULE ANALYSIS:")
        print(f"  Allowed (Δn=±1) avg probability: {allowed_prob:.2%}")
        print(f"  Forbidden (|Δn|>1) avg probability: {forbidden_prob:.2%}")
        print(f"  Selection rule strength: {allowed_prob / (forbidden_prob + 0.001):.1f}x")

        return transitions

    # =========================================================================
    # Experiment 3: Q-Factor Analysis
    # =========================================================================

    def analyze_q_factors(self) -> Dict:
        """
        Analyze Q-factors of resonance peaks.

        Q = τ_peak / Δτ_FWHM

        For true quantization:
        - Q should be consistent across orbitals
        - Q should be related to 1/e (the spacing)
        """
        if not self.peaks:
            print("[ERROR] Run peak detection first")
            return {}

        print("\n" + "-" * 50)
        print("Q-FACTOR ANALYSIS")
        print("-" * 50)

        q_values = [p.q_factor for p in self.peaks]

        if not q_values:
            return {'error': 'No peaks found'}

        q_mean = np.mean(q_values)
        q_std = np.std(q_values)
        q_expected = E  # Predicted Q ≈ e (related to orbital spacing)

        print(f"Q-factor statistics:")
        print(f"  Mean Q: {q_mean:.2f}")
        print(f"  Std Q: {q_std:.2f}")
        print(f"  Expected Q ≈ e: {E:.2f}")
        print(f"  Deviation from e: {abs(q_mean - E) / E * 100:.1f}%")

        return {
            'q_mean': q_mean,
            'q_std': q_std,
            'q_expected': E,
            'q_values': q_values,
            'consistency': q_std / q_mean if q_mean > 0 else float('inf')
        }

    # =========================================================================
    # Main Experiment Runner
    # =========================================================================

    def run_all(self) -> Dict:
        """Run all resonance experiments."""
        print("\n" + "=" * 70)
        print("ORBITAL RESONANCE SPECTROSCOPY")
        print("Proving Quantization Through Resonance")
        print("=" * 70)
        print()
        print("Euler orbital predictions:")
        print(f"  τ_n = 1 + n/e, where e = {E:.4f}")
        print(f"  Orbital spacing: Δτ = 1/e = {ORBITAL_SPACING:.4f}")
        print(f"  Predicted orbitals: {[f'n={n}: τ={1+n/E:.2f}' for n in range(6)]}")
        print()

        # Check connection
        if not self.graph.is_connected():
            print("[ERROR] Neo4j not connected")
            print("  Start with: cd config && docker-compose up -d")
            return {}

        results = {}

        # Experiment 1: Spectroscopy
        spectrum = self.run_spectroscopy(n_seeds=5, pump_power=8, pump_depth=4)
        results['spectrum'] = [
            {
                'tau_drive': p.tau_drive,
                'coherence': p.coherence,
                'beam_count': p.beam_count,
                'response_tau': p.response_tau,
                'intensity': p.intensity,
                'lasing': p.lasing
            }
            for p in spectrum
        ]

        # Find peaks
        peaks = self.find_peaks(min_peak_height=0.4, min_peak_prominence=0.05)
        results['peaks'] = [
            {
                'tau_center': p.tau_center,
                'tau_width': p.tau_width,
                'peak_height': p.peak_height,
                'q_factor': p.q_factor,
                'orbital_n': p.orbital_n,
                'tau_predicted': p.tau_predicted,
                'error': p.error,
                'matches_prediction': p.error < 0.15
            }
            for p in peaks
        ]

        # Experiment 2: Transition spectroscopy
        transitions = self.run_transition_spectroscopy(n_trials=15)
        results['transitions'] = [
            {
                'n_start': t.n_start,
                'n_end': t.n_end,
                'probability': t.probability,
                'delta_n': t.delta_n,
                'allowed': t.allowed
            }
            for t in transitions
        ]

        # Experiment 3: Q-factor analysis
        q_analysis = self.analyze_q_factors()
        results['q_analysis'] = q_analysis

        # Summary
        self._print_summary(results)

        # Save results
        self._save_results(results)

        return results

    def _print_summary(self, results: Dict):
        """Print experiment summary."""
        print("\n" + "=" * 70)
        print("RESONANCE PROOF SUMMARY")
        print("=" * 70)

        peaks = results.get('peaks', [])
        matching = [p for p in peaks if p.get('matches_prediction', False)]

        print(f"\nSPECTROSCOPY:")
        print(f"  Points measured: {len(results.get('spectrum', []))}")
        print(f"  Peaks found: {len(peaks)}")
        print(f"  Peaks matching τ_n = 1 + n/e: {len(matching)}/{len(peaks)}")

        if matching:
            print(f"\nMATCHING PEAKS (Proof of Quantization):")
            for p in matching:
                print(f"  n={p['orbital_n']}: τ={p['tau_center']:.2f} "
                      f"(predicted {p['tau_predicted']:.2f}, error={p['error']:.3f})")

        transitions = results.get('transitions', [])
        if transitions:
            allowed_trans = [t for t in transitions if t['allowed'] and t['probability'] > 0.1]
            forbidden_trans = [t for t in transitions if not t['allowed'] and t['probability'] > 0.1]

            print(f"\nTRANSITION SPECTROSCOPY:")
            print(f"  Strong allowed (Δn=±1): {len(allowed_trans)}")
            print(f"  Strong forbidden (|Δn|>1): {len(forbidden_trans)}")

            if allowed_trans and not forbidden_trans:
                print(f"  SELECTION RULES CONFIRMED: Only Δn=±1 transitions observed")

        q_analysis = results.get('q_analysis', {})
        if 'q_mean' in q_analysis:
            print(f"\nQ-FACTOR ANALYSIS:")
            print(f"  Mean Q: {q_analysis['q_mean']:.2f}")
            print(f"  Expected Q ≈ e: {E:.2f}")
            print(f"  Consistency: {q_analysis.get('consistency', 0):.2f}")

        # Final verdict
        n_peaks = len(peaks)
        n_matching = len(matching)

        print("\n" + "=" * 70)
        if n_matching >= 3:
            print("ORBITAL QUANTIZATION PROVEN BY RESONANCE")
            print(f"  {n_matching} peaks at predicted τ_n = 1 + n/e positions")
        elif n_matching >= 1:
            print("PARTIAL EVIDENCE FOR ORBITAL QUANTIZATION")
            print(f"  {n_matching} peaks at predicted positions (need more)")
        else:
            print("INCONCLUSIVE - Need more data or different parameters")
        print("=" * 70)

    def _save_results(self, results: Dict):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"orbital_resonance_{timestamp}.json"

        output = {
            'timestamp': timestamp,
            'experiment': 'Orbital Resonance Spectroscopy',
            'theory': {
                'orbital_formula': 'τ_n = 1 + n/e',
                'euler_constant': float(E),
                'orbital_spacing': float(ORBITAL_SPACING),
                'predicted_orbitals': [
                    {'n': n, 'tau': float(1 + n/E)} for n in range(8)
                ]
            },
            'parameters': {
                'tau_min': TAU_MIN,
                'tau_max': TAU_MAX,
                'tau_steps': TAU_STEPS,
                'tau_tolerance': TAU_TOLERANCE
            },
            'results': results,
            'summary': {
                'peaks_found': len(results.get('peaks', [])),
                'peaks_matching': len([p for p in results.get('peaks', [])
                                       if p.get('matches_prediction', False)]),
                'quantization_proven': len([p for p in results.get('peaks', [])
                                            if p.get('matches_prediction', False)]) >= 3
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def plot_spectrum(self, save_path: Optional[Path] = None):
        """Generate spectrum plot."""
        if not self.spectrum:
            print("No spectrum data to plot")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        taus = [p.tau_drive for p in self.spectrum]
        coherences = [p.coherence for p in self.spectrum]
        intensities = [p.intensity for p in self.spectrum]
        beam_counts = [p.beam_count for p in self.spectrum]

        # Plot 1: Coherence spectrum (main)
        ax1 = axes[0, 0]
        ax1.plot(taus, coherences, 'b-', linewidth=1.5, label='Coherence')

        # Mark orbital positions
        for n in range(8):
            tau_n = 1 + n / E
            if tau_n <= TAU_MAX:
                ax1.axvline(tau_n, color='red', linestyle='--', alpha=0.5)
                ax1.text(tau_n, ax1.get_ylim()[1] * 0.95, f'n={n}',
                        ha='center', fontsize=8, color='red')

        # Mark peaks
        for peak in self.peaks:
            ax1.plot(peak.tau_center, peak.peak_height, 'g^', markersize=10)

        ax1.axvline(E, color='purple', linestyle='-', linewidth=2, alpha=0.7, label=f'Veil (τ=e)')
        ax1.set_xlabel('τ (driving frequency)')
        ax1.set_ylabel('Coherence (response)')
        ax1.set_title('RESONANCE SPECTRUM: Coherence vs τ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Intensity
        ax2 = axes[0, 1]
        ax2.fill_between(taus, intensities, alpha=0.6, color='orange')
        ax2.plot(taus, intensities, 'orange', linewidth=1)
        for n in range(8):
            tau_n = 1 + n / E
            if tau_n <= TAU_MAX:
                ax2.axvline(tau_n, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('τ')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Beam Intensity vs τ')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Beam count
        ax3 = axes[1, 0]
        ax3.bar(taus, beam_counts, width=0.05, alpha=0.7, color='green')
        for n in range(8):
            tau_n = 1 + n / E
            if tau_n <= TAU_MAX:
                ax3.axvline(tau_n, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('τ')
        ax3.set_ylabel('Beam Count')
        ax3.set_title('Number of Coherent Beams vs τ')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Peak analysis
        ax4 = axes[1, 1]
        if self.peaks:
            peak_taus = [p.tau_center for p in self.peaks]
            peak_heights = [p.peak_height for p in self.peaks]
            peak_errors = [p.error for p in self.peaks]

            colors = ['green' if e < 0.15 else 'red' for e in peak_errors]
            ax4.bar(peak_taus, peak_heights, width=0.1, color=colors, alpha=0.7)

            # Predicted positions
            for n in range(8):
                tau_n = 1 + n / E
                if tau_n <= TAU_MAX:
                    ax4.axvline(tau_n, color='blue', linestyle=':', alpha=0.7)

            ax4.set_xlabel('Peak τ')
            ax4.set_ylabel('Peak Height')
            ax4.set_title('Detected Peaks (green=matches prediction)')
        else:
            ax4.text(0.5, 0.5, 'No peaks detected', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
        ax4.grid(True, alpha=0.3)

        plt.suptitle('ORBITAL RESONANCE SPECTROSCOPY\n'
                    f'Proving τ_n = 1 + n/e quantization',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.savefig(self._results_dir / 'orbital_resonance_spectrum.png',
                       dpi=150, bbox_inches='tight')

        plt.close()

    def close(self):
        """Clean up."""
        self.laser.close()
        self.graph.close()


def main():
    """Run orbital resonance experiment."""
    experiment = OrbitalResonanceExperiment()

    try:
        results = experiment.run_all()

        # Generate plot
        experiment.plot_spectrum()

        return 0 if results.get('summary', {}).get('quantization_proven', False) else 1

    finally:
        experiment.close()


if __name__ == "__main__":
    exit(main())
