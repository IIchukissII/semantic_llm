#!/usr/bin/env python3
"""
Semantic Thermodynamics

Exploring thermodynamic analogies in semantic space:

1. ENTROPY (S): Disorder/diversity of semantic paths
   - Path entropy: How many ways can meaning flow?
   - State entropy: How diverse is the concept distribution?

2. TEMPERATURE (T): Controls exploration/exploitation
   - Low T: Deterministic, follows highest-weight paths
   - High T: Random, explores all paths equally

3. FREE ENERGY (F = U - TS): Balance of order and disorder
   - Internal energy U ≈ potential φ
   - Entropy term TS rewards exploration
   - Minimum F = stable semantic state

4. HEAT CAPACITY: How much "thermal" energy to change τ
   - dτ/dT at different τ-levels

5. PHASE TRANSITIONS: Qualitative changes in behavior
   - Does coherence Φ show critical behavior?
   - Is there a "melting point" for semantic structure?

"Meaning flows like heat, from high potential to low"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import sys

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from chain_core.storm_logos import Storm, Logos, StormLogosBuilder


@dataclass
class ThermodynamicState:
    """Thermodynamic state of a semantic walk."""
    temperature: float
    n_steps: int

    # Entropy measures
    path_entropy: float       # Shannon entropy of path choices
    state_entropy: float      # Entropy of visited concept distribution

    # Energy measures
    avg_potential: float      # Average φ = λτ - μg
    potential_variance: float

    # Free energy
    free_energy: float        # F = U - TS

    # τ dynamics
    avg_tau: float
    tau_variance: float
    avg_delta_tau: float

    # Coherence
    coherence: float


@dataclass
class PhasePoint:
    """A point in the phase diagram."""
    temperature: float
    coherence: float
    free_energy: float
    avg_tau: float
    path_entropy: float


class SemanticThermodynamics:
    """Explore thermodynamic properties of semantic space."""

    LAMBDA = 0.5  # gravitational constant
    MU = 0.5      # lift constant

    def __init__(self):
        self.graph = MeaningGraph()
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

    def _compute_phi(self, tau: float, g: float) -> float:
        """Semantic potential φ = +λτ - μg"""
        return self.LAMBDA * tau - self.MU * g

    def _compute_path_entropy(self, choice_probs: List[List[float]]) -> float:
        """
        Compute Shannon entropy of path choices.

        H = -Σ p log p

        Higher entropy = more random/uncertain choices
        Lower entropy = more deterministic paths
        """
        if not choice_probs:
            return 0.0

        entropies = []
        for probs in choice_probs:
            if not probs or len(probs) < 2:
                continue
            probs = np.array(probs)
            probs = probs[probs > 1e-10]  # Remove zeros
            if len(probs) > 0:
                H = -np.sum(probs * np.log(probs))
                entropies.append(H)

        return np.mean(entropies) if entropies else 0.0

    def _compute_state_entropy(self, visit_counts: Dict[str, int]) -> float:
        """
        Compute entropy of visited state distribution.

        Uniform visits = high entropy
        Concentrated visits = low entropy
        """
        if not visit_counts:
            return 0.0

        counts = np.array(list(visit_counts.values()))
        probs = counts / counts.sum()
        probs = probs[probs > 1e-10]

        if len(probs) == 0:
            return 0.0

        return -np.sum(probs * np.log(probs))

    def measure_at_temperature(self, seeds: List[str], temperature: float,
                                n_walks: int = 10, steps_per_walk: int = 10
                                ) -> ThermodynamicState:
        """
        Measure thermodynamic properties at a given temperature.
        """
        storm = Storm(temperature=temperature)

        # Track quantities
        potentials = []
        taus = []
        delta_taus = []
        choice_probs_all = []
        visit_counts = defaultdict(int)

        for seed in seeds:
            seed_concept = self.graph.get_concept(seed)
            if not seed_concept:
                continue

            for walk in range(n_walks):
                current = seed
                current_tau = seed_concept.get('tau', 3.0)
                current_g = seed_concept.get('g', 0.0)

                for step in range(steps_per_walk):
                    visit_counts[current] += 1
                    potentials.append(self._compute_phi(current_tau, current_g))
                    taus.append(current_tau)

                    # Get transitions and compute probabilities
                    transitions = storm._get_transitions(current)
                    if not transitions:
                        break

                    # Compute Boltzmann probabilities
                    weights = np.array([t[1] for t in transitions])
                    T = max(0.01, temperature)
                    exp_w = np.exp(weights / T)
                    probs = exp_w / np.sum(exp_w)
                    choice_probs_all.append(probs.tolist())

                    # Sample next
                    next_word = storm._sample_next(transitions)
                    if not next_word:
                        break

                    next_concept = self.graph.get_concept(next_word)
                    if not next_concept:
                        break

                    next_tau = next_concept.get('tau', 3.0)
                    delta_taus.append(next_tau - current_tau)

                    current = next_word
                    current_tau = next_tau
                    current_g = next_concept.get('g', 0.0)

        storm.close()

        # Compute thermodynamic quantities
        path_entropy = self._compute_path_entropy(choice_probs_all)
        state_entropy = self._compute_state_entropy(dict(visit_counts))

        avg_potential = np.mean(potentials) if potentials else 0
        potential_var = np.var(potentials) if potentials else 0

        # Free energy: F = U - TS (using path entropy)
        free_energy = avg_potential - temperature * path_entropy

        avg_tau = np.mean(taus) if taus else 0
        tau_var = np.var(taus) if taus else 0
        avg_delta_tau = np.mean(delta_taus) if delta_taus else 0

        # Compute coherence using Logos
        builder = StormLogosBuilder(storm_temperature=temperature,
                                    n_walks=n_walks, steps_per_walk=steps_per_walk)
        try:
            _, pattern = builder.build(seeds, [], f"Thermo test T={temperature}")
            coherence = pattern.coherence
        except:
            coherence = 0.0
        finally:
            builder.close()

        return ThermodynamicState(
            temperature=temperature,
            n_steps=len(potentials),
            path_entropy=path_entropy,
            state_entropy=state_entropy,
            avg_potential=avg_potential,
            potential_variance=potential_var,
            free_energy=free_energy,
            avg_tau=avg_tau,
            tau_variance=tau_var,
            avg_delta_tau=avg_delta_tau,
            coherence=coherence
        )

    def scan_temperature(self, seeds: List[str],
                         T_range: Tuple[float, float] = (0.1, 5.0),
                         n_points: int = 15) -> List[PhasePoint]:
        """
        Scan across temperature range to find phase transitions.
        """
        temperatures = np.linspace(T_range[0], T_range[1], n_points)
        phase_points = []

        print(f"\nScanning temperature from {T_range[0]} to {T_range[1]}...")
        print(f"{'T':>6} {'Coherence':>10} {'Free E':>10} {'τ avg':>8} {'Path H':>10}")
        print("-" * 50)

        for T in temperatures:
            state = self.measure_at_temperature(seeds, T)
            point = PhasePoint(
                temperature=T,
                coherence=state.coherence,
                free_energy=state.free_energy,
                avg_tau=state.avg_tau,
                path_entropy=state.path_entropy
            )
            phase_points.append(point)

            print(f"{T:>6.2f} {state.coherence:>10.3f} {state.free_energy:>10.3f} "
                  f"{state.avg_tau:>8.2f} {state.path_entropy:>10.3f}")

        return phase_points

    def find_critical_temperature(self, phase_points: List[PhasePoint]) -> Optional[float]:
        """
        Find critical temperature where coherence drops sharply.

        Look for maximum |dΦ/dT| - steepest coherence change.
        """
        if len(phase_points) < 3:
            return None

        temperatures = [p.temperature for p in phase_points]
        coherences = [p.coherence for p in phase_points]

        # Compute derivative
        dC_dT = np.gradient(coherences, temperatures)

        # Find steepest drop (most negative derivative)
        min_idx = np.argmin(dC_dT)

        if dC_dT[min_idx] < -0.05:  # Significant drop
            return temperatures[min_idx]
        return None

    def compute_heat_capacity(self, seeds: List[str], T: float,
                              delta_T: float = 0.1) -> float:
        """
        Compute semantic "heat capacity" C = dU/dT

        How much the average potential changes with temperature.
        """
        state_low = self.measure_at_temperature(seeds, T - delta_T)
        state_high = self.measure_at_temperature(seeds, T + delta_T)

        dU = state_high.avg_potential - state_low.avg_potential
        C = dU / (2 * delta_T)

        return C

    def analyze_equilibrium(self, seeds: List[str], temperature: float = 1.5,
                            n_walks: int = 20, max_steps: int = 50) -> Dict:
        """
        Analyze approach to equilibrium.

        Track τ distribution as walk progresses to see if it reaches steady state.
        """
        storm = Storm(temperature=temperature)

        # Track τ at each step
        tau_by_step = defaultdict(list)

        for seed in seeds:
            seed_concept = self.graph.get_concept(seed)
            if not seed_concept:
                continue

            for walk in range(n_walks):
                current = seed
                current_tau = seed_concept.get('tau', 3.0)

                for step in range(max_steps):
                    tau_by_step[step].append(current_tau)

                    transitions = storm._get_transitions(current)
                    next_word = storm._sample_next(transitions)

                    if not next_word:
                        break

                    next_concept = self.graph.get_concept(next_word)
                    if not next_concept:
                        break

                    current = next_word
                    current_tau = next_concept.get('tau', 3.0)

        storm.close()

        # Compute statistics at each step
        equilibrium_analysis = {
            "step": [],
            "avg_tau": [],
            "std_tau": [],
            "n_samples": []
        }

        for step in sorted(tau_by_step.keys()):
            taus = tau_by_step[step]
            equilibrium_analysis["step"].append(step)
            equilibrium_analysis["avg_tau"].append(np.mean(taus))
            equilibrium_analysis["std_tau"].append(np.std(taus))
            equilibrium_analysis["n_samples"].append(len(taus))

        # Estimate equilibrium τ (average of last 10 steps)
        if len(equilibrium_analysis["avg_tau"]) >= 10:
            eq_tau = np.mean(equilibrium_analysis["avg_tau"][-10:])
        else:
            eq_tau = equilibrium_analysis["avg_tau"][-1] if equilibrium_analysis["avg_tau"] else 0

        equilibrium_analysis["equilibrium_tau"] = eq_tau

        return equilibrium_analysis

    def run_full_analysis(self, seeds: List[str] = None):
        """Run complete thermodynamic analysis."""
        if seeds is None:
            seeds = ["love", "truth", "beauty", "wisdom"]

        print("\n" + "=" * 70)
        print("SEMANTIC THERMODYNAMICS ANALYSIS")
        print("=" * 70)
        print(f"Seeds: {seeds}")

        # 1. Temperature scan
        print("\n--- PHASE DIAGRAM: Temperature Scan ---")
        phase_points = self.scan_temperature(seeds, T_range=(0.2, 4.0), n_points=12)

        # Find critical temperature
        T_c = self.find_critical_temperature(phase_points)
        if T_c:
            print(f"\n→ Critical temperature T_c ≈ {T_c:.2f}")
            print("  (Coherence drops sharply here)")
        else:
            print("\n→ No sharp phase transition detected")

        # 2. Detailed analysis at key temperatures
        print("\n--- THERMODYNAMIC STATES ---")
        print(f"{'T':>6} {'Path H':>10} {'State H':>10} {'φ avg':>10} {'F':>10} {'Δτ':>10}")
        print("-" * 60)

        for T in [0.5, 1.0, 1.5, 2.0, 3.0]:
            state = self.measure_at_temperature(seeds, T, n_walks=15)
            print(f"{T:>6.1f} {state.path_entropy:>10.3f} {state.state_entropy:>10.3f} "
                  f"{state.avg_potential:>10.3f} {state.free_energy:>10.3f} "
                  f"{state.avg_delta_tau:>+10.3f}")

        # 3. Equilibrium analysis
        print("\n--- APPROACH TO EQUILIBRIUM ---")
        eq_analysis = self.analyze_equilibrium(seeds, temperature=1.5, max_steps=30)

        print(f"{'Step':>6} {'Avg τ':>10} {'Std τ':>10}")
        print("-" * 30)
        for i in range(0, len(eq_analysis["step"]), 5):
            print(f"{eq_analysis['step'][i]:>6} {eq_analysis['avg_tau'][i]:>10.3f} "
                  f"{eq_analysis['std_tau'][i]:>10.3f}")

        print(f"\n→ Equilibrium τ ≈ {eq_analysis['equilibrium_tau']:.2f}")

        # 4. Heat capacity
        print("\n--- HEAT CAPACITY ---")
        for T in [0.5, 1.0, 1.5, 2.0, 3.0]:
            C = self.compute_heat_capacity(seeds, T)
            print(f"  C(T={T:.1f}) = {C:+.4f}")

        # 5. Summary
        print("\n" + "=" * 70)
        print("THERMODYNAMIC SUMMARY")
        print("=" * 70)

        low_T = self.measure_at_temperature(seeds, 0.5, n_walks=10)
        high_T = self.measure_at_temperature(seeds, 3.0, n_walks=10)

        print(f"""
Thermodynamic Properties:

LOW TEMPERATURE (T=0.5):
  - Path entropy: {low_T.path_entropy:.3f} (deterministic)
  - Coherence Φ:  {low_T.coherence:.3f} (ordered)
  - Free energy:  {low_T.free_energy:.3f}
  - Avg τ:        {low_T.avg_tau:.2f}

HIGH TEMPERATURE (T=3.0):
  - Path entropy: {high_T.path_entropy:.3f} (random)
  - Coherence Φ:  {high_T.coherence:.3f} (disordered?)
  - Free energy:  {high_T.free_energy:.3f}
  - Avg τ:        {high_T.avg_tau:.2f}

INTERPRETATION:
  - Low T: Follows strongest semantic connections (ordered phase)
  - High T: Random exploration (disordered phase)
  - Equilibrium τ ≈ {eq_analysis['equilibrium_tau']:.2f} (ground level)
  - {"Phase transition at T_c ≈ " + f"{T_c:.2f}" if T_c else "Smooth crossover (no sharp transition)"}
""")

        # Save results
        self._save_results(phase_points, eq_analysis)

        return {
            "phase_points": phase_points,
            "equilibrium": eq_analysis,
            "critical_T": T_c
        }

    def _save_results(self, phase_points: List[PhasePoint], eq_analysis: Dict):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"thermodynamics_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "phase_diagram": [
                {
                    "T": p.temperature,
                    "coherence": p.coherence,
                    "free_energy": p.free_energy,
                    "avg_tau": p.avg_tau,
                    "path_entropy": p.path_entropy
                }
                for p in phase_points
            ],
            "equilibrium": eq_analysis
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def close(self):
        if self.graph:
            self.graph.close()


def main():
    thermo = SemanticThermodynamics()
    try:
        thermo.run_full_analysis(["love", "truth", "beauty", "wisdom"])
    finally:
        thermo.close()


if __name__ == "__main__":
    main()
