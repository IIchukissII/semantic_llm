#!/usr/bin/env python3
"""
Gravity-Aware Storm: Physics-Enhanced Semantic Navigation

Extends the Storm phase with gravitational dynamics:
- Transitions prefer lower potential (falling is natural)
- Potential φ = λτ - μA (altitude costs, affirmation lifts)
- Walks "feel" gravity toward human reality (low τ)

Coordinate System (Jan 2026):
    A = Affirmation (PC1, 83.3% variance) - formerly 'g'
    S = Sacred (PC2, 11.7% variance)
    τ = Abstraction level [1=concrete, 6=abstract]

"Meaning falls like rain toward common ground"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json
import sys

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from chain_core.storm_logos import Storm, StormState, StormResult, Logos, LogosPattern


# =============================================================================
# Physics Constants
# =============================================================================

LAMBDA = 0.5      # Gravitational constant (τ coupling)
MU = 0.5          # Lift constant (A coupling) - Affirmation
VEIL_TAU = 3.5    # Quasi-Lagrange point (boundary between realms)


# =============================================================================
# Enhanced Data Structures
# =============================================================================

@dataclass
class PhysicsState(StormState):
    """Storm state with physics properties."""
    phi: float = 0.0              # Semantic potential: φ = λτ - μA
    realm: str = "human"          # "human" (τ<3.5) or "transcendental" (τ≥3.5)

    @staticmethod
    def from_storm_state(state: StormState) -> 'PhysicsState':
        """Convert StormState to PhysicsState."""
        # φ = λτ - μA (using state.affirmation, with fallback to g for legacy)
        A = getattr(state, 'affirmation', state.g) if hasattr(state, 'affirmation') else state.g
        phi = LAMBDA * state.tau - MU * A
        realm = "transcendental" if state.tau >= VEIL_TAU else "human"
        return PhysicsState(
            word=state.word,
            affirmation=A,
            tau=state.tau,
            j=state.j,
            activation=state.activation,
            phi=phi,
            realm=realm
        )


@dataclass
class PhysicsTrajectory:
    """Track physics observables during a walk."""
    tau_values: List[float] = field(default_factory=list)
    phi_values: List[float] = field(default_factory=list)
    affirmation_values: List[float] = field(default_factory=list)  # A values
    delta_tau: List[float] = field(default_factory=list)  # τ changes
    delta_phi: List[float] = field(default_factory=list)  # φ changes
    veil_crossings: int = 0
    words: List[str] = field(default_factory=list)

    def add_step(self, word: str, tau: float, affirmation: float):
        """Record a step in the walk."""
        phi = LAMBDA * tau - MU * affirmation  # φ = λτ - μA

        if self.tau_values:
            self.delta_tau.append(tau - self.tau_values[-1])
            self.delta_phi.append(phi - self.phi_values[-1])

            # Check for veil crossing
            prev_realm = "transcendental" if self.tau_values[-1] >= VEIL_TAU else "human"
            curr_realm = "transcendental" if tau >= VEIL_TAU else "human"
            if prev_realm != curr_realm:
                self.veil_crossings += 1

        self.tau_values.append(tau)
        self.phi_values.append(phi)
        self.affirmation_values.append(affirmation)
        self.words.append(word)

    @property
    def avg_tau(self) -> float:
        return np.mean(self.tau_values) if self.tau_values else 0.0

    @property
    def avg_phi(self) -> float:
        return np.mean(self.phi_values) if self.phi_values else 0.0

    @property
    def net_delta_tau(self) -> float:
        """Net change in τ (negative = falling)."""
        return sum(self.delta_tau) if self.delta_tau else 0.0

    @property
    def net_delta_phi(self) -> float:
        """Net change in potential (negative = releasing energy)."""
        return sum(self.delta_phi) if self.delta_phi else 0.0

    @property
    def gravity_compliance(self) -> float:
        """Fraction of steps that lowered potential (followed gravity)."""
        if not self.delta_phi:
            return 0.0
        downward = sum(1 for dp in self.delta_phi if dp < 0)
        return downward / len(self.delta_phi)


@dataclass
class GravityStormResult(StormResult):
    """Storm result with physics tracking."""
    trajectories: List[PhysicsTrajectory] = field(default_factory=list)
    physics_thoughts: List[PhysicsState] = field(default_factory=list)

    @property
    def avg_tau(self) -> float:
        if not self.trajectories:
            return 0.0
        return np.mean([t.avg_tau for t in self.trajectories])

    @property
    def avg_phi(self) -> float:
        if not self.trajectories:
            return 0.0
        return np.mean([t.avg_phi for t in self.trajectories])

    @property
    def total_veil_crossings(self) -> int:
        return sum(t.veil_crossings for t in self.trajectories)

    @property
    def gravity_compliance(self) -> float:
        """Average compliance with gravity across all walks."""
        if not self.trajectories:
            return 0.0
        return np.mean([t.gravity_compliance for t in self.trajectories])

    @property
    def realm_distribution(self) -> Dict[str, float]:
        """Fraction of time spent in each realm."""
        all_tau = []
        for t in self.trajectories:
            all_tau.extend(t.tau_values)
        if not all_tau:
            return {"human": 0.5, "transcendental": 0.5}
        human = sum(1 for tau in all_tau if tau < VEIL_TAU)
        return {
            "human": human / len(all_tau),
            "transcendental": 1 - human / len(all_tau)
        }


# =============================================================================
# Gravity-Aware Storm
# =============================================================================

class GravityStorm(Storm):
    """
    Storm phase with gravitational awareness.

    Transitions are weighted by potential difference:
    - Falling (Δφ < 0) is energetically favored
    - Rising (Δφ > 0) requires "work" against gravity

    The gravity_strength parameter controls how strongly gravity affects sampling:
    - 0.0: Pure edge-weight sampling (original behavior)
    - 1.0: Full gravitational influence
    """

    def __init__(self, temperature: float = 1.5, gravity_strength: float = 0.5):
        """
        Args:
            temperature: Controls exploration (higher = more random)
            gravity_strength: How much gravity affects transitions [0, 1]
        """
        super().__init__(temperature=temperature)
        self.gravity_strength = gravity_strength
        self._current_state: Optional[PhysicsState] = None

    def _get_physics_state(self, word: str) -> Optional[PhysicsState]:
        """Get physics-aware state for a concept."""
        state = self._get_concept(word)
        if state is None:
            return None
        return PhysicsState.from_storm_state(state)

    def _compute_transition_energy(self, current: PhysicsState,
                                    next_word: str, edge_weight: float) -> float:
        """
        Compute effective energy for a transition.

        E_eff = -w + α·Δφ

        Where:
            w = edge weight (higher = more likely)
            α = gravity_strength
            Δφ = φ(next) - φ(current) = potential change

        Lower E_eff = more likely transition.
        """
        # Get next concept properties
        next_concept = self._graph.get_concept(next_word) if self._graph else None
        if not next_concept:
            return -edge_weight  # Fallback to pure edge weight

        next_tau = next_concept.get('tau', 3.0)
        next_A = next_concept.get('g', 0.0)  # g ≈ A in database

        # Compute potential change: φ = λτ - μA
        phi_current = current.phi
        phi_next = LAMBDA * next_tau - MU * next_A
        delta_phi = phi_next - phi_current

        # Effective energy: edge weight (inverted) + gravitational potential
        # Negative edge weight because higher weight = lower energy = more likely
        E_eff = -edge_weight + self.gravity_strength * delta_phi

        return E_eff

    def _sample_next_with_gravity(self, current: PhysicsState,
                                   transitions: List[Tuple[str, float]]) -> Optional[str]:
        """
        Sample next concept using gravity-aware Boltzmann distribution.

        P(next) ∝ exp(-E_eff / T)

        Where E_eff combines edge weight and gravitational potential.
        """
        if not transitions:
            return None

        words = [t[0] for t in transitions]
        weights = [t[1] for t in transitions]

        # Compute effective energies
        energies = np.array([
            self._compute_transition_energy(current, word, weight)
            for word, weight in transitions
        ])

        # Boltzmann sampling
        T = max(0.01, self.temperature)
        exp_neg_E = np.exp(-energies / T)
        probs = exp_neg_E / np.sum(exp_neg_E)

        return np.random.choice(words, p=probs)

    def generate(self, seeds: List[str], n_walks: int = 5,
                 steps_per_walk: int = 8) -> GravityStormResult:
        """
        Generate gravity-aware storm of thoughts.

        Returns enhanced result with physics tracking.
        """
        thoughts = []
        physics_thoughts = []
        visit_counts = Counter()
        total_steps = 0
        trajectories = []

        for seed in seeds:
            seed_state = self._get_physics_state(seed)
            if not seed_state:
                continue

            for walk in range(n_walks):
                trajectory = PhysicsTrajectory()
                current = seed_state
                visit_counts[current.word] += 1
                trajectory.add_step(current.word, current.tau, current.affirmation)

                for step in range(steps_per_walk):
                    transitions = self._get_transitions(current.word)

                    # Use gravity-aware sampling
                    next_word = self._sample_next_with_gravity(current, transitions)

                    if not next_word:
                        break

                    # Get next state with physics
                    next_state = self._get_physics_state(next_word)
                    if not next_state:
                        break

                    # Activation decay with distance
                    next_state.activation = 1.0 / (1.0 + step * 0.2)

                    # Record
                    thoughts.append(next_state)
                    physics_thoughts.append(next_state)
                    visit_counts[next_word] += 1
                    trajectory.add_step(next_word, next_state.tau, next_state.affirmation)

                    current = next_state
                    total_steps += 1

                trajectories.append(trajectory)

        return GravityStormResult(
            seeds=seeds,
            thoughts=thoughts,
            visit_counts=dict(visit_counts),
            total_steps=total_steps,
            trajectories=trajectories,
            physics_thoughts=physics_thoughts
        )


# =============================================================================
# Comparison Test
# =============================================================================

class GravityStormTest:
    """Test gravity-aware vs standard storm."""

    def __init__(self):
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

    def compare_storms(self, seeds: List[str], n_walks: int = 10,
                       steps_per_walk: int = 10,
                       gravity_strengths: List[float] = None) -> Dict:
        """
        Compare storm behavior at different gravity strengths.
        """
        if gravity_strengths is None:
            gravity_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

        results = {
            "seeds": seeds,
            "n_walks": n_walks,
            "steps_per_walk": steps_per_walk,
            "comparisons": []
        }

        print("\n" + "=" * 70)
        print("GRAVITY STORM COMPARISON")
        print("=" * 70)
        print(f"Seeds: {seeds}")
        print(f"Walks: {n_walks} × {steps_per_walk} steps each")

        print(f"\n{'Gravity':>8} {'Avg τ':>8} {'Avg φ':>8} {'Gravity':>10} {'Veil':>6} {'Human%':>8}")
        print(f"{'Strength':>8} {'':>8} {'':>8} {'Compliance':>10} {'Cross':>6} {'':>8}")
        print("-" * 60)

        for g_strength in gravity_strengths:
            storm = GravityStorm(temperature=1.5, gravity_strength=g_strength)

            try:
                result = storm.generate(seeds, n_walks, steps_per_walk)

                realm_dist = result.realm_distribution

                comparison = {
                    "gravity_strength": g_strength,
                    "avg_tau": result.avg_tau,
                    "avg_phi": result.avg_phi,
                    "gravity_compliance": result.gravity_compliance,
                    "veil_crossings": result.total_veil_crossings,
                    "human_fraction": realm_dist["human"],
                    "total_steps": result.total_steps
                }
                results["comparisons"].append(comparison)

                print(f"{g_strength:>8.2f} {result.avg_tau:>8.2f} {result.avg_phi:>8.2f} "
                      f"{result.gravity_compliance:>10.1%} {result.total_veil_crossings:>6} "
                      f"{realm_dist['human']:>8.1%}")

            finally:
                storm.close()

        # Analysis
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)

        if len(results["comparisons"]) >= 2:
            no_gravity = results["comparisons"][0]
            full_gravity = results["comparisons"][-1]

            tau_change = full_gravity["avg_tau"] - no_gravity["avg_tau"]
            phi_change = full_gravity["avg_phi"] - no_gravity["avg_phi"]
            compliance_change = full_gravity["gravity_compliance"] - no_gravity["gravity_compliance"]

            print(f"\nEffect of gravity (α=0 → α=1):")
            print(f"  Δτ:          {tau_change:+.3f} ({'lower' if tau_change < 0 else 'higher'} altitude)")
            print(f"  Δφ:          {phi_change:+.3f} ({'lower' if phi_change < 0 else 'higher'} potential)")
            print(f"  Compliance:  {compliance_change:+.1%} ({'more' if compliance_change > 0 else 'less'} gravity-following)")

            if tau_change < 0 and compliance_change > 0:
                print("\n✓ GRAVITY IS WORKING: Stronger gravity → lower τ, more downward motion")
            else:
                print("\n? UNEXPECTED: Gravity effect not as predicted")

        return results

    def test_altitude_dependence(self, n_walks: int = 15,
                                  steps_per_walk: int = 12) -> Dict:
        """
        Test if gravity effect depends on starting altitude.

        Hypothesis: Gravity should be stronger from high τ (more room to fall).
        """
        # Seeds at different altitudes
        test_cases = [
            ("Ground (τ≈1-2)", ["love", "heart", "life", "way"]),
            ("Mid (τ≈3-4)", ["understanding", "essence", "principle"]),
            ("Sky (τ≈5-6)", ["transcendence", "ineffable", "ultimate"]),
        ]

        print("\n" + "=" * 70)
        print("ALTITUDE-DEPENDENT GRAVITY TEST")
        print("=" * 70)

        results = {"test_cases": []}

        for name, seeds in test_cases:
            print(f"\n--- {name} ---")
            print(f"Seeds: {seeds}")

            # Compare no gravity vs full gravity
            no_grav = GravityStorm(temperature=1.5, gravity_strength=0.0)
            full_grav = GravityStorm(temperature=1.5, gravity_strength=1.0)

            try:
                result_no = no_grav.generate(seeds, n_walks, steps_per_walk)
                result_full = full_grav.generate(seeds, n_walks, steps_per_walk)

                delta_tau = result_full.avg_tau - result_no.avg_tau
                delta_compliance = result_full.gravity_compliance - result_no.gravity_compliance

                case_result = {
                    "name": name,
                    "seeds": seeds,
                    "no_gravity": {
                        "avg_tau": result_no.avg_tau,
                        "gravity_compliance": result_no.gravity_compliance
                    },
                    "full_gravity": {
                        "avg_tau": result_full.avg_tau,
                        "gravity_compliance": result_full.gravity_compliance
                    },
                    "delta_tau": delta_tau,
                    "delta_compliance": delta_compliance
                }
                results["test_cases"].append(case_result)

                print(f"  No gravity:   τ={result_no.avg_tau:.2f}, compliance={result_no.gravity_compliance:.1%}")
                print(f"  Full gravity: τ={result_full.avg_tau:.2f}, compliance={result_full.gravity_compliance:.1%}")
                print(f"  Effect:       Δτ={delta_tau:+.3f}, Δcompliance={delta_compliance:+.1%}")

            finally:
                no_grav.close()
                full_grav.close()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for case in results["test_cases"]:
            effect = "STRONG" if case["delta_tau"] < -0.1 else "WEAK" if case["delta_tau"] < 0 else "NONE"
            print(f"  {case['name']:20s}: Δτ={case['delta_tau']:+.3f} [{effect}]")

        return results

    def run_full_test(self) -> Dict:
        """Run complete gravity storm test suite."""
        results = {}

        # Test 1: Gravity strength comparison
        print("\n\n" + "=" * 70)
        print("TEST 1: GRAVITY STRENGTH COMPARISON")
        print("=" * 70)
        results["strength_comparison"] = self.compare_storms(
            seeds=["love", "truth", "beauty", "wisdom"],
            gravity_strengths=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        # Test 2: Altitude dependence
        print("\n\n" + "=" * 70)
        print("TEST 2: ALTITUDE DEPENDENCE")
        print("=" * 70)
        results["altitude_dependence"] = self.test_altitude_dependence()

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self._results_dir / f"gravity_storm_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run gravity storm tests."""
    test = GravityStormTest()
    test.run_full_test()


if __name__ == "__main__":
    main()
