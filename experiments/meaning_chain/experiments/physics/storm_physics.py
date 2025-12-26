#!/usr/bin/env python3
"""
Storm-Logos Physics Observer

Observes semantic physics during Storm-Logos dialogue:
- Tracks τ-transitions during storm phase
- Measures gravity (toward low τ) vs lift (toward high τ)
- Computes potential φ = +λτ - μg at each step
- Validates corrected physics model in real-time

"Watch the storm obey semantic gravity"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import sys

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from chain_core.storm_logos import Storm, Logos, StormLogosBuilder, StormState, StormResult, LogosPattern
from graph.meaning_graph import MeaningGraph


@dataclass
class PhysicsStep:
    """Record of a single τ-transition."""
    from_word: str
    to_word: str
    from_tau: float
    to_tau: float
    delta_tau: float  # to_tau - from_tau
    from_g: float
    to_g: float
    phi_from: float   # potential at source
    phi_to: float     # potential at target
    delta_phi: float  # phi_to - phi_from


@dataclass
class PhysicsObservation:
    """Physics observation for one storm-logos cycle."""
    timestamp: str
    seeds: List[str]

    # Storm phase physics
    n_steps: int
    n_falling: int      # Δτ < 0 (toward ground)
    n_rising: int       # Δτ > 0 (toward sky)
    n_flat: int         # |Δτ| < 0.1
    fall_ratio: float   # falling / rising

    # τ statistics
    avg_delta_tau: float
    tau_start_avg: float
    tau_end_avg: float

    # Potential φ = +λτ - μg
    avg_phi_start: float
    avg_phi_end: float
    phi_decreased: int   # Steps where φ decreased (following gradient)
    phi_increased: int   # Steps where φ increased (against gradient)

    # Logos phase
    pattern_tau: float
    pattern_g: float
    pattern_coherence: float
    convergence_point: Optional[str]

    # Individual steps (for deep analysis)
    steps: List[PhysicsStep] = field(default_factory=list)


class StormPhysicsObserver:
    """Observes physics during Storm-Logos dialogue."""

    LAMBDA = 0.5  # gravitational constant
    MU = 0.5      # lift constant

    def __init__(self):
        self.graph = MeaningGraph()
        self.observations: List[PhysicsObservation] = []
        self._results_dir = _PHYSICS_DIR / "results"
        self._results_dir.mkdir(exist_ok=True)

    def _get_concept(self, word: str) -> Optional[Dict]:
        """Get concept properties from graph."""
        if not self.graph.is_connected():
            return None
        return self.graph.get_concept(word)

    def _compute_phi(self, tau: float, g: float) -> float:
        """Compute semantic potential: φ = +λτ - μg"""
        return self.LAMBDA * tau - self.MU * g

    def observe_storm(self, seeds: List[str],
                      n_walks: int = 5,
                      steps_per_walk: int = 8,
                      temperature: float = 1.5) -> PhysicsObservation:
        """
        Observe physics during storm phase.

        Runs storm and tracks every τ-transition.
        """
        print(f"\n[PHYSICS] Observing storm from seeds: {seeds}")

        storm = Storm(temperature=temperature)
        steps: List[PhysicsStep] = []

        # Track transitions manually
        for seed in seeds:
            seed_concept = self._get_concept(seed)
            if not seed_concept:
                continue

            seed_tau = seed_concept.get('tau', 3.0)
            seed_g = seed_concept.get('g', 0.0)

            for walk in range(n_walks):
                current_word = seed
                current_tau = seed_tau
                current_g = seed_g
                current_phi = self._compute_phi(current_tau, current_g)

                for step in range(steps_per_walk):
                    transitions = storm._get_transitions(current_word)
                    next_word = storm._sample_next(transitions)

                    if not next_word:
                        break

                    next_concept = self._get_concept(next_word)
                    if not next_concept:
                        break

                    next_tau = next_concept.get('tau', 3.0)
                    next_g = next_concept.get('g', 0.0)
                    next_phi = self._compute_phi(next_tau, next_g)

                    # Record physics step
                    physics_step = PhysicsStep(
                        from_word=current_word,
                        to_word=next_word,
                        from_tau=current_tau,
                        to_tau=next_tau,
                        delta_tau=next_tau - current_tau,
                        from_g=current_g,
                        to_g=next_g,
                        phi_from=current_phi,
                        phi_to=next_phi,
                        delta_phi=next_phi - current_phi
                    )
                    steps.append(physics_step)

                    # Move to next
                    current_word = next_word
                    current_tau = next_tau
                    current_g = next_g
                    current_phi = next_phi

        storm.close()

        # Analyze physics
        n_falling = sum(1 for s in steps if s.delta_tau < -0.1)
        n_rising = sum(1 for s in steps if s.delta_tau > 0.1)
        n_flat = sum(1 for s in steps if abs(s.delta_tau) <= 0.1)

        fall_ratio = n_falling / n_rising if n_rising > 0 else float('inf')

        avg_delta_tau = np.mean([s.delta_tau for s in steps]) if steps else 0
        tau_start_avg = np.mean([s.from_tau for s in steps]) if steps else 0
        tau_end_avg = np.mean([s.to_tau for s in steps]) if steps else 0

        avg_phi_start = np.mean([s.phi_from for s in steps]) if steps else 0
        avg_phi_end = np.mean([s.phi_to for s in steps]) if steps else 0
        phi_decreased = sum(1 for s in steps if s.delta_phi < 0)
        phi_increased = sum(1 for s in steps if s.delta_phi > 0)

        obs = PhysicsObservation(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            seeds=seeds,
            n_steps=len(steps),
            n_falling=n_falling,
            n_rising=n_rising,
            n_flat=n_flat,
            fall_ratio=fall_ratio,
            avg_delta_tau=avg_delta_tau,
            tau_start_avg=tau_start_avg,
            tau_end_avg=tau_end_avg,
            avg_phi_start=avg_phi_start,
            avg_phi_end=avg_phi_end,
            phi_decreased=phi_decreased,
            phi_increased=phi_increased,
            pattern_tau=0.0,
            pattern_g=0.0,
            pattern_coherence=0.0,
            convergence_point=None,
            steps=steps
        )

        return obs

    def observe_full_cycle(self, nouns: List[str], verbs: List[str],
                           n_walks: int = 5, steps_per_walk: int = 8,
                           temperature: float = 1.5) -> PhysicsObservation:
        """
        Observe physics through full storm-logos cycle.
        """
        print(f"\n[PHYSICS] Full cycle observation")
        print(f"  Seeds: {nouns}")
        print(f"  Verbs: {verbs}")

        # First observe storm phase
        obs = self.observe_storm(nouns, n_walks, steps_per_walk, temperature)

        # Now run logos phase to get pattern
        builder = StormLogosBuilder(
            storm_temperature=temperature,
            n_walks=n_walks,
            steps_per_walk=steps_per_walk
        )

        try:
            _, pattern = builder.build(nouns, verbs, f"Physics test: {nouns}")
            obs.pattern_tau = pattern.tau_level
            obs.pattern_g = pattern.g_direction
            obs.pattern_coherence = pattern.coherence
            obs.convergence_point = pattern.convergence_point
        finally:
            builder.close()

        self.observations.append(obs)
        return obs

    def print_observation(self, obs: PhysicsObservation):
        """Print physics observation summary."""
        print("\n" + "=" * 60)
        print("STORM PHYSICS OBSERVATION")
        print("=" * 60)

        print(f"\nSeeds: {obs.seeds}")
        print(f"Steps: {obs.n_steps}")

        print(f"\n--- τ-Transitions ---")
        print(f"  Falling (→ ground): {obs.n_falling} ({obs.n_falling/obs.n_steps*100:.1f}%)")
        print(f"  Rising (→ sky):     {obs.n_rising} ({obs.n_rising/obs.n_steps*100:.1f}%)")
        print(f"  Flat:               {obs.n_flat} ({obs.n_flat/obs.n_steps*100:.1f}%)")
        print(f"  Fall ratio:         {obs.fall_ratio:.3f}")

        gravity_ok = obs.fall_ratio > 1.0
        print(f"\n  Gravity test: {'✓ PASS' if gravity_ok else '✗ FAIL'} (ratio > 1.0)")

        print(f"\n--- τ-Level Statistics ---")
        print(f"  Start avg τ: {obs.tau_start_avg:.2f}")
        print(f"  End avg τ:   {obs.tau_end_avg:.2f}")
        print(f"  Avg Δτ:      {obs.avg_delta_tau:+.3f}")

        drift_down = obs.avg_delta_tau < 0
        print(f"\n  τ-drift test: {'✓ PASS' if drift_down else '✗ FAIL'} (avg Δτ < 0)")

        print(f"\n--- Potential φ = +λτ - μg ---")
        print(f"  Start avg φ: {obs.avg_phi_start:.3f}")
        print(f"  End avg φ:   {obs.avg_phi_end:.3f}")
        print(f"  φ decreased: {obs.phi_decreased} ({obs.phi_decreased/obs.n_steps*100:.1f}%)")
        print(f"  φ increased: {obs.phi_increased} ({obs.phi_increased/obs.n_steps*100:.1f}%)")

        gradient_ok = obs.phi_decreased > obs.phi_increased
        print(f"\n  Gradient test: {'✓ PASS' if gradient_ok else '✗ FAIL'} (more decreasing)")

        print(f"\n--- Logos Pattern ---")
        print(f"  Pattern τ:    {obs.pattern_tau:.2f}")
        print(f"  Pattern g:    {obs.pattern_g:+.3f}")
        print(f"  Coherence Φ:  {obs.pattern_coherence:.3f}")
        print(f"  Convergence:  {obs.convergence_point}")

        # Overall assessment
        tests_passed = sum([gravity_ok, drift_down, gradient_ok])
        print(f"\n{'=' * 60}")
        print(f"PHYSICS VALIDATION: {tests_passed}/3 tests passed")
        print("=" * 60)

    def save_observations(self, filename: str = None):
        """Save observations to JSON."""
        if not filename:
            filename = f"storm_physics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_file = self._results_dir / filename

        # Convert to serializable format
        data = []
        for obs in self.observations:
            obs_dict = {
                "timestamp": obs.timestamp,
                "seeds": obs.seeds,
                "n_steps": obs.n_steps,
                "n_falling": obs.n_falling,
                "n_rising": obs.n_rising,
                "n_flat": obs.n_flat,
                "fall_ratio": obs.fall_ratio,
                "avg_delta_tau": obs.avg_delta_tau,
                "tau_start_avg": obs.tau_start_avg,
                "tau_end_avg": obs.tau_end_avg,
                "avg_phi_start": obs.avg_phi_start,
                "avg_phi_end": obs.avg_phi_end,
                "phi_decreased": obs.phi_decreased,
                "phi_increased": obs.phi_increased,
                "pattern_tau": obs.pattern_tau,
                "pattern_g": obs.pattern_g,
                "pattern_coherence": obs.pattern_coherence,
                "convergence_point": obs.convergence_point,
                "steps": [
                    {
                        "from": s.from_word,
                        "to": s.to_word,
                        "delta_tau": s.delta_tau,
                        "delta_phi": s.delta_phi
                    } for s in obs.steps[:50]  # Limit steps for file size
                ]
            }
            data.append(obs_dict)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved to: {output_file}")

    def close(self):
        if self.graph:
            self.graph.close()


def run_physics_dialogue():
    """Run a dialogue session with physics observation."""
    print("\n" + "=" * 70)
    print("STORM-LOGOS PHYSICS TEST")
    print("=" * 70)
    print("Testing corrected physics during dialogue:")
    print("  - Gravity pulls toward LOW τ (ground)")
    print("  - Potential φ = +λτ - μg should decrease")
    print("  - Pattern should settle at low τ")
    print()

    observer = StormPhysicsObserver()

    # Test cases with different semantic domains
    test_cases = [
        (["love", "dream"], ["understand", "find"]),
        (["truth", "beauty"], ["seek", "discover"]),
        (["fear", "hope"], ["overcome", "embrace"]),
        (["wisdom", "knowledge"], ["gain", "share"]),
        (["life", "death"], ["understand", "accept"]),
    ]

    try:
        for nouns, verbs in test_cases:
            obs = observer.observe_full_cycle(nouns, verbs)
            observer.print_observation(obs)
            print()

        # Summary
        print("\n" + "=" * 70)
        print("AGGREGATE PHYSICS SUMMARY")
        print("=" * 70)

        all_obs = observer.observations

        total_steps = sum(o.n_steps for o in all_obs)
        total_falling = sum(o.n_falling for o in all_obs)
        total_rising = sum(o.n_rising for o in all_obs)

        print(f"\nTotal steps observed: {total_steps}")
        print(f"Total falling (→ ground): {total_falling} ({total_falling/total_steps*100:.1f}%)")
        print(f"Total rising (→ sky): {total_rising} ({total_rising/total_steps*100:.1f}%)")
        print(f"Overall fall ratio: {total_falling/total_rising:.3f}")

        avg_coherence = np.mean([o.pattern_coherence for o in all_obs])
        avg_pattern_tau = np.mean([o.pattern_tau for o in all_obs])

        print(f"\nAverage pattern τ: {avg_pattern_tau:.2f}")
        print(f"Average coherence Φ: {avg_coherence:.3f}")

        # Physics validation
        gravity_passed = total_falling > total_rising
        tau_passed = avg_pattern_tau < 3.0

        print(f"\n{'=' * 70}")
        print("CORRECTED PHYSICS VALIDATION")
        print("=" * 70)
        print(f"  [{'✓' if gravity_passed else '✗'}] Gravity dominates (falling > rising)")
        print(f"  [{'✓' if tau_passed else '✗'}] Patterns settle at ground (τ < 3)")
        print(f"\n  Overall: {'PHYSICS CONFIRMED' if gravity_passed and tau_passed else 'NEEDS INVESTIGATION'}")

        observer.save_observations()

    finally:
        observer.close()


if __name__ == "__main__":
    run_physics_dialogue()
