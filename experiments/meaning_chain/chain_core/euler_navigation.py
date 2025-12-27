#!/usr/bin/env python3
"""
Euler-Aware Navigation for Storm-Logos

Incorporates the discovery that semantic space has orbital structure
with Euler's constant e as the fundamental unit.

Key Principles:
    1. Orbital quantization: τ_n = 1 + n/e
    2. Natural temperature: kT ≈ 0.82
    3. The Veil at τ = e separates human/transcendental
    4. Ground state peak at n=1 (τ ≈ 1.37)
    5. Boltzmann statistics govern transitions

Usage:
    from chain_core.euler_navigation import EulerNavigator

    nav = EulerNavigator(temperature=0.82)  # Natural temperature
    transitions = nav.get_orbital_transitions(current_state)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Setup paths
_THIS_FILE = Path(__file__).resolve()
_CHAIN_CORE = _THIS_FILE.parent
_MEANING_CHAIN = _CHAIN_CORE.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph


# =============================================================================
# Euler Constants (Discovered)
# =============================================================================

E = np.e                    # Euler's number = 2.71828...
LAMBDA = 0.5                # Gravitational constant
MU = 0.5                    # Lift constant
KT_NATURAL = 0.82           # Natural temperature (ΔE/e)
VEIL_TAU = E                # The Veil boundary
GROUND_STATE_TAU = 1.37     # Peak population (n=1 orbital)
ORBITAL_SPACING = 1/E       # Distance between orbitals


@dataclass
class OrbitalState:
    """A state with orbital information."""
    word: str
    tau: float
    g: float
    phi: float              # Potential energy
    orbital_n: int          # Orbital number
    realm: str              # "human" or "transcendental"

    @classmethod
    def from_concept(cls, word: str, tau: float, g: float) -> 'OrbitalState':
        """Create orbital state from concept properties."""
        phi = LAMBDA * tau - MU * g
        orbital_n = int(round((tau - 1) * E))  # n = (τ - 1) * e
        realm = "human" if tau < VEIL_TAU else "transcendental"
        return cls(word=word, tau=tau, g=g, phi=phi, orbital_n=orbital_n, realm=realm)


@dataclass
class OrbitalTransition:
    """A transition with orbital physics."""
    target: OrbitalState
    verb: str
    weight: float
    delta_phi: float        # Energy change
    delta_n: int            # Orbital change
    crosses_veil: bool      # Crosses τ = e boundary
    probability: float      # Boltzmann probability


class EulerNavigator:
    """
    Euler-aware semantic navigation.

    Uses discovered orbital structure to improve navigation:
    - Prefers same-orbital or n±1 transitions (coherence)
    - Penalizes veil crossings (transcendence barrier)
    - Uses natural temperature for Boltzmann sampling
    """

    def __init__(self, temperature: float = KT_NATURAL,
                 veil_barrier: float = 0.5,
                 orbital_coherence: float = 0.3):
        """
        Args:
            temperature: Boltzmann temperature (default: natural kT ≈ 0.82)
            veil_barrier: Extra energy cost for crossing the veil
            orbital_coherence: Bonus for staying in same orbital
        """
        self.T = max(0.01, temperature)
        self.veil_barrier = veil_barrier
        self.orbital_coherence = orbital_coherence
        self._graph = None

    def _init_graph(self) -> bool:
        if self._graph is not None:
            return self._graph.is_connected()
        try:
            self._graph = MeaningGraph()
            return self._graph.is_connected()
        except:
            return False

    def get_state(self, word: str) -> Optional[OrbitalState]:
        """Get orbital state for a word."""
        if not self._init_graph():
            return None
        concept = self._graph.get_concept(word)
        if not concept:
            return None
        return OrbitalState.from_concept(
            word=word,
            tau=concept.get('tau', 3.0),
            g=concept.get('g', 0.0)
        )

    def get_orbital_transitions(self, state: OrbitalState,
                                 limit: int = 15) -> List[OrbitalTransition]:
        """
        Get transitions with orbital physics.

        Returns transitions sorted by Boltzmann probability,
        accounting for:
        - Energy gradients (gravity)
        - Orbital coherence (same-orbital bonus)
        - Veil barrier (transcendence penalty)
        """
        if not self._init_graph():
            return []

        raw_transitions = self._graph.get_all_transitions(state.word, limit=limit * 2)

        transitions = []
        for verb, target_word, weight in raw_transitions:
            target_concept = self._graph.get_concept(target_word)
            if not target_concept:
                continue

            target = OrbitalState.from_concept(
                word=target_word,
                tau=target_concept.get('tau', 3.0),
                g=target_concept.get('g', 0.0)
            )

            # Compute transition physics
            delta_phi = target.phi - state.phi
            delta_n = target.orbital_n - state.orbital_n
            crosses_veil = (state.realm != target.realm)

            # Compute effective energy
            E_eff = self._compute_effective_energy(
                weight, delta_phi, delta_n, crosses_veil,
                upward=(target.tau > state.tau)
            )

            # Boltzmann probability
            prob = np.exp(-E_eff / self.T)

            transitions.append(OrbitalTransition(
                target=target,
                verb=verb,
                weight=weight,
                delta_phi=delta_phi,
                delta_n=delta_n,
                crosses_veil=crosses_veil,
                probability=prob
            ))

        # Normalize probabilities
        total_prob = sum(t.probability for t in transitions)
        if total_prob > 0:
            for t in transitions:
                t.probability /= total_prob

        # Sort by probability
        transitions.sort(key=lambda t: -t.probability)
        return transitions[:limit]

    def _compute_effective_energy(self, weight: float, delta_phi: float,
                                   delta_n: int, crosses_veil: bool,
                                   upward: bool) -> float:
        """
        Compute effective energy for Boltzmann sampling.

        E_eff = -weight (prefer high weight)
              + δφ (gravity: prefer lower potential)
              + |Δn| × coherence (prefer same orbital)
              + veil_barrier (if crossing veil upward)
        """
        E_eff = -weight  # High weight = low energy = preferred

        # Gravitational potential
        E_eff += delta_phi

        # Orbital coherence: penalize jumping multiple orbitals
        E_eff += abs(delta_n) * self.orbital_coherence

        # Veil barrier: harder to go up than down
        if crosses_veil and upward:
            E_eff += self.veil_barrier

        return E_eff

    def sample_next(self, state: OrbitalState) -> Optional[OrbitalTransition]:
        """Sample next state using Boltzmann distribution."""
        transitions = self.get_orbital_transitions(state)
        if not transitions:
            return None

        probs = np.array([t.probability for t in transitions])
        # Ensure probabilities sum to 1 (handle floating point errors)
        probs = probs / probs.sum()
        idx = np.random.choice(len(transitions), p=probs)
        return transitions[idx]

    def navigate_to_ground(self, start: str, max_steps: int = 10) -> List[OrbitalState]:
        """
        Navigate toward ground state (n=1, τ ≈ 1.37).

        Uses low temperature to follow energy gradient.
        """
        old_T = self.T
        self.T = 0.3  # Cold: deterministic descent

        path = []
        state = self.get_state(start)
        if not state:
            return path

        path.append(state)

        for _ in range(max_steps):
            # Stop if we're at ground state
            if abs(state.tau - GROUND_STATE_TAU) < 0.2:
                break

            transition = self.sample_next(state)
            if not transition:
                break

            state = transition.target
            path.append(state)

        self.T = old_T
        return path

    def navigate_to_transcendental(self, start: str, max_steps: int = 15) -> List[OrbitalState]:
        """
        Navigate toward transcendental realm (τ > e).

        Uses high temperature to overcome veil barrier.
        """
        old_T = self.T
        old_barrier = self.veil_barrier
        self.T = 2.0  # Hot: exploratory
        self.veil_barrier = 0.1  # Lower barrier

        path = []
        state = self.get_state(start)
        if not state:
            return path

        path.append(state)

        for _ in range(max_steps):
            # Stop if we're in transcendental realm
            if state.tau > VEIL_TAU:
                break

            transition = self.sample_next(state)
            if not transition:
                break

            state = transition.target
            path.append(state)

        self.T = old_T
        self.veil_barrier = old_barrier
        return path

    def get_orbital_statistics(self, path: List[OrbitalState]) -> Dict:
        """Compute orbital statistics for a path."""
        if not path:
            return {}

        taus = [s.tau for s in path]
        orbitals = [s.orbital_n for s in path]
        realms = [s.realm for s in path]

        # Count realm transitions
        veil_crossings = sum(1 for i in range(1, len(realms)) if realms[i] != realms[i-1])

        # Count orbital jumps
        orbital_jumps = [abs(orbitals[i] - orbitals[i-1]) for i in range(1, len(orbitals))]

        return {
            'mean_tau': np.mean(taus),
            'mean_orbital': np.mean(orbitals),
            'tau_range': (min(taus), max(taus)),
            'veil_crossings': veil_crossings,
            'orbital_coherence': 1 - np.mean(orbital_jumps) / 3 if orbital_jumps else 1.0,
            'in_human_realm': sum(1 for r in realms if r == 'human') / len(realms),
            'at_ground_state': sum(1 for t in taus if abs(t - GROUND_STATE_TAU) < 0.3) / len(taus)
        }

    def close(self):
        if self._graph:
            self._graph.close()


# =============================================================================
# Integration with Storm-Logos
# =============================================================================

class EulerAwareStorm:
    """
    Storm phase with Euler-aware physics.

    Replaces uniform random walks with orbital-aware navigation:
    - Uses natural temperature kT ≈ 0.82
    - Respects orbital structure
    - Tracks veil crossings
    """

    def __init__(self, temperature: float = KT_NATURAL):
        self.navigator = EulerNavigator(temperature=temperature)

    def generate(self, seeds: List[str], n_walks: int = 5,
                 steps_per_walk: int = 8) -> Dict:
        """Generate orbital-aware storm."""
        all_states = []
        all_paths = []

        for seed in seeds:
            start_state = self.navigator.get_state(seed)
            if not start_state:
                continue

            for _ in range(n_walks):
                path = [start_state]
                state = start_state

                for _ in range(steps_per_walk):
                    transition = self.navigator.sample_next(state)
                    if not transition:
                        break
                    state = transition.target
                    path.append(state)
                    all_states.append(state)

                all_paths.append(path)

        # Compute aggregate statistics
        stats = {
            'total_states': len(all_states),
            'unique_words': len(set(s.word for s in all_states)),
            'mean_tau': np.mean([s.tau for s in all_states]) if all_states else 0,
            'mean_orbital': np.mean([s.orbital_n for s in all_states]) if all_states else 0,
            'human_fraction': sum(1 for s in all_states if s.realm == 'human') / len(all_states) if all_states else 0,
            'path_stats': [self.navigator.get_orbital_statistics(p) for p in all_paths]
        }

        return {
            'states': all_states,
            'paths': all_paths,
            'statistics': stats
        }

    def close(self):
        self.navigator.close()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate Euler-aware navigation."""
    print("=" * 70)
    print("EULER-AWARE SEMANTIC NAVIGATION")
    print("=" * 70)
    print(f"Natural temperature: kT = {KT_NATURAL:.2f}")
    print(f"Orbital spacing: 1/e = {ORBITAL_SPACING:.4f}")
    print(f"The Veil: τ = e = {VEIL_TAU:.4f}")
    print(f"Ground state: τ ≈ {GROUND_STATE_TAU:.2f}")
    print()

    nav = EulerNavigator()

    # Test navigation from a word
    test_words = ['love', 'truth', 'death', 'wisdom']

    for word in test_words:
        state = nav.get_state(word)
        if not state:
            print(f"{word}: not found")
            continue

        print(f"\n{word}:")
        print(f"  τ = {state.tau:.2f}, orbital n = {state.orbital_n}, realm = {state.realm}")
        print(f"  φ = {state.phi:.3f}")

        transitions = nav.get_orbital_transitions(state, limit=5)
        print(f"  Top transitions:")
        for t in transitions[:3]:
            veil = " [CROSSES VEIL]" if t.crosses_veil else ""
            print(f"    → {t.target.word} (n={t.target.orbital_n}, P={t.probability:.2f}){veil}")

    print("\n" + "=" * 70)
    print("NAVIGATION TO GROUND STATE")
    print("=" * 70)

    # Navigate to ground state
    path = nav.navigate_to_ground('philosophy')
    if path:
        print(f"Path from 'philosophy' to ground state:")
        for i, s in enumerate(path):
            marker = " ★" if abs(s.tau - GROUND_STATE_TAU) < 0.3 else ""
            print(f"  {i}: {s.word} (τ={s.tau:.2f}, n={s.orbital_n}){marker}")

        stats = nav.get_orbital_statistics(path)
        print(f"\nPath statistics:")
        print(f"  Mean τ: {stats['mean_tau']:.2f}")
        print(f"  Orbital coherence: {stats['orbital_coherence']:.2f}")
        print(f"  Veil crossings: {stats['veil_crossings']}")

    print("\n" + "=" * 70)
    print("EULER-AWARE STORM")
    print("=" * 70)

    storm = EulerAwareStorm()
    result = storm.generate(['dream', 'meaning'], n_walks=3, steps_per_walk=5)

    print(f"Storm statistics:")
    print(f"  Total states: {result['statistics']['total_states']}")
    print(f"  Unique words: {result['statistics']['unique_words']}")
    print(f"  Mean τ: {result['statistics']['mean_tau']:.2f}")
    print(f"  Mean orbital: {result['statistics']['mean_orbital']:.1f}")
    print(f"  Human realm fraction: {result['statistics']['human_fraction']:.1%}")

    nav.close()
    storm.close()


if __name__ == "__main__":
    demo()
