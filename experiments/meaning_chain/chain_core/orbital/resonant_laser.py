"""
Resonant Laser
==============

Semantic Laser with orbital resonance tuning.

Combines all orbital modules:
- OrbitalDetector: Detect natural orbital
- OrbitalTuner: Tune seeds to target orbital
- IntentOrbitalMapper: Map intent to orbital
- VeilCrosser: Cross-veil translation

The key insight: driving at the natural orbital frequency
produces maximum coherence (proven by resonance spectroscopy).

Usage:
    laser = ResonantLaser(graph)

    # Auto-tune to natural orbital
    result = laser.lase_resonant(seeds, verbs)

    # Manual orbital targeting
    result = laser.lase_at_orbital(seeds, target_orbital=3)

    # Cross-veil synthesis
    result = laser.lase_veil_synthesis(human_seeds, transcendent_seeds)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .constants import (
    E, ORBITAL_SPACING, KT_NATURAL, VEIL_TAU, VEIL_ORBITAL,
    tau_to_orbital, orbital_to_tau
)
from .detector import OrbitalDetector, OrbitalSignature
from .tuner import OrbitalTuner, TunedSeeds
from .mapper import IntentOrbitalMapper, IntentMapping
from .veil import VeilCrosser, VeilBridge

# Import base SemanticLaser
import sys
from pathlib import Path
_THIS_FILE = Path(__file__).resolve()
_ORBITAL_DIR = _THIS_FILE.parent
_CHAIN_CORE = _ORBITAL_DIR.parent
_MEANING_CHAIN = _CHAIN_CORE.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from chain_core.semantic_laser import SemanticLaser, CoherentBeam


@dataclass
class ResonantResult:
    """
    Result of resonant lasing.

    Contains standard laser output plus orbital resonance info.
    """
    # Standard laser output
    beams: List[CoherentBeam]
    population: Dict
    metrics: Dict

    # Resonance info
    target_orbital: int
    detected_orbital: int
    resonance_quality: float  # How well we resonated [0, 1]
    tuning_applied: str       # "none", "soft", "hard"

    # Input analysis
    seed_signature: OrbitalSignature
    intent_mapping: Optional[IntentMapping]
    tuned_seeds: Optional[TunedSeeds]

    @property
    def primary_beam(self) -> Optional[CoherentBeam]:
        """Get primary (most intense) beam."""
        return self.beams[0] if self.beams else None

    @property
    def coherence(self) -> float:
        """Get primary beam coherence."""
        return self.primary_beam.coherence if self.primary_beam else 0.0

    @property
    def concepts(self) -> List[str]:
        """Get primary beam concepts."""
        return self.primary_beam.concepts if self.primary_beam else []


class ResonantLaser:
    """
    Semantic Laser with orbital resonance tuning.

    Workflow:
    1. DETECT: Find natural orbital of query (from seeds + verbs)
    2. TUNE: Optionally tune seeds to target orbital
    3. LASE: Run semantic laser
    4. ANALYZE: Measure resonance quality

    Three modes:
    - AUTO: Detect natural orbital and tune softly
    - MANUAL: User specifies target orbital
    - ADAPTIVE: Multiple passes with feedback

    Usage:
        laser = ResonantLaser(graph)

        # Auto mode (recommended)
        result = laser.lase_resonant(seeds, verbs=['understand'])

        # Manual mode
        result = laser.lase_at_orbital(seeds, target_orbital=3)
    """

    def __init__(self, graph, temperature: float = KT_NATURAL):
        """
        Initialize resonant laser.

        Args:
            graph: MeaningGraph
            temperature: Boltzmann temperature (default: natural kT)
        """
        self.graph = graph
        self.temperature = temperature

        # Initialize components
        self.base_laser = SemanticLaser(graph, temperature=temperature)
        self.detector = OrbitalDetector(graph)
        self.tuner = OrbitalTuner(graph)
        self.mapper = IntentOrbitalMapper(graph)
        self.veil_crosser = VeilCrosser(graph, self.base_laser)

        # Learn verb mappings from data
        self.mapper.learn()

    def lase_resonant(self, seeds: List[str],
                       verbs: List[str] = None,
                       tuning: str = "soft",
                       pump_power: int = 10,
                       pump_depth: int = 5) -> ResonantResult:
        """
        Resonant lasing with automatic orbital detection.

        Args:
            seeds: Input concepts
            verbs: Intent verbs (optional, improves targeting)
            tuning: "none", "soft", or "hard"
                - none: Use seeds as-is
                - soft: Blend seeds with resonant concepts
                - hard: Replace off-orbital seeds
            pump_power: Laser pump power
            pump_depth: Laser pump depth

        Returns:
            ResonantResult with beams and resonance metrics
        """
        # Step 1: Detect natural orbital
        seed_signature = self.detector.detect(seeds, use_neighbors=True)

        # Step 2: Map intent to orbital (if verbs provided)
        intent_mapping = None
        if verbs:
            intent_mapping = self.mapper.map(verbs)

        # Step 3: Determine target orbital
        if intent_mapping and intent_mapping.confidence > 0.5:
            # Blend seed orbital with intent orbital
            target_orbital = int(round(
                0.4 * seed_signature.dominant_orbital +
                0.6 * intent_mapping.target_orbital
            ))
        else:
            target_orbital = seed_signature.dominant_orbital

        # Step 4: Tune seeds (if requested)
        tuned_seeds = None
        lasing_seeds = seeds

        if tuning == "soft":
            tuned_seeds = self.tuner.tune(
                seeds, target_orbital,
                strategy="blend",
                blend_ratio=0.6
            )
            lasing_seeds = tuned_seeds.seeds
        elif tuning == "hard":
            tuned_seeds = self.tuner.tune(
                seeds, target_orbital,
                strategy="replace"
            )
            lasing_seeds = tuned_seeds.seeds

        # Step 5: Lase
        laser_result = self.base_laser.lase(
            seeds=lasing_seeds,
            pump_power=pump_power,
            pump_depth=pump_depth,
            intent_verbs=verbs
        )

        # Step 6: Analyze resonance
        resonance_quality = self._compute_resonance_quality(
            laser_result, target_orbital
        )

        return ResonantResult(
            beams=laser_result.get('beams', []),
            population=laser_result.get('population', {}),
            metrics=laser_result.get('metrics', {}),
            target_orbital=target_orbital,
            detected_orbital=seed_signature.dominant_orbital,
            resonance_quality=resonance_quality,
            tuning_applied=tuning,
            seed_signature=seed_signature,
            intent_mapping=intent_mapping,
            tuned_seeds=tuned_seeds
        )

    def lase_at_orbital(self, seeds: List[str],
                         target_orbital: int,
                         tuning: str = "soft",
                         pump_power: int = 10,
                         pump_depth: int = 5) -> ResonantResult:
        """
        Lase at specific target orbital.

        Args:
            seeds: Input concepts
            target_orbital: Target orbital (0-15)
            tuning: "none", "soft", or "hard"
            pump_power: Laser pump power
            pump_depth: Laser pump depth

        Returns:
            ResonantResult
        """
        # Detect natural orbital (for comparison)
        seed_signature = self.detector.detect(seeds)

        # Tune to target
        tuned_seeds = None
        lasing_seeds = seeds

        if tuning != "none":
            strategy = "blend" if tuning == "soft" else "replace"
            tuned_seeds = self.tuner.tune(seeds, target_orbital, strategy=strategy)
            lasing_seeds = tuned_seeds.seeds

        # Lase
        laser_result = self.base_laser.lase(
            seeds=lasing_seeds,
            pump_power=pump_power,
            pump_depth=pump_depth
        )

        # Resonance quality
        resonance_quality = self._compute_resonance_quality(
            laser_result, target_orbital
        )

        return ResonantResult(
            beams=laser_result.get('beams', []),
            population=laser_result.get('population', {}),
            metrics=laser_result.get('metrics', {}),
            target_orbital=target_orbital,
            detected_orbital=seed_signature.dominant_orbital,
            resonance_quality=resonance_quality,
            tuning_applied=tuning,
            seed_signature=seed_signature,
            intent_mapping=None,
            tuned_seeds=tuned_seeds
        )

    def lase_adaptive(self, seeds: List[str],
                       verbs: List[str] = None,
                       passes: int = 2,
                       pump_power: int = 8,
                       pump_depth: int = 4) -> ResonantResult:
        """
        Adaptive multi-pass lasing with feedback.

        Pass 1: Detect and soft-tune
        Pass 2+: Use previous output to refine target

        Args:
            seeds: Input concepts
            verbs: Intent verbs
            passes: Number of passes (1-3)
            pump_power: Pump power per pass
            pump_depth: Pump depth per pass

        Returns:
            Final ResonantResult
        """
        current_seeds = seeds
        result = None

        for pass_num in range(passes):
            # Increasing tuning strength with each pass
            tuning = "soft" if pass_num == 0 else "hard"

            result = self.lase_resonant(
                seeds=current_seeds,
                verbs=verbs,
                tuning=tuning,
                pump_power=pump_power,
                pump_depth=pump_depth
            )

            # Use output as input for next pass
            if result.primary_beam and pass_num < passes - 1:
                # Take top concepts from beam as new seeds
                current_seeds = result.primary_beam.concepts[:5]

        return result

    def lase_veil_synthesis(self, human_seeds: List[str],
                             transcendent_seeds: List[str],
                             pump_power: int = 10,
                             pump_depth: int = 5) -> Dict:
        """
        Synthesize across the Veil boundary.

        Combines human and transcendental concepts to find
        meaning at the Veil (τ ≈ e).

        Args:
            human_seeds: Seeds from human realm (τ < e)
            transcendent_seeds: Seeds from transcendental realm (τ ≥ e)
            pump_power: Pump power
            pump_depth: Pump depth

        Returns:
            Synthesis result with bridge concepts
        """
        # Find bridge concepts
        bridge = self.veil_crosser.find_bridge(human_seeds, transcendent_seeds)

        # Add bridge concepts to seeds
        combined_seeds = human_seeds + transcendent_seeds + bridge.bridge_concepts

        # Target orbital near Veil
        target_orbital = VEIL_ORBITAL  # n=5, τ ≈ 2.84

        # Lase with combined seeds
        result = self.lase_at_orbital(
            seeds=combined_seeds,
            target_orbital=target_orbital,
            tuning="soft",
            pump_power=pump_power,
            pump_depth=pump_depth
        )

        return {
            'result': result,
            'bridge': bridge,
            'synthesis_concepts': result.concepts,
            'veil_proximity': abs(result.population.get('tau_mean', 3.0) - VEIL_TAU)
        }

    def lase_grounded(self, seeds: List[str],
                       verbs: List[str] = None,
                       pump_power: int = 10,
                       pump_depth: int = 5) -> ResonantResult:
        """
        Lase with grounding (human realm, low orbital).

        Useful for practical, concrete responses.

        Args:
            seeds: Input concepts
            verbs: Intent verbs
            pump_power: Pump power
            pump_depth: Pump depth

        Returns:
            ResonantResult in human realm
        """
        return self.lase_at_orbital(
            seeds=seeds,
            target_orbital=2,  # Ground orbital (τ ≈ 1.74)
            tuning="soft",
            pump_power=pump_power,
            pump_depth=pump_depth
        )

    def lase_transcendent(self, seeds: List[str],
                           verbs: List[str] = None,
                           pump_power: int = 10,
                           pump_depth: int = 5) -> ResonantResult:
        """
        Lase with ascension (transcendental realm, high orbital).

        Useful for abstract, philosophical responses.

        Args:
            seeds: Input concepts
            verbs: Intent verbs
            pump_power: Pump power
            pump_depth: Pump depth

        Returns:
            ResonantResult in transcendental realm
        """
        return self.lase_at_orbital(
            seeds=seeds,
            target_orbital=6,  # Transcendental orbital (τ ≈ 3.21)
            tuning="soft",
            pump_power=pump_power,
            pump_depth=pump_depth
        )

    def _compute_resonance_quality(self, laser_result: Dict,
                                    target_orbital: int) -> float:
        """
        Compute resonance quality.

        High quality = output matches target orbital frequency.
        """
        population = laser_result.get('population', {})
        tau_mean = population.get('tau_mean', 3.0)
        tau_std = population.get('tau_std', 1.0)

        # Distance from target
        target_tau = orbital_to_tau(target_orbital)
        distance = abs(tau_mean - target_tau)

        # Base quality from distance
        distance_quality = np.exp(-distance / ORBITAL_SPACING)

        # Penalty for spread (diffuse = low resonance)
        spread_penalty = min(1.0, tau_std / ORBITAL_SPACING)

        # Bonus for lasing achieved
        metrics = laser_result.get('metrics', {})
        lasing_bonus = 0.2 if metrics.get('lasing_achieved', False) else 0

        quality = distance_quality * (1 - 0.3 * spread_penalty) + lasing_bonus

        return min(1.0, max(0.0, quality))

    def get_orbital_statistics(self) -> Dict:
        """Get statistics about orbital mappings."""
        return {
            'mapper_stats': self.mapper.get_statistics(),
            'temperature': self.temperature,
            'orbital_positions': {n: orbital_to_tau(n) for n in range(8)}
        }

    def close(self):
        """Clean up."""
        self.base_laser.close()


def demo():
    """Demonstrate resonant laser."""
    from graph.meaning_graph import MeaningGraph

    print("=" * 70)
    print("RESONANT LASER DEMO")
    print("=" * 70)

    graph = MeaningGraph()
    if not graph.is_connected():
        print("Graph not connected. Start Neo4j first.")
        return

    laser = ResonantLaser(graph)

    # Test seeds
    seeds = ['dream', 'love', 'meaning']
    verbs = ['understand', 'find']

    print(f"\nSeeds: {seeds}")
    print(f"Verbs: {verbs}")

    # Auto-tuned resonant lasing
    print("\n--- AUTO-TUNED RESONANT LASING ---")
    result = laser.lase_resonant(seeds, verbs, tuning="soft")

    print(f"Detected orbital: n={result.detected_orbital}")
    print(f"Target orbital: n={result.target_orbital}")
    print(f"Resonance quality: {result.resonance_quality:.2%}")
    print(f"Coherence: {result.coherence:.2f}")
    print(f"Concepts: {result.concepts[:5]}")

    # Grounded lasing
    print("\n--- GROUNDED LASING (n=2) ---")
    grounded = laser.lase_grounded(seeds)
    print(f"Target orbital: n={grounded.target_orbital}")
    print(f"Resonance quality: {grounded.resonance_quality:.2%}")
    print(f"Concepts: {grounded.concepts[:5]}")

    # Transcendent lasing
    print("\n--- TRANSCENDENT LASING (n=6) ---")
    transcendent = laser.lase_transcendent(seeds)
    print(f"Target orbital: n={transcendent.target_orbital}")
    print(f"Resonance quality: {transcendent.resonance_quality:.2%}")
    print(f"Concepts: {transcendent.concepts[:5]}")

    laser.close()
    graph.close()


if __name__ == "__main__":
    demo()
