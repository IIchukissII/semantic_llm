"""
Semantic Navigator: Unified Semantic Physics Engine
====================================================

Combines all semantic engines into one coherent navigation system:

LAYERS:
    1. SAMPLING: SemanticLaser, Storm - walk through meaning graph
    2. NAVIGATION: Orbital, Intent, Gravity - where to target
    3. STATISTICS: MonteCarlo, Logos - aggregate patterns
    4. DIALECTICAL: Paradox - find tension, generate power

GOALS:
    - accurate:  High resonance (hit target precisely)
    - deep:      High coherence (low resonance paradox)
    - stable:    High stability (Monte Carlo consensus)
    - powerful:  High tension (paradox-based)
    - grounded:  Low tau (practical, human realm)
    - balanced:  Composite optimization

QUALITY METRICS:
    - Resonance (R): How well we hit target orbital
    - Coherence (C): How aligned the beam vectors
    - Depth (D):     C/R - the paradox ratio
    - Power (P):     Tension × Stability
    - Stability (S): Monte Carlo consensus

Usage:
    from chain_core.navigator import SemanticNavigator

    nav = SemanticNavigator()
    result = nav.navigate("What is consciousness?", goal="deep")
    print(result.concepts)
    print(result.quality)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))


class NavigationGoal(Enum):
    """Navigation goals - what quality to optimize."""
    ACCURATE = "accurate"    # High resonance
    DEEP = "deep"            # High coherence (paradox: low R, high C)
    STABLE = "stable"        # Monte Carlo consensus
    POWERFUL = "powerful"    # Paradox tension
    GROUNDED = "grounded"    # Low tau, practical
    BALANCED = "balanced"    # Composite optimization
    EXPLORATORY = "exploratory"  # Storm-logos chaos
    SUPERCRITICAL = "supercritical"  # Chain reaction mode (α=0.2, λ>1)
    WISDOM = "wisdom"        # Optimal C=0.1P balance (max meaning production)


@dataclass
class NavigationQuality:
    """Unified quality metrics for navigation result."""
    resonance: float = 0.0      # How well we hit target [0, 1]
    coherence: float = 0.0      # Beam alignment [0, 1]
    stability: float = 0.0      # Monte Carlo consensus [0, 1]
    power: float = 0.0          # Paradox tension × stability
    tau_mean: float = 0.0       # Average abstraction level

    @property
    def depth(self) -> float:
        """Depth = Coherence / Resonance (the paradox ratio)."""
        if self.resonance < 0.01:
            return self.coherence * 10  # Cap at 10x when R→0
        return min(10.0, self.coherence / self.resonance)

    @property
    def composite(self) -> float:
        """Balanced composite score."""
        return (0.25 * self.resonance +
                0.25 * self.coherence +
                0.20 * self.depth / 10 +  # Normalize depth to [0,1]
                0.15 * self.stability +
                0.15 * min(1.0, self.power / 5))  # Normalize power

    def score_for_goal(self, goal: NavigationGoal) -> float:
        """Compute goal-specific score."""
        if goal == NavigationGoal.ACCURATE:
            return self.resonance
        elif goal == NavigationGoal.DEEP:
            return self.coherence * 0.6 + self.depth / 10 * 0.4
        elif goal == NavigationGoal.STABLE:
            return self.stability
        elif goal == NavigationGoal.POWERFUL:
            return min(1.0, self.power / 5)
        elif goal == NavigationGoal.GROUNDED:
            # Lower tau = better for grounded
            return max(0, 1 - (self.tau_mean - 1) / 4)
        else:  # BALANCED, EXPLORATORY
            return self.composite

    def __repr__(self):
        return (f"Quality(R={self.resonance:.2f}, C={self.coherence:.2f}, "
                f"D={self.depth:.2f}, S={self.stability:.2f}, P={self.power:.2f})")


@dataclass
class NavigationResult:
    """Result from semantic navigation."""
    # Core output
    concepts: List[str]              # The beam concepts
    quality: NavigationQuality       # Quality metrics

    # Context
    query: str                       # Original query
    goal: NavigationGoal             # Navigation goal used
    strategy: str                    # Which engine(s) used

    # Decomposition
    nouns: List[str] = field(default_factory=list)
    verbs: List[str] = field(default_factory=list)

    # Orbital info
    detected_orbital: int = 0
    target_orbital: int = 0

    # Optional: raw results from engines
    orbital_result: Optional[Any] = None
    monte_carlo_result: Optional[Any] = None
    paradox_result: Optional[Any] = None
    storm_result: Optional[Any] = None

    # Synthesis (for paradox mode)
    synthesis: List[str] = field(default_factory=list)
    thesis: Optional[str] = None
    antithesis: Optional[str] = None

    def __repr__(self):
        return (f"NavigationResult(goal={self.goal.value}, "
                f"concepts={self.concepts[:5]}, "
                f"quality={self.quality})")


class SemanticNavigator:
    """
    Unified semantic physics engine.

    Combines all navigation systems:
    - Orbital resonance (accurate targeting)
    - Monte Carlo (statistical stability)
    - Paradox detection (meaning through tension)
    - Storm-Logos (exploratory chaos)

    Select strategy based on goal, or combine multiple engines.
    """

    def __init__(self, lazy_init: bool = True):
        """
        Initialize navigator.

        Args:
            lazy_init: If True, engines are initialized on first use
        """
        self.lazy_init = lazy_init

        # Engines (lazy loaded)
        self._graph = None
        self._decomposer = None
        self._orbital_laser = None
        self._monte_carlo = None
        self._paradox_detector = None
        self._storm_logos = None

        # Goal weights for quality scoring
        self._goal_weights = {
            NavigationGoal.ACCURATE: {'r': 0.7, 'c': 0.2, 'd': 0.0, 's': 0.1, 'p': 0.0},
            NavigationGoal.DEEP: {'r': 0.1, 'c': 0.4, 'd': 0.4, 's': 0.1, 'p': 0.0},
            NavigationGoal.STABLE: {'r': 0.2, 'c': 0.2, 'd': 0.1, 's': 0.5, 'p': 0.0},
            NavigationGoal.POWERFUL: {'r': 0.1, 'c': 0.2, 'd': 0.1, 's': 0.1, 'p': 0.5},
            NavigationGoal.GROUNDED: {'r': 0.3, 'c': 0.2, 'd': 0.0, 's': 0.2, 'p': 0.0, 'tau': 0.3},
            NavigationGoal.BALANCED: {'r': 0.25, 'c': 0.25, 'd': 0.2, 's': 0.15, 'p': 0.15},
            NavigationGoal.EXPLORATORY: {'r': 0.1, 'c': 0.3, 'd': 0.2, 's': 0.1, 'p': 0.3},
            NavigationGoal.SUPERCRITICAL: {'r': 0.1, 'c': 0.2, 'd': 0.1, 's': 0.1, 'p': 0.5},
            NavigationGoal.WISDOM: {'r': 0.15, 'c': 0.35, 'd': 0.15, 's': 0.15, 'p': 0.20},
        }

        # Optimal constants for WISDOM mode (from semantic energy conservation)
        # Σ = C + 0.1P ≈ e^(1/5) ≈ 1.22 (semantic budget)
        # Optimal: C = 0.1P = Σ/2 ≈ 0.615 (maximum meaning production)
        self.SIGMA = 1.2214  # e^(1/5) - semantic budget
        self.C_OPTIMAL = 0.615  # Optimal coherence
        self.P_OPTIMAL = 6.15   # Optimal power
        self.K_COUPLING = 0.1   # Power-to-coherence weight

        # Critical intent strength for chain reactions (α ≈ 0.2)
        # At this α, chain coefficient λ > 1 (supercritical)
        self.CRITICAL_ALPHA = 0.2

        # Supercritical engines (lazy loaded with α=0.2)
        self._supercritical_mc = None
        self._supercritical_paradox = None

    # =========================================================================
    # Lazy initialization
    # =========================================================================

    def _init_graph(self):
        if self._graph is None:
            from graph.meaning_graph import MeaningGraph
            self._graph = MeaningGraph()
            if not self._graph.is_connected():
                raise RuntimeError("Neo4j not connected")
        return self._graph

    def _init_decomposer(self):
        if self._decomposer is None:
            from chain_core.decomposer import Decomposer
            self._decomposer = Decomposer()
        return self._decomposer

    def _init_orbital(self):
        if self._orbital_laser is None:
            from chain_core.orbital import ResonantLaser
            self._orbital_laser = ResonantLaser(self._init_graph())
        return self._orbital_laser

    def _init_monte_carlo(self):
        if self._monte_carlo is None:
            from chain_core.monte_carlo_renderer import MonteCarloRenderer
            self._monte_carlo = MonteCarloRenderer(n_samples=30)
        return self._monte_carlo

    def _init_paradox(self):
        if self._paradox_detector is None:
            from chain_core.paradox_detector import ParadoxDetector
            self._paradox_detector = ParadoxDetector(n_samples=25)
        return self._paradox_detector

    def _init_storm_logos(self):
        if self._storm_logos is None:
            from chain_core.storm_logos import StormLogosBuilder
            self._storm_logos = StormLogosBuilder(
                storm_temperature=1.5,
                n_walks=5,
                steps_per_walk=8
            )
        return self._storm_logos

    def _init_supercritical_mc(self):
        """Initialize Monte Carlo with critical α=0.2 for chain reactions."""
        if self._supercritical_mc is None:
            from chain_core.monte_carlo_renderer import MonteCarloRenderer
            self._supercritical_mc = MonteCarloRenderer(
                intent_strength=self.CRITICAL_ALPHA,  # α=0.2: supercritical
                n_samples=30
            )
        return self._supercritical_mc

    def _init_supercritical_paradox(self):
        """Initialize Paradox detector with critical α=0.2 for chain reactions."""
        if self._supercritical_paradox is None:
            from chain_core.paradox_detector import ParadoxDetector
            self._supercritical_paradox = ParadoxDetector(
                intent_strength=self.CRITICAL_ALPHA,  # α=0.2: supercritical
                n_samples=25
            )
        return self._supercritical_paradox

    # =========================================================================
    # Decomposition
    # =========================================================================

    def decompose(self, query: str) -> Tuple[List[str], List[str]]:
        """Decompose query into nouns and verbs."""
        decomposer = self._init_decomposer()
        result = decomposer.decompose(query)
        nouns = result.nouns if result.nouns else ['meaning']
        verbs = result.verbs if result.verbs else []
        return nouns, verbs

    # =========================================================================
    # Individual engine methods
    # =========================================================================

    def _navigate_orbital(self, nouns: List[str], verbs: List[str],
                          mode: str = "auto") -> Tuple[Any, NavigationQuality]:
        """Navigate using orbital resonance system."""
        laser = self._init_orbital()

        if mode == "ground":
            result = laser.lase_grounded(nouns, verbs)
        elif mode == "transcend":
            result = laser.lase_transcendent(nouns, verbs)
        elif mode == "adaptive":
            result = laser.lase_adaptive(nouns, verbs, passes=2)
        else:  # auto
            result = laser.lase_resonant(nouns, verbs, tuning="soft")

        quality = NavigationQuality(
            resonance=result.resonance_quality,
            coherence=result.coherence,
            stability=0.5,  # Unknown without MC
            power=0.0,      # Unknown without paradox
            tau_mean=result.population.get('tau_mean', 2.0)
        )

        return result, quality

    def _navigate_monte_carlo(self, query: str) -> Tuple[Any, NavigationQuality]:
        """Navigate using Monte Carlo sampling."""
        mc = self._init_monte_carlo()
        landscape = mc.sample_landscape(query)

        quality = NavigationQuality(
            resonance=landscape.lasing_rate,
            coherence=landscape.coherence,
            stability=landscape.concentration,
            power=0.0,
            tau_mean=landscape.tau_mean
        )

        return landscape, quality

    def _navigate_paradox(self, query: str) -> Tuple[Any, NavigationQuality]:
        """Navigate using paradox detection."""
        detector = self._init_paradox()
        landscape = detector.detect(query)

        # Compute quality from paradox
        if landscape.strongest:
            power = landscape.strongest.power
            stability = landscape.strongest.stability
        else:
            power = 0.0
            stability = 0.0

        quality = NavigationQuality(
            resonance=0.5,  # Unknown without orbital
            coherence=landscape.coherence,
            stability=stability,
            power=power,
            tau_mean=2.5  # Default
        )

        return landscape, quality

    def _navigate_storm_logos(self, nouns: List[str], verbs: List[str],
                               query: str) -> Tuple[Any, NavigationQuality]:
        """Navigate using storm-logos architecture."""
        builder = self._init_storm_logos()
        tree, pattern = builder.build(nouns, verbs, query)

        quality = NavigationQuality(
            resonance=0.5,  # Storm doesn't measure resonance
            coherence=pattern.coherence,
            stability=0.5,  # Would need MC for this
            power=0.0,
            tau_mean=pattern.tau_level
        )

        return (tree, pattern), quality

    # =========================================================================
    # Main navigation method
    # =========================================================================

    def navigate(self, query: str,
                 goal: Union[str, NavigationGoal] = "balanced") -> NavigationResult:
        """
        Navigate semantic space with goal-based strategy selection.

        Args:
            query: User's question or topic
            goal: Navigation goal - what to optimize for
                - "accurate": High resonance (orbital auto)
                - "deep": High coherence, low resonance (orbital transcend)
                - "stable": Monte Carlo consensus
                - "powerful": Paradox-based tension
                - "grounded": Low tau, practical (orbital ground)
                - "balanced": Composite optimization
                - "exploratory": Storm-logos chaos

        Returns:
            NavigationResult with concepts, quality metrics, and raw results
        """
        # Parse goal
        if isinstance(goal, str):
            goal = NavigationGoal(goal.lower())

        # Decompose query
        nouns, verbs = self.decompose(query)

        # Select strategy based on goal
        if goal == NavigationGoal.ACCURATE:
            return self._navigate_with_orbital(query, nouns, verbs, "auto", goal)

        elif goal == NavigationGoal.DEEP:
            return self._navigate_with_orbital(query, nouns, verbs, "transcend", goal)

        elif goal == NavigationGoal.GROUNDED:
            return self._navigate_with_orbital(query, nouns, verbs, "ground", goal)

        elif goal == NavigationGoal.STABLE:
            return self._navigate_with_monte_carlo(query, nouns, verbs, goal)

        elif goal == NavigationGoal.POWERFUL:
            return self._navigate_with_paradox(query, nouns, verbs, goal)

        elif goal == NavigationGoal.SUPERCRITICAL:
            return self._navigate_with_supercritical(query, nouns, verbs, goal)

        elif goal == NavigationGoal.WISDOM:
            return self._navigate_with_wisdom(query, nouns, verbs, goal)

        elif goal == NavigationGoal.EXPLORATORY:
            return self._navigate_with_storm(query, nouns, verbs, goal)

        else:  # BALANCED
            return self._navigate_balanced(query, nouns, verbs, goal)

    def _navigate_with_orbital(self, query: str, nouns: List[str],
                                verbs: List[str], mode: str,
                                goal: NavigationGoal) -> NavigationResult:
        """Navigate using orbital system."""
        result, quality = self._navigate_orbital(nouns, verbs, mode)

        return NavigationResult(
            concepts=result.concepts[:10],
            quality=quality,
            query=query,
            goal=goal,
            strategy=f"orbital_{mode}",
            nouns=nouns,
            verbs=verbs,
            detected_orbital=result.detected_orbital,
            target_orbital=result.target_orbital,
            orbital_result=result
        )

    def _navigate_with_monte_carlo(self, query: str, nouns: List[str],
                                    verbs: List[str],
                                    goal: NavigationGoal) -> NavigationResult:
        """Navigate using Monte Carlo."""
        landscape, quality = self._navigate_monte_carlo(query)

        concepts = [w for w, _ in landscape.core_attractors[:10]]

        return NavigationResult(
            concepts=concepts,
            quality=quality,
            query=query,
            goal=goal,
            strategy="monte_carlo",
            nouns=nouns,
            verbs=verbs,
            monte_carlo_result=landscape
        )

    def _navigate_with_paradox(self, query: str, nouns: List[str],
                                verbs: List[str],
                                goal: NavigationGoal) -> NavigationResult:
        """Navigate using paradox detection."""
        landscape, quality = self._navigate_paradox(query)

        # Collect concepts from paradoxes
        concepts = []
        if landscape.strongest:
            concepts.append(landscape.strongest.thesis)
            concepts.append(landscape.strongest.antithesis)
            concepts.extend(landscape.strongest.synthesis_concepts[:3])

        # Add more from other paradoxes
        for p in landscape.paradoxes[1:5]:
            if p.thesis not in concepts:
                concepts.append(p.thesis)
            if p.antithesis not in concepts:
                concepts.append(p.antithesis)

        synthesis = []
        thesis = None
        antithesis = None

        if landscape.strongest:
            thesis = landscape.strongest.thesis
            antithesis = landscape.strongest.antithesis
            synthesis = landscape.strongest.synthesis_concepts

        return NavigationResult(
            concepts=concepts[:10],
            quality=quality,
            query=query,
            goal=goal,
            strategy="paradox",
            nouns=nouns,
            verbs=verbs,
            paradox_result=landscape,
            thesis=thesis,
            antithesis=antithesis,
            synthesis=synthesis
        )

    def _navigate_with_supercritical(self, query: str, nouns: List[str],
                                      verbs: List[str],
                                      goal: NavigationGoal) -> NavigationResult:
        """
        Navigate using SUPERCRITICAL mode (α=0.2).

        This mode operates at the critical point where chain coefficient λ > 1,
        enabling meaning amplification through chain reactions.

        Use for: paradox-powered dialogue, creative explosion, power amplification
        """
        # Use supercritical paradox detector (α=0.2)
        detector = self._init_supercritical_paradox()
        landscape = detector.detect(query)

        # Compute quality from paradox
        if landscape.strongest:
            power = landscape.strongest.power
            stability = landscape.strongest.stability
        else:
            power = 0.0
            stability = 0.0

        # Also sample with supercritical MC for additional stability measure
        mc = self._init_supercritical_mc()
        mc_landscape = mc.sample_landscape(query)

        # Combine quality metrics (supercritical emphasizes power)
        quality = NavigationQuality(
            resonance=mc_landscape.lasing_rate,
            coherence=max(landscape.coherence, mc_landscape.coherence),
            stability=mc_landscape.concentration,
            power=power * 1.2,  # Boost power in supercritical mode
            tau_mean=mc_landscape.tau_mean
        )

        # Collect concepts from paradoxes
        concepts = []
        if landscape.strongest:
            concepts.append(landscape.strongest.thesis)
            concepts.append(landscape.strongest.antithesis)
            concepts.extend(landscape.strongest.synthesis_concepts[:3])

        # Add more from other paradoxes
        for p in landscape.paradoxes[1:5]:
            if p.thesis not in concepts:
                concepts.append(p.thesis)
            if p.antithesis not in concepts:
                concepts.append(p.antithesis)

        # Enrich with MC attractors
        for word, _ in mc_landscape.core_attractors[:5]:
            if word not in concepts:
                concepts.append(word)

        synthesis = []
        thesis = None
        antithesis = None

        if landscape.strongest:
            thesis = landscape.strongest.thesis
            antithesis = landscape.strongest.antithesis
            synthesis = landscape.strongest.synthesis_concepts

        return NavigationResult(
            concepts=concepts[:10],
            quality=quality,
            query=query,
            goal=goal,
            strategy=f"supercritical_α={self.CRITICAL_ALPHA}",
            nouns=nouns,
            verbs=verbs,
            paradox_result=landscape,
            monte_carlo_result=mc_landscape,
            thesis=thesis,
            antithesis=antithesis,
            synthesis=synthesis
        )

    def _navigate_with_wisdom(self, query: str, nouns: List[str],
                               verbs: List[str],
                               goal: NavigationGoal) -> NavigationResult:
        """
        Navigate using WISDOM mode: optimal C = 0.1P balance.

        This mode targets the theoretical optimum for meaning production:
            C_opt = 0.615, P_opt = 6.15
            where Meaning = C × P is maximized under Σ = C + 0.1P = 1.22

        Strategy:
            1. Detect paradoxes and compute power for each
            2. Compute synthesis coherence for non-pole concepts
            3. Select the paradox closest to C = 0.1P balance
            4. Favor coherent synthesis over raw power

        Use for: wise dialogue, balanced understanding, maximum meaning
        """
        detector = self._init_paradox()
        landscape = detector.detect(query)

        mc = self._init_monte_carlo()
        mc_landscape = mc.sample_landscape(query)

        # Collect all potential synthesis concepts from MC (non-poles)
        all_attractors = [w for w, _ in mc_landscape.core_attractors[:20]]

        best_meaning = 0.0
        best_paradox = None
        best_coherence = 0.0
        best_power = 0.0
        best_synthesis = []

        # Evaluate each paradox for meaning production
        for paradox in landscape.paradoxes[:10]:
            thesis, antithesis = paradox.thesis, paradox.antithesis
            power = paradox.power

            # Filter synthesis: non-pole concepts from attractors
            synthesis = [w for w in all_attractors
                        if w not in [thesis, antithesis]][:10]

            if len(synthesis) < 2:
                continue

            # Compute synthesis coherence
            coherence = self._compute_synthesis_coherence(synthesis)
            abs_coherence = abs(coherence)

            # Compute meaning = C × P
            meaning = abs_coherence * power

            # Also consider distance from optimal balance
            # Optimal: C = 0.1P, so ratio C/(0.1P) should be close to 1
            if power > 0:
                balance_ratio = abs_coherence / (self.K_COUPLING * power)
                # Penalty for imbalance (prefer ratio close to 1)
                balance_score = 1.0 / (1.0 + abs(1.0 - balance_ratio))
                # Weighted meaning: meaning × balance
                adjusted_meaning = meaning * (0.7 + 0.3 * balance_score)
            else:
                adjusted_meaning = meaning

            if adjusted_meaning > best_meaning:
                best_meaning = adjusted_meaning
                best_paradox = paradox
                best_coherence = coherence
                best_power = power
                best_synthesis = synthesis

        # Fallback to strongest if no good balance found
        if best_paradox is None and landscape.strongest:
            best_paradox = landscape.strongest
            best_power = best_paradox.power
            best_synthesis = [w for w in all_attractors
                             if w not in [best_paradox.thesis, best_paradox.antithesis]][:10]
            best_coherence = self._compute_synthesis_coherence(best_synthesis)

        # Build concepts list: synthesis first (high coherence), then poles
        concepts = list(best_synthesis[:6])
        if best_paradox:
            if best_paradox.thesis not in concepts:
                concepts.append(best_paradox.thesis)
            if best_paradox.antithesis not in concepts:
                concepts.append(best_paradox.antithesis)

        # Add remaining attractors
        for w, _ in mc_landscape.core_attractors[:5]:
            if w not in concepts:
                concepts.append(w)

        # Compute quality metrics
        actual_sigma = abs(best_coherence) + self.K_COUPLING * best_power
        meaning_efficiency = (abs(best_coherence) * best_power) / (self.C_OPTIMAL * self.P_OPTIMAL)

        quality = NavigationQuality(
            resonance=mc_landscape.lasing_rate,
            coherence=abs(best_coherence),
            stability=mc_landscape.concentration,
            power=best_power,
            tau_mean=mc_landscape.tau_mean
        )

        thesis = best_paradox.thesis if best_paradox else None
        antithesis = best_paradox.antithesis if best_paradox else None

        result = NavigationResult(
            concepts=concepts[:10],
            quality=quality,
            query=query,
            goal=goal,
            strategy=f"wisdom_C={abs(best_coherence):.3f}_P={best_power:.2f}_Σ={actual_sigma:.2f}_eff={meaning_efficiency:.1%}",
            nouns=nouns,
            verbs=verbs,
            paradox_result=landscape,
            monte_carlo_result=mc_landscape,
            thesis=thesis,
            antithesis=antithesis,
            synthesis=best_synthesis
        )

        return result

    def _compute_synthesis_coherence(self, concepts: List[str]) -> float:
        """Compute average pairwise coherence among synthesis concepts."""
        if len(concepts) < 2:
            return 0.0

        graph = self._init_graph()
        j_vectors = []

        for word in concepts:
            concept = graph.get_concept(word)
            if concept and concept.get('j'):
                j = np.array(concept['j'])
                if len(j) == 5:
                    j_vectors.append(j)

        if len(j_vectors) < 2:
            return 0.0

        # Average pairwise cosine similarity
        sims = []
        for i in range(len(j_vectors)):
            for k in range(i + 1, len(j_vectors)):
                v1, v2 = j_vectors[i], j_vectors[k]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-8 and n2 > 1e-8:
                    sims.append(float(np.dot(v1, v2) / (n1 * n2)))

        return float(np.mean(sims)) if sims else 0.0

    def _navigate_with_storm(self, query: str, nouns: List[str],
                              verbs: List[str],
                              goal: NavigationGoal) -> NavigationResult:
        """Navigate using storm-logos."""
        (tree, pattern), quality = self._navigate_storm_logos(nouns, verbs, query)

        return NavigationResult(
            concepts=pattern.core_concepts[:10],
            quality=quality,
            query=query,
            goal=goal,
            strategy="storm_logos",
            nouns=nouns,
            verbs=verbs,
            storm_result=(tree, pattern)
        )

    def _navigate_balanced(self, query: str, nouns: List[str],
                           verbs: List[str],
                           goal: NavigationGoal) -> NavigationResult:
        """
        Balanced navigation: run multiple engines and combine.

        Uses orbital as primary, enriched with MC stability.
        """
        # Primary: orbital auto
        orbital_result, orbital_quality = self._navigate_orbital(nouns, verbs, "auto")

        # Secondary: Monte Carlo for stability estimate
        try:
            mc_landscape, mc_quality = self._navigate_monte_carlo(query)
            stability = mc_quality.stability
        except Exception:
            stability = 0.5
            mc_landscape = None

        # Combine quality metrics
        combined_quality = NavigationQuality(
            resonance=orbital_quality.resonance,
            coherence=orbital_quality.coherence,
            stability=stability,
            power=0.0,  # Would need paradox
            tau_mean=orbital_quality.tau_mean
        )

        return NavigationResult(
            concepts=orbital_result.concepts[:10],
            quality=combined_quality,
            query=query,
            goal=goal,
            strategy="balanced_orbital+mc",
            nouns=nouns,
            verbs=verbs,
            detected_orbital=orbital_result.detected_orbital,
            target_orbital=orbital_result.target_orbital,
            orbital_result=orbital_result,
            monte_carlo_result=mc_landscape
        )

    # =========================================================================
    # Multi-engine navigation
    # =========================================================================

    def navigate_multi(self, query: str,
                       engines: List[str] = None) -> Dict[str, NavigationResult]:
        """
        Run multiple engines and return all results.

        Args:
            query: User's question
            engines: List of engines to run. Default: all
                Options: "orbital", "monte_carlo", "paradox", "storm"

        Returns:
            Dict mapping engine name to NavigationResult
        """
        if engines is None:
            engines = ["orbital", "monte_carlo", "paradox"]

        nouns, verbs = self.decompose(query)
        results = {}

        if "orbital" in engines:
            try:
                results["orbital"] = self._navigate_with_orbital(
                    query, nouns, verbs, "auto", NavigationGoal.ACCURATE
                )
            except Exception as e:
                print(f"[Navigator] Orbital failed: {e}")

        if "monte_carlo" in engines:
            try:
                results["monte_carlo"] = self._navigate_with_monte_carlo(
                    query, nouns, verbs, NavigationGoal.STABLE
                )
            except Exception as e:
                print(f"[Navigator] Monte Carlo failed: {e}")

        if "paradox" in engines:
            try:
                results["paradox"] = self._navigate_with_paradox(
                    query, nouns, verbs, NavigationGoal.POWERFUL
                )
            except Exception as e:
                print(f"[Navigator] Paradox failed: {e}")

        if "storm" in engines:
            try:
                results["storm"] = self._navigate_with_storm(
                    query, nouns, verbs, NavigationGoal.EXPLORATORY
                )
            except Exception as e:
                print(f"[Navigator] Storm failed: {e}")

        return results

    def navigate_best(self, query: str,
                      goal: Union[str, NavigationGoal] = "balanced") -> NavigationResult:
        """
        Run all engines and return the best result for the goal.

        This is more expensive but finds the optimal strategy.
        """
        if isinstance(goal, str):
            goal = NavigationGoal(goal.lower())

        # Run all engines
        results = self.navigate_multi(query)

        if not results:
            # Fallback to basic orbital
            return self.navigate(query, goal)

        # Score each result for the goal
        best_result = None
        best_score = -1

        for engine, result in results.items():
            score = result.quality.score_for_goal(goal)
            if score > best_score:
                best_score = score
                best_result = result

        # Update the goal in result
        if best_result:
            best_result.goal = goal

        return best_result

    # =========================================================================
    # Utility methods
    # =========================================================================

    def compare_strategies(self, query: str) -> Dict:
        """
        Compare all strategies on a query.

        Returns dict with scores for each goal × strategy combination.
        """
        results = self.navigate_multi(query)

        comparison = {
            'query': query,
            'strategies': {},
            'best_for_goal': {}
        }

        for engine, result in results.items():
            comparison['strategies'][engine] = {
                'concepts': result.concepts[:5],
                'quality': {
                    'resonance': result.quality.resonance,
                    'coherence': result.quality.coherence,
                    'depth': result.quality.depth,
                    'stability': result.quality.stability,
                    'power': result.quality.power
                },
                'scores': {}
            }

            for goal in NavigationGoal:
                score = result.quality.score_for_goal(goal)
                comparison['strategies'][engine]['scores'][goal.value] = score

        # Find best strategy for each goal
        for goal in NavigationGoal:
            best_engine = None
            best_score = -1
            for engine, result in results.items():
                score = result.quality.score_for_goal(goal)
                if score > best_score:
                    best_score = score
                    best_engine = engine
            comparison['best_for_goal'][goal.value] = best_engine

        return comparison

    def close(self):
        """Clean up all engines."""
        if self._orbital_laser:
            self._orbital_laser.close()
        if self._monte_carlo:
            self._monte_carlo.close()
        if self._paradox_detector:
            self._paradox_detector.close()
        if self._storm_logos:
            self._storm_logos.close()
        if self._supercritical_mc:
            self._supercritical_mc.close()
        if self._supercritical_paradox:
            self._supercritical_paradox.close()
        if self._graph:
            self._graph.close()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate unified semantic navigation."""
    print("=" * 70)
    print("SEMANTIC NAVIGATOR: Unified Physics Engine")
    print("=" * 70)

    nav = SemanticNavigator()

    queries = [
        ("What is consciousness?", "deep"),
        ("How do I fix my sleep schedule?", "grounded"),
        ("What is love?", "powerful"),
        ("What is the meaning of life?", "balanced"),
        ("What is wisdom?", "wisdom"),  # NEW: optimal C=0.1P balance
    ]

    try:
        for query, goal in queries:
            print(f"\n{'─' * 70}")
            print(f"Query: {query}")
            print(f"Goal:  {goal}")
            print("─" * 70)

            result = nav.navigate(query, goal)

            print(f"\nStrategy: {result.strategy}")
            print(f"Concepts: {result.concepts[:6]}")
            print(f"Quality:  {result.quality}")
            print(f"Goal Score: {result.quality.score_for_goal(result.goal):.2f}")

            if result.thesis and result.antithesis:
                print(f"\nParadox: {result.thesis} ↔ {result.antithesis}")
                print(f"Synthesis: {result.synthesis}")

        # Compare all strategies on one query
        print(f"\n{'=' * 70}")
        print("STRATEGY COMPARISON")
        print("=" * 70)

        comparison = nav.compare_strategies("What is wisdom?")

        print(f"\nQuery: {comparison['query']}")
        print(f"\nBest strategy for each goal:")
        for goal, engine in comparison['best_for_goal'].items():
            print(f"  {goal:12s} → {engine}")

    finally:
        nav.close()


if __name__ == "__main__":
    demo()
