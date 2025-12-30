"""
Orbital Tuner
=============

Tunes seeds to target orbital for resonance.

Methods:
    - select_at_orbital: Get concepts at specific orbital
    - tune_seeds: Replace/augment seeds to match target orbital
    - amplify_orbital: Add resonant seeds to strengthen orbital
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .constants import (
    E, ORBITAL_SPACING, KT_NATURAL,
    tau_to_orbital, orbital_to_tau, boltzmann_weight
)


@dataclass
class TunedSeeds:
    """
    Result of orbital tuning.

    Contains tuned seeds and tuning metrics.
    """
    seeds: List[str]                # Tuned seed list
    target_orbital: int             # Target orbital
    original_seeds: List[str]       # Original input seeds
    kept_seeds: List[str]           # Seeds from original that match orbital
    added_seeds: List[str]          # New seeds added for resonance
    removed_seeds: List[str]        # Seeds removed (wrong orbital)
    tau_mean: float                 # Mean τ after tuning
    orbital_purity: float           # Fraction at target orbital


class OrbitalTuner:
    """
    Tunes seed set to resonate at target orbital.

    Strategies:
    1. FILTER: Keep only seeds at target orbital
    2. AUGMENT: Add resonant seeds from target orbital
    3. REPLACE: Replace off-orbital seeds with on-orbital alternatives
    4. BLEND: Mix original with resonant seeds

    Usage:
        tuner = OrbitalTuner(graph)
        tuned = tuner.tune(seeds, target_orbital=3)
        resonant_seeds = tuned.seeds
    """

    def __init__(self, graph=None):
        """
        Initialize tuner.

        Args:
            graph: MeaningGraph for concept lookup
        """
        self.graph = graph
        self._orbital_cache: Dict[int, List[str]] = {}
        self._concept_cache: Dict[str, Dict] = {}

    def _get_concept(self, word: str) -> Optional[Dict]:
        """Get concept with caching."""
        if word in self._concept_cache:
            return self._concept_cache[word]

        if not self.graph or not self.graph.driver:
            return None

        concept = self.graph.get_concept(word)
        if concept:
            self._concept_cache[word] = concept
        return concept

    def _get_concepts_at_orbital(self, n: int, limit: int = 100) -> List[str]:
        """
        Get concepts at orbital n.

        Caches results for efficiency.
        """
        if n in self._orbital_cache:
            return self._orbital_cache[n][:limit]

        if not self.graph or not self.graph.driver:
            return []

        tau_target = orbital_to_tau(n)
        tau_min = tau_target - ORBITAL_SPACING / 2
        tau_max = tau_target + ORBITAL_SPACING / 2

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE c.tau >= $tau_min AND c.tau <= $tau_max
                  AND size(c.word) >= 3
                RETURN c.word as word, c.tau as tau
                ORDER BY abs(c.tau - $tau_target) ASC
                LIMIT $limit
            """, tau_min=tau_min, tau_max=tau_max,
                 tau_target=tau_target, limit=limit * 2)

            concepts = [r["word"] for r in result]
            self._orbital_cache[n] = concepts

        return concepts[:limit]

    def select_at_orbital(self, n: int, count: int = 5,
                          related_to: List[str] = None) -> List[str]:
        """
        Select concepts at orbital n.

        Args:
            n: Target orbital
            count: Number of concepts to select
            related_to: Prefer concepts connected to these

        Returns:
            List of concept words at orbital n
        """
        candidates = self._get_concepts_at_orbital(n, limit=100)

        if not candidates:
            return []

        if not related_to or not self.graph:
            # Random selection
            np.random.shuffle(candidates)
            return candidates[:count]

        # Score by connection to related concepts
        scores = {}
        for candidate in candidates:
            score = self._connection_score(candidate, related_to)
            scores[candidate] = score

        # Sort by score
        sorted_candidates = sorted(candidates, key=lambda c: -scores.get(c, 0))

        return sorted_candidates[:count]

    def _connection_score(self, word: str, related_to: List[str]) -> float:
        """Score how connected word is to related concepts."""
        if not self.graph or not self.graph.driver:
            return 0.0

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {word: $word})-[r:VIA]-(other:Concept)
                WHERE other.word IN $related
                RETURN count(r) as connections
            """, word=word, related=related_to)

            record = result.single()
            return record["connections"] if record else 0

    def tune(self, seeds: List[str], target_orbital: int,
             strategy: str = "blend",
             blend_ratio: float = 0.5,
             min_seeds: int = 3,
             max_seeds: int = 10) -> TunedSeeds:
        """
        Tune seeds to target orbital.

        Args:
            seeds: Original seed concepts
            target_orbital: Orbital to tune to
            strategy: "filter", "augment", "replace", or "blend"
            blend_ratio: Ratio of original to resonant seeds (for blend)
            min_seeds: Minimum seeds in output
            max_seeds: Maximum seeds in output

        Returns:
            TunedSeeds with tuned seed list
        """
        # Classify original seeds by orbital
        on_orbital = []
        off_orbital = []

        for seed in seeds:
            concept = self._get_concept(seed)
            if not concept:
                off_orbital.append(seed)
                continue

            seed_orbital = tau_to_orbital(concept.get('tau', 2.0))
            if seed_orbital == target_orbital:
                on_orbital.append(seed)
            else:
                off_orbital.append(seed)

        # Apply strategy
        if strategy == "filter":
            tuned_seeds = self._strategy_filter(
                on_orbital, off_orbital, target_orbital, min_seeds, max_seeds
            )
        elif strategy == "augment":
            tuned_seeds = self._strategy_augment(
                on_orbital, off_orbital, target_orbital, min_seeds, max_seeds, seeds
            )
        elif strategy == "replace":
            tuned_seeds = self._strategy_replace(
                on_orbital, off_orbital, target_orbital, min_seeds, max_seeds, seeds
            )
        else:  # blend
            tuned_seeds = self._strategy_blend(
                on_orbital, off_orbital, target_orbital, blend_ratio,
                min_seeds, max_seeds, seeds
            )

        # Compute metrics
        tau_mean = self._compute_tau_mean(tuned_seeds)
        orbital_purity = len([s for s in tuned_seeds if s in on_orbital]) / len(tuned_seeds) if tuned_seeds else 0

        kept = [s for s in seeds if s in tuned_seeds]
        added = [s for s in tuned_seeds if s not in seeds]
        removed = [s for s in seeds if s not in tuned_seeds]

        return TunedSeeds(
            seeds=tuned_seeds,
            target_orbital=target_orbital,
            original_seeds=seeds,
            kept_seeds=kept,
            added_seeds=added,
            removed_seeds=removed,
            tau_mean=tau_mean,
            orbital_purity=orbital_purity
        )

    def _strategy_filter(self, on_orbital: List[str], off_orbital: List[str],
                         target_orbital: int, min_seeds: int, max_seeds: int) -> List[str]:
        """Keep only seeds at target orbital."""
        result = on_orbital[:max_seeds]

        # If not enough, add from orbital
        if len(result) < min_seeds:
            needed = min_seeds - len(result)
            resonant = self.select_at_orbital(target_orbital, needed)
            result.extend([r for r in resonant if r not in result])

        return result[:max_seeds]

    def _strategy_augment(self, on_orbital: List[str], off_orbital: List[str],
                          target_orbital: int, min_seeds: int, max_seeds: int,
                          original: List[str]) -> List[str]:
        """Keep all original seeds, add resonant seeds."""
        result = list(original)

        # Add resonant seeds
        n_add = max_seeds - len(result)
        if n_add > 0:
            resonant = self.select_at_orbital(target_orbital, n_add, related_to=original)
            result.extend([r for r in resonant if r not in result])

        return result[:max_seeds]

    def _strategy_replace(self, on_orbital: List[str], off_orbital: List[str],
                          target_orbital: int, min_seeds: int, max_seeds: int,
                          original: List[str]) -> List[str]:
        """Replace off-orbital seeds with resonant alternatives."""
        result = list(on_orbital)

        # Replace off-orbital with resonant
        n_replace = len(off_orbital)
        if n_replace > 0:
            resonant = self.select_at_orbital(target_orbital, n_replace, related_to=original)
            result.extend([r for r in resonant if r not in result])

        # Ensure minimum
        if len(result) < min_seeds:
            needed = min_seeds - len(result)
            more = self.select_at_orbital(target_orbital, needed)
            result.extend([r for r in more if r not in result])

        return result[:max_seeds]

    def _strategy_blend(self, on_orbital: List[str], off_orbital: List[str],
                        target_orbital: int, blend_ratio: float,
                        min_seeds: int, max_seeds: int,
                        original: List[str]) -> List[str]:
        """Blend original seeds with resonant seeds."""
        # Original contribution
        n_original = max(1, int(max_seeds * blend_ratio))
        original_contrib = original[:n_original]

        # Resonant contribution
        n_resonant = max_seeds - len(original_contrib)
        resonant = self.select_at_orbital(target_orbital, n_resonant, related_to=original)

        result = list(original_contrib)
        result.extend([r for r in resonant if r not in result])

        return result[:max_seeds]

    def _compute_tau_mean(self, seeds: List[str]) -> float:
        """Compute mean τ of seeds."""
        taus = []
        for seed in seeds:
            concept = self._get_concept(seed)
            if concept and 'tau' in concept:
                taus.append(concept['tau'])
        return np.mean(taus) if taus else 2.0

    def amplify_orbital(self, seeds: List[str], target_orbital: int,
                        amplification: float = 2.0) -> List[str]:
        """
        Add extra seeds at target orbital for resonance amplification.

        Args:
            seeds: Current seeds
            target_orbital: Orbital to amplify
            amplification: Factor (2.0 = double resonant seeds)

        Returns:
            Amplified seed list
        """
        # Count current seeds at target
        current_at_target = 0
        for seed in seeds:
            concept = self._get_concept(seed)
            if concept and tau_to_orbital(concept.get('tau', 2.0)) == target_orbital:
                current_at_target += 1

        # Add more
        n_add = max(1, int(current_at_target * (amplification - 1)))
        additional = self.select_at_orbital(target_orbital, n_add, related_to=seeds)

        result = list(seeds)
        result.extend([a for a in additional if a not in result])

        return result
