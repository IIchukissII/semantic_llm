"""
Orbital Detector
================

Detects the natural orbital of a query or set of seeds.

The natural orbital is where the query "wants" to resonate.
Driving the laser at this orbital produces maximum coherence.

Methods:
    - detect_from_seeds: Analyze seed τ-values
    - detect_from_neighbors: Analyze connected concepts
    - detect_from_query: Full detection with intent awareness
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from .constants import (
    E, ORBITAL_SPACING, KT_NATURAL, VEIL_TAU,
    tau_to_orbital, orbital_to_tau, ORBITAL_POSITIONS
)


@dataclass
class OrbitalSignature:
    """
    The orbital signature of a query.

    Contains detected orbital and confidence metrics.
    """
    dominant_orbital: int           # Most likely orbital
    tau_mean: float                 # Mean τ of inputs
    tau_std: float                  # Spread in τ
    confidence: float               # Detection confidence [0, 1]
    orbital_distribution: Dict[int, float]  # Probability per orbital
    realm: str                      # "human" or "transcendental"
    resonance_quality: float        # Expected resonance at dominant orbital

    @property
    def tau_dominant(self) -> float:
        """τ-level of dominant orbital."""
        return orbital_to_tau(self.dominant_orbital)

    @property
    def is_focused(self) -> bool:
        """True if input is tightly focused (low spread)."""
        return self.tau_std < ORBITAL_SPACING

    @property
    def is_diffuse(self) -> bool:
        """True if input spans multiple orbitals."""
        return self.tau_std > 2 * ORBITAL_SPACING


class OrbitalDetector:
    """
    Detects natural orbital from query/seeds.

    The natural orbital is where:
    - Input concepts cluster
    - Neighbors are most connected
    - Resonance will be strongest

    Usage:
        detector = OrbitalDetector(graph)
        signature = detector.detect(seeds)
        target_orbital = signature.dominant_orbital
    """

    def __init__(self, graph=None):
        """
        Initialize detector.

        Args:
            graph: MeaningGraph for neighbor analysis (optional)
        """
        self.graph = graph
        self._concept_cache: Dict[str, Dict] = {}

    def _get_concept(self, word: str) -> Optional[Dict]:
        """Get concept properties with caching."""
        if word in self._concept_cache:
            return self._concept_cache[word]

        if not self.graph or not self.graph.driver:
            return None

        concept = self.graph.get_concept(word)
        if concept:
            self._concept_cache[word] = concept
        return concept

    def detect_from_seeds(self, seeds: List[str]) -> OrbitalSignature:
        """
        Detect orbital from seed τ-values.

        Simple method: analyze τ-distribution of input seeds.

        Args:
            seeds: Input concept words

        Returns:
            OrbitalSignature with detected orbital
        """
        taus = []
        for seed in seeds:
            concept = self._get_concept(seed)
            if concept and 'tau' in concept:
                taus.append(concept['tau'])

        if not taus:
            # Default to ground orbital
            return OrbitalSignature(
                dominant_orbital=1,
                tau_mean=1.37,
                tau_std=0.0,
                confidence=0.0,
                orbital_distribution={1: 1.0},
                realm="human",
                resonance_quality=0.0
            )

        tau_mean = np.mean(taus)
        tau_std = np.std(taus) if len(taus) > 1 else 0.0

        # Convert to orbitals
        orbitals = [tau_to_orbital(t) for t in taus]
        orbital_counts = Counter(orbitals)
        total = sum(orbital_counts.values())
        orbital_dist = {n: c / total for n, c in orbital_counts.items()}

        # Dominant orbital
        dominant = max(orbital_counts.keys(), key=lambda n: orbital_counts[n])

        # Confidence based on concentration
        dominant_fraction = orbital_counts[dominant] / total
        spread_penalty = min(1.0, tau_std / ORBITAL_SPACING)
        confidence = dominant_fraction * (1 - 0.5 * spread_penalty)

        # Resonance quality: how well-focused at dominant orbital
        resonance_quality = self._compute_resonance_quality(taus, dominant)

        # Realm
        realm = "human" if tau_mean < VEIL_TAU else "transcendental"

        return OrbitalSignature(
            dominant_orbital=dominant,
            tau_mean=tau_mean,
            tau_std=tau_std,
            confidence=confidence,
            orbital_distribution=orbital_dist,
            realm=realm,
            resonance_quality=resonance_quality
        )

    def detect_from_neighbors(self, seeds: List[str],
                               depth: int = 1) -> OrbitalSignature:
        """
        Detect orbital by analyzing neighbors.

        More sophisticated: looks at where the graph "wants" to go.

        Args:
            seeds: Starting concepts
            depth: How many hops to explore

        Returns:
            OrbitalSignature considering neighbor structure
        """
        if not self.graph or not self.graph.driver:
            return self.detect_from_seeds(seeds)

        # Collect τ-values from seeds and neighbors
        all_taus = []
        weights = []

        for seed in seeds:
            # Seed itself (weight 1.0)
            concept = self._get_concept(seed)
            if concept and 'tau' in concept:
                all_taus.append(concept['tau'])
                weights.append(1.0)

            # Neighbors (weight decays with depth)
            if depth >= 1:
                neighbors = self._get_neighbors(seed)
                for neighbor, weight in neighbors:
                    n_concept = self._get_concept(neighbor)
                    if n_concept and 'tau' in n_concept:
                        all_taus.append(n_concept['tau'])
                        weights.append(weight * 0.5)  # Decay

        if not all_taus:
            return self.detect_from_seeds(seeds)

        # Weighted statistics
        weights = np.array(weights)
        weights = weights / weights.sum()
        taus = np.array(all_taus)

        tau_mean = np.average(taus, weights=weights)
        tau_var = np.average((taus - tau_mean) ** 2, weights=weights)
        tau_std = np.sqrt(tau_var)

        # Weighted orbital distribution
        orbitals = [tau_to_orbital(t) for t in taus]
        orbital_weights = {}
        for n, w in zip(orbitals, weights):
            orbital_weights[n] = orbital_weights.get(n, 0) + w

        # Normalize
        total_w = sum(orbital_weights.values())
        orbital_dist = {n: w / total_w for n, w in orbital_weights.items()}

        # Dominant orbital (highest weight)
        dominant = max(orbital_weights.keys(), key=lambda n: orbital_weights[n])

        # Confidence
        dominant_weight = orbital_weights[dominant] / total_w
        confidence = dominant_weight * (1 - min(1.0, tau_std / ORBITAL_SPACING))

        # Resonance quality
        resonance_quality = self._compute_resonance_quality(taus, dominant, weights)

        # Realm
        realm = "human" if tau_mean < VEIL_TAU else "transcendental"

        return OrbitalSignature(
            dominant_orbital=dominant,
            tau_mean=tau_mean,
            tau_std=tau_std,
            confidence=confidence,
            orbital_distribution=orbital_dist,
            realm=realm,
            resonance_quality=resonance_quality
        )

    def detect(self, seeds: List[str],
               use_neighbors: bool = True,
               neighbor_depth: int = 1) -> OrbitalSignature:
        """
        Full orbital detection.

        Args:
            seeds: Input concepts
            use_neighbors: Whether to analyze neighbor structure
            neighbor_depth: How deep to explore

        Returns:
            OrbitalSignature with best detection
        """
        if use_neighbors and self.graph:
            return self.detect_from_neighbors(seeds, neighbor_depth)
        return self.detect_from_seeds(seeds)

    def suggest_orbital(self, seeds: List[str],
                        intent: str = None) -> Tuple[int, float]:
        """
        Suggest target orbital for resonance.

        Args:
            seeds: Input concepts
            intent: Optional intent hint ("ground", "transcend", "balance")

        Returns:
            (suggested_orbital, confidence)
        """
        signature = self.detect(seeds)

        if intent == "ground":
            # Force to human realm
            suggested = min(signature.dominant_orbital, 3)
        elif intent == "transcend":
            # Force above Veil
            suggested = max(signature.dominant_orbital, 5)
        elif intent == "balance":
            # Near Veil boundary
            suggested = 4  # Just below Veil
        else:
            # Use natural orbital
            suggested = signature.dominant_orbital

        return suggested, signature.confidence

    def _get_neighbors(self, word: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Get neighbors with weights."""
        if not self.graph or not self.graph.driver:
            return []

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {word: $word})-[r:VIA]->(n:Concept)
                RETURN n.word as neighbor, r.weight as weight
                ORDER BY r.weight DESC
                LIMIT $limit
            """, word=word, limit=limit)

            return [(r["neighbor"], r["weight"] or 1.0) for r in result]

    def _compute_resonance_quality(self, taus: List[float],
                                    target_orbital: int,
                                    weights: np.ndarray = None) -> float:
        """
        Compute expected resonance quality at target orbital.

        High quality = taus are tightly clustered at target.
        """
        target_tau = orbital_to_tau(target_orbital)
        taus = np.array(taus)

        if weights is None:
            weights = np.ones(len(taus)) / len(taus)

        # Distance from target orbital
        distances = np.abs(taus - target_tau)

        # Weighted mean distance
        mean_distance = np.average(distances, weights=weights)

        # Quality decays with distance (exponential)
        quality = np.exp(-mean_distance / ORBITAL_SPACING)

        return float(quality)
