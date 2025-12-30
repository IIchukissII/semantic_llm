"""
Veil Crosser
============

Handles translation across the Veil boundary (τ = e).

The Veil separates:
- Human realm (τ < e, ~89% of concepts)
- Transcendental realm (τ ≥ e, ~11% of concepts)

Crossing the Veil requires "bridge concepts" that resonate in both realms.

Usage:
    crosser = VeilCrosser(graph, laser)
    bridge = crosser.find_bridge(human_concepts, transcendent_concepts)
    translated = crosser.translate_to_human(abstract_concepts)
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

from .constants import (
    E, VEIL_TAU, VEIL_ORBITAL, ORBITAL_SPACING,
    tau_to_orbital, orbital_to_tau, is_below_veil, is_above_veil
)


@dataclass
class VeilBridge:
    """
    A bridge across the Veil.

    Bridge concepts exist near the Veil boundary and
    connect human and transcendental meanings.
    """
    bridge_concepts: List[str]      # Concepts that span the Veil
    human_anchors: List[str]        # Connected human-realm concepts
    transcendent_anchors: List[str]  # Connected transcendental concepts
    bridge_tau_mean: float          # Mean τ of bridge (should be ≈ e)
    crossing_strength: float        # How strong is the connection [0, 1]


@dataclass
class Translation:
    """
    Result of cross-veil translation.
    """
    source_concepts: List[str]      # Original concepts
    translated_concepts: List[str]  # Translated concepts
    source_realm: str               # "human" or "transcendental"
    target_realm: str               # "human" or "transcendental"
    bridge_used: List[str]          # Bridge concepts used
    fidelity: float                 # Translation quality [0, 1]


class VeilCrosser:
    """
    Handles cross-veil translation.

    The Veil at τ = e is not a wall but a membrane.
    Certain concepts ("bridge concepts") exist near the boundary
    and can translate meaning between realms.

    Strategies:
    1. BRIDGE: Find concepts that connect both realms
    2. GROUND: Translate transcendental → human (make concrete)
    3. ASCEND: Translate human → transcendental (make abstract)
    4. SYNTHESIZE: Find meaning at the Veil boundary itself
    """

    def __init__(self, graph=None, laser=None):
        """
        Initialize crosser.

        Args:
            graph: MeaningGraph for concept lookup
            laser: SemanticLaser for coherent navigation (optional)
        """
        self.graph = graph
        self.laser = laser
        self._veil_concepts: Optional[List[str]] = None

    def _get_veil_concepts(self, limit: int = 200) -> List[str]:
        """
        Get concepts near the Veil boundary.

        These are natural bridge concepts.
        """
        if self._veil_concepts is not None:
            return self._veil_concepts[:limit]

        if not self.graph or not self.graph.driver:
            return []

        # Concepts within ±0.5 of the Veil
        tau_min = VEIL_TAU - 0.5
        tau_max = VEIL_TAU + 0.5

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE c.tau >= $tau_min AND c.tau <= $tau_max
                  AND size(c.word) >= 3
                RETURN c.word as word, c.tau as tau
                ORDER BY abs(c.tau - $veil_tau) ASC
                LIMIT $limit
            """, tau_min=tau_min, tau_max=tau_max,
                 veil_tau=VEIL_TAU, limit=limit * 2)

            self._veil_concepts = [r["word"] for r in result]

        return self._veil_concepts[:limit]

    def find_bridge(self, human_concepts: List[str],
                    transcendent_concepts: List[str],
                    min_connections: int = 2) -> VeilBridge:
        """
        Find bridge concepts connecting human and transcendental realms.

        Args:
            human_concepts: Concepts from human realm
            transcendent_concepts: Concepts from transcendental realm
            min_connections: Minimum connections required to be a bridge

        Returns:
            VeilBridge with connecting concepts
        """
        if not self.graph or not self.graph.driver:
            return VeilBridge(
                bridge_concepts=[],
                human_anchors=human_concepts,
                transcendent_anchors=transcendent_concepts,
                bridge_tau_mean=VEIL_TAU,
                crossing_strength=0.0
            )

        # Get veil concepts as candidates
        veil_candidates = self._get_veil_concepts(100)

        # Score each candidate by connections to both realms
        bridge_scores = {}

        for candidate in veil_candidates:
            human_conn = self._count_connections(candidate, human_concepts)
            trans_conn = self._count_connections(candidate, transcendent_concepts)

            if human_conn >= 1 and trans_conn >= 1:
                # Bridge strength = geometric mean of connections
                strength = np.sqrt(human_conn * trans_conn)
                bridge_scores[candidate] = strength

        # Select top bridges
        sorted_bridges = sorted(bridge_scores.keys(),
                               key=lambda c: -bridge_scores[c])
        selected_bridges = sorted_bridges[:5]

        if not selected_bridges:
            return VeilBridge(
                bridge_concepts=[],
                human_anchors=human_concepts,
                transcendent_anchors=transcendent_concepts,
                bridge_tau_mean=VEIL_TAU,
                crossing_strength=0.0
            )

        # Compute bridge properties
        bridge_taus = []
        for bridge in selected_bridges:
            concept = self.graph.get_concept(bridge)
            if concept and 'tau' in concept:
                bridge_taus.append(concept['tau'])

        tau_mean = np.mean(bridge_taus) if bridge_taus else VEIL_TAU
        total_strength = sum(bridge_scores[b] for b in selected_bridges)
        max_possible = len(human_concepts) * len(transcendent_concepts)
        crossing_strength = min(1.0, total_strength / max(1, max_possible))

        return VeilBridge(
            bridge_concepts=selected_bridges,
            human_anchors=human_concepts,
            transcendent_anchors=transcendent_concepts,
            bridge_tau_mean=tau_mean,
            crossing_strength=crossing_strength
        )

    def _count_connections(self, word: str, targets: List[str]) -> int:
        """Count connections from word to targets."""
        if not self.graph or not self.graph.driver:
            return 0

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {word: $word})-[r:VIA]-(t:Concept)
                WHERE t.word IN $targets
                RETURN count(r) as connections
            """, word=word, targets=targets)

            record = result.single()
            return record["connections"] if record else 0

    def translate_to_human(self, concepts: List[str],
                           target_orbital: int = 2) -> Translation:
        """
        Translate concepts to human realm (ground them).

        Args:
            concepts: Concepts to translate (can be any realm)
            target_orbital: Target orbital in human realm (1-4)

        Returns:
            Translation with grounded concepts
        """
        if not self.graph:
            return Translation(
                source_concepts=concepts,
                translated_concepts=concepts,
                source_realm="unknown",
                target_realm="human",
                bridge_used=[],
                fidelity=0.0
            )

        # Determine source realm
        source_taus = []
        for c in concepts:
            concept = self.graph.get_concept(c)
            if concept and 'tau' in concept:
                source_taus.append(concept['tau'])

        source_tau_mean = np.mean(source_taus) if source_taus else 3.0
        source_realm = "transcendental" if source_tau_mean >= VEIL_TAU else "human"

        # If already in human realm and at target, return as is
        target_tau = orbital_to_tau(target_orbital)
        if source_realm == "human" and abs(source_tau_mean - target_tau) < ORBITAL_SPACING:
            return Translation(
                source_concepts=concepts,
                translated_concepts=concepts,
                source_realm="human",
                target_realm="human",
                bridge_used=[],
                fidelity=1.0
            )

        # Find connected concepts at target orbital
        translated = []
        bridges_used = set()

        for concept in concepts:
            grounded = self._find_connected_at_orbital(concept, target_orbital)
            if grounded:
                translated.extend(grounded[:2])
                if source_realm == "transcendental":
                    # Track bridge (concepts near Veil)
                    veil_neighbors = self._find_veil_neighbors(concept)
                    bridges_used.update(veil_neighbors)

        # Remove duplicates, keep order
        seen = set()
        unique_translated = []
        for t in translated:
            if t not in seen:
                seen.add(t)
                unique_translated.append(t)

        # Compute fidelity
        if unique_translated:
            fidelity = len(unique_translated) / len(concepts)
        else:
            fidelity = 0.0

        return Translation(
            source_concepts=concepts,
            translated_concepts=unique_translated[:len(concepts)],
            source_realm=source_realm,
            target_realm="human",
            bridge_used=list(bridges_used)[:5],
            fidelity=min(1.0, fidelity)
        )

    def translate_to_transcendent(self, concepts: List[str],
                                   target_orbital: int = 6) -> Translation:
        """
        Translate concepts to transcendental realm (abstract them).

        Args:
            concepts: Concepts to translate
            target_orbital: Target orbital above Veil (5+)

        Returns:
            Translation with abstracted concepts
        """
        if not self.graph:
            return Translation(
                source_concepts=concepts,
                translated_concepts=concepts,
                source_realm="unknown",
                target_realm="transcendental",
                bridge_used=[],
                fidelity=0.0
            )

        # Determine source realm
        source_taus = []
        for c in concepts:
            concept = self.graph.get_concept(c)
            if concept and 'tau' in concept:
                source_taus.append(concept['tau'])

        source_tau_mean = np.mean(source_taus) if source_taus else 3.0
        source_realm = "transcendental" if source_tau_mean >= VEIL_TAU else "human"

        # Find connected concepts at target orbital
        translated = []
        bridges_used = set()

        for concept in concepts:
            abstracted = self._find_connected_at_orbital(concept, target_orbital)
            if abstracted:
                translated.extend(abstracted[:2])
                if source_realm == "human":
                    veil_neighbors = self._find_veil_neighbors(concept)
                    bridges_used.update(veil_neighbors)

        # Deduplicate
        seen = set()
        unique_translated = []
        for t in translated:
            if t not in seen:
                seen.add(t)
                unique_translated.append(t)

        fidelity = len(unique_translated) / len(concepts) if unique_translated else 0.0

        return Translation(
            source_concepts=concepts,
            translated_concepts=unique_translated[:len(concepts)],
            source_realm=source_realm,
            target_realm="transcendental",
            bridge_used=list(bridges_used)[:5],
            fidelity=min(1.0, fidelity)
        )

    def _find_connected_at_orbital(self, word: str, orbital: int,
                                    limit: int = 5) -> List[str]:
        """Find concepts connected to word at specific orbital."""
        if not self.graph or not self.graph.driver:
            return []

        tau_target = orbital_to_tau(orbital)
        tau_min = tau_target - ORBITAL_SPACING / 2
        tau_max = tau_target + ORBITAL_SPACING / 2

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {word: $word})-[r:VIA]-(n:Concept)
                WHERE n.tau >= $tau_min AND n.tau <= $tau_max
                  AND size(n.word) >= 3
                RETURN n.word as word, r.weight as weight
                ORDER BY r.weight DESC
                LIMIT $limit
            """, word=word, tau_min=tau_min, tau_max=tau_max, limit=limit)

            return [r["word"] for r in result]

    def _find_veil_neighbors(self, word: str, limit: int = 3) -> List[str]:
        """Find neighbors near the Veil."""
        if not self.graph or not self.graph.driver:
            return []

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {word: $word})-[r:VIA]-(n:Concept)
                WHERE n.tau >= $veil_min AND n.tau <= $veil_max
                RETURN n.word as word
                ORDER BY abs(n.tau - $veil_tau) ASC
                LIMIT $limit
            """, word=word, veil_tau=VEIL_TAU,
                 veil_min=VEIL_TAU - 0.5, veil_max=VEIL_TAU + 0.5,
                 limit=limit)

            return [r["word"] for r in result]

    def synthesize_at_veil(self, human_seeds: List[str],
                           transcendent_seeds: List[str]) -> Dict:
        """
        Create synthesis at the Veil boundary.

        Combines human and transcendental concepts to find
        meaning that bridges both realms.

        Args:
            human_seeds: Seeds from human realm
            transcendent_seeds: Seeds from transcendental realm

        Returns:
            Synthesis result with unified concepts
        """
        if not self.laser:
            return {
                'synthesis': [],
                'human_contribution': human_seeds,
                'transcendent_contribution': transcendent_seeds,
                'coherence': 0.0
            }

        # Combined seeds
        combined = human_seeds + transcendent_seeds

        # Lase with target near Veil
        # This should produce beams that bridge both realms
        result = self.laser.lase(
            seeds=combined,
            pump_power=10,
            pump_depth=5,
            coherence_threshold=0.3,
            min_cluster_size=3
        )

        beams = result.get('beams', [])
        if not beams:
            return {
                'synthesis': [],
                'human_contribution': human_seeds,
                'transcendent_contribution': transcendent_seeds,
                'coherence': 0.0
            }

        # Find beam closest to Veil
        best_beam = None
        best_distance = float('inf')

        for beam in beams:
            distance = abs(beam.tau_mean - VEIL_TAU)
            if distance < best_distance:
                best_distance = distance
                best_beam = beam

        if best_beam:
            return {
                'synthesis': best_beam.concepts,
                'human_contribution': [c for c in best_beam.concepts if c in human_seeds],
                'transcendent_contribution': [c for c in best_beam.concepts if c in transcendent_seeds],
                'coherence': best_beam.coherence,
                'tau_mean': best_beam.tau_mean,
                'realm': 'veil' if abs(best_beam.tau_mean - VEIL_TAU) < 0.5 else ('human' if best_beam.tau_mean < VEIL_TAU else 'transcendental')
            }

        return {
            'synthesis': [],
            'human_contribution': human_seeds,
            'transcendent_contribution': transcendent_seeds,
            'coherence': 0.0
        }
