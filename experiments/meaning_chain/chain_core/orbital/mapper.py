"""
Intent-Orbital Mapper
=====================

Maps intent verbs to target orbitals.

IMPORTANT: Mappings are LEARNED from graph data, not hardcoded.

The mapper observes:
1. VerbOperator delta_tau values (how verbs move in τ-space)
2. Target concept distributions (where verbs lead)
3. Transition statistics

No speculation. Only observation.

Usage:
    mapper = IntentOrbitalMapper(graph)
    mapper.learn()  # Learn from graph data
    orbital = mapper.map(['understand', 'help'])
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .constants import (
    E, VEIL_TAU, VEIL_ORBITAL,
    tau_to_orbital, orbital_to_tau
)


@dataclass
class IntentMapping:
    """
    Result of intent-to-orbital mapping.
    """
    target_orbital: int         # Recommended orbital
    confidence: float           # Mapping confidence [0, 1]
    intent_profile: str         # Derived from data, not assumed
    verb_orbitals: Dict[str, int]  # Orbital for each verb
    realm: str                  # "human" or "transcendental"
    data_source: str            # "learned", "delta_tau", or "fallback"


class IntentOrbitalMapper:
    """
    Maps intent verbs to target orbitals.

    All mappings are learned from graph data:
    1. VerbOperator.delta_tau → direction of verb effect
    2. VIA edge targets → where verbs actually lead
    3. No hardcoded assumptions

    Usage:
        mapper = IntentOrbitalMapper(graph)
        mapper.learn()  # Must call first
        mapping = mapper.map(['understand', 'help'])
    """

    def __init__(self, graph):
        """
        Initialize mapper.

        Args:
            graph: MeaningGraph (required for learning)
        """
        self.graph = graph
        self._verb_orbital_cache: Dict[str, Tuple[int, float, str]] = {}
        self._learned = False
        self._global_mean_orbital: int = 3  # Updated during learning

    def learn(self) -> Dict:
        """
        Learn verb-orbital mappings from graph data.

        Analyzes:
        1. VerbOperator delta_tau values
        2. VIA edge target distributions
        3. Global statistics

        Returns:
            Learning statistics
        """
        if not self.graph or not self.graph.driver:
            return {'error': 'No graph connection'}

        stats = {
            'verbs_from_delta_tau': 0,
            'verbs_from_targets': 0,
            'total_verbs': 0
        }

        # Method 1: Learn from delta_tau on VerbOperator nodes
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (v:VerbOperator)
                WHERE v.delta_tau IS NOT NULL
                RETURN v.verb as verb, v.delta_tau as delta_tau
            """)

            for record in result:
                verb = record["verb"]
                delta_tau = record["delta_tau"]

                # Convert delta_tau to target orbital
                # delta_tau < 0: grounding verb → lower orbital
                # delta_tau > 0: ascending verb → higher orbital
                # Map range roughly [-1, 1] to orbitals [1, 7]
                orbital = self._delta_tau_to_orbital(delta_tau)
                confidence = 0.8  # High confidence from explicit data

                self._verb_orbital_cache[verb] = (orbital, confidence, 'delta_tau')
                stats['verbs_from_delta_tau'] += 1

        # Method 2: Learn from VIA edge target distributions
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH ()-[r:VIA]->(target:Concept)
                WHERE target.tau IS NOT NULL
                RETURN r.verb as verb, avg(target.tau) as avg_tau, count(*) as count
            """)

            for record in result:
                verb = record["verb"]
                avg_tau = record["avg_tau"]
                count = record["count"]

                # Skip if already learned from delta_tau
                if verb in self._verb_orbital_cache:
                    continue

                if count >= 3:  # Minimum observations
                    orbital = tau_to_orbital(avg_tau)
                    # Confidence based on sample size
                    confidence = min(0.7, 0.3 + 0.1 * np.log(count))

                    self._verb_orbital_cache[verb] = (orbital, confidence, 'targets')
                    stats['verbs_from_targets'] += 1

        # Compute global statistics
        if self._verb_orbital_cache:
            all_orbitals = [v[0] for v in self._verb_orbital_cache.values()]
            self._global_mean_orbital = int(round(np.mean(all_orbitals)))

        stats['total_verbs'] = len(self._verb_orbital_cache)
        stats['global_mean_orbital'] = self._global_mean_orbital
        self._learned = True

        print(f"[IntentOrbitalMapper] Learned {stats['total_verbs']} verb mappings "
              f"({stats['verbs_from_delta_tau']} from delta_tau, "
              f"{stats['verbs_from_targets']} from targets)")

        return stats

    def _delta_tau_to_orbital(self, delta_tau: float) -> int:
        """
        Convert delta_tau to target orbital.

        Based on observation: delta_tau indicates direction of movement.
        Negative delta_tau → grounding → lower orbital
        Positive delta_tau → ascending → higher orbital

        The mapping is empirically calibrated:
        - Strong grounding (delta_tau < -0.3): orbital 1-2
        - Mild grounding (-0.3 < delta_tau < 0): orbital 2-3
        - Neutral (delta_tau ≈ 0): orbital 3
        - Mild ascending (0 < delta_tau < 0.3): orbital 3-4
        - Strong ascending (delta_tau > 0.3): orbital 5+
        """
        if delta_tau < -0.5:
            return 1
        elif delta_tau < -0.2:
            return 2
        elif delta_tau < 0.1:
            return 3
        elif delta_tau < 0.3:
            return 4
        elif delta_tau < 0.5:
            return 5
        else:
            return 6

    def get_verb_orbital(self, verb: str) -> Tuple[int, float, str]:
        """
        Get orbital for a single verb.

        Args:
            verb: Verb to look up

        Returns:
            (orbital, confidence, source) tuple
        """
        verb_lower = verb.lower()

        # Check learned cache
        if verb_lower in self._verb_orbital_cache:
            return self._verb_orbital_cache[verb_lower]

        # Try to learn on-the-fly from graph
        if self.graph and self.graph.driver:
            orbital, conf, source = self._lookup_verb_live(verb_lower)
            if orbital is not None:
                self._verb_orbital_cache[verb_lower] = (orbital, conf, source)
                return orbital, conf, source

        # Fallback: global mean with low confidence
        return self._global_mean_orbital, 0.2, 'fallback'

    def _lookup_verb_live(self, verb: str) -> Tuple[Optional[int], float, str]:
        """Live lookup for unknown verb."""
        # Try delta_tau first
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (v:VerbOperator {verb: $verb})
                WHERE v.delta_tau IS NOT NULL
                RETURN v.delta_tau as delta_tau
            """, verb=verb)

            record = result.single()
            if record:
                delta_tau = record["delta_tau"]
                orbital = self._delta_tau_to_orbital(delta_tau)
                return orbital, 0.7, 'delta_tau_live'

        # Try target distribution
        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH ()-[r:VIA {verb: $verb}]->(target:Concept)
                WHERE target.tau IS NOT NULL
                RETURN avg(target.tau) as avg_tau, count(*) as count
            """, verb=verb)

            record = result.single()
            if record and record["count"] and record["count"] >= 2:
                avg_tau = record["avg_tau"]
                orbital = tau_to_orbital(avg_tau)
                return orbital, 0.5, 'targets_live'

        return None, 0.0, 'none'

    def map(self, verbs: List[str],
            weighting: str = "average") -> IntentMapping:
        """
        Map list of intent verbs to target orbital.

        Args:
            verbs: List of verbs from user query
            weighting: "average", "min", "max", or "dominant"

        Returns:
            IntentMapping with target orbital
        """
        if not verbs:
            return IntentMapping(
                target_orbital=self._global_mean_orbital,
                confidence=0.0,
                intent_profile="neutral",
                verb_orbitals={},
                realm="human",
                data_source="none"
            )

        # Get orbital for each verb
        verb_orbitals = {}
        confidences = []
        sources = []

        for verb in verbs:
            orbital, conf, source = self.get_verb_orbital(verb)
            verb_orbitals[verb] = orbital
            confidences.append(conf)
            sources.append(source)

        orbitals = list(verb_orbitals.values())

        # Compute target based on weighting
        if weighting == "min":
            target = min(orbitals)
        elif weighting == "max":
            target = max(orbitals)
        elif weighting == "dominant":
            from collections import Counter
            counts = Counter(orbitals)
            target = counts.most_common(1)[0][0]
        else:  # average
            target = int(round(np.mean(orbitals)))

        # Confidence (weighted by individual confidences)
        confidence = np.mean(confidences)

        # Intent profile (derived from orbital, not assumed)
        if target <= 2:
            profile = "grounding"
        elif target >= 5:
            profile = "ascending"
        else:
            profile = "balanced"

        # Data source summary
        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1
        primary_source = max(source_counts.keys(), key=lambda k: source_counts[k])

        # Realm
        realm = "human" if target < VEIL_ORBITAL else "transcendental"

        return IntentMapping(
            target_orbital=target,
            confidence=confidence,
            intent_profile=profile,
            verb_orbitals=verb_orbitals,
            realm=realm,
            data_source=primary_source
        )

    def suggest_orbital(self, verbs: List[str],
                        nouns: List[str] = None) -> Tuple[int, float, str]:
        """
        Suggest best orbital considering both verbs and nouns.

        All data from graph observation, no speculation.

        Args:
            verbs: Intent verbs
            nouns: Query nouns

        Returns:
            (orbital, confidence, reason) tuple
        """
        # Get verb-based orbital
        verb_mapping = self.map(verbs)
        verb_orbital = verb_mapping.target_orbital
        verb_conf = verb_mapping.confidence

        # If no nouns, use verb orbital
        if not nouns or not self.graph:
            return verb_orbital, verb_conf, f"verb-based ({verb_mapping.data_source})"

        # Get noun-based orbital from actual τ values
        noun_taus = []
        for noun in nouns:
            concept = self.graph.get_concept(noun)
            if concept and 'tau' in concept:
                noun_taus.append(concept['tau'])

        if not noun_taus:
            return verb_orbital, verb_conf, f"verb-based ({verb_mapping.data_source})"

        noun_tau_mean = np.mean(noun_taus)
        noun_orbital = tau_to_orbital(noun_tau_mean)
        noun_conf = 0.9  # High confidence for direct observation

        # Blend: weight by confidence
        total_conf = verb_conf + noun_conf
        if total_conf > 0:
            blended = int(round(
                (verb_orbital * verb_conf + noun_orbital * noun_conf) / total_conf
            ))
        else:
            blended = verb_orbital

        reason = f"blended (verb={verb_orbital}@{verb_conf:.1f}, noun={noun_orbital}@{noun_conf:.1f})"
        blended_conf = (verb_conf + noun_conf) / 2

        return blended, blended_conf, reason

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        if not self._verb_orbital_cache:
            return {'learned': False, 'count': 0}

        sources = {}
        orbitals = []

        for verb, (orbital, conf, source) in self._verb_orbital_cache.items():
            sources[source] = sources.get(source, 0) + 1
            orbitals.append(orbital)

        return {
            'learned': self._learned,
            'total_verbs': len(self._verb_orbital_cache),
            'sources': sources,
            'orbital_mean': np.mean(orbitals),
            'orbital_std': np.std(orbitals),
            'global_mean_orbital': self._global_mean_orbital
        }
