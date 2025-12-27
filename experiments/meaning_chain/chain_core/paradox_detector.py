"""
Paradox Detector: Find semantic tension in meaning space.

Uses Monte Carlo sampling to find attractors, then detects
paradoxes - concepts that are both stable AND opposing.

Paradox = semantic dynamite. Power comes from holding both poles.

"To gain your life, you must lose it"
"The only way out is through"
"Strength is vulnerability"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from chain_core.monte_carlo_renderer import MonteCarloRenderer, SemanticLandscape
from graph.meaning_graph import MeaningGraph


# J-vector dimensions
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


# Archetypal words - detected by literature, not hardcoded meaning
# These are words the corpus uses to encode archetypal energy
ARCHETYPAL_MARKERS = {'she', 'he', 'thee', 'thou', 'thy', 'i', 'we', 'they'}


@dataclass
class Paradox:
    """A detected semantic paradox - tension between opposites."""
    thesis: str           # First pole
    antithesis: str       # Opposite pole
    thesis_j: np.ndarray  # J-vector of thesis
    antithesis_j: np.ndarray  # J-vector of antithesis

    # Metrics
    tension: float        # How opposed (negative dot product)
    stability: float      # How stable both are (product of frequencies)
    power: float          # Combined power score

    # Type classification
    paradox_type: str = "conceptual"  # "archetypal" or "conceptual"

    # Synthesis
    synthesis_concepts: List[str] = field(default_factory=list)
    dominant_dimension: str = ""  # Which j-dimension has most tension

    def __repr__(self):
        return (f"Paradox({self.thesis} ↔ {self.antithesis}, "
                f"tension={self.tension:.2f}, power={self.power:.2f})")


@dataclass
class ParadoxLandscape:
    """Full paradox analysis of a question."""
    question: str
    paradoxes: List[Paradox]
    strongest: Optional[Paradox]

    # Double explosion: archetypal + conceptual
    archetypal_paradoxes: List[Paradox] = field(default_factory=list)
    conceptual_paradoxes: List[Paradox] = field(default_factory=list)
    strongest_archetypal: Optional[Paradox] = None
    strongest_conceptual: Optional[Paradox] = None

    # From MC sampling
    n_samples: int = 0
    total_attractors: int = 0
    coherence: float = 0.0

    # Synthesis
    synthesis_statement: str = ""
    double_explosion: str = ""  # Combined archetypal + conceptual statement


class ParadoxDetector:
    """
    Detect semantic paradoxes in meaning space.

    A paradox occurs when Monte Carlo sampling finds STABLE attractors
    that pull in OPPOSITE directions (opposing j-vectors).

    These are points of maximum meaning - where tension creates power.
    """

    def __init__(self, intent_strength: float = 0.3, n_samples: int = 30):
        self.mc_renderer = MonteCarloRenderer(
            intent_strength=intent_strength,
            n_samples=n_samples
        )
        self.graph = self.mc_renderer.laser.graph
        self.n_samples = n_samples

    def detect(self, question: str,
               min_tension: float = 0.2,
               top_k: int = 15) -> ParadoxLandscape:
        """
        Detect paradoxes in a question's semantic landscape.

        Args:
            question: The question to analyze
            min_tension: Minimum tension threshold (negative dot product)
            top_k: Number of top attractors to analyze

        Returns:
            ParadoxLandscape with detected paradoxes
        """
        print(f"[ParadoxDetector] Sampling semantic landscape...")

        # 1. Sample the landscape
        landscape = self.mc_renderer.sample_landscape(question)

        # 2. Get j-vectors for top attractors
        attractors_with_j = self._get_attractor_vectors(
            landscape.core_attractors[:top_k]
        )

        print(f"[ParadoxDetector] Found {len(attractors_with_j)} attractors with j-vectors")

        # 3. Find opposing pairs
        paradoxes = self._find_paradoxes(attractors_with_j, min_tension)

        print(f"[ParadoxDetector] Detected {len(paradoxes)} paradoxes")

        # 4. Find synthesis concepts for each paradox
        for p in paradoxes:
            p.synthesis_concepts = self._find_synthesis(p, attractors_with_j)
            p.dominant_dimension = self._get_dominant_tension_dim(p)

        # 5. Separate archetypal and conceptual
        archetypal = [p for p in paradoxes if p.paradox_type == "archetypal"]
        conceptual = [p for p in paradoxes if p.paradox_type == "conceptual"]

        # Sort each by power
        archetypal.sort(key=lambda p: -p.power)
        conceptual.sort(key=lambda p: -p.power)
        paradoxes.sort(key=lambda p: -p.power)

        strongest = paradoxes[0] if paradoxes else None
        strongest_arch = archetypal[0] if archetypal else None
        strongest_conc = conceptual[0] if conceptual else None

        print(f"[ParadoxDetector] Archetypal: {len(archetypal)}, Conceptual: {len(conceptual)}")

        # 6. Generate synthesis statements
        synthesis = ""
        if strongest:
            synthesis = self._generate_synthesis_statement(strongest)

        # 7. Generate double explosion if we have both types
        double_explosion = ""
        if strongest_arch and strongest_conc:
            double_explosion = self._generate_double_explosion(
                strongest_arch, strongest_conc
            )

        return ParadoxLandscape(
            question=question,
            paradoxes=paradoxes,
            strongest=strongest,
            archetypal_paradoxes=archetypal,
            conceptual_paradoxes=conceptual,
            strongest_archetypal=strongest_arch,
            strongest_conceptual=strongest_conc,
            n_samples=landscape.n_samples,
            total_attractors=len(landscape.core_attractors),
            coherence=landscape.coherence,
            synthesis_statement=synthesis,
            double_explosion=double_explosion
        )

    def _get_attractor_vectors(self, attractors: List[Tuple[str, int]],
                                min_j_magnitude: float = 0.3,
                                min_tau: float = 1.2) -> Dict[str, Dict]:
        """
        Get j-vectors for attractors from graph.

        Filtering based on semantic properties (not hardcoded lists):
        - min_j_magnitude: concepts with weak j-vectors are semantically neutral (noise)
        - min_tau: very low tau concepts are too concrete/functional

        Theory: meaningful concepts have direction (j) and abstraction (tau).
        """
        result = {}

        for word, count in attractors:
            concept = self.graph.get_concept(word)
            if not concept or not concept.get('j'):
                continue

            j = np.array(concept['j'])
            if len(j) != 5:
                continue

            # Semantic filtering based on properties
            j_magnitude = np.linalg.norm(j)
            tau = concept.get('tau', 2.0)

            # Skip semantically weak concepts (no direction = noise)
            if j_magnitude < min_j_magnitude:
                continue

            # Skip too-concrete concepts (functional words)
            if tau < min_tau:
                continue

            result[word] = {
                'j': j,
                'j_magnitude': j_magnitude,
                'g': concept.get('g', 0),
                'tau': tau,
                'count': count,
                'stability': count / self.n_samples
            }

        return result

    def _find_paradoxes(self, attractors: Dict[str, Dict],
                        min_tension: float) -> List[Paradox]:
        """Find pairs of attractors with opposing j-vectors."""
        paradoxes = []
        words = list(attractors.keys())

        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                a1 = attractors[w1]
                a2 = attractors[w2]

                # Compute dot product (negative = opposing)
                j1 = a1['j'] / (np.linalg.norm(a1['j']) + 1e-8)
                j2 = a2['j'] / (np.linalg.norm(a2['j']) + 1e-8)
                dot = np.dot(j1, j2)

                # Tension is negative dot product (opposing = high tension)
                tension = -dot

                if tension >= min_tension:
                    # Stability = geometric mean of frequencies
                    stability = np.sqrt(a1['stability'] * a2['stability'])

                    # Power = tension * stability (both oppose AND stable)
                    power = tension * stability * 10  # Scale for readability

                    # Classify: archetypal if either pole is archetypal marker
                    is_archetypal = (w1.lower() in ARCHETYPAL_MARKERS or
                                     w2.lower() in ARCHETYPAL_MARKERS)
                    paradox_type = "archetypal" if is_archetypal else "conceptual"

                    paradoxes.append(Paradox(
                        thesis=w1,
                        antithesis=w2,
                        thesis_j=a1['j'],
                        antithesis_j=a2['j'],
                        tension=tension,
                        stability=stability,
                        power=power,
                        paradox_type=paradox_type
                    ))

        return paradoxes

    def _find_synthesis(self, paradox: Paradox,
                        attractors: Dict[str, Dict]) -> List[str]:
        """
        Find concepts that bridge thesis and antithesis.

        Synthesis concepts have j-vectors between the two poles.
        """
        synthesis = []

        # Target direction: average of thesis and antithesis
        # (normalized, so we're looking for concepts in the "middle")
        midpoint = (paradox.thesis_j + paradox.antithesis_j) / 2
        midpoint_norm = midpoint / (np.linalg.norm(midpoint) + 1e-8)

        for word, data in attractors.items():
            if word in [paradox.thesis, paradox.antithesis]:
                continue

            j_norm = data['j'] / (np.linalg.norm(data['j']) + 1e-8)

            # How close to midpoint?
            similarity = np.dot(j_norm, midpoint_norm)

            # Also check it's not too aligned with either pole
            sim_thesis = np.dot(j_norm, paradox.thesis_j / (np.linalg.norm(paradox.thesis_j) + 1e-8))
            sim_anti = np.dot(j_norm, paradox.antithesis_j / (np.linalg.norm(paradox.antithesis_j) + 1e-8))

            # Good synthesis: close to midpoint, not too close to either pole
            if similarity > 0.3 and abs(sim_thesis - sim_anti) < 0.5:
                synthesis.append((word, similarity))

        # Sort by similarity to midpoint
        synthesis.sort(key=lambda x: -x[1])

        return [w for w, _ in synthesis[:5]]

    def _get_dominant_tension_dim(self, paradox: Paradox) -> str:
        """Find which j-dimension has the most tension."""
        diff = paradox.thesis_j - paradox.antithesis_j
        max_idx = np.argmax(np.abs(diff))

        sign = '+' if diff[max_idx] > 0 else '-'
        return f"{sign}{J_DIMS[max_idx]}"

    def _generate_synthesis_statement(self, paradox: Paradox) -> str:
        """Generate a powerful synthesis statement from the paradox."""
        diff = paradox.thesis_j - paradox.antithesis_j
        sorted_dims = np.argsort(-np.abs(diff))
        dim1 = J_DIMS[sorted_dims[0]]

        synthesis_words = paradox.synthesis_concepts[:3]
        synth_str = ', '.join(synthesis_words) if synthesis_words else 'truth'

        return (f"The tension between {paradox.thesis} and {paradox.antithesis} "
                f"creates {synth_str} — where {dim1} meets its shadow.")

    def _generate_double_explosion(self, archetypal: Paradox,
                                    conceptual: Paradox) -> str:
        """
        Generate statement from BOTH archetypal and conceptual paradoxes.

        Double explosion: the archetype carries the concept's tension.
        """
        # Archetypal poles
        arch_t = archetypal.thesis
        arch_a = archetypal.antithesis

        # Conceptual poles
        conc_t = conceptual.thesis
        conc_a = conceptual.antithesis

        # Find the archetypal dimension (what kind of archetype?)
        arch_diff = archetypal.thesis_j - archetypal.antithesis_j
        arch_dim = J_DIMS[np.argmax(np.abs(arch_diff))]

        # Find the conceptual dimension
        conc_diff = conceptual.thesis_j - conceptual.antithesis_j
        conc_dim = J_DIMS[np.argmax(np.abs(conc_diff))]

        # Double explosion: archetype embodies concept
        return (f"Through {arch_t}, {conc_t} speaks; through {arch_a}, {conc_a} answers. "
                f"The {arch_dim} of {arch_t}/{arch_a} carries the {conc_dim} of {conc_t}/{conc_a}.")

    def _generate_powerful_statements(self, paradox: Paradox) -> List[str]:
        """Generate paradoxical statements that hold both poles."""
        t = paradox.thesis
        a = paradox.antithesis
        s = paradox.synthesis_concepts[0] if paradox.synthesis_concepts else 'truth'

        statements = [
            f"In {t} lives the seed of {a}; in {a}, the echo of {t}.",
            f"True {t} contains {a} — not as opposite, but as completion.",
            f"Where {t} and {a} meet, {s} is born.",
            f"The {t} that excludes {a} is only half alive.",
            f"To know {t} fully is to embrace its shadow: {a}.",
        ]

        return statements

    def speak_powerfully(self, question: str) -> Dict:
        """
        Analyze question and generate powerful paradoxical speech.

        Returns the paradox analysis plus suggested powerful statements.
        Includes double explosion when both archetypal and conceptual paradoxes exist.
        """
        landscape = self.detect(question)

        if not landscape.strongest:
            return {
                'question': question,
                'paradox_found': False,
                'message': "No strong paradox detected - meaning is unified here."
            }

        result = {
            'question': question,
            'paradox_found': True,
        }

        # Archetypal layer
        if landscape.strongest_archetypal:
            p = landscape.strongest_archetypal
            result['archetypal'] = {
                'thesis': p.thesis,
                'antithesis': p.antithesis,
                'tension': p.tension,
                'power': p.power,
                'dimension': p.dominant_dimension
            }

        # Conceptual layer
        if landscape.strongest_conceptual:
            p = landscape.strongest_conceptual
            result['conceptual'] = {
                'thesis': p.thesis,
                'antithesis': p.antithesis,
                'tension': p.tension,
                'power': p.power,
                'dimension': p.dominant_dimension,
                'synthesis': p.synthesis_concepts
            }
            # Generate statements from conceptual paradox
            result['statements'] = self._generate_powerful_statements(p)

        # Double explosion
        if landscape.double_explosion:
            result['double_explosion'] = landscape.double_explosion

        # Overall strongest for backwards compatibility
        p = landscape.strongest
        result['thesis'] = p.thesis
        result['antithesis'] = p.antithesis
        result['tension'] = p.tension
        result['power'] = p.power
        result['dominant_dimension'] = p.dominant_dimension
        result['synthesis'] = p.synthesis_concepts

        result['all_paradoxes'] = [(px.thesis, px.antithesis, px.power, px.paradox_type)
                                   for px in landscape.paradoxes[:5]]

        return result

    def close(self):
        self.mc_renderer.close()


def demo():
    """Demonstrate paradox detection with double explosion."""
    detector = ParadoxDetector(n_samples=20)

    questions = [
        "What is love?",
        "How do I find meaning in suffering?",
        "What is the relationship between life and death?",
    ]

    for q in questions:
        print("\n" + "="*70)
        print(f"QUESTION: {q}")
        print("="*70)

        landscape = detector.detect(q)

        print(f"\nParadoxes found: {len(landscape.paradoxes)}")

        # Show archetypal layer
        if landscape.strongest_archetypal:
            p = landscape.strongest_archetypal
            print(f"\n[ARCHETYPAL LAYER]")
            print(f"  {p.thesis} ↔ {p.antithesis}")
            print(f"  Tension: {p.tension:.2f} | Power: {p.power:.2f}")
            print(f"  Dimension: {p.dominant_dimension}")

        # Show conceptual layer
        if landscape.strongest_conceptual:
            p = landscape.strongest_conceptual
            print(f"\n[CONCEPTUAL LAYER]")
            print(f"  {p.thesis} ↔ {p.antithesis}")
            print(f"  Tension: {p.tension:.2f} | Power: {p.power:.2f}")
            print(f"  Dimension: {p.dominant_dimension}")
            print(f"  Synthesis: {p.synthesis_concepts[:3]}")

        # Double explosion
        if landscape.double_explosion:
            print(f"\n[DOUBLE EXPLOSION]")
            print(f"  {landscape.double_explosion}")

        # Show synthesis statement
        if landscape.synthesis_statement:
            print(f"\n[SYNTHESIS]")
            print(f"  {landscape.synthesis_statement}")

    detector.close()


if __name__ == "__main__":
    demo()
