"""Genre Classifier based on A/S/τ boundary patterns.

THEORY (Three Axes = Three Functions):

  τ (Abstraction) = TENSION MARKER
      τ < 0.9  → "breathing" text, abstraction floats (dramatic mode)
      τ ≈ 1.0  → "holding" text, controlled discourse (balanced mode)

  A (Affirmation) = EMOTIONAL INTENSITY
      A > 1.15 → high emotional jumps at boundaries
      A ≈ 1.0  → measured, calm emotional flow

  S (Sacred) = CONCEPTUAL STRUCTURE
      S > 1.1  → philosophical/sacred shifts at boundaries
      S ≈ 1.0  → mundane, everyday concepts

THREE CLUSTERS (Genre Vector = τ, A, S):

  DRAMATIC     (0.88, 1.18, 1.11) - Gothic, Dostoevsky
               All axes at max. Tension + emotion + philosophy.

  IRONIC       (0.88, 1.14, 0.99) - Poe, Kafka
               Tension + emotion, but S ≈ 1.0 (horror of ordinary).
               "The monster is in the mundane."

  BALANCED     (1.01, 1.06, 1.06) - Austen, Plato
               τ holds steady. Controlled discourse.
               Reason over passion.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..core.semantic_rc_v2 import SemanticRCv2, TrajectoryV2
from ..core.bond_extractor import BondExtractor, TextBonds


@dataclass
class GenreSignature:
    """Semantic signature of a text based on boundary patterns."""
    # Boundary ratios (boundary / within-sentence)
    s_ratio: float      # Sacred axis jump ratio
    a_ratio: float      # Affirmation axis jump ratio
    tau_ratio: float    # Abstraction axis jump ratio

    # Autocorrelations (semantic memory)
    s_autocorr: float
    a_autocorr: float
    tau_autocorr: float

    # Derived
    n_bonds: int
    n_sentences: int

    @property
    def dominant_axis(self) -> str:
        """Which axis dominates boundaries?"""
        if self.a_ratio > self.s_ratio and self.a_ratio > self.tau_ratio:
            return 'A'  # Emotional/dramatic
        elif self.s_ratio > self.a_ratio and self.s_ratio > self.tau_ratio:
            return 'S'  # Conceptual/sacred
        else:
            return 'τ'  # Descriptive/neutral

    @property
    def narrative_style(self) -> str:
        """Classify narrative style based on (τ, A, S) vector.

        THREE MODES:
          DRAMATIC: τ < 0.92, A > 1.12, S > 1.05
                   All axes active. Gothic, Russian novels.

          IRONIC:   τ < 0.92, S < 1.02
                   Tension without philosophy. "Horror of ordinary."
                   Kafka, Poe - the mundane becomes terrifying.

          BALANCED: τ >= 0.98
                   Controlled discourse. Austen, Plato.
        """
        # τ is the primary tension marker
        if self.tau_ratio >= 0.98:
            return 'balanced'      # τ ≈ 1.0: controlled, analytical

        # τ < 0.92: "breathing" text, tension mode
        if self.tau_ratio < 0.92:
            # S distinguishes dramatic from ironic
            if self.s_ratio < 1.02:
                return 'ironic'    # Low S = horror of ordinary (Kafka)
            elif self.a_ratio > 1.12 and self.s_ratio > 1.05:
                return 'dramatic'  # All axes high (Dostoevsky)
            else:
                return 'dramatic'  # Default for tense texts

        # Middle ground
        return 'narrative'         # Standard storytelling

    @property
    def emotional_intensity(self) -> float:
        """Emotional intensity score (0-1)."""
        # A_ratio typically ranges 1.0-1.25
        score = (self.a_ratio - 1.0) / 0.25
        return max(0, min(1, score))

    def as_dict(self) -> dict:
        return {
            's_ratio': self.s_ratio,
            'a_ratio': self.a_ratio,
            'tau_ratio': self.tau_ratio,
            's_autocorr': self.s_autocorr,
            'a_autocorr': self.a_autocorr,
            'tau_autocorr': self.tau_autocorr,
            'dominant_axis': self.dominant_axis,
            'narrative_style': self.narrative_style,
            'emotional_intensity': self.emotional_intensity,
            'n_bonds': self.n_bonds,
            'n_sentences': self.n_sentences,
        }


class GenreClassifier:
    """Classify text genre by semantic boundary patterns.

    Uses distance-based classification to cluster centers:
      DRAMATIC:  (τ=0.88, A=1.18, S=1.11) - Gothic, Russian novels
      IRONIC:    (τ=0.88, A=1.14, S=0.99) - Poe, Kafka
      BALANCED:  (τ=1.01, A=1.06, S=1.06) - Austen, Plato
    """

    # Cluster centers from unsupervised analysis
    CLUSTER_CENTERS = {
        'dramatic': np.array([0.88, 1.18, 1.11]),   # τ, A, S
        'ironic':   np.array([0.88, 1.14, 0.99]),
        'balanced': np.array([1.01, 1.06, 1.06]),
    }

    # Weights for distance calculation (normalized by variance)
    WEIGHTS = np.array([5.0, 7.0, 7.0])  # τ, A, S

    def __init__(
        self,
        extractor: Optional[BondExtractor] = None,
        rc_model: Optional[SemanticRCv2] = None,
    ):
        self.extractor = extractor or BondExtractor()
        self.rc = rc_model or SemanticRCv2(decay=0.05, dt=0.5)

    def extract_signature(self, text: str) -> Optional[GenreSignature]:
        """Extract genre signature from text.

        Args:
            text: Input text (recommend 10K-60K chars)

        Returns:
            GenreSignature or None if too short
        """
        bonds = self.extractor.extract(text)
        self.rc.reset()
        traj = self.rc.process_text(bonds)

        if len(traj.states) < 100:
            return None

        return self._compute_signature(traj)

    def extract_signature_from_trajectory(
        self,
        traj: TrajectoryV2
    ) -> Optional[GenreSignature]:
        """Extract signature from existing trajectory."""
        if len(traj.states) < 100:
            return None
        return self._compute_signature(traj)

    def _compute_signature(self, traj: TrajectoryV2) -> GenreSignature:
        """Compute signature from trajectory."""
        Q_A, Q_S, Q_tau = traj.Q_A, traj.Q_S, traj.Q_tau
        boundaries = set(traj.sentence_boundaries)

        # Compute deltas
        delta_A = np.abs(np.diff(Q_A))
        delta_S = np.abs(np.diff(Q_S))
        delta_tau = np.abs(np.diff(Q_tau))

        # Mark boundaries
        is_boundary = np.zeros(len(delta_A), dtype=bool)
        for b in boundaries:
            if 0 < b <= len(delta_A):
                is_boundary[b-1] = True

        # Compute ratios (boundary / within)
        eps = 1e-6
        s_ratio = delta_S[is_boundary].mean() / (delta_S[~is_boundary].mean() + eps)
        a_ratio = delta_A[is_boundary].mean() / (delta_A[~is_boundary].mean() + eps)
        tau_ratio = delta_tau[is_boundary].mean() / (delta_tau[~is_boundary].mean() + eps)

        # Autocorrelations
        def autocorr(x):
            if len(x) < 2:
                return 0.0
            return float(np.corrcoef(x[:-1], x[1:])[0, 1])

        return GenreSignature(
            s_ratio=s_ratio,
            a_ratio=a_ratio,
            tau_ratio=tau_ratio,
            s_autocorr=autocorr(Q_S),
            a_autocorr=autocorr(Q_A),
            tau_autocorr=autocorr(Q_tau),
            n_bonds=len(traj.states),
            n_sentences=len(boundaries),
        )

    def _classify_by_distance(self, sig: GenreSignature) -> tuple[str, dict]:
        """Classify by distance to cluster centers.

        Returns:
            (predicted_style, distances_dict)
        """
        vec = np.array([sig.tau_ratio, sig.a_ratio, sig.s_ratio])

        distances = {}
        for name, center in self.CLUSTER_CENTERS.items():
            dist = np.sum(self.WEIGHTS * (vec - center)**2)
            distances[name] = float(dist)

        predicted = min(distances, key=distances.get)
        return predicted, distances

    def classify(self, text: str) -> dict:
        """Classify text and return full analysis.

        Uses distance-based classification to cluster centers.
        Accuracy: ~86% on held-out classic literature.

        Returns:
            Dict with signature, style, distances, and interpretation
        """
        sig = self.extract_signature(text)
        if sig is None:
            return {'error': 'Text too short (need 100+ bonds)'}

        predicted, distances = self._classify_by_distance(sig)

        return {
            'signature': sig.as_dict(),
            'style': predicted,  # Distance-based classification
            'style_heuristic': sig.narrative_style,  # Heuristic-based
            'distances': distances,
            'dominant_axis': sig.dominant_axis,
            'emotional_intensity': sig.emotional_intensity,
            'interpretation': self._interpret(sig, predicted),
        }

    def _interpret(self, sig: GenreSignature, predicted_style: str) -> str:
        """Generate human-readable interpretation."""
        lines = []

        # Narrative style (from distance-based classification)
        style_desc = {
            'dramatic': 'DRAMATIC: High tension, emotional intensity, philosophical shifts',
            'ironic': 'IRONIC: Tension without philosophy (horror of ordinary)',
            'balanced': 'BALANCED: Controlled discourse, measured emotional flow',
        }
        lines.append(style_desc.get(predicted_style, 'Unknown style'))

        # τ interpretation
        if sig.tau_ratio < 0.92:
            lines.append(f"τ = {sig.tau_ratio:.2f} → 'breathing' text (tension mode)")
        else:
            lines.append(f"τ = {sig.tau_ratio:.2f} → 'holding' text (controlled)")

        # A interpretation
        if sig.a_ratio > 1.15:
            lines.append(f"A = {sig.a_ratio:.2f} → high emotional intensity")
        elif sig.a_ratio > 1.08:
            lines.append(f"A = {sig.a_ratio:.2f} → moderate emotional engagement")
        else:
            lines.append(f"A = {sig.a_ratio:.2f} → calm, measured tone")

        # S interpretation
        if sig.s_ratio > 1.1:
            lines.append(f"S = {sig.s_ratio:.2f} → philosophical/sacred shifts")
        elif sig.s_ratio < 1.02:
            lines.append(f"S = {sig.s_ratio:.2f} → mundane concepts (horror of ordinary)")
        else:
            lines.append(f"S = {sig.s_ratio:.2f} → moderate conceptual variation")

        return '\n'.join(lines)


def print_genre_report(result: dict):
    """Print genre classification report."""
    print("=" * 60)
    print("GENRE CLASSIFICATION REPORT")
    print("=" * 60)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    sig = result['signature']

    print(f"\nNarrative Style: {result['style'].upper()}")
    print(f"Dominant Axis: {result['dominant_axis']}")
    print(f"Emotional Intensity: {result['emotional_intensity']:.1%}")

    print("\n" + "-" * 40)
    print("BOUNDARY PATTERNS")
    print("-" * 40)
    print(f"  A_ratio (emotional): {sig['a_ratio']:.3f}")
    print(f"  S_ratio (sacred):    {sig['s_ratio']:.3f}")
    print(f"  τ_ratio (carrier):   {sig['tau_ratio']:.3f}")

    print("\n" + "-" * 40)
    print("AUTOCORRELATION (semantic memory)")
    print("-" * 40)
    print(f"  τ (carrier):  {sig['tau_autocorr']:.3f}")
    print(f"  A (emotion):  {sig['a_autocorr']:.3f}")
    print(f"  S (sacred):   {sig['s_autocorr']:.3f}")

    print("\n" + "-" * 40)
    print("INTERPRETATION")
    print("-" * 40)
    print(result['interpretation'])

    print("\n" + "=" * 60)


def quick_test():
    """Quick test of genre classifier."""
    classifier = GenreClassifier()

    # Test text
    dramatic_text = """
    The dark shadows crept along the ancient walls. Her heart pounded
    with terrible fear. The creature's eyes gleamed with malevolent hunger.
    She ran through the endless corridors, gasping for breath.
    Death itself seemed to follow her desperate flight.
    The cold wind howled through broken windows. No escape remained.
    Her screams echoed through the empty castle halls.
    The monster drew closer with each passing moment.
    """

    balanced_text = """
    The philosophical implications of this argument are significant.
    We must consider the nature of truth itself. What constitutes
    knowledge remains a fundamental question. The ancients grappled
    with these same problems. Modern thinkers have proposed various
    solutions to these enduring puzzles. The relationship between
    mind and reality continues to fascinate scholars.
    """

    print("DRAMATIC TEXT:")
    result = classifier.classify(dramatic_text * 5)  # Repeat for length
    print_genre_report(result)

    print("\n" * 2)

    print("BALANCED TEXT:")
    result = classifier.classify(balanced_text * 5)
    print_genre_report(result)


if __name__ == "__main__":
    quick_test()
