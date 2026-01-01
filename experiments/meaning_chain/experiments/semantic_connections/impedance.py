"""
Semantic Impedance: Complex Resistance in Meaning Space
========================================================

Z = R + jX = τ + jA

Where:
  R = τ (abstraction level, real resistance)
  X = A (Affirmation, imaginary reactance)

Coordinate System (Jan 2026):
    A = Affirmation (PC1, 83.3% variance) - main semantic direction
    S = Sacred (PC2, 11.7% variance) - orthogonal direction
    τ = Abstraction level [1=concrete, 6=abstract]

Impedance matching enables maximum meaning transfer.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class SemanticImpedance:
    """
    Complex impedance of a semantic concept

    Z = R + jX = τ + jA where:
      R = τ (abstraction resistance)
      X = A (Affirmation projection)
    """

    concept: str
    R: float          # Real part: τ (abstraction level)
    X: float          # Imaginary part: A (Affirmation)
    tau: float        # Original τ value
    j_vector: np.ndarray  # Original j-vector (for computing A)

    @property
    def Z(self) -> complex:
        """Complex impedance"""
        return complex(self.R, self.X)

    @property
    def magnitude(self) -> float:
        """Impedance magnitude |Z|"""
        return abs(self.Z)

    @property
    def phase(self) -> float:
        """Impedance phase angle (radians)"""
        return np.angle(self.Z)

    @property
    def phase_degrees(self) -> float:
        """Impedance phase angle (degrees)"""
        return np.degrees(self.phase)

    def conjugate(self) -> 'SemanticImpedance':
        """Complex conjugate Z* = R - jX"""
        return SemanticImpedance(
            concept=self.concept + "*",
            R=self.R,
            X=-self.X,
            tau=self.tau,
            j_vector=self.j_vector,
        )

    def __repr__(self):
        sign = "+" if self.X >= 0 else "-"
        return f"Z({self.concept}) = {self.R:.2f} {sign} j{abs(self.X):.2f}"


def compute_impedance(
    concept: str,
    tau: float,
    j_vector: np.ndarray,
    intent_direction: Optional[np.ndarray] = None
) -> SemanticImpedance:
    """
    Compute semantic impedance for a concept

    Args:
        concept: Concept name
        tau: Abstraction level [1, 6]
        j_vector: 5D semantic direction vector
        intent_direction: Direction of query intent (for projection)

    Returns:
        SemanticImpedance with Z = R + jX
    """
    # R = τ (abstraction as resistance)
    R = tau

    # X = j-vector projection onto intent direction
    if intent_direction is not None:
        # Project j onto intent direction
        intent_norm = np.linalg.norm(intent_direction)
        if intent_norm > 1e-6:
            X = float(np.dot(j_vector, intent_direction) / intent_norm)
        else:
            X = 0.0
    else:
        # Use j-vector magnitude as default
        X = float(np.linalg.norm(j_vector))

    return SemanticImpedance(
        concept=concept,
        R=R,
        X=X,
        tau=tau,
        j_vector=j_vector,
    )


def series_impedance(*impedances: SemanticImpedance) -> complex:
    """
    Series combination of impedances: Z_total = Z₁ + Z₂ + Z₃

    In series:
    - Impedances ADD
    - Current (meaning flow) is same through all
    - Voltage (semantic potential) divides
    """
    return sum(z.Z for z in impedances)


def parallel_impedance(*impedances: SemanticImpedance) -> complex:
    """
    Parallel combination of impedances: 1/Z_total = 1/Z₁ + 1/Z₂ + 1/Z₃

    In parallel:
    - Impedances combine reciprocally
    - Voltage (semantic potential) is same across all
    - Current (meaning flow) divides
    """
    if not impedances:
        return complex(0, 0)

    reciprocal_sum = sum(1.0 / z.Z for z in impedances if abs(z.Z) > 1e-6)

    if abs(reciprocal_sum) < 1e-6:
        return complex(float('inf'), 0)

    return 1.0 / reciprocal_sum


def reflection_coefficient(
    Z_load: complex,
    Z_source: complex
) -> complex:
    """
    Reflection coefficient Γ = (Z_load - Z_source) / (Z_load + Z_source)

    Measures impedance mismatch:
    - |Γ| → 0: Perfect match, all meaning transfers
    - |Γ| → 1: Total mismatch, meaning reflects back

    Args:
        Z_load: Impedance of destination (concept)
        Z_source: Impedance of source (query)

    Returns:
        Complex reflection coefficient
    """
    denominator = Z_load + Z_source
    if abs(denominator) < 1e-6:
        return complex(1, 0)  # Total reflection

    return (Z_load - Z_source) / denominator


def impedance_match_quality(
    Z_load: complex,
    Z_source: complex
) -> float:
    """
    Quality of impedance match [0, 1]

    Quality = 1 - |Γ|

    Higher quality = better meaning transfer
    """
    gamma = reflection_coefficient(Z_load, Z_source)
    return max(0.0, 1.0 - abs(gamma))


def standing_wave_ratio(gamma: complex) -> float:
    """
    Voltage Standing Wave Ratio (VSWR)

    VSWR = (1 + |Γ|) / (1 - |Γ|)

    - VSWR = 1: Perfect match
    - VSWR → ∞: Total mismatch
    """
    gamma_mag = abs(gamma)
    if gamma_mag >= 1.0:
        return float('inf')
    return (1 + gamma_mag) / (1 - gamma_mag)


def power_transfer_efficiency(
    Z_load: complex,
    Z_source: complex
) -> float:
    """
    Power transfer efficiency [0, 1]

    Maximum (100%) when Z_load = Z_source* (conjugate match)

    Efficiency = 1 - |Γ|²
    """
    gamma = reflection_coefficient(Z_load, Z_source)
    return max(0.0, 1.0 - abs(gamma) ** 2)


@dataclass
class ImpedanceMatchResult:
    """Result of impedance matching analysis"""

    Z_source: complex
    Z_load: complex
    gamma: complex
    match_quality: float
    vswr: float
    power_efficiency: float
    optimal_load: complex  # Conjugate match

    @property
    def is_good_match(self) -> bool:
        """Match quality > 0.7"""
        return self.match_quality > 0.7

    @property
    def is_perfect_match(self) -> bool:
        """Match quality > 0.95"""
        return self.match_quality > 0.95


def analyze_impedance_match(
    Z_source: complex,
    Z_load: complex
) -> ImpedanceMatchResult:
    """
    Complete impedance matching analysis

    Args:
        Z_source: Source impedance (query)
        Z_load: Load impedance (concept)

    Returns:
        ImpedanceMatchResult with all metrics
    """
    gamma = reflection_coefficient(Z_load, Z_source)

    return ImpedanceMatchResult(
        Z_source=Z_source,
        Z_load=Z_load,
        gamma=gamma,
        match_quality=impedance_match_quality(Z_load, Z_source),
        vswr=standing_wave_ratio(gamma),
        power_efficiency=power_transfer_efficiency(Z_load, Z_source),
        optimal_load=complex(Z_source.real, -Z_source.imag),  # Conjugate
    )


class ImpedanceNetwork:
    """
    Network of semantic impedances for complex connections
    """

    def __init__(self):
        self.impedances: dict[str, SemanticImpedance] = {}
        self.connections: list[Tuple[str, str, str]] = []  # (from, to, type)

    def add_concept(self, impedance: SemanticImpedance):
        """Add a concept to the network"""
        self.impedances[impedance.concept] = impedance

    def connect_series(self, concept1: str, concept2: str):
        """Connect two concepts in series"""
        self.connections.append((concept1, concept2, "series"))

    def connect_parallel(self, concept1: str, concept2: str):
        """Connect two concepts in parallel"""
        self.connections.append((concept1, concept2, "parallel"))

    def total_impedance(self) -> complex:
        """
        Compute total network impedance

        Currently supports simple series/parallel chains
        """
        if not self.impedances:
            return complex(0, 0)

        if len(self.impedances) == 1:
            return list(self.impedances.values())[0].Z

        # Group by connection type
        series_groups = []
        parallel_groups = []

        # Simple implementation: assume linear chain
        Z_total = complex(0, 0)

        for c1, c2, conn_type in self.connections:
            z1 = self.impedances.get(c1)
            z2 = self.impedances.get(c2)

            if z1 and z2:
                if conn_type == "series":
                    Z_total = series_impedance(z1, z2)
                else:
                    Z_total = parallel_impedance(z1, z2)

        return Z_total


# =============================================================================
# Query Impedance & Matching
# =============================================================================

@dataclass
class QueryImpedance:
    """
    Impedance of a query based on intent verbs.

    Z_query = R + jX where:
      R = target τ level (from verb Δτ properties)
      X = intent direction magnitude
    """

    query: str
    R: float              # Target abstraction level
    X: float              # Intent direction strength
    intent_direction: np.ndarray  # Combined verb j-vector
    verbs: list

    @property
    def Z(self) -> complex:
        return complex(self.R, self.X)

    @property
    def magnitude(self) -> float:
        return abs(self.Z)

    @property
    def target_tau(self) -> float:
        """Target τ level for concept matching"""
        return self.R

    def __repr__(self):
        sign = "+" if self.X >= 0 else "-"
        return f"Z_query = {self.R:.2f} {sign} j{abs(self.X):.2f}"


# Default verb → τ target mappings (from semantic physics)
VERB_TAU_TARGETS = {
    # Grounding verbs (low τ)
    "find": 1.4, "get": 1.3, "use": 1.4, "make": 1.5,
    "do": 1.3, "have": 1.4, "see": 1.5, "take": 1.4,

    # Medium verbs
    "know": 1.8, "think": 1.9, "feel": 1.7, "believe": 2.0,
    "learn": 1.8, "discover": 2.0, "experience": 1.7,

    # Ascending verbs (high τ)
    "understand": 2.3, "realize": 2.4, "contemplate": 2.6,
    "transcend": 3.0, "enlighten": 3.2, "awaken": 2.8,

    # Default
    "be": 1.7, "mean": 2.0,
}

# Default verb j-vector directions (simplified)
VERB_J_DIRECTIONS = {
    # Life-affirming
    "love": np.array([0.3, 0.8, 0.2, 0.5, 0.9]),
    "live": np.array([0.2, 0.9, 0.1, 0.4, 0.5]),
    "feel": np.array([0.4, 0.6, 0.2, 0.3, 0.7]),

    # Knowledge-seeking
    "know": np.array([0.5, 0.3, 0.4, 0.6, 0.3]),
    "understand": np.array([0.6, 0.4, 0.5, 0.7, 0.4]),
    "learn": np.array([0.5, 0.5, 0.3, 0.6, 0.4]),

    # Sacred/transcendent
    "believe": np.array([0.3, 0.2, 0.8, 0.5, 0.4]),
    "transcend": np.array([0.4, 0.3, 0.9, 0.4, 0.3]),

    # Default neutral
    "be": np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
}


def compute_query_impedance(
    query: str,
    verbs: list,
    graph=None
) -> QueryImpedance:
    """
    Compute impedance of a query from its intent verbs.

    Args:
        query: The query string
        verbs: List of intent verbs extracted from query
        graph: Optional MeaningGraph for verb properties

    Returns:
        QueryImpedance with Z = R + jX
    """
    if not verbs:
        # Default: medium abstraction, neutral direction
        return QueryImpedance(
            query=query,
            R=1.8,
            X=0.3,
            intent_direction=np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
            verbs=[]
        )

    # Compute target τ (R) from verb mappings
    tau_targets = []
    j_vectors = []

    for verb in verbs:
        verb_lower = verb.lower()

        # Get τ target
        if graph and hasattr(graph, 'get_verb_operator'):
            vo = graph.get_verb_operator(verb_lower)
            if vo and 'delta_tau' in vo:
                # Use verb's delta_tau to estimate target
                tau_targets.append(1.5 + vo['delta_tau'])
            elif verb_lower in VERB_TAU_TARGETS:
                tau_targets.append(VERB_TAU_TARGETS[verb_lower])
        elif verb_lower in VERB_TAU_TARGETS:
            tau_targets.append(VERB_TAU_TARGETS[verb_lower])

        # Get j-vector direction
        if graph and hasattr(graph, 'get_verb_operator'):
            vo = graph.get_verb_operator(verb_lower)
            if vo and 'j' in vo:
                j = vo['j']
                if isinstance(j, list):
                    j_vectors.append(np.array(j))
                else:
                    j_vectors.append(j)
            elif verb_lower in VERB_J_DIRECTIONS:
                j_vectors.append(VERB_J_DIRECTIONS[verb_lower])
        elif verb_lower in VERB_J_DIRECTIONS:
            j_vectors.append(VERB_J_DIRECTIONS[verb_lower])

    # Average τ target
    R = np.mean(tau_targets) if tau_targets else 1.8

    # Average j-vector direction
    if j_vectors:
        intent_direction = np.mean(j_vectors, axis=0)
    else:
        intent_direction = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

    # X = magnitude of intent direction
    X = float(np.linalg.norm(intent_direction))

    return QueryImpedance(
        query=query,
        R=R,
        X=X,
        intent_direction=intent_direction,
        verbs=verbs
    )


class ImpedanceMatcher:
    """
    Matches query impedance to concept impedances.

    Finds concepts with best Z-match for optimal meaning transfer.
    """

    def __init__(self, graph):
        self.graph = graph

    def compute_concept_impedance(
        self,
        concept: str,
        intent_direction: np.ndarray
    ) -> SemanticImpedance:
        """Compute impedance for a concept given query intent."""
        props = self.graph.get_concept(concept)
        if not props:
            return SemanticImpedance(
                concept=concept,
                R=1.5,
                X=0.0,
                tau=1.5,
                j_vector=np.zeros(5)
            )

        tau = props.get('tau', 1.5)
        j = props.get('j', [0, 0, 0, 0, 0])
        if isinstance(j, list):
            j = np.array(j)

        return compute_impedance(concept, tau, j, intent_direction)

    def match_concepts(
        self,
        query_z: QueryImpedance,
        candidates: list,
        top_k: int = 10
    ) -> list:
        """
        Find concepts that best match query impedance.

        Args:
            query_z: Query impedance
            candidates: List of candidate concept names
            top_k: Number of best matches to return

        Returns:
            List of (concept, match_quality, impedance) tuples
        """
        matches = []

        for concept in candidates:
            concept_z = self.compute_concept_impedance(
                concept, query_z.intent_direction
            )

            quality = impedance_match_quality(concept_z.Z, query_z.Z)
            matches.append((concept, quality, concept_z))

        # Sort by match quality (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:top_k]

    def find_resonant_concepts(
        self,
        query_z: QueryImpedance,
        seed_concepts: list,
        depth: int = 2,
        threshold: float = 0.5
    ) -> list:
        """
        Find concepts that resonate with query impedance.

        Explores from seeds, keeping only well-matched concepts.

        Args:
            query_z: Query impedance
            seed_concepts: Starting concepts
            depth: Exploration depth
            threshold: Minimum match quality

        Returns:
            List of resonant concepts with match quality
        """
        visited = set()
        resonant = []

        def explore(concept: str, current_depth: int):
            if concept in visited or current_depth > depth:
                return
            visited.add(concept)

            # Compute match
            concept_z = self.compute_concept_impedance(
                concept, query_z.intent_direction
            )
            quality = impedance_match_quality(concept_z.Z, query_z.Z)

            if quality >= threshold:
                resonant.append((concept, quality, concept_z))

                # Explore neighbors if good match
                if hasattr(self.graph, 'get_all_transitions'):
                    transitions = self.graph.get_all_transitions(concept, limit=10)
                    for verb, target, weight in transitions:
                        explore(target, current_depth + 1)

        # Start exploration from seeds
        for seed in seed_concepts:
            explore(seed, 0)

        # Sort by quality
        resonant.sort(key=lambda x: x[1], reverse=True)

        return resonant
