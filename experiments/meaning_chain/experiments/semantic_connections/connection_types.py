"""
Connection Types: Series & Parallel Semantic Connections
=========================================================

Series: A → B → C (sequential, depth accumulates)
Parallel: A →[B,C,D]→ synthesis (simultaneous, coverage expands)

Electrical Analogy:
  Series:   R_total = R₁ + R₂ + R₃
  Parallel: 1/R_total = 1/R₁ + 1/R₂ + 1/R₃
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from enum import Enum


class ConnectionType(Enum):
    SERIES = "series"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass
class ConnectionResult:
    """Result of a connection operation"""

    connection_type: ConnectionType
    source: str
    concepts: List[str]

    # Quality metrics
    coherence: float          # Combined coherence
    tau_mean: float           # Average abstraction level
    tau_delta: float          # Change in abstraction
    coverage: float           # Unique concepts / total

    # Connection-specific
    paths: List[List[str]] = field(default_factory=list)
    synthesis: Optional[str] = None

    # Detailed metrics
    path_coherences: List[float] = field(default_factory=list)
    path_taus: List[float] = field(default_factory=list)

    # Impedance (if computed)
    impedance: Optional[complex] = None
    match_quality: Optional[float] = None

    def __repr__(self):
        return (f"ConnectionResult({self.connection_type.value}, "
                f"concepts={len(self.concepts)}, C={self.coherence:.3f}, "
                f"τ={self.tau_mean:.2f}, Δτ={self.tau_delta:+.2f})")


@dataclass
class SemanticPath:
    """A single path through semantic space"""

    concepts: List[str]
    coherence: float
    tau_start: float
    tau_end: float
    j_vectors: List[np.ndarray] = field(default_factory=list)

    @property
    def tau_delta(self) -> float:
        return self.tau_end - self.tau_start

    @property
    def length(self) -> int:
        return len(self.concepts)

    def j_mean(self) -> np.ndarray:
        """Average j-vector along path"""
        if not self.j_vectors:
            return np.zeros(5)
        return np.mean(self.j_vectors, axis=0)


class SeriesConnection:
    """
    Series connection: concepts in sequence

    Properties:
    - Coherence MULTIPLIES: C_total = C₁ × C₂ × C₃
    - τ ACCUMULATES: Δτ_total = Δτ₁ + Δτ₂ + Δτ₃
    - Impedance ADDS: Z_total = Z₁ + Z₂ + Z₃

    Use for: Deep exploration, following chains of meaning
    """

    def __init__(self, graph):
        self.graph = graph

    def connect(
        self,
        seed: str,
        depth: int = 5,
        direction: str = "ascending"  # ascending (higher τ) or descending
    ) -> ConnectionResult:
        """
        Navigate in series: seed → concept₁ → concept₂ → ...

        Args:
            seed: Starting concept
            depth: Number of steps
            direction: "ascending" (toward abstract) or "descending" (toward concrete)

        Returns:
            ConnectionResult with series-combined metrics
        """
        path = [seed]
        coherences = []
        taus = [self._get_tau(seed)]
        j_vectors = [self._get_j(seed)]

        current = seed
        for _ in range(depth):
            # Get neighbors sorted by τ
            neighbors = self._get_neighbors_with_tau(current)
            if not neighbors:
                break

            # Filter by direction
            if direction == "ascending":
                candidates = [(n, t) for n, t in neighbors if t > taus[-1]]
            else:
                candidates = [(n, t) for n, t in neighbors if t < taus[-1]]

            if not candidates:
                # No candidates in desired direction, take closest
                candidates = neighbors

            # Select by coherence with current j-vector
            best = self._select_by_coherence(candidates, j_vectors[-1])
            if best is None:
                break

            next_concept, next_tau = best
            path.append(next_concept)
            taus.append(next_tau)
            j_vectors.append(self._get_j(next_concept))

            # Compute step coherence
            step_coherence = self._compute_coherence(j_vectors[-2], j_vectors[-1])
            coherences.append(step_coherence)

            current = next_concept

        # Series: coherence multiplies
        total_coherence = np.prod(coherences) if coherences else 0.0

        return ConnectionResult(
            connection_type=ConnectionType.SERIES,
            source=seed,
            concepts=path,
            coherence=total_coherence,
            tau_mean=np.mean(taus),
            tau_delta=taus[-1] - taus[0] if len(taus) > 1 else 0.0,
            coverage=len(set(path)) / len(path) if path else 0.0,
            paths=[path],
            path_coherences=coherences,
            path_taus=taus,
        )

    def _get_tau(self, concept: str) -> float:
        """Get τ value for concept"""
        props = self.graph.get_concept(concept)
        return props.get('tau', 1.0) if props else 1.0

    def _get_j(self, concept: str) -> np.ndarray:
        """Get j-vector for concept"""
        props = self.graph.get_concept(concept)
        if props and 'j' in props:
            j = props['j']
            if isinstance(j, list):
                return np.array(j)
            return j
        return np.zeros(5)

    def _get_neighbors_with_tau(self, concept: str) -> List[Tuple[str, float]]:
        """Get neighbors with their τ values"""
        # Support both mock graph (get_neighbors) and real MeaningGraph (get_all_transitions)
        if hasattr(self.graph, 'get_all_transitions'):
            # Real MeaningGraph: returns [(verb, target, weight), ...]
            transitions = self.graph.get_all_transitions(concept, limit=30)
            result = []
            for verb, target, weight in transitions:
                tau = self._get_tau(target)
                result.append((target, tau))
            return result
        else:
            # Mock graph: returns [neighbor, ...]
            neighbors = self.graph.get_neighbors(concept)
            result = []
            for n in neighbors:
                tau = self._get_tau(n)
                result.append((n, tau))
            return result

    def _select_by_coherence(
        self,
        candidates: List[Tuple[str, float]],
        current_j: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """Select candidate with best j-vector coherence"""
        if not candidates:
            return None

        best = None
        best_coherence = -1

        for concept, tau in candidates:
            j = self._get_j(concept)
            coherence = self._compute_coherence(current_j, j)
            if coherence > best_coherence:
                best_coherence = coherence
                best = (concept, tau)

        return best

    def _compute_coherence(self, j1: np.ndarray, j2: np.ndarray) -> float:
        """Compute coherence between two j-vectors (cosine similarity)"""
        norm1 = np.linalg.norm(j1)
        norm2 = np.linalg.norm(j2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(j1, j2) / (norm1 * norm2))


class ParallelConnection:
    """
    Parallel connection: multiple paths simultaneously

    Properties:
    - Coherence AVERAGES: C_total = (C₁ + C₂ + C₃) / 3
    - τ AVERAGES: τ_total = (τ₁ + τ₂ + τ₃) / 3
    - Impedance: 1/Z_total = 1/Z₁ + 1/Z₂ + 1/Z₃

    Use for: Broad exploration, capturing multiple aspects
    """

    def __init__(self, graph, n_paths: int = 3):
        self.graph = graph
        self.n_paths = n_paths

    def connect(
        self,
        seed: str,
        depth: int = 3,
        diversity_threshold: float = 0.3
    ) -> ConnectionResult:
        """
        Navigate in parallel: seed →[path₁, path₂, path₃]→ synthesis

        Args:
            seed: Starting concept
            depth: Steps per path
            diversity_threshold: Min difference between paths (Jaccard)

        Returns:
            ConnectionResult with parallel-combined metrics
        """
        # Find diverse starting directions
        neighbors = self._get_diverse_neighbors(seed, self.n_paths)

        if len(neighbors) < 2:
            # Fall back to series if not enough diversity
            series = SeriesConnection(self.graph)
            return series.connect(seed, depth)

        # Navigate each path
        paths = []
        path_coherences = []
        path_taus = []
        all_concepts = set()

        for neighbor in neighbors:
            path = self._navigate_path(neighbor, depth)
            paths.append(path)
            path_coherences.append(path.coherence)
            path_taus.append(path.tau_end)
            all_concepts.update(path.concepts)

        # Parallel: coherence averages
        avg_coherence = np.mean(path_coherences) if path_coherences else 0.0
        avg_tau = np.mean(path_taus) if path_taus else 1.0

        # Find synthesis concept (central to all paths)
        synthesis = self._find_synthesis(paths)

        # Compute coverage
        total_concepts = sum(len(p.concepts) for p in paths)
        coverage = len(all_concepts) / total_concepts if total_concepts > 0 else 0.0

        return ConnectionResult(
            connection_type=ConnectionType.PARALLEL,
            source=seed,
            concepts=list(all_concepts),
            coherence=avg_coherence,
            tau_mean=avg_tau,
            tau_delta=avg_tau - self._get_tau(seed),
            coverage=coverage,
            paths=[p.concepts for p in paths],
            synthesis=synthesis,
            path_coherences=path_coherences,
            path_taus=path_taus,
        )

    def _get_tau(self, concept: str) -> float:
        """Get τ value for concept"""
        props = self.graph.get_concept(concept)
        return props.get('tau', 1.0) if props else 1.0

    def _get_j(self, concept: str) -> np.ndarray:
        """Get j-vector for concept"""
        props = self.graph.get_concept(concept)
        if props and 'j' in props:
            j = props['j']
            if isinstance(j, list):
                return np.array(j)
            return j
        return np.zeros(5)

    def _get_diverse_neighbors(self, seed: str, n: int) -> List[str]:
        """Get n diverse neighbors (low j-vector similarity)"""
        # Support both mock graph and real MeaningGraph
        if hasattr(self.graph, 'get_all_transitions'):
            transitions = self.graph.get_all_transitions(seed, limit=30)
            neighbors = [target for verb, target, weight in transitions]
        else:
            neighbors = self.graph.get_neighbors(seed)

        if len(neighbors) <= n:
            return neighbors

        # Get j-vectors
        j_vectors = [(n, self._get_j(n)) for n in neighbors]

        # Greedy selection for diversity
        selected = [j_vectors[0]]
        remaining = j_vectors[1:]

        while len(selected) < n and remaining:
            # Find most different from all selected
            best_idx = 0
            best_min_dist = -1

            for i, (_, j) in enumerate(remaining):
                min_dist = min(
                    1 - self._cosine_sim(j, sel_j)
                    for _, sel_j in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return [name for name, _ in selected]

    def _cosine_sim(self, j1: np.ndarray, j2: np.ndarray) -> float:
        """Cosine similarity between j-vectors"""
        norm1 = np.linalg.norm(j1)
        norm2 = np.linalg.norm(j2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(j1, j2) / (norm1 * norm2))

    def _navigate_path(self, start: str, depth: int) -> SemanticPath:
        """Navigate a single path from start"""
        series = SeriesConnection(self.graph)
        result = series.connect(start, depth)

        return SemanticPath(
            concepts=result.concepts,
            coherence=result.coherence,
            tau_start=result.path_taus[0] if result.path_taus else 1.0,
            tau_end=result.path_taus[-1] if result.path_taus else 1.0,
        )

    def _find_synthesis(self, paths: List[SemanticPath]) -> Optional[str]:
        """Find concept that synthesizes all paths (closest to j-centroid)"""
        if not paths:
            return None

        # Compute centroid j-vector
        j_means = [p.j_mean() for p in paths]
        centroid = np.mean(j_means, axis=0)

        # Find concept closest to centroid
        all_concepts = set()
        for p in paths:
            all_concepts.update(p.concepts)

        best_concept = None
        best_similarity = -1

        for concept in all_concepts:
            j = self._get_j(concept)
            sim = self._cosine_sim(j, centroid)
            if sim > best_similarity:
                best_similarity = sim
                best_concept = concept

        return best_concept


class HybridConnection:
    """
    Hybrid connection: parallel paths with series depth

    Architecture:
        seed → [parallel branches] → [series deepening each] → synthesis

    Combines:
    - Parallel's coverage
    - Series's depth
    """

    def __init__(self, graph, n_paths: int = 3, series_depth: int = 3):
        self.graph = graph
        self.n_paths = n_paths
        self.series_depth = series_depth
        self.parallel = ParallelConnection(graph, n_paths)
        self.series = SeriesConnection(graph)

    def connect(self, seed: str) -> ConnectionResult:
        """
        Hybrid navigation: parallel then series on each branch
        """
        # First: parallel expansion
        parallel_result = self.parallel.connect(seed, depth=1)

        # Then: series deepening on each branch
        deep_paths = []
        deep_coherences = []
        deep_taus = []
        all_concepts = set()

        for path in parallel_result.paths:
            if path:
                end_concept = path[-1]
                series_result = self.series.connect(end_concept, self.series_depth)

                full_path = path + series_result.concepts[1:]  # Avoid duplicate
                deep_paths.append(full_path)
                deep_coherences.append(series_result.coherence)
                deep_taus.append(series_result.tau_mean)
                all_concepts.update(full_path)

        # Combine metrics
        # Coherence: average of series coherences
        avg_coherence = np.mean(deep_coherences) if deep_coherences else 0.0
        avg_tau = np.mean(deep_taus) if deep_taus else 1.0

        total_concepts = sum(len(p) for p in deep_paths)
        coverage = len(all_concepts) / total_concepts if total_concepts > 0 else 0.0

        return ConnectionResult(
            connection_type=ConnectionType.HYBRID,
            source=seed,
            concepts=list(all_concepts),
            coherence=avg_coherence,
            tau_mean=avg_tau,
            tau_delta=avg_tau - self._get_tau(seed),
            coverage=coverage,
            paths=deep_paths,
            path_coherences=deep_coherences,
            path_taus=deep_taus,
        )

    def _get_tau(self, concept: str) -> float:
        props = self.graph.get_concept(concept)
        return props.get('tau', 1.0) if props else 1.0
