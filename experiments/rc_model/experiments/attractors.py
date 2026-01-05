"""Experiment 7.3: Attractor Analysis

Do text trajectories converge to clusters in semantic space?

Hypothesis: Texts of similar genre/style should have similar endpoints (attractors).

Method:
1. Process many texts through RC model
2. Extract final states (n, θ, r) or (A, S)
3. Cluster endpoints
4. Analyze: do genres cluster together?
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from ..core.bond_extractor import BondExtractor, TextBonds
from ..core.semantic_rc import SemanticRC, Trajectory
from ..core.coord_loader import get_loader


@dataclass
class AttractorResult:
    """Result of attractor analysis."""
    # Endpoint statistics
    n_texts: int
    endpoints: np.ndarray        # (n_texts, 3) array of (n, θ, r)
    endpoints_AS: np.ndarray     # (n_texts, 2) array of (A, S)
    labels: list[str]            # Text labels/genres

    # Clustering results
    cluster_labels: np.ndarray   # Cluster assignment for each text
    n_clusters: int
    cluster_centers: np.ndarray  # Cluster centroids

    # Clustering quality
    silhouette_score: float
    inertia: float


def extract_endpoint(trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Extract endpoint from trajectory.

    Returns:
        (Q_final, AS_final) where Q = (n, θ, r) and AS = (A, S)
    """
    if len(trajectory.states) == 0:
        return np.zeros(3), np.zeros(2)

    final_state = trajectory.states[-1]
    Q = final_state.Q.copy()

    # Convert to Cartesian (A, S)
    A = Q[2] * math.cos(Q[1])  # r * cos(θ)
    S = Q[2] * math.sin(Q[1])  # r * sin(θ)

    return Q, np.array([A, S])


def extract_mean_state(trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean state from trajectory (more stable than endpoint).

    Returns:
        (Q_mean, AS_mean)
    """
    if len(trajectory.states) == 0:
        return np.zeros(3), np.zeros(2)

    Q_arr = trajectory.Q_array
    Q_mean = np.mean(Q_arr, axis=0)

    # Mean in (A, S) space
    A = Q_arr[:, 2] * np.cos(Q_arr[:, 1])
    S = Q_arr[:, 2] * np.sin(Q_arr[:, 1])
    AS_mean = np.array([np.mean(A), np.mean(S)])

    return Q_mean, AS_mean


def run_attractor_analysis(
    texts: list[str],
    labels: Optional[list[str]] = None,
    use_mean: bool = True,
    n_clusters: int = 3,
    extractor: Optional[BondExtractor] = None,
    rc_model: Optional[SemanticRC] = None,
) -> AttractorResult:
    """Analyze trajectory endpoints for clustering.

    Args:
        texts: List of texts to analyze
        labels: Optional labels for each text (e.g., genre)
        use_mean: Use mean state instead of endpoint (more stable)
        n_clusters: Number of clusters for K-means
        extractor: Bond extractor
        rc_model: RC model

    Returns:
        AttractorResult with clustering analysis
    """
    if extractor is None:
        extractor = BondExtractor()
    if rc_model is None:
        rc_model = SemanticRC()
    if labels is None:
        labels = [f"text_{i}" for i in range(len(texts))]

    endpoints = []
    endpoints_AS = []

    for text in texts:
        rc_model.reset()
        bonds = extractor.extract(text)
        trajectory = rc_model.process_text(bonds)

        if use_mean:
            Q, AS = extract_mean_state(trajectory)
        else:
            Q, AS = extract_endpoint(trajectory)

        endpoints.append(Q)
        endpoints_AS.append(AS)

    endpoints = np.array(endpoints)
    endpoints_AS = np.array(endpoints_AS)

    # Clustering on (A, S) space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(endpoints_AS)

    # Silhouette score (if enough samples)
    from sklearn.metrics import silhouette_score
    if len(texts) > n_clusters:
        sil_score = silhouette_score(endpoints_AS, cluster_labels)
    else:
        sil_score = 0.0

    return AttractorResult(
        n_texts=len(texts),
        endpoints=endpoints,
        endpoints_AS=endpoints_AS,
        labels=labels,
        cluster_labels=cluster_labels,
        n_clusters=n_clusters,
        cluster_centers=kmeans.cluster_centers_,
        silhouette_score=sil_score,
        inertia=kmeans.inertia_,
    )


def plot_attractors(
    result: AttractorResult,
    title: str = "Text Attractors in Semantic Space",
    figsize: tuple[int, int] = (10, 8),
    show_labels: bool = True,
) -> plt.Figure:
    """Plot attractor analysis results.

    Args:
        result: AttractorResult from run_attractor_analysis
        title: Plot title
        figsize: Figure size
        show_labels: Show text labels

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot colored by cluster
    scatter = ax.scatter(
        result.endpoints_AS[:, 0],
        result.endpoints_AS[:, 1],
        c=result.cluster_labels,
        cmap='tab10',
        s=100,
        alpha=0.7,
    )

    # Cluster centers
    ax.scatter(
        result.cluster_centers[:, 0],
        result.cluster_centers[:, 1],
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        linewidths=2,
        label='Cluster centers',
    )

    # Labels
    if show_labels:
        for i, label in enumerate(result.labels):
            ax.annotate(
                label[:10],  # Truncate long labels
                (result.endpoints_AS[i, 0], result.endpoints_AS[i, 1]),
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel('A (Affirmation)')
    ax.set_ylabel('S (Sacred)')
    ax.set_title(f"{title}\n(Silhouette: {result.silhouette_score:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_attractor_report(result: AttractorResult):
    """Print attractor analysis report."""
    print("=" * 60)
    print("ATTRACTOR ANALYSIS: Trajectory Endpoints")
    print("=" * 60)

    print(f"\nTexts analyzed: {result.n_texts}")
    print(f"Clusters found: {result.n_clusters}")

    print("\n" + "-" * 40)
    print("CLUSTERING QUALITY")
    print("-" * 40)
    print(f"  Silhouette score: {result.silhouette_score:.3f}")
    print(f"  Inertia:          {result.inertia:.3f}")

    if result.silhouette_score > 0.5:
        print("  ✓ Strong clustering structure")
    elif result.silhouette_score > 0.25:
        print("  ~ Moderate clustering structure")
    else:
        print("  ✗ Weak clustering structure")

    print("\n" + "-" * 40)
    print("CLUSTER COMPOSITION")
    print("-" * 40)

    for i in range(result.n_clusters):
        members = [result.labels[j] for j in range(len(result.labels))
                   if result.cluster_labels[j] == i]
        center = result.cluster_centers[i]
        print(f"\n  Cluster {i}: ({len(members)} texts)")
        print(f"    Center: A={center[0]:.3f}, S={center[1]:.3f}")
        print(f"    Members: {', '.join(members[:5])}" +
              (f"... (+{len(members)-5})" if len(members) > 5 else ""))

    print("\n" + "-" * 40)
    print("ENDPOINT STATISTICS")
    print("-" * 40)
    A_mean, S_mean = result.endpoints_AS.mean(axis=0)
    A_std, S_std = result.endpoints_AS.std(axis=0)
    print(f"  A: {A_mean:.3f} ± {A_std:.3f}")
    print(f"  S: {S_mean:.3f} ± {S_std:.3f}")

    # Are endpoints concentrated or spread?
    spread = np.sqrt(A_std**2 + S_std**2)
    mean_r = np.sqrt(A_mean**2 + S_mean**2)
    concentration = mean_r / (spread + 0.01)

    print(f"\n  Spread: {spread:.3f}")
    print(f"  Mean distance from origin: {mean_r:.3f}")
    print(f"  Concentration ratio: {concentration:.2f}")

    print("=" * 60)


# Sample texts for testing
SAMPLE_TEXTS = {
    'dark_fantasy': [
        """The ancient warrior stood before the dark gates of the cursed castle.
        His heavy sword gleamed with cold moonlight. The evil spirits whispered
        dark secrets through the empty halls. Death awaited within.""",

        """The old witch cackled in her dark tower. Black ravens circled overhead.
        The cursed forest stretched endlessly before the tired traveler.
        No hope remained in this forsaken land.""",
    ],
    'romance': [
        """Her beautiful eyes sparkled with gentle tears of joy. His warm hands
        held her soft face with tender love. The sweet spring flowers bloomed
        around their happy reunion.""",

        """The young lovers walked slowly through the bright garden.
        Their hearts beat with passionate hope. The soft music played
        as they danced beneath the silver stars.""",
    ],
    'science': [
        """The complex molecular structure demonstrated interesting properties.
        The experimental results confirmed theoretical predictions.
        Further analysis revealed significant correlations in the data.""",

        """The quantum mechanical system exhibited unusual behavior.
        Statistical analysis showed strong evidence for the hypothesis.
        The research team documented their careful observations.""",
    ],
}


def quick_test():
    """Run quick attractor analysis test."""
    texts = []
    labels = []

    for genre, genre_texts in SAMPLE_TEXTS.items():
        for i, text in enumerate(genre_texts):
            texts.append(text)
            labels.append(f"{genre}_{i}")

    print("Running attractor analysis...")
    result = run_attractor_analysis(texts, labels, n_clusters=3)
    print_attractor_report(result)

    return result


if __name__ == "__main__":
    quick_test()
