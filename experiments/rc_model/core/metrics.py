"""Metrics for RC-Model experiments.

Primary metrics (from THEORY.md):
    - raw_coherence: mean cos(Δθ) between consecutive input bonds
    - raw_bond_strength: mean cos(Δθ) × exp(-|Δn|/kT)

Secondary metrics:
    - σ_within: variance within sentences
    - smoothness: Σ|Q(t+1) - Q(t)| within sentences
    - autocorr(Q): autocorrelation (structure)
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from scipy import stats

if TYPE_CHECKING:
    from .semantic_rc import Trajectory

# From semantic physics
KT = math.exp(-1/5)  # ≈ 0.819


@dataclass
class TrajectoryMetrics:
    """Computed metrics for a trajectory."""
    # PRIMARY: Raw input coherence (before RC smoothing)
    raw_coherence: float      # mean cos(Δθ) of consecutive inputs
    raw_bond_strength: float  # mean cos(Δθ) × exp(-|Δn|/kT)

    # Within-sentence metrics
    sigma_within: float      # mean variance within sentences
    smoothness: float        # mean |dQ/dt| within sentences

    # Between-sentence metrics
    mean_jump_size: float    # mean |Q(end) - Q(start)| between sentences
    std_jump_size: float     # std of jump sizes

    # Global metrics
    sigma_global: float      # global variance of Q
    autocorr_lag1: float     # lag-1 autocorrelation
    autocorr_lag5: float     # lag-5 autocorrelation

    # Coverage
    n_bonds: int             # total bonds processed
    n_sentences: int         # number of sentences
    coverage: float          # fraction of words with coordinates

    def as_dict(self) -> dict:
        return {
            'raw_coherence': self.raw_coherence,
            'raw_bond_strength': self.raw_bond_strength,
            'sigma_within': self.sigma_within,
            'smoothness': self.smoothness,
            'mean_jump_size': self.mean_jump_size,
            'std_jump_size': self.std_jump_size,
            'sigma_global': self.sigma_global,
            'autocorr_lag1': self.autocorr_lag1,
            'autocorr_lag5': self.autocorr_lag5,
            'n_bonds': self.n_bonds,
            'n_sentences': self.n_sentences,
            'coverage': self.coverage,
        }


def _angular_diff(theta1: float, theta2: float) -> float:
    """Compute angular difference wrapped to [-π, π]."""
    diff = theta1 - theta2
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def compute_raw_coherence(input_coords: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Compute raw coherence and bond strength from input coordinates.

    Args:
        input_coords: List of (n, theta, r) tuples

    Returns:
        (mean_coherence, mean_bond_strength)
    """
    if len(input_coords) < 2:
        return (1.0, 1.0)

    coherences = []
    strengths = []

    for i in range(1, len(input_coords)):
        n1, theta1, _ = input_coords[i-1]
        n2, theta2, _ = input_coords[i]

        delta_theta = _angular_diff(theta1, theta2)
        delta_n = abs(n2 - n1)

        coh = math.cos(delta_theta)
        strength = coh * math.exp(-delta_n / KT)

        coherences.append(coh)
        strengths.append(strength)

    return (np.mean(coherences), np.mean(strengths))


def compute_metrics(trajectory) -> TrajectoryMetrics:
    """Compute all metrics for a trajectory.

    Args:
        trajectory: RC model trajectory

    Returns:
        TrajectoryMetrics with all computed values
    """
    Q = trajectory.Q_array
    boundaries = trajectory.sentence_boundaries

    if len(Q) < 2:
        return TrajectoryMetrics(
            raw_coherence=0, raw_bond_strength=0,
            sigma_within=0, smoothness=0,
            mean_jump_size=0, std_jump_size=0,
            sigma_global=0, autocorr_lag1=0, autocorr_lag5=0,
            n_bonds=len(Q), n_sentences=len(boundaries),
            coverage=0,
        )

    # Within-sentence metrics
    sigma_within_list = []
    smoothness_list = []

    start = 0
    for end in boundaries:
        if end - start >= 2:
            segment = Q[start:end]

            # Variance within sentence
            sigma_within_list.append(np.var(segment, axis=0).mean())

            # Smoothness (mean step size)
            diffs = np.diff(segment, axis=0)
            smoothness_list.append(np.linalg.norm(diffs, axis=1).mean())

        start = end

    sigma_within = np.mean(sigma_within_list) if sigma_within_list else 0
    smoothness = np.mean(smoothness_list) if smoothness_list else 0

    # Between-sentence metrics (jump sizes)
    jump_sizes = []
    for i in range(len(boundaries) - 1):
        end_idx = boundaries[i] - 1
        start_idx = boundaries[i]
        if start_idx < len(Q):
            jump = np.linalg.norm(Q[start_idx] - Q[end_idx])
            jump_sizes.append(jump)

    mean_jump = np.mean(jump_sizes) if jump_sizes else 0
    std_jump = np.std(jump_sizes) if jump_sizes else 0

    # Global metrics
    sigma_global = np.var(Q, axis=0).mean()

    # Autocorrelation (on magnitude r)
    r_vals = Q[:, 2]  # r is the third component
    autocorr_lag1 = _autocorrelation(r_vals, 1)
    autocorr_lag5 = _autocorrelation(r_vals, 5)

    # Coverage
    n_skipped = len(trajectory.skipped_words)
    n_total = len(Q) + n_skipped
    coverage = len(Q) / n_total if n_total > 0 else 0

    # Raw coherence from input coordinates (before RC smoothing)
    input_coords = []
    for state in trajectory.states:
        if state.input_coords is not None:
            input_coords.append(state.input_coords)

    if len(input_coords) >= 2:
        raw_coh, raw_str = compute_raw_coherence(input_coords)
    else:
        raw_coh, raw_str = 0.0, 0.0

    return TrajectoryMetrics(
        raw_coherence=raw_coh,
        raw_bond_strength=raw_str,
        sigma_within=sigma_within,
        smoothness=smoothness,
        mean_jump_size=mean_jump,
        std_jump_size=std_jump,
        sigma_global=sigma_global,
        autocorr_lag1=autocorr_lag1,
        autocorr_lag5=autocorr_lag5,
        n_bonds=len(Q),
        n_sentences=len(boundaries),
        coverage=coverage,
    )


def _autocorrelation(x: np.ndarray, lag: int) -> float:
    """Compute autocorrelation at given lag."""
    if len(x) <= lag:
        return 0
    x_mean = np.mean(x)
    x_centered = x - x_mean
    var = np.var(x)
    if var == 0:
        return 0
    autocov = np.mean(x_centered[:-lag] * x_centered[lag:])
    return autocov / var


def compare_metrics(
    original: TrajectoryMetrics,
    shuffled: TrajectoryMetrics,
) -> dict:
    """Compare original vs shuffled metrics.

    Args:
        original: Metrics from original text
        shuffled: Metrics from shuffled text

    Returns:
        Dict with ratios and differences
    """
    def safe_ratio(a, b):
        return a / b if b != 0 else float('inf') if a > 0 else 0

    return {
        'sigma_within_ratio': safe_ratio(original.sigma_within, shuffled.sigma_within),
        'smoothness_ratio': safe_ratio(original.smoothness, shuffled.smoothness),
        'autocorr_diff': original.autocorr_lag1 - shuffled.autocorr_lag1,
        'sigma_global_ratio': safe_ratio(original.sigma_global, shuffled.sigma_global),
        'original': original.as_dict(),
        'shuffled': shuffled.as_dict(),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return 0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_ttest(
    original_values: list[float],
    shuffled_values: list[float],
) -> dict:
    """Perform paired t-test between original and shuffled.

    Args:
        original_values: Metric values from original texts
        shuffled_values: Metric values from shuffled versions

    Returns:
        Dict with t-statistic, p-value, effect size
    """
    orig = np.array(original_values)
    shuf = np.array(shuffled_values)

    if len(orig) < 2:
        return {'t': 0, 'p': 1.0, 'cohens_d': 0, 'n': len(orig)}

    t_stat, p_value = stats.ttest_rel(orig, shuf)

    return {
        't': t_stat,
        'p': p_value,
        'cohens_d': cohens_d(orig, shuf),
        'n': len(orig),
        'mean_orig': np.mean(orig),
        'mean_shuf': np.mean(shuf),
    }
