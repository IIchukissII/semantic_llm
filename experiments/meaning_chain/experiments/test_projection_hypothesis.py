#!/usr/bin/env python3
"""
TEST: Projection Hierarchy Hypothesis
======================================

HYPOTHESIS:
    Nouns are "projections of projections" of transcendentals.

    TRANSCENDENTALS (A, S)     ← source
            │
            ↓ projection₁
    ADJECTIVES (j-vectors)     ← describe qualities
            │
            ↓ projection₂
    NOUNS (n, θ, r)            ← derived states

    Therefore:
        noun.(A, S) = weighted_average(adjective.(A, S))
        noun.θ = atan2(S_derived, A_derived)
        noun.r = sqrt(A_derived² + S_derived²)
        noun.n = f(entropy) -- already known

TEST:
    1. Load nouns with STORED (A, S, τ) from word vectors
    2. Load adjective profiles for each noun
    3. Compute DERIVED (A, S) as weighted average of adjective (A, S)
    4. Compare stored vs derived

    If r² > 0.8: coordinates ARE derived from structure
    If r² < 0.5: theory is fragmented, needs revision
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from scipy import stats as scipy_stats
from dataclasses import dataclass

# Add paths
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_EXPERIMENTS = _MEANING_CHAIN.parent
_SEMANTIC_LLM = _EXPERIMENTS.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.j_space import JSpace, PC1_AFFIRMATION, PC2_SACRED

J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']


@dataclass
class ComparisonResult:
    """Result of comparing stored vs derived coordinates."""
    word: str

    # Stored values (from word_vectors)
    A_stored: float
    S_stored: float
    tau_stored: float
    theta_stored: float
    r_stored: float

    # Derived values (from adjective weighted average)
    A_derived: float
    S_derived: float
    theta_derived: float
    r_derived: float

    # Errors
    A_error: float
    S_error: float
    theta_error: float
    r_error: float

    # Metadata
    n_adjectives: int
    coverage: float  # fraction of adjectives with vectors


def compute_stored_coordinates(word_data: Dict) -> Tuple[float, float, float, float, float]:
    """
    Compute (A, S, τ, θ, r) from stored word data.

    Returns:
        (A, S, τ, θ, r)
    """
    if not word_data.get('j'):
        return None

    j_5d = np.array([word_data['j'].get(d, 0) for d in J_DIMS])
    tau = word_data.get('tau', 3.0)

    # Project to (A, S)
    A = float(np.dot(j_5d, PC1_AFFIRMATION))
    S = float(np.dot(j_5d, PC2_SACRED))

    # Compute polar (θ, r)
    theta = np.arctan2(S, A)
    r = np.sqrt(A**2 + S**2)

    return A, S, tau, theta, r


def compute_derived_coordinates(adj_profile: Dict[str, int],
                                 adj_vectors: Dict[str, np.ndarray]) -> Tuple[float, float, float, float, float]:
    """
    Compute (A, S, θ, r, coverage) from weighted average of adjective vectors.

    Args:
        adj_profile: {adjective: count}
        adj_vectors: {adjective: j_5d}

    Returns:
        (A, S, θ, r, coverage)
    """
    if not adj_profile:
        return None

    total = sum(adj_profile.values())
    if total == 0:
        return None

    # Weighted average of adjective j-vectors
    j_centroid = np.zeros(5)
    weight_sum = 0.0
    found = 0

    for adj, count in adj_profile.items():
        if adj in adj_vectors:
            weight = count / total
            j_centroid += weight * adj_vectors[adj]
            weight_sum += weight
            found += 1

    if weight_sum < 0.01:  # Too few adjectives found
        return None

    # Normalize
    j_centroid /= weight_sum
    coverage = found / len(adj_profile)

    # Project to (A, S)
    A = float(np.dot(j_centroid, PC1_AFFIRMATION))
    S = float(np.dot(j_centroid, PC2_SACRED))

    # Compute polar (θ, r)
    theta = np.arctan2(S, A)
    r = np.sqrt(A**2 + S**2)

    return A, S, theta, r, coverage


def run_experiment(min_adjectives: int = 10, min_coverage: float = 0.3):
    """
    Run the projection hypothesis test.

    Args:
        min_adjectives: Minimum adjectives per noun to include
        min_coverage: Minimum fraction of adjectives with known vectors
    """
    print("=" * 70)
    print("PROJECTION HIERARCHY HYPOTHESIS TEST")
    print("=" * 70)
    print()
    print("HYPOTHESIS: noun.(A,S) = weighted_average(adjective.(A,S))")
    print()

    # Load data
    loader = DataLoader()

    print("[1] Loading word vectors...")
    word_vectors = loader.load_word_vectors()
    print(f"    Loaded {len(word_vectors)} word vectors")

    print("[2] Loading adjective profiles...")
    adj_profiles = loader.load_noun_adj_profiles()
    print(f"    Loaded profiles for {len(adj_profiles)} nouns")

    # Build adjective vector lookup (adjectives are words that appear in profiles)
    print("[3] Building adjective vector lookup...")
    unique_adjs = set()
    for profile in adj_profiles.values():
        unique_adjs.update(profile.keys())

    adj_vectors = {}
    for word in unique_adjs:
        if word in word_vectors and word_vectors[word].get('j'):
            j_5d = np.array([word_vectors[word]['j'].get(d, 0) for d in J_DIMS])
            adj_vectors[word] = j_5d

    print(f"    Found vectors for {len(adj_vectors)} / {len(unique_adjs)} unique adjectives")
    print(f"    Coverage: {100*len(adj_vectors)/len(unique_adjs):.1f}%")

    # Compare stored vs derived for each noun
    print("[4] Comparing stored vs derived coordinates...")
    print()

    results = []
    skipped_no_stored = 0
    skipped_no_derived = 0
    skipped_few_adj = 0
    skipped_low_coverage = 0

    for noun, adj_profile in adj_profiles.items():
        # Check minimum adjectives
        if len(adj_profile) < min_adjectives:
            skipped_few_adj += 1
            continue

        # Get stored coordinates
        if noun not in word_vectors:
            skipped_no_stored += 1
            continue

        stored = compute_stored_coordinates(word_vectors[noun])
        if stored is None:
            skipped_no_stored += 1
            continue

        A_stored, S_stored, tau_stored, theta_stored, r_stored = stored

        # Get derived coordinates
        derived = compute_derived_coordinates(adj_profile, adj_vectors)
        if derived is None:
            skipped_no_derived += 1
            continue

        A_derived, S_derived, theta_derived, r_derived, coverage = derived

        # Check coverage threshold
        if coverage < min_coverage:
            skipped_low_coverage += 1
            continue

        # Compute errors
        A_error = abs(A_stored - A_derived)
        S_error = abs(S_stored - S_derived)

        # Angle error (handle wraparound)
        theta_error = abs(theta_stored - theta_derived)
        if theta_error > np.pi:
            theta_error = 2*np.pi - theta_error

        r_error = abs(r_stored - r_derived)

        results.append(ComparisonResult(
            word=noun,
            A_stored=A_stored,
            S_stored=S_stored,
            tau_stored=tau_stored,
            theta_stored=theta_stored,
            r_stored=r_stored,
            A_derived=A_derived,
            S_derived=S_derived,
            theta_derived=theta_derived,
            r_derived=r_derived,
            A_error=A_error,
            S_error=S_error,
            theta_error=theta_error,
            r_error=r_error,
            n_adjectives=len(adj_profile),
            coverage=coverage
        ))

    print(f"    Analyzed: {len(results)} nouns")
    print(f"    Skipped (< {min_adjectives} adj): {skipped_few_adj}")
    print(f"    Skipped (no stored): {skipped_no_stored}")
    print(f"    Skipped (no derived): {skipped_no_derived}")
    print(f"    Skipped (coverage < {min_coverage}): {skipped_low_coverage}")
    print()

    if len(results) < 10:
        print("ERROR: Not enough data for statistical analysis")
        return

    # Statistical analysis
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Extract arrays
    A_stored_arr = np.array([r.A_stored for r in results])
    A_derived_arr = np.array([r.A_derived for r in results])
    S_stored_arr = np.array([r.S_stored for r in results])
    S_derived_arr = np.array([r.S_derived for r in results])
    theta_stored_arr = np.array([r.theta_stored for r in results])
    theta_derived_arr = np.array([r.theta_derived for r in results])
    r_stored_arr = np.array([r.r_stored for r in results])
    r_derived_arr = np.array([r.r_derived for r in results])

    # Correlations
    r_A, p_A = scipy_stats.pearsonr(A_stored_arr, A_derived_arr)
    r_S, p_S = scipy_stats.pearsonr(S_stored_arr, S_derived_arr)
    r_theta, p_theta = scipy_stats.pearsonr(theta_stored_arr, theta_derived_arr)
    r_r, p_r = scipy_stats.pearsonr(r_stored_arr, r_derived_arr)

    print("CORRELATION: stored vs derived")
    print("-" * 40)
    print(f"  A (Affirmation): r = {r_A:.4f}  (p = {p_A:.2e})  r² = {r_A**2:.4f}")
    print(f"  S (Sacred):      r = {r_S:.4f}  (p = {p_S:.2e})  r² = {r_S**2:.4f}")
    print(f"  θ (Phase):       r = {r_theta:.4f}  (p = {p_theta:.2e})  r² = {r_theta**2:.4f}")
    print(f"  r (Magnitude):   r = {r_r:.4f}  (p = {p_r:.2e})  r² = {r_r**2:.4f}")
    print()

    # Mean errors
    A_errors = np.array([r.A_error for r in results])
    S_errors = np.array([r.S_error for r in results])
    theta_errors = np.array([r.theta_error for r in results])
    r_errors = np.array([r.r_error for r in results])

    print("MEAN ABSOLUTE ERROR")
    print("-" * 40)
    print(f"  A (Affirmation): {np.mean(A_errors):.4f} ± {np.std(A_errors):.4f}")
    print(f"  S (Sacred):      {np.mean(S_errors):.4f} ± {np.std(S_errors):.4f}")
    print(f"  θ (Phase):       {np.degrees(np.mean(theta_errors)):.1f}° ± {np.degrees(np.std(theta_errors)):.1f}°")
    print(f"  r (Magnitude):   {np.mean(r_errors):.4f} ± {np.std(r_errors):.4f}")
    print()

    # Hypothesis verdict
    print("=" * 70)
    print("HYPOTHESIS VERDICT")
    print("=" * 70)

    avg_r2 = (r_A**2 + r_S**2) / 2

    if avg_r2 > 0.8:
        verdict = "STRONGLY CONFIRMED"
        explanation = "Noun coordinates ARE derived from adjective structure"
    elif avg_r2 > 0.6:
        verdict = "PARTIALLY CONFIRMED"
        explanation = "Significant relationship exists, but other factors contribute"
    elif avg_r2 > 0.4:
        verdict = "WEAK SUPPORT"
        explanation = "Some relationship, but theory needs revision"
    else:
        verdict = "NOT SUPPORTED"
        explanation = "Noun coordinates are NOT simply derived from adjectives"

    print()
    print(f"  Average r² (A, S): {avg_r2:.4f}")
    print(f"  Verdict: {verdict}")
    print(f"  {explanation}")
    print()

    # Best and worst examples
    print("=" * 70)
    print("EXAMPLES")
    print("=" * 70)

    # Sort by total error
    results_sorted = sorted(results, key=lambda r: r.A_error + r.S_error)

    print("\nBEST MATCHES (theory works):")
    print("-" * 40)
    for r in results_sorted[:5]:
        print(f"  {r.word:20s}  A_err={r.A_error:.3f}  S_err={r.S_error:.3f}  ({r.n_adjectives} adj)")

    print("\nWORST MATCHES (theory fails):")
    print("-" * 40)
    for r in results_sorted[-5:]:
        print(f"  {r.word:20s}  A_err={r.A_error:.3f}  S_err={r.S_error:.3f}  ({r.n_adjectives} adj)")

    # Relationship with tau
    print()
    print("=" * 70)
    print("RELATIONSHIP WITH τ (ABSTRACTION)")
    print("=" * 70)

    tau_arr = np.array([r.tau_stored for r in results])
    total_error = A_errors + S_errors

    r_tau_error, p_tau = scipy_stats.pearsonr(tau_arr, total_error)
    print(f"\n  Correlation (τ vs error): r = {r_tau_error:.4f}  (p = {p_tau:.2e})")

    if r_tau_error > 0.2:
        print("  → Higher τ (concrete nouns) have LARGER errors")
        print("  → Abstract nouns better match the projection model")
    elif r_tau_error < -0.2:
        print("  → Higher τ (concrete nouns) have SMALLER errors")
        print("  → Concrete nouns better match the projection model")
    else:
        print("  → No significant relationship between τ and error")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    N = {len(results)} nouns tested

    Key findings:
    1. A (Affirmation): r² = {r_A**2:.3f}
    2. S (Sacred):      r² = {r_S**2:.3f}
    3. Average:         r² = {avg_r2:.3f}

    Conclusion: {verdict}

    If r² < 0.5, the theory needs revision:
    - Stored (A, S) are NOT simply weighted averages of adjective (A, S)
    - There may be additional factors (context, etymology, usage patterns)
    - The projection chain may be more complex
    """)

    return results


if __name__ == "__main__":
    results = run_experiment(min_adjectives=10, min_coverage=0.3)
