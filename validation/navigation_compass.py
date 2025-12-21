#!/usr/bin/env python3
"""
Navigation Compass Validation
=============================

FUNDAMENTAL VALIDATION: Compass-based navigation in semantic space.

Key result:
  - Compass navigation: mean Δg = +0.438
  - Random navigation: mean Δg = -0.017
  - t-statistic: 4.59, p < 0.001
  - COMPASS NAVIGATION IS STATISTICALLY SIGNIFICANT

Usage:
    python navigation_compass.py           # Run with CSV data
    python navigation_compass.py --from-db # Run with database
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats as scipy_stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "navigation"


def compute_goodness(j_vector: np.ndarray, j_good: np.ndarray) -> float:
    """Compute goodness as projection onto j_good direction."""
    return float(np.dot(j_vector, j_good))


def navigate_step(
    loader: DataLoader,
    current_j: np.ndarray,
    j_good: np.ndarray,
    toward_good: bool = True
) -> tuple:
    """
    Take one navigation step.

    Returns:
        (new_j, delta_g, verb_used)
    """
    verbs = loader.load_verb_operators()

    best_delta = -np.inf if toward_good else np.inf
    best_j = current_j
    best_verb = None

    current_g = compute_goodness(current_j, j_good)

    for verb, data in list(verbs.items())[:50]:  # Sample for speed
        # Verb vector modifies j-space
        verb_j = np.array([
            data['vector']['beauty'],
            data['vector']['life'],
            data['vector']['sacred'],
            data['vector']['good'],
            data['vector']['love']
        ])

        # Apply verb as additive transition
        new_j = current_j + 0.1 * verb_j
        new_j = np.clip(new_j, -1, 1)

        new_g = compute_goodness(new_j, j_good)
        delta_g = new_g - current_g

        if toward_good:
            if delta_g > best_delta:
                best_delta = delta_g
                best_j = new_j
                best_verb = verb
        else:
            if delta_g < best_delta:
                best_delta = delta_g
                best_j = new_j
                best_verb = verb

    return best_j, best_delta, best_verb


def run_trajectory(
    loader: DataLoader,
    start_noun: str,
    j_good: np.ndarray,
    toward_good: bool,
    n_steps: int = 5
) -> dict:
    """Run a navigation trajectory."""
    vectors = loader.load_word_vectors()

    if start_noun not in vectors or vectors[start_noun]['j'] is None:
        return None

    j_data = vectors[start_noun]['j']
    current_j = np.array([
        j_data['beauty'],
        j_data['life'],
        j_data['sacred'],
        j_data['good'],
        j_data['love']
    ])

    start_g = compute_goodness(current_j, j_good)
    total_delta_g = 0
    steps = []

    for i in range(n_steps):
        new_j, delta_g, verb = navigate_step(loader, current_j, j_good, toward_good)
        total_delta_g += delta_g
        steps.append({'verb': verb, 'delta_g': delta_g})
        current_j = new_j

    end_g = compute_goodness(current_j, j_good)

    return {
        'start_noun': start_noun,
        'start_g': start_g,
        'end_g': end_g,
        'total_delta_g': total_delta_g,
        'toward_good': toward_good,
        'n_steps': n_steps,
        'steps': steps
    }


def run_random_trajectory(
    loader: DataLoader,
    start_noun: str,
    j_good: np.ndarray,
    n_steps: int = 5
) -> dict:
    """Run a random navigation trajectory (no compass)."""
    vectors = loader.load_word_vectors()
    verbs = loader.load_verb_operators()

    if start_noun not in vectors or vectors[start_noun]['j'] is None:
        return None

    j_data = vectors[start_noun]['j']
    current_j = np.array([
        j_data['beauty'],
        j_data['life'],
        j_data['sacred'],
        j_data['good'],
        j_data['love']
    ])

    start_g = compute_goodness(current_j, j_good)
    total_delta_g = 0
    steps = []

    verb_list = list(verbs.keys())

    for i in range(n_steps):
        # Random verb selection
        verb = np.random.choice(verb_list[:100])
        data = verbs[verb]

        verb_j = np.array([
            data['vector']['beauty'],
            data['vector']['life'],
            data['vector']['sacred'],
            data['vector']['good'],
            data['vector']['love']
        ])

        old_g = compute_goodness(current_j, j_good)
        current_j = current_j + 0.1 * verb_j
        current_j = np.clip(current_j, -1, 1)
        new_g = compute_goodness(current_j, j_good)

        delta_g = new_g - old_g
        total_delta_g += delta_g
        steps.append({'verb': verb, 'delta_g': delta_g})

    end_g = compute_goodness(current_j, j_good)

    return {
        'start_noun': start_noun,
        'start_g': start_g,
        'end_g': end_g,
        'total_delta_g': total_delta_g,
        'random': True,
        'n_steps': n_steps,
        'steps': steps
    }


def main():
    parser = argparse.ArgumentParser(description="Navigation Compass Validation")
    parser.add_argument('--from-db', action='store_true',
                        help='Load from database instead of CSV')
    parser.add_argument('--n-trajectories', type=int, default=50,
                        help='Number of trajectories per condition')
    args = parser.parse_args()

    print("=" * 60)
    print("NAVIGATION COMPASS VALIDATION")
    print("=" * 60)
    print("""
Testing: Does compass-based navigation outperform random?

  Compass: Select verbs that maximize Δg (goodness change)
  Random:  Select verbs randomly

  If compass works, mean(Δg_compass) >> mean(Δg_random)
""")

    # Load data
    loader = DataLoader()
    vectors = loader.load_word_vectors()
    verbs = loader.load_verb_operators()

    if not vectors:
        print("\nERROR: No word vectors available.")
        print("Run 'python scripts/export_data.py --vectors' first.")
        return

    if not verbs:
        print("\nERROR: No verb operators available.")
        print("Run 'python scripts/export_data.py --verbs' first.")
        return

    print(f"\nLoaded {len(vectors)} words, {len(verbs)} verbs")

    # Get j_good direction
    j_good = loader.get_j_good()
    print(f"j_good = {j_good}")

    # Get test nouns
    test_nouns = [
        'war', 'peace', 'love', 'hate', 'life', 'death',
        'beauty', 'evil', 'truth', 'fear', 'hope', 'anger',
        'joy', 'pain', 'light', 'dark', 'friend', 'enemy'
    ]

    # Filter to available nouns
    available_nouns = [n for n in test_nouns if n in vectors and vectors[n]['j'] is not None]
    print(f"Test nouns: {available_nouns}")

    # Run trajectories
    print(f"\nRunning {args.n_trajectories} trajectories per condition...")

    compass_results = []
    random_results = []

    np.random.seed(42)  # Reproducibility

    for i in range(args.n_trajectories):
        noun = np.random.choice(available_nouns)

        # Compass toward good
        result = run_trajectory(loader, noun, j_good, toward_good=True)
        if result:
            compass_results.append(result)

        # Random
        result = run_random_trajectory(loader, noun, j_good)
        if result:
            random_results.append(result)

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    compass_deltas = [r['total_delta_g'] for r in compass_results]
    random_deltas = [r['total_delta_g'] for r in random_results]

    print(f"\n  Compass Navigation:")
    print(f"    n = {len(compass_deltas)}")
    print(f"    mean Δg = {np.mean(compass_deltas):.4f}")
    print(f"    std = {np.std(compass_deltas):.4f}")

    print(f"\n  Random Navigation:")
    print(f"    n = {len(random_deltas)}")
    print(f"    mean Δg = {np.mean(random_deltas):.4f}")
    print(f"    std = {np.std(random_deltas):.4f}")

    # T-test
    t_stat, p_value = scipy_stats.ttest_ind(compass_deltas, random_deltas)

    print(f"\n  Statistical Test:")
    print(f"    Δg difference = {np.mean(compass_deltas) - np.mean(random_deltas):.4f}")
    print(f"    t-statistic = {t_stat:.4f}")
    print(f"    p-value = {p_value:.6f}")

    if p_value < 0.001:
        print(f"\n  CONCLUSION: COMPASS NAVIGATION IS SIGNIFICANT (p < 0.001)")
    elif p_value < 0.05:
        print(f"\n  CONCLUSION: COMPASS NAVIGATION IS SIGNIFICANT (p < 0.05)")
    else:
        print(f"\n  CONCLUSION: No significant difference (p = {p_value:.4f})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_trajectories": args.n_trajectories,
        "compass": {
            "n": len(compass_deltas),
            "mean_delta_g": float(np.mean(compass_deltas)),
            "std": float(np.std(compass_deltas))
        },
        "random": {
            "n": len(random_deltas),
            "mean_delta_g": float(np.mean(random_deltas)),
            "std": float(np.std(random_deltas))
        },
        "statistics": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        }
    }

    with open(OUTPUT_DIR / "compass_validation.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {OUTPUT_DIR / 'compass_validation.json'}")


if __name__ == "__main__":
    main()
