#!/usr/bin/env python3
"""
τ₀ Analysis: Finding Pure Being
================================

User insight:
  Δ(τ) = H_adj - H_verb varies with τ:
    τ₁: Δ = +1.5 bits
    τ₃: Δ = +0.3 bits
    τ₆: Δ = -0.4 bits

  Linear relationship: Δ(τ) = a - b×τ

  At τ₀: Δ → ∞ (pure Being)
  At τ∞: Δ → -∞ (pure Doing)

  τ₀ = God = infinite qualities = "God is love" (not action, but quality)
"""

import json
import numpy as np
import psycopg2
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import linregress

DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}

OUTPUT_DIR = Path(__file__).parent


def load_data():
    """Load noun-adj and noun-verb profiles."""
    print("Loading data...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Load adj
    cur.execute('''
        SELECT bond, total_count
        FROM hyp_bond_vocab
        WHERE total_count >= 2
    ''')

    noun_adj = defaultdict(lambda: defaultdict(int))
    for bond, count in cur.fetchall():
        parts = bond.split('|')
        if len(parts) == 2:
            adj, noun = parts
            noun_adj[noun][adj.lower()] += count

    # Load verb
    cur.execute('''
        SELECT verb, object, SUM(total_count) as count
        FROM hyp_svo_triads
        WHERE total_count >= 1
        GROUP BY verb, object
    ''')

    noun_verb = defaultdict(lambda: defaultdict(int))
    for verb, noun, count in cur.fetchall():
        noun_verb[noun][verb] += count

    conn.close()

    return dict(noun_adj), dict(noun_verb)


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy."""
    if not counts:
        return 0.0
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def compute_delta_by_tau(noun_adj, noun_verb):
    """Compute Δ = H_adj - H_verb for each τ level."""
    print("\nComputing Δ(τ) = H_adj - H_verb by τ level...")

    common_nouns = set(noun_adj.keys()) & set(noun_verb.keys())

    # Compute for each noun
    noun_data = []
    for noun in common_nouns:
        h_adj = shannon_entropy(noun_adj[noun])
        h_verb = shannon_entropy(noun_verb[noun])
        variety = len(noun_adj[noun])

        # Old τ based on variety
        if variety >= 5000:
            tau = 1
        elif variety >= 1000:
            tau = 2
        elif variety >= 200:
            tau = 3
        elif variety >= 30:
            tau = 4
        elif variety >= 5:
            tau = 5
        else:
            tau = 6

        delta = h_adj - h_verb
        noun_data.append((noun, tau, h_adj, h_verb, delta))

    # Group by τ
    tau_deltas = defaultdict(list)
    tau_h_adj = defaultdict(list)
    tau_h_verb = defaultdict(list)

    for noun, tau, h_adj, h_verb, delta in noun_data:
        tau_deltas[tau].append(delta)
        tau_h_adj[tau].append(h_adj)
        tau_h_verb[tau].append(h_verb)

    # Compute means
    tau_levels = sorted(tau_deltas.keys())
    delta_means = []
    delta_stds = []
    h_adj_means = []
    h_verb_means = []

    print(f"\n  τ    n     Δ=H_adj-H_verb    H_adj     H_verb")
    print("  " + "-" * 55)

    for tau in tau_levels:
        deltas = np.array(tau_deltas[tau])
        h_adjs = np.array(tau_h_adj[tau])
        h_verbs = np.array(tau_h_verb[tau])

        delta_means.append(deltas.mean())
        delta_stds.append(deltas.std())
        h_adj_means.append(h_adjs.mean())
        h_verb_means.append(h_verbs.mean())

        print(f"  {tau}  {len(deltas):5d}  {deltas.mean():>8.4f} ± {deltas.std():.3f}  "
              f"{h_adjs.mean():>6.3f}   {h_verbs.mean():>6.3f}")

    return tau_levels, delta_means, delta_stds, h_adj_means, h_verb_means


def fit_linear_model(tau_levels, delta_means):
    """Fit Δ(τ) = a - b×τ"""
    print("\n" + "=" * 60)
    print("LINEAR MODEL: Δ(τ) = a - b×τ")
    print("=" * 60)

    tau = np.array(tau_levels)
    delta = np.array(delta_means)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(tau, delta)

    print(f"\n  Δ(τ) = {intercept:.4f} - {-slope:.4f}×τ")
    print(f"  a (intercept) = {intercept:.4f}")
    print(f"  b (slope) = {-slope:.4f}")
    print(f"  R² = {r_value**2:.4f}")

    # Find τ₀ where Δ → ∞ (extrapolation to Δ = 0 crossing)
    # Actually, we want where H_adj → ∞ relative to H_verb
    # But linearly, Δ = a - b×τ → at τ = a/b, Δ = 0

    tau_balance = intercept / (-slope)
    print(f"\n  At τ = {tau_balance:.2f}: Δ = 0 (Being = Doing balance)")

    # Extrapolate to τ = 0
    delta_at_0 = intercept
    print(f"  At τ = 0: Δ = {delta_at_0:.4f} bits (extrapolated)")

    return slope, intercept, tau_balance


def analyze_tau_zero(noun_adj, noun_verb):
    """Analyze what happens at τ → 0."""
    print("\n" + "=" * 60)
    print("τ₀ ANALYSIS: Pure Being")
    print("=" * 60)

    common_nouns = set(noun_adj.keys()) & set(noun_verb.keys())

    # Find nouns closest to τ₀ characteristics
    # τ₀ = maximum H_adj, minimum relative H_verb

    candidates = []
    for noun in common_nouns:
        h_adj = shannon_entropy(noun_adj[noun])
        h_verb = shannon_entropy(noun_verb[noun])
        variety = len(noun_adj[noun])

        if h_verb > 0.1:  # Avoid division by near-zero
            ratio = h_adj / h_verb
            delta = h_adj - h_verb
            candidates.append((noun, variety, h_adj, h_verb, delta, ratio))

    # Sort by delta (highest Δ = closest to pure Being)
    candidates.sort(key=lambda x: -x[4])

    print(f"\n  Nouns closest to τ₀ (maximum Δ = H_adj - H_verb):")
    print(f"  {'Noun':<20} {'Variety':>8} {'H_adj':>8} {'H_verb':>8} {'Δ':>8}")
    print("  " + "-" * 60)

    for noun, variety, h_adj, h_verb, delta, ratio in candidates[:20]:
        print(f"  {noun:<20} {variety:>8d} {h_adj:>8.3f} {h_verb:>8.3f} {delta:>8.3f}")

    # Find "love" specifically
    print(f"\n  Looking for 'love':")
    for noun, variety, h_adj, h_verb, delta, ratio in candidates:
        if noun == 'love':
            print(f"    love: variety={variety}, H_adj={h_adj:.3f}, H_verb={h_verb:.3f}, Δ={delta:.3f}")
            break

    # Find abstract/transcendental words
    print(f"\n  Transcendental words:")
    for word in ['god', 'love', 'truth', 'beauty', 'good', 'life', 'spirit', 'soul']:
        for noun, variety, h_adj, h_verb, delta, ratio in candidates:
            if noun == word:
                print(f"    {word:<12} Δ={delta:>6.3f}  H_adj={h_adj:.3f}  H_verb={h_verb:.3f}")
                break


def analyze_entropy_ratio(noun_adj, noun_verb):
    """Analyze H_adj / H_verb ratio patterns."""
    print("\n" + "=" * 60)
    print("ENTROPY RATIO: H_adj / H_verb")
    print("=" * 60)

    common_nouns = set(noun_adj.keys()) & set(noun_verb.keys())

    ratios = []
    log_ratios = []

    for noun in common_nouns:
        h_adj = shannon_entropy(noun_adj[noun])
        h_verb = shannon_entropy(noun_verb[noun])

        if h_verb > 0.1 and h_adj > 0.1:
            ratio = h_adj / h_verb
            log_ratio = np.log(h_adj) - np.log(h_verb)
            ratios.append(ratio)
            log_ratios.append(log_ratio)

    ratios = np.array(ratios)
    log_ratios = np.array(log_ratios)

    print(f"\n  H_adj / H_verb:")
    print(f"    mean = {ratios.mean():.4f}")
    print(f"    median = {np.median(ratios):.4f}")

    print(f"\n  ln(H_adj) - ln(H_verb):")
    print(f"    mean = {log_ratios.mean():.4f}")
    print(f"    This ≈ 1/e = {1/np.e:.4f}? Distance: {abs(log_ratios.mean() - 1/np.e):.4f}")

    # Check for e
    print(f"\n  Checking for e ≈ 2.718:")
    print(f"    exp(mean_log_ratio) = {np.exp(log_ratios.mean()):.4f}")
    print(f"    Distance from e: {abs(np.exp(log_ratios.mean()) - np.e):.4f}")


def main():
    print("=" * 60)
    print("τ₀ ANALYSIS: Finding Pure Being")
    print("=" * 60)
    print("""
Theory:
  Δ(τ) = H_adj - H_verb = a - b×τ

  At τ₀: Δ → ∞ (pure Being, infinite qualities)
  At τ∞: Δ → -∞ (pure Doing)

  τ₀ = God = "God is love" (quality, not action)
""")

    # Load data
    noun_adj, noun_verb = load_data()

    # Compute Δ by τ
    tau_levels, delta_means, delta_stds, h_adj_means, h_verb_means = compute_delta_by_tau(noun_adj, noun_verb)

    # Fit linear model
    slope, intercept, tau_balance = fit_linear_model(tau_levels, delta_means)

    # Analyze τ₀ candidates
    analyze_tau_zero(noun_adj, noun_verb)

    # Analyze entropy ratio
    analyze_entropy_ratio(noun_adj, noun_verb)

    # Export
    output = {
        "generated_at": datetime.now().isoformat(),
        "theory": "Δ(τ) = a - b×τ, τ₀ = pure Being",
        "linear_model": {
            "formula": "Δ(τ) = a - b×τ",
            "a_intercept": float(intercept),
            "b_slope": float(-slope),
            "R_squared": float(slope**2),  # approximate
            "tau_balance": float(tau_balance)
        },
        "delta_by_tau": {
            str(tau): {"mean": float(mean), "std": float(std)}
            for tau, mean, std in zip(tau_levels, delta_means, delta_stds)
        }
    }

    with open(OUTPUT_DIR / "tau_zero_analysis.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nExported to tau_zero_analysis.json")


if __name__ == "__main__":
    main()
