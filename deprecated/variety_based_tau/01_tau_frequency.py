#!/usr/bin/env python3
"""
EXPERIMENT 01: τ Level vs Word Frequency
=========================================

Hypothesis: τ levels should correlate with word frequency.
- τ₁ (abstract) words → HIGH frequency (thing, person, way)
- τ₆ (specific) words → LOW frequency (cardiologist, Beethoven)

This validates that our τ assignment is meaningful.
"""

import json
import numpy as np
import psycopg2
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}

print("=" * 60)
print("EXPERIMENT 01: τ Level vs Word Frequency")
print("=" * 60)

# Load τ levels
print("\n[1] Loading τ levels...")
tau_data = json.load(open(Path(__file__).parent.parent.parent.parent /
                          "data/semantic_vectors/tau_levels.json"))

noun_tau = tau_data['nouns']
verb_tau = tau_data['verbs']
print(f"  Nouns: {len(noun_tau)}, Verbs: {len(verb_tau)}")

# Load word frequencies from bond counts
print("\n[2] Loading word frequencies from bonds...")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Noun frequency = total adj-noun bond count
cur.execute('''
    SELECT
        split_part(bond, '|', 2) as noun,
        SUM(total_count) as freq
    FROM hyp_bond_vocab
    WHERE bond LIKE '%|%'
    GROUP BY split_part(bond, '|', 2)
''')
noun_freq = {row[0]: row[1] for row in cur.fetchall()}

# Verb frequency = total verb-noun bond count
cur.execute('''
    SELECT
        split_part(bond, '|', 1) as verb,
        SUM(total_count) as freq
    FROM hyp_verb_bond_vocab
    GROUP BY split_part(bond, '|', 1)
''')
verb_freq = {row[0]: row[1] for row in cur.fetchall()}

conn.close()
print(f"  Noun frequencies: {len(noun_freq)}")
print(f"  Verb frequencies: {len(verb_freq)}")

# Analyze correlation
print("\n[3] Analyzing τ vs frequency correlation...")

def analyze_tau_freq(tau_dict, freq_dict, word_type):
    """Analyze correlation between τ and frequency."""

    # Group by τ level
    tau_groups = defaultdict(list)

    for word, tau in tau_dict.items():
        if word in freq_dict:
            tau_groups[tau].append(freq_dict[word])

    print(f"\n  {word_type.upper()} Analysis:")
    print(f"  {'τ':<4} {'Count':<8} {'Mean Freq':<12} {'Median':<12} {'Examples'}")
    print("  " + "-" * 70)

    tau_means = []
    tau_levels = []

    for tau in sorted(tau_groups.keys()):
        freqs = tau_groups[tau]
        mean_freq = np.mean(freqs)
        median_freq = np.median(freqs)

        tau_means.append(mean_freq)
        tau_levels.append(tau)

        # Get example words
        words_at_tau = [(w, freq_dict[w]) for w, t in tau_dict.items()
                        if t == tau and w in freq_dict]
        words_at_tau.sort(key=lambda x: -x[1])
        examples = [w for w, f in words_at_tau[:3]]

        print(f"  τ{tau:<3} {len(freqs):<8} {mean_freq:<12.1f} {median_freq:<12.1f} {', '.join(examples)}")

    # Compute correlation
    all_tau = []
    all_freq = []
    for word, tau in tau_dict.items():
        if word in freq_dict:
            all_tau.append(tau)
            all_freq.append(np.log1p(freq_dict[word]))  # Log frequency

    correlation = np.corrcoef(all_tau, all_freq)[0, 1]
    print(f"\n  Correlation (τ vs log_freq): r = {correlation:.4f}")

    return tau_levels, tau_means, correlation


# Analyze nouns
noun_levels, noun_means, noun_corr = analyze_tau_freq(noun_tau, noun_freq, "noun")

# Analyze verbs
verb_levels, verb_means, verb_corr = analyze_tau_freq(verb_tau, verb_freq, "verb")

# Plot results
print("\n[4] Generating plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Noun plot
ax = axes[0]
ax.bar(noun_levels, noun_means, color='steelblue', alpha=0.7)
ax.set_xlabel('τ Level')
ax.set_ylabel('Mean Frequency')
ax.set_title(f'Noun: τ vs Frequency (r={noun_corr:.3f})')
ax.set_yscale('log')
ax.set_xticks(noun_levels)
ax.set_xticklabels([f'τ{t}' for t in noun_levels])

# Add trend annotation
if noun_corr < 0:
    ax.annotate('✓ Expected: Higher τ → Lower Freq',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=10, color='green')

# Verb plot
ax = axes[1]
ax.bar(verb_levels, verb_means, color='coral', alpha=0.7)
ax.set_xlabel('τ Level')
ax.set_ylabel('Mean Frequency')
ax.set_title(f'Verb: τ vs Frequency (r={verb_corr:.3f})')
ax.set_yscale('log')
ax.set_xticks(verb_levels)
ax.set_xticklabels([f'τ{t}' for t in verb_levels])

if verb_corr < 0:
    ax.annotate('✓ Expected: Higher τ → Lower Freq',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=10, color='green')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tau_vs_frequency.png', dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR / 'tau_vs_frequency.png'}")

# Summary
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"""
Correlation (τ vs log_frequency):
  Nouns: r = {noun_corr:.4f}
  Verbs: r = {verb_corr:.4f}

Expected: NEGATIVE correlation
  (τ₁ = high variety = common words = high frequency)
  (τ₆ = low variety = rare words = low frequency)

Interpretation:
""")

if noun_corr < -0.3 and verb_corr < -0.3:
    print("  ✓ VALIDATED: τ levels strongly correlate with word frequency")
    print("  → Our abstraction hierarchy is meaningful")
    validation = "PASSED"
elif noun_corr < 0 and verb_corr < 0:
    print("  ~ PARTIAL: τ levels weakly correlate with frequency")
    print("  → Some signal, may need refinement")
    validation = "PARTIAL"
else:
    print("  ✗ FAILED: τ levels don't correlate as expected")
    print("  → Need to investigate")
    validation = "FAILED"

# Save results
results = {
    'noun_correlation': float(noun_corr),
    'verb_correlation': float(verb_corr),
    'validation': validation,
    'noun_tau_counts': {str(t): len([w for w, tau in noun_tau.items() if tau == t])
                        for t in range(1, 7)},
    'verb_tau_counts': {str(t): len([w for w, tau in verb_tau.items() if tau == t])
                        for t in range(1, 7)}
}

with open(OUTPUT_DIR / 'tau_frequency_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'tau_frequency_results.json'}")
