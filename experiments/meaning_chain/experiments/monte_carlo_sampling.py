"""
Monte Carlo Semantic Sampling Experiment

Run the same question N times through the semantic laser and collect statistics.
Like throwing particles many times to see where they land - finding stable orbits.

Goal: Map the "shape" of semantic space for a given question.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import sys

# Add paths
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from chain_core.semantic_laser import SemanticLaser, CoherentBeam, ExcitedState
from chain_core.decomposer import Decomposer
from core.data_loader import DataLoader


@dataclass
class SampleResult:
    """Single sample result."""
    sample_id: int
    seeds: List[str]
    intent_verbs: List[str]

    # Beam data
    beam_count: int
    beams: List[Dict]  # Full beam data

    # Population stats
    total_excited: int
    tau_mean: float
    tau_std: float
    tau_min: float
    tau_max: float
    g_mean: float
    orbital_dist: Dict[int, int]
    dominant_orbital: int
    human_fraction: float

    # Intent collapse
    intent_fraction: float
    intent_collapsed: int

    # Laser metrics
    coherence: float  # Average beam coherence
    pump_energy: float
    medium_quality: float
    mirror_alignment: float
    output_power: float
    lasing_achieved: bool

    # All excited words (for word frequency analysis)
    excited_words: List[str]
    excited_taus: List[float]
    excited_visits: List[int]


class MonteCarloSampler:
    """Run Monte Carlo sampling on semantic space."""

    def __init__(self, intent_strength: float = 0.3):
        """
        Args:
            intent_strength: α parameter (0=pure Boltzmann, 0.3=soft wind, 1+=hard wall)
        """
        self.intent_strength = intent_strength
        self.laser = SemanticLaser(intent_strength=intent_strength)
        self.decomposer = Decomposer(include_proper_nouns=True)
        self.results_dir = _MEANING_CHAIN / "results" / "monte_carlo"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_single_sample(self, question: str, sample_id: int,
                          with_intent: bool = True) -> SampleResult:
        """Run one sample of the laser on a question."""
        # Decompose question
        decomposed = self.decomposer.decompose(question)
        seeds = decomposed.nouns if decomposed.nouns else ['meaning']  # fallback
        intent_verbs = decomposed.verbs if with_intent else []

        # Run laser (with or without intent)
        result = self.laser.lase(
            seeds=seeds,
            pump_power=10,
            pump_depth=5,
            coherence_threshold=0.3,
            min_cluster_size=3,
            intent_verbs=intent_verbs
        )

        # Extract beam data
        beams_data = []
        for beam in result['beams']:
            beams_data.append({
                'concepts': beam.concepts,
                'coherence': float(beam.coherence),
                'tau_mean': float(beam.tau_mean),
                'tau_spread': float(beam.tau_spread),
                'g_polarity': float(beam.g_polarity),
                'intensity': float(beam.intensity),
                'j_centroid': beam.j_centroid.tolist(),
                'orbital_mean': float(getattr(beam, 'orbital_mean', 0)),
                'orbital_spread': float(getattr(beam, 'orbital_spread', 0))
            })

        # Population stats
        pop = result['population']

        # Extract excited states data
        excited = result['excited']
        excited_words = list(excited.keys())
        excited_taus = [excited[w].tau for w in excited_words]
        excited_visits = [excited[w].visits for w in excited_words]

        # Average coherence
        avg_coherence = np.mean([b.coherence for b in result['beams']]) if result['beams'] else 0.0

        metrics = result['metrics']

        return SampleResult(
            sample_id=sample_id,
            seeds=seeds,
            intent_verbs=intent_verbs,
            beam_count=len(result['beams']),
            beams=beams_data,
            total_excited=pop.get('total_excited', 0),
            tau_mean=pop.get('tau_mean', 0),
            tau_std=pop.get('tau_std', 0),
            tau_min=pop.get('tau_min', 0),
            tau_max=pop.get('tau_max', 0),
            g_mean=pop.get('g_mean', 0),
            orbital_dist=pop.get('orbital_dist', {}),
            dominant_orbital=pop.get('dominant_orbital', 0),
            human_fraction=pop.get('human_fraction', 0),
            intent_fraction=pop.get('intent_fraction', 0),
            intent_collapsed=pop.get('intent_collapsed', 0),
            coherence=avg_coherence,
            pump_energy=metrics.get('pump_energy', 0),
            medium_quality=metrics.get('medium_quality', 0),
            mirror_alignment=metrics.get('mirror_alignment', 0),
            output_power=metrics.get('output_power', 0),
            lasing_achieved=metrics.get('lasing_achieved', False),
            excited_words=excited_words,
            excited_taus=excited_taus,
            excited_visits=excited_visits
        )

    def run_experiment(self, question: str, n_samples: int = 100,
                       label: str = None, with_intent: bool = True) -> Dict[str, Any]:
        """
        Run N samples for a question and collect aggregate statistics.
        """
        mode = "WITH INTENT" if with_intent else "PURE BOLTZMANN"
        print(f"\n{'='*70}")
        print(f"MONTE CARLO SAMPLING: {question}")
        print(f"Mode: {mode} | Samples: {n_samples}")
        print(f"{'='*70}\n")

        samples = []
        word_counts = defaultdict(int)  # Track word frequency
        word_taus = defaultdict(list)   # Track tau per word

        for i in range(n_samples):
            if (i + 1) % 10 == 0:
                print(f"  Sample {i+1}/{n_samples}...")

            sample = self.run_single_sample(question, i, with_intent=with_intent)
            samples.append(sample)

            # Aggregate word frequency
            for word, visits in zip(sample.excited_words, sample.excited_visits):
                word_counts[word] += visits
            for word, tau in zip(sample.excited_words, sample.excited_taus):
                word_taus[word].append(tau)

        # Compute aggregate statistics
        aggregate = self._compute_aggregate(samples, word_counts, word_taus)

        # Build full result
        result = {
            'question': question,
            'label': label or question[:30],
            'n_samples': n_samples,
            'timestamp': datetime.now().isoformat(),
            'aggregate': aggregate,
            'samples': [asdict(s) for s in samples]
        }

        # Save to file
        safe_label = (label or question[:20]).replace(' ', '_').replace('?', '')
        filename = f"mc_{safe_label}_{n_samples}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nSaved to: {filepath}")

        # Print summary
        self._print_summary(aggregate, question)

        return result

    def _compute_aggregate(self, samples: List[SampleResult],
                           word_counts: Dict[str, int],
                           word_taus: Dict[str, List[float]]) -> Dict:
        """Compute aggregate statistics across all samples."""

        # Basic stats
        n = len(samples)

        # Tau statistics
        tau_means = [s.tau_mean for s in samples]
        tau_stds = [s.tau_std for s in samples]

        # Coherence statistics
        coherences = [s.coherence for s in samples]

        # Beam count statistics
        beam_counts = [s.beam_count for s in samples]

        # Orbital distribution (aggregate)
        orbital_aggregate = defaultdict(int)
        for s in samples:
            for orbital, count in s.orbital_dist.items():
                orbital_aggregate[int(orbital)] += count

        # Intent collapse
        intent_fractions = [s.intent_fraction for s in samples]

        # Lasing success rate
        lasing_rate = sum(1 for s in samples if s.lasing_achieved) / n

        # Top words (attractors)
        top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:50]

        # Word tau means (where each word tends to live)
        word_tau_means = {
            word: np.mean(taus)
            for word, taus in word_taus.items()
            if len(taus) >= 5  # Only words that appeared 5+ times
        }

        # Cluster attractors by tau
        attractors_by_orbital = defaultdict(list)
        for word, tau in word_tau_means.items():
            orbital = int(round((tau - 1) * np.e))
            attractors_by_orbital[orbital].append((word, word_counts[word], tau))

        # Sort attractors within each orbital
        for orbital in attractors_by_orbital:
            attractors_by_orbital[orbital].sort(key=lambda x: -x[1])

        return {
            # Tau distribution
            'tau_mean': float(np.mean(tau_means)),
            'tau_std_mean': float(np.mean(tau_stds)),
            'tau_variance_across_samples': float(np.var(tau_means)),

            # Coherence
            'coherence_mean': float(np.mean(coherences)),
            'coherence_std': float(np.std(coherences)),
            'coherence_min': float(np.min(coherences)) if coherences else 0,
            'coherence_max': float(np.max(coherences)) if coherences else 0,

            # Beams
            'beam_count_mean': float(np.mean(beam_counts)),
            'beam_count_std': float(np.std(beam_counts)),

            # Orbitals
            'orbital_distribution': dict(orbital_aggregate),
            'dominant_orbitals': sorted(
                orbital_aggregate.items(),
                key=lambda x: -x[1]
            )[:5],

            # Intent collapse
            'intent_fraction_mean': float(np.mean(intent_fractions)),
            'intent_fraction_std': float(np.std(intent_fractions)),

            # Lasing
            'lasing_success_rate': float(lasing_rate),

            # Attractors
            'top_attractors': [(w, c) for w, c in top_words],
            'attractors_by_orbital': {
                k: [(w, c, round(t, 2)) for w, c, t in v[:10]]
                for k, v in sorted(attractors_by_orbital.items())
            },

            # Shape metrics
            'n_unique_words': len(word_counts),
            'concentration': sum(c for _, c in top_words[:10]) / sum(word_counts.values()) if word_counts else 0
        }

    def _print_summary(self, agg: Dict, question: str):
        """Print a nice summary of results."""
        print(f"\n{'='*70}")
        print(f"SUMMARY: {question}")
        print(f"{'='*70}")

        print(f"\n  TAU DISTRIBUTION:")
        print(f"    Mean: {agg['tau_mean']:.3f} ± {agg['tau_std_mean']:.3f}")
        print(f"    Variance across samples: {agg['tau_variance_across_samples']:.4f}")

        print(f"\n  COHERENCE:")
        print(f"    Mean: {agg['coherence_mean']:.3f} ± {agg['coherence_std']:.3f}")
        print(f"    Range: [{agg['coherence_min']:.3f}, {agg['coherence_max']:.3f}]")

        print(f"\n  BEAMS:")
        print(f"    Count: {agg['beam_count_mean']:.1f} ± {agg['beam_count_std']:.1f}")
        print(f"    Lasing success rate: {agg['lasing_success_rate']:.1%}")

        print(f"\n  INTENT COLLAPSE:")
        print(f"    Fraction: {agg['intent_fraction_mean']:.1%} ± {agg['intent_fraction_std']:.1%}")

        print(f"\n  ORBITAL DISTRIBUTION (dominant):")
        for orbital, count in agg['dominant_orbitals']:
            bar = '█' * min(int(count / 50), 40)
            veil = " ← VEIL" if orbital == 5 else ""
            print(f"    n={orbital}: {bar} ({count}){veil}")

        print(f"\n  TOP ATTRACTORS (convergence points):")
        for word, count in agg['top_attractors'][:15]:
            print(f"    {word}: {count}")

        print(f"\n  ATTRACTORS BY ORBITAL:")
        for orbital, words in sorted(agg['attractors_by_orbital'].items()):
            print(f"    n={orbital}: {', '.join([w for w, _, _ in words[:5]])}")

        print(f"\n  SHAPE METRICS:")
        print(f"    Unique words seen: {agg['n_unique_words']}")
        print(f"    Concentration (top-10): {agg['concentration']:.1%}")

    def close(self):
        self.laser.close()


def run_comparative_experiment(with_intent: bool = True, intent_strength: float = 0.3):
    """Run all three question types."""
    sampler = MonteCarloSampler(intent_strength=intent_strength)

    if not with_intent:
        mode = "boltzmann"
    elif intent_strength >= 1.0:
        mode = f"hard_a{intent_strength:.1f}"
    else:
        mode = f"soft_a{intent_strength:.1f}"

    questions = [
        ("What is meaning?", f"abstract_{mode}", 100),
        ("What is a tree?", f"concrete_{mode}", 100),
        ("What do my dreams mean?", f"personal_{mode}", 100)
    ]

    results = []
    for question, label, n_samples in questions:
        result = sampler.run_experiment(question, n_samples, label, with_intent=with_intent)
        results.append(result)

    # Save comparison
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'with_intent': with_intent,
        'intent_strength': intent_strength,
        'experiments': [
            {
                'question': r['question'],
                'label': r['label'],
                'aggregate': r['aggregate']
            }
            for r in results
        ]
    }

    comparison_path = sampler.results_dir / f"comparison_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n\nComparison saved to: {comparison_path}")

    # Print comparative summary
    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)

    headers = ['Metric', 'Abstract', 'Concrete', 'Personal']
    data = [
        ['τ mean', *[f"{r['aggregate']['tau_mean']:.3f}" for r in results]],
        ['Coherence', *[f"{r['aggregate']['coherence_mean']:.3f}" for r in results]],
        ['Lasing %', *[f"{r['aggregate']['lasing_success_rate']:.1%}" for r in results]],
        ['Intent %', *[f"{r['aggregate']['intent_fraction_mean']:.1%}" for r in results]],
        ['Unique words', *[f"{r['aggregate']['n_unique_words']}" for r in results]],
        ['Concentration', *[f"{r['aggregate']['concentration']:.1%}" for r in results]],
    ]

    # Print table
    col_width = 12
    print(f"\n{'':15} " + " | ".join(f"{h:>{col_width}}" for h in headers[1:]))
    print("-" * 55)
    for row in data:
        print(f"{row[0]:15} " + " | ".join(f"{v:>{col_width}}" for v in row[1:]))

    sampler.close()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo Semantic Sampling')
    parser.add_argument('--no-intent', action='store_true',
                        help='Run pure Boltzmann exploration without intent')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Intent strength α (0=Boltzmann, 0.3=soft wind, 1+=hard wall)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples per question')
    args = parser.parse_args()

    run_comparative_experiment(
        with_intent=not args.no_intent,
        intent_strength=args.alpha
    )
