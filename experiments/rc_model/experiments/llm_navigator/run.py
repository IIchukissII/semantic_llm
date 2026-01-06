#!/usr/bin/env python3
"""Run LLM Navigator experiment.

Usage:
    python run.py                    # Run all genres
    python run.py --genre dramatic   # Run specific genre
    python run.py --save             # Save results to JSON
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from core import navigate_text, load_coordinates, GENRE_TARGETS


def run_experiment(genres: list, n_sentences: int = 5,
                   save: bool = False, verbose: bool = True):
    """Run navigation experiment for specified genres."""

    coord_dict = load_coordinates()
    print(f"Loaded {len(coord_dict)} word coordinates\n")

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_sentences': n_sentences,
            'n_coordinates': len(coord_dict),
        },
        'samples': []
    }

    for genre in genres:
        text, trajectory = navigate_text(
            genre=genre,
            n_sentences=n_sentences,
            coord_dict=coord_dict,
            verbose=verbose
        )

        results['samples'].append({
            'genre': genre,
            'text': text,
            'trajectory': [
                {'A': p.A, 'S': p.S, 'tau': p.tau}
                for p in trajectory
            ],
            'target': {
                'A': GENRE_TARGETS[genre].A,
                'S': GENRE_TARGETS[genre].S,
                'tau': GENRE_TARGETS[genre].tau,
            }
        })

    if save:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"navigator_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Navigator Experiment")
    parser.add_argument('--genre', type=str, choices=['dramatic', 'ironic', 'balanced'],
                        help='Run specific genre only')
    parser.add_argument('--sentences', type=int, default=5,
                        help='Number of sentences to generate')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')

    args = parser.parse_args()

    genres = [args.genre] if args.genre else ['dramatic', 'ironic', 'balanced']

    run_experiment(
        genres=genres,
        n_sentences=args.sentences,
        save=args.save,
        verbose=not args.quiet
    )
