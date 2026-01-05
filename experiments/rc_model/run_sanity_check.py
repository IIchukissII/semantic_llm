#!/usr/bin/env python3
"""Run the coherent vs random sanity check.

Usage:
    python run_sanity_check.py                    # Quick test
    python run_sanity_check.py --text "..."       # Custom text
    python run_sanity_check.py --file path.txt    # From file
    python run_sanity_check.py --plot             # Show plots
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rc_model.experiments.coherent_vs_random import (
    run_sanity_check,
    print_report,
    evaluate_result,
)
from rc_model.core.cardiogram import plot_cardiogram, plot_comparison


SAMPLE_TEXTS = {
    'short': """
    The old man walked slowly through the dark forest.
    He carried a heavy burden on his tired shoulders.
    The ancient trees whispered mysterious secrets.
    """,

    'medium': """
    The old man walked slowly through the dark forest.
    He carried a heavy burden on his tired shoulders.
    The ancient trees whispered mysterious secrets.
    A cold wind blew through the empty branches.
    The lonely traveler continued his difficult journey.
    Hope remained strong in his weary heart.
    The distant mountains promised safe refuge.
    But the dangerous path lay ahead.
    He gathered his remaining strength.
    The journey must continue.
    """,

    'bible': """
    In the beginning God created the heaven and the earth.
    And the earth was without form and void.
    And darkness was upon the face of the deep.
    And the Spirit of God moved upon the face of the waters.
    And God said Let there be light and there was light.
    And God saw the light that it was good.
    And God divided the light from the darkness.
    And God called the light Day and the darkness he called Night.
    And the evening and the morning were the first day.
    """,
}


def main():
    parser = argparse.ArgumentParser(description="Run RC-Model sanity check")
    parser.add_argument('--text', type=str, help="Text to analyze")
    parser.add_argument('--file', type=str, help="File containing text")
    parser.add_argument('--sample', choices=SAMPLE_TEXTS.keys(), default='medium',
                        help="Use sample text")
    parser.add_argument('--plot', action='store_true', help="Show plots")
    parser.add_argument('--save', type=str, help="Save plots to directory")

    args = parser.parse_args()

    # Get text
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text()
    else:
        text = SAMPLE_TEXTS[args.sample]

    print(f"Text length: {len(text)} chars")
    print(f"Preview: {text[:100]}...\n")

    # Run sanity check
    result = run_sanity_check(text)
    print_report(result)

    # Evaluate
    evaluation = evaluate_result(result)

    # Plots
    if args.plot or args.save:
        import matplotlib.pyplot as plt

        # Cardiogram comparison
        fig1 = plot_comparison(
            {
                'Original': result.original,
                'Full Shuffle': result.full_shuffle,
            },
            title="Coherent vs Random: Semantic Potential",
            component='potential',
        )

        fig2 = plot_cardiogram(
            result.original,
            title="Original Text: Semantic Cardiogram",
            show_potential=True,
        )

        fig3 = plot_cardiogram(
            result.full_shuffle,
            title="Shuffled Text: Semantic Cardiogram",
            show_potential=True,
        )

        if args.save:
            save_dir = Path(args.save)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(save_dir / "comparison.png", dpi=150)
            fig2.savefig(save_dir / "original.png", dpi=150)
            fig3.savefig(save_dir / "shuffled.png", dpi=150)
            print(f"\nPlots saved to {save_dir}")

        if args.plot:
            plt.show()

    return evaluation


if __name__ == "__main__":
    main()
