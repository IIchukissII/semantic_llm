#!/usr/bin/env python3
"""Semantic Mirror Demo: Psychoanalyst Session.

Run a simulated therapy session demonstrating:
- Detection of semantic position
- Defense identification
- Dialectical analysis
- Trajectory tracking
"""

from ..agents import SemanticMirror
from ..core import get_data


def demo_session():
    """Run demo psychoanalyst session."""
    data = get_data()
    print(f"\nLoaded {data.n_coordinates:,} coordinates")
    print(f"  Nouns: {data.n_nouns:,}")

    mirror = SemanticMirror(data)

    # Simulated therapy session
    test_messages = [
        "I guess everything is fine. It's always fine.",
        "Work is just... work. Nothing special. Nothing terrible.",
        "Sometimes I wonder if any of this matters at all.",
        "But then again, who am I to complain? Others have it worse.",
        "Maybe I should just be grateful for what I have.",
        "The existential weight of being presses upon the soul.",
        "Just grab some coffee and let's get this done.",
        "Everything is FINE. Totally FINE. Why wouldn't it be?",
    ]

    print("\n" + "=" * 70)
    print("PSYCHOANALYST SESSION")
    print("=" * 70)
    health = mirror.health_target
    print(f"Health target: A={health.A}, S={health.S}, τ={health.tau}")

    for i, msg in enumerate(test_messages):
        print(f"\n[Turn {i+1}] Patient: {msg}")

        state = mirror.observe(msg)
        diagnosis = mirror.diagnose()
        dialectic = mirror.dialectic()

        print(f"  Position:  A={state.A:+.2f}, S={state.S:+.2f}, τ={state.tau:.2f}")

        if state.irony > 0.2 or state.sarcasm > 0.2:
            print(f"  Markers:   irony={state.irony:.0%}, sarcasm={state.sarcasm:.0%}")

        if diagnosis['defenses']:
            print(f"  Defenses:  {', '.join(diagnosis['defenses'])}")

        # Dialectical analysis
        th = dialectic['thesis']
        an = dialectic['antithesis']
        sy = dialectic['synthesis']
        mv = dialectic['intervention']

        print(f"  THESIS:     ({th['A']:+.1f}, {th['S']:+.1f}, {th['tau']:.1f}) {th['description']}")
        print(f"  ANTITHESIS: ({an['A']:+.1f}, {an['S']:+.1f}, {an['tau']:.1f}) {an['description']}")
        print(f"  SYNTHESIS:  ({sy['A']:+.1f}, {sy['S']:+.1f}, {sy['tau']:.1f}) {sy['description']}")

        if dialectic['blocking_defense']:
            print(f"  Block: {dialectic['blocking_defense']}")

        print(f"  Intervention: {mv['direction']} (axis={mv['primary_axis']}, mag={mv['magnitude']:.2f})")

    # Session summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    summary = mirror.summary()
    print(f"Turns: {summary['n_turns']}")
    print(f"Mean position: A={summary['mean_position']['A']:+.2f}, "
          f"S={summary['mean_position']['S']:+.2f}, τ={summary['mean_position']['tau']:.2f}")
    print(f"Mean irony: {summary['mean_irony']:.0%}")
    print(f"Defenses: {summary['defenses_observed'] or 'none'}")
    print(f"Resistance: {summary['overall_resistance']:.0%}")
    print(f"Distance to health: {summary['distance_to_health']:.2f}")


def interactive_session():
    """Run interactive session with user input."""
    mirror = SemanticMirror()

    print("\n" + "=" * 70)
    print("SEMANTIC MIRROR - Interactive Session")
    print("=" * 70)
    print("Type your thoughts. Type 'quit' to exit, 'summary' for session summary.")
    print()

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue
        if text.lower() == 'quit':
            break
        if text.lower() == 'summary':
            summary = mirror.summary()
            print(f"\n  Mean: A={summary['mean_position']['A']:+.2f}, "
                  f"S={summary['mean_position']['S']:+.2f}, τ={summary['mean_position']['tau']:.2f}")
            print(f"  Irony: {summary['mean_irony']:.0%}, "
                  f"Resistance: {summary['overall_resistance']:.0%}")
            print(f"  Defenses: {summary['defenses_observed'] or 'none'}\n")
            continue

        state = mirror.observe(text)
        diagnosis = mirror.diagnose()

        # Brief feedback
        print(f"  [{state.A:+.2f}, {state.S:+.2f}, τ={state.tau:.1f}]", end="")
        if state.irony > 0.3:
            print(f" irony:{state.irony:.0%}", end="")
        if diagnosis['defenses']:
            print(f" [{', '.join(diagnosis['defenses'])}]", end="")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        interactive_session()
    else:
        demo_session()
