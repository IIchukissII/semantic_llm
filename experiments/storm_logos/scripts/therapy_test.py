#!/usr/bin/env python3
"""Test therapy conversation: Claude as patient, Storm-Logos as therapist.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python -m storm_logos.scripts.therapy_test
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import anthropic
from storm_logos.applications.therapist import Therapist


# Load API key from env file if not set
def load_api_key():
    """Load API key from env file."""
    if os.environ.get('ANTHROPIC_API_KEY'):
        return os.environ['ANTHROPIC_API_KEY']

    env_paths = [
        Path(__file__).parent.parent / '.env',  # storm_logos/.env
        Path('/home/chukiss/text_project/hypothesis/experiments/semantic_llm/archive/phase3_generation/.env'),
        Path('/home/chukiss/text_project/hypothesis/validation/.env'),
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('ANTHROPIC_API_KEY='):
                        key = line.split('=', 1)[1].strip()
                        os.environ['ANTHROPIC_API_KEY'] = key
                        return key

    raise ValueError("ANTHROPIC_API_KEY not found")


def create_patient_client(api_key: str):
    """Create Claude client for patient role."""
    return anthropic.Anthropic(api_key=api_key)


def patient_respond(client, therapist_message: str, history: list) -> str:
    """Generate patient response using Claude."""

    system = """You are a therapy PATIENT (not a therapist).

You are dealing with:
- Feeling empty and disconnected
- Using humor and irony to avoid real feelings
- Anger that you don't fully understand
- Fear of being truly seen

IMPORTANT:
- You are NOT the therapist. You are the patient.
- Respond naturally as someone in therapy
- Sometimes deflect with jokes
- Sometimes get defensive
- Sometimes open up a little
- Never give therapy advice or ask "how does that make you feel"

Keep responses 1-3 sentences."""

    # Build message history
    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"Therapist: {therapist_message}\n\nRespond as the patient:"})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        system=system,
        messages=messages,
    )

    return response.content[0].text.strip()


def run_therapy_session(n_turns: int = 10):
    """Run a therapy session with Claude as patient."""
    print("=" * 60)
    print("STORM-LOGOS THERAPY TEST")
    print("Claude as Patient, Storm-Logos as Therapist")
    print("=" * 60)
    print()

    # Load API key
    api_key = load_api_key()
    print(f"API key loaded: {api_key[:20]}...")
    print()

    # Create therapist (uses same API key)
    therapist = Therapist(model='claude', api_key=api_key)

    # Create patient client
    patient_client = create_patient_client(api_key)

    # Initial patient message
    patient_text = "I don't know why I'm here. My wife made me come. Everything's fine, I guess."

    history = []

    print("-" * 60)
    print(f"PATIENT: {patient_text}")
    print("-" * 60)

    for turn in range(n_turns):
        # Therapist responds
        therapist_response = therapist.respond(patient_text)
        print(f"\nTHERAPIST [{turn+1}]: {therapist_response}")

        # Get current state
        state = therapist.trajectory.current
        if state:
            print(f"  [Analysis: A={state.A:+.2f}, S={state.S:+.2f}, irony={state.irony:.0%}]")

        # Add to history
        history.append({"role": "assistant", "content": therapist_response})

        # Patient responds
        patient_text = patient_respond(patient_client, therapist_response, history)
        print(f"\nPATIENT [{turn+1}]: {patient_text}")

        history.append({"role": "user", "content": patient_text})

        # Check for natural ending
        if any(phrase in patient_text.lower() for phrase in ["goodbye", "see you next", "thanks for"]):
            print("\n[Session ending naturally]")
            break

    print()
    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)

    # Show trajectory
    traj = therapist.get_trajectory()
    states = traj.history
    print(f"\nTurns: {len(states)}")

    if states:
        avg_A = sum(s.A for s in states) / len(states)
        avg_irony = sum(s.irony for s in states) / len(states)
        print(f"Average A (affirmation): {avg_A:+.2f}")
        print(f"Average irony: {avg_irony:.0%}")

        # Movement
        if len(states) >= 2:
            first = states[0]
            last = states[-1]
            print(f"\nMovement:")
            print(f"  A: {first.A:+.2f} → {last.A:+.2f} ({last.A - first.A:+.2f})")
            print(f"  Irony: {first.irony:.0%} → {last.irony:.0%}")

    # Save session
    saved_path = therapist.save_session()
    print(f"\nSession saved to: {saved_path}")


def interactive_session():
    """Run interactive therapy session where user is the patient."""
    print("=" * 60)
    print("STORM-LOGOS INTERACTIVE THERAPY")
    print("Type 'quit' to exit")
    print("=" * 60)
    print()

    api_key = load_api_key()
    therapist = Therapist(model='claude', api_key=api_key)

    print("Therapist: What brings you here today?")
    print()

    while True:
        try:
            patient_text = input("You: ").strip()
            if patient_text.lower() in ('quit', 'exit', 'q'):
                break
            if not patient_text:
                continue

            response = therapist.respond(patient_text)
            print(f"\nTherapist: {response}")

            state = therapist.trajectory.current
            if state:
                print(f"  [A={state.A:+.2f}, S={state.S:+.2f}, irony={state.irony:.0%}]")
            print()

        except KeyboardInterrupt:
            break

    # Save session if there were turns
    if therapist._turns:
        saved_path = therapist.save_session()
        print(f"\nSession saved to: {saved_path}")
    print("\n[Session ended]")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Storm-Logos Therapy Test')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode (you are the patient)')
    parser.add_argument('--turns', '-n', type=int, default=10,
                        help='Number of turns for automated session')
    args = parser.parse_args()

    if args.interactive:
        interactive_session()
    else:
        run_therapy_session(n_turns=args.turns)
