#!/usr/bin/env python3
"""Conversation: Claude as patient, Mistral as therapist.

Tests the Semantic Mirror system by having:
- Claude play a patient with specific psychological patterns
- Mistral render therapeutic responses based on semantic analysis
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

from ..agents.therapist import Therapist
from ..core import get_data


# Patient prompts for Claude to roleplay
PATIENT_PROMPTS = {
    'ironic': """You are roleplaying as a patient in therapy. Your character uses irony and sarcasm as defense mechanisms.
You say things are "fine" when they're not. You minimize your feelings. You use humor to deflect.
Respond naturally as this patient would. Keep responses 1-3 sentences.""",

    'intellectual': """You are roleplaying as a patient in therapy. Your character intellectualizes emotions.
You talk about feelings in abstract, theoretical terms. You analyze rather than feel.
You reference philosophy, psychology concepts to avoid direct emotional engagement.
Respond naturally as this patient would. Keep responses 1-3 sentences.""",

    'negative': """You are roleplaying as a patient in therapy. Your character is in a state of negation.
You dismiss positive possibilities. You focus on what's wrong. You doubt yourself and others.
But underneath there's a longing for connection and meaning.
Respond naturally as this patient would. Keep responses 1-3 sentences.""",

    'mixed': """You are roleplaying as a patient in therapy. Your character has complex defenses.
Sometimes you're ironic, sometimes you intellectualize, sometimes you get genuinely emotional.
You're seeking something but not sure what. You have both dark and light moments.
Respond naturally as this patient would. Keep responses 1-3 sentences.""",
}


def run_conversation(patient_type: str = 'mixed', n_turns: int = 8):
    """Run automated conversation between Claude (patient) and Mistral (therapist)."""

    # Load Claude API key
    env_path = "/home/chukiss/text_project/hypothesis/experiments/semantic_llm/archive/phase3_generation/.env"
    load_dotenv(env_path)

    # Initialize
    claude = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    therapist = Therapist()

    patient_prompt = PATIENT_PROMPTS.get(patient_type, PATIENT_PROMPTS['mixed'])

    print("\n" + "=" * 70)
    print(f"SEMANTIC MIRROR CONVERSATION TEST")
    print(f"Patient type: {patient_type}")
    print("=" * 70)

    # Start with opening from patient
    patient_messages = [
        {"role": "user", "content": "The session begins. Say something to start the therapy session."}
    ]

    conversation = []

    for turn in range(n_turns):
        # 1. Claude generates patient response
        patient_response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=patient_prompt,
            messages=patient_messages
        )
        patient_text = patient_response.content[0].text

        print(f"\n[Turn {turn + 1}]")
        print(f"Patient: {patient_text}")

        # 2. Mistral generates therapist response (computes receptivity inside)
        therapist_text = therapist.respond(patient_text)

        # 3. Get semantic analysis (already computed in respond)
        state = therapist.mirror.trajectory.current
        diagnosis = therapist.mirror.diagnose()

        # Show physics parameters
        receptivity = therapist._current_receptivity
        tokens = therapist._current_response_length

        print(f"  [A={state.A:+.2f}, S={state.S:+.2f}, τ={state.tau:.1f}]", end="")
        print(f" recv:{receptivity:.0%} tok:{tokens}", end="")
        if state.irony > 0.2:
            print(f" irony:{state.irony:.0%}", end="")
        if diagnosis['defenses']:
            print(f" [{', '.join(diagnosis['defenses'])}]", end="")
        print()

        print(f"Therapist: {therapist_text}")

        # 4. Update Claude's context
        patient_messages.append({"role": "assistant", "content": patient_text})
        patient_messages.append({"role": "user", "content": f"The therapist responds: {therapist_text}"})

        conversation.append({
            'turn': turn + 1,
            'patient': patient_text,
            'therapist': therapist_text,
            'state': {
                'A': state.A,
                'S': state.S,
                'tau': state.tau,
                'irony': state.irony,
            },
            'defenses': diagnosis['defenses'],
        })

    # Summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    summary = therapist.summary()
    print(f"Turns: {summary['n_turns']}")
    print(f"Mean position: A={summary['mean_position']['A']:+.2f}, "
          f"S={summary['mean_position']['S']:+.2f}, "
          f"τ={summary['mean_position']['tau']:.2f}")
    print(f"Mean irony: {summary['mean_irony']:.0%}")
    print(f"Defenses observed: {summary['defenses_observed'] or 'none'}")
    print(f"Distance to health: {summary['distance_to_health']:.2f}")

    return conversation


def interactive_with_claude():
    """Interactive mode: you type, Claude responds as patient, then Mistral as therapist."""

    env_path = "/home/chukiss/text_project/hypothesis/experiments/semantic_llm/archive/phase3_generation/.env"
    load_dotenv(env_path)

    claude = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    therapist = Therapist()

    patient_prompt = PATIENT_PROMPTS['mixed']
    patient_messages = []

    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("You provide context, Claude plays patient, Mistral responds as therapist")
    print("Type 'quit' to exit")
    print("=" * 70)

    while True:
        try:
            context = input("\nContext for Claude-patient: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if context.lower() == 'quit':
            break

        # Claude generates patient response
        patient_messages.append({"role": "user", "content": context})
        patient_response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=patient_prompt,
            messages=patient_messages
        )
        patient_text = patient_response.content[0].text
        patient_messages.append({"role": "assistant", "content": patient_text})

        print(f"\nPatient (Claude): {patient_text}")

        # Analyze and respond
        state = therapist.mirror.observe(patient_text)
        diagnosis = therapist.mirror.diagnose()

        print(f"  [A={state.A:+.2f}, S={state.S:+.2f}, τ={state.tau:.1f}]", end="")
        if diagnosis['defenses']:
            print(f" [{', '.join(diagnosis['defenses'])}]", end="")
        print()

        therapist_text = therapist.respond(patient_text)
        print(f"\nTherapist (Mistral): {therapist_text}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '-i':
            interactive_with_claude()
        else:
            patient_type = sys.argv[1]
            run_conversation(patient_type)
    else:
        run_conversation('mixed')
