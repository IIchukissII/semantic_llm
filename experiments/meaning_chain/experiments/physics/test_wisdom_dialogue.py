#!/usr/bin/env python3
"""
WISDOM vs POWERFUL Dialogue Test
=================================

Runs parallel dialogues with Claude using WISDOM and POWERFUL modes,
then compares the quality metrics and responses.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python test_wisdom_dialogue.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

_THIS_FILE = Path(__file__).resolve()
_PHYSICS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _PHYSICS_DIR.parent.parent

sys.path.insert(0, str(_MEANING_CHAIN))

from chain_core.navigator import SemanticNavigator, NavigationGoal


def compute_synthesis_coherence(graph, concepts: list) -> float:
    """Compute average pairwise coherence among concepts (consistent metric)."""
    import numpy as np

    if len(concepts) < 2:
        return 0.0

    j_vectors = []
    for word in concepts:
        concept = graph.get_concept(word)
        if concept and concept.get('j'):
            j = np.array(concept['j'])
            if len(j) == 5:
                j_vectors.append(j)

    if len(j_vectors) < 2:
        return 0.0

    sims = []
    for i in range(len(j_vectors)):
        for k in range(i + 1, len(j_vectors)):
            v1, v2 = j_vectors[i], j_vectors[k]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                sims.append(float(np.dot(v1, v2) / (n1 * n2)))

    return float(np.mean(sims)) if sims else 0.0


def test_single_response(navigator, question: str, goal: str) -> dict:
    """Get a single navigation response with consistent metrics."""
    result = navigator.navigate(question, goal=goal)

    q = result.quality
    P = q.power

    # CONSISTENT METRIC: Compute synthesis coherence the same way for all modes
    # Use non-pole concepts from the result
    synthesis_concepts = []
    poles = {result.thesis, result.antithesis} if result.thesis else set()

    for c in result.concepts:
        if c not in poles:
            synthesis_concepts.append(c)

    # Compute synthesis coherence consistently
    graph = navigator._init_graph()
    C_synth = compute_synthesis_coherence(graph, synthesis_concepts[:10])
    C_synth = abs(C_synth)

    # Also store landscape coherence for comparison
    C_landscape = q.coherence

    meaning = C_synth * P
    sigma = C_synth + 0.1 * P
    balance_ratio = C_synth / (0.1 * P) if P > 0 else 0
    efficiency = meaning / 3.78 if meaning > 0 else 0

    return {
        'goal': goal,
        'strategy': result.strategy,
        'concepts': result.concepts[:8],
        'synthesis_concepts': synthesis_concepts[:10],
        'quality': {
            'resonance': q.resonance,
            'coherence_landscape': C_landscape,  # Original landscape coherence
            'coherence_synthesis': C_synth,       # Consistent synthesis coherence
            'depth': q.depth,
            'stability': q.stability,
            'power': P,
            'tau_mean': q.tau_mean
        },
        'meaning_metrics': {
            'C': C_synth,  # Use synthesis coherence for C
            'P': P,
            'meaning': meaning,
            'sigma': sigma,
            'balance_ratio': balance_ratio,
            'efficiency': efficiency
        },
        'paradox': {
            'thesis': result.thesis,
            'antithesis': result.antithesis,
            'synthesis': result.synthesis[:5] if result.synthesis else []
        } if result.thesis else None
    }


def test_with_claude(question: str, nav_result: dict) -> str:
    """Generate Claude response based on navigation."""
    import anthropic

    client = anthropic.Anthropic()

    q = nav_result['quality']
    m = nav_result['meaning_metrics']
    concepts = nav_result['concepts']
    goal = nav_result['goal']

    # Build system prompt based on goal
    if goal == "wisdom":
        tone = "wise and balanced, holding both depth and accessibility"
        guidance = "Aim for harmonious synthesis - coherent yet substantial."
    else:  # powerful
        tone = "powerful and paradoxical, embracing tension"
        guidance = "Hold opposites together - let the paradox resonate."

    system_prompt = f"""You respond based on semantic navigation with goal: {goal}

Quality: R={q['resonance']:.2f}, C_synth={m['C']:.3f}, P={m['P']:.1f}
Tone: {tone}

Use these navigated concepts naturally: {', '.join(concepts[:6])}

{guidance}
2-4 sentences, genuine and substantive."""

    paradox_info = ""
    if nav_result.get('paradox'):
        p = nav_result['paradox']
        paradox_info = f"\n\nParadox: {p['thesis']} ↔ {p['antithesis']}\nSynthesis: {p.get('synthesis', [])[:3]}"

    user_prompt = f"""Question: {question}

Navigation Result:
  Goal: {goal}
  Strategy: {nav_result['strategy']}
  Concepts: {concepts}
  Synthesis concepts: {nav_result.get('synthesis_concepts', [])[:5]}

Quality Metrics:
  Resonance: {q['resonance']:.0%}
  C_synthesis: {m['C']:.3f}
  C_landscape: {q['coherence_landscape']:.3f}
  Power: {m['P']:.2f}
  Meaning (C×P): {m['meaning']:.2f}
  Σ (C + 0.1P): {m['sigma']:.2f}
  Balance ratio: {m['balance_ratio']:.2f}
  Efficiency: {m['efficiency']:.1%}{paradox_info}

Respond to the question using the navigated concepts."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text


def run_dialogue_comparison():
    """Compare WISDOM and POWERFUL in full dialogues."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        return None

    print("=" * 70)
    print("WISDOM vs POWERFUL Dialogue Comparison")
    print("=" * 70)
    print("\nOptimal target: C=0.615, P=6.15, Meaning=3.78")
    print()

    questions = [
        "What is wisdom?",
        "What is the meaning of life?",
        "What is consciousness?",
        "What is love?",
        "What is truth?",
    ]

    nav = SemanticNavigator()

    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'title': 'WISDOM vs POWERFUL Dialogue Comparison',
        'comparisons': [],
        'summary': {}
    }

    wisdom_total_meaning = 0
    powerful_total_meaning = 0

    try:
        for question in questions:
            print(f"\n{'=' * 60}")
            print(f"  {question}")
            print(f"{'=' * 60}")

            # Get navigation results
            wisdom_nav = test_single_response(nav, question, "wisdom")
            powerful_nav = test_single_response(nav, question, "powerful")

            # Get Claude responses
            print("\n  Generating Claude responses...")
            wisdom_response = test_with_claude(question, wisdom_nav)
            powerful_response = test_with_claude(question, powerful_nav)

            # Print comparison with consistent metrics
            wm = wisdom_nav['meaning_metrics']
            pm = powerful_nav['meaning_metrics']

            print(f"\n  WISDOM Mode:")
            print(f"    C_synth={wm['C']:.3f}, P={wm['P']:.2f}")
            print(f"    Meaning={wm['meaning']:.3f}, Σ={wm['sigma']:.2f}, Balance={wm['balance_ratio']:.2f}")
            print(f"    Efficiency: {wm['efficiency']:.1%}")
            print(f"    Synthesis: {wisdom_nav.get('synthesis_concepts', [])[:4]}")
            if wisdom_nav.get('paradox'):
                print(f"    Paradox: {wisdom_nav['paradox']['thesis']} ↔ {wisdom_nav['paradox']['antithesis']}")
            print(f"\n    Response: {wisdom_response[:200]}...")

            print(f"\n  POWERFUL Mode:")
            print(f"    C_synth={pm['C']:.3f}, P={pm['P']:.2f}")
            print(f"    Meaning={pm['meaning']:.3f}, Σ={pm['sigma']:.2f}, Balance={pm['balance_ratio']:.2f}")
            print(f"    Efficiency: {pm['efficiency']:.1%}")
            print(f"    Synthesis: {powerful_nav.get('synthesis_concepts', [])[:4]}")
            if powerful_nav.get('paradox'):
                print(f"    Paradox: {powerful_nav['paradox']['thesis']} ↔ {powerful_nav['paradox']['antithesis']}")
            print(f"\n    Response: {powerful_response[:200]}...")

            # Accumulate
            wisdom_total_meaning += wisdom_nav['meaning_metrics']['meaning']
            powerful_total_meaning += powerful_nav['meaning_metrics']['meaning']

            # Store
            results['comparisons'].append({
                'question': question,
                'wisdom': {
                    'navigation': wisdom_nav,
                    'response': wisdom_response
                },
                'powerful': {
                    'navigation': powerful_nav,
                    'response': powerful_response
                }
            })

        # Summary
        n = len(questions)
        results['summary'] = {
            'n_questions': n,
            'wisdom_avg_meaning': wisdom_total_meaning / n,
            'powerful_avg_meaning': powerful_total_meaning / n,
        }

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nQuestions: {n}")
        print(f"WISDOM avg meaning: {wisdom_total_meaning/n:.2f}")
        print(f"POWERFUL avg meaning: {powerful_total_meaning/n:.2f}")

        # Save
        results_dir = _PHYSICS_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / f"wisdom_dialogue_{results['timestamp']}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

        return results

    finally:
        nav.close()


if __name__ == "__main__":
    run_dialogue_comparison()
