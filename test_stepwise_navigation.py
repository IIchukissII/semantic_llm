#!/usr/bin/env python3
"""
Step-by-Step Navigation Test
=============================

Tests the hybrid quantum-LLM architecture with:
1. Long navigation (30 steps)
2. Step-by-step visualization
3. Local LLM rendering (Ollama)
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.hybrid_llm import QuantumCore, OllamaRenderer, Trajectory


def print_step(step_num: int, transition, cumulative_g: float):
    """Print a single navigation step."""
    t = transition
    spin_marker = " [SPIN]" if t.is_spin else ""
    print(f"  Step {step_num:2d}: {t.from_state.word:15s} --[{t.verb:12s}]--> "
          f"{t.to_state.word:15s}  Δg={t.delta_g:+.3f}  g={cumulative_g:+.3f}{spin_marker}")


def visualize_trajectory(trajectory: Trajectory):
    """Visualize full trajectory with energy profile."""
    print("\n" + "=" * 70)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 70)

    print(f"\nStart: {trajectory.start.word} (g={trajectory.start.goodness:+.3f}, τ={trajectory.start.tau:.2f})")
    print(f"Goal:  {trajectory.goal}")
    print(f"Steps: {len(trajectory.transitions)}")
    print(f"Total Δg: {trajectory.total_delta_g:+.3f}")

    print("\n" + "-" * 70)
    print("Step-by-step navigation:")
    print("-" * 70)

    cumulative_g = trajectory.start.goodness
    for i, t in enumerate(trajectory.transitions, 1):
        cumulative_g = t.to_state.goodness
        print_step(i, t, cumulative_g)

    print("\n" + "-" * 70)
    print("Energy diagram:")
    print("-" * 70)
    print(trajectory.energy_diagram())

    print("\n" + "-" * 70)
    print("Word sequence:")
    print("-" * 70)
    seq = trajectory.to_sequence()
    print(" → ".join(seq))


def render_in_chunks(trajectory: Trajectory, renderer: OllamaRenderer, chunk_size: int = 5):
    """Render trajectory in chunks using local LLM."""
    print("\n" + "=" * 70)
    print("LLM RENDERING (chunk by chunk)")
    print("=" * 70)

    seq = trajectory.to_sequence()

    # Split into chunks (each chunk: noun → verb → noun → verb → noun...)
    # We need overlapping chunks for continuity
    chunk_texts = []

    for start_idx in range(0, len(seq) - 2, chunk_size * 2):
        end_idx = min(start_idx + chunk_size * 2 + 1, len(seq))
        chunk_seq = seq[start_idx:end_idx]

        if len(chunk_seq) >= 3:  # Need at least noun-verb-noun
            print(f"\nChunk {len(chunk_texts) + 1}: {' → '.join(chunk_seq)}")

            # Create mini-trajectory for this chunk
            chunk_text = renderer._template_render(
                type('obj', (object,), {
                    'to_sequence': lambda s=chunk_seq: s,
                    'transitions': [],
                    'goal': trajectory.goal
                })(),
                style="narrative"
            )
            print(f"  → {chunk_text}")
            chunk_texts.append(chunk_text)

    return " ".join(chunk_texts)


def main():
    print("=" * 70)
    print("STEP-BY-STEP QUANTUM NAVIGATION TEST")
    print("=" * 70)

    # Initialize
    print("\nInitializing QuantumCore...")
    core = QuantumCore()

    print("\nInitializing OllamaRenderer...")
    renderer = OllamaRenderer(model="qwen2.5:1.5b")

    # Test words
    test_cases = [
        ("war", "good", 30),
        ("fear", "good", 20),
        ("hate", "good", 15),
    ]

    for start_word, goal, steps in test_cases:
        print("\n" + "=" * 70)
        print(f"NAVIGATION: {start_word} → {goal} ({steps} steps)")
        print("=" * 70)

        # Navigate
        trajectory = core.navigate(start_word, goal, steps=steps, temperature=0.3)

        if trajectory:
            # Visualize
            visualize_trajectory(trajectory)

            # Render with LLM
            if renderer.available:
                print("\n" + "-" * 70)
                print("Full LLM rendering:")
                print("-" * 70)
                rendered = renderer.render(trajectory, style="narrative")
                print(rendered)
            else:
                # Use template rendering in chunks
                render_in_chunks(trajectory, renderer, chunk_size=5)
        else:
            print(f"  Could not find path from {start_word}")

        print("\n")


if __name__ == "__main__":
    main()
