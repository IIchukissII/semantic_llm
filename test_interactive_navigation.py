#!/usr/bin/env python3
"""
Interactive Step-by-Step Navigation
====================================

Navigate the semantic space step by step, seeing each decision.
Uses local LLM (Ollama) for rendering after each step.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.hybrid_llm import QuantumCore, OllamaRenderer, Trajectory, Transition
import numpy as np


class InteractiveNavigator:
    """Interactive step-by-step navigator with LLM rendering."""

    def __init__(self, model: str = "qwen2.5:1.5b"):
        print("Initializing...")
        self.core = QuantumCore()
        self.renderer = OllamaRenderer(model=model)
        print(f"Ready! Loaded {len(self.core.states)} states, {len(self.core.verb_objects)} verbs")

    def show_state(self, state, step: int = 0):
        """Display current state."""
        print(f"\n{'='*60}")
        print(f"Step {step}: {state.word}")
        print(f"{'='*60}")
        print(f"  Position: τ={state.tau:.2f} (abstraction)")
        print(f"  Goodness: g={state.goodness:+.3f}")
        print(f"  j-vector: [{', '.join(f'{v:.2f}' for v in state.j)}]")

    def show_options(self, transitions, top_n: int = 10):
        """Show top transition options."""
        # Sort by delta_g (toward good)
        sorted_trans = sorted(transitions, key=lambda t: t.delta_g, reverse=True)[:top_n]

        print(f"\nTop {len(sorted_trans)} options (sorted by Δg):")
        print("-" * 60)

        for i, t in enumerate(sorted_trans, 1):
            spin = " [SPIN]" if t.is_spin else ""
            print(f"  {i:2d}. {t.verb:12s} → {t.to_state.word:15s}  "
                  f"Δg={t.delta_g:+.3f}  τ={t.to_state.tau:.1f}{spin}")

        return sorted_trans

    def render_path(self, trajectory: Trajectory, style: str = "poetic"):
        """Render current path with LLM."""
        if not trajectory.transitions:
            return

        print(f"\n{'─'*60}")
        print("Current path:")
        seq = trajectory.to_sequence()
        print(f"  {' → '.join(seq)}")

        if self.renderer.available:
            print(f"\nLLM rendering ({style}):")
            text = self.renderer.render(trajectory, style=style)
            print(f"  \"{text}\"")
        else:
            print(f"\nTemplate rendering:")
            text = self.renderer._template_render(trajectory, style)
            print(f"  \"{text}\"")

    def navigate_interactive(self, start: str, goal: str = "good", max_steps: int = 30):
        """Interactive navigation with step-by-step control."""
        state = self.core.get_state(start)
        if not state:
            print(f"Unknown word: {start}")
            return None

        trajectory = Trajectory(start=state, transitions=[], goal=goal)
        visited = {start}

        print(f"\n{'#'*60}")
        print(f"# NAVIGATION: {start} → {goal}")
        print(f"# Max steps: {max_steps}")
        print(f"{'#'*60}")

        for step in range(max_steps):
            self.show_state(state, step)

            # Get and show options
            all_trans = self.core.get_transitions(state)
            all_trans = [t for t in all_trans if t.to_state.word not in visited]

            if not all_trans:
                print("\nNo more transitions available!")
                break

            options = self.show_options(all_trans)

            # Auto-select best option (could be made interactive)
            # For now: temperature-based selection
            scores = np.array([t.delta_g for t in options])
            temperature = 0.3
            exp_scores = np.exp((scores - np.max(scores)) / temperature)
            probs = exp_scores / exp_scores.sum()
            idx = np.random.choice(len(options), p=probs)

            chosen = options[idx]
            print(f"\n→ Selected: {chosen.verb} → {chosen.to_state.word} (Δg={chosen.delta_g:+.3f})")

            trajectory.transitions.append(chosen)
            visited.add(chosen.to_state.word)
            state = chosen.to_state

            # Render current path every 5 steps
            if (step + 1) % 5 == 0:
                self.render_path(trajectory)

        # Final summary
        print(f"\n{'='*60}")
        print("FINAL TRAJECTORY")
        print(f"{'='*60}")
        print(f"Steps: {len(trajectory.transitions)}")
        print(f"Total Δg: {trajectory.total_delta_g:+.3f}")
        print(f"Start g: {trajectory.start.goodness:+.3f}")
        print(f"End g:   {trajectory.end.goodness:+.3f}")

        self.render_path(trajectory, style="narrative")

        return trajectory

    def navigate_greedy(self, start: str, goal: str = "good", steps: int = 30):
        """Greedy navigation - always pick best option."""
        state = self.core.get_state(start)
        if not state:
            print(f"Unknown word: {start}")
            return None

        trajectory = Trajectory(start=state, transitions=[], goal=goal)
        visited = {start}

        print(f"\nGreedy navigation: {start} → {goal} ({steps} steps)")
        print("-" * 60)

        for step in range(steps):
            all_trans = self.core.get_transitions(state)
            all_trans = [t for t in all_trans if t.to_state.word not in visited]

            if not all_trans:
                break

            # Greedy: always pick highest Δg
            best = max(all_trans, key=lambda t: t.delta_g)

            spin = " [SPIN]" if best.is_spin else ""
            print(f"  {step+1:2d}. {state.word:12s} --{best.verb:10s}--> "
                  f"{best.to_state.word:12s}  Δg={best.delta_g:+.3f}{spin}")

            trajectory.transitions.append(best)
            visited.add(best.to_state.word)
            state = best.to_state

        print(f"\nTotal Δg: {trajectory.total_delta_g:+.3f}")
        self.render_path(trajectory, style="narrative")

        return trajectory


def main():
    nav = InteractiveNavigator()

    # Test cases
    test_words = ["war", "fear", "darkness", "pain", "anger"]

    for word in test_words:
        print("\n" + "#" * 70)
        trajectory = nav.navigate_greedy(word, "good", steps=15)
        print()


if __name__ == "__main__":
    main()
