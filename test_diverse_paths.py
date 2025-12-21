#!/usr/bin/env python3
"""
Diverse Path Navigation
=======================

Tests different navigation strategies to find more interesting paths.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.hybrid_llm import QuantumCore, OllamaRenderer, Trajectory
import numpy as np


def navigate_with_momentum(core: QuantumCore, start: str, goal: str = "good",
                           steps: int = 15, momentum: float = 0.3) -> Trajectory:
    """Navigate with momentum - prefer continuing in same direction."""
    state = core.get_state(start)
    if not state:
        return None

    trajectory = Trajectory(start=state, transitions=[], goal=goal)
    visited = {start}
    last_verb = None

    for step in range(steps):
        all_trans = core.get_transitions(state)
        all_trans = [t for t in all_trans if t.to_state.word not in visited]

        if not all_trans:
            break

        # Score: Δg + momentum bonus for same verb
        scores = []
        for t in all_trans:
            score = t.delta_g
            if last_verb and t.verb == last_verb:
                score += momentum
            scores.append(score)

        # Softmax selection
        scores = np.array(scores)
        exp_scores = np.exp((scores - np.max(scores)) / 0.5)
        probs = exp_scores / exp_scores.sum()
        idx = np.random.choice(len(all_trans), p=probs)

        chosen = all_trans[idx]
        trajectory.transitions.append(chosen)
        visited.add(chosen.to_state.word)
        state = chosen.to_state
        last_verb = chosen.verb

    return trajectory


def navigate_with_tau_constraint(core: QuantumCore, start: str, goal: str = "good",
                                  steps: int = 15, target_tau: float = 3.0) -> Trajectory:
    """Navigate while staying near a target abstraction level."""
    state = core.get_state(start)
    if not state:
        return None

    trajectory = Trajectory(start=state, transitions=[], goal=goal)
    visited = {start}

    for step in range(steps):
        all_trans = core.get_transitions(state)
        all_trans = [t for t in all_trans if t.to_state.word not in visited]

        if not all_trans:
            break

        # Score: Δg - penalty for moving away from target τ
        scores = []
        for t in all_trans:
            tau_penalty = abs(t.to_state.tau - target_tau) * 0.3
            score = t.delta_g - tau_penalty
            scores.append(score)

        # Select best
        idx = np.argmax(scores)
        chosen = all_trans[idx]
        trajectory.transitions.append(chosen)
        visited.add(chosen.to_state.word)
        state = chosen.to_state

    return trajectory


def navigate_exploratory(core: QuantumCore, start: str, goal: str = "good",
                         steps: int = 15) -> Trajectory:
    """Exploratory navigation - high temperature, diverse paths."""
    state = core.get_state(start)
    if not state:
        return None

    trajectory = Trajectory(start=state, transitions=[], goal=goal)
    visited = {start}

    for step in range(steps):
        all_trans = core.get_transitions(state)
        all_trans = [t for t in all_trans if t.to_state.word not in visited]

        if not all_trans:
            break

        # High temperature selection
        scores = np.array([t.delta_g for t in all_trans])
        temperature = 1.0  # High temperature = more random
        exp_scores = np.exp((scores - np.max(scores)) / temperature)
        probs = exp_scores / exp_scores.sum()
        idx = np.random.choice(len(all_trans), p=probs)

        chosen = all_trans[idx]
        trajectory.transitions.append(chosen)
        visited.add(chosen.to_state.word)
        state = chosen.to_state

    return trajectory


def display_trajectory(trajectory: Trajectory, renderer: OllamaRenderer, title: str):
    """Display trajectory with rendering."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"Steps: {len(trajectory.transitions)}, Total Δg: {trajectory.total_delta_g:+.3f}")
    print(f"Start: {trajectory.start.word} (g={trajectory.start.goodness:+.3f})")
    print(f"End:   {trajectory.end.word} (g={trajectory.end.goodness:+.3f})")

    print("\nPath:")
    for i, t in enumerate(trajectory.transitions, 1):
        spin = " [SPIN]" if t.is_spin else ""
        print(f"  {i:2d}. {t.from_state.word:12s} --{t.verb:10s}--> "
              f"{t.to_state.word:12s}  Δg={t.delta_g:+.3f}  τ={t.to_state.tau:.1f}{spin}")

    seq = trajectory.to_sequence()
    print(f"\nSequence: {' → '.join(seq)}")

    if renderer.available:
        print(f"\nLLM Rendering:")
        text = renderer.render(trajectory, style="poetic")
        print(f"  {text}")


def main():
    print("Loading...")
    core = QuantumCore()
    renderer = OllamaRenderer(model="qwen2.5:1.5b")

    start_words = ["war", "hate", "fear"]

    for start in start_words:
        print(f"\n{'#'*70}")
        print(f"# Starting from: {start}")
        print(f"{'#'*70}")

        # Try different strategies
        strategies = [
            ("Greedy (τ ≈ 2.5)", lambda: navigate_with_tau_constraint(core, start, "good", 10, 2.5)),
            ("Greedy (τ ≈ 4.0)", lambda: navigate_with_tau_constraint(core, start, "good", 10, 4.0)),
            ("Momentum", lambda: navigate_with_momentum(core, start, "good", 10)),
            ("Exploratory", lambda: navigate_exploratory(core, start, "good", 10)),
        ]

        for name, nav_func in strategies:
            traj = nav_func()
            if traj and traj.transitions:
                display_trajectory(traj, renderer, f"{start} → good ({name})")


if __name__ == "__main__":
    main()
