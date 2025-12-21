#!/usr/bin/env python3
"""
Visualize Book Journey: Run and visualize a semantic journey through a book.

Usage:
    python visualize_book_journey.py --book heart_of_darkness
    python visualize_book_journey.py --book divine_comedy --steps 100
    python visualize_book_journey.py --book crime_and_punishment --save journey.png
"""

import sys
from pathlib import Path

# Add paths
_THIS_DIR = Path(__file__).parent.resolve()
_SEMANTIC_RL_SRC = _THIS_DIR.parent / "src"
_SEMANTIC_LLM = _THIS_DIR.parent.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_SEMANTIC_RL_SRC))

import numpy as np
from typing import List, Dict

from environment.book_world import BookWorld, BookConfig, get_book_path, CLASSIC_BOOKS
from agents import QuantumAgent
from visualization.journey_viz import JourneyVisualizer, JourneyData, VIZ_OUTPUT_DIR


def run_and_visualize(book_key: str = "heart_of_darkness",
                     max_steps: int = 80,
                     save_dir: str = None,
                     show: bool = True) -> Dict:
    """
    Run a journey through a book and visualize it.

    Args:
        book_key: Key from CLASSIC_BOOKS or filename
        max_steps: Maximum steps for the journey
        save_dir: Directory to save visualizations
        show: Whether to display plots

    Returns:
        Dictionary with journey results
    """
    print("=" * 70)
    print(f"SEMANTIC JOURNEY VISUALIZATION")
    print(f"Book: {book_key}")
    print("=" * 70)

    # Setup save directory (default: semantic_rl/visualizations/)
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = VIZ_OUTPUT_DIR
        save_path.mkdir(exist_ok=True)

    # Load book
    book_path = get_book_path(book_key)
    print(f"\nLoading: {book_path}")

    world = BookWorld(
        book_file=str(book_path),
        book_config=BookConfig(max_states=300, transition_threshold=0.5),
        render_mode=None  # No text rendering during run
    )

    # Create agent
    agent = QuantumAgent(
        believe=0.5,
        temperature=1.0,
        cooling_rate=0.99,
        tunnel_bonus=0.3
    )

    # Run journey and collect data
    print(f"\nRunning journey: {world.start_word} → {world.goal_word}")
    print("-" * 70)

    obs, info = world.reset()
    agent.on_episode_start()

    path = [info["current_word"]]
    rewards = []
    believe_history = [agent.believe]
    tunnel_events = []
    step_details = []

    for step in range(max_steps):
        valid = world.get_valid_actions()
        action = agent.choose_action(obs, valid, info)

        next_obs, reward, term, trunc, info = world.step(action)
        agent.update(obs, action, reward, next_obs, term or trunc, info)

        rewards.append(reward)
        believe_history.append(agent.believe)

        if info.get("success"):
            to_word = info.get("to", "?")
            path.append(to_word)

            action_name = world.action_to_name.get(action, "?")

            if action == 0:  # Tunnel
                tunnel_events.append({
                    "step": step + 1,
                    "from": info.get("from", "?"),
                    "to": to_word,
                    "believe": agent.believe
                })
                print(f"  Step {step+1}: ⚡ {info.get('from', '?')} ══> {to_word}")
            else:
                print(f"  Step {step+1}: {info.get('from', '?')} --{action_name}--> {to_word}")

            step_details.append({
                "step": step + 1,
                "from": info.get("from", "?"),
                "to": to_word,
                "action": action_name if action > 0 else "tunnel",
                "reward": reward,
                "believe": agent.believe
            })

        obs = next_obs

        if term:
            print(f"\n  *** REACHED GOAL: {world.goal_word} ***")
            break

    # Summary
    print("\n" + "=" * 70)
    print("JOURNEY COMPLETE")
    print("=" * 70)
    print(f"Path length: {len(path)} states")
    print(f"Unique states: {len(set(path))}")
    print(f"Tunnel events: {len(tunnel_events)}")
    print(f"Total reward: {sum(rewards):+.2f}")
    print(f"Final believe: {agent.believe:.2f}")

    # Build states info
    states_info = {}
    for word in set(path):
        state = world.graph.get_state(word)
        if state:
            states_info[word] = {
                "tau": state.tau,
                "goodness": state.goodness
            }

    # Create journey data
    journey_data = JourneyData(
        path=path,
        rewards=rewards,
        believe_history=believe_history,
        tunnel_events=tunnel_events,
        states_info=states_info
    )

    # Visualizations
    print("\n" + "-" * 70)
    print("Creating visualizations...")
    print("-" * 70)

    viz = JourneyVisualizer()

    # 1. Semantic graph with path
    graph_file = save_path / f"{book_key}_graph.png"
    print(f"\n1. Semantic landscape: {graph_file}")
    viz.plot_semantic_graph(
        world.graph,
        path=path,
        tunnel_events=tunnel_events,
        title=f"Semantic Journey: {book_key.replace('_', ' ').title()}",
        save_path=str(graph_file),
        show=show
    )

    # 2. Narrative arc
    arc_file = save_path / f"{book_key}_arc.png"
    print(f"2. Narrative arc: {arc_file}")
    viz.plot_narrative_arc(
        path=path,
        states_info=states_info,
        believe_history=believe_history,
        title=f"Narrative Arc: {book_key.replace('_', ' ').title()}",
        save_path=str(arc_file),
        show=show
    )

    # 3. Journey summary
    summary_file = save_path / f"{book_key}_summary.png"
    print(f"3. Journey summary: {summary_file}")
    viz.plot_journey_summary(
        journey_data,
        graph=world.graph,
        title=f"Journey Through {book_key.replace('_', ' ').title()}",
        save_path=str(summary_file),
        show=show
    )

    print("\n" + "=" * 70)
    print(f"Visualizations saved to: {save_path}")
    print("=" * 70)

    return {
        "path": path,
        "rewards": rewards,
        "believe_history": believe_history,
        "tunnel_events": tunnel_events,
        "states_info": states_info,
        "journey_data": journey_data,
        "world": world,
        "visualizations": {
            "graph": str(graph_file),
            "arc": str(arc_file),
            "summary": str(summary_file)
        }
    }


def compare_books(books: List[str], max_steps: int = 60, show: bool = True):
    """
    Run and compare journeys through multiple books.
    """
    print("=" * 70)
    print("COMPARATIVE BOOK JOURNEYS")
    print("=" * 70)

    results = {}
    for book in books:
        print(f"\n{'─' * 70}")
        print(f"Processing: {book}")
        print(f"{'─' * 70}")

        result = run_and_visualize(book, max_steps=max_steps, show=False)
        results[book] = result

    # Comparative visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(books), 1, figsize=(14, 4 * len(books)), sharex=True)
    if len(books) == 1:
        axes = [axes]

    viz = JourneyVisualizer()

    for ax, (book, result) in zip(axes, results.items()):
        path = result["path"]
        states_info = result["states_info"]

        goodness = [states_info.get(w, {}).get("goodness", 0) for w in path]
        steps = range(len(path))
        colors = [viz.goodness_cmap((g + 1) / 2) for g in goodness]

        ax.fill_between(steps, goodness, alpha=0.3, color='green')
        ax.scatter(steps, goodness, c=colors, s=50, edgecolors='black', linewidths=0.5)
        ax.plot(steps, goodness, '-', color='darkgreen', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_ylabel('Goodness')
        ax.set_title(f'{book.replace("_", " ").title()}: {path[0]} → {path[-1]}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Journey Step')

    plt.suptitle('Comparative Semantic Journeys', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = VIZ_OUTPUT_DIR / "comparison.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"\nComparison saved: {save_path}")

    if show:
        plt.show()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize semantic book journey")
    parser.add_argument("--book", default="heart_of_darkness",
                       help=f"Book to visualize. Options: {list(CLASSIC_BOOKS.keys())}")
    parser.add_argument("--steps", type=int, default=80, help="Max journey steps")
    parser.add_argument("--save-dir", default=None, help="Directory to save visualizations")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--compare", nargs="+", default=None,
                       help="Compare multiple books (e.g., --compare heart_of_darkness divine_comedy)")

    args = parser.parse_args()

    if args.compare:
        compare_books(args.compare, max_steps=args.steps, show=not args.no_show)
    else:
        run_and_visualize(
            book_key=args.book,
            max_steps=args.steps,
            save_dir=args.save_dir,
            show=not args.no_show
        )
