#!/usr/bin/env python3
"""
Real Semantic Journey Experiment

Navigate the real 19K word semantic space.
Core principle: "Only believe what was lived is knowledge"
"""

import sys
from pathlib import Path

# Add paths - order matters!
_THIS_DIR = Path(__file__).parent.resolve()
_SEMANTIC_RL_SRC = _THIS_DIR.parent / "src"
_SEMANTIC_LLM = _THIS_DIR.parent.parent.parent  # semantic_llm directory

# Add semantic_llm first (for core.hybrid_llm)
sys.path.insert(0, str(_SEMANTIC_LLM))
# Then add semantic_rl/src
sys.path.insert(0, str(_SEMANTIC_RL_SRC))

import numpy as np
import random

from core.semantic_loader import SemanticLoader
from core.knowledge import KnowledgeBase
from environment import SemanticWorld
from agents import QuantumAgent


def run_journey(start: str = "darkness",
                goal: str = "wisdom",
                max_steps: int = 50,
                believe: float = 0.5):
    """
    Run a journey through real semantic space.

    The agent learns through lived experience:
    - Can only tunnel to states connected to what was lived
    - Believe changes based on state properties
    - Knowledge accumulates along the path
    """
    print("=" * 70)
    print("SEMANTIC RL: REAL JOURNEY")
    print("=" * 70)
    print(f"\nCore principle: 'Only believe what was lived is knowledge'")
    print(f"\nJourney: {start} → {goal}")
    print(f"Initial believe: {believe}")
    print("=" * 70)

    # Load real semantic data
    print("\nLoading semantic space...")
    loader = SemanticLoader()
    graph = loader.build_graph(max_states=500, sample_verbs=200)

    # Create environment with real graph
    env = SemanticWorld(
        semantic_graph=graph,
        start_word=start,
        goal_word=goal,
        render_mode="human"
    )

    # Create agent
    agent = QuantumAgent(
        believe=believe,
        temperature=1.0,
        cooling_rate=0.98,
        tunnel_bonus=0.3
    )

    # Run journey
    obs, info = env.reset()
    agent.on_episode_start()

    print(f"\n{'─' * 70}")
    print("JOURNEY BEGINS")
    print(f"{'─' * 70}")

    total_reward = 0
    step = 0
    done = False

    # Track interesting events
    tunnel_events = []
    path = [start]

    while not done and step < max_steps:
        step += 1
        current = info["current_word"]

        # Get valid actions
        valid_actions = env.get_valid_actions()
        action_names = [env.action_to_name.get(a, "?") for a in valid_actions]

        # Agent chooses
        action = agent.choose_action(obs, valid_actions, info)
        action_name = env.action_to_name.get(action, "unknown")

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update agent
        agent.update(obs, action, reward, next_obs, terminated or truncated, info)

        total_reward += reward

        # Log interesting events
        if info.get("success", False):
            new_word = info.get("to", current)
            path.append(new_word)

            if action == 0:  # Tunnel
                tunnel_events.append({
                    "step": step,
                    "from": current,
                    "to": new_word,
                    "believe": agent.believe
                })
                print(f"\n  Step {step}: ⚡TUNNEL {current} ══> {new_word}")
                print(f"           believe={agent.believe:.2f}, reward={reward:+.2f}")
            else:
                print(f"\n  Step {step}: {current} --{action_name}--> {new_word}")
                print(f"           g={info.get('delta_g', 0):+.2f}, reward={reward:+.2f}")

        obs = next_obs
        done = terminated or truncated

        if terminated:
            print(f"\n  *** REACHED GOAL: {goal} ***")

    # Journey summary
    print(f"\n{'=' * 70}")
    print("JOURNEY COMPLETE")
    print(f"{'=' * 70}")

    print(f"\nPath ({len(path)} states):")
    # Print path in chunks
    for i in range(0, len(path), 5):
        chunk = path[i:i+5]
        print(f"  {' → '.join(chunk)}")

    print(f"\nStatistics:")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {total_reward:+.2f}")
    print(f"  Reached goal: {terminated}")
    print(f"  Final believe: {agent.believe:.2f}")
    print(f"  Tunnel events: {len(tunnel_events)}")

    knowledge = env.knowledge.get_journey_summary()
    print(f"\nKnowledge gained:")
    print(f"  Unique states lived: {knowledge['unique_states']}")
    print(f"  Tunnels discovered: {knowledge['total_tunnels']}")
    print(f"  Connections made: {knowledge['connections']}")

    if tunnel_events:
        print(f"\nTunnel events (insights):")
        for event in tunnel_events:
            print(f"  Step {event['step']}: {event['from']} ══> {event['to']} "
                  f"(believe={event['believe']:.2f})")

    return {
        "path": path,
        "total_reward": total_reward,
        "reached_goal": terminated,
        "tunnel_events": tunnel_events,
        "knowledge": knowledge,
        "final_believe": agent.believe
    }


def explore_semantic_space():
    """Explore the semantic space interactively."""
    print("=" * 70)
    print("SEMANTIC SPACE EXPLORER")
    print("=" * 70)

    loader = SemanticLoader()
    graph = loader.build_graph(max_states=200, sample_verbs=100)

    print("\nAvailable concepts for journey:")
    words = list(graph.states.keys())[:20]
    for i, word in enumerate(words):
        state = graph.get_state(word)
        print(f"  {i+1:2}. {word:15} (g={state.goodness:+.2f}, τ={state.tau:.2f})")

    print("\nSpin pairs (can tunnel between):")
    for w1, w2 in loader.get_spin_pairs()[:10]:
        if w1 in graph.states and w2 in graph.states:
            p = loader.tunnel_probability(w1, w2)
            print(f"  {w1} ↔ {w2} (P={p:.3f})")


def run_multiple_journeys(n_journeys: int = 5):
    """Run multiple journeys and compare results."""
    print("=" * 70)
    print(f"RUNNING {n_journeys} JOURNEYS")
    print("=" * 70)

    results = []
    start_words = ["darkness", "fear", "chaos", "despair", "isolation"]
    goal_words = ["wisdom", "love", "peace", "hope", "connection"]

    for i in range(n_journeys):
        start = random.choice(start_words)
        goal = random.choice(goal_words)
        believe = 0.3 + random.random() * 0.4  # 0.3 to 0.7

        print(f"\n{'─' * 70}")
        print(f"Journey {i+1}: {start} → {goal} (believe={believe:.2f})")
        print(f"{'─' * 70}")

        result = run_journey(start, goal, max_steps=30, believe=believe)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("JOURNEY COMPARISON")
    print("=" * 70)

    for i, result in enumerate(results):
        path = result["path"]
        print(f"\nJourney {i+1}:")
        print(f"  Path: {' → '.join(path[:5])}...")
        print(f"  Reward: {result['total_reward']:+.2f}")
        print(f"  Reached goal: {result['reached_goal']}")
        print(f"  Tunnels: {len(result['tunnel_events'])}")
        print(f"  Final believe: {result['final_believe']:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real semantic journey")
    parser.add_argument("--mode", choices=["journey", "explore", "multi"],
                        default="journey", help="Run mode")
    parser.add_argument("--start", default="darkness", help="Starting concept")
    parser.add_argument("--goal", default="wisdom", help="Goal concept")
    parser.add_argument("--believe", type=float, default=0.5, help="Initial believe")
    parser.add_argument("--steps", type=int, default=50, help="Max steps")

    args = parser.parse_args()

    if args.mode == "journey":
        run_journey(args.start, args.goal, args.steps, args.believe)
    elif args.mode == "explore":
        explore_semantic_space()
    elif args.mode == "multi":
        run_multiple_journeys(5)
