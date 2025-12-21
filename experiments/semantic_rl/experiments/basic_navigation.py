#!/usr/bin/env python3
"""
Basic Navigation Experiment

Test the semantic RL environment with different agents.
Demonstrates the core principle: "Only believe what was lived is knowledge"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from typing import Dict, List

from environment import SemanticWorld
from agents import QuantumAgent
from agents.base_agent import RandomAgent


def run_episode(env: SemanticWorld, agent, render: bool = False) -> Dict:
    """Run a single episode."""
    obs, info = env.reset()
    agent.on_episode_start()

    total_reward = 0
    steps = 0
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(obs, valid_actions, info)

        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, next_obs, terminated or truncated, info)

        total_reward += reward
        steps += 1
        obs = next_obs
        done = terminated or truncated

        if render:
            env.render()

    agent.on_episode_end(total_reward, steps)

    return {
        "total_reward": total_reward,
        "steps": steps,
        "reached_goal": terminated,
        "final_word": info.get("current_word", "unknown"),
        "knowledge": env.knowledge.get_journey_summary(),
    }


def run_experiment(num_episodes: int = 100, render_interval: int = 10):
    """Run full experiment comparing agents."""

    print("=" * 60)
    print("SEMANTIC RL: BASIC NAVIGATION EXPERIMENT")
    print("=" * 60)
    print("\nCore principle: 'Only believe what was lived is knowledge'")
    print("\nGoal: Navigate from 'darkness' to 'wisdom'")
    print("=" * 60)

    # Create environment
    env = SemanticWorld(
        start_word="darkness",
        goal_word="wisdom",
        render_mode="human"
    )

    # Create agents
    agents = {
        "Quantum": QuantumAgent(believe=0.5, temperature=1.0),
        "Random": RandomAgent(),
    }

    results = {name: [] for name in agents}

    for name, agent in agents.items():
        print(f"\n{'─' * 60}")
        print(f"Testing {name} Agent")
        print(f"{'─' * 60}")

        for episode in range(num_episodes):
            render = (episode % render_interval == 0) and (episode > 0)

            result = run_episode(env, agent, render=render)
            results[name].append(result)

            if episode % 10 == 0:
                recent_rewards = [r["total_reward"] for r in results[name][-10:]]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean([r["reached_goal"] for r in results[name][-10:]])

                print(f"  Episode {episode}: "
                      f"avg_reward={avg_reward:.2f}, "
                      f"success_rate={success_rate:.0%}")

                if hasattr(agent, "believe"):
                    print(f"    believe={agent.believe:.2f}, "
                          f"T={agent.temperature:.3f}")

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for name, agent_results in results.items():
        total_rewards = [r["total_reward"] for r in agent_results]
        success_rate = np.mean([r["reached_goal"] for r in agent_results])
        avg_steps = np.mean([r["steps"] for r in agent_results])

        print(f"\n{name} Agent:")
        print(f"  Average reward: {np.mean(total_rewards):.3f}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average steps: {avg_steps:.1f}")

        if name == "Quantum":
            agent = agents[name]
            stats = agent.get_stats()
            print(f"  Final believe: {stats['believe']:.2f}")
            print(f"  Tunnel success rate: {stats['tunnel_success_rate']:.1%}")
            print(f"  Lived words: {stats['lived_words']}")

    # Show final knowledge from best run
    print("\n" + "=" * 60)
    print("KNOWLEDGE ANALYSIS (Quantum Agent)")
    print("=" * 60)

    # Run one more episode with full rendering
    print("\nFinal episode with rendering:")
    env_render = SemanticWorld(
        start_word="darkness",
        goal_word="wisdom",
        render_mode="human"
    )

    final_result = run_episode(env_render, agents["Quantum"], render=True)

    print(f"\nJourney summary:")
    for key, value in final_result["knowledge"].items():
        print(f"  {key}: {value}")


def demo_simple():
    """Simple demo showing core concepts."""

    print("=" * 60)
    print("SEMANTIC RL: SIMPLE DEMO")
    print("=" * 60)
    print("\nCore principle: 'Only believe what was lived is knowledge'")
    print("\nThis demo shows how an agent learns through lived experience.")
    print("=" * 60)

    # Create environment
    env = SemanticWorld(
        start_word="darkness",
        goal_word="wisdom",
        render_mode="human"
    )

    # Create agent
    agent = QuantumAgent(believe=0.5, temperature=1.0)

    print("\nStarting from 'darkness', goal is 'wisdom'")
    print("Agent has believe=0.5, temperature=1.0")

    obs, info = env.reset()
    env.render()

    steps = 0
    max_steps = 20

    while steps < max_steps:
        valid_actions = env.get_valid_actions()

        print(f"\nValid actions: {[env.action_to_name.get(a, a) for a in valid_actions]}")

        action = agent.choose_action(obs, valid_actions, info)
        action_name = env.action_to_name.get(action, "unknown")
        print(f"Agent chooses: {action_name}")

        next_obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Result: reward={reward:.2f}, success={info.get('success', False)}")

        if info.get("success", False):
            print(f"  Moved: {info.get('from', '?')} -> {info.get('to', '?')}")

        agent.update(obs, action, reward, next_obs, terminated or truncated, info)

        env.render()

        obs = next_obs
        steps += 1

        if terminated:
            print(f"\n*** REACHED GOAL: wisdom ***")
            break

    print(f"\nFinal agent stats: {agent.get_stats()}")
    print(f"Knowledge: {env.knowledge.get_journey_summary()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic RL experiments")
    parser.add_argument("--mode", choices=["demo", "experiment"], default="demo",
                        help="Run mode: demo (simple) or experiment (full)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes for experiment")

    args = parser.parse_args()

    if args.mode == "demo":
        demo_simple()
    else:
        run_experiment(num_episodes=args.episodes)
