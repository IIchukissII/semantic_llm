#!/usr/bin/env python3
"""Basic tests for the semantic RL environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


def test_semantic_state():
    """Test SemanticState creation and properties."""
    from core.semantic_state import SemanticState

    state = SemanticState(
        word="courage",
        tau=1.2,
        goodness=0.8,
        j_vector=np.random.randn(16)
    )

    print(f"Created: {state}")
    print(f"  altitude: {state.altitude:.2f}")
    print(f"  luminance: {state.luminance:.2f}")
    print(f"  mass: {state.mass:.2f}")
    print(f"  friction: {state.friction:.2f}")

    # Test tunnel probability
    other = SemanticState(
        word="fear",
        tau=0.6,
        goodness=0.3,
        j_vector=np.random.randn(16)
    )

    prob = state.tunnel_probability(other)
    print(f"Tunnel probability to 'fear': {prob:.3f}")

    return True


def test_knowledge_base():
    """Test KnowledgeBase for lived experience."""
    from core.knowledge import KnowledgeBase

    kb = KnowledgeBase()

    # Record visits
    kb.record_visit("darkness")
    kb.record_visit("fear")
    kb.record_visit("struggle")

    print(f"Knowledge: {kb}")
    print(f"Has lived 'fear': {kb.has_lived('fear')}")
    print(f"Has lived 'wisdom': {kb.has_lived('wisdom')}")
    print(f"Knowledge of 'fear': {kb.knowledge_of('fear'):.2f}")
    print(f"Knowledge of 'wisdom': {kb.knowledge_of('wisdom'):.2f}")

    # Record tunnel
    kb.record_tunnel("struggle", "hope", 0.8, 0.5)

    print(f"\nAfter tunnel:")
    print(f"Knowledge of 'hope': {kb.knowledge_of('hope'):.2f}")
    print(f"Can tunnel to 'hope': {kb.can_tunnel_to('hope')}")

    print(f"\nJourney: {kb.get_journey_summary()}")

    return True


def test_environment():
    """Test SemanticWorld environment."""
    from environment import SemanticWorld

    env = SemanticWorld(
        start_word="darkness",
        goal_word="wisdom"
    )

    obs, info = env.reset()

    print(f"Environment created")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Current state: {info['current_word']}")
    print(f"Valid actions: {env.get_valid_actions()}")

    # Take a few random steps
    for i in range(5):
        action = np.random.choice(env.get_valid_actions())
        action_name = env.action_to_name.get(action, "?")

        obs, reward, term, trunc, info = env.step(action)

        print(f"\nStep {i+1}: action={action_name}")
        print(f"  Result: reward={reward:.2f}, state={info['current_word']}")
        print(f"  Believe: {info['believe']:.2f}, Temp: {info['temperature']:.2f}")

        if term:
            print("  *** REACHED GOAL ***")
            break

    return True


def test_quantum_agent():
    """Test QuantumAgent."""
    from environment import SemanticWorld
    from agents import QuantumAgent

    env = SemanticWorld()
    agent = QuantumAgent(believe=0.5, temperature=1.0)

    obs, info = env.reset()

    print(f"Agent: {agent}")

    # Run a few steps
    for i in range(10):
        valid = env.get_valid_actions()
        action = agent.choose_action(obs, valid, info)
        action_name = env.action_to_name.get(action, "?")

        next_obs, reward, term, trunc, info = env.step(action)

        agent.update(obs, action, reward, next_obs, term or trunc, info)

        print(f"Step {i+1}: {action_name} -> {info.get('current_word', '?')}, "
              f"reward={reward:.2f}")

        obs = next_obs

        if term:
            break

    print(f"\nFinal agent stats: {agent.get_stats()}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC RL: TESTS")
    print("=" * 60)

    tests = [
        ("SemanticState", test_semantic_state),
        ("KnowledgeBase", test_knowledge_base),
        ("SemanticWorld", test_environment),
        ("QuantumAgent", test_quantum_agent),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'─' * 60}")
        print(f"Testing: {name}")
        print(f"{'─' * 60}")
        try:
            success = test_fn()
            results.append((name, success))
            print(f"\n✓ {name}: PASSED")
        except Exception as e:
            results.append((name, False))
            print(f"\n✗ {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
