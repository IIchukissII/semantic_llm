#!/usr/bin/env python3
"""
Dialogue: Conversation between two semantic agents.

Each agent uses Storm-Logos to process the other's message
and generate a response through meaning space.

Usage:
    python dialogue.py [--exchanges 5] [--topic "What is consciousness?"]
"""

import sys
from pathlib import Path
from typing import Optional
import argparse

# Add paths
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _APP_DIR.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.decomposer import Decomposer
from chain_core.storm_logos import StormLogosBuilder, LogosPattern
from chain_core.renderer import Renderer, RendererConfig
from models.types import MeaningTree


class SemanticAgent:
    """An agent that thinks through meaning space."""

    def __init__(self, name: str, temperature: float = 0.7,
                 storm_temperature: float = 1.5):
        self.name = name
        self.loader = DataLoader()
        self.decomposer = Decomposer(self.loader)
        self.storm_logos = StormLogosBuilder(
            storm_temperature=storm_temperature,
            n_walks=5,
            steps_per_walk=8
        )
        self.renderer = Renderer(RendererConfig(
            model="mistral:7b",
            temperature=temperature,
            max_tokens=256
        ))
        self.history = []

    def think(self, message: str) -> dict:
        """Process message through storm-logos and generate response."""
        # Decompose
        decomposed = self.decomposer.decompose(message)

        if not decomposed.nouns:
            return {
                'response': "I sense something but cannot grasp it clearly.",
                'pattern': None,
                'tree': None
            }

        # Storm-Logos
        tree, pattern = self.storm_logos.build(
            decomposed.nouns,
            decomposed.verbs,
            message
        )

        # Render response
        result = self.renderer.render(tree, message)

        self.history.append({
            'input': message,
            'response': result['response'],
            'pattern': pattern
        })

        return {
            'response': result['response'],
            'pattern': pattern,
            'tree': tree
        }

    def close(self):
        self.storm_logos.close()


def run_dialogue(topic: str, exchanges: int = 5, verbose: bool = True):
    """Run a dialogue between two semantic agents."""

    print("=" * 70)
    print("SEMANTIC DIALOGUE")
    print("Two minds exploring meaning space")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print(f"Exchanges: {exchanges}")
    print()

    # Create two agents with slightly different temperatures
    agent_a = SemanticAgent("Seeker", temperature=0.7, storm_temperature=1.5)
    agent_b = SemanticAgent("Guide", temperature=0.8, storm_temperature=1.3)

    current_message = topic
    current_speaker = agent_a
    other_speaker = agent_b

    try:
        for i in range(exchanges):
            print("-" * 70)
            print(f"Exchange {i + 1}/{exchanges}")
            print("-" * 70)

            # Current speaker responds
            print(f"\n[{current_speaker.name}] processing...")

            result = current_speaker.think(current_message)

            if verbose and result['pattern']:
                p = result['pattern']
                print(f"  Storm → Logos:")
                print(f"    Convergence: {p.convergence_point}")
                print(f"    Core: {p.core_concepts[:4]}")
                print(f"    Coherence: {p.coherence:.0%}")
                print(f"    G: {p.g_direction:+.2f}")

            response = result['response']
            print(f"\n{current_speaker.name}:")
            print(f"  {response}")

            # Swap speakers
            current_message = response
            current_speaker, other_speaker = other_speaker, current_speaker

        print("\n" + "=" * 70)
        print("DIALOGUE COMPLETE")
        print("=" * 70)

        # Summary
        print("\nSeeker's journey:")
        for h in agent_a.history:
            if h['pattern']:
                print(f"  → {h['pattern'].convergence_point} (coh: {h['pattern'].coherence:.0%})")

        print("\nGuide's journey:")
        for h in agent_b.history:
            if h['pattern']:
                print(f"  → {h['pattern'].convergence_point} (coh: {h['pattern'].coherence:.0%})")

    finally:
        agent_a.close()
        agent_b.close()


def main():
    parser = argparse.ArgumentParser(description="Semantic dialogue between agents")
    parser.add_argument("--exchanges", "-e", type=int, default=5,
                        help="Number of exchanges (default: 5)")
    parser.add_argument("--topic", "-t", type=str,
                        default="What is the meaning of love and how does it transform us?",
                        help="Starting topic")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Less verbose output")

    args = parser.parse_args()

    run_dialogue(
        topic=args.topic,
        exchanges=args.exchanges,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
