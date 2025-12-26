#!/usr/bin/env python3
"""
Dialogue between Meaning Chain (Storm-Logos) and Claude.

One agent uses semantic space navigation (local),
the other uses Claude API (cloud).

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python dialogue_claude.py --exchanges 5
"""

import sys
import os
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
from chain_core.storm_logos import StormLogosBuilder
from chain_core.renderer import Renderer, RendererConfig


class MeaningChainAgent:
    """Agent that thinks through semantic space."""

    def __init__(self, name: str = "Semantic", gravity_strength: float = 0.0):
        """
        Args:
            name: Agent identifier
            gravity_strength: Physics mode [0-1]
                - 0.0: Standard mode
                - 0.5: Recommended gravity mode (grounded responses)
        """
        self.name = name
        self.gravity_strength = gravity_strength
        self.loader = DataLoader()
        self.decomposer = Decomposer(self.loader)
        self.storm_logos = StormLogosBuilder(
            storm_temperature=1.5,
            n_walks=5,
            steps_per_walk=8,
            gravity_strength=gravity_strength
        )
        self.renderer = Renderer(RendererConfig(
            model="mistral:7b",
            temperature=0.7,
            max_tokens=300
        ))

    def respond(self, message: str) -> dict:
        """Process through storm-logos."""
        decomposed = self.decomposer.decompose(message)

        if not decomposed.nouns:
            return {
                'response': "The meaning escapes me in this moment.",
                'pattern': None
            }

        tree, pattern = self.storm_logos.build(
            decomposed.nouns,
            decomposed.verbs,
            message
        )

        result = self.renderer.render(tree, message)

        return {
            'response': result['response'],
            'pattern': pattern,
            'convergence': pattern.convergence_point if pattern else None,
            'coherence': pattern.coherence if pattern else 0
        }

    def close(self):
        self.storm_logos.close()


class ClaudeAgent:
    """Agent using Claude API."""

    def __init__(self, name: str = "Claude", model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = None
        self.history = []

        # System prompt for philosophical dialogue
        self.system = """You are engaged in a philosophical dialogue about meaning, consciousness, and human experience.
Your partner thinks through semantic space - navigating concepts like 'love', 'fear', 'meaning', 'truth'.
Respond thoughtfully and concisely (2-4 sentences).
Build on what they say, ask questions, or offer new perspectives.
Be genuine and exploratory, not didactic."""

    def _init_client(self):
        if self.client is None:
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("pip install anthropic")

    def respond(self, message: str) -> dict:
        """Get response from Claude."""
        self._init_client()

        # Add to history
        self.history.append({"role": "user", "content": message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=self.system,
            messages=self.history
        )

        reply = response.content[0].text
        self.history.append({"role": "assistant", "content": reply})

        return {
            'response': reply,
            'model': self.model
        }


def run_dialogue(topic: str, exchanges: int = 5, verbose: bool = True,
                 gravity_strength: float = 0.0, save_results: bool = True):
    """Run dialogue between MeaningChain and Claude."""
    import json
    from datetime import datetime

    mode = "GRAVITY" if gravity_strength > 0 else "STANDARD"

    print("=" * 70)
    print(f"SEMANTIC DIALOGUE: Meaning Chain ↔ Claude ({mode} MODE)")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print(f"Exchanges: {exchanges}")
    if gravity_strength > 0:
        print(f"Gravity: α={gravity_strength}")
    print()

    # Create agents
    semantic = MeaningChainAgent("Semantic", gravity_strength=gravity_strength)
    claude = ClaudeAgent("Claude")

    # Track dialogue for saving
    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "gravity_strength": gravity_strength,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    # Claude starts by responding to the topic
    current_message = topic
    speakers = [(claude, semantic), (semantic, claude)]  # Alternating pairs

    try:
        for i in range(exchanges):
            speaker, listener = speakers[i % 2]

            print("-" * 70)
            print(f"Exchange {i + 1}/{exchanges}: {speaker.name}")
            print("-" * 70)

            result = speaker.respond(current_message)
            response = result['response']

            # Build turn record
            turn = {
                "exchange": i + 1,
                "speaker": speaker.name,
                "input": current_message,
                "response": response
            }

            if verbose and speaker.name == "Semantic" and result.get('pattern'):
                p = result['pattern']
                print(f"\n  [Storm-Logos]")
                print(f"    Convergence: {p.convergence_point}")
                print(f"    Core: {p.core_concepts[:4]}")
                print(f"    Coherence: {p.coherence:.0%}")
                print(f"    τ-level: {p.tau_level:.1f}, G: {p.g_direction:+.2f}")
                if gravity_strength > 0:
                    print(f"    Physics: realm={p.realm}, φ={p.avg_phi:.2f}, compliance={p.gravity_compliance:.0%}")

                # Add pattern info to turn
                turn["pattern"] = {
                    "convergence": p.convergence_point,
                    "core_concepts": p.core_concepts[:5],
                    "coherence": p.coherence,
                    "tau_level": p.tau_level,
                    "g_direction": p.g_direction
                }
                if gravity_strength > 0:
                    turn["pattern"]["physics"] = {
                        "realm": p.realm,
                        "avg_phi": p.avg_phi,
                        "gravity_compliance": p.gravity_compliance,
                        "veil_crossings": p.veil_crossings
                    }

            dialogue_log["turns"].append(turn)

            print(f"\n{speaker.name}:")
            # Wrap long lines
            words = response.split()
            line = "  "
            for word in words:
                if len(line) + len(word) > 75:
                    print(line)
                    line = "  " + word
                else:
                    line += " " + word if line != "  " else word
            if line.strip():
                print(line)

            current_message = response

        print("\n" + "=" * 70)
        print("DIALOGUE COMPLETE")
        print("=" * 70)

        # Save results
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_claude"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gravity_tag = f"_gravity{gravity_strength}" if gravity_strength > 0 else ""
            filename = results_dir / f"dialogue{gravity_tag}_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(dialogue_log, f, indent=2, default=str)

            print(f"\nResults saved to: {filename}")

    finally:
        semantic.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchanges", "-e", type=int, default=5)
    parser.add_argument("--topic", "-t", type=str,
                        default="What does it mean to truly understand something - not just know it intellectually, but to understand it with one's whole being?")
    parser.add_argument("--gravity", "-g", type=float, default=0.0,
                        help="Gravity strength [0-1] (default: 0, use 0.5 for grounded mode)")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    run_dialogue(
        topic=args.topic,
        exchanges=args.exchanges,
        verbose=not args.quiet,
        gravity_strength=args.gravity,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
