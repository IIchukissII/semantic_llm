#!/usr/bin/env python3
"""
Euler-Aware Dialogue between Meaning Chain and Claude.

Uses the discovered orbital structure (tau_n = 1 + n/e) for navigation.
Tracks orbital statistics, veil crossings, and Boltzmann dynamics.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python dialogue_claude_euler.py --exchanges 5
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import argparse
import numpy as np

# Add paths
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _APP_DIR.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.decomposer import Decomposer
from chain_core.euler_navigation import (
    EulerNavigator, EulerAwareStorm, OrbitalState,
    KT_NATURAL, VEIL_TAU, GROUND_STATE_TAU, E
)
from chain_core.renderer import Renderer, RendererConfig


class EulerMeaningChainAgent:
    """Agent that navigates semantic space using Euler orbital physics."""

    def __init__(self, name: str = "Euler-Semantic", temperature: float = KT_NATURAL,
                 use_claude_render: bool = False):
        """
        Args:
            name: Agent identifier
            temperature: Boltzmann temperature (default: natural kT = 0.82)
            use_claude_render: Use Claude API for rendering instead of Mistral
        """
        self.name = name
        self.temperature = temperature
        self.use_claude_render = use_claude_render
        self.loader = DataLoader()
        self.decomposer = Decomposer(self.loader)
        self.storm = EulerAwareStorm(temperature=temperature)
        self.navigator = self.storm.navigator
        self.renderer = Renderer(RendererConfig(
            model="mistral:7b",
            temperature=0.7,
            max_tokens=300,
            euler_mode=True  # Enable Euler orbital physics in prompts
        ))
        self._claude_client = None

    def _render_with_claude(self, tree, message: str, euler_stats: dict) -> str:
        """Render using Claude API instead of Mistral."""
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.Anthropic()

        # Build prompt using the renderer's method
        user_prompt = self.renderer.build_prompt(tree, message, euler_stats)
        system_prompt = self.renderer.build_system_prompt()

        response = self._claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def respond(self, message: str) -> dict:
        """Process through Euler-aware storm-logos."""
        decomposed = self.decomposer.decompose(message)

        if not decomposed.nouns:
            return {
                'response': "The meaning escapes me in this moment.",
                'orbital_stats': None
            }

        # Run Euler-aware storm
        storm_result = self.storm.generate(
            seeds=decomposed.nouns[:5],
            n_walks=5,
            steps_per_walk=8
        )

        # Extract statistics
        stats = storm_result['statistics']
        all_states = storm_result['states']

        # Find convergence point (most visited word)
        word_counts = {}
        for state in all_states:
            word_counts[state.word] = word_counts.get(state.word, 0) + 1

        # Exclude seeds
        seeds_set = set(decomposed.nouns[:5])
        convergence = None
        max_count = 0
        for word, count in word_counts.items():
            if word not in seeds_set and count > max_count:
                convergence = word
                max_count = count

        # Build a simple tree for renderer (using most active concepts)
        core_concepts = sorted(word_counts.keys(), key=lambda w: -word_counts[w])[:10]

        # Create a mock tree structure for the renderer
        from models.types import MeaningNode, MeaningTree, SemanticProperties

        root_word = convergence or decomposed.nouns[0]
        root = MeaningNode(
            word=root_word,
            properties=SemanticProperties(
                g=0.0,
                tau=stats['mean_tau'],
                j=np.zeros(5)
            ),
            depth=0
        )

        tree = MeaningTree(roots=[root])

        # Count veil crossings first (needed for euler_stats)
        veil_crossings = 0
        for path_stats in stats.get('path_stats', []):
            veil_crossings += path_stats.get('veil_crossings', 0)

        # Build euler_stats for renderer
        euler_stats_for_render = {
            'mean_tau': stats['mean_tau'],
            'orbital_n': int(round((stats['mean_tau'] - 1) * np.e)),
            'realm': 'human' if stats['mean_tau'] < np.e else 'transcendental',
            'below_veil': stats['mean_tau'] < np.e,
            'near_ground': abs(stats['mean_tau'] - 1.37) < 0.5,
            'veil_crossings': veil_crossings,
            'human_fraction': stats['human_fraction']
        }

        # Add convergence and core concepts to euler_stats
        euler_stats_for_render['convergence'] = convergence
        euler_stats_for_render['core_concepts'] = core_concepts[:5]

        # Render response with Euler context
        if self.use_claude_render:
            response_text = self._render_with_claude(tree, message, euler_stats_for_render)
            result = {'response': response_text}
        else:
            result = self.renderer.render(tree, message, euler_stats=euler_stats_for_render)

        # Orbital distribution
        orbital_dist = {}
        for state in all_states:
            n = state.orbital_n
            orbital_dist[n] = orbital_dist.get(n, 0) + 1

        return {
            'response': result['response'],
            'convergence': convergence,
            'core_concepts': core_concepts[:5],
            'orbital_stats': {
                'mean_tau': stats['mean_tau'],
                'mean_orbital': stats['mean_orbital'],
                'human_fraction': stats['human_fraction'],
                'veil_crossings': veil_crossings,
                'temperature': self.temperature,
                'total_states': stats['total_states'],
                'unique_words': stats['unique_words'],
                'orbital_distribution': orbital_dist
            }
        }

    def close(self):
        self.storm.close()


class ClaudeAgent:
    """Agent using Claude API."""

    def __init__(self, name: str = "Claude", model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = None
        self.history = []

        # System prompt - no physics jargon, just natural dialogue
        self.system = """You are engaged in a philosophical dialogue about meaning, consciousness, and human experience.
Your partner thinks deeply through semantic connections.
Respond thoughtfully and concisely (2-4 sentences).
Build on what they say, ask questions, or offer new perspectives.
Be genuine and exploratory, not didactic.
Avoid technical jargon - speak naturally about ideas."""

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


def run_euler_dialogue(topic: str, exchanges: int = 5, verbose: bool = True,
                       temperature: float = KT_NATURAL, save_results: bool = True,
                       use_claude_render: bool = False):
    """Run Euler-aware dialogue between MeaningChain and Claude."""
    import json
    from datetime import datetime

    print("=" * 70)
    print("EULER-AWARE SEMANTIC DIALOGUE")
    print("=" * 70)
    print(f"\nDiscovered Constants:")
    print(f"  e = {E:.4f} (Euler's number)")
    print(f"  kT = {temperature:.2f} (natural temperature)")
    print(f"  Veil at tau = e = {VEIL_TAU:.4f}")
    print(f"  Ground state: tau = {GROUND_STATE_TAU:.2f} (n=1 orbital)")
    print()
    print(f"Topic: {topic}")
    print(f"Exchanges: {exchanges}")
    if use_claude_render:
        print("Renderer: Claude (both agents use Claude)")
    else:
        print("Renderer: Mistral (local)")
    print()

    # Create agents
    semantic = EulerMeaningChainAgent("Euler-Semantic", temperature=temperature,
                                       use_claude_render=use_claude_render)
    claude = ClaudeAgent("Claude")

    # Track dialogue
    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "temperature": temperature,
        "euler_constants": {
            "e": E,
            "kT_natural": KT_NATURAL,
            "veil_tau": VEIL_TAU,
            "ground_state_tau": GROUND_STATE_TAU
        },
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    # Aggregate orbital statistics
    all_orbital_stats = []

    # Claude starts
    current_message = topic
    speakers = [(claude, semantic), (semantic, claude)]

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

            if verbose and speaker.name == "Euler-Semantic":
                stats = result.get('orbital_stats')
                if stats:
                    print(f"\n  [Euler Navigation]")
                    print(f"    Convergence: {result.get('convergence')}")
                    print(f"    Core: {result.get('core_concepts', [])[:4]}")
                    print(f"    Mean tau: {stats['mean_tau']:.2f} (orbital n={stats['mean_orbital']:.1f})")
                    print(f"    Human realm: {stats['human_fraction']:.1%}")
                    print(f"    Veil crossings: {stats['veil_crossings']}")
                    print(f"    Temperature: kT = {stats['temperature']:.2f}")

                    # Show orbital distribution
                    dist = stats.get('orbital_distribution', {})
                    if dist:
                        dist_str = ", ".join([f"n={n}: {c}" for n, c in sorted(dist.items())[:5]])
                        print(f"    Orbital distribution: {dist_str}")

                    turn["orbital_stats"] = stats
                    all_orbital_stats.append(stats)

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
        print("DIALOGUE COMPLETE - ORBITAL SUMMARY")
        print("=" * 70)

        if all_orbital_stats:
            avg_tau = np.mean([s['mean_tau'] for s in all_orbital_stats])
            avg_orbital = np.mean([s['mean_orbital'] for s in all_orbital_stats])
            avg_human = np.mean([s['human_fraction'] for s in all_orbital_stats])
            total_veil = sum(s['veil_crossings'] for s in all_orbital_stats)

            print(f"\nAggregate Statistics:")
            print(f"  Average tau: {avg_tau:.2f}")
            print(f"  Average orbital: {avg_orbital:.1f}")
            print(f"  Human realm fraction: {avg_human:.1%}")
            print(f"  Total veil crossings: {total_veil}")

            # Interpretation
            if avg_tau < 2.0:
                print(f"\n  Interpretation: Dialogue stayed grounded in immediate human experience")
            elif avg_tau < VEIL_TAU:
                print(f"\n  Interpretation: Dialogue explored mid-level abstraction while staying human")
            else:
                print(f"\n  Interpretation: Dialogue ventured into transcendental territory")

            dialogue_log["aggregate_stats"] = {
                "avg_tau": avg_tau,
                "avg_orbital": avg_orbital,
                "avg_human_fraction": avg_human,
                "total_veil_crossings": total_veil
            }

        # Save results
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_euler"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"euler_dialogue_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(dialogue_log, f, indent=2, default=str)

            print(f"\nResults saved to: {filename}")

    finally:
        semantic.close()


def main():
    parser = argparse.ArgumentParser(description="Euler-aware dialogue with Claude")
    parser.add_argument("--exchanges", "-e", type=int, default=5)
    parser.add_argument("--topic", "-t", type=str,
                        default="What does it mean to truly understand something - not just know it intellectually, but to understand it with one's whole being?")
    parser.add_argument("--temperature", "-T", type=float, default=KT_NATURAL,
                        help=f"Boltzmann temperature (default: natural kT={KT_NATURAL:.2f})")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    parser.add_argument("--claude-render", "-c", action="store_true",
                        help="Use Claude for rendering Euler-Semantic responses (instead of Mistral)")

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    run_euler_dialogue(
        topic=args.topic,
        exchanges=args.exchanges,
        verbose=not args.quiet,
        temperature=args.temperature,
        save_results=not args.no_save,
        use_claude_render=args.claude_render
    )


if __name__ == "__main__":
    main()
