#!/usr/bin/env python3
"""
Run Dialogue with Claude and Save Results for Comparison

Runs the dialogue between Meaning Chain and Claude, saving results
to compare before/after learning implementation.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python scripts/run_dialogue_compare.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Add paths
_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _SCRIPTS_DIR.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.decomposer import Decomposer
from chain_core.storm_logos import StormLogosBuilder
from chain_core.renderer import Renderer, RendererConfig
from graph.meaning_graph import MeaningGraph


class MeaningChainAgent:
    """Agent that thinks through semantic space."""

    def __init__(self, name: str = "Semantic"):
        self.name = name
        self.loader = DataLoader()
        self.decomposer = Decomposer(self.loader)
        self.storm_logos = StormLogosBuilder(
            storm_temperature=1.5,
            n_walks=5,
            steps_per_walk=8
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
            'coherence': pattern.coherence if pattern else 0,
            'core_concepts': pattern.core_concepts[:5] if pattern else []
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


def run_dialogue_and_save(topic: str, exchanges: int = 5,
                          output_dir: Path = None) -> Dict:
    """Run dialogue and save results."""

    if output_dir is None:
        output_dir = _MEANING_CHAIN / "results" / "dialogue_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("SEMANTIC DIALOGUE: Meaning Chain ↔ Claude")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print(f"Exchanges: {exchanges}")
    print()

    # Get learning stats before dialogue
    graph = MeaningGraph()
    learning_stats = {}
    if graph.is_connected():
        learning_stats = graph.get_learning_stats()
        graph_stats = graph.get_stats()
        print(f"Graph stats: {graph_stats['concepts']} concepts, {graph_stats['via_edges']} VIA edges")
        if learning_stats.get('learned_concepts'):
            print(f"Learning: {learning_stats['learned_concepts']} learned concepts, avg τ={learning_stats.get('avg_tau', 0):.2f}")
        graph.close()

    # Create agents
    semantic = MeaningChainAgent("Semantic")
    claude = ClaudeAgent("Claude")

    # Track dialogue
    dialogue_log = []
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

            # Build exchange record
            exchange = {
                'exchange': i + 1,
                'speaker': speaker.name,
                'input': current_message,
                'response': response
            }

            if speaker.name == "Semantic" and result.get('pattern'):
                p = result['pattern']
                exchange['semantic'] = {
                    'convergence': p.convergence_point,
                    'core_concepts': p.core_concepts[:5],
                    'coherence': p.coherence
                }
                print(f"\n  [Storm-Logos]")
                print(f"    Convergence: {p.convergence_point}")
                print(f"    Core: {p.core_concepts[:4]}")
                print(f"    Coherence: {p.coherence:.0%}")

            dialogue_log.append(exchange)

            print(f"\n{speaker.name}:")
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

    finally:
        semantic.close()

    # Compute summary statistics
    semantic_exchanges = [e for e in dialogue_log if e['speaker'] == 'Semantic']
    coherence_values = [e['semantic']['coherence'] for e in semantic_exchanges if 'semantic' in e]
    convergence_points = [e['semantic']['convergence'] for e in semantic_exchanges if 'semantic' in e]

    summary = {
        'timestamp': timestamp,
        'topic': topic,
        'exchanges': exchanges,
        'learning_enabled': learning_stats.get('learned_concepts', 0) > 0,
        'learning_stats': learning_stats,
        'avg_coherence': sum(coherence_values) / len(coherence_values) if coherence_values else 0,
        'convergence_points': convergence_points,
        'dialogue': dialogue_log
    }

    # Save results as JSON
    output_file = output_dir / f"dialogue_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save results as TXT
    txt_file = output_dir / f"dialogue_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SEMANTIC DIALOGUE: Meaning Chain ↔ Claude\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Topic: {topic}\n")
        f.write(f"Exchanges: {exchanges}\n")
        f.write(f"Learning Enabled: {summary['learning_enabled']}\n")
        if learning_stats.get('learned_concepts'):
            f.write(f"Learned Concepts: {learning_stats['learned_concepts']}\n")
            f.write(f"Average τ: {learning_stats.get('avg_tau', 0):.2f}\n")
        f.write("\n")

        for ex in dialogue_log:
            f.write("-" * 70 + "\n")
            f.write(f"Exchange {ex['exchange']}: {ex['speaker']}\n")
            f.write("-" * 70 + "\n")
            if 'semantic' in ex:
                f.write(f"\n[Storm-Logos]\n")
                f.write(f"  Convergence: {ex['semantic']['convergence']}\n")
                f.write(f"  Core: {ex['semantic']['core_concepts'][:4]}\n")
                f.write(f"  Coherence: {ex['semantic']['coherence']:.0%}\n")
            f.write(f"\n{ex['speaker']}:\n")
            f.write(f"{ex['response']}\n\n")

        f.write("=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Average Coherence: {summary['avg_coherence']:.0%}\n")
        f.write(f"Convergence Points: {convergence_points}\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {output_file}")
    print(f"  TXT:  {txt_file}")

    # Print summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Average coherence: {summary['avg_coherence']:.0%}")
    print(f"Convergence points: {convergence_points}")
    if learning_stats.get('learned_concepts'):
        print(f"Learning enabled: {learning_stats['learned_concepts']} concepts")

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exchanges", "-e", type=int, default=5)
    parser.add_argument("--topic", "-t", type=str,
                        default="What does it mean to truly understand something - not just know it intellectually, but to understand it with one's whole being?")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    run_dialogue_and_save(
        topic=args.topic,
        exchanges=args.exchanges
    )


if __name__ == "__main__":
    main()
