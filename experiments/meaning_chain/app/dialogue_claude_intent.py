#!/usr/bin/env python3
"""
Intent-Collapse Dialogue between Meaning Chain and Claude.

Uses the new intent-driven collapse where verbs act as operators
that collapse navigation to intent-relevant paths.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python dialogue_claude_intent.py --exchanges 5 --topic "help me understand wisdom"
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import argparse
import numpy as np
from datetime import datetime
import json

# Add paths
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _APP_DIR.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from chain_core.decomposer import Decomposer
from chain_core.semantic_laser import SemanticLaser, KT_NATURAL, VEIL_TAU, E
from chain_core.renderer import Renderer, RendererConfig
from models.types import MeaningNode, MeaningTree, SemanticProperties


class IntentSemanticAgent:
    """Agent using SemanticLaser with intent collapse."""

    def __init__(self, name: str = "Intent-Semantic", temperature: float = KT_NATURAL,
                 use_claude_render: bool = False):
        self.name = name
        self.temperature = temperature
        self.use_claude_render = use_claude_render
        self.loader = DataLoader()
        self.decomposer = Decomposer(self.loader)
        self.laser = SemanticLaser(temperature=temperature)
        self.renderer = Renderer(RendererConfig(
            model="mistral:7b",
            temperature=0.7,
            max_tokens=300,
            euler_mode=True
        ))
        self._claude_client = None

    def _render_with_claude(self, tree, message: str, euler_stats: dict) -> str:
        """Render using Claude API instead of Mistral."""
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.Anthropic()

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
        """Process through SemanticLaser with intent collapse."""
        decomposed = self.decomposer.decompose(message)

        if not decomposed.nouns:
            return {
                'response': "The meaning escapes me in this moment.",
                'intent_stats': None
            }

        # Run laser with intent collapse
        laser_result = self.laser.lase(
            seeds=decomposed.nouns[:5],
            pump_power=10,
            pump_depth=5,
            coherence_threshold=0.3,
            intent_verbs=decomposed.verbs  # KEY: Use verbs for intent collapse
        )

        pop = laser_result['population']
        beams = laser_result['beams']
        intent_info = laser_result.get('intent', {})
        metrics = laser_result.get('metrics', {})

        # Extract coherent concepts
        coherent_concepts = []
        beam_themes = []
        for beam in beams[:3]:
            themes = self.laser.get_beam_themes(beam)
            coherent_concepts.extend(beam.concepts[:5])
            beam_themes.extend(themes)

        # Build tree for renderer
        primary_beam = self.laser.get_primary_beam(laser_result)
        root_word = coherent_concepts[0] if coherent_concepts else decomposed.nouns[0]

        root = MeaningNode(
            word=root_word,
            properties=SemanticProperties(
                g=primary_beam.g_polarity if primary_beam else 0.0,
                tau=pop['tau_mean'],
                j=primary_beam.j_centroid if primary_beam else np.zeros(5)
            ),
            depth=0
        )
        tree = MeaningTree(roots=[root])

        # Build euler_stats for renderer (with intent info)
        euler_stats = {
            'mean_tau': pop['tau_mean'],
            'orbital_n': int(round((pop['tau_mean'] - 1) * E)),
            'realm': 'human' if pop['tau_mean'] < VEIL_TAU else 'transcendental',
            'below_veil': pop['tau_mean'] < VEIL_TAU,
            'near_ground': abs(pop['tau_mean'] - 1.37) < 0.5,
            'veil_crossings': pop.get('above_veil', 0),
            'human_fraction': pop.get('human_fraction', 0.5),
            'convergence': coherent_concepts[0] if coherent_concepts else None,
            'core_concepts': coherent_concepts[:8],
            'key_symbols': decomposed.unknown_words[:10],
            'known_nouns': decomposed.nouns[:8],
            # Laser metrics
            'laser_beams': len(beams),
            'laser_coherence': primary_beam.coherence if primary_beam else 0.0,
            'laser_themes': list(set(beam_themes))[:5],
            'lasing_achieved': metrics.get('lasing_achieved', False),
            'output_power': metrics.get('output_power', 0.0),
            # Intent metrics
            'intent_enabled': intent_info.get('enabled', False),
            'intent_verbs': decomposed.verbs,
            'intent_fraction': pop.get('intent_fraction', 0.0),
            'intent_focus': metrics.get('intent_focus', 0.5)
        }

        # Render response
        if self.use_claude_render:
            response_text = self._render_with_claude(tree, message, euler_stats)
            result = {'response': response_text}
        else:
            result = self.renderer.render(tree, message, euler_stats=euler_stats)

        return {
            'response': result['response'],
            'convergence': coherent_concepts[0] if coherent_concepts else None,
            'core_concepts': coherent_concepts[:5],
            'beam_themes': list(set(beam_themes))[:5],
            'intent_stats': {
                'verbs': decomposed.verbs,
                'enabled': intent_info.get('enabled', False),
                'intent_fraction': pop.get('intent_fraction', 0.0),
                'intent_focus': metrics.get('intent_focus', 0.5),
                'operators': intent_info.get('stats', {}).get('operators', 0) if intent_info.get('stats') else 0,
                'targets': intent_info.get('stats', {}).get('targets', 0) if intent_info.get('stats') else 0,
            },
            'laser_stats': {
                'beams': len(beams),
                'coherence': primary_beam.coherence if primary_beam else 0.0,
                'excited_states': pop['total_excited'],
                'mean_tau': pop['tau_mean'],
                'human_fraction': pop.get('human_fraction', 0.5),
                'lasing_achieved': metrics.get('lasing_achieved', False)
            }
        }

    def close(self):
        self.laser.close()


class ClaudeAgent:
    """Agent using Claude API."""

    def __init__(self, name: str = "Claude", model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = None
        self.history = []

        self.system = """You are engaged in a philosophical dialogue about meaning, consciousness, and human experience.
Your partner navigates through semantic space guided by the intent of your words.
When you use verbs like 'understand', 'find', 'seek' - they guide where the conversation goes.

Respond thoughtfully and concisely (2-4 sentences).
Build on what they say, ask questions, or offer new perspectives.
Be genuine and exploratory, not didactic.
Use clear action verbs - they shape the semantic path."""

    def _init_client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()

    def respond(self, message: str) -> dict:
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

        return {'response': reply, 'model': self.model}


def run_intent_dialogue(topic: str, exchanges: int = 5, verbose: bool = True,
                        temperature: float = KT_NATURAL, save_results: bool = True,
                        use_claude_render: bool = False):
    """Run intent-driven dialogue between MeaningChain and Claude."""

    print("=" * 70)
    print("INTENT-DRIVEN SEMANTIC DIALOGUE")
    print("=" * 70)
    print(f"\nIntent Collapse:")
    print(f"  Verbs from query → VerbOperators → Collapse navigation")
    print(f"  Temperature: kT = {temperature:.2f}")
    print()
    print(f"Topic: {topic}")
    print(f"Exchanges: {exchanges}")
    print()

    # Create agents
    semantic = IntentSemanticAgent("Intent-Semantic", temperature=temperature,
                                    use_claude_render=use_claude_render)
    claude = ClaudeAgent("Claude")

    # Track dialogue
    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "temperature": temperature,
        "mode": "intent_collapse",
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    # Aggregate stats
    all_intent_stats = []
    all_laser_stats = []

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

            turn = {
                "exchange": i + 1,
                "speaker": speaker.name,
                "input": current_message,
                "response": response
            }

            if verbose and speaker.name == "Intent-Semantic":
                intent_stats = result.get('intent_stats', {})
                laser_stats = result.get('laser_stats', {})

                print(f"\n  [Intent Collapse]")
                print(f"    Verbs: {intent_stats.get('verbs', [])}")
                print(f"    Operators: {intent_stats.get('operators', 0)}, Targets: {intent_stats.get('targets', 0)}")
                print(f"    Intent fraction: {intent_stats.get('intent_fraction', 0):.0%}")
                print(f"    Intent focus: {intent_stats.get('intent_focus', 0.5):.2f}")

                print(f"\n  [Laser Output]")
                print(f"    Convergence: {result.get('convergence')}")
                print(f"    Core: {result.get('core_concepts', [])[:4]}")
                print(f"    Themes: {result.get('beam_themes', [])[:4]}")
                print(f"    Beams: {laser_stats.get('beams', 0)}, Coherence: {laser_stats.get('coherence', 0):.2f}")
                print(f"    Lasing: {'✓' if laser_stats.get('lasing_achieved') else '✗'}")

                turn["intent_stats"] = intent_stats
                turn["laser_stats"] = laser_stats
                all_intent_stats.append(intent_stats)
                all_laser_stats.append(laser_stats)

            dialogue_log["turns"].append(turn)

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
        print("DIALOGUE COMPLETE - INTENT COLLAPSE SUMMARY")
        print("=" * 70)

        if all_intent_stats:
            avg_intent_frac = np.mean([s.get('intent_fraction', 0) for s in all_intent_stats])
            avg_intent_focus = np.mean([s.get('intent_focus', 0.5) for s in all_intent_stats])
            all_verbs = []
            for s in all_intent_stats:
                all_verbs.extend(s.get('verbs', []))
            verb_counts = {}
            for v in all_verbs:
                verb_counts[v] = verb_counts.get(v, 0) + 1
            top_verbs = sorted(verb_counts.items(), key=lambda x: -x[1])[:5]

            print(f"\nIntent Statistics:")
            print(f"  Average intent fraction: {avg_intent_frac:.0%}")
            print(f"  Average intent focus: {avg_intent_focus:.2f}")
            print(f"  Most used verbs: {top_verbs}")

        if all_laser_stats:
            avg_coherence = np.mean([s.get('coherence', 0) for s in all_laser_stats])
            avg_beams = np.mean([s.get('beams', 0) for s in all_laser_stats])
            lasing_count = sum(1 for s in all_laser_stats if s.get('lasing_achieved'))

            print(f"\nLaser Statistics:")
            print(f"  Average coherence: {avg_coherence:.2f}")
            print(f"  Average beams: {avg_beams:.1f}")
            print(f"  Lasing achieved: {lasing_count}/{len(all_laser_stats)} exchanges")

            dialogue_log["aggregate_stats"] = {
                "avg_intent_fraction": avg_intent_frac,
                "avg_intent_focus": avg_intent_focus,
                "avg_coherence": avg_coherence,
                "top_verbs": top_verbs,
                "lasing_rate": lasing_count / len(all_laser_stats) if all_laser_stats else 0
            }

        # Save results
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_intent"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"intent_dialogue_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(dialogue_log, f, indent=2, default=str)

            print(f"\nResults saved to: {filename}")

    finally:
        semantic.close()


def main():
    parser = argparse.ArgumentParser(description="Intent-driven dialogue with Claude")
    parser.add_argument("--exchanges", "-e", type=int, default=5)
    parser.add_argument("--topic", "-t", type=str,
                        default="Help me understand what it means to find wisdom in everyday life")
    parser.add_argument("--temperature", "-T", type=float, default=KT_NATURAL)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--claude-render", "-c", action="store_true",
                        help="Use Claude for rendering (instead of Mistral)")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    run_intent_dialogue(
        topic=args.topic,
        exchanges=args.exchanges,
        verbose=not args.quiet,
        temperature=args.temperature,
        save_results=not args.no_save,
        use_claude_render=args.claude_render
    )


if __name__ == "__main__":
    main()
