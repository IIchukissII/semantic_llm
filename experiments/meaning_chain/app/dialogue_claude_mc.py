#!/usr/bin/env python3
"""
Monte Carlo Semantic Dialogue between MeaningChain and Claude.

Instead of one path, we sample semantic space many times to discover
its shape, then use that landscape to inform the conversation.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python dialogue_claude_mc.py --exchanges 5 --topic "help me understand wisdom"
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
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

from chain_core.monte_carlo_renderer import MonteCarloRenderer, SemanticLandscape


class MonteCarloSemanticAgent:
    """Agent using Monte Carlo semantic sampling."""

    def __init__(self, name: str = "MC-Semantic",
                 intent_strength: float = 0.3,
                 n_samples: int = 30):
        self.name = name
        self.intent_strength = intent_strength
        self.n_samples = n_samples
        self.renderer = MonteCarloRenderer(
            intent_strength=intent_strength,
            n_samples=n_samples
        )
        self._claude_client = None

    def _render_with_claude(self, landscape: SemanticLandscape, question: str) -> str:
        """Render using Claude API based on semantic landscape."""
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.Anthropic()

        # Build prompt from landscape
        prompt = self._build_claude_prompt(landscape, question)

        response = self._claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _build_system_prompt(self) -> str:
        return """You respond based on Monte Carlo semantic mapping.

The sampling process revealed where this question's meaning lives in semantic space.
Core concepts appeared consistently across many samples - these are stable meaning.
The orbital hierarchy shows abstraction levels: concrete → abstract.

Respond thoughtfully using the discovered concepts naturally.
Be substantive (2-4 sentences), not formulaic.
Don't list concepts - weave them into natural language."""

    def _build_claude_prompt(self, ls: SemanticLandscape, question: str) -> str:
        """Build prompt for Claude from semantic landscape."""
        sections = []

        # Header
        sections.append("## Semantic Landscape (Monte Carlo Sampling)")
        sections.append(f"Question: {question}")
        sections.append(f"Samples: {ls.n_samples} | Coherence: {ls.coherence:.2f} | Focus: {ls.concentration:.0%}")
        sections.append("")

        # Core attractors
        core_words = [w for w, _ in ls.core_attractors[:8]]
        sections.append(f"## Core Concepts (stable)")
        sections.append(f"These appeared consistently: {', '.join(core_words)}")
        sections.append("")

        # Orbital hierarchy
        sections.append("## Abstraction Hierarchy")
        for n in sorted(ls.orbital_map.keys())[:4]:
            words = [w for w, _, _ in ls.orbital_map[n][:4]]
            if n <= 1:
                level = "concrete/grounded"
            elif n <= 2:
                level = "mid-level/connecting"
            else:
                level = "abstract/transcendent"
            sections.append(f"  {level}: {', '.join(words)}")
        sections.append("")

        # Peripheral (optional context)
        if ls.peripheral_attractors:
            peripheral = [w for w, _ in ls.peripheral_attractors[:8]]
            sections.append(f"## Context (variable): {', '.join(peripheral)}")
            sections.append("")

        # Guidance
        sections.append("## Response Guidance")
        if ls.coherence > 0.7:
            sections.append("- High coherence: confident, focused response")
        else:
            sections.append("- Lower coherence: acknowledge nuance/multiple views")

        if ls.concentration > 0.25:
            sections.append("- High focus: stay close to core concepts")
        else:
            sections.append("- Diffuse: explore connections freely")

        sections.append("- Use core concepts naturally, don't list them")
        sections.append("- 2-4 sentences")

        return "\n".join(sections)

    def respond(self, message: str) -> Dict[str, Any]:
        """Process through Monte Carlo sampling and render with Claude."""
        # Sample the landscape
        landscape = self.renderer.sample_landscape(message)

        # Render with Claude
        response_text = self._render_with_claude(landscape, message)

        return {
            'response': response_text,
            'landscape': {
                'n_samples': landscape.n_samples,
                'unique_words': landscape.unique_words,
                'concentration': landscape.concentration,
                'coherence': landscape.coherence,
                'lasing_rate': landscape.lasing_rate,
                'core_attractors': [w for w, _ in landscape.core_attractors[:8]],
                'tau_mean': landscape.tau_mean,
                'intent_influence': landscape.intent_influence
            }
        }

    def close(self):
        self.renderer.close()


class ClaudeAgent:
    """Agent using Claude API."""

    def __init__(self, name: str = "Claude", model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = None
        self.history = []

        self.system = """You are engaged in a philosophical dialogue about meaning.
Your partner maps semantic space through Monte Carlo sampling - throwing questions
into meaning-space many times to see where they land.

Respond thoughtfully and concisely (2-4 sentences).
Build on what they say, ask questions, or offer new perspectives.
Be genuine and exploratory."""

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


def run_mc_dialogue(topic: str, exchanges: int = 5, verbose: bool = True,
                    intent_strength: float = 0.3, n_samples: int = 30,
                    save_results: bool = True):
    """Run Monte Carlo semantic dialogue between MeaningChain and Claude."""

    print("=" * 70)
    print("MONTE CARLO SEMANTIC DIALOGUE")
    print("=" * 70)
    print(f"\nMonte Carlo Sampling:")
    print(f"  Samples per turn: {n_samples}")
    print(f"  Intent strength (α): {intent_strength}")
    print(f"  Mode: {'soft wind' if intent_strength < 1 else 'hard collapse'}")
    print()
    print(f"Topic: {topic}")
    print(f"Exchanges: {exchanges}")
    print()

    # Create agents
    semantic = MonteCarloSemanticAgent(
        "MC-Semantic",
        intent_strength=intent_strength,
        n_samples=n_samples
    )
    claude = ClaudeAgent("Claude")

    # Track dialogue
    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "intent_strength": intent_strength,
        "n_samples": n_samples,
        "mode": "monte_carlo",
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    # Aggregate stats
    all_landscapes = []

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

            if verbose and speaker.name == "MC-Semantic":
                ls = result.get('landscape', {})

                print(f"\n  [Semantic Landscape]")
                print(f"    Samples: {ls.get('n_samples', 0)}")
                print(f"    Unique words: {ls.get('unique_words', 0)}")
                print(f"    Coherence: {ls.get('coherence', 0):.2f}")
                print(f"    Focus: {ls.get('concentration', 0):.0%}")
                print(f"    Lasing: {ls.get('lasing_rate', 0):.0%}")
                print(f"    τ mean: {ls.get('tau_mean', 0):.2f}")
                print(f"\n  [Core Attractors]")
                print(f"    {ls.get('core_attractors', [])[:6]}")

                turn["landscape"] = ls
                all_landscapes.append(ls)

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
        print("DIALOGUE COMPLETE - MONTE CARLO SUMMARY")
        print("=" * 70)

        if all_landscapes:
            avg_coherence = np.mean([ls.get('coherence', 0) for ls in all_landscapes])
            avg_unique = np.mean([ls.get('unique_words', 0) for ls in all_landscapes])
            avg_concentration = np.mean([ls.get('concentration', 0) for ls in all_landscapes])
            avg_lasing = np.mean([ls.get('lasing_rate', 0) for ls in all_landscapes])

            # Collect all core attractors
            all_attractors = []
            for ls in all_landscapes:
                all_attractors.extend(ls.get('core_attractors', []))
            attractor_counts = {}
            for a in all_attractors:
                attractor_counts[a] = attractor_counts.get(a, 0) + 1
            top_attractors = sorted(attractor_counts.items(), key=lambda x: -x[1])[:10]

            print(f"\nLandscape Statistics:")
            print(f"  Average coherence: {avg_coherence:.2f}")
            print(f"  Average unique words: {avg_unique:.0f}")
            print(f"  Average concentration: {avg_concentration:.0%}")
            print(f"  Average lasing rate: {avg_lasing:.0%}")
            print(f"\nMost common attractors across dialogue:")
            for word, count in top_attractors[:8]:
                print(f"    {word}: {count}")

            dialogue_log["aggregate_stats"] = {
                "avg_coherence": avg_coherence,
                "avg_unique_words": avg_unique,
                "avg_concentration": avg_concentration,
                "avg_lasing_rate": avg_lasing,
                "top_attractors": top_attractors
            }

        # Save results
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_mc"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"mc_dialogue_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(dialogue_log, f, indent=2, default=str)

            print(f"\nResults saved to: {filename}")

    finally:
        semantic.close()


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo semantic dialogue with Claude")
    parser.add_argument("--exchanges", "-e", type=int, default=5)
    parser.add_argument("--topic", "-t", type=str,
                        default="What do my dreams tell me about myself?")
    parser.add_argument("--alpha", "-a", type=float, default=0.3,
                        help="Intent strength (0=Boltzmann, 0.3=soft wind, 1+=hard)")
    parser.add_argument("--samples", "-s", type=int, default=30,
                        help="Monte Carlo samples per turn")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    run_mc_dialogue(
        topic=args.topic,
        exchanges=args.exchanges,
        verbose=not args.quiet,
        intent_strength=args.alpha,
        n_samples=args.samples,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
