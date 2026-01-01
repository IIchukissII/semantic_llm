#!/usr/bin/env python3
"""
Dialogue Navigator: Unified Semantic Physics Dialogue
======================================================

Uses the SemanticNavigator to drive dialogue with goal-based navigation.

Instead of choosing between orbital/monte-carlo/paradox modes,
specify what you want: accurate, deep, stable, powerful, grounded.

Features:
- Goal-based semantic navigation
- Quality metrics displayed per turn
- Strategy comparison mode
- Multi-engine insights

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python dialogue_navigator.py --goal deep --topic "What is consciousness?"
    python dialogue_navigator.py --goal powerful --topic "What is love?"
    python dialogue_navigator.py --compare --topic "What is wisdom?"
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

from chain_core.navigator import (
    SemanticNavigator, NavigationResult, NavigationQuality, NavigationGoal
)
from chain_core.orbital import E, VEIL_TAU, orbital_to_tau


class NavigatorAgent:
    """
    Dialogue agent powered by SemanticNavigator.

    Uses unified semantic physics to generate responses.
    """

    def __init__(self, name: str = "Navigator",
                 goal: str = "balanced",
                 show_metrics: bool = True):
        self.name = name
        self.goal = NavigationGoal(goal)
        self.show_metrics = show_metrics

        self.navigator = SemanticNavigator()
        self._claude_client = None

    def _init_claude(self):
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.Anthropic()
        return self._claude_client

    def _build_system_prompt(self, result: NavigationResult) -> str:
        """Build system prompt based on navigation result."""
        quality = result.quality
        goal = result.goal.value

        # Determine tone based on quality profile
        if quality.depth > 2.0:
            tone = "contemplative and philosophical"
            style = "nuanced, exploratory language with metaphors"
        elif quality.resonance > 0.7:
            tone = "confident and precise"
            style = "clear, direct language"
        elif quality.power > 2.0:
            tone = "powerful and paradoxical"
            style = "language that holds opposites together"
        elif quality.tau_mean < 2.0:
            tone = "grounded and practical"
            style = "concrete, actionable language"
        else:
            tone = "balanced and thoughtful"
            style = "accessible yet substantive language"

        # Realm info
        realm = "human realm" if quality.tau_mean < VEIL_TAU else "transcendental realm"

        return f"""You respond based on unified semantic physics navigation.

Navigation goal: {goal}
Quality profile: R={quality.resonance:.2f}, C={quality.coherence:.2f}, D={quality.depth:.1f}
Semantic realm: {realm} (τ={quality.tau_mean:.2f})

Tone: {tone}
Style: {style}

Use the navigated concepts naturally in your response.
2-4 sentences, substantive and genuine.
Don't list concepts - weave them into natural language."""

    def _build_prompt(self, result: NavigationResult, question: str) -> str:
        """Build prompt from navigation result."""
        sections = []

        # Header
        sections.append("## Semantic Navigation Result")
        sections.append(f"Question: {question}")
        sections.append(f"Goal: {result.goal.value}")
        sections.append(f"Strategy: {result.strategy}")
        sections.append("")

        # Quality metrics
        q = result.quality
        sections.append("## Quality Metrics")
        sections.append(f"  Resonance: {q.resonance:.0%} (accuracy)")
        sections.append(f"  Coherence: {q.coherence:.2f} (alignment)")
        sections.append(f"  Depth: {q.depth:.1f} (C/R ratio)")
        sections.append(f"  Stability: {q.stability:.0%}")
        if q.power > 0:
            sections.append(f"  Power: {q.power:.1f} (tension)")
        sections.append("")

        # Concepts
        sections.append("## Navigated Concepts")
        sections.append(f"  {', '.join(result.concepts[:8])}")
        sections.append("")

        # Decomposition
        sections.append("## Query Analysis")
        sections.append(f"  Nouns: {result.nouns}")
        sections.append(f"  Verbs: {result.verbs}")
        sections.append("")

        # Orbital info if available
        if result.detected_orbital > 0 or result.target_orbital > 0:
            sections.append("## Orbital Navigation")
            sections.append(f"  Detected: n={result.detected_orbital} (τ={orbital_to_tau(result.detected_orbital):.2f})")
            sections.append(f"  Target: n={result.target_orbital} (τ={orbital_to_tau(result.target_orbital):.2f})")
            sections.append("")

        # Paradox info if available
        if result.thesis and result.antithesis:
            sections.append("## Semantic Paradox")
            sections.append(f"  Thesis: {result.thesis}")
            sections.append(f"  Antithesis: {result.antithesis}")
            sections.append(f"  Synthesis: {result.synthesis[:3]}")
            sections.append("")

        # Response guidance
        sections.append("## Response Guidance")
        if q.depth > 2.0:
            sections.append("- High depth: explore the paradox, embrace uncertainty")
        elif q.resonance > 0.7:
            sections.append("- High resonance: confident, focused response")
        elif q.power > 2.0:
            sections.append("- High power: hold both poles of the paradox")
        else:
            sections.append("- Balanced: thoughtful, substantive response")

        sections.append("- Use navigated concepts naturally")
        sections.append("- 2-4 sentences, genuine and exploratory")

        return "\n".join(sections)

    def respond(self, message: str) -> Dict[str, Any]:
        """Generate response using semantic navigation."""
        # Navigate
        result = self.navigator.navigate(message, goal=self.goal)

        # Build prompt
        system_prompt = self._build_system_prompt(result)
        prompt = self._build_prompt(result, message)

        # Generate with Claude
        client = self._init_claude()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text

        return {
            'response': response_text,
            'navigation': {
                'goal': result.goal.value,
                'strategy': result.strategy,
                'concepts': result.concepts[:8],
                'quality': {
                    'resonance': result.quality.resonance,
                    'coherence': result.quality.coherence,
                    'depth': result.quality.depth,
                    'stability': result.quality.stability,
                    'power': result.quality.power,
                    'tau_mean': result.quality.tau_mean
                },
                'decomposition': {
                    'nouns': result.nouns,
                    'verbs': result.verbs
                },
                'orbital': {
                    'detected': result.detected_orbital,
                    'target': result.target_orbital
                }
            },
            'paradox': {
                'thesis': result.thesis,
                'antithesis': result.antithesis,
                'synthesis': result.synthesis
            } if result.thesis else None
        }

    def close(self):
        self.navigator.close()


class ClaudeAgent:
    """Standard Claude agent for dialogue."""

    def __init__(self, name: str = "Claude", model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = None
        self.history = []

        self.system = """You are engaged in a philosophical dialogue.
Your partner uses unified semantic physics to navigate meaning space.
They see concepts through multiple lenses: orbital resonance, statistical stability, dialectical tension.

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


def run_dialogue(topic: str,
                 exchanges: int = 5,
                 goal: str = "balanced",
                 compare_mode: bool = False,
                 verbose: bool = True,
                 save_results: bool = True):
    """Run dialogue between Navigator agent and Claude."""

    print("=" * 70)
    print("SEMANTIC NAVIGATOR DIALOGUE")
    print("=" * 70)
    print(f"\nGoal: {goal}")
    print(f"Topic: {topic}")
    print(f"Exchanges: {exchanges}")
    if compare_mode:
        print("Mode: Strategy comparison")
    print()

    # Create agents
    try:
        navigator_agent = NavigatorAgent("Navigator", goal=goal)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    claude = ClaudeAgent("Claude")

    # Track dialogue
    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "goal": goal,
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    # Aggregate stats
    all_quality = []

    current_message = topic
    speakers = [(claude, navigator_agent), (navigator_agent, claude)]

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

            # Show navigation details
            if verbose and speaker.name == "Navigator":
                nav = result.get('navigation', {})
                quality = nav.get('quality', {})

                print(f"\n  [Navigation]")
                print(f"    Goal: {nav.get('goal')} | Strategy: {nav.get('strategy')}")
                print(f"    R={quality.get('resonance', 0):.0%} C={quality.get('coherence', 0):.2f} "
                      f"D={quality.get('depth', 0):.1f} S={quality.get('stability', 0):.0%}")

                if quality.get('power', 0) > 0:
                    print(f"    Power: {quality.get('power', 0):.1f}")

                print(f"    τ mean: {quality.get('tau_mean', 0):.2f}")
                print(f"\n  [Concepts]")
                print(f"    {nav.get('concepts', [])[:6]}")

                # Show paradox if found
                paradox = result.get('paradox')
                if paradox and paradox.get('thesis'):
                    print(f"\n  [Paradox]")
                    print(f"    {paradox['thesis']} ↔ {paradox['antithesis']}")
                    print(f"    Synthesis: {paradox['synthesis'][:3]}")

                turn["navigation"] = nav
                turn["paradox"] = paradox
                all_quality.append(quality)

            dialogue_log["turns"].append(turn)

            # Print response
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

        # Summary
        print("\n" + "=" * 70)
        print("DIALOGUE COMPLETE - NAVIGATION SUMMARY")
        print("=" * 70)

        if all_quality:
            avg_r = np.mean([q.get('resonance', 0) for q in all_quality])
            avg_c = np.mean([q.get('coherence', 0) for q in all_quality])
            avg_d = np.mean([q.get('depth', 0) for q in all_quality])
            avg_s = np.mean([q.get('stability', 0) for q in all_quality])
            avg_tau = np.mean([q.get('tau_mean', 0) for q in all_quality])

            print(f"\nAverage Quality Metrics:")
            print(f"  Resonance: {avg_r:.0%}")
            print(f"  Coherence: {avg_c:.2f}")
            print(f"  Depth: {avg_d:.1f}")
            print(f"  Stability: {avg_s:.0%}")
            print(f"  τ mean: {avg_tau:.2f}")

            # Realm distribution
            human_count = sum(1 for q in all_quality if q.get('tau_mean', 3) < VEIL_TAU)
            print(f"  Human realm turns: {human_count}/{len(all_quality)}")

            dialogue_log["aggregate_stats"] = {
                "avg_resonance": avg_r,
                "avg_coherence": avg_c,
                "avg_depth": avg_d,
                "avg_stability": avg_s,
                "avg_tau": avg_tau
            }

        # Save results
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_navigator"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"dialogue_{goal}_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(dialogue_log, f, indent=2, default=str)

            print(f"\nResults saved to: {filename}")

    finally:
        navigator_agent.close()


def run_comparison(topic: str, verbose: bool = True):
    """Run all goals on the same topic and compare."""

    print("=" * 70)
    print("NAVIGATION STRATEGY COMPARISON")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print()

    navigator = SemanticNavigator()

    try:
        # Run comparison
        comparison = navigator.compare_strategies(topic)

        print("\n## Strategy Results")
        print("-" * 70)

        for strategy, data in comparison['strategies'].items():
            print(f"\n{strategy.upper()}")
            print(f"  Concepts: {data['concepts']}")
            q = data['quality']
            print(f"  R={q['resonance']:.2f} C={q['coherence']:.2f} D={q['depth']:.1f} "
                  f"S={q['stability']:.2f} P={q['power']:.1f}")

        print("\n## Best Strategy per Goal")
        print("-" * 70)
        for goal, engine in comparison['best_for_goal'].items():
            print(f"  {goal:12s} → {engine}")

    finally:
        navigator.close()


def interactive_mode(goal: str = "balanced"):
    """Interactive dialogue mode."""

    print("=" * 70)
    print("SEMANTIC NAVIGATOR - Interactive Mode")
    print("=" * 70)
    print(f"\nGoal: {goal}")
    print("Type 'quit' to exit, 'goal <name>' to change goal")
    print("Goals: accurate, deep, grounded, stable, powerful, balanced, wisdom, supercritical")
    print()

    navigator_agent = NavigatorAgent("Navigator", goal=goal)
    current_goal = goal

    try:
        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower().startswith('goal '):
                new_goal = user_input.split(' ', 1)[1].strip()
                try:
                    navigator_agent.goal = NavigationGoal(new_goal)
                    current_goal = new_goal
                    print(f"[Goal changed to: {new_goal}]")
                    continue
                except ValueError:
                    print(f"[Unknown goal: {new_goal}]")
                    continue

            result = navigator_agent.respond(user_input)

            # Show navigation
            nav = result.get('navigation', {})
            quality = nav.get('quality', {})
            print(f"\n[{nav.get('strategy')}] R={quality.get('resonance', 0):.0%} "
                  f"C={quality.get('coherence', 0):.2f} D={quality.get('depth', 0):.1f}")
            print(f"[Concepts: {nav.get('concepts', [])[:5]}]")

            print(f"\nNavigator: {result['response']}")

    finally:
        navigator_agent.close()
        print("\n[Session ended]")


def main():
    parser = argparse.ArgumentParser(description="Semantic Navigator Dialogue")
    parser.add_argument("--exchanges", "-e", type=int, default=5)
    parser.add_argument("--topic", "-t", type=str,
                        default="What is the nature of consciousness?")
    parser.add_argument("--goal", "-g", type=str, default="balanced",
                        choices=["accurate", "deep", "grounded", "stable",
                                "powerful", "balanced", "exploratory",
                                "supercritical", "wisdom"],
                        help="Navigation goal")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Compare all strategies on the topic")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    if args.interactive:
        interactive_mode(goal=args.goal)
    elif args.compare:
        run_comparison(args.topic, verbose=not args.quiet)
    else:
        run_dialogue(
            topic=args.topic,
            exchanges=args.exchanges,
            goal=args.goal,
            verbose=not args.quiet,
            save_results=not args.no_save
        )


if __name__ == "__main__":
    main()
