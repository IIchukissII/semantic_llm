#!/usr/bin/env python3
"""
Paradox-Powered Dialogue: Double Explosion Speech Generation.

Uses Monte Carlo sampling to find semantic attractors,
then paradox detection to find archetypal and conceptual tension,
then double explosion to generate powerful statements.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python dialogue_paradox.py --topic "What is the meaning of suffering?"
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import argparse
from datetime import datetime
import json

# Add paths
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _APP_DIR.parent

sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from chain_core.paradox_detector import ParadoxDetector


class ParadoxSpeaker:
    """Agent that speaks with paradoxical power."""

    def __init__(self, name: str = "Paradox-Speaker",
                 intent_strength: float = 0.3,
                 n_samples: int = 20):
        self.name = name
        self.detector = ParadoxDetector(
            intent_strength=intent_strength,
            n_samples=n_samples
        )
        self._claude_client = None

    def _get_claude(self):
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.Anthropic()
        return self._claude_client

    def respond(self, message: str) -> Dict[str, Any]:
        """
        Respond using paradox-powered speech.

        1. Detect paradoxes in the message
        2. Build prompt with archetypal + conceptual tension
        3. Generate response through Claude
        """
        # Detect paradoxes
        analysis = self.detector.speak_powerfully(message)

        if not analysis['paradox_found']:
            # No paradox - simple response
            return self._simple_response(message)

        # Build paradox-informed prompt
        prompt = self._build_paradox_prompt(message, analysis)
        system = self._build_system_prompt()

        # Generate with Claude
        client = self._get_claude()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            'response': response.content[0].text,
            'analysis': analysis,
            'paradox_found': True
        }

    def _build_system_prompt(self) -> str:
        return """You speak with paradoxical power.

You have been given the archetypal and conceptual tensions in the question.
Use these oppositions to create speech that holds both poles at once.

Guidelines:
- Don't list the paradoxes, embody them
- Speak from the tension, not about it
- Let the archetype (she/thee, anima/other) carry the concept
- 3-5 sentences of substantive insight
- End with synthesis, not resolution"""

    def _build_paradox_prompt(self, question: str, analysis: Dict) -> str:
        sections = []

        sections.append(f"## Question\n{question}\n")

        # Archetypal layer
        if 'archetypal' in analysis:
            a = analysis['archetypal']
            sections.append(f"## Archetypal Tension")
            sections.append(f"{a['thesis']} ↔ {a['antithesis']}")
            sections.append(f"Dimension: {a['dimension']} (power: {a['power']:.1f})")
            sections.append("")

        # Conceptual layer
        if 'conceptual' in analysis:
            c = analysis['conceptual']
            sections.append(f"## Conceptual Tension")
            sections.append(f"{c['thesis']} ↔ {c['antithesis']}")
            sections.append(f"Dimension: {c['dimension']} (power: {c['power']:.1f})")
            if c.get('synthesis'):
                sections.append(f"Synthesis concepts: {', '.join(c['synthesis'][:3])}")
            sections.append("")

        # Double explosion
        if 'double_explosion' in analysis:
            sections.append(f"## Double Explosion")
            sections.append(analysis['double_explosion'])
            sections.append("")

        # Suggested statements
        if 'statements' in analysis:
            sections.append("## Suggested Formulations")
            for s in analysis['statements'][:3]:
                sections.append(f"  • {s}")
            sections.append("")

        sections.append("## Task")
        sections.append("Respond to the question, letting both tensions inform your speech.")
        sections.append("The archetype carries the concept. Hold both poles.")

        return "\n".join(sections)

    def _simple_response(self, message: str) -> Dict[str, Any]:
        """Fallback when no paradox detected."""
        client = self._get_claude()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": message}]
        )
        return {
            'response': response.content[0].text,
            'paradox_found': False
        }

    def close(self):
        self.detector.close()


def run_paradox_dialogue(topic: str, exchanges: int = 3,
                         verbose: bool = True, save_results: bool = True):
    """Run a paradox-powered dialogue with chain reaction tracking."""

    print("=" * 70)
    print("PARADOX-POWERED DIALOGUE")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print(f"Mode: Double Explosion (Archetypal + Conceptual)")
    print()

    speaker = ParadoxSpeaker(n_samples=20)

    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "mode": "paradox_double_explosion",
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    # Chain reaction tracking
    chain_metrics = {
        "powers": [],      # Paradox power over exchanges
        "tensions": [],    # Tension accumulation
        "depths": []       # Response richness (word count)
    }

    current_message = topic

    try:
        for i in range(exchanges):
            print("-" * 70)
            print(f"Exchange {i + 1}/{exchanges}")
            print("-" * 70)

            result = speaker.respond(current_message)
            response = result['response']

            turn = {
                "exchange": i + 1,
                "input": current_message,
                "response": response,
                "paradox_found": result['paradox_found']
            }

            if verbose and result['paradox_found']:
                analysis = result['analysis']
                print(f"\n  [Paradox Analysis]")

                # Track chain metrics
                max_power = 0
                total_tension = 0

                if 'archetypal' in analysis:
                    a = analysis['archetypal']
                    print(f"    Archetypal: {a['thesis']} ↔ {a['antithesis']} "
                          f"({a['dimension']}, power={a['power']:.1f})")
                    max_power = max(max_power, a['power'])
                    total_tension += a['tension']

                if 'conceptual' in analysis:
                    c = analysis['conceptual']
                    print(f"    Conceptual: {c['thesis']} ↔ {c['antithesis']} "
                          f"({c['dimension']}, power={c['power']:.1f})")
                    max_power = max(max_power, c['power'])
                    total_tension += c['tension']

                if 'double_explosion' in analysis:
                    print(f"    Double: {analysis['double_explosion'][:60]}...")

                # Record chain metrics
                chain_metrics['powers'].append(max_power)
                chain_metrics['tensions'].append(total_tension)
                chain_metrics['depths'].append(len(response.split()))

                turn['analysis'] = {
                    'archetypal': analysis.get('archetypal'),
                    'conceptual': analysis.get('conceptual'),
                    'double_explosion': analysis.get('double_explosion')
                }
            else:
                chain_metrics['powers'].append(0)
                chain_metrics['tensions'].append(0)
                chain_metrics['depths'].append(len(response.split()))

            dialogue_log["turns"].append(turn)

            # Print response
            print(f"\n[Response]")
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

            # Use response as next input (dialogue with self)
            current_message = response

        print("\n" + "=" * 70)
        print("DIALOGUE COMPLETE - CHAIN REACTION ANALYSIS")
        print("=" * 70)

        # Chain reaction summary
        if len(chain_metrics['powers']) >= 2:
            powers = chain_metrics['powers']
            tensions = chain_metrics['tensions']
            depths = chain_metrics['depths']

            print(f"\n[CHAIN REACTION METRICS]")
            print(f"  Exchange |  Power  | Tension | Depth ")
            print(f"  ---------|---------|---------|-------")
            for i, (p, t, d) in enumerate(zip(powers, tensions, depths)):
                print(f"     {i+1}     | {p:7.1f} | {t:7.2f} | {d:5d}")

            # Amplification ratios
            if powers[0] > 0 and len(powers) > 1:
                power_amp = powers[-1] / powers[0]
                print(f"\n  Power amplification: {power_amp:.1f}x")
                print(f"  (First → Last: {powers[0]:.1f} → {powers[-1]:.1f})")

            if len(depths) > 1:
                depth_amp = depths[-1] / max(depths[0], 1)
                print(f"  Depth amplification: {depth_amp:.2f}x")

            # Chain reaction coefficient
            if len(powers) > 1:
                # Average growth rate
                growth_rates = []
                for i in range(1, len(powers)):
                    if powers[i-1] > 0:
                        growth_rates.append(powers[i] / powers[i-1])
                if growth_rates:
                    avg_growth = sum(growth_rates) / len(growth_rates)
                    print(f"\n  Chain coefficient (λ): {avg_growth:.2f}")
                    if avg_growth > 1:
                        print(f"    λ > 1: SUPERCRITICAL (meaning amplifies)")
                    elif avg_growth == 1:
                        print(f"    λ = 1: CRITICAL (meaning sustains)")
                    else:
                        print(f"    λ < 1: SUBCRITICAL (meaning decays)")

            dialogue_log["chain_metrics"] = chain_metrics
            dialogue_log["chain_reaction"] = {
                "power_amplification": powers[-1] / powers[0] if powers[0] > 0 else 0,
                "chain_coefficient": avg_growth if growth_rates else 0
            }

        # Save
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_paradox"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"paradox_dialogue_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(dialogue_log, f, indent=2, default=str)

            print(f"\nResults saved to: {filename}")

    finally:
        speaker.close()


def main():
    parser = argparse.ArgumentParser(
        description="Paradox-powered dialogue with double explosion"
    )
    parser.add_argument("--topic", "-t", type=str,
                        default="What is the meaning of suffering?")
    parser.add_argument("--exchanges", "-e", type=int, default=3)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    run_paradox_dialogue(
        topic=args.topic,
        exchanges=args.exchanges,
        verbose=not args.quiet,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
