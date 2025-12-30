#!/usr/bin/env python3
"""
Resonant Orbital Dialogue - Claude with Orbital-Tuned Semantic Laser

Uses the orbital resonance system to tune responses:
- Detect natural orbital of query
- Tune to that orbital for maximum coherence
- Render with Claude using the resonant beam

Key insight: driving at natural frequency → maximum coherence (proven by spectroscopy)

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python dialogue_resonant.py --exchanges 5 --topic "help me understand wisdom"
    python dialogue_resonant.py --mode ground --topic "how do I fix my sleep?"
    python dialogue_resonant.py --mode transcend --topic "what is the nature of consciousness?"
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

from graph.meaning_graph import MeaningGraph
from chain_core.decomposer import Decomposer
from chain_core.orbital import (
    ResonantLaser, ResonantResult,
    OrbitalDetector, IntentOrbitalMapper,
    E, VEIL_TAU, orbital_to_tau, tau_to_orbital
)


class ResonantSemanticAgent:
    """
    Agent using orbital-tuned resonant laser.

    Modes:
    - auto: Detect natural orbital, tune softly
    - ground: Force to human realm (practical responses)
    - transcend: Force above Veil (philosophical responses)
    - adaptive: Multi-pass with feedback
    """

    def __init__(self, name: str = "Resonant",
                 mode: str = "auto",
                 tuning: str = "soft"):
        self.name = name
        self.mode = mode
        self.tuning = tuning

        self.graph = MeaningGraph()
        if not self.graph.is_connected():
            raise RuntimeError("Neo4j not connected. Start with: cd config && docker-compose up -d")

        self.laser = ResonantLaser(self.graph)
        self.decomposer = Decomposer()
        self._claude_client = None

    def _decompose(self, query: str) -> Dict:
        """Decompose query into nouns and verbs."""
        result = self.decomposer.decompose(query)
        return {
            'nouns': result.nouns,
            'verbs': result.verbs
        }

    def _render_with_claude(self, result: ResonantResult, question: str) -> str:
        """Render using Claude API based on resonant laser output."""
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.Anthropic()

        prompt = self._build_prompt(result, question)

        response = self._claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            system=self._build_system_prompt(result),
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _build_system_prompt(self, result: ResonantResult) -> str:
        """Build system prompt based on orbital level."""
        orbital = result.target_orbital
        tau = orbital_to_tau(orbital)

        if tau < 2.0:
            tone = "grounded and practical"
            style = "concrete, actionable language"
        elif tau < VEIL_TAU:
            tone = "balanced between concrete and abstract"
            style = "accessible yet thoughtful language"
        else:
            tone = "reflective and philosophical"
            style = "contemplative, nuanced language"

        return f"""You respond based on orbital-tuned semantic resonance.

The query resonates at orbital n={orbital} (τ={tau:.2f}).
This is {'human realm' if tau < VEIL_TAU else 'transcendental realm'}.

Tone: {tone}
Style: {style}

Respond using the coherent beam concepts naturally.
2-4 sentences, substantive, not formulaic.
Don't list concepts - weave them into natural language."""

    def _build_prompt(self, result: ResonantResult, question: str) -> str:
        """Build prompt from resonant result."""
        sections = []

        # Header
        sections.append("## Resonant Semantic Beam")
        sections.append(f"Question: {question}")
        sections.append("")

        # Orbital info
        sections.append("## Orbital Resonance")
        sections.append(f"  Detected orbital: n={result.detected_orbital} (τ={orbital_to_tau(result.detected_orbital):.2f})")
        sections.append(f"  Target orbital: n={result.target_orbital} (τ={orbital_to_tau(result.target_orbital):.2f})")
        sections.append(f"  Resonance quality: {result.resonance_quality:.0%}")
        sections.append(f"  Coherence: {result.coherence:.2f}")

        realm = "human" if orbital_to_tau(result.target_orbital) < VEIL_TAU else "transcendental"
        sections.append(f"  Realm: {realm}")
        sections.append("")

        # Coherent beam
        if result.concepts:
            sections.append("## Coherent Beam (resonant concepts)")
            sections.append(f"  {', '.join(result.concepts[:10])}")
            sections.append("")

        # Intent mapping
        if result.intent_mapping:
            im = result.intent_mapping
            sections.append(f"## Intent Analysis")
            sections.append(f"  Profile: {im.intent_profile}")
            sections.append(f"  Confidence: {im.confidence:.0%}")
            sections.append(f"  Data source: {im.data_source}")
            sections.append("")

        # Population stats
        pop = result.population
        if pop:
            sections.append("## Population Statistics")
            sections.append(f"  τ mean: {pop.get('tau_mean', 0):.2f} ± {pop.get('tau_std', 0):.2f}")
            sections.append(f"  g mean: {pop.get('g_mean', 0):.2f}")
            sections.append(f"  Human fraction: {pop.get('human_fraction', 0):.0%}")
            sections.append("")

        # Response guidance
        sections.append("## Response Guidance")
        if result.resonance_quality > 0.7:
            sections.append("- High resonance: confident, focused response")
        elif result.resonance_quality > 0.4:
            sections.append("- Medium resonance: explore the core concepts")
        else:
            sections.append("- Low resonance: acknowledge ambiguity")

        sections.append("- Use beam concepts naturally")
        sections.append("- 2-4 sentences")

        return "\n".join(sections)

    def respond(self, message: str) -> Dict[str, Any]:
        """Process through resonant laser and render with Claude."""
        # Decompose query
        decomp = self._decompose(message)
        nouns = decomp.get('nouns', [])
        verbs = decomp.get('verbs', [])

        if not nouns:
            nouns = ['meaning']  # Default seed

        # Lase based on mode
        if self.mode == "ground":
            result = self.laser.lase_grounded(nouns, verbs)
        elif self.mode == "transcend":
            result = self.laser.lase_transcendent(nouns, verbs)
        elif self.mode == "adaptive":
            result = self.laser.lase_adaptive(nouns, verbs, passes=2)
        else:  # auto
            result = self.laser.lase_resonant(nouns, verbs, tuning=self.tuning)

        # Render with Claude
        response_text = self._render_with_claude(result, message)

        return {
            'response': response_text,
            'resonance': {
                'detected_orbital': result.detected_orbital,
                'target_orbital': result.target_orbital,
                'resonance_quality': result.resonance_quality,
                'coherence': result.coherence,
                'tau_mean': result.population.get('tau_mean', 0),
                'beam_concepts': result.concepts[:8],
                'intent_profile': result.intent_mapping.intent_profile if result.intent_mapping else None,
                'tuning': result.tuning_applied
            },
            'decomposition': {
                'nouns': nouns,
                'verbs': verbs
            }
        }

    def close(self):
        self.laser.close()
        self.graph.close()


class ClaudeAgent:
    """Standard Claude agent for dialogue."""

    def __init__(self, name: str = "Claude", model: str = "claude-sonnet-4-20250514"):
        self.name = name
        self.model = model
        self.client = None
        self.history = []

        self.system = """You are engaged in a philosophical dialogue about meaning.
Your partner uses orbital-tuned semantic resonance - tuning to the natural
frequency of queries to maximize coherence.

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


def run_resonant_dialogue(topic: str,
                          exchanges: int = 5,
                          mode: str = "auto",
                          tuning: str = "soft",
                          verbose: bool = True,
                          save_results: bool = True):
    """Run orbital-resonant dialogue between MeaningChain and Claude."""

    print("=" * 70)
    print("ORBITAL RESONANT DIALOGUE")
    print("=" * 70)
    print(f"\nOrbital Resonance Mode: {mode}")
    print(f"Tuning: {tuning}")
    print(f"Euler constant e = {E:.4f}")
    print(f"Veil at τ = e ≈ {VEIL_TAU:.2f}")
    print()
    print(f"Topic: {topic}")
    print(f"Exchanges: {exchanges}")
    print()

    # Create agents
    try:
        semantic = ResonantSemanticAgent("Resonant", mode=mode, tuning=tuning)
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    claude = ClaudeAgent("Claude")

    # Track dialogue
    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "mode": mode,
        "tuning": tuning,
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    # Aggregate resonance stats
    all_resonance = []

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

            if verbose and speaker.name == "Resonant":
                res = result.get('resonance', {})
                decomp = result.get('decomposition', {})

                print(f"\n  [Query Decomposition]")
                print(f"    Nouns: {decomp.get('nouns', [])}")
                print(f"    Verbs: {decomp.get('verbs', [])}")

                print(f"\n  [Orbital Resonance]")
                print(f"    Detected: n={res.get('detected_orbital')} → Target: n={res.get('target_orbital')}")
                print(f"    Resonance: {res.get('resonance_quality', 0):.0%}")
                print(f"    Coherence: {res.get('coherence', 0):.2f}")
                print(f"    τ mean: {res.get('tau_mean', 0):.2f}")
                print(f"    Intent: {res.get('intent_profile', 'N/A')}")

                print(f"\n  [Coherent Beam]")
                print(f"    {res.get('beam_concepts', [])[:6]}")

                turn["resonance"] = res
                turn["decomposition"] = decomp
                all_resonance.append(res)

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

        # Summary
        print("\n" + "=" * 70)
        print("DIALOGUE COMPLETE - RESONANCE SUMMARY")
        print("=" * 70)

        if all_resonance:
            avg_resonance = np.mean([r.get('resonance_quality', 0) for r in all_resonance])
            avg_coherence = np.mean([r.get('coherence', 0) for r in all_resonance])
            avg_tau = np.mean([r.get('tau_mean', 0) for r in all_resonance])

            # Collect beam concepts
            all_concepts = []
            for r in all_resonance:
                all_concepts.extend(r.get('beam_concepts', []))
            concept_counts = {}
            for c in all_concepts:
                concept_counts[c] = concept_counts.get(c, 0) + 1
            top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:10]

            print(f"\nResonance Statistics:")
            print(f"  Average resonance quality: {avg_resonance:.0%}")
            print(f"  Average coherence: {avg_coherence:.2f}")
            print(f"  Average τ: {avg_tau:.2f} (orbital n≈{tau_to_orbital(avg_tau)})")

            # Realm distribution
            human_count = sum(1 for r in all_resonance if r.get('tau_mean', 3) < VEIL_TAU)
            print(f"  Human realm turns: {human_count}/{len(all_resonance)}")

            print(f"\nMost resonant concepts:")
            for concept, count in top_concepts[:8]:
                print(f"    {concept}: {count}")

            dialogue_log["aggregate_stats"] = {
                "avg_resonance_quality": avg_resonance,
                "avg_coherence": avg_coherence,
                "avg_tau": avg_tau,
                "top_concepts": top_concepts
            }

        # Save results
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_resonant"
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"resonant_dialogue_{mode}_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(dialogue_log, f, indent=2, default=str)

            print(f"\nResults saved to: {filename}")

    finally:
        semantic.close()


def main():
    parser = argparse.ArgumentParser(description="Orbital resonant dialogue with Claude")
    parser.add_argument("--exchanges", "-e", type=int, default=5)
    parser.add_argument("--topic", "-t", type=str,
                        default="What do my dreams tell me about myself?")
    parser.add_argument("--mode", "-m", type=str, default="auto",
                        choices=["auto", "ground", "transcend", "adaptive"],
                        help="Orbital mode: auto, ground, transcend, adaptive")
    parser.add_argument("--tuning", type=str, default="soft",
                        choices=["none", "soft", "hard"],
                        help="Seed tuning strength")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    run_resonant_dialogue(
        topic=args.topic,
        exchanges=args.exchanges,
        mode=args.mode,
        tuning=args.tuning,
        verbose=not args.quiet,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
