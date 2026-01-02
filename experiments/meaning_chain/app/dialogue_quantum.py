#!/usr/bin/env python3
"""
Quantum Semantic Dialogue: Navigation through (n, θ, r) space with Claude.

Uses 12-bit quantum encoding and verb operators for semantic navigation.
Claude responds to the quantum-guided semantic exploration.

Neoplatonic Structure:
    n=0-1: τὸ ἕν (The One) - Source, Origin
    n=2-3: νοῦς (Nous) - Intellect, Forms
    n=4-5: ψυχή (Soul) - Life, Motion
    n=6+:  ὕλη (Matter) - Manifestation

Usage:
    python dialogue_quantum.py --topic "What is wisdom?" --exchanges 5
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import argparse
import numpy as np
from dotenv import load_dotenv

# Add paths
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _APP_DIR.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

# Load environment variables from .env files
load_dotenv(_MEANING_CHAIN / ".env")
load_dotenv(_SEMANTIC_LLM / ".env")

from core.semantic_quantum import QuantumEncoder, QuantumWord, QuantumTrajectory
from chain_core.decomposer import Decomposer
from chain_core.renderer import Renderer, RendererConfig
from core.data_loader import DataLoader
from models.types import MeaningNode, MeaningTree, SemanticProperties


# Neoplatonic level names
NEOPLATONIC_LEVELS = {
    'one': {'name': 'The One', 'greek': 'τὸ ἕν', 'n_range': (0, 1)},
    'nous': {'name': 'Nous', 'greek': 'νοῦς', 'n_range': (2, 3)},
    'soul': {'name': 'Soul', 'greek': 'ψυχή', 'n_range': (4, 5)},
    'matter': {'name': 'Matter', 'greek': 'ὕλη', 'n_range': (6, 15)},
}


def get_ontological_level(n: int) -> dict:
    """Map orbital n to Neoplatonic level."""
    if n <= 1:
        return {'name': 'The One', 'greek': 'τὸ ἕν', 'description': 'Source, Origin'}
    elif n <= 3:
        return {'name': 'Nous', 'greek': 'νοῦς', 'description': 'Intellect, Forms'}
    elif n <= 5:
        return {'name': 'Soul', 'greek': 'ψυχή', 'description': 'Life, Motion'}
    else:
        return {'name': 'Matter', 'greek': 'ὕλη', 'description': 'Manifestation'}


class QuantumNavigatorAgent:
    """
    Agent that navigates semantic space using quantum (n, θ, r) coordinates.

    Uses verb operators to traverse meaning space following
    Neoplatonic emanation structure.
    """

    def __init__(self, name: str = "Quantum", use_claude_render: bool = False):
        self.name = name
        self.use_claude_render = use_claude_render
        self.quantum = QuantumEncoder()
        self.loader = DataLoader()
        self.decomposer = Decomposer(self.loader)

        # Renderer for building dynamic prompts
        self.renderer = Renderer(RendererConfig(
            model=os.getenv("OLLAMA_MODEL", "mistral:7b"),
            temperature=0.7,
            max_tokens=300,
            euler_mode=True  # Use Euler mode for orbital-style prompts
        ))

        # Claude client for rendering (optional)
        self._claude_client = None

        # Track navigation history
        self.trajectory_history: List[QuantumTrajectory] = []
        self.current_position: Optional[QuantumWord] = None

    def _render_with_claude(self, tree: MeaningTree, message: str, quantum_stats: dict) -> str:
        """Render using Claude API instead of Ollama."""
        if self._claude_client is None:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self._claude_client = anthropic.Anthropic(api_key=api_key)

        # Build prompt using renderer's method (dynamic, not hardcoded)
        user_prompt = self.renderer.build_prompt(tree, message, quantum_stats)
        system_prompt = self.renderer.build_system_prompt()

        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        response = self._claude_client.messages.create(
            model=model,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text

    def respond(self, message: str) -> dict:
        """
        Navigate semantic space based on message.

        Returns dict with navigation results and rendered response.
        """
        # Decompose message
        decomposed = self.decomposer.decompose(message)
        nouns = decomposed.nouns if decomposed.nouns else ["meaning"]
        verbs = decomposed.verbs if decomposed.verbs else []

        # Get seed concept
        seed = nouns[0]
        seed_qw = self.quantum.encode(seed)

        if seed_qw is None:
            seed = "meaning"
            seed_qw = self.quantum.encode(seed)

        # Navigate using verbs
        if verbs:
            trajectory = self.quantum.chain(seed, verbs[:4])
            self.trajectory_history.append(trajectory)

            if trajectory.steps:
                self.current_position = trajectory.end
        else:
            trajectory = None
            self.current_position = seed_qw

        # Collect concepts at current position
        concepts = [seed]
        if self.current_position:
            neighbors = self.quantum.nearest(self.current_position, k=5)
            for word, dist, _ in neighbors:
                if word not in concepts and dist < 1.0:
                    concepts.append(word)

        # Determine ontological level
        level = get_ontological_level(self.current_position.n if self.current_position else 3)

        # Build quantum stats for renderer (like euler_stats)
        quantum_stats = self._build_quantum_stats(
            seed, verbs, trajectory, concepts, level
        )

        # Build a tree for the renderer
        tree = self._build_meaning_tree(concepts, level)

        # Render response using dynamic prompts
        if self.use_claude_render:
            response_text = self._render_with_claude(tree, message, quantum_stats)
        else:
            result = self.renderer.render(tree, message, euler_stats=quantum_stats)
            response_text = result['response']

        return {
            'response': response_text,
            'concepts': concepts,
            'trajectory': trajectory,
            'position': self.current_position,
            'ontological_level': level,
            'nouns': nouns,
            'verbs': verbs,
            'quantum_stats': quantum_stats
        }

    def _build_quantum_stats(self, seed: str, verbs: List[str],
                             trajectory: Optional[QuantumTrajectory],
                             concepts: List[str], level: dict) -> dict:
        """Build quantum stats for renderer (similar to euler_stats)."""
        pos = self.current_position

        # Compute mean tau from concepts
        tau_vals = []
        for concept in concepts:
            qw = self.quantum.encode(concept)
            if qw:
                tau_vals.append(qw.tau)

        mean_tau = np.mean(tau_vals) if tau_vals else 2.5
        orbital_n = int(round((mean_tau - 1) * np.e)) if tau_vals else 4

        # Phase change from trajectory
        phase_change = trajectory.phase_change if trajectory else 0

        return {
            'mean_tau': mean_tau,
            'orbital_n': orbital_n,
            'realm': 'human' if mean_tau < np.e else 'transcendental',
            'below_veil': mean_tau < np.e,
            'near_ground': abs(mean_tau - 1.37) < 0.5,
            'veil_crossings': abs(phase_change) // 90 if trajectory else 0,
            'human_fraction': sum(1 for t in tau_vals if t < np.e) / len(tau_vals) if tau_vals else 0.5,
            'convergence': concepts[0] if concepts else seed,
            'core_concepts': concepts[:5],
            'known_nouns': concepts,
            'key_symbols': [seed] + verbs[:3],
            # Quantum-specific
            'ontological_level': level['name'],
            'ontological_greek': level['greek'],
            'phase_change': phase_change,
            'current_theta': pos.theta_deg if pos else 0,
            'current_r': pos.r if pos else 0,
        }

    def _build_meaning_tree(self, concepts: List[str], level: dict) -> MeaningTree:
        """Build a MeaningTree from quantum concepts."""
        if not concepts:
            concepts = ["meaning"]

        root_word = concepts[0]
        pos = self.current_position

        root = MeaningNode(
            word=root_word,
            properties=SemanticProperties(
                g=0.0,
                tau=pos.tau if pos else 2.5,
                j=np.zeros(5)
            ),
            depth=0
        )

        # Add child nodes for other concepts
        for i, concept in enumerate(concepts[1:5]):
            qw = self.quantum.encode(concept)
            if qw:
                child = MeaningNode(
                    word=concept,
                    properties=SemanticProperties(
                        g=0.0,
                        tau=qw.tau,
                        j=np.zeros(5)
                    ),
                    depth=1
                )
                root.children.append(child)

        return MeaningTree(roots=[root])


class ClaudeDialogueAgent:
    """Agent using Claude API for philosophical dialogue."""

    def __init__(self, name: str = "Claude"):
        self.name = name
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.client = None
        self.history = []

        # Use Renderer for dynamic system prompt
        self.renderer = Renderer(RendererConfig(euler_mode=True))

    def _init_client(self):
        if self.client is None:
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment")
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic")

    def respond(self, message: str) -> dict:
        """Get response from Claude using dynamic prompts."""
        self._init_client()

        # Use renderer's system prompt (not hardcoded)
        system_prompt = self.renderer.build_system_prompt()

        self.history.append({"role": "user", "content": message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system_prompt,
            messages=self.history
        )

        reply = response.content[0].text
        self.history.append({"role": "assistant", "content": reply})

        return {
            'response': reply,
            'model': self.model
        }


def run_dialogue(topic: str, exchanges: int = 5, verbose: bool = True,
                 use_claude_render: bool = False, save_results: bool = True):
    """Run quantum dialogue between Navigator and Claude."""
    import json
    from datetime import datetime

    print("=" * 70)
    print("QUANTUM SEMANTIC DIALOGUE")
    print("Navigation through (n, θ, r) Neoplatonic space")
    print("=" * 70)
    print()
    print("Ontological Structure:")
    print("  n=0-1: τὸ ἕν (The One) - Source, Origin")
    print("  n=2-3: νοῦς (Nous) - Intellect, Forms")
    print("  n=4-5: ψυχή (Soul) - Life, Motion")
    print("  n=6+:  ὕλη (Matter) - Manifestation")
    print()
    print(f"Topic: {topic}")
    print(f"Exchanges: {exchanges}")
    if use_claude_render:
        print("Renderer: Claude (both agents use Claude)")
    else:
        print("Renderer: Ollama (local)")
    print()

    # Initialize agents
    quantum_agent = QuantumNavigatorAgent("Navigator", use_claude_render=use_claude_render)
    claude_agent = ClaudeDialogueAgent("Claude")

    # Track dialogue
    dialogue_log = {
        "topic": topic,
        "exchanges": exchanges,
        "timestamp": datetime.now().isoformat(),
        "turns": []
    }

    current_message = topic
    speakers = [(claude_agent, quantum_agent), (quantum_agent, claude_agent)]

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

            if verbose and speaker.name == "Navigator":
                stats = result.get('quantum_stats')
                if stats:
                    print(f"\n  [Quantum Navigation]")
                    print(f"    Level: {stats['ontological_greek']} ({stats['ontological_level']})")
                    print(f"    Core: {stats['core_concepts'][:4]}")
                    print(f"    Mean tau: {stats['mean_tau']:.2f} (n={stats['orbital_n']})")
                    print(f"    Phase: θ={stats['current_theta']:+.0f}°, r={stats['current_r']:.2f}")
                    if stats.get('phase_change'):
                        print(f"    Phase change: Δθ={stats['phase_change']:+.0f}°")

                    turn["quantum_stats"] = {
                        'level': stats['ontological_level'],
                        'greek': stats['ontological_greek'],
                        'mean_tau': stats['mean_tau'],
                        'orbital_n': stats['orbital_n'],
                        'theta': stats['current_theta'],
                        'r': stats['current_r'],
                        'phase_change': stats.get('phase_change', 0),
                        'concepts': stats['core_concepts'][:5]
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

        # Summary
        print("\n" + "=" * 70)
        print("DIALOGUE COMPLETE")
        print("=" * 70)

        # Show trajectory
        if quantum_agent.trajectory_history:
            print("\nQuantum Trajectory:")
            for traj in quantum_agent.trajectory_history:
                if traj.steps:
                    start_hex = traj.start.to_hex() if traj.start else "?"
                    end_hex = traj.end.to_hex() if traj.end else "?"
                    print(f"  {start_hex} → {end_hex} (Δθ={traj.phase_change:+.0f}°)")

        # Final position
        if quantum_agent.current_position:
            pos = quantum_agent.current_position
            level = get_ontological_level(pos.n)
            print(f"\nFinal Position: {pos.to_hex()}")
            print(f"  n={pos.n}, θ={pos.theta_deg:+.0f}°, r={pos.r:.2f}")
            print(f"  Level: {level['greek']} ({level['name']})")

        # Save results
        if save_results:
            results_dir = _MEANING_CHAIN / "results" / "dialogue_quantum"
            results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"dialogue_{timestamp}.json"

            # Add final state
            if quantum_agent.current_position:
                dialogue_log["final_position"] = {
                    "hex": quantum_agent.current_position.to_hex(),
                    "n": quantum_agent.current_position.n,
                    "theta": quantum_agent.current_position.theta_deg,
                    "r": quantum_agent.current_position.r
                }

            with open(result_file, 'w') as f:
                json.dump(dialogue_log, f, indent=2, ensure_ascii=False, default=str)

            print(f"\nResults saved to: {result_file}")

    finally:
        pass  # No cleanup needed for quantum encoder


def demo_navigation():
    """Demo quantum navigation without Claude."""
    print("=" * 70)
    print("QUANTUM NAVIGATION DEMO (no Claude)")
    print("=" * 70)
    print()

    quantum = QuantumEncoder()

    # Test queries
    queries = [
        ("What is truth?", ["truth"], ["seek", "find", "know"]),
        ("How do we love?", ["love"], ["feel", "give", "receive"]),
        ("Understanding consciousness", ["consciousness"], ["think", "perceive", "understand"]),
    ]

    for query, nouns, verbs in queries:
        print(f"Query: {query}")
        print(f"  Seed: {nouns[0]}")
        print(f"  Verbs: {verbs}")

        # Navigate
        trajectory = quantum.chain(nouns[0], verbs)

        if trajectory.steps:
            print(f"  Trajectory:")
            for i, step in enumerate(trajectory.steps):
                level = get_ontological_level(step.n)
                print(f"    [{i}] {step.to_hex()} n={step.n} θ={step.theta_deg:+.0f}° r={step.r:.2f} ({level['name']})")

            print(f"  Phase change: Δθ={trajectory.phase_change:+.0f}°")
            print(f"  Total shift: ΔA={trajectory.total_shift[0]:+.3f}, ΔS={trajectory.total_shift[1]:+.3f}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Quantum Semantic Dialogue")
    parser.add_argument("--topic", type=str, default="What is the nature of wisdom?",
                        help="Starting topic for dialogue")
    parser.add_argument("--exchanges", type=int, default=5,
                        help="Number of dialogue exchanges")
    parser.add_argument("--demo", action="store_true",
                        help="Run navigation demo without Claude")
    parser.add_argument("--claude-render", "-c", action="store_true",
                        help="Use Claude for rendering Navigator responses")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode (less verbose)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")

    args = parser.parse_args()

    if args.demo:
        demo_navigation()
    else:
        # Check for API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Please set ANTHROPIC_API_KEY in .env file or environment")
            print("  export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)

        run_dialogue(
            topic=args.topic,
            exchanges=args.exchanges,
            verbose=not args.quiet,
            use_claude_render=args.claude_render,
            save_results=not args.no_save
        )


if __name__ == "__main__":
    main()
