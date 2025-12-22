#!/usr/bin/env python3
"""
Conversation Simulation Test with Optimization
================================================

Simulates a dialogue between:
- Speaker 1 (Claude): Provides semantic prompts/concepts
- Speaker 2 (Hybrid): Navigates using OPTIMIZATION ALGORITHMS + LLM

Uses Hill Climbing, Simulated Annealing, and Random Local Search
for smarter semantic navigation.
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Add parent directories to find core module
# Note: Order matters - local optimization_algorithms.py should take priority
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # semantic_llm dir (for core module)
sys.path.insert(0, str(Path(__file__).parent))  # local dir (for optimization_algorithms with quantum)

from core.hybrid_llm import HybridQuantumLLM, QuantumCore, Trajectory, Transition
from optimization_algorithms import SemanticOptimizer


class ConversationSimulator:
    """Simulates a conversation between two speakers."""

    def __init__(self, algorithm: str = "hill_climbing", model: str = None):
        """
        Initialize conversation simulator.

        Args:
            algorithm: "hill_climbing" | "simulated_annealing" | "random_local" | "quantum" | "compare"
            model: Optional ollama model name (e.g. "mistral:7b", "qwen2.5:1.5b")
        """
        print("=" * 60)
        print("CONVERSATION SIMULATION WITH OPTIMIZATION")
        print("=" * 60)
        print(f"\nInitializing Hybrid System... (model: {model or 'default'})")
        self.hybrid = HybridQuantumLLM(renderer="ollama", fidelity_threshold=0.4, model=model)

        print("\nInitializing Semantic Optimizer...")
        self.optimizer = SemanticOptimizer(core=self.hybrid.core, max_edges=50000)

        self.algorithm = algorithm
        self.model = model or "qwen2.5:1.5b"  # Store model name
        self.conversation_history = []
        self.all_dialogues = []  # Store all dialogues for saving
        print(f"\nUsing algorithm: {algorithm}")

    def speaker1_prompt(self, concept: str, intent: str = "good") -> dict:
        """
        Speaker 1 (Claude) provides a concept to discuss.

        Returns the semantic analysis of the concept.
        """
        state = self.hybrid.core.get_state(concept)
        if state is None:
            return {"error": f"Unknown concept: {concept}"}

        return {
            "speaker": "Claude",
            "concept": concept,
            "intent": intent,
            "goodness": state.goodness,
            "tau": state.tau,
            "message": f"Let's explore '{concept}' (g={state.goodness:+.2f}, tau={state.tau:.2f})"
        }

    def speaker2_respond(self, concept: str, goal: str = "good", steps: int = 3) -> dict:
        """
        Speaker 2 (Hybrid) navigates from the concept using OPTIMIZATION ALGORITHMS.
        """
        # Select objective function based on goal
        if goal == "good":
            objective = self.optimizer.objective_goodness
        elif goal == "evil":
            objective = lambda w: -self.optimizer.objective_goodness(w)  # Minimize goodness
        else:
            objective = self.optimizer.objective_combined

        # Run optimization algorithm
        print(f"\n  [OPTIMIZER] Running {self.algorithm}...")

        if self.algorithm == "hill_climbing":
            opt_result = self.optimizer.hill_climbing(
                concept, objective, max_steps=steps * 3, verbose=True
            )
        elif self.algorithm == "simulated_annealing":
            opt_result = self.optimizer.simulated_annealing(
                concept, objective, max_steps=steps * 5,
                initial_temp=1.0, cooling_rate=0.9, verbose=True
            )
        elif self.algorithm == "random_local":
            opt_result = self.optimizer.random_local_search(
                concept, objective, max_steps=steps * 3,
                restarts=3, verbose=True
            )
        elif self.algorithm == "quantum":
            # Quantum Annealing: combines thermal + quantum tunneling
            # PRIORITY: tunneling first, thermal as fallback
            # P(tunnel) = believe × e^(-2κd) [quantum insight]
            # P(accept) = e^(Δg/T) [thermal fallback]
            opt_result = self.optimizer.quantum_annealing(
                concept, objective, max_steps=steps * 5,
                initial_temp=1.0, cooling_rate=0.9,
                tunnel_threshold=0.3, believe=0.5, verbose=True
            )
        elif self.algorithm == "compare":
            # Compare all algorithms and pick best
            results = self.optimizer.compare_algorithms(concept, "goodness" if goal == "good" else "combined", steps * 2)
            opt_result = max(results.values(), key=lambda r: abs(r.improvement))
        else:
            opt_result = self.optimizer.hill_climbing(concept, objective, max_steps=steps * 3, verbose=True)

        print(f"  [OPTIMIZER] Path: {' -> '.join(opt_result.path)}")
        print(f"  [OPTIMIZER] Improvement: {opt_result.improvement:+.3f} in {opt_result.steps} steps")

        # Convert optimization path to trajectory for LLM rendering
        trajectory = self._path_to_trajectory(opt_result.path, goal)

        if trajectory is None:
            return {"error": "Failed to build trajectory from optimization path"}

        # FEEDBACK LOOP: Render and verify with retries
        max_retries = 3
        best_result = None
        best_fidelity = 0

        for attempt in range(max_retries):
            print(f"\n  [FEEDBACK LOOP] Attempt {attempt + 1}/{max_retries}")

            # Render with LLM
            print(f"    [LLM] Rendering trajectory...")
            text = self.hybrid.renderer.render(trajectory, "narrative")
            print(f"    [LLM] Output: \"{text[:80]}...\"" if len(text) > 80 else f"    [LLM] Output: \"{text}\"")

            # Verify fidelity
            result = self.hybrid.encoder.verify(trajectory, text, self.hybrid.fidelity_threshold)
            print(f"    [VERIFY] Fidelity: {result.fidelity:.2f} | Extracted: {len(result.extracted_words)} words")

            # Track best result
            if result.fidelity > best_fidelity:
                best_fidelity = result.fidelity
                best_result = result

            if result.accepted:
                print(f"    [FEEDBACK] ✓ ACCEPTED (fidelity >= {self.hybrid.fidelity_threshold})")
                break
            else:
                print(f"    [FEEDBACK] ✗ REJECTED (fidelity < {self.hybrid.fidelity_threshold}), retrying...")

        # Use best result even if not fully accepted
        result = best_result if best_result else result

        return {
            "speaker": "Hybrid",
            "algorithm": opt_result.algorithm,
            "trajectory": opt_result.path,
            "response": result.text,
            "delta_g": opt_result.improvement,
            "efficiency": opt_result.efficiency,
            "fidelity": result.fidelity,
            "accepted": result.accepted,
            "end_state": opt_result.end_word,
            "steps": opt_result.steps,
            "retries": attempt + 1
        }

    def _path_to_trajectory(self, path: list, goal: str) -> Trajectory:
        """Convert optimization path to Trajectory for LLM rendering."""
        if len(path) < 2:
            # Handle single-word path (local maximum already reached)
            state = self.hybrid.core.get_state(path[0])
            if state:
                # Create a "stay" trajectory with no transitions
                return Trajectory(start=state, transitions=[], goal=goal)
            return None

        start_state = self.hybrid.core.get_state(path[0])
        if start_state is None:
            return None

        transitions = []
        for i in range(len(path) - 1):
            from_word = path[i]
            to_word = path[i + 1]

            from_state = self.hybrid.core.get_state(from_word)
            to_state = self.hybrid.core.get_state(to_word)

            if from_state is None or to_state is None:
                continue

            # Find a verb that connects these (or use generic)
            verb = self._find_connecting_verb(from_word, to_word)

            delta_g = to_state.goodness - from_state.goodness
            transitions.append(Transition(
                verb=verb,
                from_state=from_state,
                to_state=to_state,
                delta_g=delta_g
            ))

        return Trajectory(start=start_state, transitions=transitions, goal=goal)

    def _find_connecting_verb(self, from_word: str, to_word: str) -> str:
        """Find a verb that connects two words, or return a generic one."""
        # Check optimizer's neighbor graph (4-tuple: neighbor, verb, delta_g, is_spin)
        neighbors = self.optimizer.get_neighbors(from_word)
        for neighbor, verb, _, is_spin in neighbors:
            if neighbor == to_word:
                # For spin transitions, use descriptive verbs
                if is_spin:
                    return f"transform into"
                return verb

        # Generic verbs based on delta_g direction
        from_state = self.hybrid.core.get_state(from_word)
        to_state = self.hybrid.core.get_state(to_word)

        if from_state and to_state:
            if to_state.goodness > from_state.goodness:
                return random.choice(["embrace", "find", "discover", "reach"])
            else:
                return random.choice(["abandon", "leave", "lose", "forget"])

        return "become"

    def run_turn(self, concept: str, intent: str = "good", steps: int = 3):
        """Run a single conversation turn."""
        print(f"\n{'─' * 60}")
        print(f"TURN {len(self.conversation_history) + 1}")
        print(f"{'─' * 60}")

        # Speaker 1
        s1 = self.speaker1_prompt(concept, intent)
        if "error" in s1:
            print(f"\n[ERROR] {s1['error']}")
            return None

        print(f"\n[CLAUDE]: {s1['message']}")
        print(f"         Intent: navigate toward '{intent}'")

        # Speaker 2
        s2 = self.speaker2_respond(concept, intent, steps)
        if "error" in s2:
            print(f"\n[ERROR] {s2['error']}")
            return None

        print(f"\n[HYBRID ({s2['algorithm']})]:")
        print(f"  Response: \"{s2['response'][:150]}{'...' if len(s2['response']) > 150 else ''}\"")
        print(f"  Path: {' -> '.join(s2['trajectory'])}")
        print(f"  Delta-g: {s2['delta_g']:+.3f} | Efficiency: {s2['efficiency']:.4f}")
        print(f"  Steps: {s2['steps']} | Fidelity: {s2['fidelity']:.2f} | Accepted: {s2['accepted']}")
        print(f"  Arrived at: '{s2['end_state']}'")

        turn = {"speaker1": s1, "speaker2": s2}
        self.conversation_history.append(turn)

        return s2['end_state']

    def run_dialogue(self, starting_concepts: list, intent: str = "good"):
        """
        Run a multi-turn dialogue.

        Each turn, Hybrid's end state can become the next topic.
        """
        print("\n" + "=" * 60)
        print("STARTING DIALOGUE")
        print(f"Initial concepts: {starting_concepts}")
        print(f"Overall intent: {intent}")
        print("=" * 60)

        current = starting_concepts[0]

        for i, concept in enumerate(starting_concepts):
            # Alternate between provided concepts and hybrid's responses
            if i == 0:
                current = concept
            else:
                # 50% chance to use hybrid's last response, 50% to use next concept
                if random.random() < 0.5 and self.conversation_history:
                    current = self.conversation_history[-1]["speaker2"]["end_state"]
                    print(f"\n[CLAUDE] (following hybrid's lead with '{current}')")
                else:
                    current = concept

            end_state = self.run_turn(current, intent, steps=3)

            if end_state is None:
                break

        self.summarize()

    def summarize(self):
        """Summarize the conversation."""
        print("\n" + "=" * 60)
        print("CONVERSATION SUMMARY")
        print("=" * 60)

        if not self.conversation_history:
            print("No conversation recorded.")
            return

        total_delta_g = sum(t["speaker2"]["delta_g"] for t in self.conversation_history)
        avg_fidelity = sum(t["speaker2"]["fidelity"] for t in self.conversation_history) / len(self.conversation_history)
        avg_efficiency = sum(t["speaker2"]["efficiency"] for t in self.conversation_history) / len(self.conversation_history)
        total_steps = sum(t["speaker2"]["steps"] for t in self.conversation_history)
        total_retries = sum(t["speaker2"].get("retries", 1) for t in self.conversation_history)
        accepted_count = sum(1 for t in self.conversation_history if t["speaker2"]["accepted"])

        print(f"\nAlgorithm: {self.algorithm}")
        print(f"Turns: {len(self.conversation_history)}")
        print(f"Total semantic movement (delta-g): {total_delta_g:+.3f}")
        print(f"Average fidelity: {avg_fidelity:.3f}")
        print(f"Average efficiency: {avg_efficiency:.4f}")
        print(f"Total optimization steps: {total_steps}")
        print(f"Total LLM retries: {total_retries}")
        print(f"Accepted responses: {accepted_count}/{len(self.conversation_history)}")

        print("\nConversation flow:")
        for i, turn in enumerate(self.conversation_history):
            s1 = turn["speaker1"]
            s2 = turn["speaker2"]
            print(f"  {i+1}. {s1['concept']} -> {s2['end_state']} "
                  f"(Dg={s2['delta_g']:+.3f}, eff={s2['efficiency']:.4f}, steps={s2['steps']})")

        # Semantic journey
        print("\nSemantic journey:")
        start_word = self.conversation_history[0]["speaker1"]["concept"]
        end_word = self.conversation_history[-1]["speaker2"]["end_state"]

        start_state = self.hybrid.core.get_state(start_word)
        end_state = self.hybrid.core.get_state(end_word)

        if start_state and end_state:
            print(f"  Started: {start_word} (g={start_state.goodness:+.3f})")
            print(f"  Ended:   {end_word} (g={end_state.goodness:+.3f})")
            print(f"  Net change: {end_state.goodness - start_state.goodness:+.3f}")

        # Store dialogue summary
        self.all_dialogues.append({
            "turns": len(self.conversation_history),
            "algorithm": self.algorithm,
            "total_delta_g": total_delta_g,
            "avg_fidelity": avg_fidelity,
            "avg_efficiency": avg_efficiency,
            "total_steps": total_steps,
            "accepted_count": accepted_count,
            "start_word": start_word,
            "end_word": end_word,
            "net_change": end_state.goodness - start_state.goodness if start_state and end_state else 0,
            "history": self.conversation_history.copy()
        })

    def save_results(self, filename: str = None):
        """Save all dialogue results to JSON file."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{self.model.replace(':', '_')}_{self.algorithm}_{timestamp}.json"

        output_file = results_dir / filename

        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "algorithm": self.algorithm,
            "renderer": "ollama",
            "dialogues": self.all_dialogues,
            "summary": {
                "total_dialogues": len(self.all_dialogues),
                "total_delta_g": sum(d["total_delta_g"] for d in self.all_dialogues),
                "avg_fidelity": sum(d["avg_fidelity"] for d in self.all_dialogues) / len(self.all_dialogues) if self.all_dialogues else 0,
            }
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"RESULTS SAVED: {output_file}")
        print(f"{'='*60}")

        return output_file


def main():
    """Run conversation simulation with different algorithms."""
    import argparse

    parser = argparse.ArgumentParser(description="Conversation simulation with optimization")
    parser.add_argument("--algorithm", "-a",
                        choices=["hill_climbing", "simulated_annealing", "random_local", "quantum", "compare"],
                        default="hill_climbing",
                        help="Optimization algorithm to use (quantum = thermal + tunneling)")
    parser.add_argument("--model", "-m",
                        default=None,
                        help="Ollama model to use (e.g. 'mistral:7b', 'qwen2.5:1.5b')")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Run quick test with fewer dialogues")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print(f"  CONVERSATION SIMULATION: {args.algorithm.upper()}")
    if args.model:
        print(f"  MODEL: {args.model}")
    print("█" * 60)

    sim = ConversationSimulator(algorithm=args.algorithm, model=args.model)

    # Dialogue 1: Journey from darkness to light
    print("\n" + "#" * 60)
    print("DIALOGUE 1: From Darkness to Light")
    print(f"Algorithm: {args.algorithm}")
    print("#" * 60)
    sim.run_dialogue(
        starting_concepts=["darkness", "fear", "hope"],
        intent="good"
    )

    if args.quick:
        print("\n[Quick mode - stopping after 1 dialogue]")
        sim.save_results()
        return

    # Reset for new dialogue
    sim.conversation_history = []

    # Dialogue 2: Corruption journey
    print("\n" + "#" * 60)
    print("DIALOGUE 2: The Corruption")
    print(f"Algorithm: {args.algorithm}")
    print("#" * 60)
    sim.run_dialogue(
        starting_concepts=["trust", "power", "silence"],
        intent="evil"
    )

    # Reset for new dialogue
    sim.conversation_history = []

    # Dialogue 3: Philosophical exploration
    print("\n" + "#" * 60)
    print("DIALOGUE 3: Philosophical Exploration")
    print(f"Algorithm: {args.algorithm}")
    print("#" * 60)
    sim.run_dialogue(
        starting_concepts=["truth", "wisdom", "freedom"],
        intent="good"
    )

    # Save all results
    sim.save_results()


def compare_algorithms():
    """Compare all algorithms on the same dialogue."""
    algorithms = ["hill_climbing", "simulated_annealing", "random_local", "quantum"]
    results = {}

    concepts = ["darkness", "hope"]
    intent = "good"

    for algo in algorithms:
        print("\n" + "█" * 60)
        print(f"  TESTING: {algo.upper()}")
        print("█" * 60)

        sim = ConversationSimulator(algorithm=algo)
        sim.run_dialogue(starting_concepts=concepts, intent=intent)

        # Collect metrics
        if sim.conversation_history:
            results[algo] = {
                "total_delta_g": sum(t["speaker2"]["delta_g"] for t in sim.conversation_history),
                "avg_fidelity": sum(t["speaker2"]["fidelity"] for t in sim.conversation_history) / len(sim.conversation_history),
                "avg_efficiency": sum(t["speaker2"]["efficiency"] for t in sim.conversation_history) / len(sim.conversation_history),
                "total_steps": sum(t["speaker2"]["steps"] for t in sim.conversation_history),
                "accepted": sum(1 for t in sim.conversation_history if t["speaker2"]["accepted"])
            }

    # Summary comparison
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<25} {'Delta-g':<10} {'Fidelity':<10} {'Efficiency':<12} {'Steps':<8} {'Accepted':<8}")
    print("-" * 70)

    for algo, metrics in results.items():
        print(f"{algo:<25} {metrics['total_delta_g']:+.3f}     "
              f"{metrics['avg_fidelity']:.3f}      {metrics['avg_efficiency']:.4f}       "
              f"{metrics['total_steps']:<8} {metrics['accepted']}")


if __name__ == "__main__":
    import sys
    if "--compare-all" in sys.argv:
        compare_algorithms()
    else:
        main()
