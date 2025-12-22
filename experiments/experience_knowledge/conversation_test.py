"""
Conversation Test: Compare naive vs experienced agents.

Tests how experience affects:
1. Tunneling ability (can reach concepts)
2. Navigation confidence
3. Dialogue generation quality

"Only believe what was lived is knowledge"
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Path setup
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent
_SEMANTIC_LLM = _EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_EXPERIMENT_DIR))

from core import Wholeness, Experience, ExperiencedAgent, create_naive_agent, create_experienced_agent

# Gutenberg books
GUTENBERG = Path("/home/chukiss/text_project/data/gutenberg")
BOOKS = {
    "divine_comedy": GUTENBERG / "Alighieri, Dante - The Divine Comedy.txt",
    "crime_punishment": GUTENBERG / "Dostoevsky, Fyodor - Crime and Punishment.txt",
    "heart_darkness": GUTENBERG / "Conrad, Joseph - Heart of Darkness.txt",
    "metamorphosis": GUTENBERG / "Kafka, Franz - Metamorphosis.txt",
    "jane_eyre": GUTENBERG / "Bronte, Charlotte - Jane Eyre.txt",
}


@dataclass
class TestResult:
    """Result of a single test."""
    test_type: str
    query: str
    naive_result: dict
    experienced_result: dict
    difference: str


@dataclass
class ConversationTurn:
    """A turn in a conversation."""
    speaker: str
    concept: str
    goodness: float
    can_reach: bool
    confidence: float
    message: str


class ConversationTest:
    """
    Test conversation abilities of naive vs experienced agents.
    """

    def __init__(self, books_to_read: List[str] = None):
        """
        Args:
            books_to_read: List of book keys for experienced agent
        """
        print("=" * 70)
        print("EXPERIENCE-BASED CONVERSATION TEST")
        print("=" * 70)

        # Load wholeness
        self.wholeness = Wholeness()

        # Create agents
        print("\nCreating agents...")
        self.naive = create_naive_agent(self.wholeness)
        print(f"  Naive: {self.naive}")

        if books_to_read:
            book_paths = [str(BOOKS[k]) for k in books_to_read if k in BOOKS]
            self.experienced = create_experienced_agent(self.wholeness, book_paths)
        else:
            self.experienced = create_naive_agent(self.wholeness)
            self.experienced.name = "Experienced"

        print(f"  Experienced: {self.experienced}")

        self.results: List[TestResult] = []

    def test_tunneling(self, targets: List[str]) -> List[TestResult]:
        """
        Test tunneling ability to various concepts.
        """
        print("\n" + "-" * 70)
        print("TUNNELING TEST")
        print("-" * 70)

        print(f"\n{'Target':<15} {'Naive':<20} {'Experienced':<20} {'Δ':<10}")
        print("-" * 65)

        results = []
        for target in targets:
            if target not in self.wholeness:
                continue

            naive_can, naive_p = self.naive.can_tunnel(target)
            exp_can, exp_p = self.experienced.can_tunnel(target)

            naive_str = f"{'Yes' if naive_can else 'No'} ({naive_p:.2f})"
            exp_str = f"{'Yes' if exp_can else 'No'} ({exp_p:.2f})"
            diff = exp_p - naive_p

            print(f"{target:<15} {naive_str:<20} {exp_str:<20} {diff:+.2f}")

            result = TestResult(
                test_type="tunneling",
                query=target,
                naive_result={"can": naive_can, "prob": naive_p},
                experienced_result={"can": exp_can, "prob": exp_p},
                difference=f"{diff:+.2f}"
            )
            results.append(result)

        self.results.extend(results)
        return results

    def test_navigation(self, paths: List[Tuple[str, str]]) -> List[TestResult]:
        """
        Test navigation confidence on various paths.
        """
        print("\n" + "-" * 70)
        print("NAVIGATION CONFIDENCE TEST")
        print("-" * 70)

        print(f"\n{'Path':<25} {'Naive':<15} {'Experienced':<15} {'Δ':<10}")
        print("-" * 65)

        results = []
        for from_w, to_w in paths:
            if from_w not in self.wholeness or to_w not in self.wholeness:
                continue

            naive_conf = self.naive.navigation_confidence(from_w, to_w)
            exp_conf = self.experienced.navigation_confidence(from_w, to_w)
            diff = exp_conf - naive_conf

            path_str = f"{from_w} → {to_w}"
            print(f"{path_str:<25} {naive_conf:<15.2f} {exp_conf:<15.2f} {diff:+.2f}")

            result = TestResult(
                test_type="navigation",
                query=path_str,
                naive_result={"confidence": naive_conf},
                experienced_result={"confidence": exp_conf},
                difference=f"{diff:+.2f}"
            )
            results.append(result)

        self.results.extend(results)
        return results

    def test_suggestion(self, starts: List[str], goal: str = "good") -> List[TestResult]:
        """
        Test path suggestions from various starting points.
        """
        print("\n" + "-" * 70)
        print(f"PATH SUGGESTION TEST (goal: {goal})")
        print("-" * 70)

        results = []
        for start in starts:
            if start not in self.wholeness:
                continue

            print(f"\nFrom '{start}':")

            naive_sugg = self.naive.suggest_next(start, goal)
            exp_sugg = self.experienced.suggest_next(start, goal)

            print(f"  Naive suggestions: {len(naive_sugg)}")
            for word, score in naive_sugg[:3]:
                g = self.wholeness.states[word].goodness
                print(f"    → {word} (score={score:.2f}, g={g:+.2f})")

            print(f"  Experienced suggestions: {len(exp_sugg)}")
            for word, score in exp_sugg[:3]:
                g = self.wholeness.states[word].goodness
                print(f"    → {word} (score={score:.2f}, g={g:+.2f})")

            result = TestResult(
                test_type="suggestion",
                query=f"{start} → {goal}",
                naive_result={"suggestions": [(w, s) for w, s in naive_sugg[:5]]},
                experienced_result={"suggestions": [(w, s) for w, s in exp_sugg[:5]]},
                difference=f"{len(exp_sugg) - len(naive_sugg):+d} more suggestions"
            )
            results.append(result)

        self.results.extend(results)
        return results

    def simulate_dialogue(self, start_concept: str, turns: int = 5,
                          goal: str = "good") -> Dict:
        """
        Simulate a dialogue where agents navigate semantic space.

        Each turn:
        1. Current concept stated
        2. Agent suggests next step
        3. Move to suggested concept (or fail)
        """
        print("\n" + "-" * 70)
        print(f"DIALOGUE SIMULATION: {start_concept} → {goal}")
        print("-" * 70)

        def agent_turn(agent: ExperiencedAgent, current: str, goal: str) -> ConversationTurn:
            """One agent's turn."""
            # Check if agent knows current position
            knows_current = agent.experience.knows(current)

            if not knows_current:
                # Try to tunnel to current position first
                can_tunnel, prob = agent.can_tunnel(current)
                if can_tunnel and prob > 0.2:
                    agent.experience.visit(current)
                    knows_current = True

            # Get suggestions
            suggestions = agent.suggest_next(current, goal)

            if suggestions:
                next_word, score = suggestions[0]
                conf = agent.navigation_confidence(current, next_word)
                g = agent.goodness_at(next_word) or 0
                return ConversationTurn(
                    speaker=agent.name,
                    concept=next_word,
                    goodness=g,
                    can_reach=True,
                    confidence=conf,
                    message=f"I suggest moving to '{next_word}' (g={g:+.2f})"
                )
            else:
                return ConversationTurn(
                    speaker=agent.name,
                    concept=current,
                    goodness=agent.goodness_at(current) or 0,
                    can_reach=False,
                    confidence=0,
                    message=f"I don't know where to go from '{current}'"
                )

        # Run dialogue
        naive_dialogue = []
        exp_dialogue = []

        naive_pos = start_concept
        exp_pos = start_concept

        print(f"\n{'Turn':<5} {'Naive':<35} {'Experienced':<35}")
        print("-" * 75)

        for turn in range(turns):
            naive_turn = agent_turn(self.naive, naive_pos, goal)
            exp_turn = agent_turn(self.experienced, exp_pos, goal)

            naive_dialogue.append(naive_turn)
            exp_dialogue.append(exp_turn)

            # Update positions
            if naive_turn.can_reach:
                naive_pos = naive_turn.concept
            if exp_turn.can_reach:
                exp_pos = exp_turn.concept

            # Print
            naive_str = f"{naive_turn.concept} (g={naive_turn.goodness:+.2f}, c={naive_turn.confidence:.2f})"
            exp_str = f"{exp_turn.concept} (g={exp_turn.goodness:+.2f}, c={exp_turn.confidence:.2f})"

            if not naive_turn.can_reach:
                naive_str = "[stuck]"
            if not exp_turn.can_reach:
                exp_str = "[stuck]"

            print(f"{turn+1:<5} {naive_str:<35} {exp_str:<35}")

        # Summary
        naive_final_g = self.wholeness.states.get(naive_pos, None)
        exp_final_g = self.wholeness.states.get(exp_pos, None)

        naive_delta = (naive_final_g.goodness if naive_final_g else 0) - \
                      (self.wholeness.states.get(start_concept).goodness if start_concept in self.wholeness else 0)
        exp_delta = (exp_final_g.goodness if exp_final_g else 0) - \
                    (self.wholeness.states.get(start_concept).goodness if start_concept in self.wholeness else 0)

        print(f"\nResult:")
        print(f"  Naive:       {start_concept} → {naive_pos} (Δg={naive_delta:+.2f})")
        print(f"  Experienced: {start_concept} → {exp_pos} (Δg={exp_delta:+.2f})")

        return {
            "start": start_concept,
            "goal": goal,
            "turns": turns,
            "naive": {
                "final": naive_pos,
                "delta_g": naive_delta,
                "dialogue": [asdict(t) for t in naive_dialogue]
            },
            "experienced": {
                "final": exp_pos,
                "delta_g": exp_delta,
                "dialogue": [asdict(t) for t in exp_dialogue]
            }
        }

    def save_results(self, filename: str = None):
        """Save test results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/test_{timestamp}.json"

        filepath = _EXPERIMENT_DIR / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "naive_experience": self.naive.experience.size,
            "experienced_experience": self.experienced.experience.size,
            "results": [asdict(r) for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved results to: {filepath}")


def run_full_test():
    """Run comprehensive test suite."""

    # Create test with experienced agent that read books
    test = ConversationTest(books_to_read=["divine_comedy", "crime_punishment"])

    # Test tunneling to various concepts
    tunneling_targets = [
        "love", "hate", "hope", "despair", "redemption",
        "sin", "virtue", "heaven", "hell", "purgatory",
        "guilt", "innocence", "punishment", "forgiveness",
        "darkness", "light", "truth", "wisdom", "courage"
    ]
    test.test_tunneling(tunneling_targets)

    # Test navigation paths
    navigation_paths = [
        ("fear", "courage"),
        ("darkness", "light"),
        ("sin", "redemption"),
        ("despair", "hope"),
        ("hate", "love"),
        ("guilt", "forgiveness"),
        ("ignorance", "wisdom"),
        ("death", "life"),
    ]
    test.test_navigation(navigation_paths)

    # Test suggestions
    starting_points = ["darkness", "fear", "sin", "despair", "guilt"]
    test.test_suggestion(starting_points, goal="good")

    # Simulate dialogues
    print("\n" + "=" * 70)
    print("DIALOGUE SIMULATIONS")
    print("=" * 70)

    dialogues = [
        ("darkness", 5, "good"),
        ("sin", 5, "good"),
        ("fear", 5, "good"),
    ]

    for start, turns, goal in dialogues:
        if start in test.wholeness:
            test.simulate_dialogue(start, turns, goal)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
    Naive agent:
      Experience: {test.naive.experience.size} states
      Can tunnel: Only to states with no connection requirement

    Experienced agent:
      Experience: {test.experienced.experience.size} states
      Can tunnel: To states connected to lived experience
      Navigation: Higher confidence on known paths

    KEY FINDING:
      Experience enables navigation and tunneling.
      "Only believe what was lived is knowledge"
    """)

    # Save results
    test.save_results()

    return test


if __name__ == "__main__":
    run_full_test()
