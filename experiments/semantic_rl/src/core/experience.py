"""
Experience: Personal subgraph of semantic space.

Wholeness = complete semantic space (all possible)
Experience = what I've lived (my knowledge)

"Only believe what was lived is knowledge"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict


@dataclass
class Experience:
    """
    Personal subgraph of semantic space.

    This IS knowledge - the territory I've walked.
    """

    # States I've visited: word -> visit count
    visited: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Transitions I've walked: (from, to) -> count
    walked: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    # Total steps taken
    total_steps: int = 0

    def visit(self, word: str):
        """Visit a state."""
        self.visited[word] += 1
        self.total_steps += 1

    def walk(self, from_word: str, to_word: str):
        """Walk a transition."""
        self.visited[from_word] += 1
        self.visited[to_word] += 1
        self.walked[(from_word, to_word)] += 1
        self.total_steps += 1

    def knows(self, word: str) -> bool:
        """Do I know this state? (Have I been there?)"""
        return word in self.visited

    def familiarity(self, word: str) -> float:
        """How familiar am I with this state? [0, 1]"""
        if self.total_steps == 0:
            return 0.0
        return min(1.0, self.visited.get(word, 0) / 10)  # Cap at 10 visits

    def walked_transition(self, from_word: str, to_word: str) -> bool:
        """Have I walked this specific transition?"""
        return (from_word, to_word) in self.walked

    def transitions_from(self, word: str) -> List[str]:
        """Where have I gone from this state?"""
        return [to for (frm, to) in self.walked.keys() if frm == word]

    def transitions_to(self, word: str) -> List[str]:
        """Where have I come from to reach this state?"""
        return [frm for (frm, to) in self.walked.keys() if to == word]

    @property
    def known_states(self) -> Set[str]:
        """All states I know."""
        return set(self.visited.keys())

    @property
    def size(self) -> int:
        """Size of my experience (unique states)."""
        return len(self.visited)

    def merge(self, other: 'Experience'):
        """Merge another experience into this one."""
        for word, count in other.visited.items():
            self.visited[word] += count
        for trans, count in other.walked.items():
            self.walked[trans] += count
        self.total_steps += other.total_steps

    def __repr__(self):
        return f"Experience({self.size} states, {len(self.walked)} transitions, {self.total_steps} steps)"


class ExperiencedAgent:
    """
    An agent with experience navigating semantic space.

    Experience determines:
    - What I can tunnel to (must be connected to experience)
    - How confident I am (familiarity)
    - What paths I know
    """

    def __init__(self, wholeness: 'Wholeness', believe: float = 0.5):
        """
        Args:
            wholeness: The complete semantic space
            believe: Capacity for leaps beyond experience [0, 1]
        """
        self.wholeness = wholeness
        self.experience = Experience()
        self.believe = believe

    def read_book(self, path: List[str]):
        """
        Read a book by walking its path.

        This is how we gain experience.
        """
        prev = None
        for word in path:
            if word in self.wholeness.states:
                self.experience.visit(word)
                if prev is not None:
                    self.experience.walk(prev, word)
                prev = word

    def can_tunnel_to(self, target: str) -> Tuple[bool, float]:
        """
        Can I tunnel to this target?

        Returns: (can_tunnel, probability)

        Tunneling requires:
        1. Target exists in wholeness
        2. Target is adjacent to my experience (or I've been there)
        3. Probability based on believe and familiarity
        """
        if target not in self.wholeness.states:
            return False, 0.0

        # If I've been there, high probability
        if self.experience.knows(target):
            p = 0.5 + 0.5 * self.experience.familiarity(target)
            return True, p * self.believe

        # If adjacent to my experience, lower probability
        neighbors = self.wholeness.neighbors(target)
        known_neighbors = [n for n in neighbors if self.experience.knows(n)]

        if known_neighbors:
            # Can tunnel - probability based on how many neighbors I know
            adjacency = len(known_neighbors) / max(1, len(neighbors))
            p = adjacency * self.believe * 0.5
            return True, p

        # Not connected to my experience - cannot tunnel
        return False, 0.0

    def navigate_confidence(self, from_word: str, to_word: str) -> float:
        """
        How confident am I in navigating from A to B?

        Based on:
        - Do I know both states?
        - Have I walked this transition?
        """
        if not self.experience.knows(from_word):
            return 0.0

        if self.experience.walked_transition(from_word, to_word):
            # I've done this before
            return 0.8 + 0.2 * self.experience.familiarity(to_word)

        if self.experience.knows(to_word):
            # I know both ends but haven't walked this path
            return 0.3 + 0.3 * self.experience.familiarity(to_word)

        # I don't know the destination
        return 0.1 * self.believe


class Wholeness:
    """
    The complete semantic space - all that can be known.

    This is τ₀ (Logos) - the wholeness from which experience is carved.
    """

    def __init__(self):
        self.states: Dict[str, dict] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self._load()

    def _load(self):
        """Load the complete semantic space."""
        import sys
        from pathlib import Path

        _THIS_FILE = Path(__file__).resolve()
        _SEMANTIC_LLM_PATH = _THIS_FILE.parent.parent.parent.parent.parent

        try:
            import importlib.util

            data_loader_path = _SEMANTIC_LLM_PATH / "core" / "data_loader.py"
            hybrid_llm_path = _SEMANTIC_LLM_PATH / "core" / "hybrid_llm.py"

            spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
            data_loader_module = importlib.util.module_from_spec(spec)
            sys.modules['core.data_loader'] = data_loader_module
            spec.loader.exec_module(data_loader_module)

            spec = importlib.util.spec_from_file_location("hybrid_llm", hybrid_llm_path)
            hybrid_llm_module = importlib.util.module_from_spec(spec)
            sys.modules['core.hybrid_llm'] = hybrid_llm_module
            spec.loader.exec_module(hybrid_llm_module)

            core = hybrid_llm_module.QuantumCore()

            # Load all states
            for word, state in core.states.items():
                self.states[word] = {
                    'g': state.goodness,
                    'tau': state.tau,
                    'j': state.j
                }

            # Load edges from verb transitions
            for verb, objects in core.verb_objects.items():
                for obj in objects:
                    if obj in self.states:
                        # Connect to semantically similar words
                        for other in list(self.states.keys())[:1000]:
                            if other != obj:
                                self.edges[obj].add(other)
                                self.edges[other].add(obj)

            # Also connect by semantic similarity (j-vector distance)
            words = list(self.states.keys())
            for i, w1 in enumerate(words[:500]):
                j1 = self.states[w1]['j']
                for w2 in words[i+1:i+50]:
                    j2 = self.states[w2]['j']
                    dist = np.linalg.norm(j1 - j2)
                    if dist < 0.5:
                        self.edges[w1].add(w2)
                        self.edges[w2].add(w1)

        except Exception as e:
            print(f"Error loading wholeness: {e}")

    def neighbors(self, word: str) -> List[str]:
        """Get neighbors of a state in the semantic graph."""
        return list(self.edges.get(word, set()))

    def __contains__(self, word: str) -> bool:
        return word in self.states

    def __len__(self) -> int:
        return len(self.states)


def demonstrate_experience():
    """
    Demonstrate the difference between no experience and experience.
    """
    print("=" * 70)
    print("EXPERIENCE DEMONSTRATION")
    print("=" * 70)

    # Load wholeness
    print("\nLoading Wholeness (complete semantic space)...")
    wholeness = Wholeness()
    print(f"  Wholeness: {len(wholeness)} states")

    # Create two agents: one naive, one experienced
    naive_agent = ExperiencedAgent(wholeness, believe=0.5)
    experienced_agent = ExperiencedAgent(wholeness, believe=0.5)

    print(f"\nNaive agent: {naive_agent.experience}")
    print(f"Experienced agent: {experienced_agent.experience}")

    # Give experienced agent some experience (simulated book path)
    print("\n" + "-" * 70)
    print("Experienced agent reads a path through semantic space...")
    print("-" * 70)

    # Simulate reading a redemption arc
    redemption_path = [
        "darkness", "fear", "despair", "struggle", "doubt",
        "hope", "courage", "strength", "wisdom", "light",
        "love", "peace", "joy", "truth", "freedom"
    ]

    # Filter to words that exist
    valid_path = [w for w in redemption_path if w in wholeness.states]
    print(f"Walking path: {' → '.join(valid_path)}")

    experienced_agent.read_book(valid_path)
    print(f"\nAfter reading:")
    print(f"  Experienced agent: {experienced_agent.experience}")
    print(f"  Known states: {experienced_agent.experience.known_states}")

    # Test tunneling capabilities
    print("\n" + "-" * 70)
    print("TUNNELING TEST")
    print("-" * 70)

    test_targets = ["hope", "love", "wisdom", "anger", "betrayal", "redemption", "chaos"]

    print(f"\n{'Target':<15} {'Naive':<25} {'Experienced':<25}")
    print("-" * 65)

    for target in test_targets:
        if target not in wholeness.states:
            continue

        naive_can, naive_p = naive_agent.can_tunnel_to(target)
        exp_can, exp_p = experienced_agent.can_tunnel_to(target)

        naive_str = f"{'Yes' if naive_can else 'No'} (p={naive_p:.2f})" if naive_can or naive_p > 0 else "No (no connection)"
        exp_str = f"{'Yes' if exp_can else 'No'} (p={exp_p:.2f})" if exp_can or exp_p > 0 else "No (no connection)"

        print(f"{target:<15} {naive_str:<25} {exp_str:<25}")

    # Test navigation confidence
    print("\n" + "-" * 70)
    print("NAVIGATION CONFIDENCE TEST")
    print("-" * 70)

    nav_tests = [
        ("fear", "hope"),
        ("darkness", "light"),
        ("hope", "love"),
        ("anger", "peace"),
    ]

    print(f"\n{'From → To':<20} {'Naive':<15} {'Experienced':<15}")
    print("-" * 50)

    for from_w, to_w in nav_tests:
        if from_w not in wholeness.states or to_w not in wholeness.states:
            continue

        naive_conf = naive_agent.navigate_confidence(from_w, to_w)
        exp_conf = experienced_agent.navigate_confidence(from_w, to_w)

        print(f"{from_w} → {to_w:<10} {naive_conf:<15.2f} {exp_conf:<15.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
    Naive agent:
      - Cannot tunnel (no experience to connect to)
      - Zero navigation confidence (hasn't been anywhere)

    Experienced agent:
      - Can tunnel to states connected to experience
      - Higher confidence navigating known territory
      - Experience IS knowledge - lived paths become navigable
    """)

    return naive_agent, experienced_agent


if __name__ == "__main__":
    demonstrate_experience()
