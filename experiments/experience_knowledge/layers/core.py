"""
Experience-Based Knowledge System

Core principle: "Only believe what was lived is knowledge"

Wholeness = complete semantic space (all possible)
Experience = personal subgraph (what I've lived)
Knowledge = ability to navigate my experience
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import json

# Path setup
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM = _THIS_FILE.parent.parent.parent


@dataclass
class SemanticState:
    """A state in semantic space."""
    word: str
    goodness: float
    tau: float
    j: np.ndarray

    def to_dict(self) -> dict:
        return {
            'word': self.word,
            'g': self.goodness,
            'tau': self.tau,
            'j': self.j.tolist()
        }


class Wholeness:
    """
    The complete semantic space - τ₀ (Logos).

    All that can be known. The universe of concepts.
    """

    def __init__(self):
        self.states: Dict[str, SemanticState] = {}
        self._neighbors: Dict[str, Set[str]] = defaultdict(set)
        self._load()

    def _load(self):
        """Load from QuantumCore."""
        try:
            import importlib.util

            data_loader_path = _SEMANTIC_LLM / "core" / "data_loader.py"
            hybrid_llm_path = _SEMANTIC_LLM / "core" / "hybrid_llm.py"

            spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
            data_loader_module = importlib.util.module_from_spec(spec)
            sys.modules['core.data_loader'] = data_loader_module
            spec.loader.exec_module(data_loader_module)

            spec = importlib.util.spec_from_file_location("hybrid_llm", hybrid_llm_path)
            hybrid_llm_module = importlib.util.module_from_spec(spec)
            sys.modules['core.hybrid_llm'] = hybrid_llm_module
            spec.loader.exec_module(hybrid_llm_module)

            print("Loading Wholeness...")
            core = hybrid_llm_module.QuantumCore()

            for word, state in core.states.items():
                self.states[word] = SemanticState(
                    word=word,
                    goodness=state.goodness,
                    tau=state.tau,
                    j=state.j
                )

            # Build neighbor graph from verb transitions
            for verb, objects in core.verb_objects.items():
                for obj in objects:
                    if obj in self.states:
                        for other in objects:
                            if other != obj and other in self.states:
                                self._neighbors[obj].add(other)
                                self._neighbors[other].add(obj)

            print(f"  {len(self.states)} states loaded")

        except Exception as e:
            print(f"Error loading Wholeness: {e}")
            import traceback
            traceback.print_exc()

    def get(self, word: str) -> Optional[SemanticState]:
        return self.states.get(word)

    def neighbors(self, word: str) -> Set[str]:
        return self._neighbors.get(word, set())

    def semantic_neighbors(self, word: str, threshold: float = 0.3) -> List[str]:
        """Get semantically similar words (by j-vector distance)."""
        if word not in self.states:
            return []

        j = self.states[word].j
        similar = []

        for other, state in self.states.items():
            if other != word:
                dist = np.linalg.norm(j - state.j)
                if dist < threshold:
                    similar.append(other)

        return similar[:50]  # Limit

    def __contains__(self, word: str) -> bool:
        return word in self.states

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class Experience:
    """
    Personal subgraph - what I've lived.

    This IS knowledge.
    """

    visited: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    transitions: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    total_steps: int = 0

    def visit(self, word: str):
        """Visit a state."""
        self.visited[word] += 1
        self.total_steps += 1

    def transition(self, from_word: str, to_word: str):
        """Walk a transition."""
        self.transitions[(from_word, to_word)] += 1

    def knows(self, word: str) -> bool:
        """Have I been to this state?"""
        return word in self.visited

    def familiarity(self, word: str) -> float:
        """How familiar? [0, 1]"""
        visits = self.visited.get(word, 0)
        return min(1.0, visits / 10.0)

    def has_walked(self, from_word: str, to_word: str) -> bool:
        """Have I walked this transition?"""
        return (from_word, to_word) in self.transitions

    @property
    def known_words(self) -> Set[str]:
        return set(self.visited.keys())

    @property
    def size(self) -> int:
        return len(self.visited)

    def save(self, path: str):
        """Save experience to file."""
        data = {
            'visited': dict(self.visited),
            'transitions': {f"{k[0]}|{k[1]}": v for k, v in self.transitions.items()},
            'total_steps': self.total_steps
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Experience':
        """Load experience from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        exp = cls()
        exp.visited = defaultdict(int, data['visited'])
        exp.transitions = defaultdict(int, {
            tuple(k.split('|')): v for k, v in data['transitions'].items()
        })
        exp.total_steps = data['total_steps']
        return exp

    def __repr__(self):
        return f"Experience({self.size} states, {len(self.transitions)} transitions)"


class ExperiencedAgent:
    """
    Agent with experience in semantic space.

    Experience determines what it can navigate and tunnel to.
    """

    def __init__(self, wholeness: Wholeness, believe: float = 0.5, name: str = "Agent"):
        self.wholeness = wholeness
        self.experience = Experience()
        self.believe = believe
        self.name = name

    def read(self, words: List[str]):
        """
        Read a sequence of words (like reading a book).

        This builds experience.
        """
        prev = None
        for word in words:
            if word in self.wholeness:
                self.experience.visit(word)
                if prev is not None:
                    self.experience.transition(prev, word)
                prev = word

    def can_tunnel(self, target: str) -> Tuple[bool, float]:
        """
        Can I tunnel to this target?

        Returns (can_tunnel, probability)

        Rules:
        1. If I've been there -> high probability
        2. If adjacent to my experience -> lower probability
        3. If not connected -> cannot tunnel
        """
        if target not in self.wholeness:
            return False, 0.0

        # Been there before
        if self.experience.knows(target):
            p = 0.5 + 0.5 * self.experience.familiarity(target)
            return True, p * self.believe

        # Adjacent to experience?
        my_words = self.experience.known_words
        target_neighbors = self.wholeness.neighbors(target)

        # Also check semantic neighbors
        if not target_neighbors:
            target_neighbors = set(self.wholeness.semantic_neighbors(target, 0.4))

        known_neighbors = my_words & target_neighbors

        if known_neighbors:
            adjacency = len(known_neighbors) / max(1, len(target_neighbors))
            p = adjacency * self.believe * 0.3
            return True, p

        # Not connected to experience
        return False, 0.0

    def navigation_confidence(self, from_word: str, to_word: str) -> float:
        """
        How confident navigating from A to B?
        """
        if not self.experience.knows(from_word):
            return 0.0

        if self.experience.has_walked(from_word, to_word):
            return 0.9

        if self.experience.knows(to_word):
            return 0.4 + 0.3 * self.experience.familiarity(to_word)

        return 0.1 * self.believe

    def goodness_at(self, word: str) -> Optional[float]:
        """Get goodness of a known word."""
        if word in self.wholeness:
            return self.wholeness.states[word].goodness
        return None

    def suggest_next(self, current: str, goal: str = "good") -> List[Tuple[str, float]]:
        """
        Suggest next steps from current position toward goal.

        Returns [(word, score), ...] sorted by score.
        """
        if not self.experience.knows(current):
            return []

        current_g = self.wholeness.states[current].goodness
        suggestions = []

        # Consider known transitions from current
        for (frm, to), count in self.experience.transitions.items():
            if frm == current and to in self.wholeness:
                to_g = self.wholeness.states[to].goodness

                if goal == "good":
                    score = to_g - current_g  # Positive = toward good
                else:
                    score = current_g - to_g  # Positive = toward evil

                score += count * 0.1  # Bonus for familiar transitions
                suggestions.append((to, score))

        # Sort by score descending
        suggestions.sort(key=lambda x: -x[1])
        return suggestions[:10]

    def save(self, path: str):
        """Save agent state."""
        self.experience.save(path)

    def load(self, path: str):
        """Load agent state."""
        self.experience = Experience.load(path)

    def __repr__(self):
        return f"{self.name}(believe={self.believe}, {self.experience})"


def create_naive_agent(wholeness: Wholeness) -> ExperiencedAgent:
    """Create agent with no experience."""
    return ExperiencedAgent(wholeness, believe=0.5, name="Naive")


def create_experienced_agent(wholeness: Wholeness, books: List[str]) -> ExperiencedAgent:
    """Create agent that has read books."""
    import re

    agent = ExperiencedAgent(wholeness, believe=0.5, name="Experienced")

    for book_path in books:
        print(f"  Reading: {Path(book_path).stem}...")
        with open(book_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Skip header/footer
        text = text[len(text)//20 : -len(text)//20]

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        agent.read(words)

    return agent
