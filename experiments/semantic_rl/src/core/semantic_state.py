"""
Semantic State: Representation of concepts in the world.

Each concept has:
- Semantic properties (τ, g, j) from the quantum-semantic model
- Physical properties derived from semantics (mass, friction, etc.)
- Connections to other states via verbs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SemanticState:
    """
    A state in the semantic world.

    Semantic Properties:
        word: The concept/word this state represents
        tau: Abstraction level (higher = more abstract)
        goodness: Moral/value dimension (-1 to +1)
        j_vector: 16D direction vector

    Physical Properties (derived):
        altitude: Height in world (from τ)
        luminance: Brightness (from g)
        mass: How "heavy" the concept feels
        friction: Resistance to change
    """

    word: str
    tau: float = 0.0
    goodness: float = 0.0
    j_vector: np.ndarray = field(default_factory=lambda: np.zeros(16))

    # Physical properties (computed from semantics)
    altitude: float = field(init=False)
    luminance: float = field(init=False)
    mass: float = field(init=False)
    friction: float = field(init=False)

    # Special modifiers
    believe_modifier: float = 0.0  # Affects agent's believe when here
    requires: List[str] = field(default_factory=list)  # Must visit these first
    unlocks: List[str] = field(default_factory=list)   # Visiting unlocks these

    def __post_init__(self):
        """Compute physical properties from semantic properties."""
        # Altitude from abstraction (τ)
        self.altitude = self.tau

        # Luminance from goodness (g)
        # Good = bright, Evil = dark
        self.luminance = (self.goodness + 1) / 2  # Normalize to 0-1

        # Mass: abstract concepts are "lighter"
        # Concrete concepts are "heavier"
        self.mass = 2.0 - self.tau  # τ ∈ [0,2], so mass ∈ [0,2]

        # Friction: concepts with strong direction resist change
        j_norm = np.linalg.norm(self.j_vector)
        self.friction = 0.5 + 0.5 * j_norm  # Higher j = more friction

    def distance_to(self, other: 'SemanticState') -> float:
        """Semantic distance to another state."""
        tau_diff = abs(self.tau - other.tau)
        j_diff = 1 - self.cosine_similarity(other)
        return np.sqrt(tau_diff**2 + j_diff**2)

    def cosine_similarity(self, other: 'SemanticState') -> float:
        """Cosine similarity of j vectors."""
        norm1 = np.linalg.norm(self.j_vector)
        norm2 = np.linalg.norm(other.j_vector)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return np.dot(self.j_vector, other.j_vector) / (norm1 * norm2)

    def tunnel_probability(self, other: 'SemanticState') -> float:
        """
        Quantum tunneling probability to another state.

        P(tunnel) = e^(-2κd)

        where:
            d = |Δτ| (abstraction barrier)
            κ = (1 - cos(j₁, j₂)) / 2 (opacity)
        """
        d = abs(self.tau - other.tau)
        cos_sim = self.cosine_similarity(other)
        kappa = (1 - cos_sim) / 2

        return np.exp(-2 * kappa * d)

    def to_observation(self) -> np.ndarray:
        """Convert to observation vector for RL."""
        return np.concatenate([
            [self.tau, self.goodness],
            self.j_vector,
            [self.altitude, self.luminance, self.mass, self.friction]
        ])

    @property
    def observation_size(self) -> int:
        """Size of observation vector."""
        return 2 + 16 + 4  # tau, g, j_vector, physical

    def __repr__(self):
        return (f"SemanticState('{self.word}', τ={self.tau:.2f}, "
                f"g={self.goodness:+.2f}, alt={self.altitude:.2f})")


@dataclass
class Transition:
    """A transition between states via a verb."""

    verb: str
    from_state: str
    to_state: str
    delta_g: float  # Change in goodness
    energy_cost: float = 0.0  # Cost to make this transition

    def __repr__(self):
        return f"{self.from_state} --{self.verb}--> {self.to_state} (Δg={self.delta_g:+.2f})"


class SemanticGraph:
    """
    Graph of semantic states connected by verb transitions.

    This is the "map" of the semantic world.
    """

    def __init__(self):
        self.states: Dict[str, SemanticState] = {}
        self.transitions: Dict[str, List[Transition]] = {}  # word -> [transitions]

    def add_state(self, state: SemanticState):
        """Add a state to the graph."""
        self.states[state.word] = state
        if state.word not in self.transitions:
            self.transitions[state.word] = []

    def add_transition(self, transition: Transition):
        """Add a transition between states."""
        if transition.from_state not in self.transitions:
            self.transitions[transition.from_state] = []
        self.transitions[transition.from_state].append(transition)

    def get_neighbors(self, word: str) -> List[Tuple[str, str, float]]:
        """Get all reachable states from a word."""
        if word not in self.transitions:
            return []
        return [
            (t.to_state, t.verb, t.delta_g)
            for t in self.transitions[word]
        ]

    def get_state(self, word: str) -> Optional[SemanticState]:
        """Get state by word."""
        return self.states.get(word)

    def __len__(self):
        return len(self.states)
