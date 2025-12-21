"""
Knowledge Base: Tracking lived experience.

Core principle: "Only believe what was lived is knowledge"

Knowledge is not information - it is embodied experience.
You can only tunnel to states connected to what you have lived.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
from datetime import datetime


@dataclass
class LivedExperience:
    """Record of a lived moment in semantic space."""

    state: str              # The word/concept experienced
    timestamp: int          # Step when experienced
    duration: int = 1       # How long stayed in this state
    intensity: float = 1.0  # Emotional intensity of experience
    context: List[str] = field(default_factory=list)  # Neighboring states at time

    def __repr__(self):
        return f"Lived('{self.state}', t={self.timestamp}, intensity={self.intensity:.2f})"


@dataclass
class TunnelMemory:
    """Memory of a successful tunnel (insight/breakthrough)."""

    from_state: str
    to_state: str
    timestamp: int
    probability: float      # The P(tunnel) when it happened
    delta_g: float          # Change in goodness
    was_believed: bool      # Did agent believe before?

    def __repr__(self):
        return f"Tunnel('{self.from_state}' ══> '{self.to_state}', Δg={self.delta_g:+.2f})"


class KnowledgeBase:
    """
    Tracks what the agent has lived and learned.

    Knowledge accumulates through:
    1. Visiting states (lived experience)
    2. Successful tunnels (insights)
    3. Connections between lived states

    Knowledge enables:
    - Tunneling to connected states
    - Understanding of semantic barriers
    - Growth of believe parameter
    """

    def __init__(self, similarity_threshold: float = 0.7, decay_rate: float = 0.0):
        """
        Initialize knowledge base.

        Args:
            similarity_threshold: How similar a state must be to count as "known"
            decay_rate: How fast knowledge fades (0 = permanent)
        """
        self.similarity_threshold = similarity_threshold
        self.decay_rate = decay_rate

        # Core knowledge structures
        self.lived: Dict[str, LivedExperience] = {}  # word -> experience
        self.tunnels: List[TunnelMemory] = []        # successful tunnels
        self.path: List[str] = []                     # full journey

        # Derived knowledge
        self._connections: Dict[str, Set[str]] = {}   # word -> connected words
        self._barriers_discovered: Set[Tuple[str, str]] = set()  # known thick barriers

        # Statistics
        self.total_steps = 0
        self.total_tunnels = 0
        self.total_distance = 0.0

    def record_visit(self, state: str, context: List[str] = None):
        """
        Record visiting a state (lived experience).

        Args:
            state: The word/concept visited
            context: Neighboring states at time of visit
        """
        self.total_steps += 1

        if state in self.lived:
            # Revisiting - increase duration and intensity
            self.lived[state].duration += 1
            self.lived[state].intensity = min(2.0, self.lived[state].intensity + 0.1)
        else:
            # New experience
            self.lived[state] = LivedExperience(
                state=state,
                timestamp=self.total_steps,
                context=context or []
            )

        # Update path
        self.path.append(state)

        # Update connections
        if len(self.path) >= 2:
            prev = self.path[-2]
            self._add_connection(prev, state)

    def record_tunnel(self, from_state: str, to_state: str,
                      probability: float, delta_g: float, believed: bool = True):
        """
        Record a successful tunnel (insight/breakthrough).

        This is special knowledge - a discovered shortcut through semantic space.
        """
        self.total_tunnels += 1

        tunnel = TunnelMemory(
            from_state=from_state,
            to_state=to_state,
            timestamp=self.total_steps,
            probability=probability,
            delta_g=delta_g,
            was_believed=believed
        )
        self.tunnels.append(tunnel)

        # Tunnels create strong bidirectional connections
        self._add_connection(from_state, to_state)
        self._add_connection(to_state, from_state)

        # Mark this as lived
        self.record_visit(to_state, context=[from_state])

    def record_barrier(self, from_state: str, to_state: str):
        """Record discovering a thick barrier (failed tunnel attempt)."""
        self._barriers_discovered.add((from_state, to_state))

    def _add_connection(self, from_word: str, to_word: str):
        """Add a connection between two words."""
        if from_word not in self._connections:
            self._connections[from_word] = set()
        self._connections[from_word].add(to_word)

    def has_lived(self, state: str) -> bool:
        """Check if agent has lived (visited) this state."""
        return state in self.lived

    def knowledge_of(self, state: str) -> float:
        """
        How much knowledge does agent have of this state?

        Returns value in [0, 1]:
        - 1.0 = directly lived
        - 0.5-0.9 = connected to lived
        - 0.0 = unknown
        """
        if state in self.lived:
            # Directly lived - full knowledge
            # Intensity and duration affect knowledge
            exp = self.lived[state]
            base = 0.8
            intensity_bonus = min(0.2, exp.intensity * 0.1)
            return base + intensity_bonus

        # Check if connected to lived states
        for lived_state, connections in self._connections.items():
            if state in connections:
                # Connected to something we lived
                return 0.5

        # Check if reachable through tunnel paths
        for tunnel in self.tunnels:
            if tunnel.from_state == state or tunnel.to_state == state:
                return 0.4

        return 0.0

    def can_tunnel_to(self, target: str, from_state: str = None) -> Tuple[bool, float]:
        """
        Check if agent can tunnel to target state.

        Core principle: "Only believe what was lived is knowledge"
        - Can tunnel to lived states
        - Can tunnel to states connected to lived
        - Cannot tunnel to pure abstractions

        Returns:
            (can_tunnel, knowledge_factor)
        """
        knowledge = self.knowledge_of(target)

        if knowledge >= 0.4:
            return True, knowledge
        else:
            return False, 0.0

    def get_lived_states(self) -> Set[str]:
        """Get all states that have been lived."""
        return set(self.lived.keys())

    def get_tunnel_targets(self, from_state: str) -> List[str]:
        """Get all valid tunnel targets from current state."""
        valid = []
        for state in self.lived.keys():
            if state != from_state:
                valid.append(state)

        # Add tunnel destinations
        for tunnel in self.tunnels:
            if tunnel.from_state == from_state and tunnel.to_state not in valid:
                valid.append(tunnel.to_state)

        return valid

    def get_journey_summary(self) -> Dict:
        """Get summary of the agent's journey."""
        return {
            "total_steps": self.total_steps,
            "unique_states": len(self.lived),
            "total_tunnels": self.total_tunnels,
            "path_length": len(self.path),
            "connections": sum(len(c) for c in self._connections.values()),
            "barriers_found": len(self._barriers_discovered),
            "most_visited": max(self.lived.values(),
                               key=lambda x: x.duration).state if self.lived else None,
            "most_intense": max(self.lived.values(),
                               key=lambda x: x.intensity).state if self.lived else None,
        }

    def __repr__(self):
        return (f"KnowledgeBase(lived={len(self.lived)}, "
                f"tunnels={len(self.tunnels)}, "
                f"connections={sum(len(c) for c in self._connections.values())})")
