"""
Semantic Physics: How semantic properties become physical.

Maps the 16D semantic space to a physical simulation where:
- τ (abstraction) → altitude
- g (goodness) → light/reward
- j (direction) → force direction
- believe → jump power
- temperature → speed/energy
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from core.semantic_state import SemanticState


@dataclass
class PhysicsConfig:
    """Configuration for semantic physics."""

    # Mapping scales
    tau_to_altitude: float = 1.0
    goodness_to_reward: float = 1.0

    # Physical constants
    gravity: float = 0.1
    friction_base: float = 0.5
    max_speed: float = 2.0

    # Tunneling physics
    tunnel_energy_cost: float = 0.1
    barrier_visibility_range: float = 2.0


class SemanticPhysics:
    """
    Physics engine for semantic space.

    Converts semantic properties to physical interactions.
    """

    def __init__(self, config: PhysicsConfig = None):
        self.config = config or PhysicsConfig()

    def compute_movement_cost(self, from_state: SemanticState,
                              to_state: SemanticState,
                              temperature: float = 1.0) -> float:
        """
        Compute energy cost of moving between states.

        Cost depends on:
        - Altitude change (climbing is hard)
        - Friction of destination
        - Temperature (lower T = harder to move)
        """
        # Altitude cost (going up is harder)
        altitude_diff = to_state.altitude - from_state.altitude
        altitude_cost = max(0, altitude_diff) * self.config.gravity

        # Friction cost
        friction_cost = to_state.friction * self.config.friction_base

        # Temperature modifier (cold = stuck)
        temp_modifier = 1.0 / max(0.1, temperature)

        return (altitude_cost + friction_cost) * temp_modifier

    def compute_tunnel_cost(self, from_state: SemanticState,
                            to_state: SemanticState,
                            believe: float = 0.5) -> float:
        """
        Compute energy cost of tunneling.

        Tunneling cost = base_cost / believe
        (Higher believe = easier tunneling)
        """
        distance = from_state.distance_to(to_state)
        base_cost = distance * self.config.tunnel_energy_cost

        # Believe reduces cost
        return base_cost / max(0.1, believe)

    def compute_reward(self, from_state: SemanticState,
                       to_state: SemanticState,
                       is_tunnel: bool = False) -> float:
        """
        Compute reward for a transition.

        Reward = Δg (change in goodness) * scale
        Bonus for tunneling to better states
        """
        delta_g = to_state.goodness - from_state.goodness
        base_reward = delta_g * self.config.goodness_to_reward

        # Tunnel bonus if positive
        if is_tunnel and delta_g > 0:
            base_reward *= 1.5

        return base_reward

    def compute_believe_modifier(self, state: SemanticState,
                                  current_believe: float) -> float:
        """
        Compute how a state modifies believe.

        Some states boost believe (hope, courage)
        Some states reduce believe (fear, despair)
        """
        # Base modifier from state
        modifier = state.believe_modifier

        # Goodness also affects believe
        goodness_effect = state.goodness * 0.1

        return current_believe + modifier + goodness_effect

    def compute_temperature_change(self, state: SemanticState,
                                    current_temp: float,
                                    cooling_rate: float = 0.95) -> float:
        """
        Compute temperature change at a state.

        Some states "warm up" thinking (creative states)
        Some states "cool down" thinking (decisive states)
        """
        # Natural cooling
        new_temp = current_temp * cooling_rate

        # Altitude affects temperature (higher = cooler)
        altitude_effect = -state.altitude * 0.01

        # Abstract concepts cool thinking (more focused)
        if state.tau > 1.5:
            altitude_effect -= 0.05

        return max(0.01, new_temp + altitude_effect)

    def is_traversable(self, from_state: SemanticState,
                       to_state: SemanticState,
                       temperature: float,
                       believe: float) -> Tuple[bool, str]:
        """
        Check if transition is possible.

        Returns (can_traverse, reason)
        """
        # Check altitude (can't climb too high without energy)
        altitude_diff = to_state.altitude - from_state.altitude
        if altitude_diff > temperature * 2:
            return False, "altitude_too_high"

        # Check requirements
        if to_state.requires:
            # These would need to check against knowledge
            pass

        return True, "ok"

    def compute_barrier_opacity(self, from_state: SemanticState,
                                 to_state: SemanticState) -> float:
        """
        Compute opacity of barrier between states.

        κ = (1 - cos(j₁, j₂)) / 2

        High opacity = thick barrier = hard to tunnel
        """
        cos_sim = from_state.cosine_similarity(to_state)
        return (1 - cos_sim) / 2

    def visualize_as_position(self, state: SemanticState) -> Tuple[float, float, float]:
        """
        Convert semantic state to 3D position for visualization.

        x = j[0] (first component of direction)
        y = j[1] (second component)
        z = τ (altitude/abstraction)
        """
        x = state.j_vector[0] if len(state.j_vector) > 0 else 0
        y = state.j_vector[1] if len(state.j_vector) > 1 else 0
        z = state.altitude

        return (x, y, z)

    def visualize_as_color(self, state: SemanticState) -> Tuple[int, int, int]:
        """
        Convert semantic state to RGB color.

        R = (1 - goodness) / 2  (evil = red)
        G = goodness / 2 + 0.5  (good = green)
        B = τ / 2               (abstract = blue)
        """
        r = int(((1 - state.goodness) / 2) * 255)
        g = int(((state.goodness + 1) / 2) * 255)
        b = int((state.tau / 2) * 255)

        return (r, g, b)
