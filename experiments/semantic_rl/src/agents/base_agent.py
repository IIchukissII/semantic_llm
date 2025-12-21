"""
Base Agent: Abstract interface for semantic RL agents.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseAgent(ABC):
    """
    Abstract base class for agents in semantic RL.

    Agents navigate semantic space by choosing actions:
    - Verb actions (thermal moves)
    - Tunnel action (quantum jumps)
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.episode_rewards: List[float] = []
        self.total_steps = 0

    @abstractmethod
    def choose_action(self, observation: np.ndarray,
                      valid_actions: List[int],
                      info: Dict[str, Any]) -> int:
        """
        Choose an action given the current observation.

        Args:
            observation: Current state observation
            valid_actions: List of valid action indices
            info: Additional information from environment

        Returns:
            Action index to take
        """
        pass

    @abstractmethod
    def update(self, observation: np.ndarray, action: int,
               reward: float, next_observation: np.ndarray,
               done: bool, info: Dict[str, Any]):
        """
        Update agent after taking an action.

        Args:
            observation: State before action
            action: Action taken
            reward: Reward received
            next_observation: State after action
            done: Whether episode ended
            info: Additional information
        """
        pass

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.episode_rewards = []

    def on_episode_end(self, total_reward: float, steps: int):
        """Called at the end of each episode."""
        self.total_steps += steps

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "name": self.name,
            "total_steps": self.total_steps,
        }


class RandomAgent(BaseAgent):
    """Simple random agent for baseline comparison."""

    def __init__(self):
        super().__init__(name="RandomAgent")

    def choose_action(self, observation: np.ndarray,
                      valid_actions: List[int],
                      info: Dict[str, Any]) -> int:
        """Choose a random valid action."""
        return np.random.choice(valid_actions)

    def update(self, observation: np.ndarray, action: int,
               reward: float, next_observation: np.ndarray,
               done: bool, info: Dict[str, Any]):
        """Random agent doesn't learn."""
        pass
