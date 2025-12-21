"""
Quantum Agent: Agent that uses believe and tunneling.

Implements the core principle: "Only believe what was lived is knowledge"

The agent learns to balance:
- Thermal exploration (gradual, safe)
- Quantum tunneling (risky but potentially breakthrough)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .base_agent import BaseAgent


class QuantumAgent(BaseAgent):
    """
    Agent that navigates semantic space using quantum-inspired decisions.

    Key parameters:
    - believe: Capacity for breakthrough (affects tunnel probability)
    - temperature: Exploration vs exploitation balance

    Learning:
    - Q-values for verb actions
    - Tunnel value estimation based on lived experience
    """

    def __init__(self,
                 believe: float = 0.5,
                 temperature: float = 1.0,
                 cooling_rate: float = 0.99,
                 learning_rate: float = 0.1,
                 discount: float = 0.99,
                 tunnel_bonus: float = 0.5):
        """
        Initialize quantum agent.

        Args:
            believe: Initial belief in breakthrough (0-1)
            temperature: Initial exploration temperature
            cooling_rate: How fast temperature decreases
            learning_rate: Learning rate for Q-updates
            discount: Discount factor for future rewards
            tunnel_bonus: Bonus for attempting tunnels
        """
        super().__init__(name="QuantumAgent")

        self.believe = believe
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.learning_rate = learning_rate
        self.discount = discount
        self.tunnel_bonus = tunnel_bonus

        # Q-table: state_word -> action -> value
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Tunnel memory: from_word -> to_word -> (success_count, fail_count)
        self.tunnel_memory: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )

        # Experience tracking
        self.lived_words: set = set()
        self.tunnel_successes: int = 0
        self.tunnel_failures: int = 0

    def choose_action(self, observation: np.ndarray,
                      valid_actions: List[int],
                      info: Dict[str, Any]) -> int:
        """
        Choose action using quantum-inspired strategy.

        Strategy:
        1. FIRST: Consider tunneling if believe is high
        2. FALLBACK: Use thermal (verb) actions with Q-values

        Uses softmax over Q-values with temperature.
        """
        current_word = info.get("current_word", "unknown")
        self.lived_words.add(current_word)

        # Get Q-values for valid actions
        q_values = []
        for action in valid_actions:
            q = self.q_table[current_word][action]

            # Tunnel action (action=0) gets bonus based on believe
            if action == 0:
                q += self.believe * self.tunnel_bonus

            q_values.append(q)

        q_values = np.array(q_values)

        # Softmax with temperature
        if self.temperature > 0.01:
            probs = self._softmax(q_values / self.temperature)
        else:
            # Greedy
            probs = np.zeros_like(q_values)
            probs[np.argmax(q_values)] = 1.0

        # Sample action
        action_idx = np.random.choice(len(valid_actions), p=probs)
        return valid_actions[action_idx]

    def update(self, observation: np.ndarray, action: int,
               reward: float, next_observation: np.ndarray,
               done: bool, info: Dict[str, Any]):
        """
        Update agent after taking action.

        Uses Q-learning update + tunnel memory update.
        """
        current_word = info.get("current_word", "unknown")

        # Get next state info
        next_word = info.get("to", current_word)
        self.lived_words.add(next_word)

        # Q-learning update
        if done:
            target = reward
        else:
            # Max Q-value of next state
            next_q_values = [self.q_table[next_word][a]
                           for a in range(10)]  # Assume max 10 actions
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.discount * max_next_q

        # Update Q-value
        old_q = self.q_table[current_word][action]
        self.q_table[current_word][action] = (
            old_q + self.learning_rate * (target - old_q)
        )

        # Update tunnel memory if this was a tunnel action
        if action == 0 and info.get("action") == "tunnel":
            if info.get("success", False):
                self.tunnel_successes += 1
                to_word = info.get("to", "")
                if to_word:
                    self.tunnel_memory[current_word][to_word][0] += 1
                    # Successful tunnel boosts believe
                    self.believe = min(1.0, self.believe + 0.05)
            else:
                self.tunnel_failures += 1
                attempted = info.get("attempted_target", "")
                if attempted:
                    self.tunnel_memory[current_word][attempted][1] += 1
                # Failed tunnel slightly reduces believe
                self.believe = max(0.1, self.believe - 0.02)

        # Cool temperature
        self.temperature *= self.cooling_rate

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_tunnel_success_rate(self, from_word: str, to_word: str) -> float:
        """Get estimated success rate for a tunnel."""
        counts = self.tunnel_memory[from_word][to_word]
        total = counts[0] + counts[1]
        if total == 0:
            return 0.5  # Unknown, assume 50%
        return counts[0] / total

    def on_episode_start(self):
        """Reset episode-specific state."""
        super().on_episode_start()
        # Temperature might reset or continue
        # For now, keep cooling across episodes

    def on_episode_end(self, total_reward: float, steps: int):
        """Update after episode."""
        super().on_episode_end(total_reward, steps)
        # Could adjust believe based on episode outcome
        if total_reward > 0:
            self.believe = min(1.0, self.believe + 0.02)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = super().get_stats()
        stats.update({
            "believe": self.believe,
            "temperature": self.temperature,
            "lived_words": len(self.lived_words),
            "tunnel_successes": self.tunnel_successes,
            "tunnel_failures": self.tunnel_failures,
            "tunnel_success_rate": (
                self.tunnel_successes / max(1, self.tunnel_successes + self.tunnel_failures)
            ),
            "q_table_size": sum(len(v) for v in self.q_table.values()),
        })
        return stats

    def __repr__(self):
        return (f"QuantumAgent(believe={self.believe:.2f}, "
                f"T={self.temperature:.2f}, "
                f"lived={len(self.lived_words)})")
