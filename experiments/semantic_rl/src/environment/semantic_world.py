"""
Semantic World: The main RL environment.

A Gymnasium-compatible environment where agents navigate semantic space,
learning through lived experience.

Core principle: "Only believe what was lived is knowledge"
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from core.semantic_state import SemanticState, SemanticGraph, Transition
from core.knowledge import KnowledgeBase
from environment.physics import SemanticPhysics, PhysicsConfig


@dataclass
class WorldConfig:
    """Configuration for the semantic world."""
    max_steps: int = 100
    reward_scale: float = 1.0
    tunnel_threshold: float = 0.3
    knowledge_required: bool = True

    # Exploration bonuses
    exploration_bonus: float = 0.5      # Bonus for visiting new states
    revisit_penalty: float = -0.1       # Penalty for revisiting same state
    loop_penalty: float = -0.3          # Extra penalty for oscillating
    goal_distance_weight: float = 0.2   # Weight for goal-distance shaping


class SemanticWorld(gym.Env):
    """
    Gymnasium environment for semantic space navigation.

    Observation: Current state's semantic properties
    Actions: Verbs (thermal moves) or tunnel attempts (quantum jumps)

    The agent learns through lived experience - only states that
    have been visited (or connected to visited) can be tunneled to.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 semantic_graph: SemanticGraph = None,
                 start_word: str = "darkness",
                 goal_word: str = "wisdom",
                 config: WorldConfig = None,
                 render_mode: str = None):
        """
        Initialize semantic world.

        Args:
            semantic_graph: The semantic graph (states + transitions)
            start_word: Starting concept
            goal_word: Goal concept
            config: World configuration
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config or WorldConfig()
        self.render_mode = render_mode

        # Initialize graph (or create default)
        if semantic_graph is not None:
            self.graph = semantic_graph
        else:
            self.graph = self._create_default_graph()

        # Physics engine
        self.physics = SemanticPhysics()

        # Knowledge base (tracks lived experience)
        self.knowledge = KnowledgeBase()

        # World state
        self.start_word = start_word
        self.goal_word = goal_word
        self.current_state: Optional[SemanticState] = None
        self.current_step = 0

        # Agent state
        self.believe = 0.5
        self.temperature = 1.0

        # Build action space
        self._build_action_space()

        # Observation space: semantic + physical properties
        obs_size = 2 + 16 + 4 + 2  # tau, g, j_vector, physical, believe, temp
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _create_default_graph(self) -> SemanticGraph:
        """Create a simple default semantic graph for testing."""
        graph = SemanticGraph()

        # Create some basic states
        concepts = {
            "darkness": {"tau": 0.5, "g": 0.1, "believe_mod": -0.1},
            "fear": {"tau": 0.6, "g": 0.3, "believe_mod": -0.2},
            "struggle": {"tau": 0.8, "g": -0.2, "believe_mod": 0.1},
            "hope": {"tau": 1.0, "g": 0.7, "believe_mod": 0.3},
            "courage": {"tau": 1.2, "g": 0.8, "believe_mod": 0.2, "requires": ["fear"]},
            "wisdom": {"tau": 1.8, "g": 0.9, "believe_mod": 0.1, "requires": ["struggle"]},
            "love": {"tau": 1.5, "g": 1.0, "believe_mod": 0.5},
            "despair": {"tau": 0.3, "g": -0.5, "believe_mod": -0.3},
            "truth": {"tau": 1.6, "g": 0.6, "believe_mod": 0.1},
            "change": {"tau": 1.0, "g": 0.5, "believe_mod": 0.2},
        }

        for word, props in concepts.items():
            j_vector = np.random.randn(16) * 0.5
            j_vector = j_vector / (np.linalg.norm(j_vector) + 1e-8)

            state = SemanticState(
                word=word,
                tau=props["tau"],
                goodness=props["g"],
                j_vector=j_vector,
                believe_modifier=props.get("believe_mod", 0),
                requires=props.get("requires", [])
            )
            graph.add_state(state)

        # Create some transitions (verbs)
        transitions = [
            ("darkness", "fear", "feel"),
            ("darkness", "struggle", "embrace"),
            ("fear", "courage", "face"),
            ("fear", "despair", "surrender"),
            ("struggle", "hope", "find"),
            ("struggle", "wisdom", "learn"),
            ("hope", "love", "open"),
            ("hope", "change", "believe"),
            ("courage", "truth", "seek"),
            ("courage", "wisdom", "earn"),
            ("despair", "darkness", "fall"),
            ("change", "wisdom", "grow"),
            ("love", "wisdom", "understand"),
            ("truth", "wisdom", "realize"),
        ]

        for from_word, to_word, verb in transitions:
            from_state = graph.get_state(from_word)
            to_state = graph.get_state(to_word)
            if from_state and to_state:
                delta_g = to_state.goodness - from_state.goodness
                graph.add_transition(Transition(
                    verb=verb,
                    from_state=from_word,
                    to_state=to_word,
                    delta_g=delta_g
                ))

        return graph

    def _build_action_space(self):
        """Build action space from available verbs + tunnel action."""
        # Collect all unique actions
        self.verb_actions = []
        for transitions in self.graph.transitions.values():
            for t in transitions:
                if t.verb not in self.verb_actions:
                    self.verb_actions.append(t.verb)

        # Action 0 = tunnel, rest = verbs
        self.num_actions = 1 + len(self.verb_actions)
        self.action_space = spaces.Discrete(self.num_actions)

        # Action mapping
        self.action_to_name = {0: "tunnel"}
        for i, verb in enumerate(self.verb_actions):
            self.action_to_name[i + 1] = verb

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset world state
        self.current_state = self.graph.get_state(self.start_word)
        self.goal_state = self.graph.get_state(self.goal_word)
        self.current_step = 0

        # Reset agent state
        self.believe = 0.5
        self.temperature = 1.0

        # Loop detection - track recent states
        self.recent_states = []  # Last N states for loop detection
        self.visit_counts = {}   # Count visits per state

        # Reset knowledge (or keep it for transfer learning)
        if options and options.get("keep_knowledge", False):
            pass  # Keep existing knowledge
        else:
            self.knowledge = KnowledgeBase()

        # Record starting state as lived
        self.knowledge.record_visit(self.start_word)
        self._record_visit(self.start_word)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _record_visit(self, word: str):
        """Record a state visit for loop detection."""
        self.recent_states.append(word)
        if len(self.recent_states) > 10:  # Keep last 10
            self.recent_states.pop(0)
        self.visit_counts[word] = self.visit_counts.get(word, 0) + 1

    def _detect_loop(self) -> bool:
        """Detect if agent is stuck in a loop."""
        if len(self.recent_states) < 4:
            return False
        # Check for A-B-A-B pattern
        last4 = self.recent_states[-4:]
        return last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]

    def _compute_exploration_reward(self, to_word: str, base_reward: float) -> float:
        """Add exploration bonuses and penalties."""
        reward = base_reward

        # Exploration bonus for new states
        if self.visit_counts.get(to_word, 0) == 0:
            reward += self.config.exploration_bonus

        # Revisit penalty
        elif self.visit_counts.get(to_word, 0) > 1:
            reward += self.config.revisit_penalty * min(self.visit_counts[to_word], 5)

        # Loop penalty
        if self._detect_loop():
            reward += self.config.loop_penalty

        # Goal distance shaping (if goal exists)
        if self.goal_state is not None:
            to_state = self.graph.get_state(to_word)
            if to_state:
                # Reward for getting closer to goal in semantic space
                dist_to_goal = to_state.distance_to(self.goal_state)
                current_dist = self.current_state.distance_to(self.goal_state)
                progress = current_dist - dist_to_goal  # Positive if getting closer
                reward += progress * self.config.goal_distance_weight

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: 0 = tunnel, 1+ = verb actions

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        old_word = self.current_state.word

        if action == 0:
            # Tunnel action
            reward, success, info = self._do_tunnel()
        else:
            # Verb action
            verb = self.action_to_name.get(action, None)
            reward, success, info = self._do_verb(verb)

        # Apply exploration rewards if transition succeeded
        if success and info.get("to"):
            to_word = info["to"]
            reward = self._compute_exploration_reward(to_word, reward)
            self._record_visit(to_word)
            info["is_new_state"] = self.visit_counts.get(to_word, 0) == 1
            info["in_loop"] = self._detect_loop()

        # Update temperature
        self.temperature = self.physics.compute_temperature_change(
            self.current_state, self.temperature
        )

        # Update believe based on current state
        self.believe = self.physics.compute_believe_modifier(
            self.current_state, self.believe
        )
        self.believe = np.clip(self.believe, 0.0, 1.0)

        # Check termination
        terminated = self.current_state.word == self.goal_word
        truncated = self.current_step >= self.config.max_steps

        # Build observation
        obs = self._get_observation()
        info.update(self._get_info())

        return obs, reward, terminated, truncated, info

    def _do_tunnel(self) -> Tuple[float, bool, dict]:
        """
        Attempt quantum tunneling.

        Can only tunnel to states with lived connection.
        P(success) = believe × e^(-2κd) × knowledge(target)
        """
        info = {"action": "tunnel", "success": False}

        # Get valid tunnel targets (based on knowledge)
        valid_targets = self.knowledge.get_tunnel_targets(self.current_state.word)

        if not valid_targets:
            # No valid targets - failed tunnel
            info["reason"] = "no_valid_targets"
            return -0.1, False, info

        # Choose best target (highest goodness that we can reach)
        best_target = None
        best_score = -np.inf

        for target_word in valid_targets:
            target_state = self.graph.get_state(target_word)
            if target_state is None:
                continue

            # Compute tunnel probability
            base_prob = self.current_state.tunnel_probability(target_state)
            can_tunnel, knowledge_factor = self.knowledge.can_tunnel_to(target_word)

            if can_tunnel:
                effective_prob = self.believe * base_prob * knowledge_factor
                score = target_state.goodness * effective_prob

                if score > best_score:
                    best_score = score
                    best_target = target_word

        if best_target is None:
            info["reason"] = "no_reachable_target"
            return -0.1, False, info

        # Attempt tunnel
        target_state = self.graph.get_state(best_target)
        base_prob = self.current_state.tunnel_probability(target_state)
        _, knowledge_factor = self.knowledge.can_tunnel_to(best_target)
        effective_prob = self.believe * base_prob * knowledge_factor

        if np.random.random() < effective_prob:
            # Successful tunnel!
            old_state = self.current_state
            self.current_state = target_state

            # Record in knowledge
            delta_g = target_state.goodness - old_state.goodness
            self.knowledge.record_tunnel(
                old_state.word, target_state.word,
                effective_prob, delta_g, self.believe > 0.5
            )

            # Compute reward
            reward = self.physics.compute_reward(old_state, target_state, is_tunnel=True)

            info["success"] = True
            info["from"] = old_state.word
            info["to"] = target_state.word
            info["probability"] = effective_prob
            info["delta_g"] = delta_g

            return reward, True, info
        else:
            # Failed tunnel - record barrier
            self.knowledge.record_barrier(self.current_state.word, best_target)
            info["reason"] = "probability_failed"
            info["attempted_target"] = best_target
            return -0.05, False, info

    def _do_verb(self, verb: str) -> Tuple[float, bool, dict]:
        """
        Execute a verb action (thermal move).

        This is gradual, always possible if transition exists.
        """
        info = {"action": "verb", "verb": verb, "success": False}

        if verb is None:
            info["reason"] = "invalid_verb"
            return -0.1, False, info

        # Find transition with this verb from current state
        neighbors = self.graph.get_neighbors(self.current_state.word)
        valid_transition = None

        for to_word, trans_verb, delta_g in neighbors:
            if trans_verb == verb:
                valid_transition = (to_word, delta_g)
                break

        if valid_transition is None:
            info["reason"] = "no_such_transition"
            return -0.05, False, info

        to_word, delta_g = valid_transition
        to_state = self.graph.get_state(to_word)

        if to_state is None:
            info["reason"] = "invalid_target_state"
            return -0.1, False, info

        # Check if traversable
        can_traverse, reason = self.physics.is_traversable(
            self.current_state, to_state, self.temperature, self.believe
        )

        if not can_traverse:
            info["reason"] = reason
            return -0.05, False, info

        # Execute transition
        old_state = self.current_state
        self.current_state = to_state

        # Record in knowledge
        self.knowledge.record_visit(to_word, context=[old_state.word])

        # Compute reward
        reward = self.physics.compute_reward(old_state, to_state, is_tunnel=False)

        info["success"] = True
        info["from"] = old_state.word
        info["to"] = to_state.word
        info["delta_g"] = delta_g

        return reward, True, info

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        state_obs = self.current_state.to_observation()

        # Add agent state
        agent_obs = np.array([self.believe, self.temperature])

        return np.concatenate([state_obs, agent_obs]).astype(np.float32)

    def _get_info(self) -> dict:
        """Build info dictionary."""
        return {
            "current_word": self.current_state.word,
            "current_goodness": self.current_state.goodness,
            "believe": self.believe,
            "temperature": self.temperature,
            "step": self.current_step,
            "knowledge_summary": self.knowledge.get_journey_summary(),
            "valid_actions": self.get_valid_actions(),
        }

    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices."""
        valid = []

        # Tunnel is always attemptable (might fail)
        valid.append(0)

        # Check which verbs are valid from current state
        neighbors = self.graph.get_neighbors(self.current_state.word)
        available_verbs = {trans_verb for _, trans_verb, _ in neighbors}

        for i, verb in enumerate(self.verb_actions):
            if verb in available_verbs:
                valid.append(i + 1)

        return valid

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def _render_text(self):
        """Text-based rendering."""
        print(f"\n{'='*50}")
        print(f"Step {self.current_step}")
        print(f"State: {self.current_state.word}")
        print(f"  τ={self.current_state.tau:.2f}, g={self.current_state.goodness:+.2f}")
        print(f"Agent: believe={self.believe:.2f}, T={self.temperature:.2f}")
        print(f"Knowledge: {len(self.knowledge.lived)} lived, "
              f"{len(self.knowledge.tunnels)} tunnels")
        print(f"Path: {' -> '.join(self.knowledge.path[-5:])}")
        print(f"{'='*50}")

    def _render_rgb(self) -> np.ndarray:
        """RGB array rendering (for visualization)."""
        # Simple placeholder - could be expanded
        return np.zeros((100, 100, 3), dtype=np.uint8)
