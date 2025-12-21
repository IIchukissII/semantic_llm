"""
LLM Storyteller Agent: Navigate semantic space with narrative understanding.

The agent uses an LLM to:
1. Understand the current semantic context
2. Choose meaningful actions based on narrative coherence
3. Generate poetic descriptions of the journey
4. Explain tunneling events as moments of insight

"Only believe what was lived is knowledge"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class NarrativeContext:
    """Context for the LLM to understand the journey."""
    book_title: str
    current_word: str
    goal_word: str
    path_so_far: List[str]
    available_actions: List[Dict]
    believe: float
    goodness: float
    tau: float
    tunnel_count: int
    step: int


class LLMClient:
    """
    Client for LLM inference.

    Supports:
    - Ollama (local)
    - OpenAI API
    - Anthropic API
    """

    def __init__(self,
                 backend: str = "ollama",
                 model: str = "qwen2.5:1.5b",
                 base_url: str = "http://localhost:11434"):
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.available = False

        if not HAS_REQUESTS:
            print("LLMClient: requests library not available")
            return

        if backend == "ollama":
            self._init_ollama()
        elif backend == "openai":
            self._init_openai()

    def _init_ollama(self):
        """Initialize Ollama connection."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                # Check for model with or without :latest suffix
                model_base = self.model.split(':')[0]
                if any(model_base in m for m in models):
                    self.available = True
                    print(f"LLMClient: Connected to Ollama ({self.model})")
                else:
                    print(f"LLMClient: Model {self.model} not found. Available: {models[:5]}")
        except Exception as e:
            print(f"LLMClient: Cannot connect to Ollama ({e})")

    def _init_openai(self):
        """Initialize OpenAI connection."""
        import os
        if os.getenv("OPENAI_API_KEY"):
            self.available = True
            print(f"LLMClient: OpenAI API available")
        else:
            print("LLMClient: OPENAI_API_KEY not set")

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        if not self.available:
            return self._fallback_generate(prompt)

        if self.backend == "ollama":
            return self._ollama_generate(prompt, max_tokens, temperature)
        elif self.backend == "openai":
            return self._openai_generate(prompt, max_tokens, temperature)

        return self._fallback_generate(prompt)

    def _ollama_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Ollama."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
        except Exception as e:
            print(f"Ollama error: {e}")
        return self._fallback_generate(prompt)

    def _openai_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using OpenAI."""
        import os
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenAI error: {e}")
        return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when LLM not available."""
        # Extract key info and generate template response
        return "[LLM not available - using template]"


class LLMStorytellerAgent:
    """
    An agent that uses LLM to navigate semantic space narratively.

    The agent generates poetic descriptions and chooses actions
    based on narrative coherence rather than just rewards.
    """

    def __init__(self,
                 llm_backend: str = "ollama",
                 llm_model: str = "qwen2.5:1.5b",
                 believe: float = 0.5,
                 narrative_weight: float = 0.5,
                 verbose: bool = True):
        """
        Initialize storyteller agent.

        Args:
            llm_backend: "ollama" or "openai"
            llm_model: Model name
            believe: Initial believe parameter
            narrative_weight: How much to weight LLM suggestions vs rewards
            verbose: Whether to print narrative
        """
        self.llm = LLMClient(backend=llm_backend, model=llm_model)
        self.believe = believe
        self.narrative_weight = narrative_weight
        self.verbose = verbose

        # Journey state
        self.path = []
        self.narratives = []
        self.insights = []  # Tunnel events as insights
        self.book_title = ""
        self.goal_word = ""

    def on_episode_start(self, book_title: str = "", goal_word: str = ""):
        """Called at start of journey."""
        self.path = []
        self.narratives = []
        self.insights = []
        self.book_title = book_title
        self.goal_word = goal_word

    def choose_action(self,
                      obs: np.ndarray,
                      valid_actions: List[int],
                      info: Dict,
                      action_names: Dict[int, str],
                      graph = None) -> Tuple[int, str]:
        """
        Choose action using LLM guidance.

        Returns:
            (action_index, narrative_reason)
        """
        current_word = info.get("current_word", "unknown")
        self.path.append(current_word)

        # Build context for LLM
        context = self._build_context(info, valid_actions, action_names, graph)

        # Get LLM suggestion
        action, reason = self._llm_choose_action(context, valid_actions, action_names, graph)

        if self.verbose:
            print(f"\n  Narrator: {reason}")

        return action, reason

    def _build_context(self, info: Dict, valid_actions: List[int],
                       action_names: Dict[int, str], graph) -> NarrativeContext:
        """Build context for LLM."""
        current_word = info.get("current_word", "unknown")

        # Get action details
        actions = []
        if graph:
            neighbors = graph.get_neighbors(current_word)
            for to_word, verb, delta_g in neighbors:
                actions.append({
                    "verb": verb,
                    "to": to_word,
                    "delta_g": delta_g
                })

        # Always include tunnel option
        actions.insert(0, {"verb": "tunnel", "to": "insight", "delta_g": 0})

        return NarrativeContext(
            book_title=self.book_title,
            current_word=current_word,
            goal_word=self.goal_word,
            path_so_far=self.path.copy(),
            available_actions=actions[:10],  # Limit for prompt
            believe=info.get("believe", self.believe),
            goodness=info.get("current_goodness", 0),
            tau=info.get("tau", 1.0),
            tunnel_count=len(self.insights),
            step=len(self.path)
        )

    def _llm_choose_action(self, context: NarrativeContext,
                           valid_actions: List[int],
                           action_names: Dict[int, str],
                           graph) -> Tuple[int, str]:
        """Use LLM to choose action and generate reason."""

        # Build prompt
        prompt = self._build_choice_prompt(context)

        # Get LLM response
        response = self.llm.generate(prompt, max_tokens=150, temperature=0.8)

        # Parse response for action choice
        action, reason = self._parse_response(response, context, valid_actions, action_names, graph)

        return action, reason

    def _build_choice_prompt(self, context: NarrativeContext) -> str:
        """Build prompt for action selection."""
        path_str = " → ".join(context.path_so_far[-5:])

        actions_str = ""
        for i, action in enumerate(context.available_actions[:8]):
            actions_str += f"  {i}. {action['verb']} → {action['to']}\n"

        prompt = f"""You are narrating a journey through the semantic landscape of "{context.book_title}".

Current state: {context.current_word}
Goal: {context.goal_word}
Journey so far: {path_str}
Believe level: {context.believe:.2f}
Step: {context.step}

Available actions:
{actions_str}

As the narrator, choose the most narratively meaningful next step.
If believe is high ({context.believe:.2f}), tunneling (instant insight) might be powerful.

Respond in this format:
ACTION: [number 0-7]
REASON: [One poetic sentence explaining the choice]"""

        return prompt

    def _parse_response(self, response: str, context: NarrativeContext,
                        valid_actions: List[int], action_names: Dict[int, str],
                        graph) -> Tuple[int, str]:
        """Parse LLM response to extract action and reason."""

        # Default fallback
        default_action = valid_actions[0] if valid_actions else 0
        default_reason = f"The journey continues through {context.current_word}..."

        if not response or "[LLM not available" in response:
            # Use heuristic when LLM unavailable
            return self._heuristic_choice(context, valid_actions, action_names, graph)

        try:
            # Extract action number
            action_num = 0
            for line in response.split('\n'):
                if 'ACTION:' in line.upper():
                    # Find number in line
                    nums = [int(s) for s in line.split() if s.isdigit()]
                    if nums:
                        action_num = nums[0]
                        break

            # Extract reason
            reason = default_reason
            for line in response.split('\n'):
                if 'REASON:' in line.upper():
                    reason = line.split(':', 1)[1].strip()
                    break

            # Map action number to valid action
            if action_num < len(context.available_actions):
                if action_num == 0:  # Tunnel
                    action = 0
                else:
                    # Find corresponding verb action
                    verb = context.available_actions[action_num].get('verb', '')
                    for a_idx, a_name in action_names.items():
                        if a_name == verb and a_idx in valid_actions:
                            action = a_idx
                            break
                    else:
                        action = default_action
            else:
                action = default_action

            return action, reason

        except Exception as e:
            return default_action, default_reason

    def _heuristic_choice(self, context: NarrativeContext,
                          valid_actions: List[int],
                          action_names: Dict[int, str],
                          graph) -> Tuple[int, str]:
        """Heuristic choice when LLM unavailable."""

        # If high believe, consider tunneling
        if context.believe > 0.7 and 0 in valid_actions:
            if np.random.random() < context.believe * 0.3:
                return 0, f"A sudden insight pierces through {context.current_word}..."

        # Otherwise, choose action toward goal (highest delta_g)
        best_action = valid_actions[1] if len(valid_actions) > 1 else valid_actions[0]
        best_delta_g = -float('inf')

        if graph:
            neighbors = graph.get_neighbors(context.current_word)
            for to_word, verb, delta_g in neighbors:
                for a_idx, a_name in action_names.items():
                    if a_name == verb and a_idx in valid_actions:
                        if delta_g > best_delta_g:
                            best_delta_g = delta_g
                            best_action = a_idx

        verb_name = action_names.get(best_action, "continue")
        return best_action, f"The path leads from {context.current_word}..."

    def narrate_transition(self, from_word: str, to_word: str,
                          verb: str, is_tunnel: bool,
                          delta_g: float) -> str:
        """Generate narrative for a transition."""

        if is_tunnel:
            prompt = f"""Generate a single poetic sentence describing a moment of sudden insight,
where understanding leaps from "{from_word}" to "{to_word}".
This is a quantum tunnel - an instant transformation of understanding.
Make it mystical and profound."""
        else:
            prompt = f"""Generate a single poetic sentence describing the journey
from "{from_word}" through the action of "{verb}" to arrive at "{to_word}".
The change in goodness is {delta_g:+.2f}.
Make it evocative and narrative."""

        response = self.llm.generate(prompt, max_tokens=80, temperature=0.9)

        if "[LLM not available" in response:
            if is_tunnel:
                return f"In a flash of insight, {from_word} becomes {to_word}."
            else:
                return f"Through {verb}, {from_word} transforms into {to_word}."

        return response.strip()

    def update(self, obs, action, reward, next_obs, done, info):
        """Update agent after action."""
        if info.get("success"):
            to_word = info.get("to", "")
            from_word = info.get("from", "")

            if action == 0:  # Tunnel
                self.insights.append({
                    "from": from_word,
                    "to": to_word,
                    "believe": self.believe
                })

            # Update believe based on success
            if reward > 0:
                self.believe = min(1.0, self.believe + 0.05)
            else:
                self.believe = max(0.1, self.believe - 0.02)

    def get_journey_narrative(self) -> str:
        """Generate full narrative of the journey."""
        if not self.path:
            return "The journey has not yet begun."

        prompt = f"""Write a short poetic narrative (3-4 sentences) about a journey through these concepts:
{' → '.join(self.path)}

The journey is through the book "{self.book_title}".
There were {len(self.insights)} moments of sudden insight (tunneling).
The goal was to reach "{self.goal_word}".

Write in a mystical, philosophical tone."""

        response = self.llm.generate(prompt, max_tokens=200, temperature=0.8)

        if "[LLM not available" in response:
            return f"The journey led through {len(self.path)} states of understanding, " \
                   f"with {len(self.insights)} moments of insight, " \
                   f"seeking {self.goal_word}."

        return response.strip()


def run_interactive_book_journey(book_key: str = "heart_of_darkness",
                                  max_steps: int = 30,
                                  llm_model: str = "qwen2.5:1.5b"):
    """
    Run an interactive LLM-narrated journey through a book.
    """
    import sys
    from pathlib import Path

    # Add paths
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

    from environment.book_world import BookWorld, get_book_path

    print("=" * 70)
    print("LLM STORYTELLER: Interactive Book Journey")
    print("=" * 70)

    # Load book
    book_path = get_book_path(book_key)
    print(f"\nBook: {book_key}")
    print(f"Loading semantic landscape...")

    world = BookWorld(book_file=str(book_path))

    # Create storyteller agent
    agent = LLMStorytellerAgent(
        llm_model=llm_model,
        believe=0.5,
        verbose=True
    )

    print(f"\nJourney: {world.start_word} → {world.goal_word}")
    print("=" * 70)

    # Run journey
    obs, info = world.reset()
    agent.on_episode_start(book_title=book_key, goal_word=world.goal_word)

    print(f"\n{'─' * 70}")
    print("THE JOURNEY BEGINS")
    print(f"{'─' * 70}")

    for step in range(max_steps):
        valid = world.get_valid_actions()
        action, reason = agent.choose_action(
            obs, valid, info,
            world.action_to_name,
            world.graph
        )

        next_obs, reward, term, trunc, info = world.step(action)
        agent.update(obs, action, reward, next_obs, term or trunc, info)

        if info.get("success"):
            from_word = info.get("from", "?")
            to_word = info.get("to", "?")
            verb = world.action_to_name.get(action, "tunnel")

            # Generate transition narrative
            if agent.llm.available and step % 3 == 0:  # Every 3rd step
                narrative = agent.narrate_transition(
                    from_word, to_word, verb,
                    action == 0, info.get("delta_g", 0)
                )
                print(f"\n  {narrative}")

        obs = next_obs

        if term:
            print(f"\n  *** REACHED: {world.goal_word} ***")
            break

    # Final narrative
    print(f"\n{'=' * 70}")
    print("JOURNEY'S END")
    print(f"{'=' * 70}")

    print(f"\nPath: {' → '.join(agent.path[:10])}{'...' if len(agent.path) > 10 else ''}")
    print(f"States visited: {len(set(agent.path))}")
    print(f"Insights (tunnels): {len(agent.insights)}")

    print(f"\n{'─' * 70}")
    print("NARRATOR'S REFLECTION")
    print(f"{'─' * 70}")
    print(agent.get_journey_narrative())

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Storyteller Journey")
    parser.add_argument("--book", default="heart_of_darkness")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--model", default="qwen2.5:1.5b")

    args = parser.parse_args()

    run_interactive_book_journey(
        book_key=args.book,
        max_steps=args.steps,
        llm_model=args.model
    )
