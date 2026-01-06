"""LLM as Navigation Agent in Semantic Space.

Instead of skeleton → LLM, we do:
  Position + Gradient + Target → LLM → Sentence → Measure → Repeat

The LLM navigates through (A, S, τ) space guided by qualitative descriptions.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import requests
import re


# Constants
KT = math.exp(-1/5)  # ≈ 0.819
LAMBDA = 0.5  # Gravity toward concrete
MU = 0.5      # Gravity toward good


@dataclass
class SemanticPosition:
    """Position in semantic space."""
    A: float  # Affirmation (-1 to +1)
    S: float  # Sacred (-1 to +1)
    tau: float  # Abstraction (0.5 to 4.0)

    def copy(self):
        return SemanticPosition(self.A, self.S, self.tau)


# Genre target centers (from THEORY)
GENRE_TARGETS = {
    'dramatic': SemanticPosition(A=0.4, S=0.3, tau=1.8),   # High A, elevated S, concrete τ
    'ironic': SemanticPosition(A=0.2, S=-0.2, tau=2.5),    # Moderate A, mundane S, mid τ
    'balanced': SemanticPosition(A=0.1, S=0.1, tau=2.8),   # Neutral A/S, abstract τ
}


def interpret_A(A: float) -> str:
    """Qualitative description of Affirmation axis."""
    if A > 0.5:
        return "strongly affirming, positive, life-embracing"
    elif A > 0.2:
        return "moderately affirming, hopeful"
    elif A > -0.2:
        return "neutral, observational"
    elif A > -0.5:
        return "skeptical, questioning, uncertain"
    else:
        return "negating, dark, pessimistic"


def interpret_S(S: float) -> str:
    """Qualitative description of Sacred axis."""
    if S > 0.4:
        return "philosophical, sacred, transcendent"
    elif S > 0.1:
        return "elevated, meaningful, significant"
    elif S > -0.1:
        return "everyday, ordinary, mundane"
    elif S > -0.4:
        return "banal, trivial, unremarkable"
    else:
        return "profane, base, crude"


def interpret_tau(tau: float) -> str:
    """Qualitative description of Abstraction axis."""
    if tau > 3.5:
        return "highly abstract, conceptual, theoretical"
    elif tau > 2.8:
        return "abstract, general, ideational"
    elif tau > 2.2:
        return "balanced abstraction, moderate specificity"
    elif tau > 1.5:
        return "concrete, specific, sensory"
    else:
        return "very concrete, visceral, immediate"


def direction_instruction(delta: float, axis: str) -> str:
    """Convert delta to movement instruction."""
    if axis == 'A':
        if delta > 0.2:
            return "become MORE AFFIRMING (embrace light, hope, goodness)"
        elif delta < -0.2:
            return "become MORE NEGATING (embrace shadow, doubt, darkness)"
        else:
            return "HOLD your affirmation level"

    elif axis == 'S':
        if delta > 0.15:
            return "ELEVATE toward sacred (philosophy, meaning, transcendence)"
        elif delta < -0.15:
            return "DESCEND toward mundane (everyday objects, ordinary life)"
        else:
            return "HOLD your sacred/mundane level"

    elif axis == 'tau':
        if delta > 0.3:
            return "become MORE ABSTRACT (ideas, concepts, generalizations)"
        elif delta < -0.3:
            return "become MORE CONCRETE (senses, objects, specific details)"
        else:
            return "HOLD your abstraction level"


def gravity_description(Q: SemanticPosition) -> str:
    """Describe the natural gravitational pull."""
    # φ = λτ - μA
    # Gravity pulls toward: lower τ (concrete) and higher A (good)

    pull_tau = "toward concrete, sensory details" if Q.tau > 2.0 else "resisting abstraction"
    pull_A = "toward affirmation and light" if Q.A < 0.3 else "holding positive ground"

    return f"Natural flow pulls you {pull_tau} and {pull_A}."


def navigation_prompt(Q: SemanticPosition, genre: str,
                      previous_sentence: Optional[str] = None) -> str:
    """Generate navigation prompt for LLM."""

    target = GENRE_TARGETS[genre]

    # Compute deltas
    delta_A = target.A - Q.A
    delta_S = target.S - Q.S
    delta_tau = target.tau - Q.tau

    # Position description
    pos_A = interpret_A(Q.A)
    pos_S = interpret_S(Q.S)
    pos_tau = interpret_tau(Q.tau)

    # Movement instructions
    move_A = direction_instruction(delta_A, 'A')
    move_S = direction_instruction(delta_S, 'S')
    move_tau = direction_instruction(delta_tau, 'tau')

    # Gravity
    gravity = gravity_description(Q)

    # Genre flavor
    genre_hints = {
        'dramatic': "Gothic intensity, emotional weight, atmospheric tension.",
        'ironic': "Kafka-like mundanity hiding unease, ordinary becoming strange.",
        'balanced': "Austen-like observation, measured wit, social clarity.",
    }

    # Build prompt
    prompt = f"""You are navigating through semantic space, writing one sentence at a time.

CURRENT POSITION:
  Emotional tone: {pos_A}
  Conceptual depth: {pos_S}
  Abstraction level: {pos_tau}

{gravity}

TARGET DIRECTION ({genre.upper()}):
  • {move_A}
  • {move_S}
  • {move_tau}

STYLE: {genre_hints[genre]}
"""

    if previous_sentence:
        prompt += f"\nPREVIOUS SENTENCE: {previous_sentence}\n"

    prompt += """
Write ONE sentence that continues naturally while moving toward the target.
Do not explain. Do not use quotes. Just write the single sentence."""

    return prompt


class OllamaNavigator:
    """LLM navigator using Ollama."""

    def __init__(self, model: str = "mistral:7b"):
        self.model = model
        self.base_url = "http://localhost:11434"

    def generate(self, prompt: str) -> str:
        """Generate one sentence from prompt."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 100,
                }
            }
        )

        if response.status_code == 200:
            text = response.json().get('response', '').strip()
            # Extract first sentence
            sentences = re.split(r'[.!?]+', text)
            if sentences and sentences[0].strip():
                return sentences[0].strip() + '.'
            return text
        return ""


def estimate_position(sentence: str, coord_dict: dict) -> SemanticPosition:
    """Estimate semantic position from sentence words.

    Simple approach: average coordinates of known words.
    """
    words = re.findall(r'\b[a-z]+\b', sentence.lower())

    A_vals, S_vals, tau_vals = [], [], []

    for word in words:
        if word in coord_dict:
            coords = coord_dict[word]
            if isinstance(coords, dict):
                A_vals.append(coords.get('A', 0))
                S_vals.append(coords.get('S', 0))
                tau_vals.append(coords.get('n', 2.5))
            else:
                A_vals.append(coords[0])
                S_vals.append(coords[1])
                tau_vals.append(coords[2])

    if not A_vals:
        return SemanticPosition(A=0, S=0, tau=2.5)

    return SemanticPosition(
        A=sum(A_vals) / len(A_vals),
        S=sum(S_vals) / len(S_vals),
        tau=sum(tau_vals) / len(tau_vals),
    )


def update_position(Q: SemanticPosition, Q_new: SemanticPosition,
                    dt: float = 0.3, decay: float = 0.05) -> SemanticPosition:
    """RC dynamics update: blend current position with new observation."""
    return SemanticPosition(
        A=Q.A + dt * (Q_new.A - Q.A) - decay * Q.A,
        S=Q.S + dt * (Q_new.S - Q.S) - decay * Q.S,
        tau=Q.tau + dt * (Q_new.tau - Q.tau),
    )


def navigate_text(genre: str, n_sentences: int = 5,
                  coord_dict: Optional[dict] = None,
                  verbose: bool = True) -> Tuple[str, List[SemanticPosition]]:
    """Navigate through semantic space, generating text.

    Returns:
        Tuple of (generated_text, trajectory)
    """
    # Initialize near genre center with some noise
    import random
    target = GENRE_TARGETS[genre]
    Q = SemanticPosition(
        A=target.A + random.gauss(0, 0.2),
        S=target.S + random.gauss(0, 0.2),
        tau=target.tau + random.gauss(0, 0.3),
    )

    navigator = OllamaNavigator()
    sentences = []
    trajectory = [Q.copy()]

    if verbose:
        print(f"\n{'='*60}")
        print(f"NAVIGATING: {genre.upper()}")
        print(f"{'='*60}")
        print(f"Start: A={Q.A:.2f}, S={Q.S:.2f}, τ={Q.tau:.2f}")
        print(f"Target: A={target.A:.2f}, S={target.S:.2f}, τ={target.tau:.2f}")

    prev_sentence = None

    for i in range(n_sentences):
        # Generate prompt
        prompt = navigation_prompt(Q, genre, prev_sentence)

        if verbose:
            print(f"\n[Step {i+1}]")

        # Generate sentence
        sentence = navigator.generate(prompt)
        sentences.append(sentence)

        if verbose:
            print(f"  Generated: {sentence[:80]}...")

        # Estimate new position (if we have coordinates)
        if coord_dict:
            Q_measured = estimate_position(sentence, coord_dict)
            Q = update_position(Q, Q_measured)

            if verbose:
                print(f"  Position: A={Q.A:.2f}, S={Q.S:.2f}, τ={Q.tau:.2f}")
        else:
            # Without coordinates, simulate movement toward target
            Q = SemanticPosition(
                A=Q.A + 0.1 * (target.A - Q.A),
                S=Q.S + 0.1 * (target.S - Q.S),
                tau=Q.tau + 0.1 * (target.tau - Q.tau),
            )

        trajectory.append(Q.copy())
        prev_sentence = sentence

    return ' '.join(sentences), trajectory


def load_coordinates() -> dict:
    """Load word coordinates from JSON file."""
    import json
    from pathlib import Path

    # Navigate from llm_navigator/ -> experiments/ -> rc_model/ -> .. -> meaning_chain/
    base = Path(__file__).parent.parent.parent.parent  # semantic_llm/experiments/
    coord_path = base / "meaning_chain/data/derived_coordinates.json"

    try:
        with open(coord_path) as f:
            data = json.load(f)
        return data.get('coordinates', {})
    except Exception as e:
        print(f"Warning: Could not load coordinates from {coord_path}: {e}")
        return {}


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    coord_dict = load_coordinates()
    print(f"Loaded {len(coord_dict)} word coordinates")

    for genre in ['dramatic', 'ironic', 'balanced']:
        text, trajectory = navigate_text(
            genre=genre,
            n_sentences=4,
            coord_dict=coord_dict,
            verbose=True
        )

        print(f"\n{'─'*60}")
        print(f"FINAL TEXT ({genre}):")
        print(f"{'─'*60}")
        print(text)

        print(f"\nTrajectory:")
        for i, pos in enumerate(trajectory):
            print(f"  {i}: A={pos.A:.2f}, S={pos.S:.2f}, τ={pos.tau:.2f}")
