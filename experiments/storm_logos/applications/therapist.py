"""Therapist Application: Therapeutic conversation agent.

Uses the full Storm-Logos system for therapy:
- Tracks patient semantic position
- Detects defenses and irony
- Generates therapeutic responses
- Adapts approach based on feedback

Supports Ollama (local), Claude (API), and Groq (API) backends.
"""

from typing import Optional, List, Dict, Any
import os
import json
from datetime import datetime
from pathlib import Path
import requests

from ..data.models import SemanticState, Metrics, ConversationTrajectory
from ..metrics.engine import MetricsEngine
from ..feedback.engine import FeedbackEngine
from ..controller.engine import AdaptiveController
from ..semantic.dialectic import Dialectic
from ..generation.renderer import Renderer


class Therapist:
    """Therapeutic conversation agent.

    Implements the full adaptive loop for therapy:
    1. Analyze patient utterance
    2. Detect defenses and irony
    3. Compute dialectical direction
    4. Generate therapeutic response
    5. Adapt parameters based on patient reaction
    """

    # Therapy-speak phrases to avoid
    THERAPY_SPEAK = [
        "journey together", "self-compassion", "growth", "safe space",
        "I think", "I believe", "let's explore", "work through",
        "navigate", "process this", "healing", "recovery",
        "I hear you", "I understand", "that must be",
    ]

    def __init__(self,
                 model: str = 'claude',
                 base_url: str = 'http://localhost:11434',
                 api_key: str = None):
        """Initialize therapist.

        Args:
            model: 'claude' for Claude API, 'groq:model-name' for Groq API,
                   or Ollama model name (e.g., 'mistral:7b')
            base_url: Ollama base URL (only used for Ollama models)
            api_key: API key (reads from env if not provided)
        """
        self.metrics = MetricsEngine()
        self.feedback = FeedbackEngine()
        self.feedback.use_preset('therapeutic')
        self.controller = AdaptiveController(context='therapeutic')
        self.dialectic = Dialectic()

        self.model = model
        self.use_claude = model.lower().startswith('claude')
        self.use_groq = model.lower().startswith('groq:')

        if self.use_claude:
            self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._claude_client = None
        elif self.use_groq:
            self.api_key = api_key or os.environ.get('GROQ_API_KEY')
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not set")
            self.groq_model = model.split(':', 1)[1]  # e.g., 'llama-3.3-70b-versatile'
            self._groq_client = None
        else:
            self.renderer = Renderer(model=model, base_url=base_url)
            self.base_url = base_url

        self.trajectory = ConversationTrajectory()
        self._prev_irony = 0.0
        self._conversation_history = []  # For Claude context
        self._turns: List[Dict[str, Any]] = []  # Full conversation record
        self._session_start = datetime.now()

    def respond(self, patient_text: str, max_retries: int = 3) -> str:
        """Generate therapeutic response to patient.

        Args:
            patient_text: Patient's utterance
            max_retries: Maximum regeneration attempts

        Returns:
            Therapeutic response
        """
        # 1. Analyze patient
        metrics = self.metrics.measure(text=patient_text)
        state = SemanticState(
            A=metrics.A_position,
            S=metrics.S_position,
            irony=metrics.irony,
        )
        self.trajectory.add(state)

        # 2. Check for rising irony (missed patient)
        irony_delta = metrics.irony - self._prev_irony
        self._prev_irony = metrics.irony
        irony_rising = irony_delta > 0.1

        # 3. Compute errors and adapt
        errors = self.feedback.compute_errors(metrics)
        params = self.controller.adapt(errors)

        # 4. Get dialectical direction
        dial = self.dialectic.analyze(state)

        # 5. Build context for LLM
        context = self._build_context(state, dial, metrics, irony_rising)

        # 6. Generate response (with retries)
        for attempt in range(max_retries):
            response = self._generate_response(patient_text, context, params)

            # Evaluate
            score = self._evaluate_response(response, state, dial)

            if score >= 0.6:
                break

            # Add feedback for retry
            context += f"\n[Previous attempt scored {score:.2f}. Be more direct.]"

        # Record turn
        self._turns.append({
            'turn': len(self._turns) + 1,
            'patient': patient_text,
            'therapist': response,
            'state': {
                'A': state.A,
                'S': state.S,
                'tau': state.tau,
                'irony': state.irony,
            },
            'metrics': {
                'coherence': metrics.coherence,
                'tension': metrics.tension_score,
                'defenses': metrics.defenses,
            },
            'timestamp': datetime.now().isoformat(),
        })

        return response

    def _build_context(self, state: SemanticState, dial: dict,
                       metrics: Metrics, irony_rising: bool) -> str:
        """Build context for LLM prompt."""
        parts = []

        # Position
        parts.append(f"[POSITION] A={state.A:+.2f}, S={state.S:+.2f}, τ={state.tau:.2f}")

        # Defenses
        if metrics.defenses:
            parts.append(f"[DEFENSES] {', '.join(metrics.defenses)}")

        # Dialectic
        if dial:
            thesis = dial.get('thesis', {}).get('description', '')
            antithesis = dial.get('antithesis', {}).get('description', '')
            parts.append(f"[THESIS] {thesis}")
            parts.append(f"[ANTITHESIS] {antithesis}")

            intervention = dial.get('intervention', {})
            if intervention:
                direction = intervention.get('direction', '')
                parts.append(f"[DIRECTION] {direction}")

        # Warning if irony rising
        if irony_rising:
            parts.append("""
[WARNING] Patient's IRONY IS RISING. You missed them.
DO NOT offer solutions or positivity.
DO: Acknowledge exactly where they are.
Mirror the darkness BEFORE offering anything else.
""")

        return '\n'.join(parts)

    def _generate_response(self, patient_text: str, context: str,
                           params) -> str:
        """Generate response via LLM (Claude or Ollama)."""
        # Compute response length from receptivity
        receptivity = self._compute_receptivity()
        max_tokens = int(15 + 85 * receptivity)  # 15-100 tokens

        system_prompt = """You are a voice for a semantic analysis system acting as therapist.

The system provides semantic analysis of the patient. Follow the DIRECTIVES in the analysis.

Rules:
- Be present and authentic
- Avoid therapy-speak: "journey", "explore", "growth", "safe space", "I hear you"
- NO "I think", "I believe", "I feel that" statements
- Stay grounded and direct
- Match patient's emotional level
- Keep response SHORT (1-2 sentences max)

Examples of GOOD responses:
- "That joke hides something."
- "The anger is about something else."
- "You keep saying 'fine'."
- "Tell me more about the emptiness."
"""

        user_message = f"""[SEMANTIC ANALYSIS]
{context}

[PATIENT]
{patient_text}

[YOUR RESPONSE]"""

        if self.use_claude:
            return self._generate_claude(system_prompt, user_message, max_tokens)
        elif self.use_groq:
            return self._generate_groq(system_prompt, user_message, max_tokens)
        else:
            return self._generate_ollama(system_prompt, user_message, max_tokens)

    def _generate_claude(self, system: str, user: str, max_tokens: int) -> str:
        """Generate response via Claude API."""
        try:
            import anthropic

            if self._claude_client is None:
                self._claude_client = anthropic.Anthropic(api_key=self.api_key)

            response = self._claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": user}
                ]
            )

            return response.content[0].text.strip()

        except Exception as e:
            return f"[Claude error: {e}]"

    def _generate_groq(self, system: str, user: str, max_tokens: int) -> str:
        """Generate response via Groq API."""
        try:
            from groq import Groq

            if self._groq_client is None:
                self._groq_client = Groq(api_key=self.api_key)

            response = self._groq_client.chat.completions.create(
                model=self.groq_model,
                max_tokens=max_tokens,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[Groq error: {e}]"

    def _generate_ollama(self, system: str, user: str, max_tokens: int) -> str:
        """Generate response via Ollama."""
        prompt = f"{system}\n\n{user}"

        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': max_tokens,
                        'temperature': 0.7,
                    }
                },
                timeout=60,
            )

            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return "[Unable to generate response]"

        except Exception as e:
            return f"[Ollama error: {e}]"

    def _evaluate_response(self, response: str, patient_state: SemanticState,
                           dial: dict) -> float:
        """Evaluate response quality."""
        score = 1.0

        # Penalize therapy-speak
        response_lower = response.lower()
        for phrase in self.THERAPY_SPEAK:
            if phrase in response_lower:
                score -= 0.15

        # Penalize too many "I" statements
        i_count = response_lower.count(' i ')
        if i_count > 3:
            score -= 0.1 * (i_count - 3)

        # Check for appropriate length
        word_count = len(response.split())
        if word_count < 5:
            score -= 0.3
        elif word_count > 100:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _compute_receptivity(self) -> float:
        """Compute patient receptivity (affects response length)."""
        if not self.trajectory.current:
            return 0.7

        state = self.trajectory.current
        receptivity = 1.0

        # Irony reduces receptivity
        receptivity *= (1.0 - state.irony)

        # Negative A reduces receptivity
        if state.A < 0:
            receptivity *= (1.0 + state.A)

        return max(0.2, min(1.0, receptivity))

    def reset(self):
        """Reset therapist state."""
        self.trajectory.clear()
        self._prev_irony = 0.0
        self._turns = []
        self._session_start = datetime.now()
        self.controller.reset()
        self.feedback.reset()

    def get_trajectory(self) -> ConversationTrajectory:
        """Get conversation trajectory."""
        return self.trajectory

    def get_session_data(self) -> Dict[str, Any]:
        """Get full session data for export."""
        states = self.trajectory.history
        return {
            'session_id': self._session_start.strftime('%Y%m%d_%H%M%S'),
            'start_time': self._session_start.isoformat(),
            'end_time': datetime.now().isoformat(),
            'model': self.model,
            'n_turns': len(self._turns),
            'turns': self._turns,
            'summary': {
                'avg_A': sum(s.A for s in states) / len(states) if states else 0,
                'avg_S': sum(s.S for s in states) / len(states) if states else 0,
                'avg_irony': sum(s.irony for s in states) / len(states) if states else 0,
                'A_movement': states[-1].A - states[0].A if len(states) >= 2 else 0,
                'irony_movement': states[-1].irony - states[0].irony if len(states) >= 2 else 0,
            }
        }

    def save_session(self, output_dir: str = None) -> str:
        """Save session to JSON file.

        Args:
            output_dir: Directory to save to. Defaults to storm_logos/sessions/

        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'sessions'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        session_data = self.get_session_data()
        filename = f"session_{session_data['session_id']}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        return str(filepath)
