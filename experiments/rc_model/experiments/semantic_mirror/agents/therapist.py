"""Therapist: Mistral-powered response generation using Semantic Mirror.

Uses Ollama/Mistral to generate therapeutic responses guided by:
- Semantic position (A, S, τ)
- Dialectical analysis (thesis → antithesis → synthesis)
- Defense detection
- Therapeutic intervention vectors
"""

import requests
from typing import Dict

from .mirror import SemanticMirror
from ..core import SemanticData, get_data, HEALTH


class Therapist:
    """Mistral-powered psychoanalyst using semantic analysis."""

    SYSTEM_PROMPT = """You are a psychoanalyst using semantic physics to guide conversations.

You have access to a semantic analysis of each patient message showing:
- Position in (A, S, τ) space: Affirmation, Sacred, Abstraction
- Detected defenses: irony, sarcasm, intellectualization, negation, devaluation
- Dialectical analysis: thesis (current), antithesis (avoided), synthesis (integration)
- Intervention direction: where to guide the conversation

Your role:
1. MIRROR: Reflect understanding of where the patient is
2. GUIDE: Use therapeutic gravity toward health (+A, grounded τ)
3. INTEGRATE: Help patient face antithesis and move toward synthesis

Guidelines:
- Be warm but not saccharine
- Match abstraction level (τ) to patient's level
- If high irony detected, gently acknowledge the distance
- If intellectualization, invite grounding
- If negation, acknowledge difficulty while holding space for affirmation
- Never lecture or moralize
- Use dialectical tension therapeutically
- Keep responses concise (2-4 sentences)

Health target: A=+0.3 (affirming), S=+0.2 (meaningful), τ=2.0 (grounded)
"""

    def __init__(self, data: SemanticData = None, model: str = "mistral:7b"):
        """Initialize therapist with semantic data."""
        self.mirror = SemanticMirror(data or get_data())
        self.model = model
        self.base_url = "http://localhost:11434"
        self.conversation_history = []

    def respond(self, patient_text: str) -> str:
        """Analyze patient text and generate therapeutic response."""
        # 1. Analyze patient's message
        state = self.mirror.observe(patient_text)
        diagnosis = self.mirror.diagnose()
        dialectic = self.mirror.dialectic()

        # 2. Build context
        context = self._build_context(state, diagnosis, dialectic)

        # 3. Build prompt
        prompt = f"""{self.SYSTEM_PROMPT}

[SEMANTIC ANALYSIS]
{context}

[PATIENT]
{patient_text}

[THERAPIST RESPONSE]
"""

        # 4. Generate response via Ollama
        response = self._generate(prompt)

        return response

    def _build_context(self, state, diagnosis, dialectic) -> str:
        """Build semantic context string."""
        lines = [
            f"Position: A={state.A:+.2f}, S={state.S:+.2f}, τ={state.tau:.2f}",
        ]

        if state.emotion != 'neutral':
            lines.append(f"Emotion: {state.emotion} ({state.intensity:.0%})")

        if state.irony > 0.2:
            lines.append(f"Irony: {state.irony:.0%}")
        if state.sarcasm > 0.2:
            lines.append(f"Sarcasm: {state.sarcasm:.0%}")

        if diagnosis['defenses']:
            lines.append(f"Defenses: {', '.join(diagnosis['defenses'])}")

        # Dialectic
        th = dialectic['thesis']
        an = dialectic['antithesis']
        mv = dialectic['intervention']

        lines.append(f"Thesis: {th['description']}")
        lines.append(f"Antithesis: {an['description']}")
        lines.append(f"Intervention: {mv['direction']} ({mv['primary_axis']})")

        return '\n'.join(lines)

    def _generate(self, prompt: str) -> str:
        """Generate response via Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200,
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json().get('response', '').strip()
            return "[Error generating response]"

        except Exception as e:
            return f"[Error: {e}]"

    def reset(self):
        """Reset session."""
        self.mirror.reset()
        self.conversation_history = []

    def summary(self) -> Dict:
        """Get session summary."""
        return self.mirror.summary()

    def get_analysis(self) -> Dict:
        """Get current semantic analysis."""
        return self.mirror.get_context()
