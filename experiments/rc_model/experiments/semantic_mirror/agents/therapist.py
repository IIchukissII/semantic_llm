"""Therapist: Mistral-powered response generation using Semantic Mirror.

Uses Ollama/Mistral to generate therapeutic responses guided by:
- Semantic position (A, S, τ)
- Dialectical analysis (thesis → antithesis → synthesis)
- Defense detection
- Therapeutic intervention vectors

FEEDBACK LOOP: Analyzes generated response and regenerates if not moving toward health.
"""

import requests
from typing import Dict, Tuple
from dataclasses import dataclass

from .mirror import SemanticMirror
from ..core import SemanticData, get_data, HEALTH
from ..core.physics import gravity_potential, resistance, therapeutic_vector, KT
from ..detection import SemanticDetector


@dataclass
class ResponseEvaluation:
    """Evaluation of therapist response quality."""
    text: str
    A: float
    S: float
    tau: float
    toward_health: bool  # Does response move toward health?
    score: float  # Quality score (higher = better)
    feedback: str  # Why rejected (if rejected)


class Therapist:
    """Mistral-powered psychoanalyst with semantic feedback loop."""

    # Thresholds for acceptable response
    MIN_A = -0.1  # Response should not be too negative
    MIN_S = -0.3  # Allow some mundane grounding
    MAX_TAU = 3.5  # Not too abstract
    MIN_SCORE = 0.3  # Minimum quality score

    # Therapy-speak to AVOID (patient explicitly criticizes these)
    THERAPY_SPEAK = [
        "journey together",
        "our journey",
        "this journey",
        "self-compassion",
        "growth",
        "strategies",
        "navigate",
        "explore together",
        "safe space",
        "let's delve",
        "delve deeper",
        "unpack",
        "process this",
        "work through",
        "tools to",
        "one step at a time",
        "it's okay to",
        "it's natural to",
        "it's understandable",
        "I hear you",
        "I appreciate your",
        "thank you for sharing",
        # I-statements (therapist is not a guru)
        "I believe",
        "I think",
        "I feel that",
        "I want you to",
        "I encourage you",
        "I understand",
        "I sense",
        "I see that",
        "I can see",
    ]

    # Max "I" count in response (therapist is not guru)
    MAX_I_COUNT = 2

    SYSTEM_PROMPT = """You are a voice for a semantic analysis system. Follow the DIRECTIVES exactly.

The system tells you:
- THESIS: what patient shows
- ANTITHESIS: what patient hides
- YOUR MOVE: what direction to go

JUST DO WHAT THE DIRECTIVES SAY.

Rules:
- 1 sentence maximum
- No "You seem", no "It sounds like"
- No therapy words: journey, explore, growth, safe
- No "I" statements
- Be DIRECT. Be SHORT.

Examples of GOOD responses:
- "That joke."
- "What happened?"
- "The anger is about something else."
- "You keep saying 'fine'."
"""

    def __init__(self, data: SemanticData = None, model: str = "mistral:7b",
                 max_retries: int = 3, verbose: bool = False):
        """Initialize therapist with semantic data."""
        self.data = data or get_data()
        self.mirror = SemanticMirror(self.data)
        self.detector = SemanticDetector(self.data)
        self.model = model
        self.base_url = "http://localhost:11434"
        self.max_retries = max_retries
        self.verbose = verbose
        self.conversation_history = []
        self.last_evaluation = None
        self.prev_irony = 0.0  # Track irony trend
        self.missed_patient = False  # Flag: did we miss them?
        self._current_response_length = 50  # Default
        self._current_receptivity = 0.5  # Default

    def respond(self, patient_text: str) -> str:
        """Analyze patient text and generate therapeutic response with feedback loop."""
        # 1. Analyze patient's message
        patient_state = self.mirror.observe(patient_text)
        diagnosis = self.mirror.diagnose()
        dialectic = self.mirror.dialectic()

        # 2. Detect if we MISSED the patient (irony rising = closing off)
        irony_delta = patient_state.irony - self.prev_irony
        self.missed_patient = irony_delta > 0.1  # Rising irony = we missed them

        if self.verbose and self.missed_patient:
            print(f"  ⚠ IRONY RISING: {self.prev_irony:.0%} → {patient_state.irony:.0%} (patient closing off)")

        # 3. Build context with dialectical awareness
        context = self._build_context(patient_state, diagnosis, dialectic)
        warning = self._build_warning(patient_state, irony_delta)

        # 4. Generate with feedback loop
        best_response = None
        best_score = -1.0

        for attempt in range(self.max_retries):
            # Build prompt (with feedback if retry)
            feedback_text = ""
            if attempt > 0 and self.last_evaluation:
                feedback_text = f"\n[FEEDBACK ON PREVIOUS ATTEMPT]\n{self.last_evaluation.feedback}\nPlease regenerate with this guidance.\n"

            prompt = f"""{self.SYSTEM_PROMPT}
{warning}
[SEMANTIC ANALYSIS]
{context}
{feedback_text}
[PATIENT]
{patient_text}

[THERAPIST RESPONSE]
"""

            # Generate
            response = self._generate(prompt, temperature=0.7 + attempt * 0.1)

            # Evaluate
            evaluation = self._evaluate_response(response, patient_state, dialectic)
            self.last_evaluation = evaluation

            if self.verbose:
                print(f"  [Attempt {attempt+1}] A={evaluation.A:+.2f}, S={evaluation.S:+.2f}, "
                      f"τ={evaluation.tau:.2f}, score={evaluation.score:.2f}, "
                      f"receptivity={self._current_receptivity:.0%}, tokens={self._current_response_length}")

            # Accept if good enough
            if evaluation.toward_health and evaluation.score >= self.MIN_SCORE:
                self.prev_irony = patient_state.irony
                return response

            # Track best
            if evaluation.score > best_score:
                best_score = evaluation.score
                best_response = response

        # Update irony tracking
        self.prev_irony = patient_state.irony

        # Return best attempt if none passed threshold
        return best_response or response

    def _build_warning(self, state, irony_delta: float) -> str:
        """Build warning if we missed the patient."""
        if irony_delta > 0.1:
            return """
[⚠ CRITICAL WARNING]
Patient's IRONY IS RISING. This means your previous response MISSED THEM.
They are CLOSING OFF because they don't feel heard.

DO NOT:
- Offer solutions or positivity
- Use metaphors about light, growth, or improvement
- Redirect to something "constructive"
- Try to fix or resolve their pain

DO:
- ACKNOWLEDGE exactly where they are
- If they say "emptiness" — sit with emptiness, don't fill it
- If they say "destruction" — explore destruction, don't redirect
- Mirror their darkness BEFORE offering anything else
- Hold the tension. Don't resolve prematurely.
"""
        elif state.irony > 0.3:
            return """
[CAUTION]
Patient is using IRONY AS DISTANCE. They may not feel safe to be direct.
Acknowledge the irony gently without confronting it directly.
Create safety before depth.
"""
        elif 'negation' in str(self.mirror.diagnose().get('defenses', [])):
            return """
[DIALECTICAL GUIDANCE]
Patient is in NEGATION. Thesis is dark.
Do NOT jump to antithesis (light/hope).
Stay with thesis. Acknowledge the darkness.
Synthesis comes later, through held tension.
"""
        return ""

    def _evaluate_response(self, response: str, patient_state, dialectic) -> ResponseEvaluation:
        """Evaluate therapist response using semantic analysis."""
        # Analyze response
        resp_state = self.detector.detect(response)

        # Compute quality score
        score = 0.0
        feedback_parts = []

        # 1. Check A (should be affirming, not negative)
        if resp_state.A >= self.MIN_A:
            score += 0.3
        else:
            feedback_parts.append(f"Response too negative (A={resp_state.A:.2f}). Be more affirming.")

        # 2. Check if moving toward health on A axis
        if resp_state.A > patient_state.A:
            score += 0.2
        elif resp_state.A < patient_state.A - 0.1:
            feedback_parts.append("Response should model positive affirmation.")

        # 3. Check τ (should not be too abstract - therapist grounds)
        if resp_state.tau <= self.MAX_TAU:
            score += 0.2
        else:
            feedback_parts.append(f"Too abstract (τ={resp_state.tau:.2f}). Use concrete, grounded language.")

        # 4. Check S (meaningful but not preachy)
        if resp_state.S >= self.MIN_S:
            score += 0.2
        else:
            feedback_parts.append("Response feels too trivial. Add some depth.")

        # 5. Check irony/sarcasm (therapist should not be ironic)
        if resp_state.irony < 0.2 and resp_state.sarcasm < 0.2:
            score += 0.1
        else:
            feedback_parts.append("Avoid irony or sarcasm in therapeutic response.")

        # 6. Check for therapy-speak (penalize cliché phrases)
        therapy_speak_found = self._detect_therapy_speak(response)
        if therapy_speak_found:
            score -= 0.2 * len(therapy_speak_found)  # Penalize each phrase
            feedback_parts.append(
                f"AVOID therapy-speak: {', '.join(therapy_speak_found[:3])}. "
                "Use direct, concrete language instead."
            )

        # 7. Check "I" count (therapist is not a guru)
        i_count = self._count_i_statements(response)
        if i_count > self.MAX_I_COUNT:
            score -= 0.15 * (i_count - self.MAX_I_COUNT)
            feedback_parts.append(
                f"Too many 'I' statements ({i_count}). Focus on patient, not yourself. "
                "Use observations and questions instead of 'I think/believe/feel'."
            )

        # Determine if moving toward health overall
        toward_health = (
            resp_state.A >= self.MIN_A and
            resp_state.tau <= self.MAX_TAU and
            resp_state.irony < 0.3 and
            len(therapy_speak_found) == 0 and  # No therapy-speak!
            i_count <= self.MAX_I_COUNT  # Not ego-centered
        )

        feedback = " ".join(feedback_parts) if feedback_parts else "Good response."

        return ResponseEvaluation(
            text=response,
            A=resp_state.A,
            S=resp_state.S,
            tau=resp_state.tau,
            toward_health=toward_health,
            score=score,
            feedback=feedback
        )

    def _compute_receptivity(self, state, diagnosis) -> float:
        """Compute patient receptivity (0-1). Higher = more open to longer response."""
        # Base receptivity
        receptivity = 1.0

        # Irony reduces receptivity (closing off)
        receptivity *= (1.0 - state.irony)

        # Resistance reduces receptivity
        resist = diagnosis.get('resistance', 0.0)
        receptivity *= (1.0 - resist * 0.5)

        # Very negative A reduces receptivity (defensive)
        if state.A < 0:
            receptivity *= (1.0 + state.A)  # A=-1 → 0, A=0 → 1

        # High tau (abstract) = less grounded = less receptive
        if state.tau > 2.5:
            receptivity *= 0.8

        return max(0.1, min(1.0, receptivity))

    def _compute_response_length(self, receptivity: float) -> int:
        """Compute optimal response length (tokens) from receptivity."""
        # receptivity 0.1 → 20 tokens (very short: "That joke.")
        # receptivity 0.5 → 50 tokens (medium)
        # receptivity 1.0 → 100 tokens (can elaborate)
        min_tokens = 15
        max_tokens = 100
        return int(min_tokens + receptivity * (max_tokens - min_tokens))

    def _build_context(self, state, diagnosis, dialectic) -> str:
        """Build context from PHYSICS - computed parameters, not descriptions."""

        # ═══════════════════════════════════════════════════════════
        # COMPUTE PHYSICS
        # ═══════════════════════════════════════════════════════════

        # 1. Gravity potential: φ = λτ - μA
        phi = gravity_potential(state)

        # 2. Therapeutic vector: direction toward health
        vec = therapeutic_vector(state, HEALTH)
        vec_A, vec_S, vec_tau = vec

        # 3. Receptivity: how open is patient?
        receptivity = self._compute_receptivity(state, diagnosis)
        response_length = self._compute_response_length(receptivity)

        # 4. Distance to health
        dist = ((state.A - HEALTH.A)**2 +
                (state.S - HEALTH.S)**2 +
                (state.tau - HEALTH.tau)**2) ** 0.5

        # ═══════════════════════════════════════════════════════════
        # DETERMINE ACTION FROM PHYSICS
        # ═══════════════════════════════════════════════════════════

        # Which axis needs most movement?
        primary_axis = 'A' if abs(vec_A) >= max(abs(vec_S), abs(vec_tau)) else \
                       'S' if abs(vec_S) >= abs(vec_tau) else 'tau'

        # Direction on primary axis
        if primary_axis == 'A':
            if vec_A > 0:
                action = "OPEN toward affirmation (current A is low)"
            else:
                action = "HOLD in current affirmation"
        elif primary_axis == 'S':
            if vec_S > 0:
                action = "MOVE toward meaning (current S is low)"
            else:
                action = "STAY in mundane (S is fine)"
        else:  # tau
            if vec_tau < 0:
                action = "GROUND - bring down from abstraction"
            else:
                action = "LET abstract if needed"

        # ═══════════════════════════════════════════════════════════
        # BUILD DIRECTIVE
        # ═══════════════════════════════════════════════════════════

        # Store for use in generation
        self._current_response_length = response_length
        self._current_receptivity = receptivity

        lines = [
            f"PHYSICS:",
            f"  Position: A={state.A:+.2f}, S={state.S:+.2f}, τ={state.tau:.1f}",
            f"  Potential: φ={phi:.2f}",
            f"  Receptivity: {receptivity:.0%}",
            f"  Response length: {response_length} tokens",
            f"",
            f"ACTION: {action}",
        ]

        # Add warnings based on receptivity
        if receptivity < 0.5:
            lines.append("⚠ LOW RECEPTIVITY - keep response VERY short (1-3 words)")
        elif receptivity < 0.7:
            lines.append("Keep response short (1 sentence)")

        # Add irony warning if needed
        if state.irony > 0.3:
            lines.append("HIGH IRONY - just name what you see, no elaboration")

        # Add dialectic
        th = dialectic['thesis']
        an = dialectic['antithesis']
        lines.append(f"THESIS: {th['description']}")
        lines.append(f"ANTITHESIS: {an['description']}")

        # Length-specific instruction
        if response_length < 30:
            lines.append("RESPOND: 1-5 words only.")
        elif response_length < 50:
            lines.append("RESPOND: 1 short sentence.")
        else:
            lines.append("RESPOND: 1-2 sentences.")

        return '\n'.join(lines)

    def _detect_therapy_speak(self, text: str) -> list:
        """Detect therapy-speak phrases in response."""
        text_lower = text.lower()
        found = []
        for phrase in self.THERAPY_SPEAK:
            if phrase.lower() in text_lower:
                found.append(f'"{phrase}"')
        return found

    def _count_i_statements(self, text: str) -> int:
        """Count 'I' statements in response (therapist should not be ego-centered)."""
        import re
        # Count standalone "I" as word (not in "it", "in", etc.)
        # Matches: "I ", "I'", "I," at word boundaries
        matches = re.findall(r'\bI\b', text)
        return len(matches)

    def _generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response via Ollama. Length determined by receptivity."""
        # Use computed response length
        num_tokens = self._current_response_length

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_tokens,
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
