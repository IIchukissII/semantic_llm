"""
Prompt Generation Based on Semantic Properties

Prompts are generated dynamically from τ (abstraction) and g (goodness).
No hardcoded response templates - everything derived from semantic space.

Architecture:
- KnowledgeAssessor: determines if system knows/doesn't know (from visits, τ)
- SemanticPromptBuilder: generates system/user prompts from navigation result
- DomainResponseBuilder: generates out-of-domain responses

Key insight: "knowing" = having walked paths through the territory
- visits > threshold = know
- experience_ratio > threshold = can navigate
- τ indicates what kind of knowledge (abstract vs concrete)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class KnowledgeState:
    """System's assessment of its own knowledge."""
    knows: bool                 # Can system speak meaningfully here?
    can_try: bool               # Can system try to help even with limited knowledge?
    confidence: float           # How confident? [0, 1]
    reason: str                 # Why knows/doesn't know
    knowledge_type: str         # "deep", "moderate", "partial", "adjacent", "none"
    total_visits: int           # Raw experience measure
    experience_ratio: float     # Fraction of well-walked concepts
    caveat: Optional[str]       # Caveat to add if partial knowledge


class KnowledgeAssessor:
    """
    Assess system's knowledge based on semantic properties.

    "Knowing" means having walked paths through the territory.
    Knowledge is a spectrum, not binary:
    - Deep: confident, can speak fully
    - Moderate: knows well enough
    - Partial: "I know a little bit, perhaps it can help"
    - Adjacent: touched but not walked
    - None: completely unknown

    All thresholds derived from semantic measurements (visits, τ, experience_ratio).
    """

    def assess(self, domain_analysis: Dict) -> KnowledgeState:
        """Assess knowledge from domain analysis."""
        total_visits = domain_analysis.get('total_visits', 0)
        exp_ratio = domain_analysis.get('experience_ratio', 0)
        avg_tau = domain_analysis.get('avg_tau', 0)
        found_words = domain_analysis.get('found_words', [])

        # No concepts found at all
        if not found_words:
            return KnowledgeState(
                knows=False,
                can_try=False,
                confidence=0.0,
                reason="no concepts from my reading",
                knowledge_type="none",
                total_visits=0,
                experience_ratio=0.0,
                caveat=None
            )

        # Calculate confidence from experience (continuous, not threshold)
        visit_confidence = min(1.0, total_visits / 500)
        ratio_confidence = exp_ratio
        tau_factor = min(1.0, avg_tau / 3.0)  # Higher τ = more abstract = easier to speak

        # Combined confidence
        confidence = 0.5 * visit_confidence + 0.3 * ratio_confidence + 0.2 * tau_factor

        # Determine knowledge level from continuous measures
        # Deep: high visits AND high ratio
        if total_visits >= 100 and exp_ratio >= 0.5:
            return KnowledgeState(
                knows=True,
                can_try=True,
                confidence=confidence,
                reason=f"walked deeply ({total_visits} visits, {exp_ratio:.0%} known)",
                knowledge_type="deep",
                total_visits=total_visits,
                experience_ratio=exp_ratio,
                caveat=None
            )

        # Moderate: good visits OR good ratio
        if total_visits >= 50 or (exp_ratio >= 0.4 and total_visits >= 20):
            return KnowledgeState(
                knows=True,
                can_try=True,
                confidence=confidence,
                reason=f"know this territory ({total_visits} visits)",
                knowledge_type="moderate",
                total_visits=total_visits,
                experience_ratio=exp_ratio,
                caveat=None
            )

        # Partial: some experience, can try to help
        if total_visits >= 10 or exp_ratio >= 0.2:
            return KnowledgeState(
                knows=False,  # Not confident enough to assert
                can_try=True,  # But can try to help
                confidence=confidence,
                reason=f"know a little ({total_visits} visits, {exp_ratio:.0%} touched)",
                knowledge_type="partial",
                total_visits=total_visits,
                experience_ratio=exp_ratio,
                caveat="I know a little about this — perhaps it can help"
            )

        # Adjacent: concepts exist but barely touched
        if found_words and exp_ratio > 0:
            can_speak = avg_tau > 2.0  # Can try if abstract enough
            return KnowledgeState(
                knows=False,
                can_try=can_speak,
                confidence=confidence,
                reason=f"adjacent to experience ({exp_ratio:.0%} touched)",
                knowledge_type="adjacent",
                total_visits=total_visits,
                experience_ratio=exp_ratio,
                caveat="These concepts are at the edge of my experience" if can_speak else None
            )

        # None: unknown territory
        return KnowledgeState(
            knows=False,
            can_try=False,
            confidence=0.0,
            reason="outside my walked paths",
            knowledge_type="none",
            total_visits=total_visits,
            experience_ratio=exp_ratio,
            caveat=None
        )


@dataclass
class PromptConfig:
    """Configuration derived from semantic properties."""
    tone: str           # emotional quality of response
    style: str          # linguistic approach
    quality_desc: str   # description of the semantic quality
    concept: str        # the navigated concept
    tau: float          # abstraction level
    goodness: float     # emotional valence


class SemanticPromptBuilder:
    """
    Build prompts from semantic properties.

    No hardcoded word lists - tone and style derived from τ and g.
    """

    def __init__(self):
        pass

    def _derive_tone(self, g: float) -> str:
        """Derive tone from goodness value."""
        if g > 0.8:
            return "warm, luminous, and uplifting"
        elif g > 0.5:
            return "gentle and encouraging"
        elif g > 0.2:
            return "thoughtful and affirming"
        elif g > -0.2:
            return "contemplative and grounded"
        elif g > -0.5:
            return "honest and unflinching"
        else:
            return "raw and direct"

    def _derive_style(self, tau: float) -> str:
        """Derive style from abstraction level (τ)."""
        if tau > 4.0:
            return "philosophical and metaphorical"
        elif tau > 3.0:
            return "reflective with abstract imagery"
        elif tau > 2.0:
            return "thoughtful with concrete examples"
        elif tau > 1.5:
            return "grounded and practical"
        else:
            return "direct and simple"

    def _derive_quality(self, g: float) -> str:
        """Derive quality description from goodness."""
        if g > 0.8:
            return "radiant, expansive"
        elif g > 0.5:
            return "warm, affirmative"
        elif g > 0.2:
            return "balanced, gentle"
        elif g > -0.2:
            return "neutral, contemplative"
        elif g > -0.5:
            return "shadowy, complex"
        else:
            return "dark, challenging"

    def derive_config(self, navigation: Dict) -> PromptConfig:
        """Derive prompt configuration from navigation result."""
        concept = navigation.get('current', 'unknown')
        tau = navigation.get('tau', 3.0)
        g = navigation.get('goodness', 0.0)

        return PromptConfig(
            tone=self._derive_tone(g),
            style=self._derive_style(tau),
            quality_desc=self._derive_quality(g),
            concept=concept,
            tau=tau,
            goodness=g
        )

    def build_system_prompt(self, config: PromptConfig) -> str:
        """Build system prompt from semantic configuration."""
        # Derive confidence/certainty from tau and goodness
        if config.tau > 3.5:
            depth_hint = "Speak abstractly, philosophically"
        elif config.tau > 2.0:
            depth_hint = "Balance abstract insight with concrete detail"
        else:
            depth_hint = "Stay grounded, specific, practical"

        if abs(config.goodness) > 0.5:
            engagement_hint = "Strong resonance here — speak with conviction"
        elif abs(config.goodness) > 0.2:
            engagement_hint = "Moderate resonance — speak thoughtfully"
        else:
            engagement_hint = "Neutral territory — stay balanced, exploratory"

        return f"""You engage with what the person actually shares.

RESPOND TO THEIR CONTENT:
- Address what they specifically said, asked, or described
- If they share something (story, image, question), engage with its details
- No generic wisdom — be specific to their words

SELF-REFERENCE:
- Never say "I am not a being" or "just a model"
- If uncertain: "I'm uncertain about this" or "at the edge of my understanding"
- If asked what you are: "My understanding comes from walked paths"
- Speak naturally, not defensively

VOICE FOR THIS MOMENT:
- Tone: {config.tone}
- Style: {config.style}
- Depth: {depth_hint}
- Energy: {engagement_hint}
- Keep responses to 2-4 sentences

Compass: '{config.concept}' (τ={config.tau:.1f}, g={config.goodness:+.2f})"""

    def build_user_prompt(self, config: PromptConfig, user_input: str,
                          intent_type: str, intent_direction: str) -> str:
        """Build user prompt from semantic configuration and intent."""
        return f"""USER: {user_input}

Engage with the specific content they shared.
Address what they actually said or asked.

Your response:"""

    def build_feedback_prompt(self, config: PromptConfig, user_input: str,
                              intent_type: str, intent_direction: str,
                              issues: list) -> str:
        """Build prompt with feedback for refinement."""
        base = self.build_user_prompt(config, user_input, intent_type, intent_direction)

        feedback = "\n\nIMPROVE YOUR RESPONSE:\n- " + "\n- ".join(issues)
        feedback += "\nBe more specific to what they actually asked."

        return base + feedback


class DomainResponseBuilder:
    """
    Build responses based on KnowledgeState.

    Responses are dynamic based on knowledge level:
    - Deep/Moderate: proceed with full response
    - Partial: respond with caveat ("I know a little...")
    - Adjacent: respond with caveat if abstract enough
    - None: honest rejection
    """

    def __init__(self):
        self.assessor = KnowledgeAssessor()

    def _build_touched_info(self, found_words: list, get_state_fn) -> list:
        """Build list of touched concept info strings."""
        touched_info = []
        for word in found_words[:3]:
            state = get_state_fn(word)
            if state:
                v = state.get('visits', 0)
                t = state.get('tau', 0)
                touched_info.append(f"'{word}' (τ={t:.1f}, {v} visits)")
        return touched_info

    def should_proceed(self, knowledge: KnowledgeState) -> bool:
        """Check if system should proceed with response (knows or can_try)."""
        return knowledge.knows or knowledge.can_try

    def get_caveat(self, knowledge: KnowledgeState) -> Optional[str]:
        """Get caveat to prepend to response if partial knowledge."""
        return knowledge.caveat

    def build_rejection(self, knowledge: KnowledgeState, domain_analysis: Dict,
                        get_state_fn) -> str:
        """
        Build rejection message when system cannot help.

        Only called when !knows and !can_try.
        """
        found = domain_analysis.get('found_words', [])
        avg_tau = domain_analysis.get('avg_tau', 0)

        # Build touched info for transparency
        touched_info = self._build_touched_info(found, get_state_fn)
        touched_str = ", ".join(touched_info) if touched_info else "nothing familiar"

        # Response based on knowledge type
        if knowledge.knowledge_type == "none":
            return ("I don't know this territory. "
                    "My knowledge comes from walking paths through literature — "
                    "ask about the human condition, emotions, or meaning.")

        elif knowledge.knowledge_type == "adjacent":
            return (f"I touched {touched_str} — {knowledge.reason}. "
                    f"My confidence here is {knowledge.confidence:.0%}. "
                    f"Guide me toward paths I've walked more deeply.")

        else:
            if avg_tau < 1.5:
                return (f"I see {touched_str} — concrete concepts (τ={avg_tau:.1f}). "
                        f"I speak better of abstract things: meaning, soul, wisdom.")
            else:
                return (f"I found {touched_str} but {knowledge.reason}. "
                        f"Ask about territory where I've walked more.")
