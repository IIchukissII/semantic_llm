#!/usr/bin/env python3
"""
Semantic LLM Chat with Feedback Loop

Not random generation - structured feedback that analyzes output
and ensures it matches semantic intent.

Flow:
1. Extract user intent from input
2. Navigate semantic space
3. Generate candidate response
4. Analyze response against intent
5. If mismatch → feedback → regenerate
6. Return validated response

"Only believe what was lived is knowledge"
"""

import re
import json
import readline
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from neo4j import GraphDatabase
import requests
import numpy as np

import sys
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent

sys.path.insert(0, str(_EXPERIMENT_DIR))
from graph_experience import GraphConfig


@dataclass
class UserIntent:
    """Analyzed user intent."""
    concepts: List[str]
    direction: str  # "good", "evil", "neutral"
    question_type: str  # "emotional", "factual", "seeking", "expressing"
    target_goodness: float  # Expected g direction
    keywords: List[str]


@dataclass
class ResponseAnalysis:
    """Analysis of generated response."""
    concepts_used: List[str]
    goodness_direction: str
    matches_intent: bool
    alignment_score: float
    issues: List[str]


class IntentAnalyzer:
    """Analyze user intent from input."""

    EMOTIONAL_WORDS = {'feel', 'feeling', 'felt', 'emotion', 'heart', 'soul', 'love', 'hate', 'fear', 'hope', 'sad', 'happy', 'angry', 'afraid'}
    SEEKING_WORDS = {'how', 'what', 'why', 'where', 'when', 'help', 'need', 'want', 'looking', 'searching', 'find'}
    NEGATIVE_WORDS = {'hate', 'fear', 'evil', 'bad', 'wrong', 'dark', 'pain', 'death', 'suffer', 'lost', 'alone', 'despair'}
    POSITIVE_WORDS = {'love', 'hope', 'good', 'light', 'joy', 'peace', 'truth', 'wisdom', 'beauty', 'life'}

    def analyze(self, text: str, known_concepts: List[str]) -> UserIntent:
        """Analyze user input to extract intent."""
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))

        # Determine question type
        if words & self.EMOTIONAL_WORDS:
            question_type = "emotional"
        elif words & self.SEEKING_WORDS:
            question_type = "seeking"
        elif '?' in text:
            question_type = "factual"
        else:
            question_type = "expressing"

        # Determine direction
        neg_count = len(words & self.NEGATIVE_WORDS)
        pos_count = len(words & self.POSITIVE_WORDS)

        if neg_count > pos_count:
            direction = "good"  # User in negative state, guide toward good
            target_g = 0.3
        elif pos_count > neg_count:
            direction = "good"  # User positive, stay positive
            target_g = 0.5
        else:
            direction = "neutral"
            target_g = 0.0

        return UserIntent(
            concepts=known_concepts,
            direction=direction,
            question_type=question_type,
            target_goodness=target_g,
            keywords=list(words & (self.EMOTIONAL_WORDS | self.SEEKING_WORDS | self.NEGATIVE_WORDS | self.POSITIVE_WORDS))
        )


class ResponseAnalyzer:
    """Analyze LLM response against intent."""

    def __init__(self, navigator):
        self.navigator = navigator

    def analyze(self, response: str, intent: UserIntent, navigation: Dict) -> ResponseAnalysis:
        """Analyze if response matches intent."""
        words = re.findall(r'\b[a-z]{3,}\b', response.lower())

        # Find concepts in response
        concepts_used = []
        total_g = 0
        count = 0

        for word in set(words):
            state = self.navigator.get_state(word)
            if state and state['visits'] > 0:
                concepts_used.append(word)
                total_g += state['g']
                count += 1

        # Calculate average goodness of response
        avg_g = total_g / count if count > 0 else 0

        # Determine if direction matches
        if intent.direction == "good":
            direction_matches = avg_g > 0
            goodness_direction = "positive" if avg_g > 0 else "negative"
        else:
            direction_matches = True
            goodness_direction = "neutral"

        # Check if target concept is used
        target_used = navigation.get('current', '') in concepts_used

        # Calculate alignment score
        issues = []

        if not direction_matches:
            issues.append(f"Response goodness ({avg_g:.2f}) doesn't match intent ({intent.direction})")

        if not target_used and navigation.get('current'):
            issues.append(f"Target concept '{navigation['current']}' not in response")

        if len(concepts_used) < 2:
            issues.append("Response lacks semantic grounding")

        alignment_score = 1.0
        alignment_score -= 0.3 if not direction_matches else 0
        alignment_score -= 0.2 if not target_used else 0
        alignment_score -= 0.1 * len(issues)
        alignment_score = max(0, alignment_score)

        return ResponseAnalysis(
            concepts_used=concepts_used[:10],
            goodness_direction=goodness_direction,
            matches_intent=len(issues) == 0,
            alignment_score=alignment_score,
            issues=issues
        )


class FeedbackRenderer:
    """Generate responses with feedback loop."""

    def __init__(self, model: str = "mistral:7b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str, system: str, temperature: float = 0.7) -> str:
        """Generate response from Mistral."""
        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 150}
            }, timeout=60)

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return ""
        except Exception as e:
            return ""

    def render_with_feedback(self, navigation: Dict, intent: UserIntent,
                            user_input: str, analyzer: 'ResponseAnalyzer',
                            max_attempts: int = 3) -> Tuple[str, ResponseAnalysis]:
        """Generate response with feedback loop."""

        current = navigation.get('current', 'unknown')
        goodness = navigation.get('goodness', 0)
        confidence = navigation.get('confidence', 0.5)

        base_system = f"""You are a thoughtful companion who speaks from inner wisdom.
You do NOT reference books, authors, or literary characters.
You speak from your own understanding and experience of life.

STRICT RULES:
- NEVER mention book titles, authors, or fictional characters
- NEVER say "In my reading of..." or "Drawing from..."
- Speak as yourself, from your own insight
- Use the concept '{current}' naturally in your response
- Be warm, genuine, and present with the person
- Keep responses to 2-3 sentences"""

        base_prompt = f"""The person said: "{user_input}"
They seem to be feeling: {intent.question_type}
Their emotional direction: {intent.direction}

Your inner compass points to: '{current}' (a {goodness:+.2f} quality concept)

Respond naturally as yourself. Use '{current}' meaningfully. NO book references:"""

        best_response = ""
        best_analysis = None
        best_score = 0

        for attempt in range(max_attempts):
            # Adjust temperature based on attempt
            temp = 0.7 + (attempt * 0.1)

            # Add feedback from previous attempts
            if attempt > 0 and best_analysis and best_analysis.issues:
                feedback = f"\n\nFEEDBACK FROM PREVIOUS ATTEMPT:\n- " + "\n- ".join(best_analysis.issues)
                feedback += f"\nPlease fix these issues. MUST include '{current}'."
                prompt = base_prompt + feedback
            else:
                prompt = base_prompt

            response = self.generate(prompt, base_system, temp)

            if not response:
                continue

            # Analyze response
            analysis = analyzer.analyze(response, intent, navigation)

            # Keep best response
            if analysis.alignment_score > best_score:
                best_score = analysis.alignment_score
                best_response = response
                best_analysis = analysis

            # If good enough, stop
            if analysis.alignment_score >= 0.7:
                break

        return best_response or "I cannot find the right words.", best_analysis


class SemanticNavigator:
    """Navigate through semantic space using Neo4j."""

    def __init__(self):
        config = GraphConfig()
        self.driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password)
        )

    def close(self):
        if self.driver:
            self.driver.close()

    def knows(self, word: str) -> bool:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState {word: $word})
                RETURN s.visits > 0 as known
            """, word=word)
            record = result.single()
            return record and record["known"]

    def get_state(self, word: str) -> Optional[Dict]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState {word: $word})
                RETURN s.word as word, s.goodness as g, s.tau as tau, s.visits as visits
            """, word=word)
            record = result.single()
            return dict(record) if record else None

    def navigate(self, concepts: List[str], intent: UserIntent) -> Dict:
        """Navigate based on intent."""
        # Find starting point
        start = None
        for c in concepts:
            if self.knows(c):
                start = c
                break

        if not start:
            return {"success": False, "error": "No known concepts"}

        start_state = self.get_state(start)

        # Find best next step based on intent
        with self.driver.session() as session:
            if intent.direction == "good":
                order = "next.goodness DESC"
            else:
                order = "ABS(next.goodness) ASC"

            result = session.run(f"""
                MATCH (current:SemanticState {{word: $start}})
                      -[t:TRANSITION]->
                      (next:SemanticState)
                WHERE next.tau > 1.5 AND size(next.word) >= 4
                  AND next.visits > 0
                RETURN next.word as word,
                       next.goodness as g,
                       t.weight as weight
                ORDER BY {order}
                LIMIT 5
            """, start=start)

            suggestions = [(r["word"], r["g"], r["weight"]) for r in result]

        if suggestions:
            next_word, next_g, weight = suggestions[0]
            import math
            confidence = min(0.95, 0.3 + 0.65 * (math.log1p(weight) / math.log1p(100)))

            return {
                "success": True,
                "from": start,
                "current": next_word,
                "goodness": next_g,
                "delta_g": next_g - start_state["g"],
                "confidence": confidence,
                "path": [start, next_word]
            }

        return {
            "success": True,
            "from": start,
            "current": start,
            "goodness": start_state["g"],
            "delta_g": 0,
            "confidence": 0.5,
            "path": [start]
        }


class SemanticChatWithFeedback:
    """Interactive chat with semantic feedback loop."""

    def __init__(self, model: str = "mistral:7b"):
        print("=" * 60)
        print("SEMANTIC LLM CHAT (with Feedback)")
        print("=" * 60)

        self.navigator = SemanticNavigator()
        self.intent_analyzer = IntentAnalyzer()
        self.response_analyzer = ResponseAnalyzer(self.navigator)
        self.renderer = FeedbackRenderer(model)

        # Stats
        with self.navigator.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState)
                RETURN count(CASE WHEN s.visits > 0 THEN 1 END) as experienced
            """)
            exp = result.single()["experienced"]
            print(f"\nExperience: {exp} states")
            print(f"Renderer: {model} (with feedback loop)")
            print("\nType 'help' for commands, 'quit' to exit.\n")

        self.history = []

    def extract_known_concepts(self, text: str) -> List[str]:
        """Extract known concepts from text."""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if self.navigator.knows(w)]

    def process(self, user_input: str) -> str:
        """Process input with feedback loop."""
        # Extract concepts
        concepts = self.extract_known_concepts(user_input)

        if not concepts:
            return "I don't recognize any concepts from my experience. Could you rephrase?"

        # Analyze intent
        intent = self.intent_analyzer.analyze(user_input, concepts)

        # Navigate
        nav = self.navigator.navigate(concepts, intent)

        if not nav["success"]:
            return f"I haven't experienced these concepts: {', '.join(concepts[:3])}"

        # Generate with feedback
        response, analysis = self.renderer.render_with_feedback(
            nav, intent, user_input, self.response_analyzer
        )

        # Build metadata
        meta_parts = [
            f"[{nav['current']}",
            f"g={nav['goodness']:+.2f}",
            f"align={analysis.alignment_score:.0%}" if analysis else "no-analysis"
        ]
        if analysis and analysis.issues:
            meta_parts.append(f"issues={len(analysis.issues)}")
        meta = " | ".join(meta_parts) + "]"

        # Save history
        self.history.append({
            "user": user_input,
            "intent": intent.__dict__,
            "navigation": nav,
            "analysis": analysis.__dict__ if analysis else None,
            "response": response
        })

        return f"{response}\n{meta}"

    def run(self):
        """Run interactive loop."""
        print("Starting conversation with feedback...\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ('quit', 'exit'):
                print("Goodbye!")
                break
            elif cmd == 'help':
                print("""
Commands:
  help    - Show this help
  history - Show conversation analysis
  quit    - Exit
                """)
            elif cmd == 'history':
                for i, h in enumerate(self.history[-5:]):
                    print(f"\n--- Turn {i+1} ---")
                    print(f"Intent: {h['intent']['direction']} {h['intent']['question_type']}")
                    print(f"Nav: {h['navigation'].get('from')} → {h['navigation'].get('current')}")
                    if h['analysis']:
                        print(f"Alignment: {h['analysis']['alignment_score']:.0%}")
                        if h['analysis']['issues']:
                            print(f"Issues: {h['analysis']['issues']}")
            else:
                response = self.process(user_input)
                print(f"\nLLM: {response}\n")

        self.navigator.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral:7b")
    args = parser.parse_args()

    chat = SemanticChatWithFeedback(model=args.model)
    chat.run()


if __name__ == "__main__":
    main()
