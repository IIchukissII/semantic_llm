#!/usr/bin/env python3
"""
Semantic LLM Chat with Neo4j Experience

The semantic LLM navigates through semantic space based on experience.
Mistral renders the navigation into natural language.

Architecture:
- Neo4j: Stores semantic space + experience (walked paths)
- Navigation: Find paths through experienced territory
- Mistral: Renders semantic navigation into human language

"Only believe what was lived is knowledge"
"""

import re
import json
import readline  # For input history
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from neo4j import GraphDatabase
import requests

import sys
_THIS_FILE = Path(__file__).resolve()
_EXPERIMENT_DIR = _THIS_FILE.parent

sys.path.insert(0, str(_EXPERIMENT_DIR))
from graph_experience import GraphConfig


class MistralRenderer:
    """Render semantic navigation into natural language using Mistral."""

    def __init__(self, model: str = "mistral:7b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def render(self, navigation: Dict, user_input: str) -> str:
        """Render navigation result into natural language."""
        system = f"""You are a wise entity that has read many classic books.
You speak from lived experience, not abstract knowledge.
Your current position in semantic space: '{navigation.get('current', 'unknown')}'
Goodness: {navigation.get('goodness', 0):+.2f} (positive = good, negative = bad)
Confidence: {navigation.get('confidence', 0):.0%}

Respond thoughtfully in 2-3 sentences.
If confidence is low, express uncertainty.
Reference your reading experience when relevant."""

        prompt = f"""User said: "{user_input}"

You navigated from '{navigation.get('from', '?')}' to '{navigation.get('current', '?')}'
Path: {' → '.join(navigation.get('path', []))}
Goodness change: {navigation.get('delta_g', 0):+.2f}

Respond naturally, incorporating the concept '{navigation.get('current', '')}':"""

        try:
            response = requests.post(self.url, json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 150}
            }, timeout=60)

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"[Mistral error: {response.status_code}]"
        except Exception as e:
            return f"[Mistral error: {e}]"


class SemanticNavigator:
    """Navigate through semantic space using Neo4j experience."""

    def __init__(self):
        config = GraphConfig()
        self.driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password)
        )
        self.current_position = None

    def close(self):
        if self.driver:
            self.driver.close()

    def knows(self, word: str) -> bool:
        """Check if word is in experience."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState {word: $word})
                RETURN s.visits > 0 as known
            """, word=word)
            record = result.single()
            return record and record["known"]

    def get_state(self, word: str) -> Optional[Dict]:
        """Get semantic state from Neo4j."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState {word: $word})
                RETURN s.word as word, s.goodness as g, s.tau as tau, s.visits as visits
            """, word=word)
            record = result.single()
            if record:
                return dict(record)
            return None

    def find_path(self, from_word: str, to_word: str, max_len: int = 5) -> List[str]:
        """Find path through experience."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (start:SemanticState {word: $from_word}),
                      (end:SemanticState {word: $to_word}),
                      path = shortestPath((start)-[:TRANSITION*1..""" + str(max_len) + """]->(end))
                RETURN [n in nodes(path) | n.word] as words
            """, from_word=from_word, to_word=to_word)
            record = result.single()
            return record["words"] if record else []

    def suggest_next(self, current: str, goal: str = "good", top_k: int = 5) -> List[Tuple[str, float]]:
        """Suggest next steps from current position."""
        with self.driver.session() as session:
            # Get transitions sorted by goodness direction + weight
            if goal == "good":
                order = "next.goodness - current.goodness + log(t.weight)/10 DESC"
            else:
                order = "current.goodness - next.goodness + log(t.weight)/10 DESC"

            result = session.run(f"""
                MATCH (current:SemanticState {{word: $current}})
                      -[t:TRANSITION]->
                      (next:SemanticState)
                WHERE next.tau > 1.5 AND size(next.word) >= 4
                RETURN next.word as word,
                       next.goodness as g,
                       t.weight as weight
                ORDER BY {order}
                LIMIT $top_k
            """, current=current, top_k=top_k)

            return [(r["word"], r["g"]) for r in result]

    def navigate(self, user_concepts: List[str], intent: str = "good") -> Dict:
        """
        Navigate semantic space based on user input.

        Returns navigation result with path, current position, confidence.
        """
        # Find starting point from user concepts
        start = None
        for concept in user_concepts:
            if self.knows(concept):
                start = concept
                break

        if not start:
            # Try to find any known concept close to user input
            return {
                "success": False,
                "error": "No known concepts in input",
                "user_concepts": user_concepts
            }

        # Get current state
        start_state = self.get_state(start)

        # Suggest next step
        suggestions = self.suggest_next(start, intent)

        if suggestions:
            next_word, next_g = suggestions[0]
            next_state = self.get_state(next_word)

            # Calculate confidence based on transition weight
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:SemanticState {word: $from})-[t:TRANSITION]->(b:SemanticState {word: $to})
                    RETURN t.weight as weight
                """, **{"from": start, "to": next_word})
                record = result.single()
                weight = record["weight"] if record else 1

            # Confidence based on weight (more walks = more confident)
            import math
            confidence = min(0.95, 0.3 + 0.65 * (math.log1p(weight) / math.log1p(100)))

            return {
                "success": True,
                "from": start,
                "current": next_word,
                "path": [start, next_word],
                "goodness": next_g,
                "delta_g": next_g - start_state["g"],
                "confidence": confidence,
                "suggestions": suggestions[:3]
            }
        else:
            return {
                "success": True,
                "from": start,
                "current": start,
                "path": [start],
                "goodness": start_state["g"],
                "delta_g": 0,
                "confidence": 0.5,
                "suggestions": []
            }


class SemanticChat:
    """Interactive chat with semantic LLM."""

    def __init__(self, model: str = "mistral:7b"):
        print("=" * 60)
        print("SEMANTIC LLM CHAT")
        print("=" * 60)

        self.navigator = SemanticNavigator()
        self.renderer = MistralRenderer(model)
        self.history = []

        # Check experience
        with self.navigator.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState)
                RETURN count(CASE WHEN s.visits > 0 THEN 1 END) as experienced,
                       sum(s.visits) as total_visits
            """)
            stats = result.single()
            print(f"\nExperience: {stats['experienced']} states, {stats['total_visits']} visits")

        print(f"Renderer: {model}")
        print("\nType 'help' for commands, 'quit' to exit.\n")

    def extract_concepts(self, text: str) -> List[str]:
        """Extract semantic concepts from text."""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        # Filter to known words
        known = []
        for w in words:
            if self.navigator.knows(w):
                known.append(w)
        return known

    def process_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        # Extract concepts
        concepts = self.extract_concepts(user_input)

        if not concepts:
            return "I don't recognize any concepts in your message. Could you rephrase?"

        # Determine intent
        negative = {'hate', 'fear', 'evil', 'bad', 'wrong', 'dark', 'pain', 'death'}
        intent = "evil" if any(c in negative for c in concepts) else "good"

        # Navigate
        nav = self.navigator.navigate(concepts, intent)

        if not nav["success"]:
            return f"I haven't experienced concepts like {', '.join(concepts[:3])}."

        # Render with Mistral
        response = self.renderer.render(nav, user_input)

        # Add metadata
        meta = f"\n[{nav['current']} | g={nav['goodness']:+.2f} | conf={nav['confidence']:.0%}]"

        # Save to history
        self.history.append({
            "user": user_input,
            "concepts": concepts,
            "navigation": nav,
            "response": response
        })

        return response + meta

    def show_status(self):
        """Show current status."""
        with self.navigator.driver.session() as session:
            result = session.run("""
                MATCH (s:SemanticState)
                RETURN count(CASE WHEN s.visits > 0 THEN 1 END) as experienced,
                       sum(s.visits) as total_visits,
                       count(s) as total
            """)
            stats = result.single()

            result2 = session.run("MATCH ()-[t:TRANSITION]->() RETURN count(t) as transitions")
            trans = result2.single()["transitions"]

        print(f"""
Status:
  Semantic space: {stats['total']} states
  Experienced: {stats['experienced']} ({100*stats['experienced']/stats['total']:.1f}%)
  Total visits: {stats['total_visits']}
  Transitions: {trans}
  Conversation turns: {len(self.history)}
        """)

    def query_word(self, word: str):
        """Query a specific word."""
        state = self.navigator.get_state(word)
        if state:
            suggestions = self.navigator.suggest_next(word, "good", 5)
            print(f"""
Word: {state['word']}
  Goodness: {state['g']:+.3f}
  Tau: {state['tau']:.2f}
  Visits: {state['visits']}
  Suggestions: {[s[0] for s in suggestions]}
            """)
        else:
            print(f"'{word}' not in semantic space")

    def run(self):
        """Run interactive chat loop."""
        print("Starting conversation...\n")

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
  help         - Show this help
  status       - Show current status
  query <word> - Query a word
  path <a> <b> - Find path between words
  quit         - Exit
                """)
            elif cmd == 'status':
                self.show_status()
            elif cmd.startswith('query '):
                word = cmd[6:].strip()
                self.query_word(word)
            elif cmd.startswith('path '):
                parts = cmd[5:].strip().split()
                if len(parts) >= 2:
                    path = self.navigator.find_path(parts[0], parts[1])
                    if path:
                        print(f"Path: {' → '.join(path)}")
                    else:
                        print("No path found")
            else:
                response = self.process_input(user_input)
                print(f"\nLLM: {response}\n")

        self.navigator.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic LLM Chat")
    parser.add_argument("--model", default="mistral:7b", help="Ollama model")
    args = parser.parse_args()

    chat = SemanticChat(model=args.model)
    chat.run()


if __name__ == "__main__":
    main()
