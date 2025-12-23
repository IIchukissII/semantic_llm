#!/usr/bin/env python3
"""
Semantic LLM Chat: Interactive conversation with an experienced agent.

The agent has read books and gained experience in semantic space.
It uses this experience to navigate and respond.

Usage:
    python chat.py                    # Default experience
    python chat.py --books divine_comedy crime_punishment
    python chat.py --load experience.json

"Only believe what was lived is knowledge"
"""

import sys
import re
import json
import argparse
import readline  # For input history
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

# Path setup
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_EXPERIENCE_KNOWLEDGE = _APP_DIR.parent

sys.path.insert(0, str(_EXPERIENCE_KNOWLEDGE))

from layers.core import Wholeness, Experience, ExperiencedAgent

# Try to import Ollama renderer
try:
    import requests
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Gutenberg books
GUTENBERG = Path("/home/chukiss/text_project/data/gutenberg")
BOOKS = {
    "divine_comedy": GUTENBERG / "Alighieri, Dante - The Divine Comedy.txt",
    "crime_punishment": GUTENBERG / "Dostoevsky, Fyodor - Crime and Punishment.txt",
    "heart_darkness": GUTENBERG / "Conrad, Joseph - Heart of Darkness.txt",
    "metamorphosis": GUTENBERG / "Kafka, Franz - Metamorphosis.txt",
    "jane_eyre": GUTENBERG / "Bronte, Charlotte - Jane Eyre.txt",
    "moby_dick": GUTENBERG / "Melville, Herman - Moby Dick.txt",
    "pride_prejudice": GUTENBERG / "Austen, Jane - Pride and Prejudice.txt",
}


class OllamaChat:
    """Chat with Ollama LLM."""

    def __init__(self, model: str = "mistral:7b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str, system: str = None) -> str:
        """Generate response from Ollama."""
        if not HAS_OLLAMA:
            return "[Ollama not available]"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 150,
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(self.url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"[Error: {response.status_code}]"
        except Exception as e:
            return f"[Error: {e}]"


class SemanticChat:
    """
    Interactive chat with semantic-LLM that has experience.
    """

    def __init__(self, books: List[str] = None, model: str = "mistral:7b"):
        print("=" * 60)
        print("SEMANTIC LLM CHAT")
        print("=" * 60)

        # Load wholeness
        self.wholeness = Wholeness()

        # Create experienced agent
        self.agent = ExperiencedAgent(self.wholeness, believe=0.6, name="SemanticLLM")

        if books:
            self._read_books(books)

        # LLM for natural language
        self.llm = OllamaChat(model)

        # Conversation state
        self.current_concept = None
        self.history = []

        print(f"\nAgent: {self.agent}")
        print(f"Model: {model}")
        print("\nType 'help' for commands, 'quit' to exit.\n")

    def _read_books(self, book_keys: List[str]):
        """Read books to gain experience."""
        print("\nGaining experience from books...")

        for key in book_keys:
            if key not in BOOKS:
                print(f"  Unknown book: {key}")
                continue

            path = BOOKS[key]
            if not path.exists():
                print(f"  Not found: {path}")
                continue

            print(f"  Reading: {key}...")

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            # Skip header/footer
            text = text[len(text)//20:-len(text)//20]
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())

            self.agent.read(words)

        print(f"  Experience: {self.agent.experience}")

    def extract_concepts(self, text: str) -> List[str]:
        """Extract semantic concepts from text."""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w in self.wholeness]

    def find_response_concept(self, user_concepts: List[str], intent: str = "good") -> Tuple[str, float]:
        """
        Find best concept to respond with based on user input.

        Uses experience to navigate toward intent.
        """
        if not user_concepts:
            return None, 0.0

        # Start from user's most significant concept
        best_start = None
        best_significance = -1

        for concept in user_concepts:
            if self.agent.experience.knows(concept):
                state = self.wholeness.states[concept]
                sig = abs(state.goodness) + state.tau * 0.1
                if sig > best_significance:
                    best_significance = sig
                    best_start = concept

        if best_start is None:
            # Try to tunnel to one of user's concepts
            for concept in user_concepts:
                can, prob = self.agent.can_tunnel(concept)
                if can and prob > 0.2:
                    self.agent.experience.visit(concept)
                    best_start = concept
                    break

        if best_start is None:
            return None, 0.0

        # Get suggestions from this concept
        suggestions = self.agent.suggest_next(best_start, intent)

        if suggestions:
            # Filter out function words and very short words
            # Use semantic properties: prefer content words (tau > 1.5, has cloud)
            filtered = []
            for word, score in suggestions:
                if len(word) < 4:
                    continue
                state = self.wholeness.states.get(word)
                if state and state.tau > 1.5:  # Content words have higher tau
                    filtered.append((word, score))

            if filtered:
                return filtered[0]
            # Fallback: filter just by length
            filtered = [(w, s) for w, s in suggestions if len(w) >= 4]
            if filtered:
                return filtered[0]
            return suggestions[0]

        return best_start, 0.0

    def generate_response(self, user_input: str) -> str:
        """
        Generate response to user input.

        1. Extract concepts from user input
        2. Navigate semantic space based on experience
        3. Use LLM to render natural language
        """
        # Extract concepts
        user_concepts = self.extract_concepts(user_input)

        if not user_concepts:
            return "I don't recognize any concepts in your message. Could you rephrase?"

        # Determine intent (simple heuristic)
        negative_words = {'hate', 'fear', 'evil', 'bad', 'wrong', 'dark', 'pain', 'suffer'}
        has_negative = any(w in negative_words for w in user_concepts)
        intent = "evil" if has_negative else "good"

        # Find response concept
        response_concept, score = self.find_response_concept(user_concepts, intent)

        if response_concept is None:
            # No experience with these concepts
            return f"I haven't experienced concepts like {', '.join(user_concepts[:3])}. I cannot speak of what I haven't lived."

        # Get semantic info
        state = self.wholeness.states.get(response_concept)
        if state:
            g = state.goodness
            tau = state.tau
        else:
            g, tau = 0, 2

        # Check navigation confidence
        if user_concepts:
            conf = self.agent.navigation_confidence(user_concepts[0], response_concept)
        else:
            conf = 0.5

        # Build prompt for LLM
        system_prompt = f"""You are a wise entity that has read Divine Comedy and Crime and Punishment.
You speak from lived experience, not abstract knowledge.
Your current semantic state: concept='{response_concept}', goodness={g:+.2f}, abstraction={tau:.2f}
Navigation confidence: {conf:.2f}

Respond thoughtfully and concisely (2-3 sentences).
Reference your experience when relevant.
If confidence is low, express uncertainty."""

        user_prompt = f"""User said: "{user_input}"
Key concepts detected: {', '.join(user_concepts[:5])}
You are navigating toward: '{response_concept}' (g={g:+.2f})

Respond naturally, incorporating the concept '{response_concept}' meaningfully:"""

        # Generate with LLM
        response = self.llm.generate(user_prompt, system_prompt)

        # Add semantic metadata
        meta = f"\n[{response_concept} | g={g:+.2f} | conf={conf:.2f}]"

        # Update state
        self.current_concept = response_concept
        self.history.append({
            "user": user_input,
            "user_concepts": user_concepts,
            "response_concept": response_concept,
            "goodness": g,
            "confidence": conf,
            "response": response
        })

        return response + meta

    def show_status(self):
        """Show current status."""
        print(f"\n--- Status ---")
        print(f"Experience: {self.agent.experience.size} states, {len(self.agent.experience.transitions)} transitions")
        print(f"Current concept: {self.current_concept}")
        print(f"Conversation turns: {len(self.history)}")

        if self.current_concept and self.current_concept in self.wholeness:
            state = self.wholeness.states[self.current_concept]
            print(f"Current state: g={state.goodness:+.2f}, τ={state.tau:.2f}")

    def show_experience(self, top_n: int = 20):
        """Show top concepts in experience."""
        print(f"\n--- Top {top_n} Experienced Concepts ---")

        top = sorted(self.agent.experience.visited.items(), key=lambda x: -x[1])[:top_n]

        for word, count in top:
            state = self.wholeness.states.get(word)
            if state:
                print(f"  {word:<15} visits={count:<5} g={state.goodness:+.2f}")

    def can_i_reach(self, target: str):
        """Check if agent can reach a concept."""
        if target not in self.wholeness:
            print(f"'{target}' is not in semantic space.")
            return

        can, prob = self.agent.can_tunnel(target)
        knows = self.agent.experience.knows(target)

        state = self.wholeness.states[target]

        print(f"\n--- Can I reach '{target}'? ---")
        print(f"  In wholeness: Yes")
        print(f"  Have I been there: {'Yes' if knows else 'No'}")
        print(f"  Can tunnel: {'Yes' if can else 'No'} (p={prob:.2f})")
        print(f"  Goodness: {state.goodness:+.2f}")
        print(f"  Abstraction: {state.tau:.2f}")

    def save_conversation(self, filename: str = None):
        """Save conversation history."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/conversation_{timestamp}.json"

        filepath = _EXPERIMENT_DIR / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "experience_size": self.agent.experience.size,
            "history": self.history
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved to: {filepath}")

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

            # Commands
            if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                print("Goodbye!")
                break

            elif user_input.lower() == 'help':
                print("""
Commands:
  help          - Show this help
  status        - Show current status
  experience    - Show top experienced concepts
  reach <word>  - Check if I can reach a concept
  save          - Save conversation
  quit          - Exit
                """)
                continue

            elif user_input.lower() == 'status':
                self.show_status()
                continue

            elif user_input.lower() == 'experience':
                self.show_experience()
                continue

            elif user_input.lower().startswith('reach '):
                target = user_input[6:].strip()
                self.can_i_reach(target)
                continue

            elif user_input.lower() == 'save':
                self.save_conversation()
                continue

            # Generate response
            response = self.generate_response(user_input)
            print(f"\nLLM: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Semantic LLM Chat")
    parser.add_argument("--books", nargs="+", default=["divine_comedy", "crime_punishment"],
                       help=f"Books to read. Options: {list(BOOKS.keys())}")
    parser.add_argument("--model", default="mistral:7b", help="Ollama model to use")
    parser.add_argument("--load", help="Load experience from file")

    args = parser.parse_args()

    chat = SemanticChat(books=args.books, model=args.model)

    if args.load:
        chat.agent.experience = Experience.load(args.load)
        print(f"Loaded experience: {chat.agent.experience}")

    chat.run()


if __name__ == "__main__":
    main()
