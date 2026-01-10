#!/usr/bin/env python3
"""Dream Therapy Session: Claude as Dreamer, LLM as Therapist.

Simulates a dream analysis therapy session where:
- Claude generates dream content and associations
- Therapist (Groq/Ollama) helps interpret using psychoanalytic methods
- Storm-Logos tracks semantic coordinates and corpus resonances

Usage:
    python -m storm_logos.scripts.dream_therapy --turns 10
    python -m storm_logos.scripts.dream_therapy --model claude --turns 5
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_env():
    """Load environment variables from .env files."""
    locations = [
        Path.home() / '.env',
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent.parent.parent / '.env',
        Path.cwd() / '.env',
    ]

    for env_path in locations:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key in ('ANTHROPIC_API_KEY', 'GROQ_API_KEY') and not os.environ.get(key):
                            os.environ[key] = value


load_env()

from storm_logos.data.postgres import get_data
from storm_logos.data.neo4j import get_neo4j
from storm_logos.data.models import Bond


@dataclass
class DreamTurn:
    """A single turn in dream therapy."""
    turn: int
    dreamer: str  # Claude's dream/association
    therapist: str  # Therapist interpretation
    symbols: List[Dict] = field(default_factory=list)
    state: Dict = field(default_factory=dict)
    corpus_resonances: List[Dict] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class DreamSession:
    """Complete dream therapy session."""
    session_id: str
    start_time: str
    end_time: str = ""
    therapist_model: str = ""
    turns: List[DreamTurn] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


class DreamTherapist:
    """Conducts dream analysis therapy sessions."""

    def __init__(self, therapist_model: str = "groq:llama-3.3-70b-versatile"):
        self.therapist_model = therapist_model
        self.data = None
        self.neo4j = None
        self._claude = None
        self._therapist = None

    def connect(self) -> bool:
        """Initialize connections."""
        print("Loading semantic data...")
        self.data = get_data()
        print(f"Total: {self.data.n_coordinates:,} coordinates")

        self.neo4j = get_neo4j()
        if not self.neo4j.connect():
            print("Warning: Neo4j not connected (corpus search disabled)")

        # Initialize Claude (dreamer)
        import anthropic
        self._claude = anthropic.Anthropic()
        print(f"API key loaded: {os.environ.get('ANTHROPIC_API_KEY', '')[:20]}...")

        # Initialize therapist
        if self.therapist_model.startswith("groq:"):
            from groq import Groq
            self._therapist = ("groq", Groq(), self.therapist_model.split(":", 1)[1])
        elif self.therapist_model == "claude":
            self._therapist = ("claude", self._claude, "claude-sonnet-4-20250514")
        else:
            self._therapist = ("ollama", None, self.therapist_model)

        return True

    def _call_claude(self, system: str, messages: List[Dict]) -> str:
        """Call Claude as dreamer."""
        response = self._claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def _call_therapist(self, system: str, user: str) -> str:
        """Call therapist LLM."""
        client_type, client, model = self._therapist

        if client_type == "groq":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=512,
                temperature=0.7,
            )
            return response.choices[0].message.content

        elif client_type == "claude":
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text

        else:  # Ollama
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"{system}\n\nUser: {user}",
                    "stream": False,
                },
            )
            return response.json()["response"]

    def extract_symbols(self, text: str) -> List[Dict]:
        """Extract dream symbols from text."""
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        symbols = []
        seen = set()

        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ == "amod":
                if token.head.pos_ == "NOUN":
                    adj = token.lemma_.lower()
                    noun = token.head.lemma_.lower()
                    key = f"{adj}_{noun}"

                    if key not in seen:
                        seen.add(key)
                        adj_coords = self.data.get(adj)
                        noun_coords = self.data.get(noun)

                        if adj_coords and noun_coords:
                            A = (adj_coords.A + noun_coords.A) / 2
                            S = (adj_coords.S + noun_coords.S) / 2
                            tau = (adj_coords.tau + noun_coords.tau) / 2
                        elif adj_coords:
                            A, S, tau = adj_coords.A, adj_coords.S, adj_coords.tau
                        elif noun_coords:
                            A, S, tau = noun_coords.A, noun_coords.S, noun_coords.tau
                        else:
                            A, S, tau = 0.0, 0.0, 2.5

                        symbols.append({
                            "text": f"{adj} {noun}",
                            "A": A, "S": S, "tau": tau,
                        })

            elif token.pos_ == "NOUN" and token.dep_ in ("nsubj", "dobj", "pobj"):
                noun = token.lemma_.lower()
                if noun not in seen and len(noun) > 2:
                    coords = self.data.get(noun)
                    if coords:
                        seen.add(noun)
                        symbols.append({
                            "text": noun,
                            "A": coords.A, "S": coords.S, "tau": coords.tau,
                        })

        return symbols[:10]  # Limit

    def find_resonances(self, symbols: List[Dict], limit: int = 5) -> List[Dict]:
        """Find corpus resonances for symbols."""
        if not self.neo4j or not self.neo4j._connected:
            return []

        resonances = []

        for sym in symbols[:5]:
            text = sym["text"]
            bond_id = text.replace(" ", "_")

            query = """
            MATCH (book:Book)-[:CONTAINS]->(bond:Bond)
            WHERE bond.id CONTAINS $term
            RETURN book.title as book, book.author as author, bond.id as bond
            LIMIT 2
            """

            try:
                with self.neo4j._driver.session() as session:
                    result = session.run(query, term=bond_id)
                    for record in result:
                        resonances.append({
                            "symbol": text,
                            "book": record["book"],
                            "author": record["author"],
                        })
            except:
                pass

        return resonances[:limit]

    def run_session(self, n_turns: int = 10) -> DreamSession:
        """Run a complete dream therapy session."""

        session = DreamSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat(),
            therapist_model=self.therapist_model,
        )

        # Dreamer (Claude) system prompt
        dreamer_system = """You are a person in a dream therapy session. You have vivid, symbolic dreams
that you want to understand. In this session:

1. First, describe a dream you had (detailed, with imagery and emotions)
2. When the therapist asks questions, share your associations and feelings
3. Be open but also uncertain - you're trying to understand what your dreams mean
4. Your dreams contain archetypal imagery: shadows, guides, transformations, journeys
5. Express emotions: confusion, curiosity, fear, wonder

Start by describing a dream. Be vivid and specific with imagery."""

        # Therapist system prompt
        therapist_system = """You are a psychoanalyst trained in Jungian and Freudian dream interpretation.
Your approach:

1. Listen carefully to the dream narrative
2. Identify key symbols and their archetypal meanings
3. Ask about the dreamer's associations and feelings
4. Connect symbols to possible psychological meaning
5. Be curious, not dogmatic - dreams have multiple meanings
6. Use terms like: shadow, anima/animus, self, projection, unconscious
7. Help the dreamer discover meaning, don't impose it

Keep responses focused (2-4 sentences). Ask one key question at a time."""

        # Conversation history for Claude
        claude_messages = []

        print("\n" + "=" * 60)
        print("DREAM THERAPY SESSION")
        print(f"Dreamer: Claude | Therapist: {self.therapist_model}")
        print("=" * 60)

        # Initial dream
        print("-" * 60)
        print("DREAMER: *describing initial dream*")

        initial = self._call_claude(dreamer_system, [
            {"role": "user", "content": "Please describe a dream you've had recently that feels significant to you."}
        ])
        print(f"\n{initial}")
        claude_messages.append({"role": "assistant", "content": initial})

        # Extract symbols
        symbols = self.extract_symbols(initial)
        resonances = self.find_resonances(symbols)

        # Calculate state
        if symbols:
            avg_A = sum(s["A"] for s in symbols) / len(symbols)
            avg_S = sum(s["S"] for s in symbols) / len(symbols)
        else:
            avg_A, avg_S = 0.0, 0.0

        # Therapist first response
        therapist_prompt = f"""The patient describes this dream:

"{initial}"

Symbols detected: {', '.join(s['text'] for s in symbols[:5])}
Emotional valence: {avg_A:+.2f}

Respond as a dream analyst. Acknowledge the dream briefly and ask one focused question about the most significant symbol or feeling."""

        therapist_response = self._call_therapist(therapist_system, therapist_prompt)
        print(f"\nTHERAPIST [1]: {therapist_response}")
        print(f"  [Symbols: {', '.join(s['text'] for s in symbols[:3])} | A={avg_A:+.2f}]")

        session.turns.append(DreamTurn(
            turn=1,
            dreamer=initial,
            therapist=therapist_response,
            symbols=symbols,
            state={"A": avg_A, "S": avg_S},
            corpus_resonances=resonances,
            timestamp=datetime.now().isoformat(),
        ))

        # Continue conversation
        for turn in range(2, n_turns + 1):
            # Claude responds to therapist
            claude_messages.append({"role": "user", "content": f"Therapist: {therapist_response}"})

            dreamer_response = self._call_claude(dreamer_system, claude_messages)
            print(f"\nDREAMER [{turn}]: {dreamer_response}")
            claude_messages.append({"role": "assistant", "content": dreamer_response})

            # Extract new symbols
            symbols = self.extract_symbols(dreamer_response)
            resonances = self.find_resonances(symbols)

            if symbols:
                avg_A = sum(s["A"] for s in symbols) / len(symbols)
                avg_S = sum(s["S"] for s in symbols) / len(symbols)

            # Therapist responds
            therapist_prompt = f"""Continue the dream therapy session.

Patient's response:
"{dreamer_response}"

New symbols detected: {', '.join(s['text'] for s in symbols[:5]) if symbols else 'none'}
Emotional valence: {avg_A:+.2f}

Previous context: You've been exploring their dream. Continue with empathy and insight. Ask about associations, feelings, or offer a gentle interpretation."""

            therapist_response = self._call_therapist(therapist_system, therapist_prompt)
            print(f"\nTHERAPIST [{turn}]: {therapist_response}")
            print(f"  [A={avg_A:+.2f}, S={avg_S:+.2f}]")

            # Add corpus resonances occasionally
            if resonances and turn % 3 == 0:
                print(f"  [Corpus: {resonances[0]['symbol']} in {resonances[0]['book']}]")

            session.turns.append(DreamTurn(
                turn=turn,
                dreamer=dreamer_response,
                therapist=therapist_response,
                symbols=symbols,
                state={"A": avg_A, "S": avg_S},
                corpus_resonances=resonances,
                timestamp=datetime.now().isoformat(),
            ))

        # Generate summary
        all_symbols = []
        a_values = []
        for t in session.turns:
            all_symbols.extend(t.symbols)
            if t.state.get("A"):
                a_values.append(t.state["A"])

        session.end_time = datetime.now().isoformat()
        session.summary = {
            "n_turns": n_turns,
            "total_symbols": len(all_symbols),
            "avg_A": sum(a_values) / len(a_values) if a_values else 0,
            "A_start": a_values[0] if a_values else 0,
            "A_end": a_values[-1] if a_values else 0,
            "top_symbols": [s["text"] for s in all_symbols[:10]],
        }

        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Turns: {n_turns}")
        print(f"Total symbols: {len(all_symbols)}")
        print(f"Average A: {session.summary['avg_A']:+.2f}")
        print(f"A trajectory: {session.summary['A_start']:+.2f} → {session.summary['A_end']:+.2f}")
        print(f"Key symbols: {', '.join(session.summary['top_symbols'][:5])}")

        # Save session
        sessions_dir = Path(__file__).parent.parent / "sessions"
        sessions_dir.mkdir(exist_ok=True)

        filepath = sessions_dir / f"dream_therapy_{session.session_id}.json"

        # Convert to serializable dict
        session_dict = {
            "session_id": session.session_id,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "therapist_model": session.therapist_model,
            "turns": [
                {
                    "turn": t.turn,
                    "dreamer": t.dreamer,
                    "therapist": t.therapist,
                    "symbols": t.symbols,
                    "state": t.state,
                    "corpus_resonances": t.corpus_resonances,
                    "timestamp": t.timestamp,
                }
                for t in session.turns
            ],
            "summary": session.summary,
        }

        with open(filepath, 'w') as f:
            json.dump(session_dict, f, indent=2)

        print(f"\nSession saved to: {filepath}")

        return session


def main():
    parser = argparse.ArgumentParser(
        description="Dream Therapy: Claude as Dreamer, LLM as Therapist",
        prog="dream_therapy",
    )

    parser.add_argument("--turns", type=int, default=10,
                        help="Number of conversation turns")
    parser.add_argument("--model", type=str, default="groq:llama-3.3-70b-versatile",
                        help="Therapist model")

    args = parser.parse_args()

    therapist = DreamTherapist(therapist_model=args.model)
    therapist.connect()
    therapist.run_session(n_turns=args.turns)


if __name__ == "__main__":
    main()
