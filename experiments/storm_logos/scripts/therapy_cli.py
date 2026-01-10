#!/usr/bin/env python3
"""Therapy CLI: Interactive dream analysis and therapy sessions.

A context-aware CLI that guides users through:
- Dream exploration and Jungian analysis
- Therapeutic conversation
- Automatic mode detection based on user input

Usage:
    python -m storm_logos.scripts.therapy_cli
    python -m storm_logos.scripts.therapy_cli --mode dream
    python -m storm_logos.scripts.therapy_cli --mode therapy
"""

import argparse
import os
import sys
import json
import readline  # Enable arrow keys and history
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_env():
    """Load environment variables from .env files."""
    locations = [
        Path(__file__).parent.parent / '.env',
        Path.home() / '.env',
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
                        if not os.environ.get(key):
                            os.environ[key] = value


load_env()


class SessionMode(Enum):
    UNKNOWN = "unknown"
    DREAM = "dream"
    THERAPY = "therapy"
    HYBRID = "hybrid"


class ResponseType(Enum):
    DREAM_CONTENT = "dream_content"
    ASSOCIATION = "association"
    EMOTION = "emotion"
    QUESTION = "question"
    REFLECTION = "reflection"
    RESISTANCE = "resistance"
    UNCLEAR = "unclear"
    GOODBYE = "goodbye"


@dataclass
class SessionState:
    """Tracks the current session state."""
    mode: SessionMode = SessionMode.UNKNOWN
    turn: int = 0
    dream_text: Optional[str] = None
    symbols: List[Dict] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    semantic_trajectory: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    needs_clarification: bool = False
    last_question: Optional[str] = None
    # Archetype manifestations (qualitative, not scores)
    archetype_manifestations: List[Dict] = field(default_factory=list)


class TherapyCLI:
    """Interactive therapy and dream analysis CLI."""

    def __init__(self, model: str = "groq:llama-3.3-70b-versatile"):
        self.model = model
        self.state = SessionState()
        self._engine = None
        self._therapist = None
        self._llm_client = None
        self._data = None
        self._user_graph = None
        self._current_user = None
        self._session_start = datetime.now()

    def connect(self) -> bool:
        """Initialize connections."""
        from storm_logos.applications import DreamEngine
        from storm_logos.data.postgres import get_data
        from storm_logos.data.user_graph import get_user_graph

        print("Initializing...")
        self._engine = DreamEngine(model=self.model)
        self._engine.connect()
        self._data = self._engine._data

        # Connect to user graph
        self._user_graph = get_user_graph()
        if self._user_graph.connect():
            print("User tracking enabled")
        else:
            print("Warning: User tracking unavailable (Neo4j not connected)")

        # Initialize LLM client
        if self.model.startswith("groq:"):
            from groq import Groq
            self._llm_client = ("groq", Groq(), self.model.split(":", 1)[1])
        elif self.model.startswith("claude"):
            import anthropic
            self._llm_client = ("claude", anthropic.Anthropic(), "claude-sonnet-4-20250514")
        else:
            self._llm_client = ("ollama", None, self.model)

        return True

    def _call_llm(self, system: str, user: str, max_tokens: int = 400) -> str:
        """Call LLM for responses."""
        client_type, client, model_name = self._llm_client

        if client_type == "groq":
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            return response.choices[0].message.content.strip()

        elif client_type == "claude":
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text.strip()

        else:  # Ollama
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"{system}\n\nUser: {user}",
                    "stream": False,
                },
                timeout=60,
            )
            return response.json().get("response", "")

    def _analyze_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to understand intent and content."""
        # Quick check for greetings - don't send to LLM
        greetings = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]
        if user_input.lower().strip() in greetings:
            return {
                "type": "greeting",
                "mode_hint": "unclear",
                "contains_dream": False,
                "emotions_detected": [],
                "key_symbols": [],
                "needs_clarification": False,
                "summary": "User greeting"
            }

        system = """Analyze the user's input in a therapy/dream context. Return JSON:
{
    "type": "greeting|dream_content|association|emotion|question|reflection|resistance|unclear|goodbye",
    "mode_hint": "dream|therapy|unclear",
    "contains_dream": true/false,
    "emotions_detected": ["list", "of", "emotions"],
    "key_symbols": ["list", "of", "symbols"],
    "needs_clarification": true/false,
    "clarification_needed": "what to ask if unclear",
    "summary": "brief summary of what user shared"
}

Types:
- greeting: user says hello, hi, hey (NOT goodbye)
- dream_content: user is sharing/describing a dream
- association: user is making connections, remembering related things
- emotion: user is expressing feelings
- question: user is asking something
- reflection: user is thinking about meaning
- resistance: user seems defensive or avoidant
- unclear: can't determine what user means
- goodbye: user explicitly wants to end (bye, goodbye, quit, exit, I need to go)

IMPORTANT: "hello", "hi", "hey" are GREETINGS not goodbyes!"""

        prompt = f"""Current session mode: {self.state.mode.value}
Turn: {self.state.turn}
Previous themes: {', '.join(self.state.themes[-5:]) if self.state.themes else 'none'}

User input:
"{user_input}"

Analyze this input. Return only valid JSON."""

        try:
            response = self._call_llm(system, prompt, max_tokens=300)
            # Extract JSON from response
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}")+1]
                return json.loads(json_str)
        except:
            pass

        return {
            "type": "unclear",
            "mode_hint": "unclear",
            "contains_dream": False,
            "emotions_detected": [],
            "key_symbols": [],
            "needs_clarification": True,
            "clarification_needed": "Could you tell me more about what you'd like to explore?",
            "summary": user_input[:100]
        }

    def _generate_response(self, user_input: str, analysis: Dict) -> str:
        """Generate contextual therapeutic response."""
        # Build context
        history_context = ""
        if self.state.history:
            recent = self.state.history[-3:]
            history_context = "\n".join([
                f"User: {h['user'][:100]}...\nTherapist: {h['therapist'][:100]}..."
                for h in recent
            ])

        dream_context = ""
        if self.state.dream_text:
            dream_context = f"Dream being explored: {self.state.dream_text[:300]}..."

        symbols_context = ""
        if self.state.symbols:
            symbols_context = f"Symbols identified: {', '.join([s.get('text', str(s)) for s in self.state.symbols[:5]])}"

        system = f"""You are a depth psychologist conducting a {self.state.mode.value} session.

Your approach:
1. Listen deeply and reflect what you hear
2. Ask focused questions - one at a time
3. Connect symbols to psychological meaning when appropriate
4. If something is unclear, ask for clarification gently
5. Never interpret too quickly - let meaning emerge
6. Match the emotional tone of the user
7. Use Jungian concepts naturally: shadow, anima/animus, Self, archetypes
8. Keep responses concise (2-4 sentences usually)

{f"Context: {dream_context}" if dream_context else ""}
{f"Previous: {history_context}" if history_context else ""}
{f"{symbols_context}" if symbols_context else ""}

Response type needed: {analysis.get('type', 'unknown')}
Emotions present: {', '.join(analysis.get('emotions_detected', []))}
Key symbols: {', '.join(analysis.get('key_symbols', []))}"""

        prompt = f"""User says:
"{user_input}"

Analysis: {analysis.get('summary', '')}
{f"Needs clarification about: {analysis.get('clarification_needed', '')}" if analysis.get('needs_clarification') else ""}

Respond therapeutically. If this is the start of a dream, acknowledge it and ask about feelings or a specific image. If unclear, ask a gentle clarifying question."""

        return self._call_llm(system, prompt)

    def _extract_symbols(self, text: str) -> List[Dict]:
        """Extract symbols from text using DreamEngine."""
        if self._engine:
            symbols = self._engine.extract_symbols(text)
            return [{"text": s.raw_text, "archetype": s.archetype, "A": s.bond.A, "S": s.bond.S}
                    for s in symbols]
        return []

    def _detect_mode(self, user_input: str, analysis: Dict) -> SessionMode:
        """Detect or confirm session mode from user input."""
        input_lower = user_input.lower()

        # Explicit mode indicators
        if any(w in input_lower for w in ["dream", "dreamed", "dreamt", "nightmare", "vision"]):
            return SessionMode.DREAM
        if any(w in input_lower for w in ["feeling", "anxious", "depressed", "stressed", "help me"]):
            return SessionMode.THERAPY

        # Use analysis hint
        mode_hint = analysis.get("mode_hint", "unclear")
        if mode_hint == "dream":
            return SessionMode.DREAM
        if mode_hint == "therapy":
            return SessionMode.THERAPY

        # Check for dream content
        if analysis.get("contains_dream"):
            return SessionMode.DREAM

        return self.state.mode if self.state.mode != SessionMode.UNKNOWN else SessionMode.HYBRID

    def process_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        user_input = user_input.strip()

        if not user_input:
            return "I'm here. Take your time."

        # Check for commands
        if user_input.startswith("/"):
            return self._handle_command(user_input)

        # Analyze input
        analysis = self._analyze_input(user_input)

        # Check for goodbye
        if analysis.get("type") == "goodbye":
            return self._end_session()

        # Handle greeting
        if analysis.get("type") == "greeting":
            if self.state.mode == SessionMode.DREAM:
                return "Welcome. I'm here to explore your dreams with you. Tell me about a dream that's been on your mind."
            elif self.state.mode == SessionMode.THERAPY:
                return "Welcome. I'm here to listen. What's on your mind today?"
            else:
                return "Welcome. I'm here to help you explore. You can share a dream, talk about what's weighing on you, or start wherever feels right. What brings you here today?"

        # Update mode
        if self.state.mode == SessionMode.UNKNOWN:
            self.state.mode = self._detect_mode(user_input, analysis)

        # Extract symbols if dream content
        if analysis.get("contains_dream") or analysis.get("type") == "dream_content":
            if not self.state.dream_text:
                self.state.dream_text = user_input
            new_symbols = self._extract_symbols(user_input)
            self.state.symbols.extend(new_symbols)

        # Track emotions
        if analysis.get("emotions_detected"):
            self.state.emotions.extend(analysis["emotions_detected"])

        # Track themes/symbols
        if analysis.get("key_symbols"):
            self.state.themes.extend(analysis["key_symbols"])

        # Generate response
        response = self._generate_response(user_input, analysis)

        # Update state
        self.state.turn += 1
        self.state.history.append({
            "turn": self.state.turn,
            "user": user_input,
            "therapist": response,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })

        return response

    def _handle_command(self, cmd: str) -> str:
        """Handle CLI commands."""
        cmd = cmd.lower().strip()

        if cmd in ["/quit", "/exit", "/bye"]:
            return self._end_session()

        if cmd == "/symbols":
            if self.state.symbols:
                lines = ["Symbols detected:"]
                for s in self.state.symbols[:10]:
                    arch = f" [{s.get('archetype')}]" if s.get('archetype') else ""
                    lines.append(f"  - {s.get('text', s)}{arch}")
                return "\n".join(lines)
            return "No symbols detected yet. Share a dream to begin."

        if cmd == "/state":
            return f"""Session state:
  Mode: {self.state.mode.value}
  Turn: {self.state.turn}
  Symbols: {len(self.state.symbols)}
  Emotions: {', '.join(set(self.state.emotions[-5:])) if self.state.emotions else 'none'}
  Themes: {', '.join(set(self.state.themes[-5:])) if self.state.themes else 'none'}"""

        if cmd == "/dream":
            self.state.mode = SessionMode.DREAM
            return "Let's explore a dream. Tell me about a dream that's been on your mind."

        if cmd == "/therapy":
            self.state.mode = SessionMode.THERAPY
            return "I'm here to listen. What would you like to talk about?"

        if cmd == "/save":
            filepath = self._save_session()
            return f"Session saved to {filepath}"

        if cmd == "/help":
            user_cmds = ""
            if self._current_user:
                user_cmds = """
  /profile  - Show your archetype profile
  /evolution <archetype> - Show archetype evolution
  /history  - Show session history
  /logout   - Log out"""
            else:
                user_cmds = """
  /login    - Log in to track evolution
  /register - Create new account"""

            return f"""Commands:
  /dream    - Start dream exploration
  /therapy  - Start therapy mode
  /symbols  - Show detected symbols
  /state    - Show session state
  /save     - Save session
  /quit     - End session{user_cmds}"""

        # User authentication
        if cmd == "/login":
            return self._login_prompt()

        if cmd == "/register":
            return self._register_prompt()

        if cmd == "/logout":
            if self._current_user:
                name = self._current_user.username
                self._current_user = None
                return f"Logged out. Goodbye, {name}."
            return "Not logged in."

        if cmd == "/profile":
            return self._show_profile()

        if cmd.startswith("/evolution"):
            parts = cmd.split()
            archetype = parts[1] if len(parts) > 1 else "shadow"
            return self._show_evolution(archetype)

        if cmd == "/history":
            return self._show_history()

        return f"Unknown command: {cmd}. Type /help for commands."

    def _login_prompt(self) -> str:
        """Handle login flow."""
        if not self._user_graph or not self._user_graph._connected:
            return "User tracking unavailable (Neo4j not connected)"

        try:
            username = input("Username: ").strip()
            if not username:
                return "Login cancelled."

            import getpass
            password = getpass.getpass("Password: ")

            user = self._user_graph.authenticate(username, password)
            if user:
                self._current_user = user
                return f"Welcome back, {username}. Your archetype evolution is being tracked."
            return "Invalid username or password."
        except (KeyboardInterrupt, EOFError):
            return "\nLogin cancelled."

    def _register_prompt(self) -> str:
        """Handle registration flow."""
        if not self._user_graph or not self._user_graph._connected:
            return "User tracking unavailable (Neo4j not connected)"

        try:
            username = input("Choose username: ").strip()
            if not username:
                return "Registration cancelled."

            # Check if exists
            if self._user_graph.get_user(username):
                return f"Username '{username}' already exists. Try /login"

            import getpass
            password = getpass.getpass("Choose password: ")
            confirm = getpass.getpass("Confirm password: ")

            if password != confirm:
                return "Passwords don't match."

            user = self._user_graph.create_user(username, password)
            if user:
                self._current_user = user
                return f"Account created. Welcome, {username}. Your journey begins."
            return "Error creating account."
        except (KeyboardInterrupt, EOFError):
            return "\nRegistration cancelled."

    def _show_profile(self) -> str:
        """Show user's archetype profile."""
        if not self._current_user:
            return "Not logged in. Use /login or /register to track your evolution."

        profile = self._user_graph.get_user_archetype_profile(self._current_user.user_id)
        if not profile["total_sessions"]:
            return "No sessions recorded yet. Complete a session to begin tracking."

        lines = [f"Archetype Profile for {self._current_user.username}:"]
        lines.append(f"  Total sessions: {profile['total_sessions']}")

        if profile["dominant_archetypes"]:
            lines.append(f"  Dominant: {', '.join(profile['dominant_archetypes'])}")

        lines.append("\n  Archetype presence:")
        for arch, count in sorted(profile["archetypes"].items(), key=lambda x: -x[1]):
            pct = count / profile["total_sessions"] * 100
            bar = "█" * int(pct / 5)
            lines.append(f"    {arch:15} {bar} ({count} sessions)")

        return "\n".join(lines)

    def _show_evolution(self, archetype: str) -> str:
        """Show evolution of a specific archetype."""
        if not self._current_user:
            return "Not logged in. Use /login to track evolution."

        evolution = self._user_graph.get_archetype_evolution(
            self._current_user.user_id, archetype
        )

        if not evolution:
            return f"No '{archetype}' manifestations recorded yet."

        lines = [f"Evolution of {archetype.upper()}:"]
        for e in evolution[-10:]:  # Last 10
            date = e["timestamp"][:10] if e["timestamp"] else "unknown"
            symbols = ", ".join(e["symbols"][:3]) if e["symbols"] else "none"
            emotions = ", ".join(e["emotions"][:2]) if e["emotions"] else "none"
            lines.append(f"  [{date}] symbols: {symbols} | felt: {emotions}")

        # Show common symbols
        arch_symbols = self._user_graph.get_archetype_symbols(
            self._current_user.user_id, archetype
        )
        if arch_symbols:
            top_symbols = [s["symbol"] for s in arch_symbols[:5]]
            lines.append(f"\n  Common symbols: {', '.join(top_symbols)}")

        return "\n".join(lines)

    def _show_history(self) -> str:
        """Show session history."""
        if not self._current_user:
            return "Not logged in."

        sessions = self._user_graph.get_user_sessions(self._current_user.user_id, limit=10)
        if not sessions:
            return "No sessions recorded yet."

        lines = ["Recent sessions:"]
        for s in sessions:
            date = s["timestamp"][:10] if s["timestamp"] else "unknown"
            archs = ", ".join(s["archetypes"][:3]) if s["archetypes"] else "none"
            lines.append(f"  [{date}] {s['mode']:8} archetypes: {archs}")

        return "\n".join(lines)

    def _end_session(self) -> str:
        """End the session gracefully."""
        # Extract archetypes qualitatively
        archetypes = self._extract_archetypes()

        # Save to file
        filepath = self._save_session()

        # Save to user graph if logged in
        user_saved = False
        if self._current_user and self._user_graph and self._user_graph._connected:
            try:
                self._save_to_user_graph(archetypes)
                user_saved = True
            except Exception as e:
                print(f"Warning: Could not save to user graph: {e}")

        # Build summary
        arch_summary = ""
        if archetypes:
            arch_lines = []
            for a in archetypes:
                symbols = ", ".join(a.get("symbols", [])[:3])
                emotions = ", ".join(a.get("emotions", [])[:2])
                arch_lines.append(f"  {a['archetype']}: {symbols} (felt: {emotions})")
            arch_summary = "\nArchetypes manifested:\n" + "\n".join(arch_lines)

        user_note = ""
        if self._current_user and user_saved:
            user_note = f"\nEvolution tracked for {self._current_user.username}."
        elif not self._current_user:
            user_note = "\nTip: Use /register to track your archetype evolution over time."

        summary = f"""
Our time together is ending. Here's what we explored:

Turns: {self.state.turn}
Mode: {self.state.mode.value}
Symbols found: {len(self.state.symbols)}
Key themes: {', '.join(set(self.state.themes[-5:])) if self.state.themes else 'none explored'}
{arch_summary}

Session saved to: {filepath}{user_note}

Take care of yourself. The work of understanding continues in waking life too.
"""
        return summary

    def _extract_archetypes(self) -> List[Dict]:
        """Extract archetypes qualitatively from session - as symbols and emotions."""
        if not self.state.history:
            return []

        # Collect all content from session
        all_text = " ".join([
            h.get("user", "") + " " + h.get("therapist", "")
            for h in self.state.history
        ])

        # Use LLM to extract archetype manifestations qualitatively
        system = """Analyze this therapy/dream session and identify which Jungian archetypes manifested.
For each archetype present, identify:
- The symbols through which it appeared
- The emotions associated with it
- Brief context of how it manifested

Return JSON array:
[
    {
        "archetype": "shadow|anima_animus|self|mother|father|hero|trickster|death_rebirth",
        "symbols": ["symbol1", "symbol2"],
        "emotions": ["emotion1", "emotion2"],
        "context": "Brief description of manifestation"
    }
]

Only include archetypes that clearly manifested. Quality over quantity.
Return [] if no clear archetypes present."""

        # Add symbol context
        symbol_text = ""
        if self.state.symbols:
            symbol_text = "\n\nSymbols detected: " + ", ".join([
                s.get("text", str(s)) for s in self.state.symbols[:10]
            ])

        emotion_text = ""
        if self.state.emotions:
            emotion_text = "\n\nEmotions expressed: " + ", ".join(set(self.state.emotions))

        prompt = f"""Session content:
{all_text[:3000]}
{symbol_text}
{emotion_text}

Extract archetype manifestations. Return only valid JSON array."""

        try:
            response = self._call_llm(system, prompt, max_tokens=500)
            # Extract JSON
            if "[" in response:
                json_str = response[response.index("["):response.rindex("]")+1]
                archetypes = json.loads(json_str)
                self.state.archetype_manifestations = archetypes
                return archetypes
        except Exception as e:
            print(f"Warning: Could not extract archetypes: {e}")

        return []

    def _save_to_user_graph(self, archetypes: List[Dict]):
        """Save session to user graph for evolution tracking."""
        from storm_logos.data.user_graph import SessionRecord, ArchetypeManifestation

        session_id = self._session_start.strftime("%Y%m%d_%H%M%S")

        # Convert archetypes to manifestations
        manifestations = []
        for a in archetypes:
            manifestations.append(ArchetypeManifestation(
                archetype=a.get("archetype", "unknown"),
                symbols=a.get("symbols", []),
                emotions=a.get("emotions", []),
                context=a.get("context", ""),
            ))

        # Create session record
        record = SessionRecord(
            session_id=session_id,
            user_id=self._current_user.user_id,
            mode=self.state.mode.value,
            timestamp=self._session_start.isoformat(),
            dream_text=self.state.dream_text,
            archetypes=manifestations,
            symbols=[s.get("text", str(s)) for s in self.state.symbols],
            emotions=list(set(self.state.emotions)),
            themes=list(set(self.state.themes)),
            summary=f"{self.state.turn} turns, {len(self.state.symbols)} symbols",
        )

        self._user_graph.save_session(record)

    def _save_session(self) -> str:
        """Save session to file."""
        sessions_dir = Path(__file__).parent.parent / "sessions"
        sessions_dir.mkdir(exist_ok=True)

        session_id = self._session_start.strftime("%Y%m%d_%H%M%S")
        filepath = sessions_dir / f"cli_session_{session_id}.json"

        data = {
            "session_id": session_id,
            "mode": self.state.mode.value,
            "start_time": self._session_start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "turns": self.state.turn,
            "dream_text": self.state.dream_text,
            "symbols": self.state.symbols,
            "themes": list(set(self.state.themes)),
            "emotions": list(set(self.state.emotions)),
            "history": self.state.history,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def run(self, initial_mode: Optional[str] = None):
        """Run the interactive CLI."""
        # Set initial mode if specified
        if initial_mode:
            self.state.mode = SessionMode(initial_mode)

        # Print welcome
        print("\n" + "=" * 60)
        print("STORM-LOGOS THERAPY CLI")
        print("=" * 60)

        # Login status
        if self._current_user:
            print(f"Logged in as: {self._current_user.username}")
        else:
            print("Not logged in. Use /login or /register to track evolution.")

        if self.state.mode == SessionMode.DREAM:
            print("\nDream Exploration Mode")
            print("Share a dream you'd like to explore.\n")
        elif self.state.mode == SessionMode.THERAPY:
            print("\nTherapy Mode")
            print("I'm here to listen. What's on your mind?\n")
        else:
            print("\nWelcome. I'm here to help you explore.")
            print("You can share a dream, talk about what's on your mind,")
            print("or just start wherever feels right.\n")
            print("Type /help for commands, /quit to exit.\n")

        # Main loop
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                response = self.process_input(user_input)
                print(f"\nTherapist: {response}\n")

                # Check if session ended
                if "Our time together is ending" in response:
                    break

            except KeyboardInterrupt:
                print("\n")
                response = self.process_input("/quit")
                print(f"\nTherapist: {response}\n")
                break
            except EOFError:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Interactive therapy and dream analysis CLI",
        prog="therapy_cli"
    )
    parser.add_argument("--mode", choices=["dream", "therapy"],
                        help="Start in specific mode")
    parser.add_argument("--model", default="groq:llama-3.3-70b-versatile",
                        help="LLM model to use")

    args = parser.parse_args()

    cli = TherapyCLI(model=args.model)
    cli.connect()
    cli.run(initial_mode=args.mode)


if __name__ == "__main__":
    main()
