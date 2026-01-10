#!/usr/bin/env python3
"""Dream Analysis: Psychoanalytic interpretation using Storm-Logos.

Analyzes dreams through:
1. Bond extraction (symbols)
2. Semantic coordinate mapping (A, S, τ)
3. Corpus resonance (Jung, Freud, mythology)
4. Archetypal pattern matching
5. LLM-based interpretation

Usage:
    python -m storm_logos.scripts.dream_analysis --interactive
    python -m storm_logos.scripts.dream_analysis --dream "I was falling..."
    python -m storm_logos.scripts.dream_analysis --turns 5
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_env():
    """Load environment variables from .env files."""
    # Check multiple locations
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
from storm_logos.data.book_parser import BookParser


@dataclass
class DreamSymbol:
    """A symbol extracted from a dream."""
    bond: Bond
    raw_text: str
    archetype: str = ""
    interpretation: str = ""
    corpus_matches: List[str] = field(default_factory=list)


@dataclass
class DreamAnalysis:
    """Complete analysis of a dream."""
    dream_text: str
    symbols: List[DreamSymbol]
    overall_A: float  # Emotional valence
    overall_S: float  # Sacred/mundane
    overall_tau: float  # Abstraction level
    dominant_archetype: str
    interpretation: str
    corpus_resonances: List[Dict]
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "dream_text": self.dream_text,
            "symbols": [
                {
                    "bond": f"{s.bond.adj} {s.bond.noun}",
                    "A": s.bond.A,
                    "S": s.bond.S,
                    "tau": s.bond.tau,
                    "archetype": s.archetype,
                    "interpretation": s.interpretation,
                }
                for s in self.symbols
            ],
            "overall": {
                "A": self.overall_A,
                "S": self.overall_S,
                "tau": self.overall_tau,
            },
            "dominant_archetype": self.dominant_archetype,
            "interpretation": self.interpretation,
            "corpus_resonances": self.corpus_resonances,
            "timestamp": self.timestamp,
        }


# Archetypal patterns based on Jung
ARCHETYPES = {
    "shadow": {
        "keywords": ["dark", "black", "monster", "enemy", "hidden", "unknown",
                     "threatening", "evil", "dangerous", "fear", "chase", "attack"],
        "description": "The repressed, unknown aspects of the self",
        "A_range": (-1.0, 0.0),
    },
    "anima_animus": {
        "keywords": ["woman", "man", "beautiful", "mysterious", "lover", "guide",
                     "feminine", "masculine", "attractive", "strange"],
        "description": "The contrasexual aspect of the psyche",
        "A_range": (-0.5, 0.5),
    },
    "self": {
        "keywords": ["center", "whole", "mandala", "circle", "light", "unity",
                     "god", "divine", "complete", "transcendent", "sacred"],
        "description": "The archetype of wholeness and integration",
        "A_range": (0.3, 1.0),
    },
    "mother": {
        "keywords": ["mother", "earth", "water", "cave", "womb", "nurturing",
                     "protective", "devouring", "nature", "birth"],
        "description": "The nurturing/devouring maternal principle",
        "A_range": (-0.5, 0.8),
    },
    "father": {
        "keywords": ["father", "king", "authority", "sky", "law", "order",
                     "strict", "wise", "powerful", "judgment"],
        "description": "Authority, order, and spiritual principle",
        "A_range": (-0.3, 0.7),
    },
    "hero": {
        "keywords": ["hero", "journey", "battle", "dragon", "quest", "victory",
                     "sword", "brave", "fight", "overcome", "rescue"],
        "description": "The ego's journey toward individuation",
        "A_range": (0.0, 0.8),
    },
    "trickster": {
        "keywords": ["fool", "clown", "trick", "chaos", "change", "boundary",
                     "joke", "transform", "animal", "wild"],
        "description": "The agent of change and boundary-crossing",
        "A_range": (-0.3, 0.3),
    },
    "death_rebirth": {
        "keywords": ["death", "dying", "dead", "grave", "rebirth", "transform",
                     "end", "beginning", "decay", "renewal", "phoenix"],
        "description": "Transformation through symbolic death",
        "A_range": (-0.8, 0.5),
    },
}

# Common dream symbols (Freudian + Jungian)
DREAM_SYMBOLS = {
    "water": "The unconscious, emotions, the maternal",
    "falling": "Loss of control, anxiety, letting go",
    "flying": "Freedom, transcendence, escape from limitations",
    "house": "The self, the psyche, different rooms = different aspects",
    "snake": "Transformation, healing, repressed sexuality, wisdom",
    "teeth": "Power, aggression, anxiety about appearance/aging",
    "chase": "Avoidance, running from aspects of self",
    "death": "Transformation, ending of a phase, fear of change",
    "naked": "Vulnerability, exposure, authenticity",
    "exam": "Self-evaluation, fear of judgment, unpreparedness",
    "stairs": "Transition, levels of consciousness, progress",
    "door": "Opportunity, transition, new possibilities",
    "forest": "The unconscious, the unknown, nature",
    "fire": "Passion, transformation, destruction/renewal",
    "child": "New beginnings, innocence, vulnerability, potential",
}


class DreamAnalyzer:
    """Analyzes dreams using Storm-Logos semantic system."""

    def __init__(self, model: str = "groq:llama-3.3-70b-versatile"):
        self.model = model
        self.data = None
        self.neo4j = None
        self.parser = None
        self._llm_client = None

    def connect(self) -> bool:
        """Initialize connections."""
        print("Loading semantic data...")
        self.data = get_data()
        print(f"Total: {self.data.n_coordinates:,} coordinates")

        self.neo4j = get_neo4j()
        if not self.neo4j.connect():
            print("Warning: Could not connect to Neo4j (corpus search disabled)")

        self.parser = BookParser()
        return True

    def _get_llm_client(self):
        """Get LLM client based on model."""
        if self._llm_client is not None:
            return self._llm_client

        if self.model.startswith("groq:"):
            from groq import Groq
            self._llm_client = ("groq", Groq())
        elif self.model == "claude":
            import anthropic
            self._llm_client = ("claude", anthropic.Anthropic())
        else:
            # Assume Ollama
            import requests
            self._llm_client = ("ollama", None)

        return self._llm_client

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM."""
        client_type, client = self._get_llm_client()

        if client_type == "groq":
            model_name = self.model.split(":", 1)[1]
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content

        elif client_type == "claude":
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        else:  # Ollama
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\nUser: {user_prompt}",
                    "stream": False,
                },
            )
            return response.json()["response"]

    def extract_symbols(self, dream_text: str) -> List[DreamSymbol]:
        """Extract symbolic bonds from dream text."""
        # Use spaCy to extract bonds
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(dream_text)

        symbols = []
        seen = set()

        for token in doc:
            # Look for adjective-noun pairs
            if token.pos_ == "ADJ" and token.dep_ == "amod":
                if token.head.pos_ == "NOUN":
                    adj = token.lemma_.lower()
                    noun = token.head.lemma_.lower()
                    key = f"{adj}_{noun}"

                    if key not in seen:
                        seen.add(key)

                        # Get coordinates
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

                        bond = Bond(adj=adj, noun=noun, A=A, S=S, tau=tau)

                        # Identify archetype
                        archetype = self._identify_archetype(adj, noun, A)

                        symbol = DreamSymbol(
                            bond=bond,
                            raw_text=f"{token.text} {token.head.text}",
                            archetype=archetype,
                        )
                        symbols.append(symbol)

            # Also look for significant nouns
            elif token.pos_ == "NOUN" and token.dep_ in ("nsubj", "dobj", "pobj"):
                noun = token.lemma_.lower()
                if noun in DREAM_SYMBOLS and noun not in seen:
                    seen.add(noun)

                    coords = self.data.get(noun)
                    if coords:
                        A, S, tau = coords.A, coords.S, coords.tau
                    else:
                        A, S, tau = 0.0, 0.0, 2.5

                    bond = Bond(adj="", noun=noun, A=A, S=S, tau=tau)
                    archetype = self._identify_archetype("", noun, A)

                    symbol = DreamSymbol(
                        bond=bond,
                        raw_text=token.text,
                        archetype=archetype,
                        interpretation=DREAM_SYMBOLS.get(noun, ""),
                    )
                    symbols.append(symbol)

        return symbols

    def _identify_archetype(self, adj: str, noun: str, A: float) -> str:
        """Identify the archetype a symbol belongs to."""
        words = [adj.lower(), noun.lower()]

        best_match = ""
        best_score = 0

        for archetype, info in ARCHETYPES.items():
            score = 0
            for word in words:
                if word in info["keywords"]:
                    score += 2
                # Partial match
                for kw in info["keywords"]:
                    if kw in word or word in kw:
                        score += 1

            # Check A range
            a_min, a_max = info["A_range"]
            if a_min <= A <= a_max:
                score += 1

            if score > best_score:
                best_score = score
                best_match = archetype

        return best_match if best_score >= 2 else ""

    def find_corpus_resonances(self, symbols: List[DreamSymbol],
                                limit: int = 5) -> List[Dict]:
        """Find resonances with corpus (Jung, Freud, mythology)."""
        if not self.neo4j or not self.neo4j._connected:
            return []

        resonances = []

        for symbol in symbols[:5]:  # Limit to top 5 symbols
            bond_id = f"{symbol.bond.adj}_{symbol.bond.noun}" if symbol.bond.adj else symbol.bond.noun

            # Find where this bond appears in corpus
            query = """
            MATCH (book:Book)-[:CONTAINS]->(bond:Bond {id: $bond_id})
            RETURN book.title as book, book.author as author
            LIMIT 3
            """

            try:
                with self.neo4j._driver.session() as session:
                    result = session.run(query, bond_id=bond_id)
                    for record in result:
                        resonances.append({
                            "symbol": symbol.raw_text,
                            "book": record["book"],
                            "author": record["author"],
                        })
            except:
                pass

            # Find following bonds in corpus
            query = """
            MATCH (b:Bond {id: $bond_id})-[:FOLLOWS]->(next:Bond)
            RETURN next.adj + ' ' + next.noun as following, count(*) as freq
            ORDER BY freq DESC
            LIMIT 3
            """

            try:
                with self.neo4j._driver.session() as session:
                    result = session.run(query, bond_id=bond_id)
                    for record in result:
                        if record["following"].strip():
                            resonances.append({
                                "symbol": symbol.raw_text,
                                "corpus_follows": record["following"],
                                "frequency": record["freq"],
                            })
            except:
                pass

        return resonances[:limit]

    def analyze_dream(self, dream_text: str) -> DreamAnalysis:
        """Perform complete dream analysis."""
        # Extract symbols
        symbols = self.extract_symbols(dream_text)

        # Calculate overall coordinates
        if symbols:
            overall_A = sum(s.bond.A for s in symbols) / len(symbols)
            overall_S = sum(s.bond.S for s in symbols) / len(symbols)
            overall_tau = sum(s.bond.tau for s in symbols) / len(symbols)
        else:
            overall_A, overall_S, overall_tau = 0.0, 0.0, 2.5

        # Find dominant archetype
        archetype_counts = {}
        for s in symbols:
            if s.archetype:
                archetype_counts[s.archetype] = archetype_counts.get(s.archetype, 0) + 1

        dominant = max(archetype_counts.items(), key=lambda x: x[1])[0] if archetype_counts else "unknown"

        # Find corpus resonances
        resonances = self.find_corpus_resonances(symbols)

        # Generate interpretation using LLM
        interpretation = self._generate_interpretation(
            dream_text, symbols, overall_A, overall_S, dominant, resonances
        )

        return DreamAnalysis(
            dream_text=dream_text,
            symbols=symbols,
            overall_A=overall_A,
            overall_S=overall_S,
            overall_tau=overall_tau,
            dominant_archetype=dominant,
            interpretation=interpretation,
            corpus_resonances=resonances,
            timestamp=datetime.now().isoformat(),
        )

    def _generate_interpretation(self, dream_text: str, symbols: List[DreamSymbol],
                                  A: float, S: float, dominant: str,
                                  resonances: List[Dict]) -> str:
        """Generate psychoanalytic interpretation using LLM."""

        system_prompt = """You are a psychoanalyst trained in both Freudian and Jungian dream interpretation.
Analyze dreams by:
1. Identifying manifest content (surface story) vs latent content (hidden meaning)
2. Recognizing archetypal patterns and symbols
3. Connecting to the dreamer's emotional state
4. Suggesting possible meanings without being prescriptive

Be insightful but not dogmatic. Use clear, accessible language."""

        symbol_text = "\n".join([
            f"- {s.raw_text}: A={s.bond.A:.2f} (emotional valence), "
            f"archetype={s.archetype or 'none'}"
            for s in symbols[:8]
        ])

        resonance_text = "\n".join([
            f"- '{r.get('symbol', '')}' appears in {r.get('book', r.get('corpus_follows', ''))}"
            for r in resonances[:5]
        ]) if resonances else "No direct corpus matches found."

        archetype_desc = ARCHETYPES.get(dominant, {}).get("description", "")

        user_prompt = f"""Analyze this dream:

"{dream_text}"

Extracted symbols:
{symbol_text}

Overall emotional valence (A): {A:.2f} (range: -1 negative to +1 positive)
Sacred/mundane dimension (S): {S:.2f} (range: -1 mundane to +1 sacred)
Dominant archetype: {dominant} - {archetype_desc}

Corpus resonances (symbols found in Jung, Freud, mythology texts):
{resonance_text}

Provide a 2-3 paragraph interpretation that:
1. Describes what the dream might be expressing
2. Connects the symbols to possible psychological meaning
3. Suggests what the unconscious might be communicating"""

        return self._call_llm(system_prompt, user_prompt)


def run_interactive_analysis(analyzer: DreamAnalyzer, turns: int = 5):
    """Run interactive dream analysis session."""
    print("\n" + "=" * 60)
    print("STORM-LOGOS DREAM ANALYSIS")
    print("Psychoanalytic interpretation using semantic space")
    print("=" * 60)
    print("\nDescribe your dream. Type 'quit' to exit.\n")

    session_data = {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "analyses": [],
    }

    for turn in range(1, turns + 1):
        print(f"\n--- Dream {turn}/{turns} ---")

        # Get dream
        lines = []
        print("Enter your dream (empty line to finish):")
        while True:
            line = input()
            if line.lower() == 'quit':
                print("\nSession ended.")
                return session_data
            if not line:
                break
            lines.append(line)

        dream_text = " ".join(lines)
        if not dream_text.strip():
            print("No dream entered. Skipping...")
            continue

        print("\nAnalyzing dream...")
        analysis = analyzer.analyze_dream(dream_text)

        # Display results
        print("\n" + "-" * 40)
        print("SYMBOLS EXTRACTED:")
        for s in analysis.symbols[:10]:
            arch = f" [{s.archetype}]" if s.archetype else ""
            print(f"  - {s.raw_text}: A={s.bond.A:+.2f}, S={s.bond.S:+.2f}{arch}")

        print(f"\nOVERALL COORDINATES:")
        print(f"  Emotional valence (A): {analysis.overall_A:+.2f}")
        print(f"  Sacred/mundane (S): {analysis.overall_S:+.2f}")
        print(f"  Abstraction (τ): {analysis.overall_tau:.2f}")
        print(f"  Dominant archetype: {analysis.dominant_archetype}")

        if analysis.corpus_resonances:
            print(f"\nCORPUS RESONANCES:")
            for r in analysis.corpus_resonances[:3]:
                if 'book' in r:
                    print(f"  - '{r['symbol']}' in {r['book']} by {r['author']}")
                elif 'corpus_follows' in r:
                    print(f"  - '{r['symbol']}' → {r['corpus_follows']} (freq: {r['frequency']})")

        print(f"\nINTERPRETATION:")
        print(analysis.interpretation)

        session_data["analyses"].append(analysis.to_dict())

    # Save session
    sessions_dir = Path(__file__).parent.parent / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    filepath = sessions_dir / f"dream_session_{session_data['session_id']}.json"
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)

    print(f"\n\nSession saved to: {filepath}")
    return session_data


def analyze_single_dream(analyzer: DreamAnalyzer, dream_text: str):
    """Analyze a single dream."""
    print("\n" + "=" * 60)
    print("STORM-LOGOS DREAM ANALYSIS")
    print("=" * 60)

    print(f"\nDream: {dream_text[:100]}...")
    print("\nAnalyzing...")

    analysis = analyzer.analyze_dream(dream_text)

    print("\n" + "-" * 40)
    print("SYMBOLS EXTRACTED:")
    for s in analysis.symbols[:10]:
        arch = f" [{s.archetype}]" if s.archetype else ""
        interp = f" - {s.interpretation}" if s.interpretation else ""
        print(f"  - {s.raw_text}: A={s.bond.A:+.2f}, S={s.bond.S:+.2f}{arch}{interp}")

    print(f"\nOVERALL COORDINATES:")
    print(f"  Emotional valence (A): {analysis.overall_A:+.2f}")
    print(f"  Sacred/mundane (S): {analysis.overall_S:+.2f}")
    print(f"  Abstraction (τ): {analysis.overall_tau:.2f}")
    print(f"  Dominant archetype: {analysis.dominant_archetype}")

    if analysis.corpus_resonances:
        print(f"\nCORPUS RESONANCES:")
        for r in analysis.corpus_resonances[:5]:
            if 'book' in r:
                print(f"  - '{r['symbol']}' in {r['book']} by {r['author']}")

    print(f"\nINTERPRETATION:")
    print(analysis.interpretation)

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Dream Analysis using Storm-Logos",
        prog="dream_analysis",
    )

    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive dream analysis session")
    parser.add_argument("--dream", type=str, default=None,
                        help="Analyze a single dream")
    parser.add_argument("--turns", type=int, default=5,
                        help="Number of dreams to analyze in interactive mode")
    parser.add_argument("--model", type=str, default="groq:llama-3.3-70b-versatile",
                        help="LLM model for interpretation")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DreamAnalyzer(model=args.model)
    analyzer.connect()

    if args.dream:
        analyze_single_dream(analyzer, args.dream)
    elif args.interactive:
        run_interactive_analysis(analyzer, args.turns)
    else:
        # Demo dream
        demo_dream = """
        I was walking through a dark forest at night. The trees seemed alive,
        their twisted branches reaching toward me. I could hear water somewhere
        but couldn't find it. Then I saw an old woman with a lantern. She pointed
        toward a cave, and I knew I had to enter it. Inside the cave was a mirror,
        but my reflection wasn't me - it was a child version of myself, crying.
        """
        analyze_single_dream(analyzer, demo_dream)


if __name__ == "__main__":
    main()
