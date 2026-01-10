"""Dream Engine: Dream analysis and therapy application.

Uses Storm-Logos for Jungian dream interpretation:
- Symbol extraction with semantic coordinates
- Archetype detection (configurable patterns)
- Corpus resonance (finding similar themes in literature)
- Interactive dream exploration with LLM

Supports Claude, Groq, and Ollama backends.
"""

from typing import Optional, List, Dict, Any
import os
import json
from datetime import datetime
from pathlib import Path
import requests

from ..data.models import (
    Bond, SemanticState, DreamState, DreamSymbol, DreamAnalysis
)
from ..data.postgres import get_data
from ..data.neo4j import get_neo4j
from ..metrics.analyzers.archetype import get_archetype_analyzer


class DreamEngine:
    """Dream analysis and therapy engine.

    Implements Jungian dream interpretation using:
    1. Symbol extraction from dream text
    2. Coordinate assignment from corpus
    3. Archetype detection (config-based patterns)
    4. Corpus resonance search
    5. LLM-assisted interpretation
    """

    def __init__(self,
                 model: str = 'groq:llama-3.3-70b-versatile',
                 api_key: str = None):
        """Initialize dream engine.

        Args:
            model: 'claude' for Claude API, 'groq:model-name' for Groq API,
                   or Ollama model name (e.g., 'mistral:7b')
            api_key: API key (reads from env if not provided)
        """
        self.model = model
        self.use_claude = model.lower().startswith('claude')
        self.use_groq = model.lower().startswith('groq:')

        if self.use_claude:
            self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = None
        elif self.use_groq:
            self.api_key = api_key or os.environ.get('GROQ_API_KEY')
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not set")
            self.groq_model = model.split(':', 1)[1]
            self._client = None
        else:
            self.base_url = 'http://localhost:11434'
            self._client = None

        # Data sources
        self._data = None
        self._neo4j = None
        self._nlp = None

        # Session tracking
        self._session_start = datetime.now()
        self._analyses: List[DreamAnalysis] = []

    def connect(self) -> bool:
        """Initialize connections to data sources."""
        self._data = get_data()
        self._neo4j = get_neo4j()

        if not self._neo4j.connect():
            print("Warning: Neo4j not connected (corpus search disabled)")

        # Set LLM caller on archetype analyzer for dynamic detection
        archetype_analyzer = get_archetype_analyzer()
        archetype_analyzer.set_llm_caller(self._call_llm_short)

        return True

    def _call_llm_short(self, system: str, user: str) -> str:
        """Short LLM call for archetype detection."""
        return self._call_llm(system, user, max_tokens=100)

    def _get_nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract_symbols(self, text: str) -> List[DreamSymbol]:
        """Extract dream symbols from text.

        Uses spaCy to find noun-adjective pairs and significant nouns,
        then assigns semantic coordinates from corpus.

        Args:
            text: Dream narrative text

        Returns:
            List of DreamSymbol objects with coordinates
        """
        nlp = self._get_nlp()
        doc = nlp(text)
        symbols = []
        seen = set()

        archetype_analyzer = get_archetype_analyzer()

        for token in doc:
            # Adjective + Noun pairs
            if token.pos_ == "ADJ" and token.dep_ == "amod":
                if token.head.pos_ == "NOUN":
                    adj = token.lemma_.lower()
                    noun = token.head.lemma_.lower()
                    key = f"{adj}_{noun}"

                    if key not in seen:
                        seen.add(key)
                        bond, arch, interp = self._create_symbol(adj, noun, text)
                        if bond:
                            symbols.append(DreamSymbol(
                                bond=bond,
                                raw_text=f"{adj} {noun}",
                                archetype=arch,
                                interpretation=interp,
                            ))

            # Significant nouns
            elif token.pos_ == "NOUN" and token.dep_ in ("nsubj", "dobj", "pobj"):
                noun = token.lemma_.lower()
                if noun not in seen and len(noun) > 2:
                    coords = self._data.get(noun) if self._data else None
                    if coords:
                        seen.add(noun)
                        bond = Bond(
                            noun=noun,
                            A=coords.A, S=coords.S, tau=coords.tau,
                        )
                        arch, interp = archetype_analyzer.get_symbol_interpretation(bond)
                        symbols.append(DreamSymbol(
                            bond=bond,
                            raw_text=noun,
                            archetype=arch,
                            interpretation=interp,
                        ))

        return symbols[:15]  # Limit

    def _create_symbol(self, adj: str, noun: str, context: str) -> tuple:
        """Create a DreamSymbol from adjective+noun pair.

        Returns:
            (Bond, archetype, interpretation) or (None, "", "")
        """
        adj_coords = self._data.get(adj) if self._data else None
        noun_coords = self._data.get(noun) if self._data else None

        if adj_coords and noun_coords:
            A = (adj_coords.A + noun_coords.A) / 2
            S = (adj_coords.S + noun_coords.S) / 2
            tau = (adj_coords.tau + noun_coords.tau) / 2
        elif adj_coords:
            A, S, tau = adj_coords.A, adj_coords.S, adj_coords.tau
        elif noun_coords:
            A, S, tau = noun_coords.A, noun_coords.S, noun_coords.tau
        else:
            return None, "", ""

        bond = Bond(noun=noun, adj=adj, A=A, S=S, tau=tau)

        archetype_analyzer = get_archetype_analyzer()
        arch, interp = archetype_analyzer.get_symbol_interpretation(bond)

        return bond, arch, interp

    def find_corpus_resonances(self, symbols: List[DreamSymbol],
                                limit: int = 5) -> List[Dict]:
        """Find corpus passages that resonate with dream symbols.

        Args:
            symbols: List of DreamSymbols to search
            limit: Maximum resonances to return

        Returns:
            List of dicts with book, author, bond info
        """
        if not self._neo4j or not self._neo4j._connected:
            return []

        resonances = []

        for sym in symbols[:5]:
            text = sym.raw_text
            bond_id = text.replace(" ", "_")

            query = """
            MATCH (book:Book)-[:CONTAINS]->(bond:Bond)
            WHERE bond.id CONTAINS $term
            RETURN book.title as book, book.author as author,
                   bond.id as bond
            LIMIT 3
            """

            try:
                with self._neo4j._driver.session() as session:
                    result = session.run(query, term=bond_id)
                    for record in result:
                        resonances.append({
                            "symbol": text,
                            "book": record["book"],
                            "author": record["author"],
                            "bond": record["bond"],
                        })
            except Exception:
                pass

        return resonances[:limit]

    def analyze(self, dream_text: str) -> DreamAnalysis:
        """Perform full dream analysis.

        Args:
            dream_text: The dream narrative

        Returns:
            DreamAnalysis with symbols, state, and interpretation
        """
        # Extract symbols
        symbols = self.extract_symbols(dream_text)

        # Convert to bonds for archetype analysis
        bonds = [s.bond for s in symbols]

        # Create dream state
        archetype_analyzer = get_archetype_analyzer()
        state = archetype_analyzer.create_dream_state(dream_text, bonds)

        # Find corpus resonances
        resonances = self.find_corpus_resonances(symbols)

        # Store in symbols
        for sym in symbols:
            sym.corpus_sources = [
                r["book"] for r in resonances if r["symbol"] == sym.raw_text
            ]

        # Generate interpretation
        interpretation = self._generate_interpretation(
            dream_text, symbols, state, resonances
        )

        analysis = DreamAnalysis(
            dream_text=dream_text,
            symbols=symbols,
            state=state,
            interpretation=interpretation,
            corpus_resonances=resonances,
            timestamp=datetime.now().isoformat(),
        )

        self._analyses.append(analysis)
        return analysis

    def _generate_interpretation(self,
                                  dream_text: str,
                                  symbols: List[DreamSymbol],
                                  state: DreamState,
                                  resonances: List[Dict]) -> str:
        """Generate LLM interpretation of dream.

        Args:
            dream_text: Original dream
            symbols: Extracted symbols
            state: DreamState with archetype scores
            resonances: Corpus resonances

        Returns:
            Interpretation text
        """
        # Build symbol summary
        symbol_info = []
        for s in symbols[:8]:
            info = f"- {s.raw_text} (A={s.bond.A:+.2f}, S={s.bond.S:+.2f})"
            if s.archetype:
                info += f" [{s.archetype}]"
            symbol_info.append(info)

        # Build archetype summary
        dominant, score = state.dominant_archetype()
        arch_summary = f"Dominant archetype: {dominant} ({score:.2f})"

        # Build resonance summary
        res_info = []
        for r in resonances[:3]:
            res_info.append(f"- '{r['symbol']}' appears in {r['book']} by {r['author']}")

        system = """You are a Jungian dream analyst. Interpret dreams using:
- Archetypal psychology (shadow, anima/animus, self, mother, father, hero, trickster)
- Symbol amplification (connecting personal symbols to universal patterns)
- Emotional resonance (what feelings does the dream evoke?)

Be concise (2-3 paragraphs). Focus on psychological meaning, not literal interpretation.
Avoid clichés. Connect symbols to possible psychological processes."""

        prompt = f"""DREAM:
{dream_text}

SYMBOLS DETECTED:
{chr(10).join(symbol_info)}

ARCHETYPE ANALYSIS:
{arch_summary}
Coordinates: A={state.A:+.2f} (affirmation), S={state.S:+.2f} (sacred)
{'Transformation present' if state.transformation > 0 else ''}
{'Journey elements' if state.journey > 0 else ''}
{'Confrontation themes' if state.confrontation > 0 else ''}

{('CORPUS RESONANCES:' + chr(10) + chr(10).join(res_info)) if res_info else ''}

Provide a psychological interpretation of this dream."""

        return self._call_llm(system, prompt)

    def _call_llm(self, system: str, user: str, max_tokens: int = 512) -> str:
        """Call LLM for interpretation."""
        if self.use_claude:
            return self._call_claude(system, user, max_tokens)
        elif self.use_groq:
            return self._call_groq(system, user, max_tokens)
        else:
            return self._call_ollama(system, user, max_tokens)

    def _call_claude(self, system: str, user: str, max_tokens: int) -> str:
        """Call Claude API."""
        try:
            import anthropic

            if self._client is None:
                self._client = anthropic.Anthropic(api_key=self.api_key)

            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text.strip()

        except Exception as e:
            return f"[Claude error: {e}]"

    def _call_groq(self, system: str, user: str, max_tokens: int) -> str:
        """Call Groq API."""
        try:
            from groq import Groq

            if self._client is None:
                self._client = Groq(api_key=self.api_key)

            response = self._client.chat.completions.create(
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

    def _call_ollama(self, system: str, user: str, max_tokens: int) -> str:
        """Call Ollama API."""
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
            return "[Unable to generate response]"

        except Exception as e:
            return f"[Ollama error: {e}]"

    def explore(self, dream_text: str, question: str) -> str:
        """Explore a specific aspect of a dream.

        Args:
            dream_text: The dream to explore
            question: Question about the dream

        Returns:
            LLM response
        """
        # Get basic analysis
        symbols = self.extract_symbols(dream_text)
        symbol_info = [f"- {s.raw_text}: {s.archetype or 'unknown'}" for s in symbols[:5]]

        system = """You are a Jungian dream analyst engaging in dream exploration.
Answer questions about dreams using depth psychology.
Be curious, not dogmatic. Dreams have multiple meanings.
Connect symbols to archetypal patterns where appropriate."""

        prompt = f"""DREAM:
{dream_text}

SYMBOLS:
{chr(10).join(symbol_info) if symbol_info else 'No clear symbols extracted'}

QUESTION:
{question}"""

        return self._call_llm(system, prompt, max_tokens=256)

    def get_session_data(self) -> Dict[str, Any]:
        """Get full session data for export."""
        return {
            'session_id': self._session_start.strftime('%Y%m%d_%H%M%S'),
            'start_time': self._session_start.isoformat(),
            'end_time': datetime.now().isoformat(),
            'model': self.model,
            'n_analyses': len(self._analyses),
            'analyses': [a.as_dict() for a in self._analyses],
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
        filename = f"dream_session_{session_data['session_id']}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        return str(filepath)

    def reset(self):
        """Reset session state."""
        self._analyses.clear()
        self._session_start = datetime.now()
