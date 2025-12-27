"""
Monte Carlo Renderer: Statistical Semantic Rendering

Instead of one path through semantic space, sample many paths
and use statistics to build response structure.

The shape of the semantic landscape informs the response:
- TOP ATTRACTORS become the CORE of the answer
- ORBITAL STRUCTURE gives HIERARCHY (concrete → abstract)
- COHERENCE indicates CONFIDENCE
- CONCENTRATION indicates FOCUS level
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_MEANING_CHAIN.parent.parent))

from chain_core.semantic_laser import SemanticLaser
from chain_core.decomposer import Decomposer


@dataclass
class SemanticLandscape:
    """The statistical shape of semantic space for a question."""
    question: str
    seeds: List[str]
    intent_verbs: List[str]
    n_samples: int

    # Core statistics
    unique_words: int
    concentration: float  # How focused (top-10 / total)
    coherence: float      # Average beam coherence
    lasing_rate: float    # Fraction of samples that achieved lasing

    # Attractors by stability
    core_attractors: List[tuple]       # (word, count) - top 10, stable
    peripheral_attractors: List[tuple] # (word, count) - 11-30, variable

    # Orbital structure (concrete → abstract)
    orbital_map: Dict[int, List[tuple]]  # n -> [(word, count, tau), ...]

    # Full distribution
    tau_mean: float
    tau_std: float
    intent_influence: float


class MonteCarloRenderer:
    """
    Render responses using Monte Carlo semantic sampling.

    Instead of following one path, we throw the question into
    semantic space many times and see where it lands.

    The statistics reveal:
    - What concepts are STABLE (appear often) vs VARIABLE (appear sometimes)
    - What level of abstraction the question lives at
    - How coherent/focused the answer should be
    """

    def __init__(self, intent_strength: float = 0.3, n_samples: int = 30):
        """
        Args:
            intent_strength: α parameter (0.3 = soft wind recommended)
            n_samples: Number of samples per question (30 is fast but informative)
        """
        self.intent_strength = intent_strength
        self.n_samples = n_samples
        self.laser = SemanticLaser(intent_strength=intent_strength)
        self.decomposer = Decomposer(include_proper_nouns=True)

        # LLM client (lazy)
        self._ollama = None

    def _get_ollama(self):
        if self._ollama is None:
            try:
                import ollama
                self._ollama = ollama
            except ImportError:
                raise ImportError("ollama required: pip install ollama")
        return self._ollama

    def sample_landscape(self, question: str, n_samples: int = None) -> SemanticLandscape:
        """
        Sample semantic space N times to discover its shape.

        Returns a SemanticLandscape with statistics about where
        the question tends to land.
        """
        n = n_samples if n_samples is not None else self.n_samples

        # Decompose question
        decomposed = self.decomposer.decompose(question)
        seeds = decomposed.nouns if decomposed.nouns else ['meaning']
        intent_verbs = decomposed.verbs

        # Collect statistics across samples
        word_counts = defaultdict(int)
        word_taus = defaultdict(list)
        coherences = []
        lasing_count = 0
        intent_influences = []

        for _ in range(n):
            # Run laser with intent
            result = self.laser.lase(
                seeds=seeds,
                pump_power=10,
                pump_depth=5,
                coherence_threshold=0.3,
                min_cluster_size=3,
                intent_verbs=intent_verbs
            )

            # Collect beam coherence
            if result['beams']:
                coherences.append(np.mean([b.coherence for b in result['beams']]))
                lasing_count += 1

            # Collect word frequencies and taus
            for word, state in result['excited'].items():
                word_counts[word] += state.visits
                word_taus[word].append(state.tau)

            # Collect intent influence
            pop = result['population']
            intent_influences.append(pop.get('intent_fraction', 0))

        # Compute aggregate statistics
        total_visits = sum(word_counts.values())

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

        # Core (stable) vs peripheral (variable) attractors
        core = sorted_words[:10]
        peripheral = sorted_words[10:30]

        # Concentration: how much top-10 captures
        top10_visits = sum(c for _, c in core)
        concentration = top10_visits / total_visits if total_visits > 0 else 0

        # Orbital map
        orbital_map = defaultdict(list)
        for word, count in sorted_words[:50]:
            if word in word_taus and len(word_taus[word]) >= 3:
                tau = np.mean(word_taus[word])
                orbital = int(round((tau - 1) * np.e))
                orbital_map[orbital].append((word, count, round(tau, 2)))

        # Sort within orbitals
        for orbital_n in orbital_map:
            orbital_map[orbital_n].sort(key=lambda x: -x[1])

        # Tau statistics
        all_taus = []
        for taus in word_taus.values():
            all_taus.extend(taus)
        tau_mean = np.mean(all_taus) if all_taus else 2.0
        tau_std = np.std(all_taus) if all_taus else 0.5

        return SemanticLandscape(
            question=question,
            seeds=seeds,
            intent_verbs=intent_verbs,
            n_samples=n,
            unique_words=len(word_counts),
            concentration=concentration,
            coherence=np.mean(coherences) if coherences else 0,
            lasing_rate=lasing_count / n,
            core_attractors=core,
            peripheral_attractors=peripheral,
            orbital_map=dict(orbital_map),
            tau_mean=tau_mean,
            tau_std=tau_std,
            intent_influence=np.mean(intent_influences) if intent_influences else 0
        )

    def build_prompt(self, landscape: SemanticLandscape) -> str:
        """
        Build LLM prompt from semantic landscape statistics.

        The prompt structure reflects what Monte Carlo revealed:
        - Core attractors become required concepts
        - Orbital structure suggests response flow
        - Coherence/concentration guide confidence
        """
        sections = []

        # Header with landscape shape
        sections.append("## Semantic Landscape (Monte Carlo)")
        sections.append(f"Question: {landscape.question}")
        sections.append(f"Samples: {landscape.n_samples} | Lasing: {landscape.lasing_rate:.0%}")
        sections.append(f"Coherence: {landscape.coherence:.2f} | Focus: {landscape.concentration:.0%}")
        sections.append("")

        # Core attractors - these MUST appear in response
        sections.append("## Core Concepts (stable across samples)")
        core_words = [w for w, _ in landscape.core_attractors[:7]]
        sections.append(f"These concepts consistently appeared: {', '.join(core_words)}")
        sections.append("")

        # Orbital structure - flow from concrete to abstract
        sections.append("## Concept Hierarchy (by abstraction)")
        for n in sorted(landscape.orbital_map.keys()):
            words_at_n = landscape.orbital_map[n][:5]
            word_list = [w for w, _, _ in words_at_n]
            level = "concrete" if n <= 1 else ("mid-level" if n <= 3 else "abstract")
            sections.append(f"  {level} (n={n}): {', '.join(word_list)}")
        sections.append("")

        # Peripheral concepts - can mention but not required
        if landscape.peripheral_attractors:
            peripheral_words = [w for w, _ in landscape.peripheral_attractors[:10]]
            sections.append(f"## Optional Context: {', '.join(peripheral_words)}")
            sections.append("")

        # Response guidance based on landscape shape
        sections.append("## Response Guidance")

        if landscape.coherence > 0.75:
            sections.append("- High coherence: Give a focused, confident answer")
        elif landscape.coherence > 0.5:
            sections.append("- Medium coherence: Answer with some nuance")
        else:
            sections.append("- Low coherence: Acknowledge multiple perspectives")

        if landscape.concentration > 0.3:
            sections.append("- High concentration: Stay close to core concepts")
        else:
            sections.append("- Low concentration: Feel free to explore connections")

        # Orbital-based structure suggestion
        if 0 in landscape.orbital_map or 1 in landscape.orbital_map:
            sections.append("- Start with concrete/grounded concepts, then expand")

        sections.append("- Use the core concepts naturally, don't list them")
        sections.append("- 2-4 sentences, substantive not generic")

        return "\n".join(sections)

    def build_system_prompt(self) -> str:
        """System prompt for Monte Carlo rendering."""
        return """You respond based on statistical semantic mapping.

The Monte Carlo process revealed where the question's meaning lives in semantic space.
Core concepts appeared consistently across many samples - use them naturally.
The hierarchy shows how to structure: concrete → abstract.

Respond thoughtfully, using the discovered concepts.
Be substantive, not formulaic."""

    def render(self, question: str, model: str = "mistral:7b",
               temperature: float = 0.7, stream: bool = False) -> Dict[str, Any]:
        """
        Render a response using Monte Carlo semantic sampling.

        Args:
            question: User's question
            model: LLM model to use
            temperature: Generation temperature
            stream: If True, return generator

        Returns:
            {
                'response': str,
                'landscape': SemanticLandscape,
                'prompt': str,
                'core_concepts': List[str]
            }
        """
        # Sample the landscape
        print(f"[MC-Render] Sampling semantic landscape ({self.n_samples} samples)...")
        landscape = self.sample_landscape(question)
        print(f"[MC-Render] Found {landscape.unique_words} unique concepts, "
              f"coherence={landscape.coherence:.2f}, focus={landscape.concentration:.0%}")

        # Build prompt
        prompt = self.build_prompt(landscape)
        system_prompt = self.build_system_prompt()

        # Generate response
        ollama = self._get_ollama()

        if stream:
            return self._render_stream(ollama, model, system_prompt, prompt, landscape)

        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": 512
                }
            )
            generated = response['message']['content']
        except Exception as e:
            generated = f"[Error: {e}]"

        return {
            'response': generated,
            'landscape': landscape,
            'prompt': prompt,
            'core_concepts': [w for w, _ in landscape.core_attractors[:7]]
        }

    def _render_stream(self, ollama, model, system_prompt, prompt, landscape):
        """Streaming render."""
        try:
            stream = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.7, "num_predict": 512},
                stream=True
            )
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"[Error: {e}]"

    def close(self):
        self.laser.close()


# Demo
def demo():
    """Demonstrate Monte Carlo rendering."""
    renderer = MonteCarloRenderer(intent_strength=0.3, n_samples=30)

    questions = [
        "What is the meaning of life?",
        "What is a tree?",
        "What do my dreams mean?"
    ]

    for q in questions:
        print("\n" + "="*70)
        print(f"QUESTION: {q}")
        print("="*70)

        result = renderer.render(q)

        print(f"\nCORE CONCEPTS: {result['core_concepts']}")
        print(f"\nRESPONSE:\n{result['response']}")

    renderer.close()


if __name__ == "__main__":
    demo()
