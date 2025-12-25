"""
Renderer: MeaningTree -> LLM Prompt -> Response

Converts meaning trees into structured prompts for LLM generation,
then invokes the LLM to render natural language.
"""

import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent paths for imports
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from models.types import MeaningTree, MeaningNode


@dataclass
class RendererConfig:
    """Configuration for rendering."""
    model: str = "mistral"           # Ollama model name
    temperature: float = 0.7
    max_tokens: int = 512
    include_properties: bool = True  # Include g, tau in prompt
    include_paths: bool = True       # Include all paths
    include_transitions: bool = True # Include verb transitions
    system_prompt: str = ""


class TreeSerializer:
    """Serializes MeaningTree into text formats for LLM prompts."""

    @staticmethod
    def to_paths_text(tree: MeaningTree) -> str:
        """Convert tree to path-based text representation."""
        lines = []

        for path in tree.get_paths():
            lines.append(" -> ".join(path))

        return "\n".join(lines)

    @staticmethod
    def to_transitions_text(tree: MeaningTree) -> str:
        """Convert tree to transition-based text representation."""
        lines = []

        for from_word, verb, to_word in tree.get_transitions():
            lines.append(f"{from_word} --({verb})--> {to_word}")

        return "\n".join(lines)

    @staticmethod
    def to_hierarchical_text(tree: MeaningTree, include_props: bool = True) -> str:
        """Convert tree to hierarchical text representation."""
        lines = []

        def render_node(node: MeaningNode, indent: int = 0):
            prefix = "  " * indent
            verb_str = f"({node.verb_from_parent}) " if node.verb_from_parent else ""

            if include_props:
                props = f"[g={node.g:.2f}, τ={node.tau:.1f}]"
                lines.append(f"{prefix}{verb_str}{node.word} {props}")
            else:
                lines.append(f"{prefix}{verb_str}{node.word}")

            for child in node.children:
                render_node(child, indent + 1)

        for i, root in enumerate(tree.roots):
            if i > 0:
                lines.append("")  # Separator between trees
            lines.append(f"# Root: {root.word}")
            render_node(root)

        return "\n".join(lines)

    @staticmethod
    def to_concepts_list(tree: MeaningTree) -> str:
        """Get unique concepts as comma-separated list."""
        return ", ".join(tree.all_words())

    @staticmethod
    def to_json(tree: MeaningTree) -> str:
        """Convert tree to JSON for structured prompts."""
        return json.dumps(tree.to_dict(), indent=2)


class Renderer:
    """
    Renders meaning trees into natural language via LLM.

    Uses the semantic structure of the tree to guide LLM generation,
    ensuring responses are grounded in the meaning space.
    """

    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config or RendererConfig()
        self.serializer = TreeSerializer()

        # LLM client (lazy loaded)
        self._ollama = None

    def _get_ollama(self):
        """Get Ollama client."""
        if self._ollama is None:
            try:
                import ollama
                self._ollama = ollama
            except ImportError:
                raise ImportError("ollama package required. Install with: pip install ollama")
        return self._ollama

    def _compute_ranges(self, tree: MeaningTree) -> Dict[str, Tuple[float, float]]:
        """Compute actual g and tau ranges from the tree."""
        nodes = tree.all_nodes()
        if not nodes:
            return {'g': (-1, 1), 'tau': (1, 7)}

        g_vals = [n.g for n in nodes if n.g is not None]
        tau_vals = [n.tau for n in nodes if n.tau is not None]

        return {
            'g': (min(g_vals) if g_vals else -1, max(g_vals) if g_vals else 1),
            'tau': (min(tau_vals) if tau_vals else 1, max(tau_vals) if tau_vals else 7)
        }

    def _relative_position(self, val: float, min_val: float, max_val: float) -> str:
        """Get relative position in range."""
        if max_val == min_val:
            return "mid"
        normalized = (val - min_val) / (max_val - min_val)
        if normalized > 0.7:
            return "high"
        elif normalized < 0.3:
            return "low"
        return "mid"

    def _get_dominant_themes(self, tree: MeaningTree) -> List[str]:
        """Extract dominant themes from j-vectors."""
        # Sum j-vectors to get dominant direction
        j_sum = [0.0] * 5
        count = 0

        for node in tree.all_nodes():
            if node.j is not None:
                for i, val in enumerate(node.j):
                    j_sum[i] += val
                count += 1

        if count == 0:
            return []

        # Normalize and interpret
        j_avg = [x / count for x in j_sum]
        dimensions = ['beauty', 'life', 'sacred', 'good', 'love']
        themes = []

        for i, (name, val) in enumerate(zip(dimensions, j_avg)):
            if val > 0.1:
                themes.append(f"strong {name}")
            elif val < -0.1:
                themes.append(f"lack of {name}")

        return themes[:3]  # Top 3 themes

    def _collapse_quality(self, tree: MeaningTree) -> Dict[str, float]:
        """
        Measure quality of semantic collapse.

        Returns metrics that indicate if we have enough information:
        - specificity: how specific/concrete the concepts are (based on tau variance)
        - richness: how many meaningful concepts we extracted
        - coherence: how aligned the concepts are (j-vector variance)
        """
        nodes = tree.all_nodes()

        if not nodes:
            return {'specificity': 0.0, 'richness': 0.0, 'coherence': 0.0}

        # Richness: number of meaningful nodes (normalized)
        richness = min(1.0, len(nodes) / 10.0)

        # Specificity: inverse of tau variance (consistent abstraction = specific)
        tau_vals = [n.tau for n in nodes if n.tau is not None]
        if len(tau_vals) > 1:
            tau_var = sum((t - sum(tau_vals)/len(tau_vals))**2 for t in tau_vals) / len(tau_vals)
            specificity = 1.0 / (1.0 + tau_var)
        else:
            specificity = 0.5  # Single concept - medium specificity

        # Coherence: j-vector alignment (low variance = coherent)
        j_vectors = [n.j for n in nodes if n.j is not None and len(n.j) == 5]
        if len(j_vectors) > 1:
            import numpy as np
            j_center = np.mean(j_vectors, axis=0)
            j_var = np.mean([np.linalg.norm(j - j_center) for j in j_vectors])
            coherence = 1.0 / (1.0 + j_var)
        else:
            coherence = 0.5

        return {
            'specificity': specificity,
            'richness': richness,
            'coherence': coherence
        }

    def build_prompt(self, tree: MeaningTree, user_query: str) -> str:
        """
        Build LLM prompt from meaning tree using semantic collapse quality.

        The collapse quality metrics tell the LLM how much information
        is available and whether to ask for more or proceed with interpretation.
        """
        sections = []

        # Measure collapse quality
        quality = self._collapse_quality(tree)
        summary = tree.summary()
        ranges = self._compute_ranges(tree)
        themes = self._get_dominant_themes(tree)

        # Semantic space context
        g_min, g_max = ranges['g']
        avg_g = summary['avg_goodness']

        sections.append("## Collapse Quality")
        sections.append(f"Richness: {quality['richness']:.2f} (concepts extracted)")
        sections.append(f"Specificity: {quality['specificity']:.2f} (content detail)")
        sections.append(f"Coherence: {quality['coherence']:.2f} (thematic focus)")
        sections.append("")

        # Concepts extracted
        if tree.roots:
            sections.append("## Concepts")
            for root in tree.roots[:3]:
                sections.append(f"- {root.word}")

            # Show some connections if available
            transitions = tree.get_transitions()[:4]
            if transitions:
                sections.append("")
                for from_word, verb, to_word in transitions:
                    sections.append(f"  {from_word} --{verb}--> {to_word}")
            sections.append("")

        # Themes from j-space
        if themes:
            sections.append(f"Themes: {', '.join(themes)}")
            sections.append("")

        # Query
        sections.append(f"Query: {user_query}")
        sections.append("")

        # Guidance based on collapse quality
        sections.append("## Guidance")
        if quality['richness'] < 0.3:
            sections.append("- Low richness: ask what specific information the user can share")
        if quality['specificity'] < 0.4:
            sections.append("- Low specificity: ask for concrete details")
        if quality['richness'] >= 0.3 and quality['specificity'] >= 0.4:
            sections.append("- Good collapse: respond using the concepts and connections")

        sections.append("- Be concise (2-3 sentences)")

        return "\n".join(sections)

    def build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        if self.config.system_prompt:
            return self.config.system_prompt

        return """You respond based on semantic meaning structures.
Follow the concept connections naturally. Match the tone to the semantic context.
Be concise and insightful, not mechanical."""

    def render(self, tree: MeaningTree, user_query: str) -> Dict[str, Any]:
        """
        Render meaning tree into natural language response.

        Args:
            tree: The meaning tree
            user_query: Original user question

        Returns:
            {
                'response': str,           # Generated text
                'prompt': str,             # Prompt used
                'tree_summary': dict,      # Tree statistics
                'model': str,              # Model used
            }
        """
        ollama = self._get_ollama()

        # Build prompts
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_prompt(tree, user_query)

        # Call LLM
        try:
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )

            generated_text = response['message']['content']

        except Exception as e:
            generated_text = f"[Error generating response: {e}]"

        return {
            'response': generated_text,
            'prompt': user_prompt,
            'system_prompt': system_prompt,
            'tree_summary': tree.summary(),
            'model': self.config.model,
            'temperature': self.config.temperature,
        }

    def render_stream(self, tree: MeaningTree, user_query: str):
        """
        Render with streaming output.

        Yields chunks of the response as they're generated.
        """
        ollama = self._get_ollama()

        system_prompt = self.build_system_prompt()
        user_prompt = self.build_prompt(tree, user_query)

        try:
            stream = ollama.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                },
                stream=True
            )

            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']

        except Exception as e:
            yield f"[Error: {e}]"
