#!/usr/bin/env python3
"""
Meaning Chain Chat

A standalone chat application that uses meaning trees for semantic navigation.

Pipeline:
    User Query -> Decompose -> Build Tree -> Render -> Feedback -> Response

Usage:
    python /path/to/experiments/meaning_chain/app/chat.py

Or as library:
    from experiments.meaning_chain.app.chat import MeaningChainChat
    chat = MeaningChainChat()
    result = chat.respond("What is the meaning of love?")
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add parent paths for imports - resolve all paths from this file
_THIS_FILE = Path(__file__).resolve()
_APP_DIR = _THIS_FILE.parent
_MEANING_CHAIN = _APP_DIR.parent
_EXPERIMENTS = _MEANING_CHAIN.parent
_SEMANTIC_LLM = _EXPERIMENTS.parent

# Add semantic_llm to path (for core.data_loader)
if str(_SEMANTIC_LLM) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_LLM))

# Add meaning_chain to path (for models.types)
if str(_MEANING_CHAIN) not in sys.path:
    sys.path.insert(0, str(_MEANING_CHAIN))

# Import from semantic_llm/core
from core.data_loader import DataLoader

# Import from meaning_chain/chain_core
from chain_core.decomposer import Decomposer
from chain_core.tree_builder import TreeBuilder, TreeBuilderConfig
from chain_core.renderer import Renderer, RendererConfig
from chain_core.feedback import FeedbackAnalyzer, FeedbackConfig, FeedbackResult
from chain_core.storm_logos import StormLogosBuilder, LogosPattern
from models.types import MeaningTree

# Import conversation learner
from graph.conversation_learner import ConversationLearner


@dataclass
class ChatConfig:
    """Configuration for the chat system."""
    # Tree building (fallback)
    max_depth: int = 3
    max_children: int = 4

    # Storm-Logos (biological emergence)
    use_storm_logos: bool = True          # Use storm-logos instead of brute force
    storm_temperature: float = 1.5        # Controls chaos in storm phase
    n_walks: int = 5                      # Parallel walks per seed
    steps_per_walk: int = 8               # Steps in each walk

    # Rendering
    model: str = "mistral:7b"
    temperature: float = 0.7
    max_tokens: int = 512

    # Feedback (for validation, not regeneration)
    min_alignment: float = 0.5

    # Learning
    learn_from_conversation: bool = True  # Learn SVO patterns from exchanges

    # Output
    verbose: bool = True
    show_tree: bool = True


class MeaningChainChat:
    """
    Complete chat system using meaning chains.

    Combines all components:
    - Decomposer: Parse input into concepts
    - TreeBuilder: Build meaning tree
    - Renderer: Generate response via LLM
    - Feedback: Validate and potentially regenerate
    """

    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()

        # Initialize data loader (shared)
        self.loader = DataLoader()

        # Initialize components
        self.decomposer = Decomposer(self.loader)

        self.tree_builder = TreeBuilder(
            self.loader,
            TreeBuilderConfig(
                max_depth=self.config.max_depth,
                max_children=self.config.max_children
            )
        )

        self.renderer = Renderer(
            RendererConfig(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        )

        self.feedback = FeedbackAnalyzer(
            self.loader,
            FeedbackConfig(
                min_alignment_score=self.config.min_alignment
            )
        )

        # Storm-Logos builder (biological emergence)
        self.storm_logos = None
        if self.config.use_storm_logos:
            self.storm_logos = StormLogosBuilder(
                storm_temperature=self.config.storm_temperature,
                n_walks=self.config.n_walks,
                steps_per_walk=self.config.steps_per_walk
            )

        # Conversation learner (if enabled)
        self.learner = None
        if self.config.learn_from_conversation:
            try:
                self.learner = ConversationLearner()
                if self.learner.graph.is_connected():
                    self._log("[Chat] Conversation learning enabled")
                else:
                    self.learner = None
            except Exception as e:
                self._log(f"[Chat] Learning disabled: {e}")
                self.learner = None

        # Conversation history
        self.history: List[Dict[str, Any]] = []

    def _log(self, message: str):
        """Log message if verbose."""
        if self.config.verbose:
            print(message)

    def respond(self, query: str) -> Dict[str, Any]:
        """
        Generate response to user query using meaning chain.

        Args:
            query: User's input

        Returns:
            {
                'response': str,
                'tree': MeaningTree,
                'feedback': FeedbackResult,
                'attempts': int,
                'metadata': dict
            }
        """
        self._log(f"\n{'='*60}")
        self._log(f"Query: {query}")
        self._log(f"{'='*60}")

        # Step 1: Decompose
        self._log("\n[1] Decomposing...")
        decomposed = self.decomposer.decompose(query)
        self._log(f"    Nouns: {decomposed.nouns}")
        self._log(f"    Verbs: {decomposed.verbs}")
        if decomposed.unknown_words:
            self._log(f"    Unknown: {decomposed.unknown_words}")

        if not decomposed.nouns:
            return {
                'response': "I couldn't identify any concepts in your query. "
                           "Could you rephrase?",
                'tree': None,
                'feedback': None,
                'attempts': 0,
                'metadata': {'error': 'no_concepts'}
            }

        # Step 2: Build tree using Storm-Logos or fallback
        pattern = None
        if self.storm_logos:
            self._log("\n[2] STORM phase (chaotic associations)...")
            tree, pattern = self.storm_logos.build(
                decomposed.nouns,
                decomposed.verbs,
                query
            )
            self._log(f"    Convergence: {pattern.convergence_point}")
            self._log(f"    Core concepts: {pattern.core_concepts[:5]}")
            self._log(f"    Coherence: {pattern.coherence:.2f}")

            self._log("\n[3] LOGOS phase (pattern extracted)")
            self._log(f"    G direction: {pattern.g_direction:+.2f}")
            self._log(f"    Tau level: {pattern.tau_level:.1f}")
        else:
            self._log("\n[2] Building meaning tree (fallback)...")
            tree = self.tree_builder.build_from_decomposition(
                decomposed.nouns,
                decomposed.verbs,
                query
            )

        if self.config.show_tree:
            self._log(f"\n{tree.pretty_print()}")

        summary = tree.summary()
        self._log(f"    Nodes: {summary['total_nodes']}, Paths: {summary['paths']}")

        # Step 4: Render ONE response from the pattern
        self._log("\n[4] Rendering response...")
        render_result = self.renderer.render(tree, query)
        final_response = render_result['response']

        # Validate with feedback (but don't regenerate - pattern is the answer)
        final_feedback = self.feedback.analyze(final_response, tree)
        self._log(f"    Alignment: {final_feedback.alignment_score:.0%}")
        self._log(f"    Coverage: {final_feedback.concept_coverage:.0%}")

        # Build result
        result = {
            'response': final_response,
            'tree': tree,
            'feedback': final_feedback,
            'pattern': pattern,  # The logos pattern
            'metadata': {
                'model': self.config.model,
                'tree_summary': summary,
                'decomposed': {
                    'nouns': decomposed.nouns,
                    'verbs': decomposed.verbs
                }
            }
        }

        # Add to history
        self.history.append({
            'query': query,
            'result': result
        })

        # Learn from this exchange (if enabled)
        if self.learner:
            learn_stats = self.learner.learn_from_exchange(query, final_response)
            if learn_stats['total'] > 0:
                self._log(f"\n[5] Learning: {learn_stats['total']} patterns "
                          f"({learn_stats['new']} new, {learn_stats['reinforced']} reinforced)")
            result['metadata']['learned'] = learn_stats

        # Final output
        self._log(f"\n{'='*60}")
        self._log("RESPONSE:")
        self._log(f"{'='*60}")
        self._log(final_response)
        if pattern:
            self._log(f"\n[Coherence: {pattern.coherence:.0%} | "
                      f"Alignment: {final_feedback.alignment_score:.0%}]")
        else:
            self._log(f"\n[Alignment: {final_feedback.alignment_score:.0%}]")

        return result

    def respond_stream(self, query: str):
        """
        Generate response with streaming output.

        Yields response chunks as they're generated.
        """
        # Decompose and build tree first
        decomposed = self.decomposer.decompose(query)

        if not decomposed.nouns:
            yield "I couldn't identify any concepts. Could you rephrase?"
            return

        tree = self.tree_builder.build_from_decomposition(
            decomposed.nouns,
            decomposed.verbs,
            query
        )

        # Stream render
        for chunk in self.renderer.render_stream(tree, query):
            yield chunk

    def get_tree_for_query(self, query: str) -> MeaningTree:
        """
        Get just the meaning tree without rendering.

        Useful for debugging or visualization.
        """
        decomposed = self.decomposer.decompose(query)
        return self.tree_builder.build_from_decomposition(
            decomposed.nouns,
            decomposed.verbs,
            query
        )


def main():
    """Interactive chat loop."""
    print("="*60)
    print("  MEANING CHAIN CHAT")
    print("  Semantic navigation through meaning space")
    print("="*60)
    print()
    print("Commands:")
    print("  /tree <query>  - Show meaning tree only")
    print("  /learn         - Show learning statistics")
    print("  /quiet         - Toggle verbose output")
    print("  /exit          - Exit")
    print()

    chat = MeaningChainChat(ChatConfig(verbose=True, show_tree=True))

    try:
        while True:
            try:
                query = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not query:
                continue

            if query == "/exit":
                print("Goodbye!")
                break

            if query == "/quiet":
                chat.config.verbose = not chat.config.verbose
                print(f"Verbose: {chat.config.verbose}")
                continue

            if query == "/learn":
                if chat.learner:
                    stats = chat.learner.get_session_stats()
                    print(f"\nLearning Statistics:")
                    print(f"  Session: {stats.get('session_id', 'N/A')}")
                    print(f"  Total patterns: {stats.get('total_patterns', 0)}")
                    print(f"  Unique: {stats.get('unique_patterns', 0)}")
                    print(f"  New: {stats.get('new', 0)}")
                    print(f"  Reinforced: {stats.get('reinforced', 0)}")
                    if stats.get('top_patterns'):
                        print("  Top patterns:")
                        for (s, v, o), count in stats['top_patterns'][:5]:
                            print(f"    {s} --[{v}]--> {o} ({count}x)")
                else:
                    print("Learning is disabled")
                continue

            if query.startswith("/tree "):
                query = query[6:]
                tree = chat.get_tree_for_query(query)
                print(tree.pretty_print())
                continue

            # Normal response
            result = chat.respond(query)

    finally:
        # Cleanup
        if chat.learner:
            stats = chat.learner.get_session_stats()
            if stats.get('total_patterns', 0) > 0:
                print(f"\nSession learned {stats['total_patterns']} patterns "
                      f"({stats['new']} new)")
            chat.learner.close()
        chat.tree_builder.close()


if __name__ == "__main__":
    main()
