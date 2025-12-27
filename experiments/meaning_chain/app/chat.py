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
from chain_core.euler_navigation import (
    EulerAwareStorm, EulerNavigator,
    KT_NATURAL, VEIL_TAU, GROUND_STATE_TAU, E
)
from models.types import MeaningTree, MeaningNode, SemanticProperties
import numpy as np

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

    # Euler Navigation (orbital physics)
    euler_mode: bool = True               # Use Euler orbital navigation
    euler_temperature: float = KT_NATURAL # Natural temperature kT ≈ 0.82

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
                max_tokens=self.config.max_tokens,
                euler_mode=self.config.euler_mode  # Enable Euler context in prompts
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
        self.euler_storm = None

        if self.config.euler_mode:
            # Euler-aware navigation with orbital physics
            self.euler_storm = EulerAwareStorm(temperature=self.config.euler_temperature)
            self._log(f"[Chat] Euler navigation enabled (kT={self.config.euler_temperature:.2f})")
        elif self.config.use_storm_logos:
            # Standard storm-logos (fallback)
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

        # Conversation history (for context)
        self.history: List[Dict[str, Any]] = []
        # Chat messages for Ollama (user/assistant format)
        self.chat_messages: List[Dict[str, str]] = []

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

        # Accumulate context nouns from conversation history for richer navigation
        if self.chat_messages:
            context_nouns = []
            for msg in self.chat_messages[-6:]:  # Last 3 exchanges
                if msg['role'] == 'user':  # Only from user messages
                    ctx_decomposed = self.decomposer.decompose(msg['content'])
                    context_nouns.extend(ctx_decomposed.nouns[:5])

            if context_nouns:
                # Combine current nouns with context, prioritizing current
                all_nouns = decomposed.nouns + [n for n in context_nouns if n not in decomposed.nouns]
                decomposed.nouns = all_nouns[:8]  # Limit to 8 seeds for navigation
                if len(context_nouns) > 0:
                    self._log(f"    With context: {decomposed.nouns}")

        if not decomposed.nouns:
                return {
                    'response': "I couldn't identify any concepts in your query. "
                               "Could you rephrase?",
                    'tree': None,
                    'feedback': None,
                    'attempts': 0,
                    'metadata': {'error': 'no_concepts'}
                }

        # Step 2: Build tree using Euler, Storm-Logos, or fallback
        pattern = None
        euler_stats = None

        if self.euler_storm:
            # Euler-aware navigation with orbital physics
            self._log("\n[2] EULER STORM phase (orbital navigation)...")
            storm_result = self.euler_storm.generate(
                seeds=decomposed.nouns[:5],
                n_walks=self.config.n_walks,
                steps_per_walk=self.config.steps_per_walk
            )

            stats = storm_result['statistics']
            all_states = storm_result['states']

            # Find convergence (most visited non-seed word)
            word_counts = {}
            for state in all_states:
                word_counts[state.word] = word_counts.get(state.word, 0) + 1

            seeds_set = set(decomposed.nouns[:5])
            convergence = None
            max_count = 0
            for word, count in word_counts.items():
                if word not in seeds_set and count > max_count:
                    convergence = word
                    max_count = count

            core_concepts = sorted(word_counts.keys(), key=lambda w: -word_counts[w])[:10]

            # Count veil crossings
            veil_crossings = sum(p.get('veil_crossings', 0) for p in stats.get('path_stats', []))

            # Orbital distribution
            orbital_dist = {}
            for state in all_states:
                n = state.orbital_n
                orbital_dist[n] = orbital_dist.get(n, 0) + 1

            self._log(f"    Convergence: {convergence}")
            self._log(f"    Core concepts: {core_concepts[:5]}")
            self._log(f"    Mean τ: {stats['mean_tau']:.2f} (orbital n={stats['mean_orbital']:.1f})")
            self._log(f"    Human realm: {stats['human_fraction']:.1%}")
            self._log(f"    Veil crossings: {veil_crossings}")

            # Build euler_stats for renderer
            euler_stats = {
                'mean_tau': stats['mean_tau'],
                'orbital_n': int(round((stats['mean_tau'] - 1) * E)),
                'realm': 'human' if stats['mean_tau'] < VEIL_TAU else 'transcendental',
                'below_veil': stats['mean_tau'] < VEIL_TAU,
                'near_ground': abs(stats['mean_tau'] - GROUND_STATE_TAU) < 0.5,
                'veil_crossings': veil_crossings,
                'human_fraction': stats['human_fraction'],
                'convergence': convergence,
                'core_concepts': core_concepts[:5],
                'key_symbols': decomposed.unknown_words[:10],  # Important symbols not in vocab
                'known_nouns': decomposed.nouns[:8]  # Nouns we could navigate
            }

            # Build tree from Euler results
            root_word = convergence or decomposed.nouns[0]
            root = MeaningNode(
                word=root_word,
                properties=SemanticProperties(
                    g=0.0,
                    tau=stats['mean_tau'],
                    j=np.zeros(5)
                ),
                depth=0
            )
            tree = MeaningTree(roots=[root])

            self._log("\n[3] ORBITAL PHYSICS")
            if euler_stats['near_ground']:
                self._log("    Position: Ground state - immediate experience")
            elif euler_stats['below_veil']:
                self._log("    Position: Human territory - everyday meaning")
            else:
                self._log("    Position: Transcendental - beyond the Veil")

        elif self.storm_logos:
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

        # Step 4: Render ONE response from the pattern (with conversation history)
        self._log("\n[4] Rendering response...")
        render_result = self.renderer.render(
            tree, query,
            euler_stats=euler_stats,
            conversation_history=self.chat_messages
        )
        final_response = render_result['response']

        # Update chat messages for next turn
        self.chat_messages.append({"role": "user", "content": query})
        self.chat_messages.append({"role": "assistant", "content": final_response})

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
            'euler_stats': euler_stats,  # Euler orbital statistics
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
            if learn_stats.get('total', 0) > 0:
                self._log(f"\n[5] Learning: {learn_stats.get('total', 0)} patterns "
                          f"({learn_stats.get('new', 0)} new, {learn_stats.get('reinforced', 0)} reinforced)")
            result['metadata']['learned'] = learn_stats

        # Final output
        self._log(f"\n{'='*60}")
        self._log("RESPONSE:")
        self._log(f"{'='*60}")
        self._log(final_response)
        if euler_stats:
            self._log(f"\n[τ={euler_stats['mean_tau']:.2f} | n={euler_stats['orbital_n']} | "
                      f"{euler_stats['realm']} | veil×{euler_stats['veil_crossings']}]")
        elif pattern:
            self._log(f"\n[Coherence: {pattern.coherence:.0%} | "
                      f"Alignment: {final_feedback.alignment_score:.0%}]")
        else:
            self._log(f"\n[Alignment: {final_feedback.alignment_score:.0%}]")

        return result

    def close(self):
        """Clean up resources."""
        if self.learner:
            self.learner.close()
        if self.euler_storm:
            self.euler_storm.close()
        if self.storm_logos:
            self.storm_logos.close()
        self.tree_builder.close()

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
    print("  Euler-Aware Semantic Navigation")
    print("="*60)
    print()
    print(f"  Euler Constants:")
    print(f"    e = {E:.4f} (orbital spacing = 1/e)")
    print(f"    kT = {KT_NATURAL:.2f} (natural temperature)")
    print(f"    Veil at τ = e (human < e < transcendental)")
    print()
    print("Commands:")
    print("  /tree <query>  - Show meaning tree only")
    print("  /learn         - Show learning statistics")
    print("  /euler         - Toggle Euler mode")
    print("  /clear         - Clear conversation history")
    print("  /quiet         - Toggle verbose output")
    print("  /exit          - Exit")
    print()

    chat = MeaningChainChat(ChatConfig(verbose=True, show_tree=False, euler_mode=True))

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

            if query == "/euler":
                chat.config.euler_mode = not chat.config.euler_mode
                if chat.config.euler_mode and not chat.euler_storm:
                    chat.euler_storm = EulerAwareStorm(temperature=chat.config.euler_temperature)
                print(f"Euler mode: {chat.config.euler_mode}")
                continue

            if query == "/clear":
                chat.chat_messages = []
                chat.history = []
                print("Conversation history cleared.")
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
                print(f"\nSession learned {stats.get('total_patterns', 0)} patterns "
                      f"({stats.get('new', 0)} new)")
            chat.learner.close()
        if chat.euler_storm:
            chat.euler_storm.close()
        if chat.storm_logos:
            chat.storm_logos.close()
        chat.tree_builder.close()


if __name__ == "__main__":
    main()
