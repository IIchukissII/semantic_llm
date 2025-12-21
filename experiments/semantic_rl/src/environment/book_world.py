"""
Book World: Navigate semantic space through literature.

The agent travels through the semantic landscape of a book,
following narrative flow and experiencing the story's journey.

"Only believe what was lived is knowledge"
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass

from core.semantic_state import SemanticState, SemanticGraph, Transition
from core.semantic_loader import SemanticLoader
from environment.semantic_world import SemanticWorld, WorldConfig


@dataclass
class BookConfig:
    """Configuration for book parsing."""
    min_word_frequency: int = 3      # Minimum occurrences to include
    window_size: int = 30            # Words to look ahead for transitions
    max_states: int = 300            # Maximum states in graph
    transition_threshold: float = 0.5  # Minimum co-occurrence for transition
    add_reverse_transitions: bool = True  # Add bidirectional transitions


class BookLoader:
    """
    Loads and parses books into semantic graphs.

    Extracts key concepts and builds transitions based on narrative flow.
    """

    def __init__(self, semantic_loader: SemanticLoader = None):
        """
        Initialize with optional semantic loader for real semantic data.
        """
        self.semantic_loader = semantic_loader or SemanticLoader()
        self.available_words = set(self.semantic_loader.core.states.keys()) if self.semantic_loader.core else set()

    def load_from_file(self, filepath: str) -> str:
        """Load text from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def load_sample_text(self) -> str:
        """Load a sample narrative for testing (Hero's Journey)."""
        return """
        In the beginning, there was only darkness and fear.
        The hero lived in ignorance, knowing nothing of the world beyond.

        Then came the call to adventure, a voice of hope in the night.
        The hero resisted at first, clinging to comfort and safety.
        But destiny would not be denied.

        Crossing the threshold, the hero entered the unknown.
        Here were tests and trials, enemies and allies.
        Each challenge brought pain but also wisdom.

        In the depths of despair, when all seemed lost,
        the hero found courage they never knew they had.
        Through struggle came transformation.

        The ordeal tested everything - body, mind, and soul.
        Death and rebirth in the darkness of the cave.
        The hero emerged changed, carrying the elixir of truth.

        The return was not easy. Old fears resurfaced.
        But armed with wisdom and love, the hero prevailed.
        What was once darkness became light.

        The master of two worlds now, the hero shared their gift.
        From fear to courage, from ignorance to wisdom,
        the journey had transformed not just the hero, but the world.

        And in the end, the hero understood:
        the treasure was not the destination, but the path.
        Every step, every struggle, every moment of despair and hope -
        all were necessary for the soul to grow.

        Peace came at last. Not the absence of conflict,
        but the presence of understanding. The hero rested,
        knowing that the journey would continue in others.

        For this is the eternal story: from darkness to light,
        from death to life, from hate to love.
        The wheel turns, and we all walk the path.
        """

    def extract_concepts(self, text: str, config: BookConfig = None) -> List[str]:
        """
        Extract semantic concepts from text.

        Only includes words that exist in our semantic space.
        """
        config = config or BookConfig()

        # Tokenize and clean
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Count frequencies
        word_counts = Counter(words)

        # Filter to words in our semantic space with minimum frequency
        concepts = []
        for word, count in word_counts.most_common():
            if count >= config.min_word_frequency:
                if word in self.available_words:
                    concepts.append(word)
                    if len(concepts) >= config.max_states:
                        break

        return concepts

    # Common words to exclude from semantic journey
    EXCLUDE_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "after", "before", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "also", "now", "about", "up", "out",
        "if", "or", "because", "until", "while", "during", "although",
        "that", "this", "these", "those", "what", "which", "who", "whom",
        "it", "its", "he", "his", "she", "her", "they", "their", "we", "our",
        "you", "your", "me", "my", "him", "them", "us", "i", "am", "ebook",
        "ebooks", "www", "http", "org", "txt", "gutenberg"
    }

    def extract_narrative_sequence(self, text: str) -> List[str]:
        """
        Extract concept sequence following narrative order.

        Returns concepts in the order they appear in the text.
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Keep only semantic words, preserving order
        sequence = []
        seen = set()
        for word in words:
            # Filter: in semantic space, not excluded, min length 3
            if (word in self.available_words and
                word not in self.EXCLUDE_WORDS and
                len(word) >= 3):
                sequence.append(word)
                # Track first occurrence
                if word not in seen:
                    seen.add(word)

        return sequence

    def build_graph(self, text: str, config: BookConfig = None) -> Tuple[SemanticGraph, Dict]:
        """
        Build semantic graph from book text.

        Transitions are based on narrative proximity - words that appear
        near each other in the text are connected.

        Returns:
            (SemanticGraph, metadata dict)
        """
        config = config or BookConfig()
        graph = SemanticGraph()

        # Get narrative sequence
        sequence = self.extract_narrative_sequence(text)
        concepts = list(set(sequence))[:config.max_states]

        print(f"Found {len(concepts)} unique concepts in narrative")

        # Add states from semantic loader
        for word in concepts:
            state = self.semantic_loader.get_state(word)
            if state:
                graph.add_state(state)

        # Build transitions based on co-occurrence
        cooccurrence = Counter()
        for i, word1 in enumerate(sequence):
            if word1 not in graph.states:
                continue
            # Look ahead in window
            for j in range(i + 1, min(i + config.window_size, len(sequence))):
                word2 = sequence[j]
                if word2 != word1 and word2 in graph.states:
                    # Weight by proximity (closer = stronger connection)
                    weight = 1.0 / (j - i)
                    cooccurrence[(word1, word2)] += weight

        # Create transitions for strong co-occurrences
        added_transitions = 0
        for (from_word, to_word), weight in cooccurrence.most_common():
            if weight >= config.transition_threshold:
                from_state = graph.get_state(from_word)
                to_state = graph.get_state(to_word)
                if from_state and to_state:
                    delta_g = to_state.goodness - from_state.goodness
                    # Use narrative-inspired verb
                    verb = self._infer_verb(from_word, to_word, delta_g)
                    graph.add_transition(Transition(
                        verb=verb,
                        from_state=from_word,
                        to_state=to_word,
                        delta_g=delta_g
                    ))
                    added_transitions += 1

                    # Add reverse transition if enabled
                    if config.add_reverse_transitions:
                        reverse_verb = self._infer_verb(to_word, from_word, -delta_g)
                        graph.add_transition(Transition(
                            verb=reverse_verb,
                            from_state=to_word,
                            to_state=from_word,
                            delta_g=-delta_g
                        ))
                        added_transitions += 1

        print(f"Created {added_transitions} narrative transitions")

        # Metadata about the book
        metadata = {
            "total_words": len(sequence),
            "unique_concepts": len(concepts),
            "transitions": added_transitions,
            "narrative_arc": self._detect_narrative_arc(sequence, graph)
        }

        return graph, metadata

    def _infer_verb(self, from_word: str, to_word: str, delta_g: float) -> str:
        """Infer a verb for the narrative transition."""
        # Simple heuristics based on goodness change
        if delta_g > 0.3:
            verbs = ["discover", "find", "embrace", "achieve"]
        elif delta_g < -0.3:
            verbs = ["face", "confront", "endure", "suffer"]
        else:
            verbs = ["experience", "encounter", "witness", "know"]

        # Use hash to deterministically pick verb
        idx = hash((from_word, to_word)) % len(verbs)
        return verbs[idx]

    def _detect_narrative_arc(self, sequence: List[str], graph: SemanticGraph) -> List[Dict]:
        """
        Detect narrative arc by tracking goodness over the story.

        Returns key moments in the narrative.
        """
        arc = []
        chunk_size = max(len(sequence) // 10, 1)

        for i in range(0, len(sequence), chunk_size):
            chunk = sequence[i:i + chunk_size]
            goodness_values = []
            for word in chunk:
                state = graph.get_state(word)
                if state:
                    goodness_values.append(state.goodness)

            if goodness_values:
                avg_g = np.mean(goodness_values)
                arc.append({
                    "position": i / len(sequence),
                    "avg_goodness": float(avg_g),
                    "key_concepts": list(set(chunk))[:5]
                })

        return arc


class BookWorld(SemanticWorld):
    """
    Semantic world based on a book's narrative.

    The agent navigates through the story's semantic landscape,
    experiencing the narrative journey.
    """

    def __init__(self,
                 book_text: str = None,
                 book_file: str = None,
                 start_word: str = None,
                 goal_word: str = None,
                 book_config: BookConfig = None,
                 world_config: WorldConfig = None,
                 render_mode: str = None):
        """
        Initialize book world.

        Args:
            book_text: Raw text of the book
            book_file: Path to book file
            start_word: Starting concept (auto-detected if None)
            goal_word: Goal concept (auto-detected if None)
            book_config: Configuration for book parsing
            world_config: Configuration for world physics
            render_mode: Rendering mode
        """
        # Load book
        self.book_loader = BookLoader()

        if book_file:
            text = self.book_loader.load_from_file(book_file)
        elif book_text:
            text = book_text
        else:
            text = self.book_loader.load_sample_text()

        self.book_text = text
        self.book_config = book_config or BookConfig()

        # Build graph from book
        graph, self.book_metadata = self.book_loader.build_graph(text, self.book_config)

        # Auto-detect start and goal from narrative
        if start_word is None or goal_word is None:
            start_word, goal_word = self._detect_journey_endpoints(graph)

        # Verify start/goal exist in graph
        if start_word not in graph.states:
            print(f"Warning: '{start_word}' not in graph, auto-detecting...")
            start_word, _ = self._detect_journey_endpoints(graph)
        if goal_word not in graph.states:
            print(f"Warning: '{goal_word}' not in graph, auto-detecting...")
            _, goal_word = self._detect_journey_endpoints(graph)

        print(f"Book journey: {start_word} → {goal_word}")

        # Initialize parent
        super().__init__(
            semantic_graph=graph,
            start_word=start_word,
            goal_word=goal_word,
            config=world_config,
            render_mode=render_mode
        )

    def _detect_journey_endpoints(self, graph: SemanticGraph) -> Tuple[str, str]:
        """
        Auto-detect start and goal from narrative arc.

        Prefers archetypal journey concepts:
        - Start: darkness, fear, ignorance, etc.
        - Goal: wisdom, truth, light, understanding, etc.
        """
        # Preferred start concepts (hero's journey beginning)
        start_preferences = [
            "darkness", "fear", "ignorance", "confusion", "despair",
            "doubt", "chaos", "night", "shadow", "unknown", "death",
            "isolation", "wilderness", "void", "silence"
        ]

        # Preferred goal concepts (hero's journey end)
        goal_preferences = [
            "wisdom", "truth", "light", "understanding", "knowledge",
            "love", "peace", "hope", "life", "freedom", "courage",
            "clarity", "meaning", "salvation", "awakening"
        ]

        # Find best start from preferences
        start_word = None
        for pref in start_preferences:
            if pref in graph.states:
                start_word = pref
                break

        # Find best goal from preferences
        goal_word = None
        for pref in goal_preferences:
            if pref in graph.states:
                goal_word = pref
                break

        # Fallback: use goodness extremes
        if start_word is None or goal_word is None:
            states = list(graph.states.values())
            if not states:
                return "darkness", "light"

            states.sort(key=lambda s: s.goodness)

            if start_word is None:
                # Find meaningful word with low goodness
                for s in states[:20]:
                    if len(s.word) > 3 and s.word not in ["ebook", "ebooks", "www", "http"]:
                        start_word = s.word
                        break

            if goal_word is None:
                # Find meaningful word with high goodness
                for s in reversed(states[-20:]):
                    if len(s.word) > 3 and s.word not in ["ebook", "ebooks", "www", "http"]:
                        goal_word = s.word
                        break

        # Ensure start has outgoing transitions
        if start_word and start_word in graph.states:
            neighbors = graph.get_neighbors(start_word)
            if len(neighbors) == 0:
                # Find a connected start word
                for pref in start_preferences:
                    if pref in graph.states and len(graph.get_neighbors(pref)) > 0:
                        start_word = pref
                        break
                else:
                    # Use most connected state with low-ish goodness
                    best_start = None
                    best_connections = 0
                    for word, state in graph.states.items():
                        n_connections = len(graph.get_neighbors(word))
                        if n_connections > best_connections:
                            best_connections = n_connections
                            best_start = word
                    if best_start:
                        start_word = best_start

        # Ensure goal is reachable (has incoming transitions)
        if goal_word and goal_word in graph.states:
            # Check if any state can reach goal
            has_incoming = False
            for word in graph.states:
                for to_word, _, _ in graph.get_neighbors(word):
                    if to_word == goal_word:
                        has_incoming = True
                        break
                if has_incoming:
                    break

            if not has_incoming:
                # Find a reachable goal
                for pref in goal_preferences:
                    if pref in graph.states:
                        for word in graph.states:
                            for to_word, _, _ in graph.get_neighbors(word):
                                if to_word == pref:
                                    goal_word = pref
                                    break
                            if goal_word == pref:
                                break

        return start_word or "darkness", goal_word or "truth"

    def get_narrative_position(self) -> float:
        """Get agent's position in the narrative arc."""
        # Based on current state's goodness relative to arc
        if not self.current_state:
            return 0.0

        arc = self.book_metadata.get("narrative_arc", [])
        if not arc:
            return 0.0

        # Find closest point in arc
        current_g = self.current_state.goodness
        best_pos = 0.0
        best_diff = float('inf')

        for point in arc:
            diff = abs(point["avg_goodness"] - current_g)
            if diff < best_diff:
                best_diff = diff
                best_pos = point["position"]

        return best_pos

    def _render_text(self):
        """Enhanced text rendering with narrative context."""
        super()._render_text()

        # Add narrative context
        pos = self.get_narrative_position()
        arc = self.book_metadata.get("narrative_arc", [])

        print(f"Narrative position: {pos*100:.0f}%")
        if arc:
            # Find current arc segment
            for point in arc:
                if abs(point["position"] - pos) < 0.1:
                    print(f"Context: {', '.join(point['key_concepts'][:3])}")
                    break


def run_book_journey(book_file: str = None,
                     start: str = None,
                     goal: str = None,
                     max_steps: int = 100):
    """Run a journey through a book's semantic landscape."""
    from agents import QuantumAgent

    print("=" * 70)
    print("BOOK WORLD: NARRATIVE JOURNEY")
    print("=" * 70)

    # Create world
    world = BookWorld(
        book_file=book_file,
        start_word=start,
        goal_word=goal,
        render_mode="human"
    )

    # Create agent
    agent = QuantumAgent(
        believe=0.5,
        temperature=1.0,
        cooling_rate=0.99,
        tunnel_bonus=0.3
    )

    # Run journey
    obs, info = world.reset()
    agent.on_episode_start()

    print(f"\n{'─' * 70}")
    print("THE JOURNEY BEGINS")
    print(f"{'─' * 70}")

    total_reward = 0
    path = [info["current_word"]]

    for step in range(max_steps):
        valid = world.get_valid_actions()
        action = agent.choose_action(obs, valid, info)

        next_obs, reward, term, trunc, info = world.step(action)
        agent.update(obs, action, reward, next_obs, term or trunc, info)

        total_reward += reward

        if info.get("success"):
            to_word = info.get("to", "?")
            path.append(to_word)

            action_name = world.action_to_name.get(action, "?")
            if action == 0:
                print(f"  Step {step+1}: ⚡TUNNEL → {to_word}")
            else:
                print(f"  Step {step+1}: --{action_name}--> {to_word}")

        obs = next_obs

        if term:
            print(f"\n  *** REACHED GOAL: {world.goal_word} ***")
            break

    # Summary
    print(f"\n{'=' * 70}")
    print("JOURNEY COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nPath: {' → '.join(path[:10])}{'...' if len(path) > 10 else ''}")
    print(f"States visited: {len(set(path))}")
    print(f"Total reward: {total_reward:+.2f}")
    print(f"Final believe: {agent.believe:.2f}")

    return path, total_reward


# Gutenberg books path
GUTENBERG_PATH = Path("/home/chukiss/text_project/data/gutenberg")

# Classic books for semantic journeys
CLASSIC_BOOKS = {
    "heart_of_darkness": "Conrad, Joseph - Heart of Darkness.txt",
    "crime_and_punishment": "Dostoevsky, Fyodor - Crime and Punishment.txt",
    "divine_comedy": "Alighieri, Dante - The Divine Comedy.txt",
    "moby_dick": "Melville, Herman - Moby Dick.txt",
    "frankenstein": "Shelley, Mary - Frankenstein.txt",
    "jane_eyre": "Bronte, Charlotte - Jane Eyre.txt",
    "great_expectations": "Dickens, Charles - Great Expectations.txt",
    "brothers_karamazov": "Dostoevsky, Fyodor - The Brothers Karamazov.txt",
    "odyssey": "Homer - The Odyssey.txt",
    "paradise_lost": "Milton, John - Paradise Lost.txt",
    "les_miserables": "Hugo, Victor - Les Miserables.txt",
    "zarathustra": "Nietzsche, Friedrich - Thus Spake Zarathustra.txt",
    "metamorphosis": "Kafka, Franz - Metamorphosis.txt",
}


def get_book_path(book_key: str) -> Path:
    """Get path to a classic book."""
    if book_key in CLASSIC_BOOKS:
        return GUTENBERG_PATH / CLASSIC_BOOKS[book_key]
    # Try direct filename
    return GUTENBERG_PATH / book_key


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Book World Journey")
    parser.add_argument("--book", default="heart_of_darkness",
                        help=f"Book to use. Options: {list(CLASSIC_BOOKS.keys())}")
    parser.add_argument("--start", default=None, help="Starting concept")
    parser.add_argument("--goal", default=None, help="Goal concept")
    parser.add_argument("--steps", type=int, default=100, help="Max steps")

    args = parser.parse_args()

    book_path = get_book_path(args.book)
    print(f"Loading: {book_path}")

    run_book_journey(
        book_file=str(book_path),
        start=args.start,
        goal=args.goal,
        max_steps=args.steps
    )
