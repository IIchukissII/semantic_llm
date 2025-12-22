"""
Book Semantic Extractor: Extract semantic journey from books.

Uses the NounCloud-based QuantumCore to map book text to a trajectory
through semantic space (goodness, tau).

"Only believe what was lived is knowledge"
"""

import re
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Add path to find QuantumCore
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM_PATH = _THIS_FILE.parent.parent.parent.parent.parent
if str(_SEMANTIC_LLM_PATH) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_LLM_PATH))


@dataclass
class SemanticPoint:
    """A point in the book's semantic journey."""
    position: float          # 0-1 position in book
    words: List[str]         # nouns in this chunk
    avg_goodness: float      # average g
    avg_tau: float           # average τ
    avg_j: np.ndarray        # average j-vector (5D)
    chunk_text: str = ""     # original text chunk


@dataclass
class BookJourney:
    """Complete semantic journey of a book."""
    title: str
    points: List[SemanticPoint]
    all_nouns: List[str]     # all nouns found in order

    @property
    def goodness_trajectory(self) -> List[float]:
        return [p.avg_goodness for p in self.points]

    @property
    def tau_trajectory(self) -> List[float]:
        return [p.avg_tau for p in self.points]

    @property
    def positions(self) -> List[float]:
        return [p.position for p in self.points]

    def narrative_arc(self) -> Dict:
        """Analyze the narrative arc."""
        g = self.goodness_trajectory
        if not g:
            return {}

        return {
            "start_g": g[0],
            "end_g": g[-1],
            "min_g": min(g),
            "max_g": max(g),
            "min_pos": g.index(min(g)) / len(g),
            "max_pos": g.index(max(g)) / len(g),
            "delta_g": g[-1] - g[0],
            "variance": np.var(g),
            "trajectory_type": self._classify_arc(g)
        }

    def _classify_arc(self, g: List[float]) -> str:
        """Classify the narrative arc type."""
        if len(g) < 3:
            return "flat"

        start, middle, end = g[0], np.mean(g[len(g)//3:2*len(g)//3]), g[-1]

        if middle < start and middle < end:
            return "descent-ascent"  # Classic hero's journey
        elif middle > start and middle > end:
            return "ascent-descent"  # Tragedy
        elif end > start:
            return "ascent"  # Comedy/redemption
        elif end < start:
            return "descent"  # Tragedy
        else:
            return "flat"


class BookExtractor:
    """
    Extract semantic journey from books using QuantumCore.

    Maps text to NounCloud space and tracks the trajectory.
    """

    # Common words to exclude
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
        "you", "your", "me", "my", "him", "them", "us", "i", "am",
        "said", "say", "says", "like", "one", "two", "much", "many",
        "thing", "things", "way", "ways", "time", "times", "man", "men",
        "woman", "women", "day", "days", "year", "years", "make", "made",
        "see", "saw", "seen", "come", "came", "go", "went", "gone",
        "know", "knew", "known", "think", "thought", "take", "took", "taken",
        "get", "got", "give", "gave", "given", "find", "found", "tell", "told",
        "put", "let", "seem", "seemed", "leave", "left", "call", "called",
        "ebook", "ebooks", "www", "http", "org", "txt", "gutenberg", "chapter"
    }

    def __init__(self):
        """Initialize with QuantumCore."""
        self.core = None
        self.available_words = set()
        self._load_core()

    def _load_core(self):
        """Load QuantumCore for semantic data."""
        try:
            # Direct import from absolute path
            import importlib.util

            hybrid_llm_path = _SEMANTIC_LLM_PATH / "core" / "hybrid_llm.py"
            data_loader_path = _SEMANTIC_LLM_PATH / "core" / "data_loader.py"

            # Load data_loader first (dependency)
            spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
            data_loader_module = importlib.util.module_from_spec(spec)
            sys.modules['core.data_loader'] = data_loader_module
            spec.loader.exec_module(data_loader_module)

            # Load hybrid_llm
            spec = importlib.util.spec_from_file_location("hybrid_llm", hybrid_llm_path)
            hybrid_llm_module = importlib.util.module_from_spec(spec)
            sys.modules['core.hybrid_llm'] = hybrid_llm_module
            spec.loader.exec_module(hybrid_llm_module)

            QuantumCore = hybrid_llm_module.QuantumCore

            print("Loading QuantumCore...")
            self.core = QuantumCore()
            self.available_words = set(self.core.states.keys())
            print(f"  Loaded {len(self.available_words)} semantic states")

        except Exception as e:
            print(f"Warning: Could not load QuantumCore: {e}")
            import traceback
            traceback.print_exc()
            self.core = None

    def load_book(self, filepath: str) -> str:
        """Load book text from file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def extract_nouns(self, text: str) -> List[str]:
        """
        Extract nouns that exist in our semantic space.

        Returns words in order of appearance.
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())

        nouns = []
        for word in words:
            if (word in self.available_words and
                word not in self.EXCLUDE_WORDS and
                len(word) >= 3):
                nouns.append(word)

        return nouns

    def extract_journey(self,
                       text: str,
                       chunk_size: int = 500,
                       title: str = "Book") -> BookJourney:
        """
        Extract semantic journey from book text.

        Args:
            text: Book text
            chunk_size: Number of words per chunk
            title: Book title

        Returns:
            BookJourney with semantic trajectory
        """
        if self.core is None:
            print("Error: QuantumCore not loaded")
            return BookJourney(title=title, points=[], all_nouns=[])

        # Get all words
        all_words = re.findall(r'\b[a-z]+\b', text.lower())
        total_words = len(all_words)

        print(f"Extracting journey from '{title}'...")
        print(f"  Total words: {total_words}")

        # Extract all nouns first
        all_nouns = self.extract_nouns(text)
        print(f"  Semantic nouns found: {len(all_nouns)}")
        print(f"  Unique semantic nouns: {len(set(all_nouns))}")

        # Process in chunks
        points = []
        for i in range(0, total_words, chunk_size):
            chunk_words = all_words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words[:50]) + "..."  # Preview

            # Extract nouns from chunk
            chunk_nouns = []
            for word in chunk_words:
                if (word in self.available_words and
                    word not in self.EXCLUDE_WORDS and
                    len(word) >= 3):
                    chunk_nouns.append(word)

            if not chunk_nouns:
                continue

            # Compute average semantic properties
            g_values = []
            tau_values = []
            j_vectors = []

            for noun in chunk_nouns:
                state = self.core.states.get(noun)
                if state:
                    g_values.append(state.goodness)
                    tau_values.append(state.tau)
                    j_vectors.append(state.j)

            if not g_values:
                continue

            avg_j = np.mean(j_vectors, axis=0) if j_vectors else np.zeros(5)

            point = SemanticPoint(
                position=i / total_words,
                words=list(set(chunk_nouns)),  # Unique nouns
                avg_goodness=np.mean(g_values),
                avg_tau=np.mean(tau_values),
                avg_j=avg_j,
                chunk_text=chunk_text
            )
            points.append(point)

        journey = BookJourney(
            title=title,
            points=points,
            all_nouns=all_nouns
        )

        print(f"  Journey points: {len(points)}")
        arc = journey.narrative_arc()
        print(f"  Arc type: {arc.get('trajectory_type', 'unknown')}")
        print(f"  Δg: {arc.get('delta_g', 0):+.3f}")

        return journey

    def extract_key_concepts(self, journey: BookJourney, top_n: int = 20) -> List[Tuple[str, int, float]]:
        """
        Extract key concepts from journey.

        Returns: [(word, count, goodness), ...]
        """
        counts = Counter(journey.all_nouns)

        results = []
        for word, count in counts.most_common(top_n):
            state = self.core.states.get(word)
            if state:
                results.append((word, count, state.goodness))

        return results

    def compare_books(self, journeys: List[BookJourney]) -> Dict:
        """Compare multiple book journeys."""
        comparison = {}

        for j in journeys:
            arc = j.narrative_arc()
            comparison[j.title] = {
                "arc_type": arc.get("trajectory_type"),
                "delta_g": arc.get("delta_g", 0),
                "variance": arc.get("variance", 0),
                "n_nouns": len(set(j.all_nouns)),
            }

        return comparison


def extract_book_journey(book_path: str,
                        chunk_size: int = 500,
                        show_key_concepts: bool = True) -> BookJourney:
    """
    Quick function to extract and display a book's semantic journey.
    """
    extractor = BookExtractor()

    # Extract title from filename
    title = Path(book_path).stem
    if " - " in title:
        title = title.split(" - ", 1)[1]

    text = extractor.load_book(book_path)
    journey = extractor.extract_journey(text, chunk_size=chunk_size, title=title)

    if show_key_concepts:
        print("\nKey concepts:")
        concepts = extractor.extract_key_concepts(journey)
        for word, count, g in concepts[:15]:
            print(f"  {word:15} (n={count:3}, g={g:+.2f})")

    return journey


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract book semantic journey")
    parser.add_argument("book", help="Path to book file or book key")
    parser.add_argument("--chunk", type=int, default=500, help="Words per chunk")
    parser.add_argument("--plot", action="store_true", help="Plot the journey")

    args = parser.parse_args()

    # Check if it's a key or path
    from environment.book_world import GUTENBERG_PATH, CLASSIC_BOOKS

    if args.book in CLASSIC_BOOKS:
        book_path = str(GUTENBERG_PATH / CLASSIC_BOOKS[args.book])
    else:
        book_path = args.book

    print(f"Book: {book_path}")
    journey = extract_book_journey(book_path, chunk_size=args.chunk)

    # Print journey summary
    print("\n" + "=" * 60)
    print("SEMANTIC JOURNEY SUMMARY")
    print("=" * 60)

    arc = journey.narrative_arc()
    print(f"\nNarrative arc: {arc.get('trajectory_type', 'unknown')}")
    print(f"  Start goodness: {arc.get('start_g', 0):+.3f}")
    print(f"  End goodness:   {arc.get('end_g', 0):+.3f}")
    print(f"  Min goodness:   {arc.get('min_g', 0):+.3f} at {arc.get('min_pos', 0)*100:.0f}%")
    print(f"  Max goodness:   {arc.get('max_g', 0):+.3f} at {arc.get('max_pos', 0)*100:.0f}%")
    print(f"  Total Δg:       {arc.get('delta_g', 0):+.3f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        positions = [p.position * 100 for p in journey.points]
        goodness = journey.goodness_trajectory
        tau = journey.tau_trajectory

        # Goodness plot
        ax1.fill_between(positions, goodness, alpha=0.3, color='green')
        ax1.plot(positions, goodness, 'o-', color='darkgreen', markersize=3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Goodness (g)')
        ax1.set_title(f'{journey.title} - Semantic Journey')
        ax1.grid(True, alpha=0.3)

        # Tau plot
        ax2.plot(positions, tau, 's-', color='blue', markersize=3, alpha=0.7)
        ax2.set_xlabel('Position in book (%)')
        ax2.set_ylabel('Abstraction (τ)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        out_path = f"visualizations/{journey.title.replace(' ', '_')}_semantic.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {out_path}")
        plt.show()
