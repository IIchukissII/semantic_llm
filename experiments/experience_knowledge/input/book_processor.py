"""
Book Processor - Transform books into experience.

Processes text content and stores as walked paths in the experience graph.
Designed for future API/endpoint integration.

Weight hierarchy:
    Books (corpus):     w_max (1.0)    - Established knowledge
    Articles:           0.8            - Curated content
    Conversation:       0.2            - Needs reinforcement
    Context-inferred:   w_min (0.1)    - Weakest

"Reading is walking through another's semantic territory."
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

import sys
_THIS_FILE = Path(__file__).resolve()
_EXPERIENCE_KNOWLEDGE = _THIS_FILE.parent.parent
sys.path.insert(0, str(_EXPERIENCE_KNOWLEDGE))

from layers.core import Wholeness
from layers.dynamics import WeightDynamics, WeightConfig
from graph.experience import ExperienceGraph, GraphConfig


@dataclass
class ProcessingResult:
    """Result of processing a book/text."""
    source_id: str
    source_type: str  # "book" | "article" | "text"

    # Statistics
    total_words: int = 0
    semantic_words: int = 0
    unique_states: int = 0
    unique_transitions: int = 0

    # Weight info
    initial_weight: float = 1.0

    # Metadata
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: int = 0

    # Status
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "statistics": {
                "total_words": self.total_words,
                "semantic_words": self.semantic_words,
                "unique_states": self.unique_states,
                "unique_transitions": self.unique_transitions,
            },
            "weight": self.initial_weight,
            "processed_at": self.processed_at,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class SourceWeight:
    """Weight configuration for different source types."""
    book: float = 1.0       # w_max - established knowledge
    article: float = 0.8    # Curated but less authoritative
    conversation: float = 0.2  # Needs reinforcement
    context: float = 0.1    # w_min - weakest


class BookProcessor:
    """
    Process books and other text sources into experience.

    Designed for:
    - CLI usage (current)
    - Future REST API endpoint
    - Batch processing pipelines
    """

    def __init__(self,
                 graph: ExperienceGraph = None,
                 wholeness: Wholeness = None,
                 weights: SourceWeight = None):
        """
        Initialize processor.

        Args:
            graph: Experience graph (creates new if None)
            wholeness: Semantic space (loads if None)
            weights: Weight configuration for source types
        """
        self.wholeness = wholeness or Wholeness()
        self.graph = graph or ExperienceGraph(
            GraphConfig(),
            self.wholeness,
            WeightDynamics()
        )
        self.weights = weights or SourceWeight()

    def process_book(self,
                     source: Union[str, Path],
                     book_id: str = None,
                     skip_header_footer: bool = True) -> ProcessingResult:
        """
        Process a book file into experience.

        Args:
            source: Path to book file
            book_id: Identifier (defaults to filename)
            skip_header_footer: Skip Gutenberg-style headers/footers

        Returns:
            ProcessingResult with statistics
        """
        start_time = datetime.now()

        # Resolve path
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                source_id=str(source),
                source_type="book",
                success=False,
                error=f"File not found: {source}"
            )

        # Extract book ID from filename
        if book_id is None:
            book_id = path.stem
            # Handle "Author - Title" format
            if " - " in book_id:
                book_id = book_id.split(" - ", 1)[1]

        # Read content
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return ProcessingResult(
                source_id=book_id,
                source_type="book",
                success=False,
                error=f"Read error: {e}"
            )

        # Process text
        result = self.process_text(
            text=text,
            source_id=book_id,
            source_type="book",
            skip_header_footer=skip_header_footer
        )

        # Add timing
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        result.processing_time_ms = int(elapsed)

        return result

    def process_text(self,
                     text: str,
                     source_id: str,
                     source_type: str = "book",
                     skip_header_footer: bool = False) -> ProcessingResult:
        """
        Process raw text into experience.

        Args:
            text: Raw text content
            source_id: Identifier for this content
            source_type: "book" | "article" | "conversation" | "context"
            skip_header_footer: Skip first/last 5% of text

        Returns:
            ProcessingResult with statistics
        """
        # Determine weight based on source type
        weight = getattr(self.weights, source_type, self.weights.context)

        # Skip header/footer if requested (for Gutenberg books)
        if skip_header_footer and len(text) > 1000:
            skip_chars = len(text) // 20  # 5%
            text = text[skip_chars:-skip_chars]

        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        total_words = len(words)

        # Filter to semantic words (in wholeness)
        semantic_words = [w for w in words if w in self.wholeness]

        if not semantic_words:
            return ProcessingResult(
                source_id=source_id,
                source_type=source_type,
                total_words=total_words,
                semantic_words=0,
                initial_weight=weight,
                success=True
            )

        # Build visits and transitions
        visits = {}
        transitions = {}
        prev = None

        for word in semantic_words:
            visits[word] = visits.get(word, 0) + 1

            if prev is not None and prev != word:
                key = (prev, word)
                transitions[key] = transitions.get(key, 0) + 1

            prev = word

        # Store in graph
        self._store_experience(visits, transitions, source_id, source_type, weight)

        return ProcessingResult(
            source_id=source_id,
            source_type=source_type,
            total_words=total_words,
            semantic_words=len(semantic_words),
            unique_states=len(visits),
            unique_transitions=len(transitions),
            initial_weight=weight,
            success=True
        )

    def _store_experience(self,
                          visits: Dict[str, int],
                          transitions: Dict[tuple, int],
                          source_id: str,
                          source_type: str,
                          weight: float):
        """Store visits and transitions in the experience graph."""
        if not self.graph.driver:
            return

        now = datetime.now().isoformat()

        # Prepare state batch
        state_batch = []
        for word, count in visits.items():
            state = self.wholeness.states.get(word)
            if not state:
                continue
            state_batch.append({
                'word': word,
                'goodness': float(state.goodness),
                'tau': float(state.tau),
                'j': state.j.tolist(),
                'count': count,
                'source_id': source_id,
                'now': now,
                'source_type': source_type
            })

        # Store states
        with self.graph.driver.session() as session:
            session.run("""
                UNWIND $batch AS item
                MERGE (s:SemanticState {word: item.word})
                ON CREATE SET
                    s.goodness = item.goodness,
                    s.tau = item.tau,
                    s.j = item.j,
                    s.visits = item.count,
                    s.books = [item.source_id],
                    s.created_at = item.now,
                    s.last_visited = item.now,
                    s.learned_from = item.source_type
                ON MATCH SET
                    s.visits = s.visits + item.count,
                    s.last_visited = item.now,
                    s.books = CASE
                        WHEN NOT item.source_id IN s.books
                        THEN s.books + item.source_id
                        ELSE s.books
                    END
            """, batch=state_batch)

        # Prepare transition batch
        trans_batch = [
            {
                'from_word': f,
                'to_word': t,
                'count': c,
                'source_id': source_id,
                'now': now,
                'weight': weight
            }
            for (f, t), c in transitions.items()
        ]

        # Store transitions in chunks
        chunk_size = 5000
        with self.graph.driver.session() as session:
            for i in range(0, len(trans_batch), chunk_size):
                chunk = trans_batch[i:i+chunk_size]
                session.run("""
                    UNWIND $batch AS item
                    MATCH (a:SemanticState {word: item.from_word})
                    MATCH (b:SemanticState {word: item.to_word})
                    MERGE (a)-[t:TRANSITION]->(b)
                    ON CREATE SET
                        t.weight = item.weight,
                        t.raw_count = item.count,
                        t.books = [item.source_id],
                        t.created_at = item.now,
                        t.last_updated = item.now
                    ON MATCH SET
                        t.raw_count = coalesce(t.raw_count, 0) + item.count,
                        t.last_updated = item.now,
                        t.books = CASE
                            WHEN NOT item.source_id IN t.books
                            THEN t.books + item.source_id
                            ELSE t.books
                        END
                """, batch=chunk)

    def process_library(self,
                        directory: Union[str, Path],
                        pattern: str = "*.txt",
                        limit: int = None) -> List[ProcessingResult]:
        """
        Process multiple books from a directory.

        Args:
            directory: Directory containing books
            pattern: Glob pattern for book files
            limit: Maximum number of books to process

        Returns:
            List of ProcessingResult for each book
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return [ProcessingResult(
                source_id=str(directory),
                source_type="library",
                success=False,
                error=f"Directory not found: {directory}"
            )]

        books = sorted(dir_path.glob(pattern))
        if limit:
            books = books[:limit]

        results = []
        print(f"\nProcessing {len(books)} books from {directory}")
        print("=" * 60)

        for i, book in enumerate(books, 1):
            print(f"[{i}/{len(books)}] {book.name[:50]}...")
            result = self.process_book(book)
            results.append(result)

            if result.success:
                print(f"         {result.unique_states} states, {result.unique_transitions} transitions")
            else:
                print(f"         ERROR: {result.error}")

        print("=" * 60)

        # Summary
        successful = [r for r in results if r.success]
        total_states = sum(r.unique_states for r in successful)
        total_trans = sum(r.unique_transitions for r in successful)

        print(f"Processed: {len(successful)}/{len(results)} books")
        print(f"Total states: {total_states}")
        print(f"Total transitions: {total_trans}")

        return results

    def close(self):
        """Close graph connection."""
        if self.graph:
            self.graph.close()


# Convenience functions for direct usage

def process_book(path: Union[str, Path],
                 book_id: str = None) -> ProcessingResult:
    """
    Process a single book file.

    Convenience function that creates processor, processes book, and cleans up.

    Args:
        path: Path to book file
        book_id: Optional identifier

    Returns:
        ProcessingResult
    """
    processor = BookProcessor()
    try:
        return processor.process_book(path, book_id)
    finally:
        processor.close()


def process_library(directory: Union[str, Path],
                    pattern: str = "*.txt",
                    limit: int = None) -> List[ProcessingResult]:
    """
    Process multiple books from a directory.

    Convenience function that creates processor, processes books, and cleans up.

    Args:
        directory: Directory containing books
        pattern: Glob pattern for files
        limit: Maximum number of books

    Returns:
        List of ProcessingResult
    """
    processor = BookProcessor()
    try:
        return processor.process_library(directory, pattern, limit)
    finally:
        processor.close()


def main():
    """CLI interface for book processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process books into experience graph"
    )
    parser.add_argument("command", choices=["book", "library", "stats"],
                       help="Command to run")
    parser.add_argument("path", nargs="?",
                       help="Path to book or directory")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of books")
    parser.add_argument("--pattern", default="*.txt",
                       help="File pattern for library")
    parser.add_argument("--id", default=None,
                       help="Book identifier")

    args = parser.parse_args()

    if args.command == "book":
        if not args.path:
            print("Error: path required for book command")
            return
        result = process_book(args.path, args.id)
        print(f"\nResult: {result.to_dict()}")

    elif args.command == "library":
        path = args.path or "/home/chukiss/text_project/data/gutenberg"
        results = process_library(path, args.pattern, args.limit)

        # Print summary as JSON-like
        print("\nResults:")
        for r in results[:5]:  # Show first 5
            print(f"  {r.source_id}: {r.unique_states} states, {r.unique_transitions} transitions")
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more")

    elif args.command == "stats":
        processor = BookProcessor()
        stats = processor.graph.get_stats()
        print("\nExperience Graph Stats:")
        print(f"  States: {stats['nodes']}")
        print(f"  Transitions: {stats['edges']}")
        print(f"  Books: {len(stats['books'])}")
        processor.close()


if __name__ == "__main__":
    main()
