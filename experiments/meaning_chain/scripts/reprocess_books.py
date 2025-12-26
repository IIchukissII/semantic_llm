#!/usr/bin/env python3
"""
Reprocess Books with Learning Enabled

Reprocesses all books to:
1. Extract SVO patterns → VIA relationships
2. Extract Adj-Noun pairs → Learn concepts (τ from entropy)

Usage:
    python scripts/reprocess_books.py
    python scripts/reprocess_books.py --book "Jung"  # Process only Jung books
"""

import sys
from pathlib import Path
from datetime import datetime

# Add paths
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))
sys.path.insert(0, str(_SEMANTIC_LLM))

from input.book_processor import BookProcessor
from graph.meaning_graph import MeaningGraph


# Book paths
BOOKS = [
    # Bible
    {
        "path": "/home/chukiss/text_project/data/bible/King_James_Bible.txt",
        "id": "King James Bible",
        "max_sentences": 100000
    },
    # Jung
    {
        "path": "/home/chukiss/text_project/data/gutenberg/Jung_Psychology_of_the_Unconscious.txt",
        "id": "Jung - Psychology of the Unconscious",
        "max_sentences": 50000
    },
    {
        "path": "/home/chukiss/text_project/data/gutenberg/Jung, Carl Gustav - Memories, Dreams, Reflections.txt",
        "id": "Jung - Memories Dreams Reflections",
        "max_sentences": 50000
    },
    {
        "path": "/home/chukiss/text_project/data/gutenberg/Jung, Carl Gustav - Four Archetypes.txt",
        "id": "Jung - Four Archetypes",
        "max_sentences": 50000
    },
    # Breath of Love
    {
        "path": "/home/chukiss/text_project/data/gutenberg/breath_of_love_english.txt",
        "id": "Breath of Love (Nerim)",
        "max_sentences": 10000
    },
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess books with learning")
    parser.add_argument("--book", type=str, default=None,
                        help="Filter books by name (partial match)")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Override max sentences per book")
    parser.add_argument("--no-learning", action="store_true",
                        help="Disable learning (SVO only)")
    args = parser.parse_args()

    print("=" * 70)
    print("Reprocess Books with Learning")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Connect to graph
    graph = MeaningGraph()
    if not graph.is_connected():
        print("\n✗ Neo4j not connected!")
        print("  Start with: cd config && docker-compose up -d")
        return 1

    # Setup learning schema
    graph.setup_schema()
    graph.setup_learning_schema()

    # Create processor
    enable_learning = not args.no_learning
    processor = BookProcessor(graph=graph, enable_learning=enable_learning)

    # Try to load adjective vectors for j-centroid computation
    try:
        from core.data_loader import DataLoader
        loader = DataLoader()
        processor.load_adj_vectors(loader)
        print(f"Loaded {len(processor.adj_vectors)} adjective vectors")
    except Exception as e:
        print(f"Warning: Could not load adjective vectors: {e}")
        print("  Concepts will have j=[0,0,0,0,0] and g=0")

    # Filter books if requested
    books_to_process = BOOKS
    if args.book:
        books_to_process = [b for b in BOOKS if args.book.lower() in b["id"].lower()]
        if not books_to_process:
            print(f"\n✗ No books matching '{args.book}'")
            return 1

    print(f"\nBooks to process: {len(books_to_process)}")
    for book in books_to_process:
        print(f"  - {book['id']}")

    # Process each book
    results = []
    total_start = datetime.now()

    for i, book in enumerate(books_to_process, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(books_to_process)}] {book['id']}")
        print("=" * 70)

        path = Path(book["path"])
        if not path.exists():
            print(f"  ✗ File not found: {path}")
            continue

        max_sentences = args.max_sentences or book.get("max_sentences", 50000)

        result = processor.process_book(
            filepath=str(path),
            book_id=book["id"],
            max_sentences=max_sentences
        )

        results.append({
            "book": book["id"],
            "success": result.success,
            "svo_patterns": result.svo_patterns,
            "adj_noun_pairs": result.adj_noun_pairs,
            "new_concepts": result.new_concepts_learned,
            "updated_concepts": result.existing_concepts_updated,
            "time_ms": result.processing_time_ms
        })

    # Summary
    total_time = (datetime.now() - total_start).total_seconds()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_svo = sum(r["svo_patterns"] for r in results)
    total_adj_noun = sum(r["adj_noun_pairs"] for r in results)
    total_new = sum(r["new_concepts"] for r in results)
    total_updated = sum(r["updated_concepts"] for r in results)

    print(f"\nBooks processed: {len(results)}")
    print(f"Total SVO patterns: {total_svo:,}")
    print(f"Total Adj-Noun pairs: {total_adj_noun:,}")
    print(f"New concepts learned: {total_new:,}")
    print(f"Concepts updated: {total_updated:,}")
    print(f"Total time: {total_time:.1f}s")

    print("\nPer-book results:")
    print("-" * 70)
    print(f"{'Book':<40} {'SVO':>8} {'Adj-N':>8} {'New':>6} {'Upd':>6} {'Time':>8}")
    print("-" * 70)

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['book'][:38]:<38} {r['svo_patterns']:>8,} {r['adj_noun_pairs']:>8,} "
              f"{r['new_concepts']:>6,} {r['updated_concepts']:>6,} {r['time_ms']/1000:>7.1f}s")

    # Get learning stats from graph
    print("\n" + "-" * 70)
    print("Learning Statistics from Neo4j:")
    stats = graph.get_learning_stats()
    if "error" not in stats:
        print(f"  Learned concepts: {stats.get('learned_concepts', 0)}")
        print(f"  Adjectives tracked: {stats.get('adjectives', 0)}")
        print(f"  Observation edges: {stats.get('observation_edges', 0)}")
        if stats.get('avg_tau'):
            print(f"  Average τ: {stats['avg_tau']:.2f}")
        if stats.get('avg_confidence'):
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")

    processor.close()

    print(f"\nCompleted: {datetime.now().isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
