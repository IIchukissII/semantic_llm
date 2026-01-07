#!/usr/bin/env python3
"""Process Gutenberg books and load bonds into Neo4j.

Usage:
    python -m storm_logos.scripts.process_books --priority
    python -m storm_logos.scripts.process_books --file "path/to/book.txt"
    python -m storm_logos.scripts.process_books --all --limit 10
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from storm_logos.data.book_parser import BookProcessor, BookParser
from storm_logos.data.neo4j import get_neo4j


# Default Gutenberg directory
DEFAULT_GUTENBERG_DIR = Path('/home/chukiss/text_project/data/gutenberg')


def process_priority_books(gutenberg_dir: Path) -> None:
    """Process the 10 priority books (5 Jung + 5 Mythology)."""
    print("=" * 60)
    print("Storm-Logos Book Processor")
    print("Processing Priority Books: Jung + Mythology")
    print("=" * 60)
    print()

    processor = BookProcessor()

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    if not processor.connect():
        print("ERROR: Could not connect to Neo4j")
        print("Make sure Neo4j is running and credentials are correct")
        return

    print("Connected!")
    print()

    # Process priority books
    start_time = datetime.now()
    results = processor.process_priority_books(gutenberg_dir)
    end_time = datetime.now()

    # Summary
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print()

    total_bonds = 0
    total_sentences = 0
    successful = 0
    failed = 0

    for r in results:
        if 'error' in r:
            failed += 1
            print(f"FAILED: {r.get('file', 'Unknown')}: {r['error']}")
        else:
            successful += 1
            total_bonds += r['n_bonds']
            total_sentences += r['n_sentences']
            print(f"OK: {r['title']} by {r['author']}: "
                  f"{r['n_bonds']} bonds, {r['n_sentences']} sentences")

    print()
    print(f"Time: {end_time - start_time}")
    print(f"Books processed: {successful}/{len(results)}")
    print(f"Total bonds: {total_bonds:,}")
    print(f"Total sentences: {total_sentences:,}")

    # Neo4j stats
    stats = processor.neo4j.stats()
    print()
    print("Neo4j Stats:")
    print(f"  Authors: {stats.get('n_authors', 0)}")
    print(f"  Books: {stats.get('n_books', 0)}")
    print(f"  Bonds: {stats.get('n_bonds', 0):,}")
    print(f"  FOLLOWS edges: {stats.get('n_follows', 0):,}")


def process_single_file(filepath: Path, author: str = '', title: str = '') -> None:
    """Process a single book file."""
    print(f"Processing: {filepath}")
    print()

    processor = BookProcessor()

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    if not processor.connect():
        print("ERROR: Could not connect to Neo4j")
        return

    print("Connected!")
    print()

    # Prepare metadata
    metadata = {}
    if author:
        metadata['author'] = author
    if title:
        metadata['title'] = title

    # Process
    start_time = datetime.now()
    result = processor.process_book(filepath, metadata if metadata else None)
    end_time = datetime.now()

    print()
    print("=" * 60)
    if 'error' in result:
        print(f"FAILED: {result['error']}")
    else:
        print(f"SUCCESS: {result['title']} by {result['author']}")
        print(f"  Bonds: {result['n_bonds']:,}")
        print(f"  Sentences: {result['n_sentences']:,}")
        print(f"  Chapters: {result['n_chapters']}")
        print(f"  Coords found: {result['coords_found']:,}")
    print(f"  Time: {end_time - start_time}")


def process_directory(directory: Path, limit: int = None) -> None:
    """Process all .txt files in a directory."""
    print(f"Processing directory: {directory}")
    print(f"Limit: {limit if limit else 'None'}")
    print()

    processor = BookProcessor()

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    if not processor.connect():
        print("ERROR: Could not connect to Neo4j")
        return

    print("Connected!")
    print()

    # Process
    start_time = datetime.now()
    results = processor.process_directory(directory, pattern='*.txt', limit=limit)
    end_time = datetime.now()

    # Summary
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    total_bonds = 0
    successful = 0
    failed = 0

    for r in results:
        if 'error' in r:
            failed += 1
        else:
            successful += 1
            total_bonds += r.get('n_bonds', 0)

    print(f"Time: {end_time - start_time}")
    print(f"Books: {successful} successful, {failed} failed")
    print(f"Total bonds: {total_bonds:,}")


def test_parser(filepath: Path) -> None:
    """Test parsing without loading to Neo4j."""
    print(f"Testing parser on: {filepath}")
    print()

    parser = BookParser()

    # Parse
    print("Parsing...")
    parsed = parser.parse_file(filepath)

    print(f"Title: {parsed.title}")
    print(f"Author: {parsed.author}")
    print(f"Chapters: {parsed.n_chapters}")
    print(f"Sentences: {parsed.n_sentences}")
    print(f"Bonds extracted: {len(parsed.bonds)}")
    print()

    # Sample bonds
    print("Sample bonds (first 20):")
    for i, bond in enumerate(parsed.bonds[:20]):
        print(f"  {i+1}. {bond.adj} {bond.noun} "
              f"(ch {bond.chapter}, sent {bond.sentence})")

    # Look up coordinates
    print()
    print("Looking up coordinates...")
    bonds_with_coords = parser.lookup_coordinates(parsed.bonds)
    found = sum(1 for eb, b in bonds_with_coords if b.A != 0 or b.S != 0 or b.tau != 2.5)
    print(f"Found coordinates for {found}/{len(bonds_with_coords)} bonds")

    # Sample with coordinates
    print()
    print("Sample bonds with coordinates:")
    count = 0
    for eb, b in bonds_with_coords:
        if b.A != 0 or b.S != 0:
            print(f"  {b.adj} {b.noun}: A={b.A:.2f}, S={b.S:.2f}, τ={b.tau:.2f}")
            count += 1
            if count >= 10:
                break


def main():
    parser = argparse.ArgumentParser(
        description='Process Gutenberg books and load bonds into Neo4j',
        prog='process_books',
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--priority', action='store_true',
                             help='Process priority books (Jung + Mythology)')
    input_group.add_argument('--file', '-f', type=Path,
                             help='Process a single book file')
    input_group.add_argument('--all', action='store_true',
                             help='Process all books in Gutenberg directory')
    input_group.add_argument('--test', type=Path,
                             help='Test parser on a file without loading to Neo4j')

    # Additional options
    parser.add_argument('--gutenberg-dir', '-d', type=Path,
                        default=DEFAULT_GUTENBERG_DIR,
                        help='Gutenberg books directory')
    parser.add_argument('--limit', '-n', type=int,
                        help='Limit number of books to process (for --all)')
    parser.add_argument('--author', '-a', type=str, default='',
                        help='Author name (for --file)')
    parser.add_argument('--title', '-t', type=str, default='',
                        help='Book title (for --file)')

    args = parser.parse_args()

    # Validate
    if args.file and not args.file.exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    if not args.gutenberg_dir.exists():
        print(f"ERROR: Gutenberg directory not found: {args.gutenberg_dir}")
        sys.exit(1)

    # Process
    if args.priority:
        process_priority_books(args.gutenberg_dir)
    elif args.file:
        process_single_file(args.file, args.author, args.title)
    elif args.all:
        process_directory(args.gutenberg_dir, args.limit)
    elif args.test:
        if not args.test.exists():
            print(f"ERROR: File not found: {args.test}")
            sys.exit(1)
        test_parser(args.test)


if __name__ == '__main__':
    main()
