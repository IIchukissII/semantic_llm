#!/usr/bin/env python3
"""
Semantic Index Pipeline - Database-backed processing.

Similar to bond_pipeline.py:
- Progress tracked in PostgreSQL
- Resumable processing
- Batch mode or continuous

Tables:
- hyp_semantic_progress: Track which books are processed
- hyp_semantic_words: Accumulated word vectors
- hyp_semantic_index: Final averaged index

Usage:
    python semantic_pipeline.py --status       # Show progress
    python semantic_pipeline.py --batch 100    # Process 100 books
    python semantic_pipeline.py --continuous   # Run until done
    python semantic_pipeline.py --finalize     # Create final index
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np

import psycopg2
from psycopg2.extras import execute_values, Json
import zstandard as zstd
import torch
import spacy

sys.path.insert(0, str(Path(__file__).parent))
from semantic_bottleneck_v2 import SemanticBottleneckV2

# Configuration
DATABASE_URL = "postgresql://bonds:bonds_secret@localhost:5432/bonds"
CORPUS_DIR = Path("/home/chukiss/text_project/hypothesis/data/corpus")
MODELS_DIR = Path(__file__).parent / "models_v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        logger.warning("Force exit")
        sys.exit(1)
    logger.warning("Shutdown requested - finishing current book...")
    SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_database():
    """Create tables if not exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS hyp_semantic_progress (
            book_path TEXT PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT NOW(),
            n_words INTEGER,
            n_nouns INTEGER,
            n_verbs INTEGER,
            n_adjs INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS hyp_semantic_words (
            word TEXT PRIMARY KEY,
            word_type INTEGER,  -- 0=noun, 1=verb, 2=adj, 3=other
            j_sum FLOAT8[],     -- accumulated j vectors
            i_sum FLOAT8[],     -- accumulated i vectors
            tau_sum FLOAT8,
            variety_sum FLOAT8,
            verb_sum FLOAT8[],  -- accumulated verb vectors (for verbs only)
            count INTEGER,      -- total occurrences
            n_books INTEGER     -- books containing this word
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS hyp_semantic_index (
            word TEXT PRIMARY KEY,
            word_type INTEGER,
            j FLOAT8[],
            i FLOAT8[],
            tau FLOAT8,
            variety FLOAT8,
            verb FLOAT8[],
            count INTEGER,
            n_books INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized")


def get_status():
    """Get processing status."""
    conn = get_connection()
    cur = conn.cursor()

    # Total books in corpus
    all_books = list(CORPUS_DIR.glob("*/*.txt.zst"))
    total_books = len(all_books)

    # Processed books
    cur.execute("SELECT COUNT(*) FROM hyp_semantic_progress")
    processed = cur.fetchone()[0]

    # Words in accumulator
    cur.execute("SELECT COUNT(*) FROM hyp_semantic_words")
    words_accumulated = cur.fetchone()[0]

    # Words in final index
    cur.execute("SELECT COUNT(*) FROM hyp_semantic_index")
    words_finalized = cur.fetchone()[0]

    cur.close()
    conn.close()

    return {
        "total_books": total_books,
        "processed_books": processed,
        "remaining_books": total_books - processed,
        "progress_pct": 100.0 * processed / total_books if total_books > 0 else 0,
        "words_accumulated": words_accumulated,
        "words_finalized": words_finalized
    }


def get_pending_books(limit: int = 100):
    """Get books that haven't been processed yet."""
    conn = get_connection()
    cur = conn.cursor()

    # Get all processed book paths
    cur.execute("SELECT book_path FROM hyp_semantic_progress")
    processed = set(row[0] for row in cur.fetchall())

    cur.close()
    conn.close()

    # Find unprocessed books
    all_books = sorted(CORPUS_DIR.glob("*/*.txt.zst"))
    pending = [b for b in all_books if str(b) not in processed]

    return pending[:limit]


def load_model():
    """Load trained semantic model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(
        MODELS_DIR / "semantic_bottleneck_v2_best.pt",
        map_location=device,
        weights_only=False
    )

    with open(MODELS_DIR / "vocabulary.json") as f:
        vocab_data = json.load(f)

    word2idx = vocab_data['word2idx']

    model = SemanticBottleneckV2(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint['embed_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        n_basis_adjectives=checkpoint['n_basis_adjectives']
    ).to(device)

    basis_ids = torch.arange(checkpoint['n_basis_adjectives'], device=device)
    model.noun_encoder.set_basis_adjectives(basis_ids)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint['epoch']}) on {device}")

    return model, word2idx, device


def load_spacy():
    """Load spaCy model."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        import subprocess
        subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.max_length = 2_000_000
    return nlp


def read_book(path: Path) -> str:
    """Read zstd-compressed book."""
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        data = f.read()
    return dctx.decompress(data).decode('utf-8', errors='ignore')


def tokenize(nlp, text: str):
    """Tokenize with spaCy."""
    if len(text) > 1_500_000:
        text = text[:1_500_000]

    doc = nlp(text)
    words = []
    word_types = []

    pos_map = {'NOUN': 0, 'PROPN': 0, 'VERB': 1, 'AUX': 1, 'ADJ': 2}

    for token in doc:
        if token.is_alpha and len(token.text) > 1:
            words.append(token.text.lower())
            word_types.append(pos_map.get(token.pos_, 3))

    return words, word_types


def extract_vectors(model, word2idx, words, word_types, device, batch_size=256):
    """Extract semantic vectors."""
    word_type_counts = defaultdict(lambda: Counter())
    for w, t in zip(words, word_types):
        if w in word2idx:
            word_type_counts[w][t] += 1

    word_to_type = {w: counts.most_common(1)[0][0]
                    for w, counts in word_type_counts.items()}
    valid_words = list(word_to_type.keys())

    if not valid_words:
        return {}

    results = {}

    for i in range(0, len(valid_words), batch_size):
        batch_words = valid_words[i:i+batch_size]
        batch_ids = torch.tensor(
            [[word2idx[w]] for w in batch_words],
            dtype=torch.long, device=device
        )
        batch_types = torch.tensor(
            [[word_to_type[w]] for w in batch_words],
            dtype=torch.long, device=device
        )

        with torch.no_grad():
            semantic = model.encode(batch_ids, batch_types)

        for j, word in enumerate(batch_words):
            wtype = word_to_type[word]
            results[word] = {
                'j': semantic['j'][j].cpu().numpy().tolist(),
                'i': semantic['i'][j].cpu().numpy().tolist(),
                'tau': semantic['tau'][j].item(),
                'variety': semantic['variety'][j].item(),
                'verb': semantic['verb'][j].cpu().numpy().tolist() if wtype == 1 else None,
                'word_type': wtype
            }

    return results


def update_word_accumulator(conn, vectors: dict, word_counts: Counter):
    """Update accumulated word vectors in database."""
    cur = conn.cursor()

    for word, vec in vectors.items():
        j = vec['j']
        i = vec['i']
        tau = vec['tau']
        variety = vec['variety']
        verb = vec['verb']
        wtype = vec['word_type']
        count = word_counts.get(word, 1)

        # Upsert: insert or update accumulated vectors
        cur.execute("""
            INSERT INTO hyp_semantic_words
                (word, word_type, j_sum, i_sum, tau_sum, variety_sum, verb_sum, count, n_books)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
            ON CONFLICT (word) DO UPDATE SET
                j_sum = ARRAY(
                    SELECT COALESCE(a.v, 0) + COALESCE(b.v, 0)
                    FROM unnest(hyp_semantic_words.j_sum) WITH ORDINALITY AS a(v, ord)
                    FULL OUTER JOIN unnest(%s::float8[]) WITH ORDINALITY AS b(v, ord)
                    ON a.ord = b.ord
                    ORDER BY COALESCE(a.ord, b.ord)
                ),
                i_sum = ARRAY(
                    SELECT COALESCE(a.v, 0) + COALESCE(b.v, 0)
                    FROM unnest(hyp_semantic_words.i_sum) WITH ORDINALITY AS a(v, ord)
                    FULL OUTER JOIN unnest(%s::float8[]) WITH ORDINALITY AS b(v, ord)
                    ON a.ord = b.ord
                    ORDER BY COALESCE(a.ord, b.ord)
                ),
                tau_sum = hyp_semantic_words.tau_sum + %s,
                variety_sum = hyp_semantic_words.variety_sum + %s,
                verb_sum = CASE
                    WHEN %s IS NOT NULL THEN ARRAY(
                        SELECT COALESCE(a.v, 0) + COALESCE(b.v, 0)
                        FROM unnest(hyp_semantic_words.verb_sum) WITH ORDINALITY AS a(v, ord)
                        FULL OUTER JOIN unnest(%s::float8[]) WITH ORDINALITY AS b(v, ord)
                        ON a.ord = b.ord
                        ORDER BY COALESCE(a.ord, b.ord)
                    )
                    ELSE hyp_semantic_words.verb_sum
                END,
                count = hyp_semantic_words.count + %s,
                n_books = hyp_semantic_words.n_books + 1
        """, (word, wtype, j, i, tau, variety, verb, count,
              j, i, tau, variety, verb, verb, count))

    conn.commit()
    cur.close()


def process_book(book_path: Path, model, word2idx, device, nlp, conn):
    """Process a single book."""
    try:
        text = read_book(book_path)
        words, word_types = tokenize(nlp, text)

        if len(words) < 100:
            # Mark as processed but skip
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO hyp_semantic_progress (book_path, n_words, n_nouns, n_verbs, n_adjs)
                VALUES (%s, %s, 0, 0, 0)
            """, (str(book_path), len(words)))
            conn.commit()
            cur.close()
            return 0

        word_counts = Counter(words)
        vectors = extract_vectors(model, word2idx, words, word_types, device)

        # Update accumulator
        update_word_accumulator(conn, vectors, word_counts)

        # Mark book as processed
        n_nouns = sum(1 for t in word_types if t == 0)
        n_verbs = sum(1 for t in word_types if t == 1)
        n_adjs = sum(1 for t in word_types if t == 2)

        cur = conn.cursor()
        cur.execute("""
            INSERT INTO hyp_semantic_progress (book_path, n_words, n_nouns, n_verbs, n_adjs)
            VALUES (%s, %s, %s, %s, %s)
        """, (str(book_path), len(words), n_nouns, n_verbs, n_adjs))
        conn.commit()
        cur.close()

        return len(vectors)

    except Exception as e:
        logger.error(f"Error processing {book_path.name}: {e}")
        # Mark as processed to skip
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO hyp_semantic_progress (book_path, n_words, n_nouns, n_verbs, n_adjs)
            VALUES (%s, 0, 0, 0, 0)
            ON CONFLICT (book_path) DO NOTHING
        """, (str(book_path),))
        conn.commit()
        cur.close()
        return 0


def finalize_index():
    """Create final averaged index from accumulated vectors."""
    logger.info("Finalizing index...")

    conn = get_connection()
    cur = conn.cursor()

    # Clear existing final index
    cur.execute("DELETE FROM hyp_semantic_index")

    # Compute averages and insert
    cur.execute("""
        INSERT INTO hyp_semantic_index (word, word_type, j, i, tau, variety, verb, count, n_books)
        SELECT
            word,
            word_type,
            ARRAY(SELECT unnest(j_sum) / n_books),
            ARRAY(SELECT unnest(i_sum) / n_books),
            tau_sum / n_books,
            variety_sum / n_books,
            CASE WHEN verb_sum IS NOT NULL
                 THEN ARRAY(SELECT unnest(verb_sum) / n_books)
                 ELSE NULL
            END,
            count,
            n_books
        FROM hyp_semantic_words
        WHERE n_books >= 1
    """)

    cur.execute("SELECT COUNT(*) FROM hyp_semantic_index")
    total = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    logger.info(f"Final index: {total} words")
    return total


def export_index(output_path: Path):
    """Export final index to JSON."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT word, word_type, j, i, tau, variety, verb, count, n_books
        FROM hyp_semantic_index
    """)

    index = {}
    for row in cur.fetchall():
        word, wtype, j, i, tau, variety, verb, count, n_books = row
        index[word] = {
            'j': list(j),
            'i': list(i),
            'tau': tau,
            'variety': variety,
            'verb': list(verb) if verb else None,
            'word_type': wtype,
            'count': count,
            'n_books': n_books
        }

    cur.close()
    conn.close()

    with open(output_path, 'w') as f:
        json.dump({'index': index, 'n_words': len(index)}, f)

    logger.info(f"Exported {len(index)} words to {output_path}")


def run_batch(batch_size: int):
    """Process a batch of books."""
    init_database()

    pending = get_pending_books(batch_size)
    if not pending:
        logger.info("No pending books")
        return

    logger.info(f"Processing {len(pending)} books...")

    model, word2idx, device = load_model()
    nlp = load_spacy()
    conn = get_connection()

    total_words = 0
    for idx, book_path in enumerate(pending):
        if SHUTDOWN_REQUESTED:
            break

        n_words = process_book(book_path, model, word2idx, device, nlp, conn)
        total_words += n_words

        if (idx + 1) % 10 == 0:
            status = get_status()
            logger.info(f"Progress: {status['processed_books']}/{status['total_books']} "
                       f"({status['progress_pct']:.1f}%) - {status['words_accumulated']} words")

    conn.close()
    logger.info(f"Batch complete: {total_words} new word entries")


def run_continuous():
    """Run until all books processed."""
    init_database()

    model, word2idx, device = load_model()
    nlp = load_spacy()

    while not SHUTDOWN_REQUESTED:
        pending = get_pending_books(100)
        if not pending:
            logger.info("All books processed!")
            break

        conn = get_connection()

        for book_path in pending:
            if SHUTDOWN_REQUESTED:
                break
            process_book(book_path, model, word2idx, device, nlp, conn)

        conn.close()

        status = get_status()
        logger.info(f"Progress: {status['processed_books']}/{status['total_books']} "
                   f"({status['progress_pct']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Semantic Index Pipeline")
    parser.add_argument('--status', action='store_true', help='Show progress')
    parser.add_argument('--batch', type=int, help='Process N books')
    parser.add_argument('--continuous', action='store_true', help='Run until done')
    parser.add_argument('--finalize', action='store_true', help='Create final index')
    parser.add_argument('--export', type=str, help='Export to JSON file')
    parser.add_argument('--init', action='store_true', help='Initialize database')
    args = parser.parse_args()

    if args.init:
        init_database()
        return

    if args.status:
        status = get_status()
        print("=" * 50)
        print("SEMANTIC INDEX STATUS")
        print("=" * 50)
        for k, v in status.items():
            print(f"  {k}: {v}")
        return

    if args.finalize:
        finalize_index()
        return

    if args.export:
        export_index(Path(args.export))
        return

    if args.batch:
        run_batch(args.batch)
        return

    if args.continuous:
        run_continuous()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
