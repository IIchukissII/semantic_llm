#!/usr/bin/env python3
"""
Populate word types (noun, verb, adjective, adverb) directly in PostgreSQL.

Uses NLTK WordNet to determine the primary part of speech for each word.
"""

import psycopg2
from collections import Counter
from tqdm import tqdm

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Install nltk: pip install nltk")
    exit(1)

# Ensure wordnet is downloaded
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Database config
DB_CONFIG = {
    'dbname': 'hyp',
    'user': 'semantic',
    'password': 'semantic',
    'host': 'localhost',
    'port': 5432
}

# Word type mapping: string -> int
# Keeping numeric for DB, will map in code
POS_MAP = {
    'n': 1,  # noun
    'v': 2,  # verb
    'a': 3,  # adjective
    's': 3,  # satellite adjective -> adjective
    'r': 4,  # adverb
}

TYPE_NAMES = {
    0: 'unknown',
    1: 'noun',
    2: 'verb',
    3: 'adjective',
    4: 'adverb',
}

# Morphological heuristics for words not in WordNet
ADJ_SUFFIXES = ['ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ial', 'ic', 'ical',
                'ant', 'ent', 'ary', 'ory', 'ile', 'ine', 'esque', 'like']
VERB_SUFFIXES = ['ize', 'ise', 'ify', 'ate', 'en']
ADV_SUFFIXES = ['ly', 'ward', 'wards', 'wise']
NOUN_SUFFIXES = ['tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'er', 'or', 'ist',
                 'ism', 'ance', 'ence', 'dom', 'ship', 'hood', 'age']


def get_word_type_wordnet(word):
    """Get word type from WordNet synsets."""
    synsets = wn.synsets(word)
    if not synsets:
        return None

    # Count POS occurrences
    pos_counts = Counter()
    for syn in synsets:
        pos = syn.pos()
        if pos in POS_MAP:
            pos_counts[POS_MAP[pos]] += 1

    if not pos_counts:
        return None

    # Return most common
    return pos_counts.most_common(1)[0][0]


def get_word_type_heuristic(word):
    """Fallback heuristic based on suffixes."""
    word_lower = word.lower()

    # Check adverbs first (most specific)
    for suffix in ADV_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return 4  # adverb

    # Check adjectives
    for suffix in ADJ_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return 3  # adjective

    # Check verbs
    for suffix in VERB_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return 2  # verb

    # Check nouns
    for suffix in NOUN_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
            return 1  # noun

    # Default to noun (most common)
    return 1


def get_word_type(word):
    """Get word type using WordNet with heuristic fallback."""
    wn_type = get_word_type_wordnet(word)
    if wn_type:
        return wn_type
    return get_word_type_heuristic(word)


def populate_word_types():
    """Main function to populate word types in PostgreSQL."""
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get all words
    print("Loading words from database...")
    cur.execute("SELECT word FROM hyp_semantic_index")
    words = [row[0] for row in cur.fetchall()]
    print(f"Found {len(words)} words")

    # Classify words
    print("Classifying word types...")
    type_counts = Counter()
    updates = []

    for word in tqdm(words):
        wtype = get_word_type(word)
        type_counts[wtype] += 1
        updates.append((wtype, word))

    # Print summary
    print(f"\nWord type distribution:")
    for wtype_id, count in type_counts.most_common():
        pct = 100 * count / len(words)
        print(f"  {TYPE_NAMES[wtype_id]:12s}: {count:6d} ({pct:.1f}%)")

    # Update database
    print(f"\nUpdating database...")
    cur.executemany(
        "UPDATE hyp_semantic_index SET word_type = %s WHERE word = %s",
        updates
    )
    conn.commit()

    print(f"Updated {len(updates)} words")

    # Verify
    cur.execute("""
        SELECT word_type, COUNT(*)
        FROM hyp_semantic_index
        GROUP BY word_type
        ORDER BY word_type
    """)
    print("\nVerification from DB:")
    for wtype, count in cur.fetchall():
        print(f"  {TYPE_NAMES.get(wtype, 'unknown'):12s}: {count}")

    cur.close()
    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    populate_word_types()
