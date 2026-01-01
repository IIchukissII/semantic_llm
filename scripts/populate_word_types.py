#!/usr/bin/env python3
"""
Populate word types (noun, verb, adjective, adverb) for all words.

Uses NLTK WordNet to determine the primary part of speech for each word.
WordNet POS tags:
  n = noun
  v = verb
  a = adjective
  r = adverb
  s = adjective satellite (treated as adjective)

Method:
  1. Look up word in WordNet
  2. Count synsets for each POS
  3. Assign most common POS as word_type
  4. If not in WordNet, use morphological heuristics
"""

import csv
from pathlib import Path
from collections import Counter
from tqdm import tqdm

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Install nltk: pip install nltk")
    print("Then run: python -c \"import nltk; nltk.download('wordnet')\"")
    exit(1)

# Ensure wordnet is downloaded
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CSV_FILE = DATA_DIR / "csv" / "word_vectors.csv"
OUTPUT_FILE = DATA_DIR / "csv" / "word_vectors.csv"  # Overwrite

# Word type mapping
POS_MAP = {
    'n': 'noun',
    'v': 'verb',
    'a': 'adjective',
    's': 'adjective',  # satellite adjective
    'r': 'adverb',
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
            return 'adverb'

    # Check adjectives
    for suffix in ADJ_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return 'adjective'

    # Check verbs
    for suffix in VERB_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return 'verb'

    # Check nouns
    for suffix in NOUN_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
            return 'noun'

    # Default to noun (most common)
    return 'noun'


def get_word_type(word):
    """Get word type using WordNet with heuristic fallback."""
    wn_type = get_word_type_wordnet(word)
    if wn_type:
        return wn_type
    return get_word_type_heuristic(word)


def populate_word_types():
    """Main function to populate word types."""
    if not CSV_FILE.exists():
        print(f"Error: {CSV_FILE} not found")
        return

    # Load existing data
    print(f"Loading {CSV_FILE}...")
    rows = []
    fieldnames = None

    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} words")

    # Classify words
    print("Classifying word types...")
    type_counts = Counter()

    for row in tqdm(rows):
        word = row['word']
        word_type = get_word_type(word)
        row['word_type'] = word_type
        type_counts[word_type] += 1

    # Print summary
    print(f"\nWord type distribution:")
    for wtype, count in type_counts.most_common():
        pct = 100 * count / len(rows)
        print(f"  {wtype:12s}: {count:6d} ({pct:.1f}%)")

    # Write output
    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done!")

    # Show samples per type
    print("\nSamples per type:")
    for wtype in ['noun', 'verb', 'adjective', 'adverb']:
        samples = [r['word'] for r in rows if r['word_type'] == wtype][:5]
        print(f"  {wtype}: {', '.join(samples)}")

    return type_counts


if __name__ == '__main__':
    populate_word_types()
