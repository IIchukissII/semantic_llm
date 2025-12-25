"""
Book Processor for Meaning Chain

Processes books to extract SVO (Subject-Verb-Object) patterns
and stores them as VIA relationships in MeaningGraph.

Uses spaCy for dependency parsing to extract grammatical relations.

"Reading is walking through another's semantic territory
 with verbs as guides."
"""

import re
import spacy
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import sys
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph, GraphConfig


@dataclass
class ProcessingResult:
    """Result of processing a book."""
    source_id: str
    source_type: str = "book"

    # Statistics
    total_sentences: int = 0
    svo_patterns: int = 0
    unique_subjects: int = 0
    unique_verbs: int = 0
    unique_objects: int = 0

    # Timing
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: int = 0

    # Status
    success: bool = True
    error: Optional[str] = None


@dataclass
class SVOPattern:
    """A Subject-Verb-Object pattern."""
    subject: str
    verb: str
    object: str
    sentence: str = ""


class BookProcessor:
    """
    Process books to extract SVO patterns for MeaningGraph.

    Uses spaCy to parse sentences and extract grammatical relations:
    - nsubj (nominal subject)
    - ROOT (verb)
    - dobj (direct object)

    Creates VIA relationships: (subject)-[:VIA {verb}]->(object)
    """

    def __init__(self, graph: MeaningGraph = None,
                 model: str = "en_core_web_sm"):
        self.graph = graph or MeaningGraph()

        # Load spaCy model
        try:
            self.nlp = spacy.load(model)
            print(f"[BookProcessor] Loaded spaCy model: {model}")
        except OSError:
            print(f"[BookProcessor] Model {model} not found, downloading...")
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)

        # Get known concepts from graph
        self.known_concepts: Set[str] = set()
        self._load_known_concepts()

    def _load_known_concepts(self):
        """Load known concepts from graph for filtering."""
        if not self.graph.is_connected():
            return

        with self.graph.driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN c.word as word")
            self.known_concepts = {r["word"] for r in result}
            print(f"[BookProcessor] Loaded {len(self.known_concepts)} known concepts")

    def extract_svo(self, text: str, max_sentences: int = 10000) -> List[SVOPattern]:
        """
        Extract SVO patterns from text using spaCy.

        Args:
            text: Raw text
            max_sentences: Maximum sentences to process

        Returns:
            List of SVOPattern
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text)

        # Process in chunks (spaCy memory)
        chunk_size = 100000
        patterns = []
        sentences_processed = 0

        for i in range(0, len(text), chunk_size):
            if sentences_processed >= max_sentences:
                break

            chunk = text[i:i+chunk_size]
            doc = self.nlp(chunk)

            for sent in doc.sents:
                if sentences_processed >= max_sentences:
                    break

                sentences_processed += 1

                # Extract SVO from sentence
                svo = self._extract_svo_from_sentence(sent)
                if svo:
                    patterns.extend(svo)

        return patterns

    def _extract_svo_from_sentence(self, sent) -> List[SVOPattern]:
        """Extract SVO patterns from a single sentence."""
        patterns = []

        # Find verbs (ROOT of sentence)
        for token in sent:
            if token.pos_ == "VERB":
                verb = token.lemma_.lower()

                # Skip auxiliary verbs
                if verb in {'be', 'have', 'do', 'will', 'would', 'could', 'should', 'may', 'might'}:
                    continue

                # Find subject
                subjects = []
                for child in token.children:
                    if child.dep_ in ('nsubj', 'nsubjpass'):
                        subj = child.lemma_.lower()
                        if len(subj) >= 3 and subj in self.known_concepts:
                            subjects.append(subj)

                # Find object
                objects = []
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        obj = child.lemma_.lower()
                        if len(obj) >= 3 and obj in self.known_concepts:
                            objects.append(obj)

                # Also check prep objects
                for child in token.children:
                    if child.dep_ == 'prep':
                        for grandchild in child.children:
                            if grandchild.dep_ == 'pobj':
                                obj = grandchild.lemma_.lower()
                                if len(obj) >= 3 and obj in self.known_concepts:
                                    objects.append(obj)

                # Create patterns
                for subj in subjects:
                    for obj in objects:
                        if subj != obj:
                            patterns.append(SVOPattern(
                                subject=subj,
                                verb=verb,
                                object=obj,
                                sentence=sent.text[:100]
                            ))

        return patterns

    def process_book(self, filepath: str, book_id: str = None,
                     skip_header_footer: bool = True,
                     max_sentences: int = 50000) -> ProcessingResult:
        """
        Process a book file and store SVO patterns in graph.

        Args:
            filepath: Path to book file
            book_id: Identifier (defaults to filename)
            skip_header_footer: Skip Gutenberg-style headers
            max_sentences: Maximum sentences to process

        Returns:
            ProcessingResult with statistics
        """
        start_time = datetime.now()

        path = Path(filepath)
        if not path.exists():
            return ProcessingResult(
                source_id=str(filepath),
                success=False,
                error=f"File not found: {filepath}"
            )

        # Extract book ID
        if book_id is None:
            book_id = path.stem
            if " - " in book_id:
                book_id = book_id.split(" - ", 1)[1]

        print(f"\n[BookProcessor] Processing: {book_id}")

        # Read file
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return ProcessingResult(
                source_id=book_id,
                success=False,
                error=f"Read error: {e}"
            )

        # Skip header/footer
        if skip_header_footer and len(text) > 2000:
            skip = len(text) // 20
            text = text[skip:-skip]

        print(f"  Text length: {len(text)} chars")

        # Extract SVO patterns
        print(f"  Extracting SVO patterns...")
        patterns = self.extract_svo(text, max_sentences)
        print(f"  Found: {len(patterns)} SVO patterns")

        if not patterns:
            return ProcessingResult(
                source_id=book_id,
                total_sentences=max_sentences,
                svo_patterns=0
            )

        # Aggregate patterns (count duplicates)
        pattern_counts = defaultdict(int)
        for p in patterns:
            key = (p.subject, p.verb, p.object)
            pattern_counts[key] += 1

        print(f"  Unique patterns: {len(pattern_counts)}")

        # Store in graph
        self._store_patterns(pattern_counts, book_id)

        # Build result
        subjects = {k[0] for k in pattern_counts.keys()}
        verbs = {k[1] for k in pattern_counts.keys()}
        objects = {k[2] for k in pattern_counts.keys()}

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        result = ProcessingResult(
            source_id=book_id,
            total_sentences=max_sentences,
            svo_patterns=len(patterns),
            unique_subjects=len(subjects),
            unique_verbs=len(verbs),
            unique_objects=len(objects),
            processing_time_ms=int(elapsed)
        )

        print(f"  Subjects: {len(subjects)}, Verbs: {len(verbs)}, Objects: {len(objects)}")
        print(f"  Time: {elapsed/1000:.1f}s")

        return result

    def _store_patterns(self, pattern_counts: Dict[Tuple, int], book_id: str):
        """Store SVO patterns as VIA relationships."""
        if not self.graph.is_connected():
            return

        transitions = []
        for (subj, verb, obj), count in pattern_counts.items():
            transitions.append({
                "subject": subj,
                "object": obj,
                "verb": verb,
                "weight": min(1.0, 0.5 + 0.1 * count),  # Weight based on frequency
                "count": count,
                "source": f"book:{book_id}"
            })

        print(f"  Storing {len(transitions)} VIA relationships...")
        self.graph.create_via_batch(transitions)

    def close(self):
        """Close connections."""
        if self.graph:
            self.graph.close()


def process_book(filepath: str, book_id: str = None) -> ProcessingResult:
    """Convenience function to process a single book."""
    processor = BookProcessor()
    try:
        return processor.process_book(filepath, book_id)
    finally:
        processor.close()


def main():
    """CLI for book processing."""
    import argparse

    parser = argparse.ArgumentParser(description="Process books for MeaningGraph")
    parser.add_argument("filepath", help="Path to book file")
    parser.add_argument("--id", default=None, help="Book identifier")
    parser.add_argument("--max-sentences", type=int, default=50000,
                        help="Maximum sentences to process")

    args = parser.parse_args()

    result = process_book(args.filepath, args.id)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
