"""
Book Processor for Meaning Chain (with Learning)

Processes books to extract:
1. SVO (Subject-Verb-Object) patterns → VIA relationships
2. Adj-Noun pairs → Learning new concepts via entropy-based τ

Uses spaCy for dependency parsing to extract grammatical relations.

Learning approach:
- Track adjective co-occurrences for each noun
- Compute τ from Shannon entropy of adjective distribution
- Compute g, j from adjective vector centroids
- New words are learned, existing words update their parameters

"Reading is walking through another's semantic territory
 with verbs as guides."
"""

import re
import spacy
import numpy as np
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
from graph.learning import Neo4jLearningStore, VectorCalculator


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

    # Learning statistics
    adj_noun_pairs: int = 0
    new_concepts_learned: int = 0
    existing_concepts_updated: int = 0

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


@dataclass
class AdjNounPair:
    """An Adjective-Noun pair for learning."""
    adjective: str
    noun: str
    sentence: str = ""


class BookProcessor:
    """
    Process books to extract SVO patterns and learn new concepts.

    Two extraction modes:
    1. SVO Extraction: Subject-Verb-Object patterns → VIA relationships
    2. Adj-Noun Extraction: Adjective-Noun pairs → Concept learning

    Learning:
    - New nouns are learned from their adjective distributions
    - τ = f(entropy) of adjective distribution
    - g, j computed from adjective vector centroids
    - Existing concepts update their parameters with new observations

    Creates VIA relationships: (subject)-[:VIA {verb}]->(object)
    Creates DESCRIBED_BY relationships: (noun)-[:DESCRIBED_BY]->(adjective)
    """

    def __init__(self, graph: MeaningGraph = None,
                 model: str = "en_core_web_sm",
                 enable_learning: bool = True,
                 include_proper_nouns: bool = True,
                 adj_vectors: Dict[str, np.ndarray] = None):
        """
        Initialize BookProcessor.

        Args:
            graph: MeaningGraph instance
            model: spaCy model name
            enable_learning: If True, learn new concepts from adj-noun pairs
            include_proper_nouns: If True, include proper nouns (names) in extraction
            adj_vectors: {adjective: 5D j-vector} for computing centroids
        """
        self.graph = graph or MeaningGraph()
        self.enable_learning = enable_learning
        self.include_proper_nouns = include_proper_nouns
        self.adj_vectors = adj_vectors or {}

        # Learning store (lazy init)
        self._learning_store = None

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

        # Setup learning if enabled
        if self.enable_learning:
            self._setup_learning()

    def _load_known_concepts(self):
        """Load known concepts from graph for filtering."""
        if not self.graph.is_connected():
            return

        with self.graph.driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN c.word as word")
            self.known_concepts = {r["word"] for r in result}
            print(f"[BookProcessor] Loaded {len(self.known_concepts)} known concepts")

    def _setup_learning(self):
        """Initialize learning components."""
        if self.graph.is_connected():
            self._learning_store = Neo4jLearningStore(self.graph.driver)
            self._learning_store.setup_schema()
            print("[BookProcessor] Learning enabled")

    def load_adj_vectors(self, data_loader):
        """
        Load adjective vectors from DataLoader for centroid computation.

        Args:
            data_loader: DataLoader instance from core.data_loader
        """
        vectors = data_loader.load_word_vectors()
        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

        for word, v in vectors.items():
            if v.get('j'):
                self.adj_vectors[word] = np.array([v['j'].get(d, 0) for d in j_dims])

        print(f"[BookProcessor] Loaded {len(self.adj_vectors)} adjective vectors")

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

    def _is_valid_noun(self, token, word: str) -> bool:
        """Check if a token is a valid noun for extraction."""
        if len(word) < 3:
            return False
        # Known concepts are always valid
        if word in self.known_concepts:
            return True
        # Proper nouns are valid if include_proper_nouns is True
        if self.include_proper_nouns and token.pos_ == "PROPN":
            return True
        return False

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

                # Find subject (include proper nouns if enabled)
                subjects = []
                for child in token.children:
                    if child.dep_ in ('nsubj', 'nsubjpass'):
                        subj = child.lemma_.lower()
                        if self._is_valid_noun(child, subj):
                            subjects.append(subj)

                # Find object (include proper nouns if enabled)
                objects = []
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        obj = child.lemma_.lower()
                        if self._is_valid_noun(child, obj):
                            objects.append(obj)

                # Also check prep objects
                for child in token.children:
                    if child.dep_ == 'prep':
                        for grandchild in child.children:
                            if grandchild.dep_ == 'pobj':
                                obj = grandchild.lemma_.lower()
                                if self._is_valid_noun(grandchild, obj):
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

    def extract_adj_noun(self, text: str, max_sentences: int = 10000) -> List[AdjNounPair]:
        """
        Extract Adjective-Noun pairs from text for learning.

        Uses spaCy to find amod (adjectival modifier) dependencies.

        Args:
            text: Raw text
            max_sentences: Maximum sentences to process

        Returns:
            List of AdjNounPair
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text)

        # Process in chunks
        chunk_size = 100000
        pairs = []
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

                # Extract adj-noun pairs (include proper nouns if enabled)
                valid_pos = ("NOUN", "PROPN") if self.include_proper_nouns else ("NOUN",)
                for token in sent:
                    if token.pos_ in valid_pos:
                        noun = token.lemma_.lower()

                        # Skip short/invalid nouns
                        if len(noun) < 3:
                            continue

                        # Find adjective modifiers
                        for child in token.children:
                            if child.dep_ == "amod" and child.pos_ == "ADJ":
                                adj = child.lemma_.lower()
                                if len(adj) >= 3:
                                    pairs.append(AdjNounPair(
                                        adjective=adj,
                                        noun=noun,
                                        sentence=sent.text[:100]
                                    ))

        return pairs

    def _aggregate_adj_noun(self, pairs: List[AdjNounPair]) -> Dict[str, Dict[str, int]]:
        """
        Aggregate adj-noun pairs into distributions.

        Returns:
            {noun: {adjective: count}}
        """
        distributions = defaultdict(lambda: defaultdict(int))

        for pair in pairs:
            distributions[pair.noun][pair.adjective] += 1

        return {noun: dict(adjs) for noun, adjs in distributions.items()}

    def process_book(self, filepath: str, book_id: str = None,
                     skip_header_footer: bool = True,
                     max_sentences: int = 50000) -> ProcessingResult:
        """
        Process a book file: extract SVO patterns and learn concepts.

        Processing pipeline:
        1. Read and clean text
        2. Extract SVO patterns → VIA relationships
        3. Extract Adj-Noun pairs → Learning (if enabled)
        4. Update concept parameters from distributions

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

        # ============================================================
        # Phase 1: Extract SVO patterns
        # ============================================================
        print(f"  Phase 1: Extracting SVO patterns...")
        patterns = self.extract_svo(text, max_sentences)
        print(f"  Found: {len(patterns)} SVO patterns")

        # Aggregate patterns
        pattern_counts = defaultdict(int)
        for p in patterns:
            key = (p.subject, p.verb, p.object)
            pattern_counts[key] += 1

        print(f"  Unique patterns: {len(pattern_counts)}")

        # Store in graph
        if pattern_counts:
            self._store_patterns(pattern_counts, book_id)

        # ============================================================
        # Phase 2: Extract Adj-Noun pairs for learning
        # ============================================================
        adj_noun_count = 0
        new_concepts = 0
        updated_concepts = 0

        if self.enable_learning and self._learning_store:
            print(f"  Phase 2: Extracting Adj-Noun pairs for learning...")
            adj_noun_pairs = self.extract_adj_noun(text, max_sentences)
            adj_noun_count = len(adj_noun_pairs)
            print(f"  Found: {adj_noun_count} adj-noun pairs")

            # Aggregate into distributions
            distributions = self._aggregate_adj_noun(adj_noun_pairs)
            print(f"  Unique nouns with adjectives: {len(distributions)}")

            # Learn concepts
            new_concepts, updated_concepts = self._learn_from_distributions(
                distributions, source=f"book:{book_id}"
            )
            print(f"  New concepts: {new_concepts}, Updated: {updated_concepts}")

        # ============================================================
        # Build result
        # ============================================================
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
            adj_noun_pairs=adj_noun_count,
            new_concepts_learned=new_concepts,
            existing_concepts_updated=updated_concepts,
            processing_time_ms=int(elapsed)
        )

        print(f"  Summary: {len(subjects)} subj, {len(verbs)} verbs, {len(objects)} obj")
        print(f"  Learning: {new_concepts} new, {updated_concepts} updated")
        print(f"  Time: {elapsed/1000:.1f}s")

        return result

    def _learn_from_distributions(self, distributions: Dict[str, Dict[str, int]],
                                   source: str) -> Tuple[int, int]:
        """
        Learn concepts from adjective distributions.

        For each noun:
        1. Store adjective observations in Neo4j
        2. Recompute τ, g, j from distribution
        3. Track if new or updated

        Args:
            distributions: {noun: {adjective: count}}
            source: Origin identifier

        Returns:
            (new_concepts_count, updated_concepts_count)
        """
        if not self._learning_store:
            return 0, 0

        new_count = 0
        updated_count = 0

        for noun, adj_counts in distributions.items():
            # Check if concept exists
            is_new = noun not in self.known_concepts

            # Store observations and update
            result = self._learning_store.learn_concept(
                noun, adj_counts, source, self.adj_vectors
            )

            if result:
                if is_new:
                    new_count += 1
                    # Add to known concepts
                    self.known_concepts.add(noun)
                else:
                    updated_count += 1

        return new_count, updated_count

    def reprocess_book(self, filepath: str, book_id: str = None,
                       skip_header_footer: bool = True,
                       max_sentences: int = 50000) -> ProcessingResult:
        """
        Reprocess a book to update concept parameters.

        Same as process_book but:
        - Clears existing observations for this source first
        - Forces re-learning of all concepts

        Args:
            filepath: Path to book file
            book_id: Identifier
            skip_header_footer: Skip Gutenberg-style headers
            max_sentences: Maximum sentences to process

        Returns:
            ProcessingResult
        """
        # For now, just call process_book
        # The DESCRIBED_BY edges will accumulate counts
        # Could add source-specific clearing if needed
        return self.process_book(filepath, book_id, skip_header_footer, max_sentences)

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
