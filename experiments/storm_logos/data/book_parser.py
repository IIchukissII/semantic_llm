"""Book Parser: spaCy-based bond extraction from books.

Extracts noun-adjective bonds from Gutenberg books and stores them in Neo4j.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator
from dataclasses import dataclass, field
from datetime import datetime

import spacy
from spacy.tokens import Doc

from .models import Bond
from .postgres import get_data
from .neo4j import Neo4jData, Author, Book as BookRecord


@dataclass
class ExtractedBond:
    """A bond extracted from text with position metadata."""
    adj: str
    noun: str
    chapter: int
    sentence: int
    position: int  # Position within book


@dataclass
class ChapterInfo:
    """Information about a detected chapter."""
    number: int
    title: str
    start_char: int


@dataclass
class ParsedBook:
    """Result of parsing a book."""
    title: str
    author: str
    bonds: List[ExtractedBond] = field(default_factory=list)
    n_sentences: int = 0
    n_chapters: int = 0
    metadata: Dict = field(default_factory=dict)


class BookParser:
    """spaCy-based book parser for bond extraction.

    Extracts noun-adjective bonds from text using spaCy's dependency parser.
    More accurate than regex-based approaches.
    """

    # Chapter detection patterns
    CHAPTER_PATTERNS = [
        r'^CHAPTER\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',
        r'^Chapter\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',
        r'^BOOK\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',
        r'^Book\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',
        r'^PART\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',
        r'^([IVXLCDM]+)\.\s+(.*)$',  # Roman numeral sections
    ]

    # Skip patterns (headers, metadata)
    SKIP_PATTERNS = [
        r'^[\*\-_=]{3,}',  # Horizontal rules
        r'^\[.*\]$',  # Editorial notes
        r'^Table of Contents',
        r'^THE END',
        r'^FINIS',
    ]

    def __init__(self, model: str = 'en_core_web_sm'):
        """Initialize parser with spaCy model.

        Args:
            model: spaCy model name. Use 'en_core_web_sm' for speed,
                   'en_core_web_md' or 'en_core_web_lg' for accuracy.
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"spaCy model '{model}' not found. Downloading...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

        # Compile patterns
        self.chapter_re = [re.compile(p, re.MULTILINE) for p in self.CHAPTER_PATTERNS]
        self.skip_re = [re.compile(p, re.IGNORECASE) for p in self.SKIP_PATTERNS]

        # Get data layer for coordinate lookup
        self._data = None

    @property
    def data(self):
        """Lazy-load data layer."""
        if self._data is None:
            self._data = get_data()
        return self._data

    def parse_file(self, filepath: Path, author: str = '',
                   title: str = '') -> ParsedBook:
        """Parse a book file and extract bonds.

        Args:
            filepath: Path to text file
            author: Author name (optional, will try to detect)
            title: Book title (optional, will try to detect)

        Returns:
            ParsedBook with extracted bonds
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Try to extract metadata from Gutenberg header
        if not author or not title:
            meta = self._extract_gutenberg_metadata(text)
            author = author or meta.get('author', 'Unknown')
            title = title or meta.get('title', filepath.stem)

        # Clean text
        text = self._clean_gutenberg_text(text)

        # Detect chapters
        chapters = self._detect_chapters(text)

        # Parse and extract bonds
        result = ParsedBook(
            title=title,
            author=author,
            n_chapters=len(chapters) if chapters else 1,
        )

        if chapters:
            for i, chapter in enumerate(chapters):
                # Get chapter text
                start = chapter.start_char
                end = chapters[i + 1].start_char if i + 1 < len(chapters) else len(text)
                chapter_text = text[start:end]

                bonds, n_sents = self._extract_bonds_from_text(
                    chapter_text,
                    chapter_num=chapter.number,
                    position_offset=len(result.bonds)
                )
                result.bonds.extend(bonds)
                result.n_sentences += n_sents
        else:
            # No chapters detected, process as single unit
            bonds, n_sents = self._extract_bonds_from_text(text, chapter_num=0)
            result.bonds = bonds
            result.n_sentences = n_sents

        return result

    def parse_text(self, text: str, author: str = 'Unknown',
                   title: str = 'Untitled') -> ParsedBook:
        """Parse raw text and extract bonds.

        Args:
            text: Raw text content
            author: Author name
            title: Book title

        Returns:
            ParsedBook with extracted bonds
        """
        # Detect chapters
        chapters = self._detect_chapters(text)

        # Parse and extract bonds
        result = ParsedBook(
            title=title,
            author=author,
            n_chapters=len(chapters) if chapters else 1,
        )

        if chapters:
            for i, chapter in enumerate(chapters):
                start = chapter.start_char
                end = chapters[i + 1].start_char if i + 1 < len(chapters) else len(text)
                chapter_text = text[start:end]

                bonds, n_sents = self._extract_bonds_from_text(
                    chapter_text,
                    chapter_num=chapter.number,
                    position_offset=len(result.bonds)
                )
                result.bonds.extend(bonds)
                result.n_sentences += n_sents
        else:
            bonds, n_sents = self._extract_bonds_from_text(text, chapter_num=0)
            result.bonds = bonds
            result.n_sentences = n_sents

        return result

    def _extract_gutenberg_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from Gutenberg header."""
        meta = {}

        # Look for title
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', text[:5000])
        if title_match:
            meta['title'] = title_match.group(1).strip()

        # Look for author
        author_match = re.search(r'Author:\s*(.+?)(?:\n|$)', text[:5000])
        if author_match:
            meta['author'] = author_match.group(1).strip()

        return meta

    def _clean_gutenberg_text(self, text: str) -> str:
        """Remove Gutenberg header/footer and clean text."""
        # Find start of actual content
        start_markers = [
            '*** START OF THE PROJECT GUTENBERG',
            '*** START OF THIS PROJECT GUTENBERG',
            '*END*THE SMALL PRINT',
        ]

        start_pos = 0
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                # Find next double newline after marker
                next_para = text.find('\n\n', pos)
                if next_para != -1:
                    start_pos = next_para + 2
                break

        # Find end of content
        end_markers = [
            '*** END OF THE PROJECT GUTENBERG',
            '*** END OF THIS PROJECT GUTENBERG',
            'End of the Project Gutenberg',
            'End of Project Gutenberg',
        ]

        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                end_pos = pos
                break

        return text[start_pos:end_pos]

    def _detect_chapters(self, text: str) -> List[ChapterInfo]:
        """Detect chapter boundaries in text."""
        chapters = []

        for line_match in re.finditer(r'^.+$', text, re.MULTILINE):
            line = line_match.group()

            for pattern in self.chapter_re:
                match = pattern.match(line)
                if match:
                    num_str = match.group(1)
                    title = match.group(2).strip() if match.lastindex >= 2 else ''

                    # Convert roman numerals
                    try:
                        num = self._roman_to_int(num_str) if not num_str.isdigit() else int(num_str)
                    except ValueError:
                        continue

                    chapters.append(ChapterInfo(
                        number=num,
                        title=title,
                        start_char=line_match.start(),
                    ))
                    break

        return chapters

    def _roman_to_int(self, s: str) -> int:
        """Convert roman numeral to integer."""
        roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        result = 0
        prev = 0
        for c in reversed(s.upper()):
            val = roman.get(c, 0)
            if val < prev:
                result -= val
            else:
                result += val
            prev = val
        return result

    def _extract_bonds_from_text(self, text: str, chapter_num: int = 0,
                                  position_offset: int = 0) -> Tuple[List[ExtractedBond], int]:
        """Extract bonds from text using spaCy.

        Args:
            text: Text to process
            chapter_num: Chapter number for metadata
            position_offset: Starting position for this chunk

        Returns:
            Tuple of (bonds list, sentence count)
        """
        bonds = []
        n_sentences = 0
        position = position_offset

        # Process in chunks to avoid memory issues
        chunk_size = 100000  # Characters

        for chunk_start in range(0, len(text), chunk_size):
            chunk = text[chunk_start:chunk_start + chunk_size]

            # Don't cut in middle of sentence
            if chunk_start + chunk_size < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:
                    chunk = chunk[:last_period + 1]

            # Process with spaCy
            doc = self.nlp(chunk)

            for sent_idx, sent in enumerate(doc.sents):
                n_sentences += 1

                # Skip short sentences
                if len(sent) < 3:
                    continue

                # Extract noun-adjective bonds from this sentence
                sent_bonds = self._extract_bonds_from_sentence(sent)

                for adj, noun in sent_bonds:
                    bonds.append(ExtractedBond(
                        adj=adj,
                        noun=noun,
                        chapter=chapter_num,
                        sentence=n_sentences,
                        position=position,
                    ))
                    position += 1

        return bonds, n_sentences

    def _extract_bonds_from_sentence(self, sent) -> List[Tuple[str, str]]:
        """Extract noun-adjective bonds from a spaCy sentence.

        Uses dependency parsing for more accurate extraction than regex.

        Args:
            sent: spaCy Span (sentence)

        Returns:
            List of (adjective, noun) tuples
        """
        bonds = []

        for token in sent:
            # Look for adjectives modifying nouns
            if token.pos_ == 'ADJ' and token.dep_ == 'amod':
                head = token.head
                if head.pos_ == 'NOUN':
                    adj = token.lemma_.lower()
                    noun = head.lemma_.lower()

                    # Filter out very short or numeric
                    if len(adj) >= 2 and len(noun) >= 2:
                        if not adj.isdigit() and not noun.isdigit():
                            bonds.append((adj, noun))

            # Also look for predicative adjectives (e.g., "the sky is blue")
            elif token.pos_ == 'ADJ' and token.dep_ in ('acomp', 'attr'):
                # Find the subject
                for child in token.head.children:
                    if child.dep_ == 'nsubj' and child.pos_ == 'NOUN':
                        adj = token.lemma_.lower()
                        noun = child.lemma_.lower()
                        if len(adj) >= 2 and len(noun) >= 2:
                            if not adj.isdigit() and not noun.isdigit():
                                bonds.append((adj, noun))
                        break

        return bonds

    def lookup_coordinates(self, bonds: List[ExtractedBond]) -> List[Tuple[ExtractedBond, Bond]]:
        """Look up coordinates for extracted bonds.

        Computes bond coordinates by averaging adjective and noun coordinates.

        Args:
            bonds: List of extracted bonds

        Returns:
            List of (extracted_bond, bond_with_coordinates) tuples
        """
        result = []

        for eb in bonds:
            # Get coordinates for both words
            adj_coords = self.data.get(eb.adj)
            noun_coords = self.data.get(eb.noun)

            if adj_coords and noun_coords:
                # Average the coordinates
                A = (adj_coords.A + noun_coords.A) / 2
                S = (adj_coords.S + noun_coords.S) / 2
                tau = (adj_coords.tau + noun_coords.tau) / 2
            elif adj_coords:
                A, S, tau = adj_coords.A, adj_coords.S, adj_coords.tau
            elif noun_coords:
                A, S, tau = noun_coords.A, noun_coords.S, noun_coords.tau
            else:
                # Use defaults for unknown words
                A, S, tau = 0.0, 0.0, 2.5

            bond = Bond(
                adj=eb.adj,
                noun=eb.noun,
                A=A,
                S=S,
                tau=tau,
            )
            result.append((eb, bond))

        return result


class BookProcessor:
    """Processes books and loads them into Neo4j."""

    # Book metadata for priority books (matched to actual filenames)
    # NOTE: Some files have incorrect content (apollodorus has wrong book, Iliad file has Odyssey)
    BOOK_METADATA = {
        # Jung (5 books)
        'Jung, Carl Gustav - Four Archetypes.txt': {
            'title': 'Four Archetypes',
            'author': 'Carl Jung',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychology',
        },
        'Jung_Carl_Gustav_-_Man_and_his_Symbols.txt': {
            'title': 'Man and His Symbols',
            'author': 'Carl Jung',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychology',
        },
        'Jung, Carl Gustav - Memories, Dreams, Reflections.txt': {
            'title': 'Memories, Dreams, Reflections',
            'author': 'Carl Jung',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychology',
        },
        'Jung_Collected_Papers.txt': {
            'title': 'Collected Papers on Analytical Psychology',
            'author': 'Carl Jung',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychology',
        },
        'Jung_Psychology_of_the_Unconscious.txt': {
            'title': 'Psychology of the Unconscious',
            'author': 'Carl Jung',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychology',
        },
        # Homer - Odyssey (using homer_odyssey.txt, the other file also has Odyssey)
        'homer_odyssey.txt': {
            'title': 'The Odyssey',
            'author': 'Homer',
            'genre': 'epic',
            'era': 'ancient',
            'domain': 'mythology',
        },
        # Homer - Odyssey (Pope translation, named as Iliad but is actually Odyssey)
        'Homer - The Iliad.txt': {
            'title': 'The Odyssey (Pope Translation)',
            'author': 'Homer',
            'genre': 'epic',
            'era': 'ancient',
            'domain': 'mythology',
        },
        # Ovid
        'ovid_metamorphoses.txt': {
            'title': 'Metamorphoses',
            'author': 'Ovid',
            'genre': 'mythology',
            'era': 'ancient',
            'domain': 'mythology',
        },
        # Bulfinch
        'bulfinch_mythology.txt': {
            'title': "Bulfinch's Mythology",
            'author': 'Thomas Bulfinch',
            'genre': 'mythology',
            'era': '19th_century',
            'domain': 'mythology',
        },
        # NOTE: apollodorus_library.txt contains wrong book (Sabatini), skipping

        # Comparative Mythology (Campbell alternatives)
        'Frazer_The_Golden_Bough.txt': {
            'title': 'The Golden Bough: A Study of Magic and Religion',
            'author': 'James George Frazer',
            'genre': 'mythology',
            'era': '19th_century',
            'domain': 'comparative_religion',
        },
        'Spence_Introduction_to_Mythology.txt': {
            'title': 'An Introduction to Mythology',
            'author': 'Lewis Spence',
            'genre': 'mythology',
            'era': '20th_century',
            'domain': 'comparative_mythology',
        },
        'Fiske_Myths_and_Myth_Makers.txt': {
            'title': 'Myths and Myth-Makers',
            'author': 'John Fiske',
            'genre': 'mythology',
            'era': '19th_century',
            'domain': 'comparative_mythology',
        },
        'Lang_Modern_Mythology.txt': {
            'title': 'Modern Mythology',
            'author': 'Andrew Lang',
            'genre': 'mythology',
            'era': '19th_century',
            'domain': 'comparative_mythology',
        },
        'Doane_Bible_Myths.txt': {
            'title': 'Bible Myths and their Parallels in Other Religions',
            'author': 'Thomas William Doane',
            'genre': 'mythology',
            'era': '19th_century',
            'domain': 'comparative_religion',
        },

        # Freud (6 books)
        'Freud, Sigmund - A General Introduction to Psychoanalysis.txt': {
            'title': 'A General Introduction to Psychoanalysis',
            'author': 'Sigmund Freud',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychoanalysis',
        },
        'Freud, Sigmund - Three Contributions to the Theory of Sex.txt': {
            'title': 'Three Contributions to the Theory of Sex',
            'author': 'Sigmund Freud',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychoanalysis',
        },
        'Freud, Sigmund - Group Psychology and the Analysis of the Ego.txt': {
            'title': 'Group Psychology and the Analysis of the Ego',
            'author': 'Sigmund Freud',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychoanalysis',
        },
        'Freud, Sigmund - Psychopathology of Everyday Life.txt': {
            'title': 'Psychopathology of Everyday Life',
            'author': 'Sigmund Freud',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychoanalysis',
        },
        'Freud, Sigmund - Totem and Taboo.txt': {
            'title': 'Totem and Taboo',
            'author': 'Sigmund Freud',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychoanalysis',
        },
        'Freud, Sigmund - Dream Psychology.txt': {
            'title': 'Dream Psychology: Psychoanalysis for Beginners',
            'author': 'Sigmund Freud',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychoanalysis',
        },

        # Otto Rank
        'Rank, Otto - The Myth of the Birth of the Hero.txt': {
            'title': 'The Myth of the Birth of the Hero',
            'author': 'Otto Rank',
            'genre': 'psychology',
            'era': '20th_century',
            'domain': 'psychoanalysis',
        },

        # Bible (KJV)
        'Bible - Genesis (KJV).txt': {
            'title': 'Genesis',
            'author': 'Bible (KJV)',
            'genre': 'religious',
            'era': 'ancient',
            'domain': 'scripture',
        },
        'Bible - Job (KJV).txt': {
            'title': 'Book of Job',
            'author': 'Bible (KJV)',
            'genre': 'religious',
            'era': 'ancient',
            'domain': 'scripture',
        },

        # Dostoevsky (4 books)
        'Dostoevsky, Fyodor - Crime and Punishment.txt': {
            'title': 'Crime and Punishment',
            'author': 'Fyodor Dostoevsky',
            'genre': 'fiction',
            'era': '19th_century',
            'domain': 'literature',
        },
        'Dostoevsky, Fyodor - Notes from the Underground.txt': {
            'title': 'Notes from Underground',
            'author': 'Fyodor Dostoevsky',
            'genre': 'fiction',
            'era': '19th_century',
            'domain': 'literature',
        },
        'Dostoevsky, Fyodor - The Brothers Karamazov.txt': {
            'title': 'The Brothers Karamazov',
            'author': 'Fyodor Dostoevsky',
            'genre': 'fiction',
            'era': '19th_century',
            'domain': 'literature',
        },
        'Dostoevsky, Fyodor - The Gambler.txt': {
            'title': 'The Gambler',
            'author': 'Fyodor Dostoevsky',
            'genre': 'fiction',
            'era': '19th_century',
            'domain': 'literature',
        },
    }

    def __init__(self, neo4j: Neo4jData = None):
        """Initialize processor.

        Args:
            neo4j: Neo4jData instance. If None, will create one.
        """
        self.parser = BookParser()
        self.neo4j = neo4j

    def connect(self) -> bool:
        """Connect to Neo4j."""
        if self.neo4j is None:
            from .neo4j import get_neo4j
            self.neo4j = get_neo4j()
        return self.neo4j.connect()

    def process_book(self, filepath: Path, metadata: Dict = None) -> Dict:
        """Process a single book and load into Neo4j.

        Args:
            filepath: Path to book file
            metadata: Optional metadata dict with title, author, genre, etc.

        Returns:
            Processing result with stats
        """
        filename = filepath.name

        # Get metadata
        meta = metadata or self.BOOK_METADATA.get(filename, {})
        author_name = meta.get('author', 'Unknown')
        title = meta.get('title', filepath.stem)
        genre = meta.get('genre', '')
        era = meta.get('era', '')
        domain = meta.get('domain', '')

        print(f"Processing: {title} by {author_name}")

        # Parse book
        parsed = self.parser.parse_file(filepath, author=author_name, title=title)
        print(f"  Extracted {len(parsed.bonds)} bonds from {parsed.n_sentences} sentences")

        # Look up coordinates
        bonds_with_coords = self.parser.lookup_coordinates(parsed.bonds)
        found_coords = sum(1 for eb, b in bonds_with_coords if b.A != 0 or b.S != 0 or b.tau != 2.5)
        print(f"  Found coordinates for {found_coords}/{len(bonds_with_coords)} bonds")

        # Create author
        author = Author(name=author_name, era=era, domain=domain)
        self.neo4j.add_author(author)

        # Create book ID from filename
        book_id = filepath.stem.replace(' ', '_').lower()

        # Create book record
        book = BookRecord(
            id=book_id,
            title=title,
            author=author_name,
            filename=filename,
            genre=genre,
            processed_at=datetime.now(),
            n_bonds=len(bonds_with_coords),
            n_sentences=parsed.n_sentences,
            n_chapters=parsed.n_chapters,
        )
        self.neo4j.add_book(book)

        # Add bonds and relationships
        prev_bond = None
        for i, (eb, bond) in enumerate(bonds_with_coords):
            # Add bond node
            bond_id = self.neo4j.add_bond(bond)

            # Add CONTAINS relationship
            self.neo4j.add_bond_to_book(
                bond_id=bond_id,
                book_id=book_id,
                chapter=eb.chapter,
                sentence=eb.sentence,
                position=eb.position,
            )

            # Add FOLLOWS relationship
            if prev_bond is not None:
                self.neo4j.add_follows(
                    source=prev_bond,
                    target=bond,
                    book_id=book_id,
                    chapter=eb.chapter,
                    sentence=eb.sentence,
                    position=eb.position,
                )

            prev_bond = bond

            # Progress
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1}/{len(bonds_with_coords)} bonds...")

        print(f"  Done: {len(bonds_with_coords)} bonds loaded")

        return {
            'book_id': book_id,
            'title': title,
            'author': author_name,
            'n_bonds': len(bonds_with_coords),
            'n_sentences': parsed.n_sentences,
            'n_chapters': parsed.n_chapters,
            'coords_found': found_coords,
        }

    def process_directory(self, directory: Path,
                          pattern: str = '*.txt',
                          limit: int = None) -> List[Dict]:
        """Process all books in a directory.

        Args:
            directory: Directory containing book files
            pattern: Glob pattern for book files
            limit: Maximum number of books to process

        Returns:
            List of processing results
        """
        files = list(directory.glob(pattern))
        if limit:
            files = files[:limit]

        results = []
        for filepath in files:
            try:
                result = self.process_book(filepath)
                results.append(result)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                results.append({
                    'file': str(filepath),
                    'error': str(e),
                })

        return results

    def process_priority_books(self, gutenberg_dir: Path) -> List[Dict]:
        """Process the priority books (Jung + Mythology).

        Args:
            gutenberg_dir: Path to Gutenberg books directory

        Returns:
            List of processing results
        """
        results = []

        for filename, meta in self.BOOK_METADATA.items():
            filepath = gutenberg_dir / filename
            if filepath.exists():
                result = self.process_book(filepath, meta)
                results.append(result)
            else:
                print(f"File not found: {filepath}")
                results.append({
                    'file': filename,
                    'error': 'File not found',
                })

        return results
