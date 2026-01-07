"""Text Extractor: Text -> Bonds.

Extracts noun-adjective bonds from text using spaCy.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from ...data.models import Bond


@dataclass
class Sentence:
    """A sentence with extracted bonds."""
    text: str
    bonds: List[Bond] = field(default_factory=list)


@dataclass
class ExtractedText:
    """Result of text extraction."""
    sentences: List[Sentence] = field(default_factory=list)

    @property
    def all_bonds(self) -> List[Bond]:
        """All bonds from all sentences."""
        bonds = []
        for sent in self.sentences:
            bonds.extend(sent.bonds)
        return bonds

    def __len__(self) -> int:
        return len(self.sentences)


class TextExtractor:
    """Extract bonds from text using spaCy or regex fallback."""

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self._nlp = None

        if use_spacy:
            self._load_spacy()

    def _load_spacy(self):
        """Load spaCy model."""
        try:
            import spacy
            self._nlp = spacy.load('en_core_web_sm')
        except (ImportError, OSError):
            print("spaCy not available, using regex fallback")
            self.use_spacy = False

    def extract(self, text: str) -> ExtractedText:
        """Extract bonds from text.

        Args:
            text: Input text

        Returns:
            ExtractedText with sentences and bonds
        """
        if self.use_spacy and self._nlp:
            return self._extract_spacy(text)
        else:
            return self._extract_regex(text)

    def _extract_spacy(self, text: str) -> ExtractedText:
        """Extract using spaCy dependency parsing."""
        doc = self._nlp(text)
        result = ExtractedText()

        for sent in doc.sents:
            sentence = Sentence(text=sent.text)

            for token in sent:
                # Pattern 1: Nouns with adjective modifiers (e.g., "dark forest")
                if token.pos_ in ('NOUN', 'PROPN'):
                    noun = token.lemma_.lower()

                    # Find adjective children
                    adjs = [
                        child.lemma_.lower()
                        for child in token.children
                        if child.pos_ == 'ADJ'
                    ]

                    if adjs:
                        for adj in adjs:
                            sentence.bonds.append(Bond(noun=noun, adj=adj))
                    else:
                        # Noun without adjective
                        sentence.bonds.append(Bond(noun=noun, adj=None))

                # Pattern 2: Predicative adjectives (e.g., "Everything is terrible")
                elif token.pos_ == 'ADJ' and token.dep_ in ('acomp', 'attr'):
                    adj = token.lemma_.lower()
                    # Find the subject
                    head = token.head
                    for child in head.children:
                        if child.dep_ == 'nsubj':
                            # Use subject as noun (even if PRON)
                            noun = child.lemma_.lower()
                            sentence.bonds.append(Bond(noun=noun, adj=adj))
                            break
                    else:
                        # No subject found, use adjective alone
                        sentence.bonds.append(Bond(noun=None, adj=adj))

                # Pattern 3: Stand-alone adjectives in context
                elif token.pos_ == 'ADJ' and token.dep_ == 'amod':
                    adj = token.lemma_.lower()
                    noun = token.head.lemma_.lower() if token.head.pos_ in ('NOUN', 'PROPN') else None
                    if noun:
                        sentence.bonds.append(Bond(noun=noun, adj=adj))

            if sentence.bonds:
                result.sentences.append(sentence)

        return result

    def _extract_regex(self, text: str) -> ExtractedText:
        """Extract using regex (fallback)."""
        result = ExtractedText()

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        # Simple adj+noun pattern
        pattern = r'\b([a-z]+)\s+([a-z]+)\b'

        for sent_text in sentences:
            sent_text = sent_text.strip()
            if not sent_text:
                continue

            sentence = Sentence(text=sent_text)

            # Find potential adj+noun pairs
            matches = re.findall(pattern, sent_text.lower())
            for word1, word2 in matches:
                # Assume word1 is adj, word2 is noun
                # This is a very rough heuristic
                sentence.bonds.append(Bond(noun=word2, adj=word1))

            if sentence.bonds:
                result.sentences.append(sentence)

        return result

    def extract_words(self, text: str) -> List[str]:
        """Extract all words from text."""
        return re.findall(r'\b[a-z]+\b', text.lower())
