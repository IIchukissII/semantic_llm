"""Bond extractor for RC-Model.

Text → Bonds (preserving sentence structure)

Unification:
  - Noun + Adj       → (noun, adj)
  - Verb + Adv       → (verb_as_noun, adv_as_adj)
  - Noun alone       → (noun, None)
  - Verb alone       → (verb_as_noun, None)
  - Pronoun          → (pronoun, None)
"""

import spacy
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Bond:
    """A semantic bond (noun/verb, adj/adv)."""
    noun: str           # lemma of noun/verb/pronoun
    adj: Optional[str]  # lemma of adj/adv or None
    noun_pos: str       # original POS: NOUN, VERB, PRON
    adj_pos: Optional[str]  # original POS: ADJ, ADV, or None

    def __repr__(self):
        if self.adj:
            return f"({self.noun}, {self.adj})"
        return f"({self.noun}, -)"

    def as_tuple(self) -> tuple[str, Optional[str]]:
        return (self.noun, self.adj)


@dataclass
class SentenceBonds:
    """Bonds from a single sentence."""
    bonds: list[Bond]
    text: str  # original sentence text

    def __len__(self):
        return len(self.bonds)

    def __iter__(self):
        return iter(self.bonds)


@dataclass
class TextBonds:
    """Bonds from a full text, preserving sentence structure."""
    sentences: list[SentenceBonds]

    def __len__(self):
        return sum(len(s) for s in self.sentences)

    @property
    def flat_bonds(self) -> list[Bond]:
        """All bonds as flat list."""
        return [b for s in self.sentences for b in s.bonds]

    @property
    def sentence_boundaries(self) -> list[int]:
        """Cumulative bond indices at sentence boundaries."""
        boundaries = []
        total = 0
        for s in self.sentences:
            total += len(s)
            boundaries.append(total)
        return boundaries

    def as_nested_tuples(self) -> list[list[tuple[str, Optional[str]]]]:
        """Return as nested list of (noun, adj) tuples."""
        return [[b.as_tuple() for b in s.bonds] for s in self.sentences]


class BondExtractor:
    """Extract semantic bonds from text using spaCy."""

    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize with spaCy model.

        Args:
            model: spaCy model name. Options:
                - "en_core_web_sm" (fast, good for most cases)
                - "en_core_web_md" (medium, better vectors)
                - "en_core_web_lg" (large, best accuracy)
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.warning(f"Model {model} not found, downloading...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)

    def extract(self, text: str) -> TextBonds:
        """Extract bonds from text, preserving sentence structure.

        Args:
            text: Input text (can be multiple sentences)

        Returns:
            TextBonds with sentence structure preserved
        """
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            bonds = []

            for token in sent:
                # Noun + (optional) adjective modifier
                if token.pos_ == "NOUN":
                    adj = self._find_adj_modifier(token)
                    bonds.append(Bond(
                        noun=token.lemma_.lower(),
                        adj=adj,
                        noun_pos="NOUN",
                        adj_pos="ADJ" if adj else None,
                    ))

                # Verb + (optional) adverb modifier (verb as noun)
                elif token.pos_ == "VERB":
                    adv = self._find_adv_modifier(token)
                    bonds.append(Bond(
                        noun=token.lemma_.lower(),
                        adj=adv,
                        noun_pos="VERB",
                        adj_pos="ADV" if adv else None,
                    ))

                # Pronoun (neutral)
                elif token.pos_ == "PRON":
                    # Skip expletive pronouns like "it" in "it rains"
                    if token.dep_ == "expl":
                        continue
                    bonds.append(Bond(
                        noun=token.lemma_.lower(),
                        adj=None,
                        noun_pos="PRON",
                        adj_pos=None,
                    ))

            if bonds:  # skip empty sentences
                sentences.append(SentenceBonds(
                    bonds=bonds,
                    text=sent.text.strip(),
                ))

        return TextBonds(sentences=sentences)

    def _find_adj_modifier(self, token) -> Optional[str]:
        """Find adjective modifier for a noun."""
        for child in token.children:
            if child.dep_ == "amod" and child.pos_ == "ADJ":
                return child.lemma_.lower()
        return None

    def _find_adv_modifier(self, token) -> Optional[str]:
        """Find adverb modifier for a verb."""
        for child in token.children:
            if child.dep_ == "advmod" and child.pos_ == "ADV":
                return child.lemma_.lower()
        return None

    def extract_simple(self, text: str) -> list[list[tuple[str, Optional[str]]]]:
        """Extract bonds as simple nested list.

        Returns:
            [
                [(noun, adj), (noun, adj), ...],  # sentence 1
                [(noun, adj), ...],               # sentence 2
                ...
            ]
        """
        return self.extract(text).as_nested_tuples()


# Singleton instance
_default_extractor: Optional[BondExtractor] = None

def extract_bonds(text: str) -> TextBonds:
    """Extract bonds using default extractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = BondExtractor()
    return _default_extractor.extract(text)


def extract_bonds_simple(text: str) -> list[list[tuple[str, Optional[str]]]]:
    """Extract bonds as simple nested list."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = BondExtractor()
    return _default_extractor.extract_simple(text)
