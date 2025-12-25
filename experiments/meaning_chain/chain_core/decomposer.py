"""
Decomposer: Sentence -> List of Meanings

Extracts semantic concepts (nouns) and operators (verbs) from input text.
Uses the semantic space vocabulary to filter to known concepts.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent paths for imports
_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM = _THIS_FILE.parent.parent.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))

from core.data_loader import DataLoader


@dataclass
class DecomposedSentence:
    """Result of sentence decomposition."""
    original: str
    nouns: List[str]           # Extracted nouns (meaning roots)
    verbs: List[str]           # Extracted verbs (operators)
    noun_verb_pairs: List[Tuple[str, str]]  # (noun, verb) associations
    unknown_words: List[str]   # Words not in semantic space


class Decomposer:
    """
    Decomposes sentences into semantic concepts.

    Uses vocabulary from the semantic space to identify:
    - Nouns: become roots of meaning trees
    - Verbs: become operators for tree expansion
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.loader = data_loader or DataLoader()

        # Load vocabularies
        self._word_vectors = None
        self._verb_operators = None
        self._noun_set = None
        self._verb_set = None

        # Common stop words to filter
        self.stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'just', 'don', 'now', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'about', 'against', 'any', 'both', 'down',
            'up', 'out', 'off', 'over', 'under', 'few', 'many', 'much'
        }

        # Minimum tau for meaningful roots (filter function words)
        self.min_tau_for_root = 2.0

        # Maximum roots to extract (focus on key concepts)
        self.max_roots = 5

        # Question words (not stop words, but don't add to meaning tree)
        self.question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which'}

    def _load_vocabularies(self):
        """Lazy load vocabularies."""
        if self._noun_set is not None:
            return

        self._word_vectors = self.loader.load_word_vectors()
        self._verb_operators = self.loader.load_verb_operators()

        # Build sets for fast lookup
        # Nouns: word_type == 0
        self._noun_set = {
            word for word, data in self._word_vectors.items()
            if data.get('word_type') == 0 and data.get('j') is not None
        }

        # Verbs: from verb operators
        self._verb_set = set(self._verb_operators.keys())

        print(f"[Decomposer] Loaded {len(self._noun_set)} nouns, {len(self._verb_set)} verbs")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens

    def _lemmatize_simple(self, word: str) -> List[str]:
        """
        Simple lemmatization without spaCy.
        Returns possible base forms of the word.
        """
        candidates = [word]

        # Common verb endings
        if word.endswith('ing'):
            candidates.append(word[:-3])       # running -> runn
            candidates.append(word[:-3] + 'e') # loving -> love
            if len(word) > 4 and word[-4] == word[-5]:
                candidates.append(word[:-4])   # running -> run
        elif word.endswith('ed'):
            candidates.append(word[:-2])       # loved -> lov
            candidates.append(word[:-1])       # loved -> love
            candidates.append(word[:-2] + 'e') # loved -> love
        elif word.endswith('s') and not word.endswith('ss'):
            candidates.append(word[:-1])       # loves -> love
            if word.endswith('es'):
                candidates.append(word[:-2])   # goes -> go
                candidates.append(word[:-2] + 'e')
            if word.endswith('ies'):
                candidates.append(word[:-3] + 'y')  # happiness -> happy
        elif word.endswith('ly'):
            candidates.append(word[:-2])       # happily -> happi
            candidates.append(word[:-2] + 'e') # nicely -> nice

        # Plural nouns
        if word.endswith('ies'):
            candidates.append(word[:-3] + 'y')
        elif word.endswith('es'):
            candidates.append(word[:-2])
            candidates.append(word[:-1])
        elif word.endswith('s') and len(word) > 3:
            candidates.append(word[:-1])

        return list(set(candidates))

    def _find_in_vocab(self, word: str, vocab_set: Set[str]) -> Optional[str]:
        """Find word or its lemma in vocabulary."""
        if word in vocab_set:
            return word

        for lemma in self._lemmatize_simple(word):
            if lemma in vocab_set:
                return lemma

        return None

    def decompose(self, text: str) -> DecomposedSentence:
        """
        Decompose a sentence into semantic concepts.

        Args:
            text: Input sentence

        Returns:
            DecomposedSentence with extracted nouns, verbs, and pairs
        """
        self._load_vocabularies()

        tokens = self._tokenize(text)

        nouns = []
        verbs = []
        unknown = []
        seen = set()

        for token in tokens:
            # Skip stop words and question words
            if token in self.stop_words or token in self.question_words:
                continue

            if token in seen:
                continue
            seen.add(token)

            found_any = False

            # Check for verb
            verb_match = self._find_in_vocab(token, self._verb_set)
            if verb_match:
                verbs.append(verb_match)
                found_any = True

            # Also check for noun - words can be BOTH verb and noun
            # e.g., "love", "dream", "help" are both concepts and operators
            noun_match = self._find_in_vocab(token, self._noun_set)
            if noun_match:
                data = self._word_vectors.get(noun_match, {})
                tau = data.get('tau', 3.0)
                if tau >= self.min_tau_for_root:
                    nouns.append((noun_match, tau))
                    found_any = True

            # Unknown word (not in either vocabulary)
            if not found_any:
                unknown.append(token)

        # Sort nouns by tau (higher = more abstract/meaningful) and limit
        nouns.sort(key=lambda x: -x[1])
        nouns = [n[0] for n in nouns[:self.max_roots]]

        # Create noun-verb pairs based on proximity
        # Simple heuristic: pair each noun with nearest verb
        pairs = []
        for noun in nouns:
            if verbs:
                # Use first verb as default operator
                pairs.append((noun, verbs[0]))

        return DecomposedSentence(
            original=text,
            nouns=nouns,
            verbs=verbs,
            noun_verb_pairs=pairs,
            unknown_words=unknown
        )

    def get_noun_properties(self, noun: str) -> Optional[Dict]:
        """Get semantic properties for a noun."""
        self._load_vocabularies()

        if noun not in self._word_vectors:
            return None

        data = self._word_vectors[noun]
        if data.get('j') is None:
            return None

        # Compute goodness (projection onto j_good)
        j_good = self.loader.get_j_good()
        j_vec = [data['j'].get(d, 0) for d in ['beauty', 'life', 'sacred', 'good', 'love']]
        goodness = sum(a * b for a, b in zip(j_vec, j_good))

        return {
            'word': noun,
            'tau': data.get('tau', 3.0),
            'j': data['j'],
            'i': data.get('i', {}),
            'goodness': goodness,
            'variety': data.get('variety', 0)
        }

    def get_verb_properties(self, verb: str) -> Optional[Dict]:
        """Get semantic properties for a verb."""
        self._load_vocabularies()

        if verb not in self._verb_operators:
            return None

        data = self._verb_operators[verb]
        return {
            'word': verb,
            'vector': data['vector'],
            'magnitude': data['magnitude']
        }
