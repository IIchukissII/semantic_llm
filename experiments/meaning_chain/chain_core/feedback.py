"""
Feedback Analyzer: Response -> Validation -> Regenerate?

Validates LLM responses against the meaning tree structure,
checking alignment and coherence.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Set, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent paths for imports
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
_SEMANTIC_LLM = _MEANING_CHAIN.parent.parent
sys.path.insert(0, str(_SEMANTIC_LLM))
sys.path.insert(0, str(_MEANING_CHAIN))

from core.data_loader import DataLoader
from models.types import MeaningTree, MeaningNode


@dataclass
class FeedbackConfig:
    """Configuration for feedback analysis."""
    min_concept_coverage: float = 0.3    # Min fraction of tree concepts in response
    min_alignment_score: float = 0.5     # Min alignment to accept
    max_regenerations: int = 2           # Max regeneration attempts
    goodness_weight: float = 0.4         # Weight for goodness alignment
    coverage_weight: float = 0.4         # Weight for concept coverage
    coherence_weight: float = 0.2        # Weight for verb coherence


@dataclass
class FeedbackResult:
    """Result of feedback analysis."""
    accepted: bool
    alignment_score: float
    concept_coverage: float
    goodness_alignment: float
    coherence_score: float
    concepts_found: List[str]
    concepts_missing: List[str]
    suggestions: List[str]

    def to_dict(self) -> Dict:
        return {
            'accepted': self.accepted,
            'alignment_score': self.alignment_score,
            'concept_coverage': self.concept_coverage,
            'goodness_alignment': self.goodness_alignment,
            'coherence_score': self.coherence_score,
            'concepts_found': self.concepts_found,
            'concepts_missing': self.concepts_missing,
            'suggestions': self.suggestions,
        }


class FeedbackAnalyzer:
    """
    Analyzes LLM responses against meaning trees.

    Checks:
    1. Concept coverage: Are tree concepts present in response?
    2. Goodness alignment: Does response match intended tone?
    3. Coherence: Are verb relationships honored?
    """

    def __init__(self, data_loader: Optional[DataLoader] = None,
                 config: Optional[FeedbackConfig] = None):
        self.loader = data_loader or DataLoader()
        self.config = config or FeedbackConfig()

        # Lazy-loaded data
        self._word_vectors = None
        self._j_good = None

    def _load_data(self):
        """Lazy load required data."""
        if self._word_vectors is not None:
            return

        self._word_vectors = self.loader.load_word_vectors()
        self._j_good = self.loader.get_j_good()

    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization for concept matching."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return set(text.split())

    def _compute_concept_coverage(self, response: str,
                                   tree: MeaningTree) -> Tuple[float, List[str], List[str]]:
        """
        Compute what fraction of tree concepts appear in response.

        Returns: (coverage_ratio, found_concepts, missing_concepts)
        """
        response_tokens = self._tokenize(response)
        tree_concepts = set(tree.all_words())

        found = []
        missing = []

        for concept in tree_concepts:
            # Check exact match and common variations
            variations = [concept, concept + 's', concept + 'ed', concept + 'ing']
            if any(v in response_tokens for v in variations):
                found.append(concept)
            else:
                missing.append(concept)

        coverage = len(found) / len(tree_concepts) if tree_concepts else 0.0
        return coverage, found, missing

    def _compute_goodness_alignment(self, response: str,
                                     tree: MeaningTree) -> float:
        """
        Check if response goodness matches tree's intended direction.

        Extracts concepts from response and compares average goodness
        to tree's average goodness.
        """
        self._load_data()

        response_tokens = self._tokenize(response)
        tree_avg_g = tree.summary()['avg_goodness']

        # Compute average goodness of response concepts
        response_goodnesses = []
        for token in response_tokens:
            if token in self._word_vectors:
                data = self._word_vectors[token]
                if data.get('j'):
                    j_vec = [data['j'].get(d, 0) for d in
                             ['beauty', 'life', 'sacred', 'good', 'love']]
                    g = sum(a * b for a, b in zip(j_vec, self._j_good))
                    response_goodnesses.append(g)

        if not response_goodnesses:
            return 0.5  # Neutral if no concepts found

        response_avg_g = np.mean(response_goodnesses)

        # Alignment: 1.0 if same sign and similar magnitude
        # Use correlation-like measure
        if tree_avg_g >= 0 and response_avg_g >= 0:
            alignment = min(1.0, response_avg_g / (tree_avg_g + 0.1))
        elif tree_avg_g < 0 and response_avg_g < 0:
            alignment = min(1.0, tree_avg_g / (response_avg_g - 0.1))
        else:
            # Opposite signs - misalignment
            alignment = 0.3

        return max(0.0, min(1.0, alignment))

    def _compute_coherence(self, response: str, tree: MeaningTree) -> float:
        """
        Check if response honors verb relationships.

        Looks for verb usage that matches tree transitions.
        """
        response_tokens = self._tokenize(response)
        transitions = tree.get_transitions()

        if not transitions:
            return 1.0  # No transitions to check

        coherent_count = 0
        for from_word, verb, to_word in transitions:
            # Check if verb appears near its expected concepts
            variations = [verb, verb + 's', verb + 'ed', verb + 'ing']
            verb_present = any(v in response_tokens for v in variations)

            from_present = from_word in response_tokens
            to_present = to_word in response_tokens

            # Coherent if verb appears when both concepts present
            # or if all three appear
            if verb_present and (from_present or to_present):
                coherent_count += 1
            elif from_present and to_present:
                coherent_count += 0.5  # Concepts present but verb missing

        return coherent_count / len(transitions)

    def analyze(self, response: str, tree: MeaningTree) -> FeedbackResult:
        """
        Analyze response against meaning tree.

        Args:
            response: LLM generated response
            tree: The meaning tree used for generation

        Returns:
            FeedbackResult with scores and recommendations
        """
        # Compute metrics
        coverage, found, missing = self._compute_concept_coverage(response, tree)
        goodness = self._compute_goodness_alignment(response, tree)
        coherence = self._compute_coherence(response, tree)

        # Weighted alignment score
        alignment = (
            self.config.coverage_weight * coverage +
            self.config.goodness_weight * goodness +
            self.config.coherence_weight * coherence
        )

        # Determine if accepted
        accepted = (
            alignment >= self.config.min_alignment_score and
            coverage >= self.config.min_concept_coverage
        )

        # Generate suggestions for improvement
        suggestions = []
        if coverage < self.config.min_concept_coverage:
            top_missing = missing[:3]
            suggestions.append(f"Include more concepts: {', '.join(top_missing)}")
        if goodness < 0.5:
            suggestions.append("Adjust tone toward more positive concepts")
        if coherence < 0.5:
            suggestions.append("Use transition verbs to connect concepts")

        return FeedbackResult(
            accepted=accepted,
            alignment_score=alignment,
            concept_coverage=coverage,
            goodness_alignment=goodness,
            coherence_score=coherence,
            concepts_found=found,
            concepts_missing=missing,
            suggestions=suggestions
        )

    def should_regenerate(self, result: FeedbackResult,
                          attempt: int) -> Tuple[bool, str]:
        """
        Decide if response should be regenerated.

        Args:
            result: Feedback analysis result
            attempt: Current attempt number (0-indexed)

        Returns:
            (should_regenerate, reason)
        """
        if result.accepted:
            return False, "Response accepted"

        if attempt >= self.config.max_regenerations:
            return False, f"Max regenerations ({self.config.max_regenerations}) reached"

        # Build reason
        reasons = []
        if result.concept_coverage < self.config.min_concept_coverage:
            reasons.append(f"low coverage ({result.concept_coverage:.0%})")
        if result.alignment_score < self.config.min_alignment_score:
            reasons.append(f"low alignment ({result.alignment_score:.0%})")

        return True, "Regenerating: " + ", ".join(reasons)

    def compare_responses(self, responses: List[str], tree: MeaningTree) -> Tuple[int, List[FeedbackResult]]:
        """
        Compare multiple candidate responses and pick the best.

        Instead of evaluating sequentially, this evaluates all candidates
        and returns the index of the best one based on composite score.

        Args:
            responses: List of candidate response strings
            tree: The meaning tree used for generation

        Returns:
            (best_index, all_results) - index of best response and all feedback results
        """
        if not responses:
            return 0, []

        # Evaluate each response separately
        results = [self.analyze(response, tree) for response in responses]

        # Compute composite score for comparison
        # Score = alignment * 0.4 + coverage * 0.3 + coherence * 0.2 + goodness * 0.1
        def composite_score(result: FeedbackResult) -> float:
            return (
                result.alignment_score * 0.4 +
                result.concept_coverage * 0.3 +
                result.coherence_score * 0.2 +
                result.goodness_alignment * 0.1
            )

        scores = [composite_score(r) for r in results]
        best_idx = int(np.argmax(scores))

        return best_idx, results
