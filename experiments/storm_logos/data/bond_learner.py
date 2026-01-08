"""Bond Learner: Runtime bond learning from conversations.

This module provides the main interface for learning new bonds during
runtime (conversations). It handles:

1. Bond extraction from text using spaCy
2. Coordinate computation/estimation
3. Storage in PostgreSQL (learned_bonds table)
4. Synchronization to Neo4j for trajectory navigation

Usage:
    from storm_logos.data.bond_learner import BondLearner

    learner = BondLearner()
    learner.connect()

    # Learn from a single text
    bonds = learner.learn_from_text("The dark forest held ancient secrets.")

    # Learn from conversation turn
    bonds = learner.learn_turn(text, conversation_id="conv_123")

    # Get learning statistics
    stats = learner.get_stats()

Learning Flow:
    Text → spaCy → Bonds → Coordinates → PostgreSQL → Neo4j
                                  ↓
                         estimate if unknown

Architecture:
    - PostgreSQL: Ground truth for learned bonds with coordinates
    - Neo4j: Trajectory graph for navigation and decay
    - Both are kept in sync by this module
"""

import re
import uuid
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .models import Bond, Trajectory
from .postgres import PostgresData, get_data
from .neo4j import Neo4jData, get_neo4j


@dataclass
class LearnedBond:
    """A bond learned from conversation with metadata."""
    adj: str
    noun: str
    A: float = 0.0
    S: float = 0.0
    tau: float = 2.5
    source: str = 'conversation'
    confidence: float = 0.5
    is_new: bool = False  # True if this was just created

    @property
    def text(self) -> str:
        return f"{self.adj} {self.noun}"

    def to_bond(self) -> Bond:
        """Convert to Bond dataclass."""
        return Bond(
            adj=self.adj,
            noun=self.noun,
            A=self.A,
            S=self.S,
            tau=self.tau,
        )


@dataclass
class LearningResult:
    """Result of learning from text."""
    bonds: List[LearnedBond] = field(default_factory=list)
    new_bonds: int = 0
    reinforced_bonds: int = 0
    trajectory_edges: int = 0
    conversation_id: str = ''

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Learned {len(self.bonds)} bonds: "
            f"{self.new_bonds} new, {self.reinforced_bonds} reinforced, "
            f"{self.trajectory_edges} edges"
        )


class BondLearner:
    """Main interface for learning bonds from conversations.

    Coordinates between spaCy extraction, PostgreSQL storage,
    and Neo4j trajectory updates.
    """

    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        """Initialize the learner.

        Args:
            spacy_model: spaCy model to use for extraction
        """
        self.spacy_model = spacy_model
        self.nlp = None
        self.postgres: Optional[PostgresData] = None
        self.neo4j: Optional[Neo4jData] = None
        self._connected = False

    def connect(self, init_tables: bool = True) -> bool:
        """Connect to databases and initialize.

        Args:
            init_tables: Create learning tables if they don't exist

        Returns:
            True if all connections successful
        """
        # Load spaCy model
        if SPACY_AVAILABLE and self.nlp is None:
            try:
                self.nlp = spacy.load(self.spacy_model)
            except OSError:
                print(f"spaCy model '{self.spacy_model}' not found. Downloading...")
                try:
                    spacy.cli.download(self.spacy_model)
                    self.nlp = spacy.load(self.spacy_model)
                except Exception as e:
                    print(f"Failed to download spaCy model: {e}")
                    return False

        # Connect to PostgreSQL
        self.postgres = get_data()
        if init_tables:
            self.postgres.init_learning_tables()

        # Connect to Neo4j
        self.neo4j = get_neo4j()
        neo4j_ok = self.neo4j.connect()

        self._connected = neo4j_ok
        return neo4j_ok

    # ========================================================================
    # BOND EXTRACTION
    # ========================================================================

    def extract_bonds(self, text: str) -> List[Tuple[str, str]]:
        """Extract adj-noun bonds from text using spaCy.

        Args:
            text: Input text

        Returns:
            List of (adjective, noun) tuples
        """
        if self.nlp is None:
            return self._extract_bonds_simple(text)

        doc = self.nlp(text)
        bonds = []

        for token in doc:
            # Look for adjectives modifying nouns
            if token.pos_ == 'ADJ' and token.dep_ == 'amod':
                head = token.head
                if head.pos_ == 'NOUN':
                    adj = token.lemma_.lower()
                    noun = head.lemma_.lower()
                    if len(adj) >= 2 and len(noun) >= 2:
                        bonds.append((adj, noun))

            # Also look for predicative adjectives
            elif token.pos_ == 'ADJ' and token.dep_ in ('acomp', 'attr'):
                for child in token.head.children:
                    if child.dep_ == 'nsubj' and child.pos_ == 'NOUN':
                        adj = token.lemma_.lower()
                        noun = child.lemma_.lower()
                        if len(adj) >= 2 and len(noun) >= 2:
                            bonds.append((adj, noun))
                        break

        return bonds

    def _extract_bonds_simple(self, text: str) -> List[Tuple[str, str]]:
        """Simple regex-based extraction fallback.

        Used when spaCy is not available.
        """
        # Simple pattern: adjective followed by noun
        pattern = r'\b([a-z]+)\s+([a-z]+(?:s|es)?)\b'

        # Very basic: assume any lowercase word pair could be adj+noun
        words = text.lower().split()
        bonds = []

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            # Clean punctuation
            w1 = re.sub(r'[^\w]', '', w1)
            w2 = re.sub(r'[^\w]', '', w2)
            if len(w1) >= 3 and len(w2) >= 3:
                if w1.isalpha() and w2.isalpha():
                    bonds.append((w1, w2))

        return bonds

    # ========================================================================
    # LEARNING
    # ========================================================================

    def learn_bond(self, adj: str, noun: str,
                   source: str = 'conversation',
                   confidence: float = 0.5) -> LearnedBond:
        """Learn a single bond.

        1. Compute/lookup coordinates
        2. Store in PostgreSQL
        3. Sync to Neo4j

        Args:
            adj: Adjective
            noun: Noun
            source: Source type
            confidence: Confidence in coordinates

        Returns:
            LearnedBond with coordinates
        """
        adj = adj.lower().strip()
        noun = noun.lower().strip()

        # Check if already exists in PostgreSQL
        existing = self.postgres.get_learned_bond(adj, noun) if self.postgres else None
        is_new = existing is None

        # Compute coordinates
        coords = self._get_coordinates(adj, noun)
        A, S, tau = coords

        # Store in PostgreSQL
        if self.postgres:
            bond = self.postgres.learn_bond(
                adj=adj, noun=noun,
                A=A, S=S, tau=tau,
                source=source,
                confidence=confidence
            )
            if bond:
                A, S, tau = bond.A, bond.S, bond.tau

        # Create learned bond result
        learned = LearnedBond(
            adj=adj, noun=noun,
            A=A, S=S, tau=tau,
            source=source,
            confidence=confidence,
            is_new=is_new,
        )

        # Sync to Neo4j
        if self.neo4j and self._connected:
            self.neo4j.learn_bond(learned.to_bond(), source=source)

        return learned

    def learn_from_text(self, text: str,
                        conversation_id: str = None,
                        source: str = 'conversation') -> LearningResult:
        """Learn all bonds from a text.

        Extracts bonds, computes coordinates, stores them,
        and creates trajectory edges if there are multiple bonds.

        Args:
            text: Input text
            conversation_id: Conversation identifier (auto-generated if None)
            source: Source type

        Returns:
            LearningResult with all learned bonds
        """
        if conversation_id is None:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        # Extract bonds
        bond_pairs = self.extract_bonds(text)

        result = LearningResult(conversation_id=conversation_id)
        bonds_for_trajectory = []

        for adj, noun in bond_pairs:
            learned = self.learn_bond(adj, noun, source=source)
            result.bonds.append(learned)
            bonds_for_trajectory.append(learned.to_bond())

            if learned.is_new:
                result.new_bonds += 1
            else:
                result.reinforced_bonds += 1

        # Create trajectory edges in Neo4j
        if self.neo4j and self._connected and len(bonds_for_trajectory) >= 2:
            result.trajectory_edges = self.neo4j.learn_trajectory(
                bonds_for_trajectory,
                conversation_id=conversation_id,
                source_type=source
            )

        return result

    def learn_turn(self, text: str,
                   conversation_id: str,
                   previous_bonds: List[Bond] = None,
                   source: str = 'conversation') -> LearningResult:
        """Learn from a conversation turn, linking to previous context.

        Like learn_from_text but also creates edges from previous
        turn's last bond to this turn's first bond.

        Args:
            text: Text of this turn
            conversation_id: Conversation identifier
            previous_bonds: Bonds from previous turn (to link)
            source: Source type

        Returns:
            LearningResult
        """
        result = self.learn_from_text(text, conversation_id, source)

        # Link to previous turn
        if (previous_bonds and result.bonds and
            self.neo4j and self._connected):
            last_prev = previous_bonds[-1] if isinstance(previous_bonds[-1], Bond) else previous_bonds[-1].to_bond()
            first_curr = result.bonds[0].to_bond()

            self.neo4j.learn_transition(
                last_prev, first_curr,
                conversation_id=conversation_id,
                source_type=source
            )
            result.trajectory_edges += 1

        return result

    # ========================================================================
    # COORDINATE COMPUTATION
    # ========================================================================

    def _get_coordinates(self, adj: str, noun: str) -> Tuple[float, float, float]:
        """Get or compute coordinates for a bond.

        Priority:
        1. Look up both words in corpus
        2. Look up in learned words
        3. Estimate from word structure

        Args:
            adj: Adjective
            noun: Noun

        Returns:
            (A, S, tau) tuple
        """
        if self.postgres:
            return self.postgres._compute_bond_coordinates(adj, noun)

        # Fallback without postgres
        return self._estimate_coordinates(adj, noun)

    def _estimate_coordinates(self, adj: str, noun: str) -> Tuple[float, float, float]:
        """Estimate coordinates using word heuristics.

        Simple heuristics based on word patterns.
        Can be extended with embeddings or LLM.
        """
        A, S, tau = 0.0, 0.0, 2.5

        # Analyze adjective
        if adj.startswith(('un', 'dis', 'non', 'anti')):
            A -= 0.3
        if adj.endswith(('ful', 'ous', 'ive')):
            A += 0.2
        if any(w in adj for w in ['dark', 'evil', 'bad', 'terrible']):
            A -= 0.4
        if any(w in adj for w in ['good', 'beautiful', 'bright', 'wonderful']):
            A += 0.4

        # Analyze noun for abstraction
        if noun.endswith(('ness', 'ity', 'tion', 'ism', 'ment')):
            tau += 0.5
        if any(w in noun for w in ['god', 'soul', 'spirit', 'heaven']):
            S += 0.5
            tau += 0.3

        # Clamp
        A = max(-1.0, min(1.0, A))
        S = max(-1.0, min(1.0, S))
        tau = max(0.5, min(4.5, tau))

        return (A, S, tau)

    # ========================================================================
    # RETRIEVAL
    # ========================================================================

    def get_learned_bonds(self, limit: int = 100,
                          min_use_count: int = 1) -> List[LearnedBond]:
        """Get all learned bonds from PostgreSQL.

        Args:
            limit: Maximum bonds to return
            min_use_count: Minimum use count filter

        Returns:
            List of learned bonds
        """
        if not self.postgres:
            return []

        bonds = self.postgres.get_all_learned_bonds(limit, min_use_count)
        return [
            LearnedBond(
                adj=b.adj,
                noun=b.noun,
                A=b.A,
                S=b.S,
                tau=b.tau,
                source='learned',
            )
            for b in bonds
        ]

    def get_conversation_trajectory(self, conversation_id: str) -> Trajectory:
        """Get the trajectory for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Trajectory of the conversation
        """
        if not self.neo4j or not self._connected:
            return Trajectory()

        return self.neo4j.get_conversation_trajectory(conversation_id)

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_stats(self) -> Dict:
        """Get combined learning statistics.

        Returns:
            Dictionary with stats from both PostgreSQL and Neo4j
        """
        stats = {
            'connected': self._connected,
            'spacy_available': self.nlp is not None,
        }

        if self.postgres:
            pg_stats = self.postgres.get_learning_stats()
            stats['postgresql'] = pg_stats

        if self.neo4j and self._connected:
            neo_stats = self.neo4j.get_learning_stats()
            stats['neo4j'] = neo_stats

        return stats


# ============================================================================
# SINGLETON
# ============================================================================

_learner_instance: Optional[BondLearner] = None


def get_learner() -> BondLearner:
    """Get the singleton BondLearner instance."""
    global _learner_instance
    if _learner_instance is None:
        _learner_instance = BondLearner()
    return _learner_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def learn_from_text(text: str, conversation_id: str = None) -> LearningResult:
    """Learn bonds from text (convenience function).

    Args:
        text: Input text
        conversation_id: Optional conversation ID

    Returns:
        LearningResult
    """
    learner = get_learner()
    if not learner._connected:
        learner.connect()
    return learner.learn_from_text(text, conversation_id)


def learn_bond(adj: str, noun: str) -> LearnedBond:
    """Learn a single bond (convenience function).

    Args:
        adj: Adjective
        noun: Noun

    Returns:
        LearnedBond
    """
    learner = get_learner()
    if not learner._connected:
        learner.connect()
    return learner.learn_bond(adj, noun)


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Bond Learner Test")
    print("=" * 50)

    learner = BondLearner()
    print(f"Connecting...")
    if learner.connect():
        print("Connected!")

        # Test text
        test_text = """
        The dark forest held ancient secrets.
        Beautiful flowers grew in the sunny garden.
        The mysterious stranger wore a black cloak.
        """

        print(f"\nTest text:\n{test_text.strip()}")
        print("\nExtracting bonds...")

        bonds = learner.extract_bonds(test_text)
        print(f"Found {len(bonds)} bonds:")
        for adj, noun in bonds:
            print(f"  - {adj} {noun}")

        print("\nLearning bonds...")
        result = learner.learn_from_text(test_text)
        print(f"\n{result.summary()}")

        print("\nLearned bonds:")
        for b in result.bonds:
            print(f"  {b.text}: A={b.A:+.2f}, S={b.S:+.2f}, tau={b.tau:.2f} {'(new)' if b.is_new else ''}")

        print("\nStats:")
        stats = learner.get_stats()
        print(f"  PostgreSQL: {stats.get('postgresql', {})}")
        print(f"  Neo4j: {stats.get('neo4j', {})}")
    else:
        print("Failed to connect!")
        sys.exit(1)
