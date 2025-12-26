"""
Conversation Learner - Learn patterns and concepts from conversations.

Extracts:
1. SVO patterns → VIA relationships
2. Adj-Noun pairs → Concept learning (new words, parameter updates)

Weight hierarchy (matching experience_knowledge):
    Books:        1.0  (established knowledge)
    Articles:     0.8  (curated content)
    Conversation: 0.2  (needs reinforcement)
    Context:      0.1  (weakest)

Learning:
- New nouns are learned from their adjective distributions
- τ = f(entropy) of adjective distribution
- Existing concepts update their parameters with new observations

"Each conversation walks new paths through meaning space"
"""

import re
import spacy
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import sys
_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph
from graph.learning import Neo4jLearningStore


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = ""


@dataclass
class LearnedPattern:
    """An SVO pattern learned from conversation."""
    subject: str
    verb: str
    object: str
    source: str        # "user" or "assistant"
    reinforced: bool = False  # True if pattern already existed


@dataclass
class LearnedConcept:
    """A concept learned from conversation."""
    noun: str
    adjectives: List[str]
    is_new: bool = True


class ConversationLearner:
    """
    Learn SVO patterns and concepts from conversations.

    Two learning modes:
    1. SVO patterns → VIA relationships (weight 0.2)
    2. Adj-Noun pairs → Concept learning (τ from entropy)

    Extracts grammatical patterns and stores them in MeaningGraph
    with conversation-level weights (0.2 - needs reinforcement).
    """

    # Weight for conversation-learned patterns
    CONVERSATION_WEIGHT = 0.2

    def __init__(self, graph: MeaningGraph = None,
                 model: str = "en_core_web_sm",
                 enable_learning: bool = True,
                 adj_vectors: Dict[str, np.ndarray] = None):
        """
        Initialize ConversationLearner.

        Args:
            graph: MeaningGraph instance
            model: spaCy model name
            enable_learning: If True, learn new concepts from adj-noun pairs
            adj_vectors: {adjective: 5D j-vector} for computing centroids
        """
        self.graph = graph or MeaningGraph()
        self.enable_learning = enable_learning
        self.adj_vectors = adj_vectors or {}

        # Learning store
        self._learning_store = None

        # Load spaCy
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)

        # Known concepts for filtering
        self.known_concepts: Set[str] = set()
        self._load_known_concepts()

        # Session tracking
        self.session_patterns: List[LearnedPattern] = []
        self.session_concepts: List[LearnedConcept] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup learning
        if self.enable_learning:
            self._setup_learning()

    def _setup_learning(self):
        """Initialize learning components."""
        if self.graph.is_connected():
            self._learning_store = Neo4jLearningStore(self.graph.driver)
            self._learning_store.setup_schema()

    def load_adj_vectors(self, data_loader):
        """Load adjective vectors from DataLoader."""
        vectors = data_loader.load_word_vectors()
        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

        for word, v in vectors.items():
            if v.get('j'):
                self.adj_vectors[word] = np.array([v['j'].get(d, 0) for d in j_dims])

    def _load_known_concepts(self):
        """Load known concepts from graph."""
        if not self.graph.is_connected():
            return

        with self.graph.driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN c.word as word")
            self.known_concepts = {r["word"] for r in result}

    def extract_svo(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract SVO patterns from text.

        Returns: [(subject, verb, object), ...]
        """
        doc = self.nlp(text)
        patterns = []

        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    verb = token.lemma_.lower()

                    # Skip auxiliaries
                    if verb in {'be', 'have', 'do', 'will', 'would', 'could', 'should'}:
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

                    # Also check prepositional objects
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
                                patterns.append((subj, verb, obj))

        return patterns

    def extract_adj_noun(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract Adjective-Noun pairs from text for learning.

        Returns: [(adjective, noun), ...]
        """
        doc = self.nlp(text)
        pairs = []

        for token in doc:
            if token.pos_ == "NOUN":
                noun = token.lemma_.lower()
                if len(noun) < 3:
                    continue

                # Find adjective modifiers
                for child in token.children:
                    if child.dep_ == "amod" and child.pos_ == "ADJ":
                        adj = child.lemma_.lower()
                        if len(adj) >= 3:
                            pairs.append((adj, noun))

        return pairs

    def _aggregate_adj_noun(self, pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
        """Aggregate adj-noun pairs into distributions."""
        distributions = defaultdict(lambda: defaultdict(int))
        for adj, noun in pairs:
            distributions[noun][adj] += 1
        return {noun: dict(adjs) for noun, adjs in distributions.items()}

    def _learn_concepts(self, distributions: Dict[str, Dict[str, int]],
                        source: str) -> List[LearnedConcept]:
        """Learn concepts from adjective distributions."""
        if not self._learning_store:
            return []

        learned = []
        for noun, adj_counts in distributions.items():
            is_new = noun not in self.known_concepts

            # Store observations and update
            result = self._learning_store.learn_concept(
                noun, adj_counts, source, self.adj_vectors
            )

            if result:
                concept = LearnedConcept(
                    noun=noun,
                    adjectives=list(adj_counts.keys()),
                    is_new=is_new
                )
                learned.append(concept)
                self.session_concepts.append(concept)

                if is_new:
                    self.known_concepts.add(noun)

        return learned

    def learn_from_turn(self, turn: ConversationTurn) -> List[LearnedPattern]:
        """
        Learn SVO patterns from a conversation turn.

        Args:
            turn: The conversation turn to learn from

        Returns:
            List of learned patterns
        """
        patterns = self.extract_svo(turn.content)
        learned = []

        for subj, verb, obj in patterns:
            # Check if pattern already exists
            existing = self._check_existing(subj, verb, obj)

            if existing:
                # Reinforce existing pattern
                self._reinforce_pattern(subj, verb, obj)
                pattern = LearnedPattern(subj, verb, obj, turn.role, reinforced=True)
            else:
                # Create new pattern
                self._create_pattern(subj, verb, obj, turn.role)
                pattern = LearnedPattern(subj, verb, obj, turn.role, reinforced=False)

            learned.append(pattern)
            self.session_patterns.append(pattern)

        return learned

    def learn_from_exchange(self, user_text: str, assistant_text: str) -> Dict:
        """
        Learn from a complete exchange (user + assistant).

        Learns both:
        1. SVO patterns → VIA relationships
        2. Adj-Noun pairs → Concept learning (if enabled)

        Args:
            user_text: User's message
            assistant_text: Assistant's response

        Returns:
            Statistics about learned patterns and concepts
        """
        # Learn SVO patterns
        user_patterns = self.learn_from_turn(
            ConversationTurn("user", user_text)
        )
        assistant_patterns = self.learn_from_turn(
            ConversationTurn("assistant", assistant_text)
        )

        all_patterns = user_patterns + assistant_patterns

        # Learn concepts from adj-noun pairs
        concepts_learned = []
        if self.enable_learning and self._learning_store:
            # Combine texts
            combined_text = f"{user_text} {assistant_text}"

            # Extract adj-noun pairs
            adj_noun_pairs = self.extract_adj_noun(combined_text)

            if adj_noun_pairs:
                # Aggregate and learn
                distributions = self._aggregate_adj_noun(adj_noun_pairs)
                concepts_learned = self._learn_concepts(
                    distributions, source=f"conversation:{self.session_id}"
                )

        return {
            "user_patterns": len(user_patterns),
            "assistant_patterns": len(assistant_patterns),
            "total_patterns": len(all_patterns),
            "patterns_reinforced": sum(1 for p in all_patterns if p.reinforced),
            "patterns_new": sum(1 for p in all_patterns if not p.reinforced),
            "concepts_learned": len(concepts_learned),
            "concepts_new": sum(1 for c in concepts_learned if c.is_new),
            "concepts_updated": sum(1 for c in concepts_learned if not c.is_new)
        }

    def _check_existing(self, subj: str, verb: str, obj: str) -> bool:
        """Check if pattern already exists."""
        if not self.graph.is_connected():
            return False

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (s:Concept {word: $subj})
                      -[r:VIA {verb: $verb}]->
                      (o:Concept {word: $obj})
                RETURN count(r) > 0 as exists
            """, subj=subj, verb=verb, obj=obj)
            record = result.single()
            return record["exists"] if record else False

    def _reinforce_pattern(self, subj: str, verb: str, obj: str):
        """Reinforce an existing pattern (increase weight)."""
        if not self.graph.is_connected():
            return

        with self.graph.driver.session() as session:
            # Increase weight slightly (learning)
            session.run("""
                MATCH (s:Concept {word: $subj})
                      -[r:VIA {verb: $verb}]->
                      (o:Concept {word: $obj})
                SET r.weight = CASE
                    WHEN r.weight < 1.0 THEN r.weight + 0.05
                    ELSE r.weight
                END,
                r.count = coalesce(r.count, 1) + 1,
                r.last_reinforced = datetime()
            """, subj=subj, verb=verb, obj=obj)

    def _create_pattern(self, subj: str, verb: str, obj: str, source: str):
        """Create a new pattern with conversation weight."""
        if not self.graph.is_connected():
            return

        with self.graph.driver.session() as session:
            session.run("""
                MATCH (s:Concept {word: $subj})
                MATCH (o:Concept {word: $obj})
                MERGE (s)-[r:VIA {verb: $verb}]->(o)
                ON CREATE SET
                    r.weight = $weight,
                    r.count = 1,
                    r.source = $source,
                    r.created_at = datetime()
            """, subj=subj, verb=verb, obj=obj,
                 weight=self.CONVERSATION_WEIGHT,
                 source=f"conversation:{source}")

    def get_session_stats(self) -> Dict:
        """Get statistics for current session."""
        # Pattern stats
        pattern_counts = defaultdict(int)
        for p in self.session_patterns:
            key = (p.subject, p.verb, p.object)
            pattern_counts[key] += 1

        # Concept stats
        concept_counts = defaultdict(list)
        for c in self.session_concepts:
            concept_counts[c.noun].extend(c.adjectives)

        return {
            "session_id": self.session_id,
            # Patterns
            "total_patterns": len(self.session_patterns),
            "unique_patterns": len(pattern_counts),
            "patterns_reinforced": sum(1 for p in self.session_patterns if p.reinforced),
            "patterns_new": sum(1 for p in self.session_patterns if not p.reinforced),
            "top_patterns": sorted(pattern_counts.items(), key=lambda x: -x[1])[:5],
            # Concepts
            "total_concepts": len(self.session_concepts),
            "unique_concepts": len(concept_counts),
            "concepts_new": sum(1 for c in self.session_concepts if c.is_new),
            "concepts_updated": sum(1 for c in self.session_concepts if not c.is_new),
            "top_concepts": sorted(
                [(noun, len(adjs)) for noun, adjs in concept_counts.items()],
                key=lambda x: -x[1]
            )[:5]
        }

    def close(self):
        """Close connections."""
        if self.graph:
            self.graph.close()


def demo():
    """Demonstrate conversation learning."""
    print("=" * 60)
    print("Conversation Learning Demo")
    print("=" * 60)

    learner = ConversationLearner()

    if not learner.graph.is_connected():
        print("Graph not connected")
        return

    # Simulate a conversation
    exchanges = [
        ("What is the meaning of life?",
         "Life finds meaning through connection and purpose."),
        ("How can I understand my dreams?",
         "Dreams reveal hidden aspects of the soul."),
        ("Tell me about the shadow self",
         "The shadow contains parts we reject but must integrate.")
    ]

    for user_text, assistant_text in exchanges:
        print(f"\nUser: {user_text}")
        print(f"Assistant: {assistant_text[:50]}...")

        stats = learner.learn_from_exchange(user_text, assistant_text)
        print(f"Learned: {stats['total']} patterns ({stats['new']} new, {stats['reinforced']} reinforced)")

    print("\n" + "=" * 60)
    print("Session Statistics")
    print("=" * 60)
    session_stats = learner.get_session_stats()
    print(f"Total patterns: {session_stats['total_patterns']}")
    print(f"Unique: {session_stats['unique_patterns']}")
    print(f"New: {session_stats['new']}")
    print(f"Reinforced: {session_stats['reinforced']}")

    learner.close()


if __name__ == "__main__":
    demo()
