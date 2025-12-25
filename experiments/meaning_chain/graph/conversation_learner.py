"""
Conversation Learner - Learn SVO patterns from conversations.

Extracts Subject-Verb-Object patterns from conversation exchanges
and stores them in MeaningGraph with conversation-level weights.

Weight hierarchy (matching experience_knowledge):
    Books:        1.0  (established knowledge)
    Articles:     0.8  (curated content)
    Conversation: 0.2  (needs reinforcement)
    Context:      0.1  (weakest)

"Each conversation walks new paths through meaning space"
"""

import re
import spacy
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


class ConversationLearner:
    """
    Learn SVO patterns from conversations.

    Extracts grammatical patterns and stores them in MeaningGraph
    with conversation-level weights (0.2 - needs reinforcement).
    """

    # Weight for conversation-learned patterns
    CONVERSATION_WEIGHT = 0.2

    def __init__(self, graph: MeaningGraph = None, model: str = "en_core_web_sm"):
        self.graph = graph or MeaningGraph()

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
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        Args:
            user_text: User's message
            assistant_text: Assistant's response

        Returns:
            Statistics about learned patterns
        """
        user_patterns = self.learn_from_turn(
            ConversationTurn("user", user_text)
        )
        assistant_patterns = self.learn_from_turn(
            ConversationTurn("assistant", assistant_text)
        )

        return {
            "user_patterns": len(user_patterns),
            "assistant_patterns": len(assistant_patterns),
            "total": len(user_patterns) + len(assistant_patterns),
            "reinforced": sum(1 for p in user_patterns + assistant_patterns if p.reinforced),
            "new": sum(1 for p in user_patterns + assistant_patterns if not p.reinforced)
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
        if not self.session_patterns:
            return {"patterns": 0}

        pattern_counts = defaultdict(int)
        for p in self.session_patterns:
            key = (p.subject, p.verb, p.object)
            pattern_counts[key] += 1

        return {
            "session_id": self.session_id,
            "total_patterns": len(self.session_patterns),
            "unique_patterns": len(pattern_counts),
            "reinforced": sum(1 for p in self.session_patterns if p.reinforced),
            "new": sum(1 for p in self.session_patterns if not p.reinforced),
            "top_patterns": sorted(pattern_counts.items(), key=lambda x: -x[1])[:5]
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
