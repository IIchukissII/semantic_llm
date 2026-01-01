"""
Intent Collapse: Verbs as Quantum Operators

Theory:
    "Intent collapses meaning like observation collapses wavefunction"

    When a user says "help me understand my dream":
    - "understand" and "help" are OPERATORS, not decorations
    - They collapse the superposition of possible meanings
    - Navigation follows paths aligned with these operators

Implementation:
    1. Extract intent verbs from user query
    2. Build intent space: concepts those verbs typically operate on
    3. During navigation, prioritize transitions that:
       a) Use intent verbs explicitly (verb on edge matches)
       b) Lead to intent-aligned targets (what those verbs act upon)
    4. Track collapse statistics for validation

This is the missing piece that makes the theory actually work.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import sys

_THIS_FILE = Path(__file__).resolve()
_MEANING_CHAIN = _THIS_FILE.parent.parent
sys.path.insert(0, str(_MEANING_CHAIN))

from graph.meaning_graph import MeaningGraph


@dataclass
class IntentOperator:
    """A verb acting as a semantic operator."""
    verb: str
    j_vector: np.ndarray  # 5D direction this verb pushes toward
    magnitude: float      # Operator strength
    targets: Set[str]     # Concepts this verb typically operates on

    @property
    def is_loaded(self) -> bool:
        """Check if operator has valid j-vector."""
        return np.linalg.norm(self.j_vector) > 1e-6


@dataclass
class CollapseState:
    """Track how intent collapsed navigation choices."""
    word: str
    tau: float
    affirmation: float  # A: Affirmation score (formerly g)
    j: np.ndarray

    # Collapse tracking
    collapsed_by_verb: bool = False    # True if verb matched intent
    collapsed_by_target: bool = False  # True if target in intent space
    collapse_verb: Optional[str] = None  # Which verb caused collapse
    intent_score: float = 0.0          # How aligned with intent [0, 1]

    @property
    def A(self) -> float:
        """Alias for affirmation."""
        return self.affirmation

    @property
    def g(self) -> float:
        """Legacy alias (g ≈ A)."""
        return self.affirmation


@dataclass
class CollapseResult:
    """Result of intent-driven navigation."""
    states: List[CollapseState]

    # Collapse statistics
    total_transitions: int = 0
    verb_collapses: int = 0      # Transitions where verb matched
    target_collapses: int = 0    # Transitions to intent targets
    random_walks: int = 0        # Transitions without intent match

    @property
    def collapse_ratio(self) -> float:
        """Fraction of transitions driven by intent (not random)."""
        if self.total_transitions == 0:
            return 0.0
        return (self.verb_collapses + self.target_collapses) / self.total_transitions

    @property
    def intent_words(self) -> Set[str]:
        """Words reached via intent collapse."""
        return {s.word for s in self.states if s.collapsed_by_verb or s.collapsed_by_target}


class IntentCollapse:
    """
    Intent-driven navigation through semantic space.

    Unlike random/Boltzmann walks that explore blindly,
    IntentCollapse uses verbs as operators that "collapse"
    the navigation to intent-relevant paths.

    PHASE SHIFT (Pirate Insight):
        Raw j-vectors are biased (all ~[-0.82, -0.97, ...]).
        We "shift the phase" by centering: j_centered = j - global_mean.
        This makes opposite verbs have negative similarity (rise/fall: -0.25).

    Usage:
        collapse = IntentCollapse(graph)
        collapse.set_intent(['understand', 'help'])  # User's verbs

        # Navigate with intent filtering
        result = collapse.navigate(
            seeds=['dream', 'meaning'],
            n_walks=5,
            steps_per_walk=8
        )

        # Check how much intent drove navigation
        print(f"Collapse ratio: {result.collapse_ratio:.0%}")
    """

    def __init__(self, graph: MeaningGraph = None, fallback_ratio: float = 0.3):
        """
        Args:
            graph: MeaningGraph with VerbOperator and OPERATES_ON data
            fallback_ratio: Fraction of transitions allowed without intent match
                           (prevents getting stuck when intent space is sparse)
        """
        self.graph = graph or MeaningGraph()
        self.fallback_ratio = fallback_ratio

        # Intent state
        self.intent_verbs: Set[str] = set()
        self.intent_operators: Dict[str, IntentOperator] = {}
        self.intent_targets: Set[str] = set()
        self.intent_j: Optional[np.ndarray] = None  # Combined intent direction

        self._j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

        # Phase shift: global mean for centering j-vectors
        self._global_j_mean: Optional[np.ndarray] = None
        self._compute_global_mean()

    def _compute_global_mean(self):
        """Compute global j-vector mean for phase shifting."""
        if not self.graph.driver:
            self._global_j_mean = np.zeros(5)
            return

        with self.graph.driver.session() as session:
            result = session.run("""
                MATCH (v:VerbOperator)
                WHERE v.j IS NOT NULL
                RETURN v.j as j
            """)

            all_j = []
            for record in result:
                j = record['j']
                if j and len(j) >= 5:
                    all_j.append(np.array(j[:5]))

            if all_j:
                self._global_j_mean = np.mean(all_j, axis=0)
            else:
                self._global_j_mean = np.zeros(5)

    def _center_j(self, j: np.ndarray) -> np.ndarray:
        """Center j-vector by subtracting global mean (phase shift)."""
        if self._global_j_mean is None:
            return j
        return j - self._global_j_mean

    def set_intent(self, verbs: List[str]) -> Dict:
        """
        Set the intent operators from user's verbs.

        This is the key step - it builds the "intent space" that will
        collapse navigation to relevant paths.

        Args:
            verbs: Verbs from user query (e.g., ['understand', 'help'])

        Returns:
            Stats about loaded intent: {operators, targets, intent_j}
        """
        self.intent_verbs = set(verbs)
        self.intent_operators = {}
        self.intent_targets = set()

        # Load operators with their j-vectors
        for verb in verbs:
            operator = self._load_operator(verb)
            if operator:
                self.intent_operators[verb] = operator
                self.intent_targets.update(operator.targets)

        # Also get targets via OPERATES_ON relationships
        graph_targets = self.graph.get_verb_targets(verbs)
        self.intent_targets.update(graph_targets)

        # Compute combined intent direction (weighted by magnitude)
        self._compute_intent_direction()

        return {
            'operators': len(self.intent_operators),
            'targets': len(self.intent_targets),
            'intent_j': self.intent_j.tolist() if self.intent_j is not None else None
        }

    def _load_operator(self, verb: str) -> Optional[IntentOperator]:
        """Load a VerbOperator from graph."""
        if not self.graph.driver:
            return None

        with self.graph.driver.session() as session:
            # Get operator properties
            result = session.run("""
                MATCH (v:VerbOperator {verb: $verb})
                OPTIONAL MATCH (v)-[:OPERATES_ON]->(c:Concept)
                RETURN v.j as j, v.magnitude as magnitude,
                       collect(c.word) as targets
            """, verb=verb)

            record = result.single()
            if not record or record['j'] is None:
                # Operator not in graph - create minimal operator
                return IntentOperator(
                    verb=verb,
                    j_vector=np.zeros(5),
                    magnitude=1.0,
                    targets=set()
                )

            j_vec = np.array(record['j'][:5] if record['j'] else [0]*5)
            targets = set(record['targets']) if record['targets'] else set()

            return IntentOperator(
                verb=verb,
                j_vector=j_vec,
                magnitude=record['magnitude'] or 1.0,
                targets=targets
            )

    def _compute_intent_direction(self):
        """
        Compute combined intent direction from operators.

        Uses PHASE-SHIFTED (centered) j-vectors so that:
        - Opposite verbs have negative contribution
        - Similar verbs reinforce each other
        - The intent direction is meaningful, not biased
        """
        if not self.intent_operators:
            self.intent_j = None
            return

        # Weighted average of CENTERED operator j-vectors
        total_weight = 0.0
        combined_j = np.zeros(5)

        for op in self.intent_operators.values():
            if op.is_loaded:
                weight = op.magnitude
                # PHASE SHIFT: Use centered j-vector
                centered_j = self._center_j(op.j_vector)
                combined_j += weight * centered_j
                total_weight += weight

        if total_weight > 0:
            combined_j /= total_weight
            # Normalize
            norm = np.linalg.norm(combined_j)
            if norm > 1e-6:
                self.intent_j = combined_j / norm
            else:
                self.intent_j = None
        else:
            self.intent_j = None

    def navigate(self, seeds: List[str],
                 n_walks: int = 5,
                 steps_per_walk: int = 8,
                 temperature: float = 0.82) -> CollapseResult:
        """
        Navigate semantic space with intent collapse.

        At each step:
        1. Try intent-driven transition (verb match or target match)
        2. If no intent match and within fallback budget, allow random
        3. Track all collapses for validation

        Args:
            seeds: Starting concepts
            n_walks: Parallel walks per seed
            steps_per_walk: Steps per walk
            temperature: Boltzmann temperature for random fallback

        Returns:
            CollapseResult with states and collapse statistics
        """
        all_states = []
        total_transitions = 0
        verb_collapses = 0
        target_collapses = 0
        random_walks = 0

        for seed in seeds:
            concept = self.graph.get_concept(seed)
            if not concept:
                continue

            for _ in range(n_walks):
                current = seed
                current_tau = concept.get('tau', 2.0)
                walk_random_count = 0
                max_random = int(steps_per_walk * self.fallback_ratio)

                for step in range(steps_per_walk):
                    # Try intent-driven transition first
                    transition = self._intent_transition(current)

                    if transition is None and walk_random_count < max_random:
                        # Fallback to random/Boltzmann transition
                        transition = self._random_transition(current, current_tau, temperature)
                        if transition:
                            transition['collapsed_by_verb'] = False
                            transition['collapsed_by_target'] = False
                            transition['collapse_verb'] = None
                            random_walks += 1
                            walk_random_count += 1

                    if transition is None:
                        break

                    # Record state
                    state = CollapseState(
                        word=transition['word'],
                        tau=transition['tau'],
                        affirmation=transition.get('g', 0.0),  # g ≈ A
                        j=np.array(transition['j']) if transition['j'] else np.zeros(5),
                        collapsed_by_verb=transition.get('collapsed_by_verb', False),
                        collapsed_by_target=transition.get('collapsed_by_target', False),
                        collapse_verb=transition.get('collapse_verb'),
                        intent_score=transition.get('intent_score', 0.0)
                    )
                    all_states.append(state)

                    # Update stats
                    total_transitions += 1
                    if state.collapsed_by_verb:
                        verb_collapses += 1
                    elif state.collapsed_by_target:
                        target_collapses += 1

                    current = transition['word']
                    current_tau = transition['tau']

        return CollapseResult(
            states=all_states,
            total_transitions=total_transitions,
            verb_collapses=verb_collapses,
            target_collapses=target_collapses,
            random_walks=random_walks
        )

    def _intent_transition(self, word: str) -> Optional[Dict]:
        """
        Get an intent-driven transition from word.

        Prioritizes:
        1. Edges where verb matches an intent verb
        2. Edges leading to intent targets

        Returns None if no intent-aligned transition found.
        """
        if not self.graph.driver:
            return None

        # Use the graph's intent query (now actually used!)
        transitions = self.graph.get_intent_transitions(
            word,
            self.intent_verbs,
            self.intent_targets,
            limit=10
        )

        if not transitions:
            return None

        # Transitions are (verb, target, score)
        # Pick best by intent score
        best = transitions[0]
        verb, target, score = best

        # Get full concept properties
        concept = self.graph.get_concept(target)
        if not concept:
            return None

        # Determine collapse type
        collapsed_by_verb = verb in self.intent_verbs
        collapsed_by_target = target in self.intent_targets

        return {
            'word': target,
            'tau': concept.get('tau', 2.0),
            'g': concept.get('g', 0.0),
            'j': concept.get('j'),
            'collapsed_by_verb': collapsed_by_verb,
            'collapsed_by_target': collapsed_by_target,
            'collapse_verb': verb if collapsed_by_verb else None,
            'intent_score': score
        }

    def _random_transition(self, word: str, current_tau: float,
                           temperature: float) -> Optional[Dict]:
        """
        Fallback: random Boltzmann-weighted transition.

        Used when intent space is sparse to prevent getting stuck.
        """
        transitions = self.graph.get_all_transitions(word, limit=20)
        if not transitions:
            return None

        # Boltzmann weighting by tau proximity
        words = []
        weights = []

        for verb, target, weight in transitions:
            concept = self.graph.get_concept(target)
            if not concept:
                continue

            target_tau = concept.get('tau', 2.0)
            delta_tau = abs(target_tau - current_tau)
            boltzmann_w = np.exp(-delta_tau / temperature) * weight

            words.append((target, concept))
            weights.append(boltzmann_w)

        if not words:
            return None

        # Sample
        weights = np.array(weights)
        probs = weights / np.sum(weights)
        idx = np.random.choice(len(words), p=probs)
        target, concept = words[idx]

        return {
            'word': target,
            'tau': concept.get('tau', 2.0),
            'g': concept.get('g', 0.0),
            'j': concept.get('j')
        }

    def compute_j_alignment(self, j: np.ndarray) -> float:
        """
        Compute how aligned a j-vector is with intent direction.

        Uses PHASE-SHIFTED comparison so that:
        - Concepts aligned with intent verbs have positive score
        - Concepts opposite to intent have negative score
        """
        if self.intent_j is None or j is None:
            return 0.0

        j = np.array(j) if not isinstance(j, np.ndarray) else j

        # PHASE SHIFT: Center the concept's j-vector too
        centered_j = self._center_j(j)

        norm = np.linalg.norm(centered_j)
        if norm < 1e-6:
            return 0.0

        j_norm = centered_j / norm
        return float(np.dot(j_norm, self.intent_j))

    def close(self):
        """Close graph connection."""
        if self.graph:
            self.graph.close()


def demo():
    """Demonstrate intent collapse."""
    print("=" * 70)
    print("INTENT COLLAPSE: Verbs as Quantum Operators")
    print("=" * 70)

    collapse = IntentCollapse()

    if not collapse.graph.is_connected():
        print("\nNot connected to Neo4j. Start with:")
        print("  cd config && docker-compose up -d")
        return

    # Test query: "help me understand my dream"
    print("\nTest query: 'help me understand my dream'")
    print("Seeds: ['dream']")
    print("Intent verbs: ['understand', 'help']")
    print()

    # Set intent
    intent_stats = collapse.set_intent(['understand', 'help'])
    print(f"Intent operators loaded: {intent_stats['operators']}")
    print(f"Intent targets: {intent_stats['targets']}")
    if intent_stats['intent_j']:
        j = intent_stats['intent_j']
        dims = ['beauty', 'life', 'sacred', 'good', 'love']
        print(f"Intent direction: {dict(zip(dims, [f'{v:.2f}' for v in j]))}")
    print()

    # Navigate with intent
    print("--- Intent-Driven Navigation ---")
    result = collapse.navigate(
        seeds=['dream'],
        n_walks=5,
        steps_per_walk=8
    )

    print(f"\nResults:")
    print(f"  Total transitions: {result.total_transitions}")
    print(f"  Verb collapses: {result.verb_collapses}")
    print(f"  Target collapses: {result.target_collapses}")
    print(f"  Random fallback: {result.random_walks}")
    print(f"  COLLAPSE RATIO: {result.collapse_ratio:.0%}")
    print()

    # Show collapsed words
    intent_words = result.intent_words
    if intent_words:
        print(f"Words reached via intent: {sorted(intent_words)[:10]}")

    # Compare with random walk
    print("\n--- Comparison: Random Walk (no intent) ---")
    collapse.intent_verbs = set()  # Disable intent
    collapse.intent_targets = set()

    random_result = collapse.navigate(
        seeds=['dream'],
        n_walks=5,
        steps_per_walk=8
    )

    print(f"Random transitions: {random_result.total_transitions}")
    print(f"Collapse ratio: {random_result.collapse_ratio:.0%} (should be 0%)")

    collapse.close()


if __name__ == "__main__":
    demo()
