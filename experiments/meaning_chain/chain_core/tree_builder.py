"""
TreeBuilder: Intent-Driven Meaning Tree Construction

Builds meaning trees using verb-mediated navigation.
User's intent verbs collapse the semantic space to relevant paths.

"Intent collapses meaning like observation collapses wavefunction"
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
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
from models.types import MeaningNode, MeaningTree, SemanticProperties

# Import MeaningGraph (new schema with verb-mediated transitions)
from graph.meaning_graph import MeaningGraph, GraphConfig


@dataclass
class TreeBuilderConfig:
    """Configuration for tree building."""
    max_depth: int = 3
    max_children: int = 4
    min_goodness: float = -1.0      # Minimum g to include
    prefer_positive: bool = True     # Prefer positive goodness
    use_meaning_graph: bool = True   # Use new MeaningGraph (vs fallback)

    # Quantum physics parameters
    temperature: float = 1.0         # Boltzmann temperature for path selection
    intent_coupling: float = 2.0     # How strongly intent affects energy landscape
    barrier_scale: float = 1.0       # Scale factor for barrier opacity


class TreeBuilder:
    """
    Builds meaning trees using intent-driven collapse.

    Key innovation: User's verbs are not decorative - they are
    operators that collapse the semantic space to relevant paths.

    Uses MeaningGraph with VIA relationships:
        (subject)-[:VIA {verb}]->(object)
    """

    def __init__(self, data_loader: Optional[DataLoader] = None,
                 config: Optional[TreeBuilderConfig] = None,
                 graph: Optional[MeaningGraph] = None):
        self.loader = data_loader or DataLoader()
        self.config = config or TreeBuilderConfig()

        # MeaningGraph for verb-mediated navigation
        self._graph = graph
        self._graph_initialized = False

        # Intent-driven collapse state
        self._intent_verbs: Set[str] = set()
        self._intent_targets: Set[str] = set()
        self._intent_j: Optional[np.ndarray] = None  # Intent direction in j-space

        # Fallback data
        self._svo_patterns = None
        self._verb_objects = None

        # Cache for j-vectors
        self._j_cache: Dict[str, np.ndarray] = {}

    def _init_graph(self) -> bool:
        """Initialize MeaningGraph connection."""
        if self._graph_initialized:
            return self._graph is not None and self._graph.is_connected()

        self._graph_initialized = True

        if not self.config.use_meaning_graph:
            return False

        try:
            if self._graph is None:
                print("[TreeBuilder] Connecting to MeaningGraph...")
                self._graph = MeaningGraph()

            if self._graph.is_connected():
                stats = self._graph.get_stats()
                print(f"[TreeBuilder] Connected: {stats['concepts']} concepts, {stats['via_edges']} VIA edges")
                return True
            else:
                print("[TreeBuilder] MeaningGraph not available, using fallback")
                return False

        except Exception as e:
            print(f"[TreeBuilder] Graph init error: {e}")
            return False

    def _load_fallback_data(self):
        """Load SVO patterns from CSV (fallback)."""
        if self._svo_patterns is not None:
            return

        print("[TreeBuilder] Loading fallback data from CSV...")
        self._svo_patterns = self.loader.load_svo_patterns()
        self._verb_objects = self.loader.load_verb_objects()
        print(f"[TreeBuilder] Loaded {len(self._svo_patterns)} SVO patterns")

    def close(self):
        """Clean up resources."""
        if self._graph:
            self._graph.close()
            self._graph = None

    # =========================================================================
    # Quantum Physics
    # =========================================================================

    def _get_j_vector(self, word: str) -> np.ndarray:
        """Get j-vector for a word (5D transcendental direction)."""
        if word in self._j_cache:
            return self._j_cache[word]

        j = np.zeros(5)
        if self._init_graph():
            concept = self._graph.get_concept(word)
            if concept and concept.get('j'):
                j = np.array(concept['j'])

        self._j_cache[word] = j
        return j

    def _compute_barrier(self, j1: np.ndarray, j2: np.ndarray) -> float:
        """
        Compute barrier opacity between two states.

        κ = (1 - cos(j₁, j₂)) / 2

        κ = 0: parallel (no barrier)
        κ = 0.5: orthogonal (medium barrier)
        κ = 1: antiparallel (maximum barrier)
        """
        norm1 = np.linalg.norm(j1)
        norm2 = np.linalg.norm(j2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.5  # Unknown direction -> medium barrier

        cos_sim = np.dot(j1, j2) / (norm1 * norm2)
        return (1 - cos_sim) / 2

    def _compute_energy(self, from_tau: float, to_tau: float,
                        barrier: float, delta_g: float) -> float:
        """
        Compute transition energy cost.

        E = altitude_cost + barrier_cost - reward

        where:
            altitude_cost = max(0, Δτ)  (climbing costs energy)
            barrier_cost = κ × barrier_scale
            reward = Δg × goodness_weight
        """
        altitude_cost = max(0, to_tau - from_tau) * 0.5
        barrier_cost = barrier * self.config.barrier_scale
        reward = delta_g * 0.3  # Negative energy = favorable

        return altitude_cost + barrier_cost - reward

    def _intent_potential(self, j_target: np.ndarray) -> float:
        """
        Compute intent potential (how much intent lowers energy for this target).

        V_intent = -coupling × cos(j_target, j_intent)

        Negative potential = favorable (intent-aligned)
        """
        if self._intent_j is None or np.linalg.norm(self._intent_j) < 1e-6:
            return 0.0

        norm_target = np.linalg.norm(j_target)
        norm_intent = np.linalg.norm(self._intent_j)

        if norm_target < 1e-6:
            return 0.0

        cos_sim = np.dot(j_target, self._intent_j) / (norm_target * norm_intent)
        return -self.config.intent_coupling * cos_sim

    def _boltzmann_probabilities(self, energies: np.ndarray) -> np.ndarray:
        """
        Compute Boltzmann probabilities for transitions.

        P_i = exp(-E_i / T) / Σ exp(-E_j / T)

        Lower energy = higher probability.
        """
        if len(energies) == 0:
            return np.array([])

        T = max(0.01, self.config.temperature)
        # Shift to avoid overflow
        shifted = energies - np.min(energies)
        exp_neg_E = np.exp(-shifted / T)
        return exp_neg_E / np.sum(exp_neg_E)

    def _compute_intent_direction(self, verbs: List[str]):
        """
        Compute intent direction in j-space from intent verbs.

        The intent j-vector is the sum of verb operator j-vectors,
        pointing toward the semantic direction the user wants to go.
        """
        if not verbs:
            self._intent_j = None
            return

        intent_j = np.zeros(5)
        verb_ops = self.loader.load_verb_operators()

        for verb in verbs:
            if verb in verb_ops:
                vec = verb_ops[verb].get('vector', [])
                if vec and len(vec) == 5:
                    try:
                        intent_j += np.array(vec, dtype=float)
                    except (ValueError, TypeError):
                        pass

        if np.linalg.norm(intent_j) > 1e-6:
            self._intent_j = intent_j / np.linalg.norm(intent_j)
        else:
            self._intent_j = None

    # =========================================================================
    # Intent-Driven Collapse
    # =========================================================================

    def _set_intent(self, verbs: List[str]):
        """
        Set intent verbs that collapse the semantic space.

        In quantum terms: intent acts as a measurement operator that
        projects the superposition of possible transitions onto
        intent-aligned states.

        The collapse is not binary filtering but energy modification:
        intent-aligned paths have lower energy (higher probability).
        """
        self._intent_verbs = set(verbs) if verbs else set()

        # Compute intent direction in j-space (the "measurement axis")
        self._compute_intent_direction(verbs)

        if not self._intent_verbs:
            self._intent_targets = set()
            return

        # Get what these verbs operate on (intent targets)
        if self._init_graph():
            self._intent_targets = self._graph.get_verb_targets(list(self._intent_verbs))
            print(f"[TreeBuilder] Intent verbs: {self._intent_verbs}")
            if self._intent_j is not None:
                print(f"[TreeBuilder] Intent j-direction: [{', '.join(f'{x:.2f}' for x in self._intent_j)}]")
            print(f"[TreeBuilder] Intent targets: {len(self._intent_targets)} concepts")
        else:
            # Fallback: use verb_objects
            self._load_fallback_data()
            self._intent_targets = set()
            for verb in self._intent_verbs:
                if verb in self._verb_objects:
                    self._intent_targets.update(self._verb_objects[verb])
            print(f"[TreeBuilder] Intent (fallback): {len(self._intent_targets)} targets")

    def _get_transitions(self, word: str) -> List[Tuple[str, str, float]]:
        """
        Get transitions using quantum physics.

        Computes energy for each transition using:
        - Barrier opacity from j-vector alignment
        - Altitude cost from tau difference
        - Intent potential from projection onto intent direction
        - Goodness reward

        Returns transitions sorted by Boltzmann probability (low energy first).
        """
        # Get raw transitions from graph
        if self._init_graph():
            raw_transitions = self._graph.get_all_transitions(word, limit=self.config.max_children * 4)
        else:
            raw_transitions = self._get_transitions_fallback(word)

        if not raw_transitions:
            return []

        # Get source properties
        source_props = self._get_semantic_properties(word)
        j_source = self._get_j_vector(word)

        # Compute energy for each transition
        energies = []
        for verb, target, _ in raw_transitions:
            target_props = self._get_semantic_properties(target)
            j_target = self._get_j_vector(target)

            # Barrier opacity from j-vector alignment
            barrier = self._compute_barrier(j_source, j_target)

            # Base energy (altitude + barrier - goodness reward)
            delta_g = target_props.g - source_props.g
            energy = self._compute_energy(
                source_props.tau, target_props.tau,
                barrier, delta_g
            )

            # Intent potential (lowers energy for aligned targets)
            energy += self._intent_potential(j_target)

            energies.append(energy)

        # Compute Boltzmann probabilities
        energies = np.array(energies)
        probs = self._boltzmann_probabilities(energies)

        # Return sorted by probability (highest first = lowest energy)
        indexed = [(raw_transitions[i], probs[i], energies[i])
                   for i in range(len(raw_transitions))]
        indexed.sort(key=lambda x: -x[1])  # Sort by probability descending

        # Return (verb, target, probability)
        return [(t[0], t[1], p) for (t, p, e) in indexed]

    def _get_transitions_fallback(self, word: str) -> List[Tuple[str, str, float]]:
        """Get transitions from SVO patterns (fallback)."""
        self._load_fallback_data()

        if word not in self._svo_patterns:
            return []

        transitions = []
        for verb, obj in self._svo_patterns[word]:
            # Score by intent alignment
            score = 0.5
            if verb in self._intent_verbs:
                score += 0.4
            if obj in self._intent_targets:
                score += 0.3

            transitions.append((verb, obj, score))

        # Sort by score
        transitions.sort(key=lambda x: -x[2])
        return transitions[:self.config.max_children * 2]

    def _get_semantic_properties(self, word: str) -> SemanticProperties:
        """Get semantic properties for a word."""
        if self._init_graph():
            concept = self._graph.get_concept(word)
            if concept:
                return SemanticProperties(
                    g=concept['g'],
                    tau=concept['tau'],
                    j=np.array(concept['j']) if concept['j'] else np.zeros(5)
                )

        # Fallback: neutral properties
        return SemanticProperties(g=0.0, tau=3.0)

    # =========================================================================
    # Tree Building
    # =========================================================================

    def _build_subtree(self, word: str, depth: int,
                       visited: Set[str]) -> MeaningNode:
        """
        Recursively build subtree from a word.

        Intent verbs guide which transitions to follow.
        """
        props = self._get_semantic_properties(word)
        node = MeaningNode(
            word=word,
            properties=props,
            depth=depth
        )

        # Stop at max depth
        if depth >= self.config.max_depth:
            return node

        # Get intent-collapsed transitions
        transitions = self._get_transitions(word)

        # Build children
        children_added = 0
        for verb, target, score in transitions:
            if children_added >= self.config.max_children:
                break

            if target in visited:
                continue

            # Filter by goodness if configured
            target_props = self._get_semantic_properties(target)
            if target_props.g < self.config.min_goodness:
                continue

            # Recursively build child
            child_node = self._build_subtree(
                target,
                depth + 1,
                visited | {target}
            )
            child_node.verb_from_parent = verb

            node.children.append(child_node)
            children_added += 1

        return node

    def _word_exists(self, word: str) -> bool:
        """Check if word exists in semantic space."""
        if self._init_graph():
            return self._graph.has_concept(word)

        # Fallback
        self._load_fallback_data()
        return word in self._svo_patterns

    def build_tree(self, roots: List[str], source_text: str = "") -> MeaningTree:
        """
        Build a meaning tree from root concepts.

        Args:
            roots: List of root words (nouns)
            source_text: Original text for reference

        Returns:
            MeaningTree with all roots expanded
        """
        tree = MeaningTree(
            roots=[],
            max_depth=self.config.max_depth,
            source_text=source_text
        )

        all_visited = set()
        for root in roots:
            if root in all_visited:
                continue

            if not self._word_exists(root):
                print(f"[TreeBuilder] '{root}' not in semantic space, skipping")
                continue

            visited = {root}
            root_node = self._build_subtree(root, 0, visited)
            tree.roots.append(root_node)

            # Track all visited
            for node in root_node.flatten():
                all_visited.add(node.word)

        return tree

    def build_from_decomposition(self, nouns: List[str], verbs: List[str],
                                  source_text: str = "") -> MeaningTree:
        """
        Build tree using INTENT-DRIVEN COLLAPSE.

        The user's verbs act as collapse operators - they observe
        the superposition of possible meanings and collapse it
        to paths aligned with intent.

        Args:
            nouns: Root concepts (the "what")
            verbs: Intent operators (the "how" - collapse operators)
            source_text: Original text

        Returns:
            MeaningTree collapsed to intent-relevant meanings
        """
        # Set intent BEFORE building
        self._set_intent(verbs)

        # Build tree with intent collapse
        tree = self.build_tree(nouns, source_text)

        # Clear intent for next query
        self._intent_verbs = set()
        self._intent_targets = set()

        return tree


class MeaningChainPipeline:
    """
    Complete pipeline: Text -> MeaningTree

    Combines Decomposer and TreeBuilder.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None,
                 tree_config: Optional[TreeBuilderConfig] = None):
        self.loader = data_loader or DataLoader()
        self.tree_config = tree_config or TreeBuilderConfig()

        from .decomposer import Decomposer

        self.decomposer = Decomposer(self.loader)
        self.tree_builder = TreeBuilder(self.loader, self.tree_config)

    def process(self, text: str) -> MeaningTree:
        """
        Process text into meaning tree with intent collapse.

        Args:
            text: Input sentence

        Returns:
            MeaningTree
        """
        # Decompose
        decomposed = self.decomposer.decompose(text)
        print(f"[Pipeline] Nouns: {decomposed.nouns}, Verbs: {decomposed.verbs}")

        # Build tree with intent
        tree = self.tree_builder.build_from_decomposition(
            decomposed.nouns,
            decomposed.verbs,
            text
        )

        print(f"[Pipeline] Tree: {tree.root_count} roots, {tree.total_nodes} nodes")
        return tree

    def close(self):
        """Clean up resources."""
        self.tree_builder.close()
