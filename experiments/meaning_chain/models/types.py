"""
Core data types for Meaning Chain experiment.

MeaningNode: A single concept in semantic space with its properties and connections
MeaningTree: A forest of meaning trees built from sentence decomposition

Coordinate System (Jan 2026):
    Every word = (A, S, τ) - three semantic coordinates
    A = Affirmation (PC1, 83.3% variance) - replaces old 'g'
    S = Sacred (PC2, 11.7% variance)
    τ = Abstraction level [1=concrete, 6=abstract]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

# Principal Component vectors for 2D projection
PC1_AFFIRMATION = np.array([-0.448, -0.519, -0.118, -0.480, -0.534])
PC2_SACRED = np.array([-0.513, +0.128, -0.732, +0.420, +0.090])


@dataclass
class SemanticProperties:
    """
    Semantic properties of a concept.

    Primary coordinates (A, S, τ):
        A = Affirmation score (projection onto PC1)
        S = Sacred score (projection onto PC2)
        tau = Abstraction level [1=concrete, 6=abstract]

    Legacy:
        g = old goodness dimension (now g ≈ A, kept for compatibility)
    """
    # Primary 2D coordinates (Jan 2026)
    affirmation: float = 0.0          # A: Affirmation (PC1 projection)
    sacred: float = 0.0               # S: Sacred (PC2 projection)
    tau: float = 3.0                  # τ: Abstraction level

    # Legacy (kept for backward compatibility)
    g: float = 0.0                    # Old goodness (≈ A, deprecated)

    # Original vectors (for recomputation if needed)
    j: Optional[np.ndarray] = None   # 5D transcendental vector
    i: Optional[np.ndarray] = None   # 11D context vector

    # Bond space properties
    h_adj_norm: float = 0.5          # Normalized adjective entropy
    variety: int = 0                  # Adjective variety count

    def __post_init__(self):
        """Compute A, S from j-vector if available and not set."""
        if self.j is not None and self.affirmation == 0.0 and self.sacred == 0.0:
            self.affirmation = float(np.dot(self.j, PC1_AFFIRMATION))
            self.sacred = float(np.dot(self.j, PC2_SACRED))
            # Keep g ≈ A for compatibility
            if self.g == 0.0:
                self.g = self.affirmation

    @property
    def A(self) -> float:
        """Alias for affirmation."""
        return self.affirmation

    @property
    def S(self) -> float:
        """Alias for sacred."""
        return self.sacred

    def to_dict(self) -> Dict[str, Any]:
        return {
            'affirmation': self.affirmation,
            'sacred': self.sacred,
            'tau': self.tau,
            'g': self.g,  # Legacy
            'j': self.j.tolist() if self.j is not None else None,
            'i': self.i.tolist() if self.i is not None else None,
            'h_adj_norm': self.h_adj_norm,
            'variety': self.variety,
        }


@dataclass
class MeaningNode:
    """
    A node in the meaning tree representing a concept.

    Each node contains:
    - The word/concept itself
    - Its semantic properties (A, S, τ)
    - How it connects to children (via verb operators)
    - The depth in the tree
    """
    word: str
    properties: SemanticProperties
    children: List['MeaningNode'] = field(default_factory=list)
    verb_from_parent: Optional[str] = None  # The verb that led here from parent
    depth: int = 0

    @property
    def A(self) -> float:
        """Affirmation score."""
        return self.properties.affirmation

    @property
    def S(self) -> float:
        """Sacred score."""
        return self.properties.sacred

    @property
    def g(self) -> float:
        """Legacy: goodness (≈ A)."""
        return self.properties.g

    @property
    def tau(self) -> float:
        return self.properties.tau

    @property
    def j(self) -> Optional[np.ndarray]:
        return self.properties.j

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def child_count(self) -> int:
        return len(self.children)

    def total_descendants(self) -> int:
        """Count all descendants recursively."""
        count = len(self.children)
        for child in self.children:
            count += child.total_descendants()
        return count

    def flatten(self) -> List['MeaningNode']:
        """Return all nodes in subtree as flat list (BFS order)."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def get_path_to_root(self) -> List[str]:
        """Get the path of words from root to this node (requires parent tracking)."""
        return [self.word]  # For now, just returns self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'word': self.word,
            'properties': self.properties.to_dict(),
            'verb_from_parent': self.verb_from_parent,
            'depth': self.depth,
            'children': [child.to_dict() for child in self.children],
        }

    def __repr__(self) -> str:
        verb_str = f" ←({self.verb_from_parent})" if self.verb_from_parent else ""
        children_str = f" [{len(self.children)} children]" if self.children else ""
        return f"MeaningNode({self.word}{verb_str}, A={self.A:.2f}, S={self.S:.2f}, τ={self.tau:.1f}{children_str})"


@dataclass
class MeaningTree:
    """
    A forest of meaning trees built from a sentence.

    Each root corresponds to a key concept extracted from the input.
    Children are built by navigating semantic space using verb operators.
    """
    roots: List[MeaningNode] = field(default_factory=list)
    max_depth: int = 3
    source_text: str = ""

    @property
    def root_count(self) -> int:
        return len(self.roots)

    @property
    def total_nodes(self) -> int:
        """Total nodes across all trees."""
        count = 0
        for root in self.roots:
            count += 1 + root.total_descendants()
        return count

    def all_words(self) -> List[str]:
        """Get all unique words in the forest."""
        words = set()
        for root in self.roots:
            for node in root.flatten():
                words.add(node.word)
        return list(words)

    def all_nodes(self) -> List[MeaningNode]:
        """Get all nodes in the forest."""
        nodes = []
        for root in self.roots:
            nodes.extend(root.flatten())
        return nodes

    def get_paths(self) -> List[List[str]]:
        """Get all root-to-leaf paths as word lists."""
        paths = []

        def traverse(node: MeaningNode, current_path: List[str]):
            current_path = current_path + [node.word]
            if node.is_leaf:
                paths.append(current_path)
            else:
                for child in node.children:
                    traverse(child, current_path)

        for root in self.roots:
            traverse(root, [])

        return paths

    def get_transitions(self) -> List[tuple]:
        """Get all (from_word, verb, to_word) transitions."""
        transitions = []

        def traverse(node: MeaningNode):
            for child in node.children:
                transitions.append((node.word, child.verb_from_parent, child.word))
                traverse(child)

        for root in self.roots:
            traverse(root)

        return transitions

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the tree."""
        all_nodes = self.all_nodes()
        return {
            'root_count': self.root_count,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'avg_affirmation': np.mean([n.A for n in all_nodes]) if all_nodes else 0,
            'avg_sacred': np.mean([n.S for n in all_nodes]) if all_nodes else 0,
            'avg_tau': np.mean([n.tau for n in all_nodes]) if all_nodes else 0,
            'paths': len(self.get_paths()),
            'transitions': len(self.get_transitions()),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_text': self.source_text,
            'max_depth': self.max_depth,
            'roots': [root.to_dict() for root in self.roots],
            'summary': self.summary(),
        }

    def pretty_print(self, indent: str = "  ") -> str:
        """Pretty print the tree structure."""
        lines = []
        lines.append(f"MeaningTree: {self.root_count} roots, {self.total_nodes} total nodes")
        lines.append(f"Source: \"{self.source_text}\"")
        lines.append("")

        def print_node(node: MeaningNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            verb_str = f"({node.verb_from_parent})→ " if node.verb_from_parent else ""
            lines.append(f"{prefix}{connector}{verb_str}{node.word} [A={node.A:.2f}, S={node.S:.2f}, τ={node.tau:.1f}]")

            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                print_node(child, child_prefix, i == len(node.children) - 1)

        for i, root in enumerate(self.roots):
            lines.append(f"Root {i+1}: {root.word}")
            for j, child in enumerate(root.children):
                print_node(child, "", j == len(root.children) - 1)
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MeaningTree(roots={self.root_count}, nodes={self.total_nodes}, depth={self.max_depth})"
