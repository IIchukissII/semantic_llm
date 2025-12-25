"""
Meaning Chain Experiment

A modular component that decomposes sentences into meaning trees,
navigates semantic space to build multi-level chains, and renders
rich context for LLM generation.

Architecture:
    Sentence → Decomposer → [meanings] → TreeBuilder → MeaningTree → Renderer → LLM
"""

from .chain_core.decomposer import Decomposer
from .chain_core.tree_builder import TreeBuilder
from .chain_core.renderer import Renderer
from .models.types import MeaningNode, MeaningTree

__all__ = [
    'Decomposer',
    'TreeBuilder',
    'Renderer',
    'MeaningNode',
    'MeaningTree',
]
