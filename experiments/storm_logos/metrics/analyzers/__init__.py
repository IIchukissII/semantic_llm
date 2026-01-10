"""Analyzers: Compute specific metrics."""

from .irony import IronyAnalyzer
from .coherence import CoherenceAnalyzer
from .tau import TauAnalyzer
from .tension import TensionAnalyzer
from .defense import DefenseAnalyzer
from .boundary import BoundaryAnalyzer
from .archetype import ArchetypeAnalyzer, get_archetype_analyzer
