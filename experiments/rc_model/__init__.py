"""RC-Model v2: Unified Bond Dynamics

Semantic text analysis through RC-circuit dynamics.

Pipeline:
    TEXT → BONDS → COORDINATES → TRAJECTORY → CARDIOGRAM

Core insight:
    Everything is (noun, adj) bonds
    Context is RC-circuit with three charges (Q_A, Q_S, Q_τ)
    Result is "cardiogram of meaning"

Usage:
    from rc_model import process_text, plot_cardiogram

    trajectory = process_text("The old man runs quickly.")
    fig = plot_cardiogram(trajectory)
"""

from .core import (
    CoordLoader,
    BondExtractor,
    SemanticRC,
    plot_cardiogram,
    plot_comparison,
    compute_metrics,
)
from .core.semantic_rc import process_text, Trajectory, KT
from .core.bond_extractor import extract_bonds, TextBonds

__version__ = "2.0.0"

__all__ = [
    # Classes
    'CoordLoader',
    'BondExtractor',
    'SemanticRC',
    'Trajectory',
    'TextBonds',

    # Functions
    'process_text',
    'extract_bonds',
    'plot_cardiogram',
    'plot_comparison',
    'compute_metrics',

    # Constants
    'KT',
]
