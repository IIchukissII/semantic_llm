"""
Semantic Physics Experiments

Read-only observers for testing semantic physics hypotheses.
These modules NEVER modify the graph - observation only.

Hypotheses tested:
- Semantic Gravity (G1-G5)
- Semantic Thermodynamics (T1-T5)
- Semantic Optics (O1-O5)
- Semantic Field Theory (F1-F5)
- Integrated Semantics (Φ1-Φ5)

Usage:
    from experiments.physics import SemanticObserver

    observer = SemanticObserver()
    results = observer.run_all_tests()
"""

from .observers import SemanticObserver, ObservationResult

__all__ = ['SemanticObserver', 'ObservationResult']
