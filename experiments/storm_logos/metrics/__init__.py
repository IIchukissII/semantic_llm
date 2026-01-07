"""Metrics Engine: Measurement and Analysis.

Provides:
    - Extractors: text -> bonds, bonds -> coordinates
    - Analyzers: irony, coherence, tau, tension, defense
    - MetricsEngine: orchestrates all measurements
"""

from .engine import MetricsEngine, get_metrics_engine
