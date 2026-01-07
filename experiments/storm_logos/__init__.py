"""Storm-Logos: Adaptive Semantic System.

A layered architecture for semantic navigation with adaptive control.

Layers:
    1. DATA LAYER - PostgreSQL (bonds, coordinates) + Neo4j (trajectories)
    2. SEMANTIC LAYER - Storm, Dialectic, Chain Reaction
    3. METRICS ENGINE - Extractors + Analyzers
    4. FEEDBACK ENGINE - Error computation against homeostatic targets
    5. GENERATION ENGINE - Storm -> Dialectic -> Chain pipeline
    6. ADAPTIVE CONTROLLER - PI control for parameter tuning
    7. ORCHESTRATION LAYER - Main loop coordination
    8. APPLICATION LAYER - Therapist, Generator, Navigator, Analyzer
"""

__version__ = "0.1.0"
__author__ = "Storm-Logos Team"

from .config import Config, get_config
