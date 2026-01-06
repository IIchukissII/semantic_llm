"""Semantic Mirror: Psychoanalyst Agent in (A, S, τ) Space.

A conversational agent that mirrors and guides human discourse through
semantic space using gravity physics and dialectical analysis.

Architecture:
    semantic_mirror/
    ├── core/           # Data structures, physics, data layer
    │   ├── models.py   # SemanticState, ConversationTrajectory
    │   ├── physics.py  # Gravity, RC dynamics, constants
    │   └── data.py     # SemanticData singleton (99K coordinates)
    ├── detection/      # Position and marker detection
    │   └── detector.py # SemanticDetector
    ├── analysis/       # Dialectical analysis
    │   └── analyzer.py # SemanticAnalyzer
    ├── agents/         # Agent implementations
    │   └── mirror.py   # SemanticMirror (psychoanalyst)
    ├── storage/        # Persistence (future)
    └── cli/            # Command-line interface
        └── run.py      # Demo

Future Modules:
    - storage/dreams.py: Neo4j semantic core with processed books
    - agents/hero.py: Hero's journey (Campbell's monomyth)
    - agents/archetypes.py: Jungian archetype detection
    - agents/alchemy.py: Alchemical transformation stages
    - agents/myth.py: Mythological pattern analysis

Usage:
    from semantic_mirror import SemanticMirror

    mirror = SemanticMirror()
    state = mirror.observe("I guess everything is fine.")
    diagnosis = mirror.diagnose()
    dialectic = mirror.dialectic()
"""

from .core import (
    SemanticState, ConversationTrajectory,
    SemanticData, get_data, WordCoordinates, Bond,
    KT, LAMBDA, MU, HEALTH, TARGETS,
    gravity_potential, gravity_force, resistance,
    rc_update, therapeutic_vector,
)
from .detection import SemanticDetector
from .analysis import SemanticAnalyzer
from .agents import SemanticMirror, get_mirror

__version__ = '0.2.0'

__all__ = [
    # Main agent
    'SemanticMirror',
    'get_mirror',

    # Models
    'SemanticState',
    'ConversationTrajectory',

    # Data
    'SemanticData',
    'get_data',
    'WordCoordinates',
    'Bond',

    # Components
    'SemanticDetector',
    'SemanticAnalyzer',

    # Physics
    'HEALTH',
    'TARGETS',
    'KT',
    'LAMBDA',
    'MU',
    'gravity_potential',
    'gravity_force',
    'resistance',
    'rc_update',
    'therapeutic_vector',
]
