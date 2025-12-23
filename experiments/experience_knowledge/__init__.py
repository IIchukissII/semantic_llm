"""
Experience-Based Knowledge System

Core principle: "Only believe what was lived is knowledge"

Architecture (5-layer):
    Layer 1: Consciousness - Meditation, Sleep, Prayer
    Layer 2: Render - LLM rendering (Mistral)
    Layer 3: Feedback - Intent/response analysis
    Layer 4: Navigation - Neo4j graph navigation
    Layer 5: Semantic Space - Base semantic states

Modules:
    layers/  - Core modules (consciousness, prompts, core classes)
    graph/   - Neo4j database utilities
    app/     - Application entry points
    tests/   - Test suite
    config/  - Configuration files
    data/    - Runtime data (results, versions)
"""

# Re-export from layers for convenience
from .layers import (
    # Core
    SemanticState,
    Wholeness,
    Experience,
    ExperiencedAgent,
    create_naive_agent,
    create_experienced_agent,
    # Prompts
    KnowledgeState,
    KnowledgeAssessor,
    PromptConfig,
    SemanticPromptBuilder,
    DomainResponseBuilder,
    # Consciousness
    CenteredState,
    SleepReport,
    Resonance,
    Meditation,
    Sleep,
    Prayer,
    create_consciousness,
)

# Re-export from graph
from .graph import (
    GraphConfig,
    ExperienceGraph,
)

__all__ = [
    # Core
    'SemanticState',
    'Wholeness',
    'Experience',
    'ExperiencedAgent',
    'create_naive_agent',
    'create_experienced_agent',
    # Prompts
    'KnowledgeState',
    'KnowledgeAssessor',
    'PromptConfig',
    'SemanticPromptBuilder',
    'DomainResponseBuilder',
    # Consciousness
    'CenteredState',
    'SleepReport',
    'Resonance',
    'Meditation',
    'Sleep',
    'Prayer',
    'create_consciousness',
    # Graph
    'GraphConfig',
    'ExperienceGraph',
]
