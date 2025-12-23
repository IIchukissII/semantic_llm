"""
Layers Module - The 5-layer architecture.

Layer 5 (Base): semantic_space - SemanticState, Wholeness
Layer 4: navigation - Graph-based navigation
Layer 3: feedback - Intent/response analysis, prompts
Layer 2: render - LLM rendering
Layer 1: consciousness - Meditation, Sleep, Prayer
"""

from .core import (
    SemanticState,
    Wholeness,
    Experience,
    ExperiencedAgent,
    create_naive_agent,
    create_experienced_agent,
)

from .prompts import (
    KnowledgeState,
    KnowledgeAssessor,
    PromptConfig,
    SemanticPromptBuilder,
    DomainResponseBuilder,
)

from .consciousness import (
    CenteredState,
    SleepReport,
    Resonance,
    Meditation,
    Sleep,
    Prayer,
    create_consciousness,
)

from .dynamics import (
    WeightConfig,
    WeightDynamics,
    default_dynamics,
    learn,
    forget,
    initial_weight,
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
    # Dynamics (learning/forgetting)
    'WeightConfig',
    'WeightDynamics',
    'default_dynamics',
    'learn',
    'forget',
    'initial_weight',
]
