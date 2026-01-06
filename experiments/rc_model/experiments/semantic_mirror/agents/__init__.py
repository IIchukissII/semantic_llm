"""Agents: Semantic Mirror and future agents.

Current:
- SemanticMirror: Psychoanalyst agent (analysis only)
- Therapist: Mistral-powered response generation

Future:
- DreamEngine: Neo4j semantic core with processed books
- HeroJourney: Narrative pattern analysis (Campbell's monomyth)
- Archetypes: Jungian archetype detection
- Alchemy: Transformation stage tracking
"""

from .mirror import SemanticMirror, get_mirror
from .therapist import Therapist

__all__ = ['SemanticMirror', 'get_mirror', 'Therapist']
