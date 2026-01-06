# Semantic Mirror: Psychoanalyst Agent

A conversational agent that mirrors and guides human discourse through semantic space using gravity physics and dialectical analysis.

## Architecture

```
semantic_mirror/
├── __init__.py         # Public API
├── README.md           # This file
│
├── core/               # Core data structures and physics
│   ├── __init__.py
│   ├── models.py       # SemanticState, ConversationTrajectory
│   ├── physics.py      # Gravity, RC dynamics, constants
│   └── data.py         # SemanticData singleton (99K coordinates)
│
├── detection/          # Position and marker detection
│   ├── __init__.py
│   └── detector.py     # SemanticDetector
│
├── analysis/           # Dialectical analysis and diagnosis
│   ├── __init__.py
│   └── analyzer.py     # SemanticAnalyzer
│
├── agents/             # Agent implementations
│   ├── __init__.py
│   └── mirror.py       # SemanticMirror (psychoanalyst)
│
├── storage/            # Persistence layer (future)
│   └── __init__.py
│
└── cli/                # Command-line interface
    ├── __init__.py
    └── run.py          # Demo CLI
```

## Core Theory

### Semantic Space (A, S, τ)

Human discourse moves through three-dimensional semantic space:

- **A (Affirmation)**: -1 to +1. Negative = denial, criticism. Positive = affirmation, hope.
- **S (Sacred)**: -1 to +1. Negative = mundane, trivial. Positive = meaningful, transcendent.
- **τ (Abstraction)**: 0.5 to 4.5. Low = concrete, sensory. High = abstract, conceptual.

### Gravity Physics

```
φ = λτ - μA
```

Semantic gravity pulls naturally toward:
- **Concrete** (lower τ): grounding in reality
- **Good** (higher A): affirmation and health

### RC Dynamics

Conversation state accumulates like charge on a capacitor:
- Recent utterances have more weight
- State decays toward neutral over time
- Memory window tracks trajectory

### Dialectical Engine

Each conversational state generates:
- **Thesis**: current position
- **Antithesis**: opposite (what patient avoids)
- **Synthesis**: integration toward health

## Usage

```python
from semantic_mirror import SemanticMirror

mirror = SemanticMirror()

# Observe human text
state = mirror.observe("I guess everything is fine. It's always fine.")

# Get diagnosis
diagnosis = mirror.diagnose()
# {'defenses': ['irony_as_distance'], 'resistance': 0.23, ...}

# Get dialectical analysis
dialectic = mirror.dialectic()
# {'thesis': {...}, 'antithesis': {...}, 'synthesis': {...}}
```

## Key Insights

### 1. Defense Detection

Defenses manifest as semantic patterns:
- **Intellectualization**: τ > 3.0 (flight into abstraction)
- **Irony as distance**: high A + low S (false positivity)
- **Negation**: A < -0.3 (refusing affirmation)
- **Devaluation**: S < -0.3 (reducing meaning)

### 2. Resistance = Fighting Gravity

Resistance = motion against therapeutic gravity:
- Moving toward -A when gravity pulls toward +A
- Moving toward +τ when gravity pulls toward grounding

### 3. No Hardcoded Interventions

Therapeutic interventions computed as vectors, not hardcoded phrases.

## Future Work

### Phase 1: Storage Layer

**Neo4j Semantic Core**
- Graph database for processed book semantic data
- Store word relationships, narrative patterns
- Query semantic neighborhoods

### Phase 2: Narrative Agents

**Hero's Journey (Campbell)**
```
agents/hero.py
```
Detect stages: Call to Adventure → Crossing Threshold → Ordeal → Return

**Archetypes (Jung)**
```
agents/archetypes.py
```
- Hero, Shadow, Anima/Animus, Self
- Trickster, Wise Old Man, Great Mother
- Map discourse to archetypal patterns

**Alchemy (Transformation)**
```
agents/alchemy.py
```
Stages: Nigredo → Albedo → Citrinitas → Rubedo
- Psychological transformation tracking
- Death/rebirth patterns

**Myth Patterns**
```
agents/myth.py
```
- Creation myths, destruction myths
- Trickster narratives
- Transformation stories

### Phase 3: Integration

- Combine semantic position with narrative stage
- Track hero's journey through (A, S, τ) space
- Map archetypal transitions to semantic gravity

## Run

```bash
# From semantic_mirror directory
cd /path/to/semantic_mirror

# Demo session
python -m cli.run

# Interactive session
python -m cli.run -i
```

## Dependencies

- Python 3.8+
- psycopg2 (PostgreSQL for word coordinates)
- Future: neo4j (for semantic core)

## References

- Freud: defense mechanisms, pleasure principle
- Jung: archetypes, collective unconscious
- Lacan: desire, symbolic order
- Campbell: hero's journey (monomyth)
- Hegel: dialectic (thesis → antithesis → synthesis)
- Alchemy: transformation stages
