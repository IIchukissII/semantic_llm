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
│   ├── mirror.py       # SemanticMirror (analysis)
│   └── therapist.py    # Therapist (Mistral + feedback loop)
│
├── storage/            # Persistence layer (future)
│   └── __init__.py
│
└── cli/                # Command-line interface
    ├── __init__.py
    ├── run.py          # Demo session
    ├── conversation.py # Claude patient + Mistral therapist
    └── visualize.py    # 3D trajectory visualization
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

Where:
  λ = 0.5 (gravitational constant)
  μ = 0.5 (lift constant)
  kT = e^(-1/5) ≈ 0.819 (thermal constant)
```

Semantic gravity pulls naturally toward:
- **Concrete** (lower τ): grounding in reality
- **Good** (higher A): affirmation and health

### RC Dynamics

Conversation state accumulates like charge on a capacitor:
```
dQ/dt = (Q_input - Q) × (1 - |Q|/Q_max) - Q × decay
```
- Recent utterances have more weight
- State decays toward neutral over time
- Memory window tracks trajectory

### Dialectical Engine

Each conversational state generates:
- **Thesis**: current position (what patient shows)
- **Antithesis**: opposite (what patient hides/avoids)
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

## Key Features

### 1. Physics-Based Response Generation

Response parameters computed from semantic physics, not hardcoded:

```python
# Gravity potential
φ = gravity_potential(state)  # φ = λτ - μA

# Therapeutic vector (direction toward health)
vec = therapeutic_vector(state, HEALTH)

# Receptivity (how open is patient?)
receptivity = compute_receptivity(state, diagnosis)

# Response length determined by receptivity
tokens = 15 + receptivity * 85  # 15-100 tokens
```

### 2. Receptivity Parameter

Patient openness computed from physics:

```python
receptivity = 1.0
receptivity *= (1.0 - irony)      # Irony closes off
receptivity *= (1.0 - resistance)  # Resistance reduces openness
if A < 0: receptivity *= (1 + A)   # Negative A = defensive
if τ > 2.5: receptivity *= 0.8     # Too abstract = less grounded

# Result: 0.1 to 1.0
# Low receptivity → short response (1-5 words)
# High receptivity → can elaborate (1-2 sentences)
```

### 3. Therapy-Speak Filter

System detects and penalizes cliché therapeutic phrases:

```python
THERAPY_SPEAK = [
    "journey together", "self-compassion", "growth",
    "strategies", "navigate", "safe space",
    "I think", "I believe", "I understand",
    "let's explore", "work through", ...
]

# Detected phrases reduce response score
# Forces regeneration with cleaner language
```

### 4. Extended Psychological Markers

Beyond basic irony/sarcasm, detect:
- **vulnerability**: openness, rawness ("honestly", "I feel", "*sighs*")
- **minimization**: downplaying ("just", "only", "no big deal")
- **self_deprecation**: self-attack ("who am I to", "pathetic", "I failed")
- **deflection**: avoiding ("anyway", "forget it", "doesn't matter")
- **projection**: blaming others ("they always", "it's their fault")
- **rationalization**: over-explaining ("because", "logically", "makes sense")
- **humor_defense**: using humor to avoid ("haha", "*laughs*", "joking")

### 5. Defense Detection

Defenses manifest as semantic patterns:
- **Intellectualization**: τ > 3.0 (flight into abstraction)
- **Irony as distance**: high irony + low vulnerability
- **Negation**: A < -0.3 (refusing affirmation)
- **Devaluation**: S < -0.3 (reducing meaning)

### 6. Irony Tracking

**Critical insight**: Rising irony = therapist missed the patient.

```python
irony_delta = current_irony - previous_irony

if irony_delta > 0.1:
    # Patient is CLOSING OFF
    # Add warning to prompt:
    # "Mirror their darkness BEFORE offering anything else"
    # Block premature solutions/positivity
```

Dialectical approach:
- Stay with THESIS (darkness) before jumping to ANTITHESIS (light)
- SYNTHESIS emerges through held tension, not forced resolution

### 7. Resistance = Fighting Gravity

```python
resistance = motion against therapeutic gravity

# High resistance when:
# - Moving toward -A when gravity pulls toward +A
# - Moving toward +τ when gravity pulls toward grounding
```

## Conversation Example

```
[Turn 4]
Patient: *lets out a bitter laugh* What if I spend years digging
         and there's just... nothing there worth finding?
  [A=+0.16, S=-0.03, τ=1.5] recv:57% tok:63 irony:40%

Therapist: Your intricate layers and hidden clues are quite evident.

# Low receptivity (57%) due to high irony (40%)
# Short response (63 tokens max)
# System detected patient closing off
```

## Future Work

### Phase 1: Neo4j Psychoanalytic Core

**Process psychoanalytic literature to generate semantic bonds:**

```
Books to process:
├── Freud
│   ├── Interpretation of Dreams
│   ├── Beyond the Pleasure Principle
│   ├── Ego and the Id
│   └── Civilization and Its Discontents
│
├── Jung
│   ├── Archetypes and the Collective Unconscious
│   ├── Psychology and Alchemy
│   ├── Man and His Symbols
│   └── Red Book
│
├── Lacan
│   ├── Écrits
│   └── Seminar Series
│
├── Klein
│   ├── Envy and Gratitude
│   └── Love, Guilt and Reparation
│
└── Winnicott
    ├── Playing and Reality
    └── The Maturational Processes
```

**Neo4j Schema:**

```cypher
// Concepts from psychoanalytic literature
(Concept {
    word: "repression",
    A: -0.2,
    S: 0.3,
    tau: 2.8,
    source: "Freud",
    book: "Interpretation of Dreams"
})

// Bonds extracted from text
(Bond {
    noun: "desire",
    adj: "unconscious",
    frequency: 847,
    context: "The unconscious desire manifests..."
})

// Relationships
(:Concept)-[:OPPOSES]->(:Concept)      // death vs life
(:Concept)-[:TRANSFORMS]->(:Concept)   // nigredo → albedo
(:Concept)-[:CONTAINS]->(:Concept)     // self contains shadow
(:Bond)-[:FROM_BOOK]->(:Book)
(:Concept)-[:IN_STAGE]->(:Stage)       // hero's journey stage
```

**Bond Generation Pipeline:**

```python
# 1. Parse psychoanalytic text
bonds = extract_bonds(book_text)

# 2. Compute semantic coordinates
for bond in bonds:
    bond.A, bond.S, bond.tau = compute_coordinates(bond)

# 3. Store in Neo4j with context
neo4j.create_bond(bond, source=book, context=sentence)

# 4. Build concept graph
neo4j.link_concepts(bonds)
```

### Phase 2: Narrative Agents

**Hero's Journey (Campbell)**
```
agents/hero.py
```
Detect stages: Call to Adventure → Crossing Threshold → Ordeal → Return
- Map patient's journey through (A, S, τ) space
- Each stage has characteristic semantic signature

**Archetypes (Jung)**
```
agents/archetypes.py
```
- Hero, Shadow, Anima/Animus, Self
- Trickster, Wise Old Man, Great Mother
- Detect archetypal patterns in discourse
- Query Neo4j for archetypal bonds

**Alchemy (Transformation)**
```
agents/alchemy.py
```
Stages: Nigredo → Albedo → Citrinitas → Rubedo
- Psychological transformation tracking
- Death/rebirth patterns
- Map stages to semantic coordinates

**Myth Patterns**
```
agents/myth.py
```
- Creation myths, destruction myths
- Trickster narratives
- Transformation stories

### Phase 3: Integration

```python
# Combine semantic position with narrative stage
class IntegratedAnalysis:
    def analyze(self, text):
        # 1. Semantic position
        state = mirror.observe(text)

        # 2. Query Neo4j for relevant concepts
        concepts = neo4j.query_neighborhood(state.A, state.S, state.tau)

        # 3. Detect hero's journey stage
        stage = hero_agent.detect_stage(trajectory)

        # 4. Find archetypal patterns
        archetype = archetype_agent.detect(text, concepts)

        # 5. Determine transformation phase
        phase = alchemy_agent.detect_phase(trajectory)

        return {
            'state': state,
            'concepts': concepts,
            'stage': stage,
            'archetype': archetype,
            'phase': phase,
            'intervention': compute_intervention(...)
        }
```

### Phase 4: Semantic Response Templates

Generate response templates from Neo4j bonds:

```python
# Query bonds for current state
bonds = neo4j.query("""
    MATCH (b:Bond)
    WHERE abs(b.A - $patient_A) < 0.3
      AND abs(b.S - $patient_S) < 0.3
    RETURN b
    ORDER BY b.frequency DESC
    LIMIT 10
""", patient_A=state.A, patient_S=state.S)

# Use bonds as semantic skeleton for response
skeleton = select_therapeutic_bonds(bonds, therapeutic_vector)
response = render_response(skeleton, receptivity)
```

## Run

```bash
# From semantic_mirror directory
cd /path/to/semantic_mirror

# Demo session
python -m cli.run

# Interactive session
python -m cli.run -i

# Test conversation (Claude patient + Mistral therapist)
python -c "from semantic_mirror.cli.conversation import run_conversation; run_conversation('mixed', 6)"
```

## Dependencies

- Python 3.8+
- psycopg2 (PostgreSQL for word coordinates)
- requests (Ollama API)
- anthropic (Claude API for patient simulation)
- Future: neo4j (for psychoanalytic semantic core)

## References

### Psychoanalysis
- Freud: defense mechanisms, pleasure principle, unconscious
- Jung: archetypes, collective unconscious, individuation
- Lacan: desire, symbolic order, the Real
- Klein: object relations, paranoid-schizoid position
- Winnicott: true self/false self, transitional space

### Narrative Theory
- Campbell: hero's journey (monomyth)
- Propp: narrative functions
- Jung: archetypal narratives

### Philosophy
- Hegel: dialectic (thesis → antithesis → synthesis)
- Alchemy: transformation stages (nigredo → rubedo)

### Physics
- Boltzmann: P ∝ exp(-ΔE/kT)
- RC dynamics: capacitor charging model
- Gravity: potential field φ = λτ - μA
