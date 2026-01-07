# Storm-Logos: Adaptive Semantic Generation System

An 8-layer adaptive system for semantic text generation based on psychoanalytic and RC-model principles.

## Overview

Storm-Logos implements a homeostatic text generation system that:

- **Navigates semantic space** defined by (A, S, τ) coordinates:
  - **A** (Affirmation): good ↔ bad axis
  - **S** (Sacred): elevated ↔ mundane axis
  - **τ** (Tau): abstraction level (0=concrete, 5=abstract)

- **Uses RC-circuit dynamics** for smooth state transitions
- **Implements Chain Reaction** for coherent bond selection via resonance
- **Adapts parameters** through PI control to maintain homeostatic targets

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 8: APPLICATIONS                                         │
│  Therapist, Generator, Navigator, Analyzer                     │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 7: ORCHESTRATION                                        │
│  Main loop: Measure → Adapt → Generate                         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 6: ADAPTIVE CONTROLLER                                  │
│  PI control: Δp = η·error + κ·∫error                          │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5: GENERATION ENGINE                                    │
│  Storm → Dialectic → Chain pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: FEEDBACK ENGINE                                      │
│  Error = target - current                                      │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: METRICS ENGINE                                       │
│  Extractors + Analyzers (coherence, irony, tension, tau)       │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: SEMANTIC LAYER                                       │
│  Storm, Dialectic, Chain, Physics, State                       │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: DATA LAYER                                           │
│  PostgreSQL (coordinates) + Neo4j (trajectories)               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### RC-Model Dynamics

State updates follow RC-circuit behavior:
```
dQ/dt = (x_w - Q) × (1 - |Q|/Q_max) - Q × decay
```

### Boltzmann Factor

Transition probability for abstraction jumps:
```
P(Δτ) ∝ exp(-|Δτ|/kT)  where kT ≈ 0.819
```

### Chain Reaction (Resonance Scoring)

Coherent bond selection through history resonance:
```python
power = Σ coherence(candidate, history[i]) × decay^i
if power > threshold:
    power = threshold + (power - threshold)²  # Lasing
```

### Homeostatic Targets

| Metric | Target |
|--------|--------|
| Coherence | 0.70 |
| Irony | 0.15 |
| Tension | 0.60 |
| τ Variance | 0.80 |
| Noise Ratio | 0.20 |
| τ Slope | -0.10 |

## Installation

```bash
# Clone the repository
cd experiments/storm_logos

# Install dependencies
pip install spacy neo4j numpy

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### CLI Commands

```bash
# Generate text
python -m storm_logos.cli.main generate --genre balanced --sentences 3

# Analyze text
python -m storm_logos.cli.main analyze "The ancient temple held sacred mysteries."

# Interactive therapy session
python -m storm_logos.cli.main therapy --interactive

# System info
python -m storm_logos.cli.main info
```

### Python API

```python
from storm_logos.data import SemanticState, Bond
from storm_logos.applications.generator import Generator
from storm_logos.applications.analyzer import Analyzer

# Generate text
gen = Generator()
text = gen.generate(genre='dramatic', n_sentences=3)
print(text)

# Analyze text
analyzer = Analyzer()
result = analyzer.analyze_text("The dark forest held ancient secrets.")
print(f"Position: A={result['position']['A']:.2f}, S={result['position']['S']:.2f}")
print(f"Coherence: {result['coherence']:.2f}")
```

### Processing Books

Extract bonds from books and store in Neo4j:

```bash
# Process priority books (Jung + Mythology)
python -m storm_logos.scripts.process_books --priority

# Process a single book
python -m storm_logos.scripts.process_books --file /path/to/book.txt

# Test parser without Neo4j
python -m storm_logos.scripts.process_books --test /path/to/book.txt
```

### Trajectory Analysis

```python
from storm_logos.data.neo4j import get_neo4j
from storm_logos.applications.analyzer import Analyzer

neo = get_neo4j()
neo.connect()

# Get book trajectory
traj = neo.get_book_trajectory('jung_collected_papers', limit=1000)

# Analyze trajectory
analyzer = Analyzer()
result = analyzer.analyze_trajectory(traj)
print(f"Coherence: {result['coherence']:.3f}")
print(f"τ slope: {result['tau_slope']:.4f}")
```

## Neo4j Schema

```cypher
(:Author {name, era, domain})
(:Book {id, title, author, filename, genre, n_bonds, n_sentences})
(:Bond {id, adj, noun, A, S, tau})

(:Author)-[:WROTE]->(:Book)
(:Book)-[:CONTAINS {chapter, sentence, position}]->(:Bond)
(:Bond)-[:FOLLOWS {book_id, chapter, sentence, position}]->(:Bond)
```

## Processed Books

| Author | Books | Bonds | Genre |
|--------|-------|-------|-------|
| Carl Jung | 5 | 34,034 | Psychology |
| Homer | 2 | 11,680 | Epic |
| Ovid | 1 | 5,155 | Mythology |
| Thomas Bulfinch | 1 | 11,774 | Mythology |

**Total:** 42,172 unique bonds, 62,634 FOLLOWS edges

## Project Structure

```
storm_logos/
├── data/               # Layer 1: Data access
│   ├── postgres.py     # PostgreSQL (coordinates)
│   ├── neo4j.py        # Neo4j (trajectories)
│   ├── book_parser.py  # spaCy-based book parser
│   └── models.py       # Bond, Trajectory, etc.
├── semantic/           # Layer 2: Semantic operations
│   ├── storm.py        # Candidate explosion
│   ├── dialectic.py    # Thesis-antithesis filtering
│   ├── chain.py        # Chain reaction selection
│   └── physics.py      # RC dynamics, gravity
├── metrics/            # Layer 3: Metrics extraction
│   ├── extractors/     # Text → bonds
│   ├── analyzers/      # Coherence, irony, etc.
│   └── engine.py       # MetricsEngine
├── feedback/           # Layer 4: Error computation
│   ├── targets.py      # Homeostatic targets
│   └── engine.py       # FeedbackEngine
├── controller/         # Layer 6: PI control
│   ├── pi_controller.py
│   └── engine.py       # AdaptiveController
├── generation/         # Layer 5: Generation pipeline
│   ├── pipeline.py     # Storm→Dialectic→Chain
│   └── engine.py       # GenerationEngine
├── orchestration/      # Layer 7: Main loop
│   ├── loop.py
│   └── engine.py       # Orchestrator
├── applications/       # Layer 8: User-facing
│   ├── therapist.py    # Therapeutic agent
│   ├── generator.py    # Text generation
│   ├── analyzer.py     # Analysis
│   └── navigator.py    # Semantic navigation
├── cli/                # Command-line interface
│   └── main.py
└── scripts/            # Utility scripts
    └── process_books.py
```

## Theory

The system is based on the RC-Model theory of semantic dynamics:

1. **Semantic Space**: Words and concepts exist in a 3D space (A, S, τ)
2. **RC Dynamics**: State changes follow resistor-capacitor circuit behavior
3. **Gravity**: There's a natural drift toward concrete (low τ) and good (high A)
4. **Chain Reaction**: Coherent sequences emerge through resonance with history
5. **Homeostasis**: The system maintains balance through adaptive control

For detailed theory, see `rc_model/THEORY_V3.md`.

## License

Research use only.
