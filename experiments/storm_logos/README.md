# Storm-Logos: Adaptive Semantic Generation System

An 8-layer adaptive system for semantic text generation based on psychoanalytic and RC-model principles. Powers therapeutic AI agents with semantic state tracking.

## Overview

Storm-Logos implements a homeostatic text generation system that:

- **Navigates semantic space** defined by (A, S, τ) coordinates:
  - **A** (Affirmation): good ↔ bad axis
  - **S** (Sacred): elevated ↔ mundane axis
  - **τ** (Tau): abstraction level (0=concrete, 5=abstract)

- **Uses RC-circuit dynamics** for smooth state transitions
- **Implements Chain Reaction** for coherent bond selection via resonance
- **Adapts parameters** through PI control to maintain homeostatic targets
- **Supports multiple LLM backends**: Claude, Groq (Llama 70B), Ollama (local models)

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
pip install spacy neo4j numpy anthropic groq

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Environment Variables

Create a `.env` file with your API keys:

```bash
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

## Usage

### Therapy Sessions

Run automated therapy sessions with Claude as patient and various LLM backends as therapist:

```bash
# Using Groq Llama 70B (recommended - fast and high quality)
python scripts/therapy_test.py --model "groq:llama-3.3-70b-versatile" --turns 15

# Using Claude as therapist
python scripts/therapy_test.py --model claude --turns 10

# Using local Ollama model
python scripts/therapy_test.py --model mistral:7b --turns 5

# Interactive mode (you are the patient)
python scripts/therapy_test.py --interactive
```

Sessions are saved to `sessions/session_YYYYMMDD_HHMMSS.json` with full trajectory data.

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
(:Bond)-[:FOLLOWS {book_id, chapter, sentence, position, weight, last_used, source}]->(:Bond)
```

## Weight Dynamics (Learning & Forgetting)

FOLLOWS edges have weights that evolve over time through learning and forgetting dynamics,
modeled as a capacitor charging/discharging.

### Core Principle: Corpus is Permanent

**IMPORTANT:** Core knowledge from books (corpus) **never decays**. Only user-walked paths
from conversations are subject to forgetting:

```
CORPUS EDGES (source='corpus'):
    - Permanent weight = 1.0
    - Never decay
    - Established knowledge from books

USER EDGES (source='conversation' or 'context'):
    - Subject to forgetting
    - Must be reinforced to stay active
    - Represents user-specific learning
```

### The Capacitor Analogy

```
dw/dt = λ · (w_target - w)

Learning  = Charging capacitor    (voltage rises toward max)
Forgetting = Discharging capacitor (voltage falls toward baseline)

    w_max ──────────────────── Learning: w rises
         ╲                    ╱
          ╲    ← current →   ╱
           ╲                ╱
    w_min ──────────────────── Forgetting: w decays

The baseline is never zero. Something always remains.
```

### Formulas

**Forgetting (decay toward w_min) - USER EDGES ONLY:**
```
w(t+dt) = w_min + (w - w_min) · e^(-λ_forget · dt)
```

**Learning (simple reinforcement):**
```
w(t+1) = min(w_max, w + 0.05)
```

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| w_min | 0.1 | Floor - never fully forgotten |
| w_max | 1.0 | Ceiling - fully learned |
| λ_learn | 0.3 | Learning rate (per reinforcement) |
| λ_forget | 0.05 | Forgetting rate (per day) |

**Key insight:** Learning is 6× faster than forgetting. Easy to learn, slow to fully forget.

### Weight Sources

| Source | Initial Weight | Decays? | Rationale |
|--------|----------------|---------|-----------|
| Corpus (books) | 1.0 | **NO** | Permanent established knowledge |
| Conversation | 0.2 | Yes | User paths - needs reinforcement |
| Context-inferred | 0.1 | Yes | Weakest, most uncertain |

### Dormancy States (User Edges Only)

| State | Condition | Meaning |
|-------|-----------|---------|
| Active | w > 0.2 | Used in navigation |
| Dormant | w ≤ 0.2 | Exists but not actively used |
| Gone | NEVER | "Knowledge is never lost" |

Dormant paths can be reactivated through reinforcement.

### Half-Life

The half-life (time for weight to decay halfway to w_min) is:
```
t_half = ln(2) / λ_forget ≈ 13.86 days
```

### Decay Examples

Starting from w = 1.0:

| Days | Weight | Status |
|------|--------|--------|
| 1 | 0.956 | Active |
| 7 | 0.763 | Active |
| 14 | 0.550 | Active |
| 30 | 0.323 | Active |
| 60 | 0.167 | Dormant |
| 100 | 0.107 | Dormant |

### Nightly Decay Job

Apply forgetting decay to edges:

```bash
# Preview decay (dry run)
python -m storm_logos.scripts.nightly_decay --dry-run

# Apply 1 day of decay
python -m storm_logos.scripts.nightly_decay

# Apply 7 days of decay (e.g., after a week offline)
python -m storm_logos.scripts.nightly_decay --days 7

# View statistics
python -m storm_logos.scripts.nightly_decay --stats

# Initialize weights on existing edges
python -m storm_logos.scripts.nightly_decay --init-weights

# Show dynamics parameters
python -m storm_logos.scripts.nightly_decay --info
```

### Cron Setup

```bash
# Install cron job (runs at 3:00 AM daily)
./scripts/setup_cron.sh install

# Check status
./scripts/setup_cron.sh status

# Remove cron job
./scripts/setup_cron.sh remove
```

### Python API

```python
from storm_logos.data.neo4j import get_neo4j
from storm_logos.data.weight_dynamics import decay_weight, learn_weight_simple

neo = get_neo4j()
neo.connect()

# Reinforce an edge (learning)
neo.reinforce_transition(source_bond, target_bond)

# Apply decay
stats = neo.apply_decay(days_elapsed=1.0, dry_run=False)
print(f"Decayed {stats['edges_affected']} edges")

# Get decay statistics
stats = neo.get_decay_stats()
print(f"Active edges: {stats['active_count']}")
print(f"Dormant edges: {stats['dormant_count']}")

# Weight calculations
from storm_logos.data.weight_dynamics import decay_weight, time_to_dormancy

w_after_week = decay_weight(1.0, days_elapsed=7)  # 0.763
days_until_dormant = time_to_dormancy(1.0)  # ~44 days
```

## Runtime Bond Learning

The system can learn new bonds during conversations, storing them in PostgreSQL and
syncing to Neo4j for trajectory navigation.

### Learning Flow

```
User Input → spaCy → Bonds → Coordinates → PostgreSQL → Neo4j
                                  ↓
                         estimate if unknown
```

### PostgreSQL Tables

```sql
-- Learned bonds from conversations
learned_bonds:
    id, adj, noun, A, S, tau, source, confidence,
    created_at, last_used, use_count

-- Learned word coordinates
learned_words:
    word, A, S, tau, source, confidence,
    created_at, last_used
```

### Initialize Learning Tables

```python
from storm_logos.data.postgres import get_data

data = get_data()
data.init_learning_tables()  # Creates tables if they don't exist
```

### Learning from Text

```python
from storm_logos.data.bond_learner import BondLearner

learner = BondLearner()
learner.connect()

# Learn from text
result = learner.learn_from_text("The dark forest held ancient secrets.")
print(result.summary())  # "Learned 2 bonds: 2 new, 0 reinforced, 1 edges"

# Learn from conversation turn (links to previous)
result = learner.learn_turn(
    text="The mysterious path led deeper.",
    conversation_id="conv_123",
    previous_bonds=result.bonds
)

# Get learning statistics
stats = learner.get_stats()
print(f"Learned bonds: {stats['postgresql']['n_learned_bonds']}")
```

### Coordinate Estimation

When words are not in the corpus, coordinates are estimated using heuristics:

| Pattern | Effect |
|---------|--------|
| Negative prefix (un-, dis-, anti-) | A decreases |
| Positive suffix (-ful, -ive, -ous) | A increases |
| Abstract suffix (-ness, -ity, -tion) | τ increases |
| Sacred words (god, soul, spirit) | S and τ increase |

```python
# Estimate coordinates for unknown word
A, S, tau = data.estimate_word_coordinates("unhappiness")
# A=-0.3 (negative prefix), tau=3.0 (abstract suffix)
```

### Conversation Trajectories

Learned bonds are connected in Neo4j with source='conversation':

```python
from storm_logos.data.neo4j import get_neo4j

neo = get_neo4j()
neo.connect()

# Learn a trajectory
neo.learn_trajectory(
    bonds=[bond1, bond2, bond3],
    conversation_id="conv_123",
    source_type='conversation'
)

# Get conversation trajectory
trajectory = neo.get_conversation_trajectory("conv_123")

# Get learning stats
stats = neo.get_learning_stats()
print(f"Learned edges: {stats['learned_edges']}")
print(f"Conversations: {stats['conversations']}")
```

### Convenience Functions

```python
from storm_logos.data.bond_learner import learn_bond, learn_from_text

# Quick learning
bond = learn_bond("dark", "forest")
result = learn_from_text("The ancient temple was mysterious.")
```

## Processed Books

| Author | Books | Bonds | Genre |
|--------|-------|-------|-------|
| Carl Jung | 5 | 34,034 | Psychology |
| Homer | 2 | 11,680 | Epic |
| Ovid | 1 | 5,155 | Mythology |
| Thomas Bulfinch | 1 | 11,774 | Mythology |
| James George Frazer | 1 | 20,046 | Comparative Religion |
| Lewis Spence | 1 | 7,290 | Mythology |
| John Fiske | 1 | 4,474 | Mythology |
| Andrew Lang | 1 | 2,719 | Mythology |
| Thomas William Doane | 1 | 10,987 | Comparative Religion |

**Total:** 14 books, 64,167 unique bonds, 108,154 FOLLOWS edges

## Project Structure

```
storm_logos/
├── data/                   # Layer 1: Data access
│   ├── postgres.py         # PostgreSQL (coordinates + learned bonds)
│   ├── neo4j.py            # Neo4j (trajectories + weight dynamics)
│   ├── weight_dynamics.py  # Learning/forgetting formulas
│   ├── bond_learner.py     # Runtime bond learning from conversations
│   ├── book_parser.py      # spaCy-based book parser
│   └── models.py           # Bond, Trajectory, etc.
├── semantic/               # Layer 2: Semantic operations
│   ├── storm.py            # Candidate explosion
│   ├── dialectic.py        # Thesis-antithesis filtering
│   ├── chain.py            # Chain reaction selection
│   └── physics.py          # RC dynamics, gravity
├── metrics/                # Layer 3: Metrics extraction
│   ├── extractors/         # Text → bonds
│   ├── analyzers/          # Coherence, irony, etc.
│   └── engine.py           # MetricsEngine
├── feedback/               # Layer 4: Error computation
│   ├── targets.py          # Homeostatic targets
│   └── engine.py           # FeedbackEngine
├── controller/             # Layer 6: PI control
│   ├── pi_controller.py
│   └── engine.py           # AdaptiveController
├── generation/             # Layer 5: Generation pipeline
│   ├── pipeline.py         # Storm→Dialectic→Chain
│   └── engine.py           # GenerationEngine
├── orchestration/          # Layer 7: Main loop
│   ├── loop.py
│   └── engine.py           # Orchestrator
├── applications/           # Layer 8: User-facing
│   ├── therapist.py        # Therapeutic agent
│   ├── generator.py        # Text generation
│   ├── analyzer.py         # Analysis
│   └── navigator.py        # Semantic navigation
├── cli/                    # Command-line interface
│   └── main.py
├── scripts/                # Utility scripts
│   ├── process_books.py    # Book → Neo4j loader
│   ├── therapy_test.py     # Therapy session runner
│   ├── nightly_decay.py    # Forgetting job (cron)
│   └── setup_cron.sh       # Cron setup helper
├── sessions/               # Saved therapy sessions (JSON)
└── tests/                  # Unit tests
    ├── test_weight_dynamics.py  # Weight dynamics tests
    └── test_bond_learning.py    # Bond learning tests
```

## Session Analysis

Therapy sessions can be analyzed for emotional trajectories:

```python
import json
from storm_logos.data.neo4j import get_neo4j

# Load session
with open('sessions/session_20260108_203922.json') as f:
    session = json.load(f)

# Analyze trajectory
states = [t['state'] for t in session['turns']]
for i, s in enumerate(states, 1):
    print(f"Turn {i}: A={s['A']:+.2f}, irony={s['irony']:.0%}")

# Compare to corpus in Neo4j
neo = get_neo4j()
neo.connect()

# Find similar emotional descents in books
query = """
MATCH path = (b1:Bond)-[:FOLLOWS]->(b2:Bond)-[:FOLLOWS]->(b3:Bond)
WHERE b1.A > 0.5 AND b3.A < 0.3 AND b2.A < b1.A
RETURN b1.adj + ' ' + b1.noun as step1,
       b2.adj + ' ' + b2.noun as step2,
       b3.adj + ' ' + b3.noun as step3
LIMIT 5
"""
result = neo._driver.execute_query(query)
```

### Session JSON Format

```json
{
  "session_id": "20260108_203922",
  "turns": [
    {
      "turn": 1,
      "patient": "I don't know why I'm here...",
      "therapist": "You keep saying 'fine'...",
      "state": {"A": 0.65, "S": -0.07, "irony": 0.0},
      "metrics": {"coherence": 1.0, "tension": 0.55}
    }
  ],
  "summary": {
    "avg_A": 0.29,
    "avg_irony": 0.13,
    "A_movement": -0.46
  }
}
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
