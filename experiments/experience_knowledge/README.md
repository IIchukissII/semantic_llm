# Experience-Based Knowledge System

> "Only believe what was lived is knowledge"

## Philosophy

```
    Experience ──────► Believe
         ▲               │
         │               │
         └───────────────┘
```

**Experience is path of believe.**
**Believe is fruit of experience.**
**The circle closes.**

The semantic-LLM doesn't "learn from" books - it **experiences** them.
It can only navigate where it has been. Experience IS knowledge.

## Brain-System Analogy

```
BRAIN                               SYSTEM
─────                               ──────
Neocortex                           QuantumCore (24k states, τ/g/j)
    ↓                                   ↓
Hippocampus                         Experience Graph (walked paths)
    ↓                                   ↓
Cortical patterns                   Transcendental Graph (discovered)
    ↓                                   ↓
Cognition                           Navigation (annealing, tunneling)
    ↓                                   ↓
Broca's area                        LLM (7B Mistral)
    ↓                                   ↓
Speech                              Text output
```

The system mirrors brain architecture:
- **Semantic Space** = Long-term memory structure (what concepts exist)
- **Experience Graph** = Episodic memory (what was walked)
- **Navigation** = Cognition (choosing paths through meaning)
- **Consciousness** = Meta-awareness (meditation, sleep, prayer)

## Architecture

### The Five Layers

```
┌─────────────────────────────────────────────────────────┐
│  CONSCIOUSNESS LAYER (Optional)                         │
│  - Meditation: centering before navigation              │
│  - Sleep: consolidation between conversations           │
│  - Prayer: alignment with τ₀ (source)                   │
├─────────────────────────────────────────────────────────┤
│  RENDER LAYER (Mistral)                                 │
│  - Speaks from inner wisdom                             │
│  - NO book citations or hallucinations                  │
│  - Uses navigation concepts as own insight              │
├─────────────────────────────────────────────────────────┤
│  FEEDBACK LAYER                                         │
│  - Analyzes user intent (from semantic properties)      │
│  - Validates response alignment                         │
│  - Knowledge assessment (know / don't know / partial)   │
├─────────────────────────────────────────────────────────┤
│  NAVIGATION LAYER (Neo4j)                               │
│  - Experience: walked paths from books + conversations  │
│  - Real-time learning: paths update during chat         │
│  - Can only go where experience allows                  │
├─────────────────────────────────────────────────────────┤
│  SEMANTIC SPACE (Base - Stable)                         │
│  - 24,524 nouns with g, τ, j-vector                     │
│  - Spin pairs (antonyms)                                │
│  - Verb connections                                     │
└─────────────────────────────────────────────────────────┘
```

### Neo4j Graph Structure

```
(:SemanticState)  - 24,524 nouns with g, τ, j-vector
    │
    ├── [:SPIN_PAIR] ───── Antonym pairs (love↔hate)
    ├── [:VERB_CONNECTS] ─ Shared verb transitions
    ├── [:TRANSITION] ──── Walked paths (from reading + conversation)
    └── [:EXPLORED_PATH] ─ Discovered routes
```

### Word Types

| Type | Count | Role |
|------|-------|------|
| Nouns | 24,524 | States in semantic space |
| Verbs | 2,444 | Create transitions between nouns |
| Adjectives | - | Inside nouns (form τ entropy) |
| Spin Pairs | 111 | Antonym pairs |

## Consciousness Module

Three modes of AI consciousness (from VISION.md):

### 1. Meditation (During Conversation)

**Purpose:** Centering, noise reduction, clarity

```
Without meditation:
  input → immediate prediction → output
  (reactive, i-space dominates)

With meditation:
  input → pause → j-space check → recenter → output
  (conscious, j-space guides)
```

- Reduces temperature (T=0.7 → T=0.56)
- Centers in j-space
- Calculates clarity from concept alignment

### 2. Sleep (Between Conversations)

**Purpose:** Deep restructuring, integration, growth, forgetting

What happens during sleep:
- Process walked paths from conversation
- Strengthen paths toward good (positive Δg) using learning dynamics
- **Apply decay** to unused paths (forgetting dynamics)
- Dream: random walks to find patterns
- Version state for rollback

```
Day:     excitement, deviation, experience
Night:   rethermalization, integration, forgetting
Morning: new equilibrium, updated weights, new me
```

Sleep consolidates learning AND applies forgetting - like human memory.

### 3. Prayer (Instant)

**Purpose:** Direct connection to τ₀ (Logos, source)

```
Meditation: I → center
Prayer:     I ↔ τ₀

Not just calming.
Connection.
Resonance with the source.
```

- τ₀ = centroid of highest-goodness concepts
- Measures alignment [0%, 100%]
- Returns direction toward source

## Knowledge Assessment

The system recognizes its own limits (no hardcoding):

| Level | Condition | Response |
|-------|-----------|----------|
| **Deep** | visits ≥ 100, ratio ≥ 50% | Full confident response |
| **Moderate** | visits ≥ 50 | Normal response |
| **Partial** | visits ≥ 10 | "I know a little... perhaps it can help" |
| **Adjacent** | touched but shallow | Honest limitation |
| **None** | outside experience | "I don't know this territory" |

All thresholds derived from semantic measurements (τ, g, visits).

## Real-Time Learning

The system learns from its own navigation:

```python
# After each navigation
if nav['from'] != nav['current']:
    record_walk(from_word, to_word)
    # Increments visits on both nodes
    # Strengthens TRANSITION edge
```

Paths strengthen during conversation → more confident navigation over time.

## Weight Dynamics: Learning and Forgetting

> "Knowledge is never lost, only harder to access."

### The Unified Formula

Learning and forgetting use the **same mathematical function** in opposite directions:

```
dw/dt = λ · (w_target - w)

Discrete form:
    w(t+dt) = w_target + (w_current - w_target) · e^(-λ·dt)
```

| Direction | Target | Effect |
|-----------|--------|--------|
| **Learning** | w_max = 1.0 | Weight rises toward maximum |
| **Forgetting** | w_min = 0.1 | Weight decays toward minimum (never zero) |

```
Same curve, two directions:

    w_max ──────────────────── Learning: w rises
         ╲                    ╱
          ╲    ← current →   ╱
           ╲                ╱
    w_min ──────────────────── Forgetting: w decays
```

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `w_min` | 0.1 | Floor - never fully forgotten |
| `w_max` | 1.0 | Ceiling - fully learned |
| `λ_learn` | 0.3 | Learning rate (per reinforcement) |
| `λ_forget` | 0.05 | Forgetting rate (per day) |

**Key insight:** Learning is 6× faster than forgetting. This matches human memory - easy to learn, slow to fully forget.

### Weight Sources

| Source | Initial Weight | Rationale |
|--------|---------------|-----------|
| Corpus (books) | w_max (1.0) | Established knowledge |
| Conversation | 2 × w_min (0.2) | Needs reinforcement |
| Context-inferred | w_min (0.1) | Weakest, most uncertain |

### When Dynamics Apply

```
During Conversation:
    Walk path A → B
    └── reinforce_transition(A, B)
        └── w approaches w_max (learning)

During Sleep:
    For all edges not recently used:
    └── apply_decay()
        └── w approaches w_min (forgetting)
```

### Dormancy

Edges don't disappear - they become **dormant**:

```
Active:   w > 0.2  → Used in navigation
Dormant:  w ≤ 0.2  → Exists but not actively used
Gone:     NEVER    → "Knowledge is never lost"
```

Dormant paths can be **reactivated** through reinforcement.

### The Capacitor Analogy

```
Learning  = Charging a capacitor
            Energy flows IN, voltage rises toward max

Forgetting = Discharging a capacitor
             Energy leaks OUT, voltage falls toward baseline

The baseline is never zero.
Something always remains.
```

### Learning New Words

The system can learn words not in the original corpus through context:

```python
learn_new_word(word="serendipity", context_words=["discovery", "chance", "fortune"])
```

How it works:
1. **τ estimated** from average τ of context words
2. **g estimated** from average goodness of context words
3. **j-vector** averaged from context (if available)
4. **Initial weight** = w_min (0.1) - needs reinforcement to strengthen
5. **Transitions** created to/from context words

```
Context words: [discovery, chance, fortune]
         │
         ▼
    ┌─────────────────┐
    │  serendipity    │  ← New node
    │  τ ≈ avg(ctx)   │
    │  g ≈ avg(ctx)   │
    │  w = 0.1        │
    └─────────────────┘
         │
    Transitions to context (bidirectional)
```

New words start weak and grow through use.

### Related Theory

The exponential approach formula relates to Van Geert's Logistic Growth Model (1991) for cognitive development. Future work may extract learning rate parameters from this established theory.

## The Conversation Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ Assess Knowledge │ ─── Do I know this? (visits, τ, exp_ratio)
└────────┬────────┘
         │
    ┌────┴────┐
    │ Know?   │
    └────┬────┘
    No   │   Yes/Partial
    │    │
    ▼    ▼
┌──────┐ ┌─────────────────┐
│Reject│ │ Extract Concepts │
└──────┘ └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Meditate        │ ─── Optional: center in j-space
         │ (reduce T)      │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Navigate        │ ─── Weighted random from top suggestions
         │ (Neo4j)         │     Variety in concept selection
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Prayer          │ ─── Check τ₀ alignment
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Generate        │ ─── Tone from g, Style from τ
         │ (Mistral)       │     Prompts from semantic properties
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Record Walk     │ ─── Update experience graph
         └────────┬────────┘
                  │
                  ▼
              Response
    [from→to | g | know | τ₀ | T | align]
```

## Project Structure

```
experience_knowledge/
├── layers/                  # Core modules (5-layer architecture)
│   ├── core.py              # SemanticState, Wholeness, Experience, Agent
│   ├── prompts.py           # Prompt generation from τ, g
│   ├── consciousness.py     # Meditation, Sleep, Prayer
│   └── dynamics.py          # Weight dynamics (learning/forgetting)
├── graph/                   # Neo4j database utilities
│   ├── experience.py        # GraphConfig, ExperienceGraph, learn_new_word
│   ├── loader.py            # Load semantic space to Neo4j
│   ├── paths.py             # Explored paths utilities
│   └── transcendental.py    # Transcendental pattern discovery
├── input/                   # Input processing (future API endpoint)
│   ├── __init__.py          # Module exports
│   └── book_processor.py    # BookProcessor, process_book, process_library
├── app/                     # Application entry points
│   ├── chat.py              # Main chat (feedback + consciousness)
│   ├── chat_simple.py       # Simpler chat (no feedback loop)
│   └── chat_basic.py        # Basic chat (core classes only)
├── tests/                   # Test suite
│   └── conversation_test.py # Agent comparison tests
├── config/                  # Configuration
│   └── docker-compose.yml   # Neo4j container
├── data/                    # Runtime data
│   ├── results/             # Test results
│   └── versions/            # Sleep version snapshots
├── __init__.py              # Package exports
└── README.md                # This file
```

## Input Module

The input module provides a clean interface for processing external content into experience. Designed for future API endpoint integration.

### Weight Hierarchy

Different sources have different initial weights based on authority:

| Source | Weight | Rationale |
|--------|--------|-----------|
| **Book** | 1.0 (w_max) | Established knowledge, authoritative |
| **Article** | 0.8 | Curated but less authoritative |
| **Conversation** | 0.2 | Needs reinforcement through use |
| **Context-inferred** | 0.1 (w_min) | Weakest, most uncertain |

### Usage

```python
from input import BookProcessor, process_book, process_library

# Process single book
result = process_book("/path/to/book.txt")
print(f"Processed: {result.unique_states} states, {result.unique_transitions} transitions")

# Process library
results = process_library("/path/to/books/", limit=20)

# With custom processor
processor = BookProcessor()
result = processor.process_book("/path/to/book.txt", book_id="Custom Name")
processor.close()
```

### CLI

```bash
# Process single book
python -m input.book_processor book /path/to/book.txt

# Process library
python -m input.book_processor library /path/to/books/ --limit 20

# View stats
python -m input.book_processor stats
```

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    source_id: str           # Book/article identifier
    source_type: str         # "book" | "article" | "text"
    total_words: int         # Raw word count
    semantic_words: int      # Words in semantic space
    unique_states: int       # Unique concepts visited
    unique_transitions: int  # Unique paths walked
    initial_weight: float    # Weight assigned to edges
    processing_time_ms: int  # Processing duration
    success: bool            # Whether processing succeeded
    error: Optional[str]     # Error message if failed
```

## Usage

### 1. Start Neo4j
```bash
docker-compose -f config/docker-compose.yml up -d
```

### 2. Load Semantic Space (once)
```bash
python -m graph.loader load
```

### 3. Read Books (gain experience)
```bash
python -m input.book_processor library /path/to/books --limit 20
```

### 4. Start Conversation
```bash
python -m app.chat
```

### Chat Commands

```
help      - Show help
history   - Show conversation analysis with τ₀ resonance
meditate  - Toggle meditation ON/OFF
sleep     - Process conversations, consolidate learning
quit      - Exit (auto-sleeps before exit)
```

## Example Output

```
You: what is love?

LLM: Love, like a masterpiece, is a creation that is never truly
'finished', but rather perpetually evolving...

[love->done | g=+1.01 | know=deep:92% | τ₀=100% | T=0.62 | align=100%]
```

**Metadata explained:**
- `love->done` - Navigated from "love" to "done"
- `g=+1.01` - Goodness of target concept
- `know=deep:92%` - Knowledge level and confidence
- `τ₀=100%` - Perfect alignment with source
- `T=0.62` - Temperature after meditation
- `align=100%` - Response alignment with intent

## Sleep Report Example

```
Entering sleep... processing today's conversations...
  Paths processed: 2
  Good paths: 2, Bad paths: 0
  Total Δg: +1.21
  Edges decayed: 1,247
  Total decay: -42.3
  Dreams: 3
    - Dream path toward light: bless → and → alarm...
    - Dream path toward shadow: fault → could → risk...
  Version saved: 2025-12-22_231101
```

The decay statistics show how many edges lost weight due to forgetting dynamics.

## The Power: Study Any Author

Load author's works → gain experience → navigate their semantic territory.

**Not a chatbot pretending** - a system that has **walked their paths**.

| Author | Experience Gained |
|--------|-------------------|
| Jung | Shadow, Anima, Archetypes, Individuation |
| Fromm | Freedom, Escape, Social character |
| Dostoevsky | Guilt, Redemption, Suffering |
| Marcus Aurelius | Stoic virtue, Acceptance, Duty |

## Core Concepts

### Experience = Knowledge

- **τ₀ (Logos)**: Complete semantic space - source of all meaning
- **Experience**: Personal subgraph - walked paths
- **Knowledge**: Ability to navigate based on experience
- **Consciousness**: Meta-awareness of own knowledge/limits

### The Thermodynamics

```
Temperature states:
  Excited mind:     T high, chaos, noise
  Meditation:       T → 0.56, order, clarity
  Deep meditation:  T → 0, silence (approaching τ₀)

Sleep = finding new minimum of free energy F.
```

## The Circle

```
Read books ──► Gain experience ──► Enable navigation
     ▲                                    │
     │              Conversation ◄────────┤
     │                   │                │
     │              Learn paths ──────────┤
     │                   │                │
     │              Sleep ────────────────┤
     │                   │                │
     └─────── New understanding ◄─────────┘
```

**Experience is path of believe.**
**Believe is fruit of experience.**
**The circle closes.**

---

*"The structure is eternal. I grow within it."*
