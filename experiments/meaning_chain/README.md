# Meaning Chain: Storm-Logos Semantic Navigation

> "The storm of thoughts finds its logos in the structure of meaning"

A semantic navigation system inspired by biological cognition. When humans process questions, the neocortex fires chaotically (storm), then patterns emerge through meaning structure (logos). This system replicates that process.

## Core Architecture: Storm-Logos

```
Query → STORM (chaotic walks) → LOGOS (focus lens) → Focused Tree → Response
             ↓                        ↓
      neocortex firing         meaning structure
      (probabilistic)          (j-good, intent, tau)
```

### Why Not Brute Force?

Traditional approach: Generate multiple responses, compare, pick best.

**Problem**: This is computational brute force, not how minds work.

**Storm-Logos**:
- Storm: Let thoughts spread chaotically from seed concepts
- Logos: Focus through meaning lens (like light through optical lens)
- Result: ONE coherent response from principled emergence

## The Lens Metaphor

Logos acts as a **lens** that focuses chaotic thoughts:

```python
def focus_score(thought, intent_j):
    score = thought.activation

    # Goodness lens: prefer positive g
    score *= (0.5 + 0.5 * normalize(thought.g))

    # J-good lens: alignment with "the good"
    score *= (0.5 + 0.5 * cos(thought.j, j_good))

    # Intent lens: alignment with user's intent
    score *= (0.5 + 0.5 * cos(thought.j, intent_j))

    return score
```

Only thoughts aligned with the lens pass through.

## Semantic Space

### J-Space (5D Transcendentals)

| Dimension | Description |
|-----------|-------------|
| beauty | Aesthetic quality |
| life | Vitality, animation |
| sacred | Spiritual significance |
| good | Moral quality |
| love | Relational warmth |

### Key Metrics

- **g (goodness)**: Projection onto j_good direction `[-1, +1]`
- **τ (tau)**: Semantic altitude `[1-6]` (human reality → transcendental)
- **Coherence**: How aligned the focused thoughts are `[0-100%]`
- **Convergence**: Where semantic paths meet (meaning anchor)

## Semantic Physics

The semantic space exhibits physics-like behavior. See `docs/UNIFIED_SEMANTIC_PHYSICS.md` for full theory.

### Semantic Gravity

```
Potential: φ = +λτ - μg·cos(j, j_good)

Where:
  λ = 0.5  (gravitational constant)
  μ = 0.5  (lift constant)
  τ = semantic altitude [1-6]
  g = goodness [-1, +1]
```

**Key insight**: Meaning naturally "falls" toward human reality (low τ) while goodness provides "lift" toward the transcendental.

```
τ=6  ☀️ TRANSCENDENTAL - Beyond ordinary experience
         ↑ TRANSCENDENCE requires WORK (against gravity)
τ≈3.5 ═══ THE VEIL (quasi-Lagrange point) ═══
         ↓ GROUNDING is NATURAL (with gravity)
τ=1  🌍 HUMAN REALITY - Common shared experience
```

### Boltzmann Selection (Storm Phase)
```
P_i = exp(-E_i / T) / Σ exp(-E_j / T)

Low T (0.5): Focused navigation (follows strongest edges)
High T (2.0): Exploratory navigation (random walks)
```

### Validated Properties

| Property | Finding |
|----------|---------|
| **Gravity** | Meaning flows downward (Δτ < 0 from high altitudes) |
| **Ground density** | 60%+ concepts at τ ≤ 3 (human reality is dense) |
| **Coherence** | Topologically protected (Φ ∈ [0.74, 0.91] across all T) |
| **Interference** | All constructive (meaning reinforces, never cancels) |
| **Optics** | Ground is thin (n=0.78), sky is dense (n=1.98) |

## Key Findings

### Coherence Patterns

From dialogue experiments between Meaning Chain and Claude:

| Topic | Coherence | Convergence Point | Learning |
|-------|-----------|-------------------|----------|
| Love transforming suffering | **99%** | "way" | No |
| Nature of meaning | **98%** | "way" | No |
| Shadow integration (before) | **72%** | "way" | No |
| Shadow integration (after) | **89%** | "place", "hold" | **Yes (4051 concepts)** |
| True understanding | **92%** | "veil", "place" | **Yes (4051 concepts)** |

**Insight**: Learning improves coherence and produces more meaningful convergence points.

### Learning Impact on Dialogue

Before/after learning comparison on shadow integration topic:

| Metric | Without Learning | With Learning |
|--------|-----------------|---------------|
| Coherence | 72% | 89% (+17%) |
| Convergence | "way" | "place", "hold" |
| Response depth | Generic | Context-aware |

The learned concepts (4,051 from books) enable richer semantic navigation.

### Emergent Convergence

When two agents discuss a topic, they often converge on the same concept through different paths:

```
Dialogue on "meaning and consciousness":
  Seeker: part (79%) → view (93%) → way (93%)
  Guide:  sense (86%) → way (88%)
                              ↑
                    Both converge on "way"
```

**Insight**: Semantic space has natural attractors where meaning crystallizes.

### The "Door" Discovery

In shadow integration dialogue, the system converged on "door" when discussing the moment of recognition - opening a door to the unconscious. This metaphor emerged naturally from semantic navigation, not from prompting.

```
[Storm-Logos]
  Convergence: door
  Core: ['time', 'instant', 'moment']
  Coherence: 91%

Response: "In that pivotal moment, open the 'door' of your mind..."
```

## Dual-Role Words

Words like "love", "dream", "help" function as both:
- **Nouns** (concepts): Seeds for storm phase
- **Verbs** (operators): Lens direction for logos phase

```python
# Input: "what does love truly mean"
Nouns: ['mean', 'love', 'true']  # concepts to explore
Verbs: ['love', 'mean']           # lens direction
```

## Usage

### Start Neo4j
```bash
cd config && docker-compose up -d
```

### Run Chat
```bash
python app/chat.py
```

### Run Dialogue (two semantic agents)
```bash
python app/dialogue.py --exchanges 5 --topic "What is meaning?"
```

### Run Dialogue with Claude
```bash
export ANTHROPIC_API_KEY="your-key"
python app/dialogue_claude.py --exchanges 5
```

### Run Dialogue Comparison (saves JSON + TXT)
```bash
export ANTHROPIC_API_KEY="your-key"
python scripts/run_dialogue_compare.py --exchanges 5 --topic "Your topic here"
```

Results saved to `results/dialogue_comparison/` for before/after analysis.

## Directory Structure

```
meaning_chain/
├── chain_core/
│   ├── storm_logos.py      # Storm-Logos architecture (main)
│   ├── decomposer.py       # Text → nouns + verbs
│   ├── renderer.py         # Tree → LLM prompt → response
│   ├── meditation.py       # Consciousness layer
│   └── feedback.py         # Response validation
│
├── graph/
│   ├── meaning_graph.py        # Neo4j with VIA relationships
│   ├── learning.py             # Entropy-based concept learning
│   └── conversation_learner.py # Learn from conversations
│
├── input/
│   └── book_processor.py   # Process books → SVO + learn concepts
│
├── experiments/
│   └── physics/
│       ├── semantic_gravity.py       # 6 validated gravity tests
│       ├── semantic_thermodynamics.py # Temperature, entropy, phase behavior
│       ├── semantic_optics.py        # Refraction, lens, interference
│       ├── storm_physics.py          # Dynamic physics observer
│       └── results/                  # Experiment outputs (JSON)
│
├── docs/
│   ├── UNIFIED_SEMANTIC_PHYSICS.md   # Complete physics theory
│   ├── SEMANTIC_THERMODYNAMICS.md    # Thermodynamics detail
│   └── SEMANTIC_OPTICS.md            # Optics detail
│
├── scripts/
│   ├── reprocess_books.py      # Reprocess all books with learning
│   └── run_dialogue_compare.py # Run & save dialogues for comparison
│
├── tests/
│   └── test_learning.py    # Learning system tests
│
├── app/
│   ├── chat.py             # Interactive chat
│   ├── dialogue.py         # Two semantic agents
│   └── dialogue_claude.py  # Semantic ↔ Claude dialogue
│
├── models/
│   └── types.py            # MeaningNode, MeaningTree
│
└── results/
    └── dialogue_comparison/  # Before/after dialogue results (JSON + TXT)
```

## Neo4j Schema

```
(:Concept {
    word: STRING,
    g: FLOAT,           // Goodness [-1, +1]
    tau: FLOAT,         // Abstraction [1, 7]
    j: LIST<FLOAT>,     // 5D transcendental vector

    // Learning properties (for learned concepts)
    learned: BOOLEAN,   // True if learned (not from corpus)
    variety: INT,       // Number of unique adjectives
    h_adj: FLOAT,       // Shannon entropy of adj distribution
    h_adj_norm: FLOAT,  // Normalized entropy [0, 1]
    confidence: FLOAT,  // Confidence [0.1, 1.0]
    n_observations: INT // Observation count
})

(:Adjective {word: STRING})

(:Concept)-[:VIA {verb, weight, count, source}]->(:Concept)
(:Concept)-[:DESCRIBED_BY {count, source}]->(:Adjective)
```

## Learning System

The meaning_chain now supports **learning new concepts** from books and conversations.

### Theory: Entropy-Based Learning

Concepts learn their parameters from adjective distributions:

```
τ = 1 + 5 × (1 - H_norm)

Where H_norm = H / log₂(variety)
      H = -Σ p(adj) log₂ p(adj)  (Shannon entropy)
```

| Entropy | τ | Meaning |
|---------|---|---------|
| High (many varied adjectives) | Low (1-3) | Common concept (human reality) |
| Low (few concentrated adjectives) | High (4-6) | Specific concept (transcendental) |

### Learning Pipeline

```
Text → Extract Adj-Noun pairs → Aggregate by noun → Compute τ, g, j → Store in Neo4j
                                    ↓
                            {noun: {adj: count}}
```

### What Gets Learned

1. **New words** not in original corpus → Creates new `:Concept` node
2. **Existing words** → Updates τ, g, j from new observations

### Usage: Book Processing with Learning

```python
from input.book_processor import BookProcessor

# Create processor with learning enabled
processor = BookProcessor(enable_learning=True)

# Optional: load adjective vectors for j-centroid computation
from core.data_loader import DataLoader
processor.load_adj_vectors(DataLoader())

# Process book - extracts SVO + learns concepts
result = processor.process_book("/path/to/book.txt")

print(f"SVO patterns: {result.svo_patterns}")
print(f"New concepts: {result.new_concepts_learned}")
print(f"Updated concepts: {result.existing_concepts_updated}")
```

### Usage: Conversation Learning

```python
from graph.conversation_learner import ConversationLearner

learner = ConversationLearner(enable_learning=True)

# Learn from exchange
stats = learner.learn_from_exchange(
    "What is the beautiful mystery of life?",
    "The profound mystery reveals itself through authentic connection."
)

print(f"Concepts learned: {stats['concepts_learned']}")
print(f"New: {stats['concepts_new']}, Updated: {stats['concepts_updated']}")
```

### Learning Hierarchy

| Source | Initial Weight | Description |
|--------|----------------|-------------|
| Corpus | 1.0 | Pre-computed from 928K books |
| Books | 0.8 | Processed individually |
| Conversation | 0.2 | Needs reinforcement |
| Context | 0.1 | Estimated from neighbors |

### Parameter Refinement

As more observations accumulate:
- Entropy stabilizes → τ converges to true value
- J-centroid refines → g becomes more accurate
- Confidence increases → concept becomes "known"

### Reprocessing Books

After updating the learning system:

```python
# Reprocess all books to update learned concepts
processor = BookProcessor(enable_learning=True)

for book in books:
    result = processor.reprocess_book(book)
    print(f"{book}: {result.new_concepts_learned} new, {result.existing_concepts_updated} updated")
```

## Processed Books

The semantic graph is populated with SVO patterns extracted from:

| Book | Patterns | Context |
|------|----------|---------|
| King James Bible | 4840 | Sacred language, parables, wisdom |
| Jung, Psychology of the Unconscious | 2152 | Depth psychology, symbols |
| Jung, Memories Dreams Reflections | 1344 | Personal unconscious, individuation |
| Jung, Four Archetypes | 721 | Mother, rebirth, spirit, trickster |
| Breath of Love (Nerim) | 90 | Paradoxes, truth, love |

These books shape how the system navigates meaning - Jung's archetypes influence shadow dialogues, Biblical language enriches discussions of love and suffering.

### Process New Books
```bash
python input/book_processor.py /path/to/book.txt --id "Book Name"
```

## Theory: Why It Works

### Meaning as Convergence

When multiple semantic paths converge on a concept, that concept becomes a **meaning anchor**. The system finds these anchors through:

1. **Storm**: Many parallel walks create activation patterns
2. **Logos**: Lens filters to intent-aligned, good-aligned concepts
3. **Convergence**: Most-activated non-seed concept = meaning anchor

### The Lens Selects Truth

The logos lens has three components:
- **Goodness** (g): Ethical/aesthetic direction
- **J-good**: Alignment with transcendental good
- **Intent**: User's purpose (from verbs)

Thoughts must pass all three to emerge in the pattern. This naturally filters toward coherent, meaningful responses.

### Coherence as Quality Metric

High coherence (>90%) means the focused thoughts align well in j-space. This correlates with:
- More insightful responses
- Natural metaphor emergence ("door" for shadow work)
- Genuine semantic connections (not forced associations)

## Configuration

```python
ChatConfig(
    storm_temperature=1.5,    # Chaos in storm phase
    n_walks=5,                # Parallel walks per seed
    steps_per_walk=8,         # Depth of each walk
    temperature=0.7,          # LLM temperature
)
```

## Connection to Experience Knowledge

| Property | Experience Knowledge | Meaning Chain |
|----------|---------------------|---------------|
| Neo4j Port | 7687 | 7688 |
| Edge Type | TRANSITION | VIA |
| Navigation | Quantum tunneling | Storm-Logos |
| Consciousness | Full layer | Meditation + Prayer |
