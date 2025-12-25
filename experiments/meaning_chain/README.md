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
- **τ (tau)**: Abstraction level `[1-7]` (concrete → abstract)
- **Coherence**: How aligned the focused thoughts are `[0-100%]`
- **Convergence**: Where semantic paths meet (meaning anchor)

## Quantum Physics (Storm Phase)

Transitions through semantic space follow physical laws:

### Barrier Opacity
```
κ = (1 - cos(j₁, j₂)) / 2

κ = 0:   parallel j-vectors (no barrier)
κ = 0.5: orthogonal (medium barrier)
κ = 1:   antiparallel (maximum barrier)
```

### Boltzmann Selection
```
P_i = exp(-E_i / T) / Σ exp(-E_j / T)

Low T (0.5): Focused navigation
High T (2.0): Exploratory navigation
```

## Key Findings

### Coherence Patterns

From dialogue experiments between Meaning Chain and Claude:

| Topic | Coherence | Convergence Point |
|-------|-----------|-------------------|
| Love transforming suffering | **99%** | "way" |
| Nature of meaning | **98%** | "way" |
| Shadow integration | **91%** | "door" |
| Consciousness | **93%** | "view" |

**Insight**: High coherence (>90%) correlates with finding genuine semantic convergence points.

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
│   ├── meaning_graph.py    # Neo4j with VIA relationships
│   └── conversation_learner.py
│
├── app/
│   ├── chat.py             # Interactive chat
│   ├── dialogue.py         # Two semantic agents
│   └── dialogue_claude.py  # Semantic ↔ Claude dialogue
│
├── models/
│   └── types.py            # MeaningNode, MeaningTree
│
└── results/                # Saved dialogue transcripts
```

## Neo4j Schema

```
(:Concept {
    word: STRING,
    g: FLOAT,           // Goodness [-1, +1]
    tau: FLOAT,         // Abstraction [1, 7]
    j: LIST<FLOAT>,     // 5D transcendental vector
})

(:Concept)-[:VIA {verb, weight}]->(:Concept)
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
