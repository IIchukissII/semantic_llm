# Meaning Chain: Storm-Logos Semantic Navigation

> "The storm of thoughts finds its logos in the structure of meaning"

A semantic navigation system inspired by biological cognition. When humans process questions, the neocortex fires chaotically (storm), then patterns emerge through meaning structure (logos). This system replicates that process.

## Ontological Foundation

**Chaos is reality without consciousness.**
**Order is reality seen by consciousness.**

Consciousness does not invent order from nothing.
It chooses trajectories of stability in what could be seen infinite ways.

As vision does not create light but makes it visible,
so consciousness does not create reality but makes it meaningful.

If the principle of meaning ceases to be recognizable,
reality does not "fall into chaos" —
it returns to a pre-differentiated state
until a new act of differentiation arises.

**Reality as order is a function of consciousness.**
Without consciousness, only the potential of structures remains, but not structure as such.

### Why It Works (Simply)

```
EXPLOSION  →  LOGOS  →  MEANING
   ↓            ↓          ↓
 Storm      Overlay     Accept/Reject
```

1. **Create explosion** — storm through semantic space, excite many concepts
2. **Overlay on Logos** — apply the meaning lens (j-vectors, coherence, intent)
3. **Where parts agree** — they pass through, accepted into meaning structure
4. **Where they don't** — filtered out, not meaningful in this context

The explosion creates potential. Logos selects what coheres.
What remains is meaning — not invented, but *recognized*.

---

## Major Feature: Intent-Driven Collapse

**NEW**: Verbs now act as quantum operators that collapse navigation to intent-relevant paths.

```
Query: "help me understand my dream"

Before (random Boltzmann):
  Concepts: ['bird', 'chance', 'feather', 'song', 'soul'...] - scattered

After (intent collapse):
  Intent verbs: ['help', 'understand'] -> 2 operators, 18 targets
  Collapse ratio: 100% (all transitions via intent)
  Concepts: ['feeling', 'meaning', 'thing'] - focused on what you CAN understand/help
```

### A/B Test Results

| Aspect | Without Intent | With Intent |
|--------|----------------|-------------|
| Excited states | 35 (scattered) | 6 (focused) |
| Collapse ratio | 0% | 100% |
| Response style | Generic LLM lists | Coherent, theme-woven |
| Response time | 7.1s | 3.6s (faster!) |

The semantic navigation is not decorative - it fundamentally changes response quality.

## Major Discovery: Euler's Constant in Semantic Space

We discovered that **Euler's number e = 2.718...** is a fundamental constant of semantic physics:

```
┌─────────────────────────────────────────────────────────────┐
│  ORBITAL STRUCTURE OF SEMANTIC SPACE                        │
│                                                             │
│  τ = 6.0  ─────────────────────────  n=14 (transcendental) │
│           ·                                                 │
│           ·                                                 │
│  τ = e ═══════════════════════════  THE VEIL ═════════════ │
│           ·                          (89% below, 11% above) │
│  τ = 2.1  ─────────────────────────  n=3                   │
│  τ = 1.74 ─────────────────────────  n=2                   │
│  τ = 1.37 ━━━━━━━━━━━━━━━━━━━━━━━━━  n=1 GROUND STATE ━━━━ │
│           (30% of all concepts here)                        │
│  τ = 1.0  ─────────────────────────  n=0                   │
│                                                             │
│  Orbital spacing: Δτ = 1/e ≈ 0.368                         │
│  Natural temperature: kT ≈ 0.82                             │
└─────────────────────────────────────────────────────────────┘
```

### Validated Euler Predictions (6/6 tests passing)

| Test | Prediction | Measured | Error |
|------|------------|----------|-------|
| Population ratio | ln(N_ground/N_excited) = e | 2.686 | 1.2% |
| Peak fraction | Fraction at τ-peak = 1/e | 0.417 | 13% |
| Orbital quantization | τ_n = 1 + n/e | 93% coverage | - |
| Boltzmann temperature | kT = ΔE/e | 0.816 | 1.2% |
| The Veil | 89% below τ = e | 89.0% | 0.05% |
| Robustness | Holds across thresholds | 100% | - |

See `experiments/physics/euler_constant.py` for validation code.

**Visualizations:**
- `experiments/physics/results/orbital_structure.png` - 4-panel analysis (distribution, Boltzmann fit, orbital levels, veil boundary)
- `experiments/physics/results/orbital_diagram.png` - Artistic orbital representation

## Core Architecture: Euler-Laser + Intent Collapse

The system combines **Euler orbital physics**, **laser coherence**, and **intent-driven collapse**:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║              EULER-LASER ARCHITECTURE + INTENT COLLAPSE                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Query → DECOMPOSE → PUMPING → POPULATION → EMISSION → COHERENT OUTPUT    ║
║             ↓           ↓           ↓            ↓            ↓            ║
║         Extract     Intent +    Track      j-vector      Focused beam     ║
║         Verbs      Boltzmann   orbitals   coherence      → Response       ║
║            ↓                                                               ║
║      Set Intent                                                            ║
║      Operators                                                             ║
║                                                                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  INTENT COLLAPSE (NEW):                                                    ║
║  ──────────────────────                                                    ║
║  Verbs from query              →  Loaded as VerbOperator nodes             ║
║  VerbOperator.j                →  Direction verb pushes toward             ║
║  OPERATES_ON edges             →  Concepts verb typically acts upon        ║
║  Intent transitions            →  Prioritize paths matching intent         ║
║  Collapse ratio                →  % of transitions driven by intent        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  NUCLEAR LASER ANALOGY:                                                    ║
║  ─────────────────────                                                     ║
║  Nuclear explosion (energy)     →  Intent-aware Storm (focused excitation)║
║  Pump the medium                →  Populate τ-levels via intent+Boltzmann ║
║  Population inversion           →  Words at excited orbitals              ║
║  Stimulated emission            →  j-vector coherence triggers lasing     ║
║  Coherent beam output           →  Focused thematic response              ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Phase 1: PUMPING (Intent + Euler Storm)

**NEW**: Pumping now uses intent-driven transitions first, with Boltzmann fallback:

```python
# At each step during pumping:
1. TRY INTENT TRANSITION:
   - Query graph.get_intent_transitions(word, intent_verbs, intent_targets)
   - Returns edges where verb matches OR target in intent space
   - If found → use this transition (collapsed by intent)

2. FALLBACK TO BOLTZMANN (max 40% of steps):
   P(transition) ∝ exp(-|Δτ| / kT)
   Where:
     Δτ = |τ_target - τ_current|  # orbital distance
     kT = 0.82                     # natural temperature

3. TRACK COLLAPSE:
   - collapsed_by_intent: bool
   - intent_score: float [0, 1]
```

Intent collapse focuses exploration on paths relevant to what the user wants to DO (understand, help, find, learn).

### Phase 2: POPULATION ANALYSIS

Track distribution across orbital levels and intent collapse:

```
Orbital Distribution:
  n=0: ██████████████████████████████ (35)
  n=1: ██████████████████████████████ (48)  ← GROUND STATE (peak)
  n=2: ████████████████████████ (24)
  n=3: ██████████ (10)
  n=4: █████ (5)
  n=5: ██ (2) ← VEIL (τ = e ≈ 2.718)

Intent Collapse Statistics (NEW):
  Intent fraction: 67%  ← % of states reached via intent
  Avg intent score: 0.82
```

Key metrics:
- `dominant_orbital`: Where population concentrates
- `human_fraction`: % below the Veil (human realm)
- `above_veil`: Count of transcendental concepts
- `intent_fraction`: % of states reached via intent collapse (NEW)
- `intent_collapsed`: Count of intent-driven states (NEW)

### Phase 3: STIMULATED EMISSION

True laser coherence requires BOTH:
1. **j-vector alignment** (polarization coherence)
2. **Orbital proximity** (frequency coherence)

```python
combined_coherence = (1 - w) * j_coherence + w * orbital_coherence

Where:
  j_coherence = cos(j₁, j₂)                    # j-vector similarity
  orbital_coherence = exp(-|n₁ - n₂| / 2)      # orbital proximity
  w = 0.3                                       # orbital weight
```

Concepts cluster into **coherent beams** - groups with aligned meaning AND similar abstraction level.

### Phase 4: COHERENT OUTPUT

Laser metrics measure extraction quality (now includes intent focus):

```
Output = pump_energy × medium_quality × mirror_alignment × intent_focus

pump_energy      = above_veil × (1 - human_fraction)
medium_quality   = count(τ > e) / total_states
mirror_alignment = mean(j_coherence across beams)
intent_focus     = 0.5 + 0.5 × intent_fraction  (NEW: range [0.5, 1.0])
spectral_purity  = dominant_orbital_count / total_states
lasing_achieved  = beam_count > 0 AND mirror_alignment > 0.5
```

Higher intent focus = more of the exploration was guided by user intent, not random.

### Example Output

```
Query: "help me understand the meaning of my dreams"
Verbs: ['help', 'understand']

[SemanticLaser] Intent set: ['help', 'understand'] -> 2 operators, 18 targets
[SemanticLaser] Pump collapse ratio: 100% (100 intent / 0 random)

[EULER-LASER: 1 beams | coherence=0.78 | τ=1.84 | n=1 | lasing=✓ | intent=67%]

COHERENT BEAMS:
  Beam 1 (human):
    Concepts: ['feeling', 'thing', 'meaning', 'dream']
    Coherence: 0.78
    τ: 1.75 ± 0.35
    Orbital: n=1 (ground state)
    Themes: ['+beauty', '+life', '+sacred', '+good', '+love']
    Intent collapsed: 67% of concepts reached via intent

Response: "Dreams often reflect our subconscious thoughts, emotions, and desires,
serving as a window into our inner world. They can also symbolize strong beauty,
life, or the sacred, offering us clues about ourselves..."
```

Notice how the response naturally incorporates the beam themes (+beauty, +life, +sacred).

## Legacy Architecture: Storm-Logos

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

### Semantic Gravity with Euler Physics

```
Potential: φ = +λτ - μg·cos(j, j_good)

Where:
  λ = 0.5  (gravitational constant)
  μ = 0.5  (lift constant)
  τ = semantic altitude [1-6]
  g = goodness [-1, +1]

Euler Constants:
  e = 2.718...  (fundamental unit)
  kT = 0.82     (natural temperature = ΔE/e)
  Δτ = 1/e      (orbital spacing)
```

**Key insight**: Meaning naturally "falls" toward human reality (low τ) while goodness provides "lift" toward the transcendental. The Veil at τ = e marks the boundary.

```
τ=6  ☀️ TRANSCENDENTAL - Beyond ordinary experience (11%)
         ↑ TRANSCENDENCE requires WORK (against gravity)
τ=e  ═══ THE VEIL (τ = 2.718) ═══════════════════════
         ↓ GROUNDING is NATURAL (with gravity)
τ=1.37 ⭐ GROUND STATE (n=1 orbital, 30% of concepts)
τ=1  🌍 HUMAN REALITY - Common shared experience (89%)
```

### Boltzmann Selection (Storm Phase)
```
P_i = exp(-E_i / kT) / Σ exp(-E_j / kT)

Natural T (0.82): Follows orbital structure
Low T (0.3): Deterministic descent to ground state
High T (2.0): Exploratory, can cross the Veil
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

### Gravity Mode Results

Comparison of gravity modes on "What is wisdom?" dialogue (6 exchanges):

| Gravity α | Avg τ | Avg Coherence | Physics |
|-----------|-------|---------------|---------|
| 0.0 | 2.38 | 76% | - |
| **0.5** | **2.48** | **87%** | realm=human, φ=1.37 |
| 0.75 | 2.36 | 80% | realm=human, φ=1.39 |

**Finding**: Gravity α=0.5 produces optimal results:
- Highest average coherence (87%)
- Grounded in human realm (τ < 3.5)
- Peak coherence of 98% in middle exchanges
- Meaningful convergence points ("way" → "end" → "permission")

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

## Dual-Role Words & Intent Collapse

Words like "love", "dream", "help" function as both:
- **Nouns** (concepts): Seeds for storm phase
- **Verbs** (operators): Intent collapse operators (NEW)

```python
# Input: "help me understand my dream"
Nouns: ['dream']                  # concepts to explore (seeds)
Verbs: ['help', 'understand']     # intent operators (collapse navigation)

# What happens:
1. Load VerbOperators for 'help' and 'understand' from graph
2. Get OPERATES_ON targets (concepts these verbs act upon)
3. During pumping, prioritize transitions to intent targets
4. Result: Navigation collapses to intent-relevant paths
```

### Intent Collapse Theory

"Intent collapses meaning like observation collapses wavefunction"

| Aspect | Before Intent | After Intent |
|--------|--------------|--------------|
| Navigation | Random Boltzmann walks | Verb-directed collapse |
| Concepts found | Abstract, scattered | Goal-oriented, actionable |
| Response style | Generic LLM knowledge | Coherent, theme-woven |

The verbs tell the system what the user wants to DO, not just what they're asking ABOUT.

## Usage

### Start Neo4j
```bash
cd config && docker-compose up -d
```

### Run Chat (Euler-Laser)
```bash
python app/chat.py
```

The chat now uses **Euler-Laser** by default:
```
============================================================
  MEANING CHAIN CHAT
  Semantic Laser Navigation
============================================================

  Semantic Laser:
    Pumping -> Population -> Stimulated Emission -> Coherent Output
    Uses j-vector alignment for thematic coherence

Commands:
  /laser  - Toggle Laser mode
  /euler  - Toggle legacy Euler mode
  /clear  - Clear conversation history
  /quiet  - Toggle verbose output
  /exit   - Exit

You: I dreamed of climbing a mountain with ancient symbols...

[2] SEMANTIC LASER (coherent extraction)...
    Excited: 139 states
    τ range: 1.00 - 4.23
    Coherent beams: 2
    Beam 1: ['passage', 'way', 'place', 'dream'] (coherence=0.70)
    Beam 2: ['god', 'sun', 'expression'] (coherence=0.75)

[3] COHERENT OUTPUT
    Primary beam: 40 concepts
    Polarity: neutral (g=-0.03)
    Themes: ['+life', '+sacred', '+good']

[EULER-LASER: 2 beams | coherence=0.70 | τ=1.51 | n=1 | lasing=✓]
```

### Run Euler Dialogue with Claude
```bash
export ANTHROPIC_API_KEY="your-key"

# Euler-aware dialogue (recommended)
python app/dialogue_claude_euler.py --exchanges 5 --topic "What is wisdom?"

# Use Claude for rendering (higher quality responses)
python app/dialogue_claude_euler.py --exchanges 5 --claude-render
```

### Run Standard Dialogue (legacy)
```bash
# Two semantic agents
python app/dialogue.py --exchanges 5 --topic "What is meaning?"

# With Claude (standard mode)
python app/dialogue_claude.py --exchanges 5 --gravity 0.5
```

Results saved to `results/dialogue_euler/` with orbital statistics.

## Directory Structure

```
meaning_chain/
├── chain_core/
│   ├── semantic_laser.py   # Euler-Laser + Intent Collapse (MAIN)
│   ├── intent_collapse.py  # Intent-driven navigation (NEW)
│   ├── euler_navigation.py # Euler orbital navigation
│   ├── storm_logos.py      # Storm-Logos + Intent (updated)
│   ├── decomposer.py       # Text → nouns + verbs (spaCy)
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
│       ├── euler_constant.py         # Euler validation (6 tests) (NEW)
│       ├── gravity_storm.py          # Gravity-aware storm prototype
│       ├── semantic_gravity.py       # 6 validated gravity tests
│       ├── semantic_thermodynamics.py # Temperature, entropy, phase behavior
│       ├── semantic_optics.py        # Refraction, lens, interference
│       ├── storm_physics.py          # Dynamic physics observer
│       └── results/
│           ├── euler_constant_*.json     # Euler validation results
│           └── orbital_structure.png     # Orbital visualization (NEW)
│
├── docs/
│   ├── UNIFIED_SEMANTIC_PHYSICS.md   # Complete physics theory + Euler
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
│   ├── chat.py                 # Interactive chat (Euler-aware)
│   ├── dialogue.py             # Two semantic agents
│   ├── dialogue_claude.py      # Semantic ↔ Claude dialogue
│   └── dialogue_claude_euler.py # Euler-aware Claude dialogue (NEW)
│
├── models/
│   └── types.py            # MeaningNode, MeaningTree
│
└── results/
    ├── dialogue_comparison/  # Before/after dialogue results
    ├── dialogue_claude/      # Claude dialogue results
    └── dialogue_euler/       # Euler-aware dialogue results (NEW)
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

(:VerbOperator {         // NEW: Verbs as semantic operators
    verb: STRING,
    j: LIST<FLOAT>,      // Direction this verb pushes toward
    magnitude: FLOAT     // Operator strength
})

(:Adjective {word: STRING})

(:Concept)-[:VIA {verb, weight, count, source}]->(:Concept)
(:Concept)-[:DESCRIBED_BY {count, source}]->(:Adjective)
(:VerbOperator)-[:OPERATES_ON]->(:Concept)  // NEW: What this verb acts upon
```

### Intent Collapse Query

The key query that enables intent-driven navigation:

```cypher
// Get intent-aligned transitions from a concept
MATCH (c:Concept {word: $word})-[r:VIA]->(target:Concept)
WHERE r.verb IN $intent_verbs           // Verb matches intent
   OR target.word IN $intent_targets    // Target in intent space
RETURN r.verb, target.word,
       CASE WHEN r.verb IN $intent_verbs THEN 2.0 ELSE 1.0 END +
       CASE WHEN target.word IN $intent_targets THEN 1.0 ELSE 0.0 END AS score
ORDER BY score DESC
LIMIT 10
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

### Euler-Laser with Intent Collapse (Recommended)

```python
from chain_core.semantic_laser import SemanticLaser, KT_NATURAL

# Euler-aware laser with natural temperature
laser = SemanticLaser(temperature=KT_NATURAL)  # kT = 0.82

# Lase with intent collapse - verbs guide navigation
result = laser.lase(
    seeds=['wisdom', 'love', 'dream'],
    pump_power=10,           # walks per seed
    pump_depth=5,            # steps per walk
    coherence_threshold=0.3, # minimum j-alignment
    intent_verbs=['find', 'understand', 'seek']  # NEW: intent operators
)

# Intent collapse statistics
intent = result['intent']
print(f"Intent enabled: {intent['enabled']}")
print(f"Intent verbs: {intent['verbs']}")

# Population statistics (now includes intent)
pop = result['population']
print(f"Excited states: {pop['total_excited']}")
print(f"Mean τ: {pop['tau_mean']:.2f}")
print(f"Human realm: {pop['human_fraction']:.1%}")
print(f"Intent fraction: {pop['intent_fraction']:.0%}")  # NEW

# Coherent beams
for beam in result['beams']:
    print(f"Beam: {beam.concepts[:5]}")
    print(f"  Coherence: {beam.coherence:.2f}")
    print(f"  Themes: {laser.get_beam_themes(beam)}")

# Laser metrics (now includes intent focus)
metrics = result['metrics']
print(f"Lasing achieved: {metrics['lasing_achieved']}")
print(f"Output power: {metrics['output_power']:.3f}")
print(f"Intent focus: {metrics['intent_focus']:.2f}")  # NEW
```

### Without Intent (comparison)

```python
# Same query but without intent verbs
result_no_intent = laser.lase(
    seeds=['wisdom', 'love', 'dream'],
    intent_verbs=None  # No intent collapse
)

# Compare: scattered exploration vs focused collapse
print(f"Without intent: {result_no_intent['population']['total_excited']} states")
print(f"With intent: {result['population']['total_excited']} states (fewer, focused)")
```

### Temperature Values

| Temperature | Mode | Effect |
|-------------|------|--------|
| 0.3 | Cold | Deterministic descent to ground state |
| **0.82** | **Natural (kT)** | Follows orbital structure |
| 1.5 | Warm | More exploration, occasional veil crossing |
| 2.0 | Hot | Exploratory, can reach transcendental |

### Legacy Configuration (Euler Navigation)

```python
# Standard mode
StormLogosBuilder(
    storm_temperature=1.5,
    n_walks=5,
    steps_per_walk=8,
    gravity_strength=0.5,     # Semantic gravity [0-1]
)
```

## Connection to Experience Knowledge

| Property | Experience Knowledge | Meaning Chain |
|----------|---------------------|---------------|
| Neo4j Port | 7687 | 7688 |
| Edge Type | TRANSITION | VIA |
| Navigation | Quantum tunneling | Storm-Logos |
| Consciousness | Full layer | Meditation + Prayer |
