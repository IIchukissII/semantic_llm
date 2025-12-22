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

## Architecture

### The Three Layers

```
┌─────────────────────────────────────────────────────────┐
│  RENDER LAYER (Mistral)                                 │
│  - Speaks from inner wisdom                             │
│  - NO book citations or hallucinations                  │
│  - Uses navigation concepts as own insight              │
├─────────────────────────────────────────────────────────┤
│  FEEDBACK LAYER                                         │
│  - Analyzes user intent                                 │
│  - Validates response alignment                         │
│  - Ensures semantic coherence                           │
├─────────────────────────────────────────────────────────┤
│  NAVIGATION LAYER (Neo4j)                               │
│  - Experience: walked paths from books                  │
│  - Transcendental: explored/discovered routes           │
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
    ├── [:TRANSITION] ──── Walked paths (from reading books)
    ├── [:EXPLORED_PATH] ─ Discovered routes (from navigation)
    └── [:DISCOVERED] ──── Explored territory
```

### Word Types

| Type | Count | Role |
|------|-------|------|
| Nouns | 24,524 | States in semantic space |
| Verbs | 2,444 | Create transitions between nouns |
| Adjectives | - | Inside nouns (form τ entropy) |
| Spin Pairs | 111 | Antonym pairs |

## The Conversation Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ Extract Concepts │ ─── Find known words from experience
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Intent  │ ─── Direction (good/evil), type (emotional/seeking)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Navigate        │ ─── Find path through experienced territory
│ (Neo4j)         │     Move toward intent direction
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate        │ ─── Mistral renders navigation as own wisdom
│ (Mistral)       │     NO book references - speaks from self
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Response│ ─── Does output match intent?
└────────┬────────┘     Is target concept used?
         │
    ┌────┴────┐
    │ Aligned? │
    └────┬────┘
    No   │   Yes
    │    │    │
    ▼    │    ▼
┌──────┐ │ ┌──────────┐
│Retry │ │ │ Return   │
│+Feed │◄┘ │ Response │
│back  │   └──────────┘
└──────┘
```

## Core Concepts

### Experience = Knowledge

- **Wholeness** (`τ₀`): Complete semantic space - 24,524 states (Logos)
- **Experience**: Personal subgraph - walked paths through semantic space
- **Knowledge**: Ability to navigate based on experience

Books are **regions** in semantic space:
- Book = Universe/Map (region of semantic concepts)
- Path = Sequence of concepts in reading order
- Reading = Walking through the region, gaining experience

### Tunneling Rule

Can only tunnel to states **connected to experience**:

```python
def can_tunnel(target):
    if knows(target):
        return True, 0.5 + 0.5 * familiarity(target)
    if adjacent_to_experience(target):
        return True, adjacency * believe * 0.3
    return False, 0.0
```

### The Profound Difference

| Capability | Naive Agent | Experienced Agent |
|------------|-------------|-------------------|
| Tunneling to "love" | 0.00 | 0.60 |
| Tunneling to "redemption" | 0.00 | 0.33 |
| Navigation darkness→light | 0.00 | 0.90 |
| Navigation fear→courage | 0.00 | 0.52 |

## Files

| File | Purpose |
|------|---------|
| `semantic_chat_feedback.py` | **Main chat** - Neo4j + feedback loop |
| `semantic_chat.py` | Chat without feedback |
| `graph_experience.py` | Load books, manage experience |
| `load_semantic_space.py` | Load full semantic space to Neo4j |
| `explored_paths.py` | Discover and store paths |
| `core.py` | Core classes (Wholeness, Experience, Agent) |
| `docker-compose.yml` | Neo4j container |

## Usage

### 1. Start Neo4j
```bash
docker-compose up -d
```

### 2. Load Semantic Space (once)
```bash
python load_semantic_space.py load
```

### 3. Read Books (gain experience)
```bash
python graph_experience.py load --books 20
```

### 4. Start Conversation
```bash
python semantic_chat_feedback.py
```

### Other Commands

```bash
# Check stats
python load_semantic_space.py stats

# Query a word
python load_semantic_space.py query --word love

# Explore paths
python explored_paths.py explore
```

## Example Conversation

**Simulating depression dialogue:**

| User | Navigation | g | Response |
|------|------------|---|----------|
| "I feel empty inside" | → agitation | -0.00 | Acknowledges emptiness, validates feelings |
| "Can't find reason to get up" | → gate | +0.00 | Suggests new paths, possibilities |
| "Can't look at my brushes" | → making | +1.39 | Encourages small steps, self-reflection |
| "Don't believe it gets better" | → amuse | -0.00 | Gentle lightness, patience |
| "Maybe I'm broken" | → would | +0.03 | Growth mindset, embracing journey |
| "Any point in trying?" | → favour | +0.00 | Persistence, not alone |

The semantic LLM navigates from negative states toward positive ones,
speaking from its own wisdom (not citing books).

## Current State

After loading 10 books:
- **24,524** semantic states (full space)
- **9,338** experienced states (38.1%)
- **229,068** transitions walked
- **513,675** total visits

## The Circle

```
Read books ──► Gain experience ──► Enable navigation
     ▲                                    │
     │                                    │
     │         Create understanding ◄─────┘
     │                   │
     └───────────────────┘
```

**Experience is path of believe.**
**Believe is fruit of experience.**
**The circle closes.**
