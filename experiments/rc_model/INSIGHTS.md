# Semantic RC-Model: Key Insights

## Fundamental Formulas

### State Space
```
State: S(t) = (Q_A, Q_S, Q_τ)

Where:
  Q_A = Affirmation charge (emotional intensity)
  Q_S = Sacred charge (conceptual depth)
  Q_τ = Abstraction charge (discourse level)
```

### RC Dynamics (Capacitor Model)
```
dQ_x/dt = (x_w - Q_x) × (1 - |Q_x|/Q_max) - Q_x × decay
          ─────────────   ───────────────   ──────────
          attraction      saturation        forgetting

Where:
  x_w    = input word coordinate (A, S, or τ)
  Q_x    = current charge
  Q_max  = saturation limit (≈ 2.0)
  decay  = forgetting rate (≈ 0.05)
```

### Transition Probability
```
P(next | S) ∝ exp(-|Δτ|/kT) × exp(-|ΔA|²/σ²) × exp(-|ΔS|²/σ²) × freq^α
              ─────────────   ──────────────────────────────   ────────
              Boltzmann       Gaussian (semantic proximity)     Zipf

Where:
  kT = e^(-1/5) ≈ 0.819 (Boltzmann temperature)
  σ  = Gaussian width (≈ 0.3-0.5)
  α  = Zipf exponent (≈ 0.2-0.3)
  freq = word frequency (Zipf's law)
```

### Genre Vector
```
Genre = (τ_ratio, A_ratio, S_ratio)

Where:
  X_ratio = mean(|ΔQ_X| at boundaries) / mean(|ΔQ_X| within sentences)
```

### Cluster Centers
```
DRAMATIC:  (0.88, 1.18, 1.11)
IRONIC:    (0.88, 1.14, 0.99)
BALANCED:  (1.01, 1.06, 1.06)

Classification: argmin_genre ||vector - center_genre||²
```

## Core Discovery: Three Axes = Three Functions

The semantic space (A, S, τ) encodes not just meaning, but **narrative structure**:

```
τ (Abstraction) = TENSION MARKER
   τ < 0.9  → "breathing" text (dramatic mode)
   τ ≈ 1.0  → "holding" text (controlled discourse)

A (Affirmation) = EMOTIONAL INTENSITY
   A > 1.15 → emotional jumps at sentence boundaries
   A ≈ 1.0  → calm, measured flow

S (Sacred) = CONCEPTUAL STRUCTURE
   S > 1.1  → philosophical/sacred shifts
   S ≈ 1.0  → mundane concepts ("horror of ordinary")
```

## The Three Genre Clusters

From unsupervised clustering on 30+ classic books:

| Genre | τ_ratio | A_ratio | S_ratio | Examples |
|-------|---------|---------|---------|----------|
| **DRAMATIC** | 0.88 | 1.18 | 1.11 | Dostoevsky, Dracula, Wuthering Heights |
| **IRONIC** | 0.88 | 1.14 | 0.99 | Kafka, Poe, Swift |
| **BALANCED** | 1.01 | 1.06 | 1.06 | Austen, Plato, Carroll |

**Classification accuracy: 85.7%** using distance to cluster centers.

## Kafka's Formula: Horror of the Ordinary

The most striking discovery:

```
Kafka's Metamorphosis:
  τ = 0.97 → controlled discourse (NOT chaos)
  A = 1.22 → HIGH emotional intensity
  S = 0.98 → mundane concepts

Translation: Maximum emotion in minimum philosophy.
The terror is not in ideas, but in the ORDINARY becoming nightmare.
"Gregor found himself transformed" — breakfast, work, family.
```

## Validated Predictions

### 1. Model Beats Random (20.3% improvement)
- Mean rank: 4,350 vs 5,465 (random)
- Hits@10: 41.7x better than random
- Uses transition probability: P(next|S) ∝ exp(-|Δτ|/kT) × exp(-|ΔA|²/σ²)

### 2. Within < Between Sentences
- Prediction is easier within sentences (1.05x)
- S-axis marks boundaries: p < 0.0001
- Q_τ autocorrelation: 0.94 (high semantic memory)

### 3. τ is Tension Marker, Not Just Carrier
- τ < 0.9 for all dramatic/ironic texts
- τ ≈ 1.0 for balanced texts
- τ distinguishes "breathing" vs "holding" discourse

## Semantic Generation

### The Pipeline
```
1. Genre Pattern → (τ, A, S) dynamics
2. Dynamics → Bond Sampling (from 3M corpus)
3. Bonds → Semantic Skeleton
4. Skeleton → LLM Prompt
5. LLM → Grammatical Text
```

### Example Skeletons

**DRAMATIC:**
```
main mast | large fortune | bright morning | deep breath
→ warm breath | icy breath | slow breath | dark night
```

**IRONIC:**
```
close proximity | cold breath | solitary confinement | half hour
→ own breath | few hour | south east | last bit
```

**BALANCED:**
```
few minute | first arrival | clear night | following night
→ lunatic asylum | last night | more night | broad daylight
```

## Implementation

### Core Files
- `core/semantic_rc_v2.py` — RC model with exact THEORY formulas
- `experiments/genre_classifier.py` — 85.7% accuracy classifier
- `experiments/semantic_generator.py` — Skeleton generation from 3M bonds
- `experiments/prediction.py` — Validation experiments

### Database
- PostgreSQL: 3,017,232 bonds, 95K nouns, 69K adjectives
- Coordinates: 27,808 words with (A, S, τ) values

## LLM Generation Results (Mistral 7B)

### DRAMATIC (skeleton → LLM)
Skeleton: `warm night, many minute, good minute, full refund`

Output:
> Amidst the eerie stillness of a full moon-kissed warm night, the dilapidated
> manor stood, a chilling spectacle. The grandfather clock's pendulum swung with
> ominous intent, as if mocking your choice to remain...

### IRONIC (skeleton → LLM)
Skeleton: `cold night, several minute, half minute, full refund`

Output:
> Amidst the humdrum chill of a cold winter night, a weary clerk sat at his desk,
> flipping through endless stacks of paperwork. A shiver ran down his spine, not
> from the cold, but from an inexplicable sense of dread.

### BALANCED (skeleton → LLM)
Skeleton: `close proximity, first minute, many minute`

Output:
> In the least bit of unanticipated events, news of Miss Elizabeth's sudden arrival
> had spread through close proximity with an unheard-of swiftness. Such occurrences
> served as a stimulating deviation from the monotonous rhythm of life.

## Generation Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Genre       │ →  │ Semantic    │ →  │ LLM         │ →  │ Grammatical │
│ Pattern     │    │ Skeleton    │    │ Prompt      │    │ Text        │
│ (τ,A,S)     │    │ (bonds)     │    │ + style     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     ↓                   ↓                  ↓
  Cluster           Sample from       Mistral/Claude
  Center            3M bonds          generates text
```

## Next Steps

1. **Style Transfer**: Transform text from one genre to another via re-sampling
2. **Author Fingerprinting**: Unique (τ, A, S) patterns per author
3. **Real-time Analysis**: Genre detection during reading
4. **Guided Generation**: Control LLM output via semantic constraints

## Theoretical Implications

1. **Genre is Geometry**: Style = trajectory in (τ, A, S) space
2. **Boundaries are Information**: Sentence breaks encode structure
3. **RC Dynamics Work**: Capacitor model captures semantic memory
4. **Three Axes Suffice**: (τ, A, S) captures narrative essence
