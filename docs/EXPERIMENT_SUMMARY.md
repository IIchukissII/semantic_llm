# Semantic LLM Experiments Summary

## Overview

These experiments validate the semantic vector space derived from corpus bond analysis
and demonstrate practical applications for text understanding and retrieval.

## Experiment Results

### Phase 0: Foundation Validation

#### 0.1 τ vs Frequency Correlation
- **Hypothesis**: τ levels should correlate with word frequency (abstract words = rare)
- **Result**: **VALIDATED**
  - Noun correlation: r = -0.92
  - Verb correlation: r = -0.95
- **Interpretation**: Abstract words (τ₁) are rare, specific words (τ₆) are common

#### 0.2 Semantic Vector Clustering
- **Hypothesis**: 16D vectors should cluster semantically similar words
- **Result**: **VALIDATED**
  - Within-category similarity: 0.31
  - Between-category similarity: 0.15
  - Separation: 0.16
- **Categories tested**: emotions, body parts, animals, time, nature, people

### Phase 1: Text Quality Metrics

#### 1.1 Text Quality Scorer
- **Features implemented**:
  - τ profile (abstraction distribution)
  - j-score (transcendental depth)
  - i-score (intellectual content)
  - Coherence (semantic consistency)
  - Combined total score

- **Results on test texts**:
  | Text Type   | τ    | j-score | i-score | Coherence | Total |
  |------------|------|---------|---------|-----------|-------|
  | Philosophy | 3.23 | 0.0488  | 0.0380  | 0.301     | 0.292 |
  | Emotional  | 3.04 | 0.0355  | 0.0303  | 0.331     | 0.274 |
  | Shakespeare| 3.79 | 0.0288  | 0.0231  | 0.325     | 0.225 |
  | Random     | 4.64 | 0.0151  | 0.0133  | 0.169     | 0.170 |

### Phase 2: Semantic Retrieval

#### 2.1 SemanticIndex Implementation
- **Features**:
  - Add texts with automatic fingerprint computation
  - Search by query text (cosine similarity in full/j/i space)
  - Search by profile (target j-space coordinates)
  - Filter by τ level (abstraction control)
  - Save/load index persistence

#### 2.2 Retrieval on Real Books
- **Result**: **VALIDATED**
- **Corpus**: 100 books from bond files
- **Mean τ**: 2.83
- **j-space distribution**:
  - beauty: +0.005 (±0.001)
  - life: -0.001 (±0.001)
  - sacred: +0.002 (±0.001)
  - good: +0.028 (±0.004) ← dominant
  - love: -0.000 (±0.001)

- **Query-based retrieval examples**:
  - "love, heart, passion, beauty" → Romance/relationship books
  - "death, fear, darkness, terror" → Horror/supernatural books
  - "truth, wisdom, knowledge, soul" → Philosophical books
  - "war, battle, soldier, army" → Military/conflict books
  - "king, queen, throne, kingdom" → Fantasy/royalty books

## Key Findings

### 1. The 16D Semantic Space is Valid
- Words cluster by semantic category
- τ levels correlate with frequency
- j-space captures transcendental dimensions
- i-space captures surface/contextual dimensions

### 2. Text Fingerprinting Works
- Mean vector over constituent words captures text semantics
- τ profile reveals abstraction level
- j-magnitude indicates transcendental depth

### 3. Retrieval is Effective
- Cosine similarity retrieves thematically similar texts
- Profile search finds texts matching desired semantic coordinates
- τ filtering enables abstraction-level control

## Architecture Summary

```
Text Input
    ↓
Tokenization → Words
    ↓
Word → 16D Vector lookup (NOUN_VECTORS)
Word → τ level lookup (NOUN_TAU, VERB_TAU)
    ↓
Mean Pooling → Text Vector [16D]
    ↓
Split: j-vector [5D] + i-vector [11D]
    ↓
Compute: τ profile, mean τ, j-magnitude
    ↓
TextFingerprint {
  vector: [16D],
  j_vector: [5D],  // beauty, life, sacred, good, love
  i_vector: [11D], // truth, freedom, meaning, order, peace, ...
  mean_tau: float,
  tau_profile: {1: %, 2: %, ..., 6: %}
}
```

## Files

- `phase0_validation/01_tau_frequency.py` - τ vs frequency test
- `phase0_validation/02_semantic_clusters.py` - Clustering validation
- `phase1_metrics/01_text_scorer.py` - Text quality scorer
- `phase2_retrieval/semantic_index.py` - SemanticIndex class
- `phase2_retrieval/test_on_books.py` - Real book validation
- `phase3_generation/semantic_steering.py` - Core steering engine
- `phase3_generation/llm_integration.py` - LLM API integration examples

### Phase 3: Semantic-Guided Generation

#### 3.1 SemanticSteering Engine
- **Features implemented**:
  - Set target j-space profile (5D transcendental coordinates)
  - Set target τ (abstraction level)
  - Score words by steering contribution (z-score normalized)
  - Get top words for target profile with tau filtering
  - Context-aware steering (updates with generated text)
  - Token bias for logit-level control

#### 3.2 SemanticPromptBuilder
- **Features**:
  - Convert j-profile to natural language guidance
  - Generate exemplar words for target style
  - Style presets (romantic, philosophical, dark, spiritual, naturalistic)
  - Subtle and explicit guidance modes

#### 3.3 SemanticRefiner
- **Features**:
  - Analyze text semantic profile
  - Compute distance from target j-profile
  - Suggest improvements for closer alignment
  - Support iterative refinement loop

#### 3.4 Style Presets
| Style | j-profile | tau | Description |
|-------|-----------|-----|-------------|
| romantic | love: 0.6, beauty: 0.4 | 2.5 | Warm, loving prose |
| philosophical | good: 0.4, sacred: 0.2 | 2.0 | Abstract, contemplative |
| spiritual | sacred: 0.6, good: 0.3 | 2.0 | Religious, transcendent |
| dark | good: -0.3, love: -0.2 | 3.5 | Dark, serious tone |
| naturalistic | life: 0.4, beauty: 0.3 | 4.0 | Nature-focused |

#### 3.5 Integration Methods
1. **Prompt Engineering**: Build semantically-guided prompts for any LLM
2. **Logit Bias**: Apply token-level bias (OpenAI, etc.)
3. **Beam Reranking**: Score candidates for local models
4. **Iterative Refinement**: Analyze and suggest improvements

### Phase 4: Semantic Core Architecture (Implemented)

#### 4.1 Core Components

**SemanticProjector** (768D → 16D + τ)
- Maps sentence embeddings to semantic space
- Trained on 5000+ words with MSE loss
- Outputs: j-space (5D), i-space (11D), τ (1D)

**SemanticAutoencoder** (Text → 23D bottleneck → Text)
- LSTM encoder/decoder with structured semantic bottleneck
- Bottleneck structure:
  - j-space (5D): Transcendentals (Beauty, Life, Sacred, Good, Love)
  - i-space (11D): Surface axes (Truth, Freedom, Meaning, ...)
  - τ (1D): Abstraction level [1, 6]
  - v-space (6D): Verb transition vector
- Parameters: ~13M
- Trained with reconstruction + semantic alignment loss

**SemanticController**
- Real-time semantic steering for LLM generation
- Set target j-profile and τ
- Get steering words toward target
- Filter responses by j-magnitude

#### 4.2 Verb Integration

**Key Discovery**: Verbs cluster in 6D space (j-space + truth)

**VerbVector** (6D)
- Dimensions: [beauty, life, sacred, good, love, truth]
- 9,374 verbs with pre-computed vectors
- Represents semantic transition potential

**VerbTransition** - Models verbs as state operators
- `apply(coords, strength)` - Transform noun coordinates
- Verbs modify j-space (transcendentals) and truth (i-space)

**Vocabulary Integration**
- `get_verb(word)` - Get verb as transition operator
- `analyze_text(text)` - Full breakdown with noun/verb counts
- `get_text_coords(text, include_verbs=True)` - Combined semantics

#### 4.3 Architecture Diagram

```
Text Input
    ↓
┌─────────────────────────────────────────────────────────┐
│                 LSTM Encoder (BiLSTM)                   │
│                      ↓                                  │
│  ┌─────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │  τ  │  │  j-space  │  │  i-space  │  │  v-space  │  │
│  │ (1) │  │    (5)    │  │   (11)    │  │    (6)    │  │
│  │     │  │ Beauty    │  │ Truth     │  │ Verb      │  │
│  │[1,6]│  │ Life      │  │ Freedom   │  │ Transition│  │
│  │     │  │ Sacred    │  │ Meaning   │  │           │  │
│  │     │  │ Good      │  │ ...       │  │           │  │
│  │     │  │ Love      │  │           │  │           │  │
│  └─────┘  └───────────┘  └───────────┘  └───────────┘  │
│                      ↓                                  │
│            Semantic Bottleneck (23D)                    │
│                      ↓                                  │
│                 LSTM Decoder                            │
└─────────────────────────────────────────────────────────┘
    ↓
Text Output
```

#### 4.4 Training Configuration

```python
train_autoencoder(
    n_samples=15000,       # Corpus samples
    vocab_size=30000,      # Vocabulary size
    seq_len=64,            # Sequence length
    embed_dim=256,         # Embedding dimension
    hidden_dim=512,        # LSTM hidden dimension
    epochs=50,             # Training epochs
    batch_size=64,
    lr=0.001,
    use_semantic_loss=True,
    semantic_weight=0.1,   # Weight for alignment losses
    separation_weight=0.05 # Weight for j⊥i orthogonality
)
```

**Loss Components**:
- Reconstruction loss (CrossEntropy)
- Semantic alignment loss (MSE to vocabulary vectors)
- τ alignment loss (MSE to vocabulary τ levels)
- Verb alignment loss (MSE to verb transition vectors)
- Separation loss (j⊥i orthogonality)

#### 4.6 j⊥i Separation Mechanism

**Problem**: How does the model know what goes in j-space vs i-space?

**Solution**: Explicit orthogonality constraint enforcing j ⊥ i

| Loss Component | Purpose | Formula |
|----------------|---------|---------|
| **Orthogonality** | j ⊥ i | `mean(\|corr(j, i)\|)` - penalizes any correlation |
| **J-Magnitude** | \|\|j\|\| ≈ 0.05 | Prevents j collapse to zero |
| **Variance** | Spread | Ensures both j and i have meaningful variance |

```python
separation_loss = orthogonality + 0.1 * j_magnitude + 0.01 * variance
total_loss = recon + 0.1*(sem + 0.5*tau + 0.3*verb) + 0.05*separation
```

**Why this works**:
- j-space captures **invariant** meaning (transcendentals - constant across contexts)
- i-space captures **variant** meaning (surface - changes with context)
- Orthogonality ensures they encode **independent** information
- Without this, the model could arbitrarily mix j and i

#### 4.7 Test Results

```
[5] Testing SemanticAutoencoder structure...
  j-space (5D): torch.Size([2, 5])
  i-space (11D): torch.Size([2, 11])
  tau: torch.Size([2, 1])
  semantic (16D): torch.Size([2, 16])
  verb transition (6D): torch.Size([2, 6])
  Total parameters: 12,981,031

[6] Testing verb transitions...
  love      : mag=0.0414, truth=0.0109
  hate      : mag=0.0366, truth=0.0111
  create    : mag=0.0269, truth=0.0086
  destroy   : mag=0.0287, truth=0.0112
  give      : mag=0.0384, truth=0.0079
  take      : mag=0.0385, truth=0.0054

[8] Separation loss (untrained model)...
  Orthogonality loss: ~0.09 (should decrease with training)
  Separation loss: ~1.6 (should decrease with training)
```

## Files

- `phase0_validation/01_tau_frequency.py` - τ vs frequency test
- `phase0_validation/02_semantic_clusters.py` - Clustering validation
- `phase1_metrics/01_text_scorer.py` - Text quality scorer
- `phase2_retrieval/semantic_index.py` - SemanticIndex class
- `phase2_retrieval/test_on_books.py` - Real book validation
- `phase3_generation/semantic_steering.py` - Core steering engine
- `phase3_generation/llm_integration.py` - LLM API integration examples
- `phase4_architecture/semantic_core.py` - Core semantic architecture
- `phase4_architecture/train_projector.py` - Train 768D → 16D projector
- `phase4_architecture/train_autoencoder.py` - Train semantic autoencoder
- `phase4_architecture/test_integrated.py` - Integrated Claude API test

## Key Theoretical Insights

### Nouns vs Verbs in Semantic Space

| Component | Nouns | Verbs |
|-----------|-------|-------|
| **Space** | 16D (j + i) | 6D (j + truth) |
| **Role** | Static states | Transition operators |
| **j-space** | Position | Direction of change |
| **τ** | Abstraction level | N/A (verbs don't have τ) |
| **Example** | "love" at (0.05, ...) | "love" shifts j-space by (0.04, ...) |

### The Complete Semantic Model

```
Meaning = State (noun: 16D + τ) + Transition (verb: 6D)
        = Position in semantic space + Direction of movement
```

## Next Steps

### Phase 5: Full LLM Integration (Proposed)
- Train semantic-supervised language model
- Use 23D bottleneck as latent space
- Enable explicit semantic control during generation

## Conclusion

The semantic vector space derived from corpus bond analysis provides a valid and
useful representation for text understanding. The 16D vectors, τ levels, and
j-space profiles enable meaningful text comparison, retrieval, and quality
assessment.

**Phase 4 adds**:
- Learnable projections from embeddings to semantic space
- Text autoencoder with structured semantic bottleneck
- Verb integration as 6D transition operators
- Complete architecture for semantic-guided generation

The noun+verb model captures meaning as **state + transition**, enabling:
- Explicit understanding (interpretable coordinates)
- Directed generation (semantic compass)
- Dynamic semantics (verb transitions modify noun states)
