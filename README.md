# Quantum Semantic Architecture

A 16-dimensional semantic space for language understanding, validated on 928K books.

## Key Discoveries

| Discovery | Formula | Result |
|-----------|---------|--------|
| **Entropy-based τ** | τ = 1 + 5×(1 - H_norm) | Abstraction from Shannon entropy |
| **One-Bit Law** | H_adj - H_verb ≈ 1.08 | Being > Doing by 1 bit |
| **Euler's e** | ln(H_adj/H_verb) ≈ 1/e | Euler's number in language |
| **Spin Operators** | prefix: τ preserved, s flipped | Prefixes as quantum operators |
| **Compass Navigation** | t=4.59, p<0.001 | Goal-directed navigation works |

## Architecture

```
16D Semantic Space = j-space (5D) ⊕ i-space (11D) + τ

j-space (transcendentals):
  beauty, life, sacred, good, love

i-space (surface/context):
  truth, freedom, meaning, order, peace,
  power, nature, time, knowledge, self, society

τ (abstraction level):
  1 = abstract (high entropy)
  6 = concrete (low entropy)
```

## Directory Structure

```
semantic_llm/
├── core/                    # Essential implementation
│   ├── semantic_core.py     # 16D space definition
│   ├── semantic_bottleneck.py # V3 model
│   ├── navigator.py         # Compass navigation
│   └── hybrid_llm.py        # Trajectory → text
│
├── validation/              # Theory proofs (algorithmic)
│   ├── entropy_tau.py       # τ = f(entropy)
│   ├── euler_constant.py    # ln(H_adj/H_verb) ≈ 1/e
│   ├── entropy_correlation.py
│   ├── unified_space.py     # r=0.9389
│   ├── spin_conservation.py # τ conserved 100%
│   └── navigation_compass.py # t=4.59
│
├── scripts/                 # Utilities
│   ├── export_data.py       # DB → JSON/CSV
│   ├── train_bottleneck.py
│   └── build_index.py
│
├── data/                    # Exported data
│   ├── json/
│   └── csv/
│
├── models/                  # V3 trained models
├── results/                 # Organized results
├── docs/                    # Documentation
└── deprecated/              # Old variety-based τ
```

## Quick Start

### Option 1: With Database (full data)
```bash
# Export data from database to CSV/JSON
python scripts/export_data.py --all

# Run validation
python validation/entropy_tau.py
python validation/navigation_compass.py
```

### Option 2: With CSV only (reproduction)
```bash
# If data/csv/ and data/json/ are already populated:
python validation/entropy_tau.py           # Works with CSV
python validation/navigation_compass.py    # Works with CSV

# All validation scripts use core/data_loader.py which:
#   1. First tries to load from CSV/JSON
#   2. Falls back to database if files not found
```

## Core Concepts

### Nouns = States
Nouns are positions in 16D semantic space:
```
|noun⟩ = |j, i, τ⟩
```

### Verbs = Transitions
Verbs are 6D operators that transform states:
```
V̂|noun₁⟩ = |noun₂⟩
```

### Adjectives = Qualities
Adjectives modify j-space (transcendentals):
```
Â|noun⟩ = |noun'⟩ with modified j
```

### Prefixes = Spin Operators
Prefixes flip direction while preserving τ:
```
un̂|happy⟩ = |unhappy⟩
τ preserved, s → -s
```

## Paradigm Shift

```
OLD (LLM):
  tokens → attention → P(next_token)
  Prediction by frequency. No understanding.

NEW (Quantum Semantic):
  state → navigate → trajectory → render
  Navigation by direction. Understanding.
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full architecture
- [THERMODYNAMICS.md](docs/THERMODYNAMICS.md) — Entropy theory
- [SPIN_OPERATORS.md](docs/SPIN_OPERATORS.md) — Prefix operators
- [SCRIPTS_INDEX.md](docs/SCRIPTS_INDEX.md) — All scripts documented
- [PROGRESS.md](docs/PROGRESS.md) — Development history

## Data Exports

| File | Contents |
|------|----------|
| `data/json/word_vectors.json` | 16D vectors + τ |
| `data/json/entropy_stats.json` | H_adj, H_verb per noun |
| `data/json/verb_operators.json` | 6D verb operators |
| `data/json/spin_pairs.json` | Prefix spin pairs |

## Validation Results

| Test | Result | Status |
|------|--------|--------|
| τ-entropy correlation | r = -0.99 | VALIDATED |
| Semantic opposites | 91% detected | VALIDATED |
| j ⊥ i orthogonality | mean \|r\| = 0.16 | VALIDATED |
| Compass navigation | t = 4.59 | VALIDATED |
| Spin τ conservation | 100% | VALIDATED |

## Database Tables

| Table | Description |
|-------|-------------|
| `hyp_bond_vocab` | Adjective\|noun bonds |
| `hyp_svo_triads` | Subject\|verb\|object triads |
| `hyp_semantic_index` | Final 16D word vectors |

## Requirements

**For reproduction (CSV-based):**
- Python 3.8+
- numpy, scipy
- Data files in `data/csv/` and `data/json/`

**For full pipeline (database):**
- PostgreSQL with bond data
- PyTorch
- spaCy (en_core_web_sm)
- psycopg2
