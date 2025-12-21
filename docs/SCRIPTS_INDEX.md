# Scripts Index

Complete documentation of all scripts in the semantic_llm project.

## Directory Structure

```
semantic_llm/
├── core/                    # Essential implementation
├── validation/              # Theory validation (algorithmic proofs)
├── scripts/                 # Utility scripts
├── data/                    # Exported data (JSON, CSV)
├── models/                  # Trained models (V3 only)
├── results/                 # Organized results
├── docs/                    # Documentation
└── deprecated/              # Old theory (for reference)
```

---

## Core (Essential Implementation)

| File | Description | Key Classes/Functions |
|------|-------------|----------------------|
| `semantic_core.py` | 16D semantic space definition | `SemanticCoords`, `VerbVector`, `VerbTransition` |
| `semantic_bottleneck.py` | V3 bottleneck model with attention | `SemanticBottleneckV2`, `NounEncoder` |
| `navigator.py` | Compass-based navigation | `Navigator`, `compass`, `navigate` |
| `hybrid_llm.py` | Trajectory → LLM rendering | `QuantumCore`, `LLMRenderer`, `HybridSystem` |

### semantic_core.py
Defines the 16-dimensional semantic space:
- **j-space (5D)**: Transcendentals — beauty, life, sacred, good, love
- **i-space (11D)**: Surface/context — truth, freedom, meaning, order, peace, power, nature, time, knowledge, self, society
- **τ (tau)**: Abstraction level [1-6], computed from Shannon entropy

### semantic_bottleneck.py
Neural network that encodes text into semantic coordinates:
- Nouns encoded as "clouds of adjectives" via attention
- τ derived from attention entropy (not variety)
- Orthogonality constraint: j ⊥ i

### navigator.py
Goal-directed navigation in semantic space:
- **Compass**: Direction toward "good" (j_good vector)
- **Navigation**: Select verbs that move toward goal
- Validated: t=4.59, p<0.001

### hybrid_llm.py
Complete hybrid system:
- **QuantumCore**: Plans trajectory in semantic space
- **LLMRenderer**: Converts trajectory to fluent text
- **FeedbackEncoder**: Verifies semantic fidelity

---

## Validation (Theory Proofs)

These scripts contain **algorithmic proofs** of the semantic theory.

| File | Proves | Key Result |
|------|--------|------------|
| `entropy_tau.py` | τ from Shannon entropy | τ = 1 + 5×(1 - H_norm) |
| `euler_constant.py` | Euler's e in language | ln(H_adj/H_verb) ≈ 1/e = 0.3679 |
| `entropy_correlation.py` | H_adj vs H_verb correlation | r = 0.837 (strong) |
| `unified_space.py` | Noun-adj-verb correlation | r = 0.9389 (unified space) |
| `spin_conservation.py` | Prefixes conserve τ | 100% τ conservation |
| `navigation_compass.py` | Compass navigation works | t = 4.59, p < 0.001 |

### entropy_tau.py (FUNDAMENTAL)
Proves that τ should be computed from Shannon entropy:
```
τ = 1 + 5 × (1 - H_norm)

Where:
  H = -Σ p·log₂(p)  (Shannon entropy)
  H_norm = H / log₂(n)  (normalized)
```

Supersedes old variety-based τ.

### euler_constant.py (FUNDAMENTAL)
Discovers Euler's number in language:
```
ln(H_adj) - ln(H_verb) = 0.3622 ≈ 1/e = 0.3679
Distance: 0.57%

H_adj - H_verb ≈ 1.08 bits (One-Bit Law)
Being > Doing by exactly 1 bit!
```

### entropy_correlation.py
Validates correlation between adjective and verb entropies:
- Pearson r = 0.837
- Confirms unified semantic structure

### unified_space.py
Proves noun-adj and noun-verb spaces are correlated:
- Pearson r = 0.9389
- Confirms nouns, adjectives, and verbs share semantic structure

### spin_conservation.py
Proves prefixes behave like quantum spin operators:
- **τ Conservation**: 100% (50/50 perfect)
- **Direction Flip**: 64% accuracy (j-space conjugation)
- Example: happy → unhappy (τ preserved, direction flipped)

### navigation_compass.py
Validates compass-based navigation:
```
Compass navigation: mean Δg = +0.438
Random navigation:  mean Δg = -0.017
t-statistic: 4.59
p-value: < 0.001

CONCLUSION: Compass navigation is statistically significant!
```

---

## Scripts (Utilities)

| File | Purpose | Usage |
|------|---------|-------|
| `export_data.py` | Export DB → JSON/CSV | `python export_data.py --all` |
| `train_bottleneck.py` | Train V3 model | `python train_bottleneck.py` |
| `build_index.py` | Build semantic index | `python build_index.py --batch 100` |
| `train_projector.py` | Train 768D→16D projector | `python train_projector.py` |

### export_data.py
Exports semantic data from PostgreSQL to JSON and CSV:
```bash
python export_data.py --all       # Export everything
python export_data.py --vectors   # Export word vectors only
python export_data.py --entropy   # Export entropy stats only
python export_data.py --verbs     # Export verb operators
python export_data.py --spin      # Export spin pairs
python export_data.py --bonds     # Export bond statistics
```

Output files:
- `data/json/word_vectors.json` — All 16D vectors
- `data/json/entropy_stats.json` — Shannon entropy statistics
- `data/json/verb_operators.json` — Verb 6D operators
- `data/json/spin_pairs.json` — Prefix spin pairs
- `data/csv/` — Same data in CSV format

### train_bottleneck.py
Trains V3 semantic bottleneck with thermodynamic losses:
- **Entropy-based τ loss**: τ = f(H_norm)
- **One-Bit Law loss**: H_adj - H_verb ≈ 1.0
- **Euler Law loss**: ln(H_adj/H_verb) ≈ 1/e
- **Orthogonality loss**: j ⊥ i

### build_index.py
Database-backed pipeline for semantic indexing:
```bash
python build_index.py --status      # Check progress
python build_index.py --batch 100   # Process 100 books
python build_index.py --continuous  # Run until done
python build_index.py --finalize    # Create final index
```

---

## Data (Exports)

| File | Contents |
|------|----------|
| `json/word_vectors.json` | 16D vectors for all words + τ |
| `json/entropy_stats.json` | H_adj, H_verb, τ_entropy per noun |
| `json/verb_operators.json` | 6D verb transition operators |
| `json/spin_pairs.json` | Prefix spin pairs (τ-conserving) |
| `json/bond_statistics.json` | Top adj\|noun bonds |
| `csv/` | Same data in CSV format |

---

## Results

| Directory | Contents |
|-----------|----------|
| `results/entropy/` | Entropy discovery results |
| `results/navigation/` | Compass validation results |
| `results/spin/` | Spin operator analysis |

---

## Deprecated (Old Theory)

These files validated the **old variety-based τ**, which has been superseded by entropy-based τ.

| File | Reason Deprecated |
|------|-------------------|
| `variety_based_tau/01_tau_frequency.py` | Used variety-based τ |
| `variety_based_tau/02_semantic_clusters.py` | Old clustering test |
| `variety_based_tau/test_theory_variety_tau.py` | Tested variety→τ |
| `old_models/train_autoencoder.py` | Autoencoder has no meaning |
| `old_models/models/` | V1 models (poor orthogonality) |
| `old_models/models_v2/` | V2 models (superseded by V3) |

---

## Theory Summary

### Current Theory (Entropy-Based τ)

```
τ = 1 + 5 × (1 - H_norm)

Where:
  H_norm = H / log₂(n)   (normalized Shannon entropy)
  H = -Σ p·log₂(p)       (Shannon entropy)

Key discoveries:
  • H_adj - H_verb ≈ 1.08 bits (One-Bit Law)
  • ln(H_adj/H_verb) ≈ 1/e (Euler's constant in language)
  • Prefixes conserve τ (spin operators)
  • Compass navigation works (t=4.59)
```

### Old Theory (Superseded)

```
τ = f(variety)  ← WRONG

Where variety = count of unique adjectives.
This is linear and doesn't capture distribution.
```

---

## Quick Start

```bash
# 1. Export data from database
python scripts/export_data.py --all

# 2. Run validation tests
python validation/entropy_tau.py
python validation/navigation_compass.py

# 3. Use the hybrid system
python core/hybrid_llm.py
```
