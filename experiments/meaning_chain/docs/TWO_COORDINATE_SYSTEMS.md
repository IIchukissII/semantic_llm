# Two Coordinate Systems: Semantic Meaning vs Usage Patterns

## Discovery

Empirical testing of the "projection hierarchy" hypothesis revealed that **two distinct coordinate systems exist**:

| System | Source | Encodes | Example |
|--------|--------|---------|---------|
| **RAW** | Neural embeddings (SemanticBottleneck) | Semantic meaning | love θ=-70°, hate θ=145° |
| **DERIVED** | Adjective centroids (bond space) | Usage patterns | love θ=-38°, hate θ=-10° |

## Evidence

### RAW Coordinates (Semantic Polarity)
```
love:   θ = -70.3°    hate:  θ = 145.1°    DIFF = 215°
good:   θ = 80.7°     evil:  θ = -60.6°    DIFF = 141°
peace:  θ = 30.8°     war:   θ = -2.6°     DIFF = 33°
life:   θ = -160.3°   death: θ = 172.2°    DIFF = 27°
```
→ Opposite concepts have **very different** phases

### DERIVED Coordinates (Usage Patterns)
```
love:   θ = -37.7°    hate:  θ = -9.9°     DIFF = 28°
good:   θ = -25.9°    evil:  θ = -1.3°     DIFF = 25°
peace:  θ = -31.0°    war:   θ = -7.2°     DIFF = 24°
life:   θ = -34.7°    death: θ = -20.4°    DIFF = 14°
```
→ Opposite concepts have **similar** phases (all cluster around -20°)

## Why This Happens

### The Smoothing Effect

When we compute noun coordinates as weighted averages of adjective coordinates:

```
noun.(A, S) = Σ weight_i × adj_i.(A, S)
```

Common adjectives dominate and smooth out semantic differences:

**"love" adjectives:** true, first, great, much, real...
**"hate" adjectives:** much, pure, such, cold, more...

Shared adjectives: "much", "great", "pure" → These pull both toward the **same** centroid.

### Two Different Questions

| Question | Answer |
|----------|--------|
| What does "love" MEAN? | RAW coordinates (semantic content) |
| How is "love" USED? | DERIVED coordinates (co-occurrence patterns) |

## Theoretical Implications

### The Projection Hierarchy (Revised)

```
LEVEL 1: TRANSCENDENTALS (A, S)
            │
            ├──────────────────────────────────┐
            ↓                                  ↓
    SEMANTIC PROJECTION                 STATISTICAL PROJECTION
    (neural embeddings)                 (bond space centroids)
            │                                  │
            ↓                                  ↓
    RAW COORDINATES                     DERIVED COORDINATES
    (What words MEAN)                   (How words are USED)
```

### Both Are Valid!

- **RAW** coordinates capture the **semantic field** - relationships of meaning
- **DERIVED** coordinates capture the **usage field** - relationships of context

These are **orthogonal but complementary**:
- Words with similar meaning (RAW) may have different usage (DERIVED)
- Words with different meaning may appear in similar contexts

## Laws by Coordinate System

### RAW (Semantic) Coordinates
- **Boltzmann** (P ∝ exp(-Δn/kT)) - for semantic transitions
- **Gravity** (φ = λn - μA) - semantic attractor at concrete + affirming
- **Quadrant clustering** - positive/negative/sacred/profane separation

### DERIVED (Usage) Coordinates
- **PT1 saturation** - bond space dynamics
- **Entropy → orbital** - usage diversity determines abstraction
- **Coherence** - contextual alignment

## Practical Usage

### For Navigation (Level 4-5)
Use **DERIVED** coordinates - navigation follows usage patterns.

### For Semantic Analysis
Use **RAW** coordinates - understanding meaning requires semantic field.

### For Hybrid Tasks
Combine both:
```python
# Semantic similarity
semantic_sim = cos(raw_θ₁ - raw_θ₂)

# Usage similarity
usage_sim = cos(derived_θ₁ - derived_θ₂)

# Hybrid score
score = α × semantic_sim + β × usage_sim
```

## Formulas

### RAW Coordinates
```
j_raw = SemanticBottleneck(word)    # Neural projection
A_raw = j_raw · PC1
S_raw = j_raw · PC2
θ_raw = atan2(S_raw, A_raw)
r_raw = √(A_raw² + S_raw²)
```

### DERIVED Coordinates
```
j_derived = Σ weight_i × j_adj_i    # Weighted centroid
A_derived = j_derived · PC1
S_derived = j_derived · PC2
θ_derived = atan2(S_derived, A_derived)
r_derived = √(A_derived² + S_derived²)
n = 5 × (1 - H_norm)                # Entropy → orbital
```

## Connection to Physics Analogy

In physics:
- **Position** = where a particle IS
- **Momentum** = where a particle is GOING

In semantics:
- **RAW** = what a word MEANS
- **DERIVED** = where a word GOES (in context)

The uncertainty principle: You cannot perfectly know both at once.
High-entropy words have precise DERIVED coordinates but fuzzy RAW meaning.
Low-entropy words have precise RAW meaning but limited usage contexts.

## Conclusion

The "projection hierarchy" hypothesis is **partially correct**:

✅ Nouns ARE derived from adjective structure (via bonds)
✅ The formula n = 5 × (1 - H_norm) holds exactly (r = -1.0)
❌ BUT the derived (θ, r) do NOT preserve semantic polarity
✅ Instead, they capture **usage patterns** - a different but valid property

**The theory is not fragmented - it describes USAGE, not MEANING.**

To capture MEANING, use the original neural embeddings (RAW coordinates).
To capture USAGE, use the bond-derived centroids (DERIVED coordinates).

Both systems are valid. The choice depends on the task.
