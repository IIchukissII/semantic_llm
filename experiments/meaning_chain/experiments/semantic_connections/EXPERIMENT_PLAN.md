# Semantic Connections: Parallel & Series Architecture

**Date**: 2026-01-01
**Status**: PLANNING

---

## 1. THEORETICAL FOUNDATION

### 1.1 Core Insight

Dynamic systems always involve oscillations and differential equations. If semantic space follows physics (PT1 dynamics, Boltzmann transitions), then concepts should be connectable in **parallel** and **series** like electrical components.

### 1.2 Electrical Circuit Analogy

```
SERIES CONNECTION:              PARALLEL CONNECTION:
A → B → C                       A ─┬─ B ─┐
                                   ├─ C ─┼─→ synthesis
R_total = R₁ + R₂ + R₃             └─ D ─┘

                                1/R_total = 1/R₁ + 1/R₂ + 1/R₃
```

### 1.3 Semantic Interpretation

| Circuit Property | Semantic Analog | Formula |
|-----------------|-----------------|---------|
| Resistance R | τ (abstraction level) | R = τ |
| Reactance X | j-vector projection | X = j · intent |
| Impedance Z | Complex semantic load | Z = R + jX |
| Capacitance C | Concept connectivity | C = degree(concept) |
| Inductance L | Semantic inertia | L = 1/variety |
| Current I | Meaning flow | I = transition_weight |
| Voltage V | Semantic potential | V = φ = λτ - μg |

### 1.4 Connection Rules

**SERIES (Sequential Deepening)**:
```python
# Coherence MULTIPLIES (like probabilities)
C_series = C₁ × C₂ × C₃

# τ ACCUMULATES
Δτ_series = Δτ₁ + Δτ₂ + Δτ₃

# Impedance ADDS
Z_series = Z₁ + Z₂ + Z₃
```

**PARALLEL (Simultaneous Expansion)**:
```python
# Coherence AVERAGES
C_parallel = (C₁ + C₂ + C₃) / 3

# τ AVERAGES (weighted)
τ_parallel = (w₁τ₁ + w₂τ₂ + w₃τ₃) / (w₁ + w₂ + w₃)

# Impedance: reciprocal sum
1/Z_parallel = 1/Z₁ + 1/Z₂ + 1/Z₃
```

---

## 2. OSCILLATION DYNAMICS

### 2.1 Resonance Frequency

```
ω₀ = 1/√(LC)

Where:
  L = semantic_inertia(concept) = 1 / adjective_variety
  C = semantic_capacity(concept) = degree(concept) / max_degree

Result:
  High L, High C → LOW ω₀  (deep concepts, slow response)
  Low L, Low C  → HIGH ω₀  (surface concepts, fast response)
```

### 2.2 Concept Classification by Frequency

| ω₀ Range | Concept Type | Examples | Response To |
|----------|--------------|----------|-------------|
| ω < 0.5 | Deep/Slow | love, god, truth | Philosophical questions |
| 0.5 ≤ ω < 1.0 | Medium | meaning, feeling | General questions |
| ω ≥ 1.0 | Fast/Surface | table, run, blue | Specific questions |

### 2.3 Resonance Matching

Maximum power transfer when:
```
ω_query ≈ ω₀_concept

# Query frequency from verb analysis
ω_query = f(intent_verbs)
  "understand" → low ω (slow, deep)
  "find" → high ω (fast, specific)
```

---

## 3. FILTER DESIGN

### 3.1 τ-Based Filters

```
LOW-PASS (GROUNDED):   Pass τ < τ_cutoff (concrete)
HIGH-PASS (DEEP):      Pass τ > τ_cutoff (abstract)
BAND-PASS (FOCUSED):   Pass τ_low < τ < τ_high
BAND-STOP (AVOID):     Block τ_low < τ < τ_high
```

### 3.2 Goal-to-Filter Mapping

| Goal | Filter Type | τ Range | Implementation |
|------|-------------|---------|----------------|
| GROUNDED | Low-pass | τ < 1.74 (n≤2) | Keep concrete |
| DEEP | High-pass | τ > 2.1 (n≥3) | Keep abstract |
| WISDOM | Band-pass | 1.3 < τ < 1.6 | Focus zone |
| POWERFUL | Band-pass | 2.5 < τ < 3.0 | Near Veil |
| ACCURATE | Band-pass | around τ_detected | Resonance |

### 3.3 Filter Transfer Function

```python
H(τ) = 1 / (1 + ((τ - τ_center) / bandwidth)^(2n))

Where:
  τ_center = target abstraction
  bandwidth = acceptable range
  n = filter order (sharpness)
```

---

## 4. COMPLEX IMPEDANCE MODEL

### 4.1 Concept as Impedance

```python
Z_concept = R + jX

Where:
  R = τ                           # Real part: abstraction resistance
  X = ||j|| × sign(j · intent)    # Imaginary: directional reactance
```

### 4.2 Impedance Matching

For maximum meaning transfer:
```python
Z_query* = Z_concept  # Conjugate match

# This means:
τ_query ≈ τ_concept           # Same abstraction level
j_query · j_concept > 0       # Same direction
```

### 4.3 Reflection Coefficient

```python
Γ = (Z_concept - Z_query) / (Z_concept + Z_query)

|Γ| → 0: Good match (meaning flows)
|Γ| → 1: Poor match (meaning reflects back)
```

---

## 5. EXPERIMENTAL DESIGN

### Experiment 1: Parallel Path Divergence

**Hypothesis**: Parallel paths capture multiple aspects of a concept.

**Setup**:
```python
# From seed "wisdom", find 3 diverse paths
paths = [
    navigate(wisdom → knowledge → truth),
    navigate(wisdom → experience → insight),
    navigate(wisdom → humility → acceptance)
]
```

**Metrics**:
- Path diversity: J(path_i, path_j) < 0.3 (low Jaccard)
- Synthesis quality: coherence of merged concepts
- Coverage: unique concepts / total concepts

**Prediction**: Parallel paths will have higher coverage and synthesis quality than single linear path.

---

### Experiment 2: Series Depth Accumulation

**Hypothesis**: Series connections accumulate depth (Δτ).

**Setup**:
```python
# Linear chain through τ levels
chain = navigate(concrete → grounded → meaningful → abstract)
# e.g., table → furniture → category → concept → idea
```

**Metrics**:
- Δτ per step: τ[i+1] - τ[i]
- Coherence decay: C_series = Π(C_step)
- Depth ratio: final_τ / initial_τ

**Prediction**: Coherence multiplies (decays), but depth accumulates.

---

### Experiment 3: Resonance Frequency Spectrum

**Hypothesis**: Concepts have characteristic resonance frequencies.

**Setup**:
```python
# For each concept, measure L and C
for concept in graph.all_concepts():
    L = 1 / concept.variety
    C = concept.degree / max_degree
    omega_0 = 1 / sqrt(L * C)
    spectrum[concept] = omega_0
```

**Metrics**:
- ω₀ distribution across concepts
- Correlation with τ levels
- Clustering by frequency bands

**Prediction**: ω₀ will correlate inversely with τ (abstract = low frequency).

---

### Experiment 4: Filter Response Curves

**Hypothesis**: Goal-based navigation acts as τ-filter.

**Setup**:
```python
# Run same query with different goals
query = "What is love?"
for goal in [GROUNDED, BALANCED, DEEP]:
    result = navigate(query, goal=goal)
    tau_distribution[goal] = [c.tau for c in result.concepts]
```

**Metrics**:
- τ distribution shape per goal
- Filter bandwidth (τ_max - τ_min)
- Center frequency (τ_mean)

**Prediction**: GROUNDED = low-pass, DEEP = high-pass, BALANCED = band-pass.

---

### Experiment 5: Impedance Matching Quality

**Hypothesis**: Better Z-match = better response quality.

**Setup**:
```python
# Compute Z for query and each result concept
Z_query = compute_impedance(query)
for concept in result.concepts:
    Z_concept = compute_impedance(concept)
    gamma = reflection_coefficient(Z_query, Z_concept)
    match_quality[concept] = 1 - abs(gamma)
```

**Metrics**:
- Average match quality
- Correlation with subjective quality rating
- Correlation with existing quality metrics (R, C, S, P)

**Prediction**: High match quality correlates with high coherence.

---

### Experiment 6: Parallel vs Series Comparison

**Hypothesis**: Different connection types optimal for different goals.

**Setup**:
```python
questions = ["What is wisdom?", "What is love?", ...]
for q in questions:
    series_result = navigate_series(q, depth=5)
    parallel_result = navigate_parallel(q, n_paths=3)
    compare(series_result, parallel_result)
```

**Metrics**:
- Coherence: series vs parallel
- Coverage: unique concepts
- Quality scores: R, C, D, S, P
- Subjective evaluation

**Prediction**:
- Series better for DEEP goal (depth accumulation)
- Parallel better for STABLE goal (multiple perspectives)
- Hybrid best for WISDOM goal

---

## 6. IMPLEMENTATION PLAN

### Phase 1: Core Infrastructure (Week 1)

```
experiments/semantic_connections/
├── __init__.py
├── connection_types.py      # Series, Parallel, Hybrid classes
├── impedance.py             # Z = R + jX calculations
├── filters.py               # Low-pass, high-pass, band-pass
├── oscillators.py           # ω₀, resonance, L, C
└── results/
```

### Phase 2: Experiments (Week 2)

```
├── exp1_parallel_divergence.py
├── exp2_series_depth.py
├── exp3_resonance_spectrum.py
├── exp4_filter_response.py
├── exp5_impedance_matching.py
├── exp6_parallel_vs_series.py
```

### Phase 3: Integration (Week 3)

```
chain_core/
├── parallel_navigator.py    # New navigator mode
├── connection_router.py     # Choose series/parallel dynamically
```

### Phase 4: Validation & Documentation

```
├── EXPERIMENT_RESULTS.md
docs/
├── CONNECTION_SEMANTICS.md
├── OSCILLATION_DYNAMICS.md
```

---

## 7. SUCCESS CRITERIA

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Parallel coverage | > 2x linear | More unique concepts |
| Series depth | Δτ > 1.0 | Meaningful depth increase |
| Resonance correlation | R² > 0.7 | ω₀ predicts response |
| Filter precision | 80% in band | Goal hits target τ range |
| Impedance correlation | R² > 0.6 | Z-match predicts quality |

---

## 8. RISKS & MITIGATIONS

| Risk | Probability | Mitigation |
|------|-------------|------------|
| L/C not measurable | Medium | Use proxy metrics (variety, degree) |
| No resonance effect | Medium | May need oscillation over dialogue turns |
| Parallel paths redundant | Low | Ensure diverse seeds |
| Computation too slow | Low | Cache impedance values |

---

## 9. NEXT STEPS

1. [ ] Create `connection_types.py` with Series/Parallel classes
2. [ ] Create `impedance.py` with Z calculations
3. [ ] Run Experiment 1 (parallel divergence)
4. [ ] Analyze results
5. [ ] Iterate on theory

---

## 10. THEORETICAL IMPLICATIONS

If validated, this framework provides:

1. **Architectural insight**: Semantic navigation = circuit design
2. **New goal type**: RESONANT (match query frequency)
3. **Quality metric**: Impedance match quality
4. **Synthesis method**: Parallel path merging
5. **Depth control**: Series chain length

The electrical analogy becomes not metaphor but **isomorphism**.
