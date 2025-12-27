# Experiment: Learning Meaningful VerbOperator J-Vectors

## Problem Statement

Current VerbOperator j-vectors are NOT meaningful:
- All values negative (mean ≈ [-0.82, -0.97, -0.92, -0.80, -0.95])
- Opposite verbs have 99% cosine similarity (give ≈ take, create ≈ destroy)
- They don't capture semantic direction

**Goal**: Learn j-vectors that represent the *semantic direction* a verb pushes toward.

## Hypothesis

If verb "love" typically connects concepts with high j_love values,
then VerbOperator("love").j should point toward +love direction.

```
love: j = [+beauty, +life, +sacred, +good, +love]  (positive love dimension)
hate: j = [-beauty, -life, -sacred, -good, -love]  (negative love dimension)
```

## What We Need

### 1. Data: VIA Relationships with Source/Target J-Vectors

```cypher
MATCH (src:Concept)-[r:VIA]->(tgt:Concept)
WHERE src.j IS NOT NULL AND tgt.j IS NOT NULL
RETURN r.verb as verb, src.j as src_j, tgt.j as tgt_j
```

**Required**:
- Concepts with j-vectors (we have ~19K)
- VIA edges with verb labels (we have these)

### 2. Learning Algorithm: Verb Direction from Transitions

For each verb, compute the average direction it pushes:

```python
def learn_verb_direction(verb: str) -> np.ndarray:
    """Learn what direction this verb pushes toward."""

    # Get all transitions using this verb
    transitions = graph.query("""
        MATCH (src:Concept)-[r:VIA {verb: $verb}]->(tgt:Concept)
        WHERE src.j IS NOT NULL AND tgt.j IS NOT NULL
        RETURN src.j as src_j, tgt.j as tgt_j
    """, verb=verb)

    # Compute delta vectors (where does the verb push?)
    deltas = []
    for t in transitions:
        delta = np.array(t['tgt_j']) - np.array(t['src_j'])
        deltas.append(delta)

    # Average direction = characteristic push direction
    verb_j = np.mean(deltas, axis=0)

    # Normalize to unit vector
    verb_j = verb_j / np.linalg.norm(verb_j)

    return verb_j
```

### 3. Validation Metrics

**A) Opposite Verb Test**
```python
# These pairs should have NEGATIVE cosine similarity
opposite_pairs = [
    ('love', 'hate'),
    ('give', 'take'),
    ('create', 'destroy'),
    ('help', 'harm'),
    ('find', 'lose'),
    ('rise', 'fall'),
    ('open', 'close'),
    ('begin', 'end')
]

for v1, v2 in opposite_pairs:
    cos_sim = np.dot(j[v1], j[v2])
    assert cos_sim < 0, f"{v1} and {v2} should be opposite"
```

**B) Similar Verb Test**
```python
# These pairs should have POSITIVE cosine similarity
similar_pairs = [
    ('love', 'adore'),
    ('help', 'assist'),
    ('find', 'discover'),
    ('create', 'make'),
    ('understand', 'comprehend')
]

for v1, v2 in similar_pairs:
    cos_sim = np.dot(j[v1], j[v2])
    assert cos_sim > 0.5, f"{v1} and {v2} should be similar"
```

**C) Semantic Clustering**
```python
# Verbs should cluster by semantic category
positive_verbs = ['love', 'help', 'give', 'create', 'heal']
negative_verbs = ['hate', 'harm', 'take', 'destroy', 'hurt']

# Average j-vector of positive verbs should be in +good direction
positive_centroid = np.mean([j[v] for v in positive_verbs], axis=0)
assert positive_centroid[3] > 0  # good dimension positive

negative_centroid = np.mean([j[v] for v in negative_verbs], axis=0)
assert negative_centroid[3] < 0  # good dimension negative
```

### 4. Alternative Learning Approaches

**A) Delta-based (proposed above)**
```
verb_j = mean(tgt_j - src_j for all VIA edges with this verb)
```

**B) Target-weighted**
```
verb_j = weighted_mean(tgt_j, weight=edge_count)
```

**C) Contrastive learning**
```
Maximize: cos(verb_j, typical_target_j)
Minimize: cos(verb_j, random_target_j)
```

**D) From word embeddings**
```
verb_j = project(word2vec[verb], onto j-space)
```

## Experiment Steps

### Step 1: Data Collection
```python
# Count available data
n_concepts_with_j = count(Concept where j IS NOT NULL)
n_via_edges = count(VIA)
n_usable_edges = count(VIA where src.j AND tgt.j NOT NULL)
```

### Step 2: Learn J-Vectors
```python
for verb in all_verbs:
    verb_j = learn_verb_direction(verb)
    store_verb_j(verb, verb_j)
```

### Step 3: Validate
```python
run_opposite_verb_test()
run_similar_verb_test()
run_clustering_test()
```

### Step 4: Compare Intent Collapse
```python
# A/B test: old j-vectors vs learned j-vectors
result_old = laser.lase(seeds, intent_verbs, use_old_j=True)
result_new = laser.lase(seeds, intent_verbs, use_learned_j=True)

compare(result_old.coherence, result_new.coherence)
compare(result_old.intent_fraction, result_new.intent_fraction)
```

## Required Resources

| Resource | Status | Notes |
|----------|--------|-------|
| Concepts with j-vectors | ✅ Have | ~19K concepts |
| VIA edges with verbs | ✅ Have | Need to count usable |
| VerbOperator nodes | ✅ Have | 499 verbs |
| OPERATES_ON edges | ✅ Have | 4297 edges |
| Learning script | ❌ Need | To be implemented |
| Validation script | ❌ Need | To be implemented |

## Expected Outcomes

**If successful**:
- Opposite verbs will have negative cosine similarity
- Similar verbs will cluster together
- Intent collapse will be more semantically meaningful
- "love" will push toward +love, "hate" toward -love

**If unsuccessful**:
- J-space may not capture verb semantics
- May need different dimensionality
- May need different learning approach

## Next Steps

1. Count usable VIA edges (src.j AND tgt.j NOT NULL)
2. Implement delta-based learning
3. Run validation tests
4. If passing, update VerbOperator nodes
5. Re-run intent collapse experiments
