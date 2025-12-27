# Verb Dual Structure: The Pirate Insight

## Discovery Summary

Verbs have the same dual structure as nouns, with parallel properties that enable meaningful semantic operations.

## The Problem

VerbOperator j-vectors were all biased toward a global mean:
```
global_mean ≈ [-0.82, -0.97, -0.92, -0.80, -0.95]
```

This made opposite verbs have 99% cosine similarity:
- love/hate: 0.99 (should be opposite!)
- create/destroy: 0.99 (should be opposite!)
- give/take: 0.99 (should be different!)

## The Pirate Insight: "Shift the Phase"

Instead of using raw j-vectors, we **center** them by subtracting the global mean:

```python
i_vector = j_vector - global_mean  # The "phase shift"
```

This is mathematically equivalent to:
- `cos(θ) → cos(θ - φ)` where φ is the bias angle
- Using sine instead of cosine (sin = cos shifted by 90°)

## Results After Phase Shift

| Verb Pair | Before | After | Effect |
|-----------|--------|-------|--------|
| rise/fall | 0.97 | **-0.25** | OPPOSITE |
| create/destroy | 0.99 | **0.04** | ORTHOGONAL |
| find/lose | 0.99 | **0.13** | ORTHOGONAL |
| begin/end | 0.99 | **0.20** | ORTHOGONAL |
| give/take | 0.99 | 0.75 | Similar (both transfer verbs!) |

## The Dual Structure

### Nouns
```
τ (tau)     = abstraction level (scalar)
g           = good/evil polarity (scalar)
j           = 5D position in meaning space (vector)
```

### Verbs (Parallel Structure)
```
Δτ (delta_tau)  = abstraction EFFECT (↑ ascend / ↓ descend)
Δg (delta_g)    = moral PUSH direction
i               = intrinsic action type (centered j-vector)
j               = effect direction (raw j-vector)
```

## Verb Properties Explained

### i-vector (Intrinsic Action Type)
The centered j-vector represents *what kind of action* the verb is:

| Verb | i-dominant | Meaning |
|------|-----------|---------|
| love | +life | Life-affirming action |
| help | +life | Life-affirming action |
| create | +sacred | Sacred/generative action |
| understand | +sacred | Sacred/cognitive action |
| find | +love | Connection-seeking action |
| learn | -love | Knowledge-seeking (away from sentiment) |
| give/take | -beauty | Transfer actions (utility over aesthetics) |

### Δτ (Delta Tau) - Abstraction Effect
How the verb transforms abstraction level:

| Verb | Δτ | Effect |
|------|-----|--------|
| love | -0.41 | **GROUNDS** - brings abstract to concrete |
| find | -0.13 | Materializes, makes real |
| give | -0.13 | Manifests through sharing |
| hate | +0.16 | **ABSTRACTS** - ideologizes |
| create/destroy | ±0.07 | Stable - transforms at same level |

### Δg (Delta G) - Moral Push
Direction the verb pushes in good/evil space:

| Verb | Δg | Interpretation |
|------|-----|----------------|
| destroy | +0.69 | Targets evil (destroys bad things) |
| find | -0.36 | Finds morally complex things |
| love | -0.64 | Loves flawed beings (redemptive) |

## Orthogonal Verbs

**Key insight**: Orthogonal verbs (like create/destroy at 87.8° angle) are not opposites - they operate in **perpendicular semantic dimensions**.

- **create** (i=+sacred): Operates in the sacred/generative dimension
- **destroy** (i=+good): Operates in the moral dimension

They're different *kinds* of actions, not opposite actions.

## Neo4j Schema

### VerbOperator Node (Updated)
```cypher
(:VerbOperator {
  verb: "love",

  // Original
  j: [-0.69, -0.84, -1.02, -0.68, -0.82],  // Raw 5D direction
  magnitude: 1.0,

  // New (Phase-Shifted)
  i: [+0.13, +0.13, -0.10, +0.12, +0.13],  // Centered j-vector
  delta_tau: -0.41,                         // Abstraction effect
  delta_g: -0.64,                           // Moral push
  transition_count: 76,                     // VIA edge count
  phase_shifted: true                       // Flag for new properties
})
```

### GlobalMean Node
```cypher
(:GlobalMean {
  name: 'verb_j_mean',
  j_mean: [-0.82, -0.97, -0.92, -0.80, -0.95],
  updated_at: datetime()
})
```

## Usage in IntentCollapse

The `IntentCollapse` class now uses phase-shifted vectors:

```python
class IntentCollapse:
    def _center_j(self, j: np.ndarray) -> np.ndarray:
        """Center j-vector by subtracting global mean (phase shift)."""
        return j - self._global_j_mean

    def _compute_intent_direction(self):
        """Uses CENTERED j-vectors for meaningful intent direction."""
        for op in self.intent_operators.values():
            centered_j = self._center_j(op.j_vector)
            combined_j += weight * centered_j
```

## Queries

### Find verbs that GROUND (descend in abstraction)
```cypher
MATCH (v:VerbOperator)
WHERE v.delta_tau < -0.2
RETURN v.verb, v.delta_tau, v.i
ORDER BY v.delta_tau ASC
```

### Find life-affirming verbs
```cypher
MATCH (v:VerbOperator)
WHERE v.i[1] > 0.1  // life dimension is index 1
RETURN v.verb, v.i[1] as life_score
ORDER BY life_score DESC
```

### Find orthogonal verb pairs
```cypher
MATCH (v1:VerbOperator), (v2:VerbOperator)
WHERE v1.verb < v2.verb
  AND v1.i IS NOT NULL AND v2.i IS NOT NULL
WITH v1, v2,
     reduce(dot = 0.0, i IN range(0, 4) | dot + v1.i[i] * v2.i[i]) as dot_product,
     sqrt(reduce(s = 0.0, x IN v1.i | s + x*x)) as norm1,
     sqrt(reduce(s = 0.0, x IN v2.i | s + x*x)) as norm2
WITH v1.verb as verb1, v2.verb as verb2,
     dot_product / (norm1 * norm2) as cosine
WHERE abs(cosine) < 0.2  // Near orthogonal
RETURN verb1, verb2, cosine
ORDER BY abs(cosine) ASC
LIMIT 20
```

## Future Work

1. **Learn verb properties from transitions**: Instead of computing from current VIA edges, learn optimal i-vectors that maximize predictive power

2. **Verb composition**: How do sequential verbs combine? (find + destroy = ?)

3. **Verb-noun interaction**: How does verb i-vector interact with noun j-vector during transition?

4. **Temporal verbs**: Do past/present/future tenses have different Δτ effects?

5. **Intent prediction**: Given a partial query, predict intent verbs from semantic trajectory

## References

- Original insight: "perhaps we should act like pirates - sin or tan similarity"
- Mathematical basis: Phase shifting in Fourier analysis
- Implementation: `chain_core/intent_collapse.py`
- Storage script: `scripts/store_verb_properties.py`
- Test script: `experiments/verb_operator_learning/phase_shift_j.py`

---

*Discovered 2025-12-27 during meaning_chain development*
