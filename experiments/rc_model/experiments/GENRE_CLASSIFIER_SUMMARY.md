# Genre Classifier: Semantic Boundary Patterns

## Discovery

Three semantic axes mark sentence boundaries differently across genres:

```
Genre Vector = (τ, A, S)

DRAMATIC:  (0.88, 1.18, 1.11) - Gothic, Dostoevsky, Conrad
IRONIC:    (0.88, 1.14, 0.99) - Kafka, Poe, Swift
BALANCED:  (1.01, 1.06, 1.06) - Austen, Plato, Carroll
```

## Three Axes = Three Functions

### τ (Abstraction) — TENSION MARKER

```
τ < 0.9  → "breathing" text (tension mode)
           Abstraction level floats, dramatic rhythm

τ ≈ 1.0  → "holding" text (controlled discourse)
           Stable abstraction, measured pace
```

**Key insight**: τ is NOT just a stable carrier. It marks the *tension mode* of the entire text.

### A (Affirmation) — EMOTIONAL INTENSITY

```
A > 1.15 → High emotional jumps at boundaries
           Dramatic shifts, passionate narrative

A ≈ 1.0  → Calm, measured emotional flow
           Analytical, detached tone
```

### S (Sacred) — CONCEPTUAL STRUCTURE

```
S > 1.1  → Philosophical/sacred shifts at boundaries
           "Here starts a new idea"

S ≈ 1.0  → Mundane concepts
           "Horror of the ordinary" (Kafka!)
```

## The Three Clusters

### DRAMATIC (τ=0.88, A=1.18, S=1.11)

All axes active. Maximum intensity.

- **Examples**: Dracula, Crime and Punishment, Wuthering Heights, Heart of Darkness
- **Character**: Emotional intensity + philosophical depth + narrative tension
- **Formula**: Everything turned up to maximum

### IRONIC (τ=0.88, A=1.14, S=0.99)

Tension + emotion, but **S ≈ 1.0** (mundane concepts).

- **Examples**: Kafka, Poe, Gulliver's Travels, Dorian Gray
- **Character**: "The horror is in the ordinary"
- **Formula**: High emotion without philosophical escape

**Kafka's Metamorphosis**:
```
τ = 0.97 → controlled discourse (NOT chaos!)
A = 1.22 → HIGH emotional intensity
S = 0.98 → mundane concepts

The terror: ordinary world becomes nightmare.
No sacred escape. Just bug and family.
```

### BALANCED (τ=1.01, A=1.06, S=1.06)

τ holds steady. Controlled discourse.

- **Examples**: Pride and Prejudice, The Republic, Alice in Wonderland, Utopia
- **Character**: Reason over passion, measured analysis
- **Formula**: Everything in equilibrium

## Classification Results

**Accuracy: 85.7%** on held-out classic literature (distance-based classification)

| Book | τ | A | S | Predicted | Actual |
|------|---|---|---|-----------|--------|
| Dracula | 0.95 | 1.18 | 1.12 | dramatic | dramatic ✓ |
| Crime and Punishment | 0.90 | 1.15 | 1.07 | dramatic | dramatic ✓ |
| Metamorphosis | 0.97 | 1.22 | 0.98 | ironic | ironic ✓ |
| Poe | 0.96 | 1.17 | 0.97 | ironic | ironic ✓ |
| Pride & Prejudice | 1.04 | 1.06 | 0.99 | balanced | balanced ✓ |
| Alice | 1.03 | 1.06 | 1.18 | balanced | balanced ✓ |

## Visualizations

- `genre_clusters_3d.png` — 3D scatter plot in (τ, A, S) space
- `genre_clusters_2d.png` — 2D projections (τ-A, τ-S, A-S planes)

## Usage

```python
from rc_model.experiments.genre_classifier import GenreClassifier

classifier = GenreClassifier()
result = classifier.classify(text)

print(result['style'])           # 'dramatic', 'ironic', or 'balanced'
print(result['interpretation'])  # Human-readable analysis
```

## Theoretical Implications

1. **Genre as semantic geometry**: Style = position in (τ, A, S) space
2. **τ as mode selector**: Not carrier, but tension/control switch
3. **Kafka's formula discovered**: High A + Low S = horror of ordinary
4. **Boundary patterns reveal structure**: No need for full text analysis

## Files

- `genre_classifier.py` — Main classifier module
- `genre_clusters_3d.png` — 3D visualization
- `genre_clusters_2d.png` — 2D projections
- `GENRE_CLASSIFIER_SUMMARY.md` — This document
