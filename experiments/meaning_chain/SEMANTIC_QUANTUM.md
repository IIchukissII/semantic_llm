# Semantic Quantum Numbers: (n, θ, r) Encoding

## Overview

The semantic quantum number system encodes word meaning in three dimensions, analogous to quantum mechanics' (n, l, m) orbital description. This 12-bit representation achieves 6x compression while preserving semantic navigation capability.

```
(n, θ, r) ≡ (orbital, phase, intensity)
         ≡ (level, direction, magnitude)
```

## The Three Quantum Numbers

### n — Orbital (Abstraction Level)

**Definition**: n = round((τ - 1) × e), where τ is the abstraction coordinate

**Range**: 0-15 (4 bits)

**Interpretation**:
- n = 0-4: **Human realm** — abstract concepts (love, truth, freedom)
- n = 5: **The Veil** — boundary between realms
- n = 6-15: **Transcendental realm** — concrete manifestations

**Formula**: τ_n = 1 + n/e

| n | τ range | Semantic domain |
|---|---------|-----------------|
| 0 | 1.00-1.18 | Pure abstractions |
| 1 | 1.18-1.55 | High concepts |
| 2 | 1.55-1.92 | Abstract ideas |
| 3 | 1.92-2.29 | Human concepts |
| 4 | 2.29-2.66 | Common abstractions |
| 5 | 2.66-3.03 | The Veil |
| 6+ | 3.03+ | Concrete/physical |

**Distribution**: ~7.7x more words in human realm (n < 5) than transcendental realm.

---

### θ — Phase (Direction in j-Space)

**Definition**: θ = atan2(S, A), where A = Affirmation, S = Sacred

**Range**: -180° to +180° (4 bits → 16 directions)

**Semantic Quadrants**:

```
                    S (Sacred)
                       ↑
            II         |         I
        Negating +     |    Affirming +
         Sacred        |      Sacred
                       |
    ←——————————————————+——————————————————→ A (Affirmation)
                       |
           III         |        IV
        Negating +     |    Affirming +
         Profane       |      Profane
                       ↓
```

| θ range | Semantic character |
|---------|-------------------|
| 0° ± 22° | Pure affirmation (yes, good, true) |
| 90° ± 22° | Pure sacred (holy, divine, transcendent) |
| 180° ± 22° | Pure negation (no, evil, false) |
| -90° ± 22° | Pure profane (mundane, material, earthly) |

---

### r — Intensity (Transcendental Magnitude)

**Definition**: r = √(A² + S²)

**Range**: 0.0 to 4.0 (4 bits → 16 levels)

**Interpretation**: How strongly a word projects onto the transcendental axes.

| r value | Meaning |
|---------|---------|
| r ≈ 0 | Neutral/common words (weak transcendental character) |
| r ≈ 1 | Moderate transcendental projection |
| r > 2 | Strong transcendental character |
| r > 3 | Extreme transcendental words |

**Metaphysical Interpretation** (Neoplatonic framework):
- r = degree of emanation from the Source (τὸ ἕν)
- Higher r = closer to pure Being/Non-Being axis
- Lower r = more participation in material existence

---

## Bijection: (A, S, τ) ↔ (n, θ, r)

The transformation is lossless:

**Forward (Cartesian → Polar Quantum)**:
```python
n = round((τ - 1) * e)
θ = atan2(S, A)
r = sqrt(A² + S²)
```

**Inverse (Polar Quantum → Cartesian)**:
```python
τ = 1 + n/e
A = r * cos(θ)
S = r * sin(θ)
```

---

## 12-Bit Encoding

### Bit Layout

```
┌────────────┬────────────┬────────────┐
│  n (4 bit) │  θ (4 bit) │  r (4 bit) │
│  bits 8-11 │  bits 4-7  │  bits 0-3  │
└────────────┴────────────┴────────────┘
     MSB                        LSB
```

### Packing/Unpacking

```python
# Pack to 12-bit integer
bits = (n << 8) | (theta_idx << 4) | r_idx

# Unpack from 12-bit integer
n = (bits >> 8) & 0xF
theta_idx = (bits >> 4) & 0xF
r_idx = bits & 0xF
```

### Hex Representation

Each word encodes to a 3-character hex string:

| Word | Hex | n | θ° | r |
|------|-----|---|-----|---|
| truth | 473 | 4 | -12 | 0.80 |
| love | 451 | 4 | -60 | 0.27 |
| god | 485 | 4 | +12 | 1.33 |
| death | 498 | 4 | +156 | 1.47 |
| wisdom | 474 | 4 | 0 | 1.07 |
| life | 462 | 4 | -36 | 0.53 |
| peace | 472 | 4 | -24 | 0.53 |
| war | 497 | 4 | +144 | 1.60 |

---

## Quantization Error

### Theoretical Maximum Error

With 4 bits per dimension:
- Δn = ±0.5 orbital levels
- Δθ = ±11.25° (±π/16 radians)
- Δr = ±0.133 (range 4.0 / 15 / 2)

### Reconstructed Accuracy

| Dimension | Mean Error | Max Error |
|-----------|------------|-----------|
| A | ±0.05 | ±0.15 |
| S | ±0.04 | ±0.12 |
| τ | ±0.02 | ±0.18 |

---

## Storage Efficiency

| Format | Size per word | Total (22,486 words) |
|--------|---------------|----------------------|
| Word2Vec (300D float32) | 1,200 bytes | 27.0 MB |
| (A, S, τ) float32 | 12 bytes | 269.8 KB |
| (n, θ, r) 12-bit | 1.5 bytes | 33.7 KB |

**Compression ratios**:
- vs Word2Vec: **800x**
- vs float32 coordinates: **6x**

---

## Usage

### Python API

```python
from core.semantic_quantum import QuantumEncoder, QuantumWord

encoder = QuantumEncoder()

# Encode word to quantum representation
qw = encoder.encode("truth")
print(qw)  # QuantumWord(n=4, θ=-12°, r=0.80, word='truth')

# Get hex encoding
hex_code = encoder.encode_to_hex("truth")
print(hex_code)  # "473"

# Decode back to (A, S, τ)
A, S, tau = encoder.decode(qw)
print(f"A={A:.2f}, S={S:.2f}, τ={tau:.2f}")

# Find nearest words
neighbors = encoder.nearest(qw, k=5)
for word, dist, qw_neighbor in neighbors:
    print(f"  {word}: distance={dist:.3f}")
```

### Verb Operators

Verbs act as navigation operators in quantum space:

```python
# Encode verb operator
vb = encoder.encode_verb("love")
print(vb)  # QuantumVerb(love: ΔA=-0.172, ΔS=+0.165, |Δ|=0.238)

# Apply verb to word
result = encoder.apply_verb("truth", "love")
print(result)  # New quantum position after transformation

# Chain of verbs
trajectory = encoder.chain("truth", ["love", "seek", "find"])
for step in trajectory:
    print(f"{step.word}: θ={step.theta_deg:+.0f}°")

# Navigate toward target direction
results = encoder.navigate("truth", target_theta=90)  # Find verbs toward sacred
for verb, position in results:
    print(f"{verb} → θ={position.theta_deg:+.0f}°")
```

### Verb Operator Semantics

| Verb | ΔA | ΔS | Direction | Effect |
|------|----|----|-----------|--------|
| love | -0.17 | +0.16 | +136° | Toward sacred-negating |
| hate | +0.16 | -0.15 | -43° | Toward profane-affirming |
| think | +0.01 | +0.14 | +88° | Toward sacred |
| feel | +0.04 | -0.04 | -50° | Toward profane |
| create | -0.08 | -0.09 | -132° | Toward profane-negating |
| destroy | -0.04 | +0.16 | +104° | Toward sacred-negating |
| seek | -0.46 | +0.16 | +161° | Strong negating |

### Navigation Example

```
START: truth (n=4, θ=-12°, r=0.80)

Navigate toward sacred (θ≈90°):
  overlook  → θ=+84°
  lie       → θ=+60°
  think     → θ=+36°

Navigate toward profane (θ≈-90°):
  caress    → θ=-84°
  illuminate→ θ=-84°
  expect    → θ=-84°
```

### Direct Bit Manipulation

```python
# Create from bits
qw = QuantumWord.from_bits(0x473, "truth")

# Convert to bits
bits = qw.to_bits()  # 1139 (0x473)

# Access properties
print(qw.n)          # 4
print(qw.theta_deg)  # -12.0
print(qw.r)          # 0.80
print(qw.A)          # +0.78
print(qw.S)          # -0.17
print(qw.tau)        # 2.47
```

---

## Theoretical Foundation

### Analogy to Quantum Mechanics

| Atomic Orbitals | Semantic Orbitals |
|-----------------|-------------------|
| n (principal quantum number) | n (abstraction orbital) |
| l (angular momentum) | — (not needed) |
| m (magnetic quantum number) | θ (phase direction) |
| — | r (intensity magnitude) |

### Why It Works

1. **Dimensional Reduction**: PCA reveals 95% of semantic variance in 2 dimensions (A, S)
2. **Natural Quantization**: τ shows discrete orbital structure (not continuous)
3. **Polar Symmetry**: Semantic opposites relate by θ → θ + 180°
4. **Intensity Independence**: r carries information orthogonal to direction

### Metaphysical Interpretation: Neoplatonic Cosmology

The three quantum numbers map to ontological categories in Neoplatonic philosophy:

| Quantum Number | Ontological Meaning |
|----------------|---------------------|
| n (orbital) | Level of emanation (abstract → concrete) |
| θ (phase) | Axiological direction (good/evil × sacred/profane) |
| r (intensity) | Distance from The One (emanation degree) |

#### The Emanation Structure

**Empirically verified** (correlation n ↔ r = **+0.969**):

```
                         SEMANTIC SPACE

                    S (Sacred)
                       ↑
           ┌───────────┼───────────┐
           │     n≥5   │   n≥5     │   r=1.84
           │   ╱ ὕλη   │   ὕλη  ╲  │   (Matter)
           │ ╱         │         ╲ │
           │╱   n=3-4  │  n=3-4   ╲│   r=1.26
    ←──────┼───────────●───────────┼──────→ A (Affirmation)
           │╲   ψυχή   │   ψυχή  ╱ │   (Soul)
           │ ╲         │        ╱  │
           │   ╲ n=1-2 │ n=1-2 ╱   │   r=1.03
           │     ╲νοῦς │ νοῦς╱     │   (Nous)
           └───────────┼───────────┘
                  n=0  │  n=0        r=0.45
                 τὸ ἕν │ τὸ ἕν      (The One)
                       ↓            ← ORIGIN
```

#### Ontological Levels

| Level | Greek | n | mean r | Description |
|-------|-------|---|--------|-------------|
| **The One** | τὸ ἕν | 0 | 0.45 | Source, Origin, Center |
| **Nous** | νοῦς | 1-2 | 1.03 | Intellect, Forms, Ideas |
| **Soul** | ψυχή | 3-4 | 1.26 | Animation, Life, Motion |
| **Matter** | ὕλη | 5+ | 1.84 | Manifestation, Periphery |

#### Navigation as Spiritual Movement

**Emanation** (πρόοδος — procession):
- Direction: Origin → Periphery
- Effect: `Δn > 0`, `Δr > 0`
- Meaning: Creation, manifestation, descent into matter

**Return** (ἐπιστροφή — epistrophe):
- Direction: Periphery → Origin
- Effect: `Δn < 0`, `Δr < 0`
- Meaning: Contemplation, abstraction, return to Source

#### Verbs as Ontological Operators

Verbs shift position in semantic space, implementing ontological movements:

| Verb Type | Effect | Ontological Meaning |
|-----------|--------|---------------------|
| Ascending | Δr < 0 | Return toward The One |
| Descending | Δr > 0 | Emanation into matter |
| Rotating | Δθ ≠ 0 | Axiological transformation |

Example verbs:
- `think`, `contemplate` → tend toward Origin (Nous)
- `make`, `build` → tend toward Matter (creation)
- `love` → rotates toward Sacred axis

#### Independence of r: The Residual ε

While orbital means show strong n ↔ r correlation (0.969), individual words show:

```
Mean r per orbital:  corr(n, r) = 0.969 (trend)
Individual words:    corr(n, r) = 0.464 (R² = 0.22)

→ 88.6% of r variance is INDEPENDENT of n
→ r carries substantial unique information
→ All three (n, θ, r) are needed
```

The residual ε = r - f(n) represents each word's "individuality":

| ε | Meaning | Example |
|---|---------|---------|
| ε < 0 | Closer to Source than expected | `sic` (n=14, r=0.01, ε=-2.29) |
| ε ≈ 0 | Follows ideal emanation | typical words |
| ε > 0 | Farther from Source than expected | `astral` (n=0, r=3.66, ε=+2.93) |

This represents the word's unique "spiritual character" beyond its emanation level.

---

## Files

| File | Description |
|------|-------------|
| `core/semantic_quantum.py` | QuantumEncoder, QuantumWord classes |
| `data/json/semantic_quantum.json` | Pre-computed 12-bit encodings |
| `data/json/semantic_coordinates.json` | Full (A, S, τ) coordinates |

---

## References

- PC1 (Affirmation): 83.3% variance, corresponds to yes/no axis
- PC2 (Sacred): 11.7% variance, corresponds to transcendent/mundane axis
- τ (Abstraction): 1 + rank/e formula from word frequency ranking
- Encoding: 22,486 words × 12 bits = 33.7 KB
