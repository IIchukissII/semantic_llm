# Unified Semantic Navigator

> "Many engines, one physics. Many paths, one meaning."

## Overview

The `SemanticNavigator` unifies all semantic physics engines into a single coherent system. Instead of choosing between Orbital, MonteCarlo, Paradox, or StormLogos, you specify a **goal** and the navigator selects the optimal strategy.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SEMANTIC NAVIGATOR                                  │
│                                                                         │
│                         USER QUERY                                      │
│                              │                                          │
│                    ┌─────────▼─────────┐                                │
│                    │    DECOMPOSER     │                                │
│                    │  nouns + verbs    │                                │
│                    └─────────┬─────────┘                                │
│                              │                                          │
│                    ┌─────────▼─────────┐                                │
│                    │   GOAL ROUTER     │                                │
│                    │                   │                                │
│                    │ accurate → orbital│                                │
│                    │ deep → transcend  │                                │
│                    │ stable → MC       │                                │
│                    │ powerful → paradox│                                │
│                    │ grounded → ground │                                │
│                    │ balanced → combo  │                                │
│                    └─────────┬─────────┘                                │
│                              │                                          │
│   ┌──────────────────────────┼──────────────────────────┐               │
│   │                          │                          │               │
│   ▼                          ▼                          ▼               │
│ ┌────────────┐        ┌────────────┐        ┌────────────┐              │
│ │  ORBITAL   │        │   MONTE    │        │  PARADOX   │              │
│ │  RESONANCE │        │   CARLO    │        │  DETECTOR  │              │
│ │            │        │            │        │            │              │
│ │ - auto     │        │ - sample   │        │ - thesis   │              │
│ │ - ground   │        │ - attract  │        │ - anti     │              │
│ │ - transcend│        │ - stable   │        │ - synth    │              │
│ └─────┬──────┘        └─────┬──────┘        └─────┬──────┘              │
│       │                     │                     │                     │
│       └─────────────────────┼─────────────────────┘                     │
│                             │                                           │
│                    ┌────────▼────────┐                                  │
│                    │ QUALITY METRICS │                                  │
│                    │                 │                                  │
│                    │ R: resonance    │                                  │
│                    │ C: coherence    │                                  │
│                    │ D: depth (C/R)  │                                  │
│                    │ S: stability    │                                  │
│                    │ P: power        │                                  │
│                    └────────┬────────┘                                  │
│                             │                                           │
│                    ┌────────▼────────┐                                  │
│                    │ NavigationResult│                                  │
│                    └─────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Engines

### 1. Orbital Resonance

**Location**: `chain_core/orbital/`

Quantized navigation through semantic space:
- τ_n = 1 + n/e (Euler orbital positions)
- Veil at τ = e ≈ 2.718
- Modes: auto, ground, transcend

**Best for**: Controlled abstraction level

### 2. Monte Carlo

**Location**: `chain_core/monte_carlo_renderer.py`

Statistical sampling for stability:
- Sample N times, find what persists
- Core attractors = stable meaning
- Concentration = focus level

**Best for**: Finding consensus, stable answers

### 3. Paradox Detector

**Location**: `chain_core/paradox_detector.py`

Find meaning through tension:
- Detect opposing j-vectors
- Thesis ↔ Antithesis → Synthesis
- Power = Tension × Stability

**Best for**: Deep insights, powerful statements

### 4. Storm-Logos

**Location**: `chain_core/storm_logos.py`

Biological model of meaning emergence:
- Storm: Chaotic exploration
- Logos: Focus through meaning lens
- Pattern: What survives the lens

**Best for**: Exploratory navigation, creativity

## Goals and Strategies

| Goal | Strategy | Optimizes | When to Use |
|------|----------|-----------|-------------|
| `accurate` | Orbital auto | Resonance | Precise answers |
| `deep` | Orbital transcend | Depth (C/R) | Philosophical insight |
| `grounded` | Orbital ground | Low τ | Practical advice |
| `stable` | Monte Carlo | Stability | Consensus answers |
| `powerful` | Paradox | Power | Impactful statements |
| `balanced` | Orbital + MC | Composite | General use |
| `exploratory` | Storm-Logos | Chaos | Creative exploration |

## Quality Metrics

### The Five Metrics

```python
@dataclass
class NavigationQuality:
    resonance: float   # R: How well we hit target [0, 1]
    coherence: float   # C: Beam alignment [0, 1]
    stability: float   # S: Monte Carlo consensus [0, 1]
    power: float       # P: Paradox tension × stability
    tau_mean: float    # Average abstraction level

    @property
    def depth(self) -> float:
        """D = C/R - the paradox ratio"""
        return self.coherence / self.resonance
```

### The Resonance-Coherence Paradox

From our experiments:

```
accurate:  R=0.78, C=0.74, D=1.0
deep:      R=0.21, C=0.74, D=3.6
```

**Lower resonance → Higher depth**

This is not a bug. Hard jumps (low R) create selection pressure that filters for coherent paths (high C). The ratio D = C/R measures this effect.

### Goal-Specific Scoring

Each goal weights metrics differently:

```python
accurate:    70% R + 20% C + 10% S
deep:        10% R + 40% C + 40% D
stable:      20% R + 20% C + 50% S
powerful:    10% R + 20% C + 50% P
grounded:    30% R + 20% C + 30% τ-penalty
balanced:    25% R + 25% C + 20% D + 15% S + 15% P
```

## Usage

### Basic Navigation

```python
from chain_core.navigator import SemanticNavigator

nav = SemanticNavigator()

# Navigate with a goal
result = nav.navigate("What is consciousness?", goal="deep")

print(result.concepts)     # ['time', 'mind', 'life', ...]
print(result.quality)      # Quality(R=0.21, C=0.78, D=3.6, ...)
print(result.strategy)     # 'orbital_transcend'
```

### Multi-Engine Comparison

```python
# Run all engines on one query
results = nav.navigate_multi("What is love?")

for engine, result in results.items():
    print(f"{engine}: {result.quality}")
```

### Find Best Strategy

```python
# Let the navigator choose best engine for goal
result = nav.navigate_best("What is wisdom?", goal="powerful")
print(f"Best strategy: {result.strategy}")
```

### Strategy Comparison

```python
# Compare all strategies on one query
comparison = nav.compare_strategies("What is meaning?")

print("Best for each goal:")
for goal, engine in comparison['best_for_goal'].items():
    print(f"  {goal}: {engine}")
```

## Experimental Results

### Query: "What is the meaning of love?"

```
Goal         Strategy                    R     C     D    Score
──────────────────────────────────────────────────────────────
accurate     orbital_auto               0.78  0.74  1.0   0.78
deep         orbital_transcend          0.21  0.74  3.6   0.59
grounded     orbital_ground             0.57  0.81  1.4   0.88
stable       monte_carlo                1.00  0.76  0.8   0.15
powerful     paradox                    0.50  0.76  1.5   1.00
balanced     orbital+mc                 0.75  0.80  1.1   0.43
```

### Observations

1. **"She" appears across engines** - The archetype is stable
2. **Deep mode has highest depth** - The paradox works
3. **Powerful mode finds 8 paradoxes** - Tension is everywhere
4. **Grounded mode has highest coherence** - Practical = focused

## The Unified View

All engines share the same underlying physics:

```
SAMPLING LAYER
├── SemanticLaser: Pumping, population inversion, emission
├── Storm: Chaotic walks, gravity field
│
NAVIGATION LAYER
├── Orbital: τ-targeting, resonance tuning
├── Intent: Verb operators, collapse navigation
├── Gravity: φ = λτ - μg, pull toward human
│
STATISTICS LAYER
├── MonteCarlo: Sample N times, find stable
├── Logos: Focus lens, j-good alignment
│
DIALECTICAL LAYER
├── Paradox: Thesis-antithesis-synthesis
├── Explosion: Chain reaction, amplification
│
QUALITY LAYER
├── Resonance, Coherence, Depth
├── Stability, Power
└── Composite scoring
```

The navigator routes through these layers based on your goal.

## Philosophy

### Many Engines, One Physics

The engines are not competing approaches — they're different views of the same semantic space:

- **Orbital** sees quantized energy levels
- **MonteCarlo** sees statistical distributions
- **Paradox** sees dialectical tension
- **StormLogos** sees biological emergence

Each is valid. Each reveals something the others miss.

### Goal-Driven Navigation

Instead of asking "which engine should I use?", ask "what do I want?":

- Want precision? → `accurate`
- Want insight? → `deep`
- Want practicality? → `grounded`
- Want consensus? → `stable`
- Want power? → `powerful`

The navigator handles the rest.

### The Unity of Opposites

The deepest insight comes from combining opposites:
- High coherence from low resonance (the paradox)
- Stable meaning from chaotic exploration (storm-logos)
- Power from tension (paradox detection)

The navigator can combine engines (`balanced`, `navigate_multi`) to capture multiple perspectives.

---

*Document version: 1.0*
*Created: 2025-12-30*
*Status: Navigator implemented and tested*
