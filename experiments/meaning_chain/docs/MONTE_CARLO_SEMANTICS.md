# Monte Carlo Semantic Sampling

## The Insight

Instead of following one path through semantic space, **throw the question many times** and see where it lands. Like a physicist throwing particles to discover stable orbits.

```
100 throws → distribution of landing points → shape of semantic space
```

## The Discovery

Running experiments with different intent modes revealed:

| Mode | Unique Words | Concentration | Lasing | Character |
|------|--------------|---------------|--------|-----------|
| Hard Collapse (α→∞) | 3 | 100% | 0% | Point |
| Soft Wind (α=0.3) | 1000+ | 17% | 99% | Cloud with direction |
| Pure Boltzmann (α=0) | 246 | 33% | 94% | Diffuse cloud |

**Hard collapse was too strong** — it collapsed everything to a single deterministic path.

**Soft wind is optimal** — it provides direction without blocking exploration.

## The Formula

```
P(next) ∝ exp(-|Δτ|/kT) × (1 + α × intent_alignment)
```

Where:
- `exp(-|Δτ|/kT)` = Boltzmann weight (orbital proximity)
- `α` = intent strength parameter
- `intent_alignment` = 1 if target is in intent space, 0 otherwise

### The α Dial

```
α = 0.0  →  Pure Boltzmann exploration (cloud)
α = 0.3  →  Soft guidance ("wind not wall") ← RECOMMENDED
α = 1.0  →  Moderate influence
α ≥ 2.0  →  Strong collapse (approaching determinism)
```

## Semantic Landscape

Monte Carlo sampling produces a **SemanticLandscape** — the statistical shape of a question in semantic space:

```python
@dataclass
class SemanticLandscape:
    question: str
    n_samples: int

    # Shape metrics
    unique_words: int      # How many concepts were visited
    concentration: float   # Top-10 / total (focus level)
    coherence: float       # Average beam coherence
    lasing_rate: float     # Fraction achieving coherent output

    # Attractors (convergence points)
    core_attractors: List[tuple]       # Stable (appear often)
    peripheral_attractors: List[tuple] # Variable (appear sometimes)

    # Orbital structure
    orbital_map: Dict[int, List[tuple]]  # n -> concepts at that level
```

## Orbital Hierarchy

The orbital map reveals abstraction levels:

```
n=0: Ground state (most concrete)
     e.g., lightning, resolution, formation

n=1: Human realm (everyday)
     e.g., dream, meaning, memory, gate

n=2: Mid-level (connecting)
     e.g., she, sense, way, image

n=3: Abstract
     e.g., thing, place, body, moment

n=4+: Transcendental
      e.g., time, hand, other
```

## Key Findings

### 1. Questions Have Different "Shapes"

- **"What is meaning?"** → Deterministic path (few alternatives in graph)
- **"What is a tree?"** → tree → death (Biblical: tree of knowledge → mortality)
- **"What do dreams mean?"** → Rich cloud (many paths, ~300 unique words)

### 2. Emergent Discoveries

Running "What do my dreams mean?" found **justice** as a core attractor.

This is Jung's "compensatory function" — dreams restore psychic balance.
The space discovered Jung **without loading Jung's texts**.

### 3. Stability vs Variability

- **Core attractors** (top 10): Appear in most samples → stable meaning
- **Peripheral attractors** (11-30): Appear sometimes → contextual meaning

This maps to response structure:
- Use core concepts **with confidence**
- Use peripheral concepts **with nuance**

## Monte Carlo Renderer

The renderer uses landscape statistics to structure LLM responses:

```python
from chain_core.monte_carlo_renderer import MonteCarloRenderer

renderer = MonteCarloRenderer(intent_strength=0.3, n_samples=30)
result = renderer.render("What do my dreams mean?")

# result contains:
# - response: LLM-generated text
# - landscape: SemanticLandscape statistics
# - core_concepts: Most stable attractors
# - prompt: The structured prompt sent to LLM
```

### Prompt Structure

The landscape informs the prompt:

```
## Semantic Landscape (Monte Carlo)
Samples: 30 | Lasing: 100%
Coherence: 0.78 | Focus: 17%

## Core Concepts (stable across samples)
These concepts consistently appeared: she, dream, image, sense, way

## Concept Hierarchy (by abstraction)
  concrete (n=0): resolution, formation, godhead
  concrete (n=1): dream, meaning, memory
  mid-level (n=2): she, image, sense, way
  abstract (n=3): place, body, thing

## Response Guidance
- High coherence: Give a focused, confident answer
- Low concentration: Feel free to explore connections
- Start with concrete/grounded concepts, then expand
```

## Usage

### Running Experiments

```bash
# Pure Boltzmann (no intent)
python experiments/monte_carlo_sampling.py --no-intent

# Soft wind (recommended)
python experiments/monte_carlo_sampling.py --alpha 0.3

# Hard collapse
python experiments/monte_carlo_sampling.py --alpha 2.0
```

### Using the Renderer

```python
from chain_core.monte_carlo_renderer import MonteCarloRenderer

# Create renderer with soft wind
renderer = MonteCarloRenderer(
    intent_strength=0.3,  # α parameter
    n_samples=30          # samples per question
)

# Render a response
result = renderer.render(
    question="What is the meaning of life?",
    model="mistral:7b",
    temperature=0.7
)

print(result['response'])
print(f"Core concepts: {result['core_concepts']}")

# Access full landscape
ls = result['landscape']
print(f"Coherence: {ls.coherence:.2f}")
print(f"Unique words: {ls.unique_words}")

renderer.close()
```

## Implications for Rendering

| Landscape Metric | Rendering Implication |
|------------------|----------------------|
| High coherence (>0.75) | Confident, focused response |
| Low coherence (<0.5) | Acknowledge multiple perspectives |
| High concentration (>30%) | Stay close to core concepts |
| Low concentration (<20%) | Explore connections freely |
| Rich orbital distribution | Structure: concrete → abstract |
| Single orbital dominance | Stay at one level of abstraction |

## Files

```
chain_core/
├── semantic_laser.py        # Intent-weighted Boltzmann (α dial)
├── monte_carlo_renderer.py  # MC-based response generation

experiments/
├── monte_carlo_sampling.py  # Sampling experiments

results/monte_carlo/
├── comparison_*.json        # Comparative results
├── mc_*.json               # Individual experiment data
```

## The Metaphor

**Intent is wind, not wall.**

A wall blocks all but one path. Wind pushes gently in a direction while allowing exploration.

```
Hard collapse:  meaning → difference → thing (always)
Soft wind:      meaning → [concept, dream, order, philosophy, mind, ...] (varied)
```

The soft wind preserves the **shape** of semantic space while providing **direction**.
