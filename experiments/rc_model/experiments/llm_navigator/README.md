# LLM Navigator: Semantic Space Navigation

## Concept

LLM as agent navigating through (A, S, τ) semantic space.

Instead of:
```
Skeleton bonds → LLM → Text
```

We do:
```
Position + Gradient + Target → LLM → Sentence → Measure → Repeat
```

## Key Idea

LLM understands its position qualitatively:
- NOT: "A=0.3, τ=2.5"
- BUT: "You are in dark, earthly, concrete territory. Move toward light."

## Prompt Structure

```
CURRENT POSITION:
  Emotional tone: [qualitative A description]
  Conceptual depth: [qualitative S description]
  Abstraction level: [qualitative τ description]

GRAVITY pulls toward: concrete (+τ) and good (+A)

TARGET DIRECTION (GENRE):
  • Move A: [instruction]
  • Move S: [instruction]
  • Move τ: [instruction]

Write ONE sentence that moves in the target direction.
```

## Gravity Model

Natural semantic flow:
- φ = λτ - μA
- Pulls toward concrete (lower τ)
- Pulls toward affirmation (higher A)

Genres resist/embrace gravity differently:
- **DRAMATIC**: Let gravity pull, but elevate S
- **IRONIC**: Hold S flat (mundane), let A rise slowly
- **BALANCED**: Resist gravity, maintain abstract τ

## Usage

```bash
cd experiments/llm_navigator

# Run all genres
python run.py

# Run specific genre
python run.py --genre dramatic

# Save results
python run.py --save --sentences 6
```

## Files

```
llm_navigator/
├── __init__.py     # Exports
├── core.py         # Main implementation
├── run.py          # CLI runner
├── README.md       # This file
└── results/        # Saved experiment results
```

## Results

Trajectory shows position evolution:
```
Step 0: A=0.47, S=0.53, τ=1.79  (start)
Step 1: A=0.39, S=0.33, τ=1.23  (after sentence 1)
Step 2: A=0.38, S=0.19, τ=0.92  (after sentence 2)
...
```

## Observations

- A axis responds well to instructions
- S axis moves toward targets
- τ keeps falling (gravity wins) — needs stronger resistance
