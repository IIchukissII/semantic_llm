# Experiment Summary: Conversation Optimization

**Date:** 2025-12-21
**Experiment ID:** conv-opt-001

## Objective

Test and compare optimization algorithms for semantic space navigation in a hybrid Quantum-LLM conversation system.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Semantic Space | 19,055 word states |
| Verbs Available | 2,444 unique verbs |
| Graph Type | Dynamic (on-demand) |
| LLM Renderer | Ollama qwen2.5:1.5b |
| Fidelity Threshold | 0.4 |
| Max Retries | 3 |

## Algorithm Comparison

### Results Summary

| Algorithm | Net Δg | Steps | Efficiency | Fidelity | Retries |
|-----------|--------|-------|------------|----------|---------|
| **Hill Climbing** | +1.161 | 3 | 0.629 | 0.80 | 5 |
| **Simulated Annealing** | +0.924 | 27 | 0.029 | 0.51 | 4 |
| **Random Local Search** | +0.692 | 4 | 0.105 | 0.67 | 5 |

### Detailed Analysis

#### Hill Climbing
- **Behavior:** Greedy, always takes best neighbor
- **Strengths:** Fast convergence, high efficiency, reliable fidelity
- **Weaknesses:** Gets stuck at local maxima quickly
- **Best path:** `darkness --wish--> luck` (Δg=+1.21 in 1 step)

#### Simulated Annealing
- **Behavior:** Probabilistic, accepts worse moves with decreasing probability
- **Strengths:** Rich exploration, diverse verb usage, escapes local maxima
- **Weaknesses:** Lower efficiency, longer paths
- **Best path:** `darkness -> package -> prayer -> kingdom -> habit -> bean -> change -> hip -> pyramid -> price -> grin -> shoulder -> figure` (12 steps, 12 unique verbs)

#### Random Local Search
- **Behavior:** Random neighbor selection with restarts
- **Strengths:** Simple, moderate exploration
- **Weaknesses:** Unpredictable paths, may restart from unrelated words
- **Note:** Restarts from random positions can lead to disconnected narratives

## Verb Diversity Analysis

| Algorithm | Unique Verbs Used | Examples |
|-----------|-------------------|----------|
| Hill Climbing | 3 | wish, refuse, suggest |
| Simulated Annealing | 26+ | hand, mutter, destroy, instill, spill, hate, span, resemble, name, sport, numb, spot, answer, reflect, acquire, void, shine, investigate |
| Random Local | 4 | (restarts break continuity) |

## Key Findings

### 1. Dynamic Graph Solves Verb Bias
Before: Only "abandon" verb used (alphabetical bias)
After: 26+ diverse verbs per conversation

### 2. Trade-off: Efficiency vs Exploration
- Hill Climbing: High efficiency (0.63), low exploration
- Simulated Annealing: Low efficiency (0.03), high exploration

### 3. Feedback Loop Essential
- Average 4-5 LLM retries per conversation
- Fidelity verification prevents incoherent outputs
- Best results when path words are incorporated naturally

### 4. Temperature Dynamics (Simulated Annealing)
```
T=1.000: Accepts large negative moves (Δ=-0.80)
T=0.500: Accepts moderate negative moves (Δ=-0.40)
T=0.250: Mostly accepts only improvements
```

## Sample Paths

### From Darkness to Light (Simulated Annealing)
```
darkness --hand--> package --mutter--> prayer --destroy--> kingdom
         --instill--> habit --spill--> bean --hate--> change
         --span--> hip --resemble--> pyramid --name--> price
         --sport--> grin --numb--> shoulder --spot--> figure
```

### Philosophical Journey (Hill Climbing)
```
hope --refuse--> food --suggest--> age
```

## Conclusions

1. **Simulated Annealing** is best for rich, diverse semantic exploration
2. **Hill Climbing** is best for efficient, direct navigation
3. **Dynamic Graph** is essential for utilizing full verb vocabulary
4. **Feedback Loop** ensures semantic coherence in LLM output

## Recommendations

- Use **Simulated Annealing** for creative/exploratory conversations
- Use **Hill Climbing** for goal-directed navigation
- Consider **hybrid approach**: SA for exploration, then HC for refinement

## Files Generated

```
experiments/conversation_optimization/
├── README.md                              # Full documentation
├── optimization_algorithms.py             # Core algorithms
├── conversation_test.py                   # Test framework
└── results/
    ├── EXPERIMENT_SUMMARY.md              # This file
    ├── hill_climbing_output.txt           # HC results
    ├── simulated_annealing_output.txt     # SA results
    └── random_local_output.txt            # RLS results
```
