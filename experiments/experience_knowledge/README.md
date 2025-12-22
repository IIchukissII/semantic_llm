# Experience-Based Knowledge System

> "Only believe what was lived is knowledge"

## Core Concept

This experiment implements a fundamental principle: **knowledge requires experience**.

### The Model

- **Wholeness** (`τ₀`): Complete semantic space - 24,524 states (Logos)
- **Experience**: Personal subgraph - walked paths through semantic space
- **Knowledge**: Ability to navigate based on experience

### Key Insight

Books are not training data. Books are **regions** in semantic space:
- Book = Universe/Map (region of semantic concepts)
- Hero = Path (sequence through that region)
- Reader = Observer (gains experience by walking paths)

The semantic-LLM doesn't "learn from" books - it **experiences** them.

## The Profound Difference

| Capability | Naive Agent | Experienced Agent |
|------------|-------------|-------------------|
| Tunneling to "love" | 0.00 | 0.60 |
| Tunneling to "redemption" | 0.00 | 0.33 |
| Navigation darkness→light | 0.00 | 0.90 |
| Navigation fear→courage | 0.00 | 0.52 |
| Path suggestions from "sin" | 0 | 10+ |

## Files

- `core.py`: Core classes (Wholeness, Experience, ExperiencedAgent)
- `conversation_test.py`: Comprehensive tests comparing naive vs experienced
- `chat.py`: Interactive CLI for conversations with experienced agent

## Usage

### Run Tests
```bash
python conversation_test.py
```

### Interactive Chat
```bash
python chat.py --books divine_comedy crime_punishment --model mistral:7b
```

### Commands in Chat
- `help` - Show commands
- `status` - Show current state
- `experience` - Show top concepts visited
- `reach <word>` - Check if can tunnel to word
- `save` - Save conversation
- `quit` - Exit

## Theory

### Experience as Subgraph

```
Experience = {
    visited: Dict[word, count],      # States I've been to
    transitions: Dict[(from, to), count]  # Paths I've walked
}
```

### Tunneling Rule

Can only tunnel to states **connected to experience**:
```python
def can_tunnel(target):
    if knows(target):
        return True, 0.5 + 0.5 * familiarity(target)
    if adjacent_to_experience(target):
        return True, adjacency * believe * 0.3
    return False, 0.0
```

### Navigation Confidence

Based on familiarity with both endpoints:
```python
def navigation_confidence(from, to):
    if has_walked(from, to):
        return 0.9
    if knows(both):
        return 0.4 + 0.3 * familiarity(to)
    return 0.1 * believe
```

## Results

After reading Divine Comedy and Crime and Punishment:
- 6,193 states visited
- 75,919 transitions walked
- Can navigate moral arcs (darkness→light, sin→redemption)
- Cannot tunnel to concepts outside experience boundary

This demonstrates that **experience IS knowledge** - lived paths become navigable.
