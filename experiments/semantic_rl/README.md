# Semantic RL: Embodied Knowledge Through Lived Experience

**Core Principle:** "Only believe what was lived is knowledge"

## Overview

A reinforcement learning environment where an agent navigates through semantic space,
learning through embodied experience. Concepts become physical objects with properties
derived from their semantic meaning.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC RL WORLD                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SEMANTIC SPACE              PHYSICAL SIMULATION                │
│  ─────────────────────────────────────────────────              │
│  τ (abstraction)      →      altitude / height                  │
│  g (goodness)         →      reward / light                     │
│  j (direction)        →      gravity direction                  │
│  believe              →      jump/tunnel power                  │
│  temperature          →      energy / speed                     │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   AGENT      │────▶│  ENVIRONMENT │────▶│  KNOWLEDGE   │    │
│  │              │     │              │     │              │    │
│  │ - believe    │     │ - concepts   │     │ - lived set  │    │
│  │ - temperature│     │ - physics    │     │ - tunnels    │    │
│  │ - actions    │     │ - rewards    │     │ - barriers   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Lived Experience
- Agent can only truly "know" states it has visited
- Tunneling (insight) requires connection to lived experience
- Knowledge = embodied, not abstract

### 2. Semantic Physics
```python
concept_properties = {
    "love":    { "mass": 2.0, "believe_bonus": +0.5, "friction": 0.8 },
    "fear":    { "mass": 0.5, "believe_penalty": -0.3, "speed": 1.5 },
    "wisdom":  { "altitude": 3.0, "requires": ["struggle", "time"] },
    "courage": { "requires": ["fear"], "unlocks": ["hope"] },
}
```

### 3. Two Movement Types
1. **Thermal (verbs):** Gradual movement, always possible
2. **Tunneling (insight):** Instant jump, requires lived connection

### 4. Believe Parameter
```
P(tunnel_success) = believe × e^(-2κd) × knowledge(target)

where:
  believe = agent's belief in change (0-1)
  κd = semantic barrier thickness
  knowledge(target) = connection to lived experience
```

## Project Structure

```
semantic_rl/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml              # Configuration
├── src/
│   ├── core/
│   │   ├── semantic_state.py     # State representation
│   │   ├── semantic_loader.py    # Load from QuantumCore (19K words)
│   │   └── knowledge.py          # Lived experience tracking
│   ├── environment/
│   │   ├── semantic_world.py     # Main environment
│   │   ├── book_world.py         # Navigate through literature
│   │   └── physics.py            # Semantic → Physical
│   ├── agents/
│   │   ├── base_agent.py         # Agent interface
│   │   └── quantum_agent.py      # Believe/tunnel agent
│   └── visualization/
│       └── journey_viz.py        # Graph + narrative arc visualization
├── experiments/
│   ├── real_semantic_journey.py  # Journey through 19K semantic space
│   └── visualize_book_journey.py # Visualize book journeys
└── tests/
    └── test_environment.py       # Unit tests
```

## Book World: Navigate Literature

Journey through the semantic landscape of classic books:

```python
from environment.book_world import BookWorld, CLASSIC_BOOKS

# Available books
print(CLASSIC_BOOKS.keys())
# heart_of_darkness, crime_and_punishment, divine_comedy,
# moby_dick, frankenstein, odyssey, metamorphosis, ...

# Create world from book
world = BookWorld(book_file="path/to/book.txt")
# Or use built-in:
world = BookWorld(book_key="heart_of_darkness")

# Run journey
obs, info = world.reset()
# ... agent navigation ...
```

## Visualization

Visualize semantic journeys with:
- **Graph view**: Semantic landscape with path highlighted
- **Narrative arc**: Goodness (g) trajectory through journey
- **Journey summary**: Combined statistics and visualization

```bash
# Single book journey
python experiments/visualize_book_journey.py --book heart_of_darkness

# Compare multiple books
python experiments/visualize_book_journey.py \
    --compare heart_of_darkness divine_comedy metamorphosis
```

![Semantic Journey](experiments/visualizations/heart_of_darkness_graph.png)
![Narrative Arc](experiments/visualizations/heart_of_darkness_arc.png)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.environment import SemanticWorld
from src.agents import QuantumAgent

# Create world
world = SemanticWorld(start="darkness", goal="wisdom")

# Create agent
agent = QuantumAgent(believe=0.5, temperature=1.0)

# Run episode
state = world.reset()
done = False

while not done:
    action = agent.choose_action(state, world.get_valid_actions())
    state, reward, done, info = world.step(action)

    # Agent learns from lived experience
    agent.update_knowledge(state, reward)

print(f"Knowledge gained: {agent.knowledge}")
print(f"Tunnels discovered: {agent.tunnel_history}")
```

## Theoretical Foundation

Based on the quantum-semantic framework:

| Component | Meaning |
|-----------|---------|
| State | Concept/word in semantic space |
| Action | Verb (thermal) or tunnel (quantum) |
| Reward | Change in goodness (Δg) |
| Episode | Journey from start to goal |
| Knowledge | Set of lived states + tunnel connections |
| Believe | Capacity for breakthrough |
| Temperature | Exploration vs exploitation |

## Future Extensions

- [ ] Multi-agent environments (dialogue simulation)
- [ ] Continuous semantic space (not discrete)
- [x] Visual rendering of semantic landscape ✓
- [x] Book-based semantic navigation ✓
- [x] Narrative arc visualization ✓
- [ ] Integration with LLM for action selection
- [ ] Transfer learning between semantic domains
- [ ] Animation of semantic journeys
- [ ] Real-time dashboard for journey monitoring
