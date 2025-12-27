# Meaning Chain - Neo4j Schema Design

## Philosophy

> "Intent collapses meaning like observation collapses wavefunction"

The user's verbs are not decorative - they are **operators** that collapse
the superposition of possible meanings to those aligned with intent.

## Separation from Main System

```
experience_knowledge/          meaning_chain/
├── Neo4j DB: "neo4j"          ├── Neo4j DB: "meaning" (NEW)
├── Password: experience123    ├── Password: experience123
├── Sequential transitions     ├── Verb-mediated transitions
└── Quantum navigation         └── Intent-driven collapse
```

**Key principle**: Completely separate database. No changes to existing system.

## Schema

### Nodes

```cypher
// Concept - semantic state (subset of main system)
(:Concept {
    word: STRING,           // The word
    g: FLOAT,               // Goodness [-2, +2]
    tau: FLOAT,             // Abstraction level [1, 7]
    j: LIST<FLOAT>,         // J-vector (5D transcendentals)
    pos: STRING             // Part of speech: "noun", "verb", "adj"
})

// VerbOperator - verbs as navigation operators (DUAL STRUCTURE)
(:VerbOperator {
    verb: STRING,           // The verb

    // Original properties
    j: LIST<FLOAT>,         // Raw j-vector (5D semantic direction)
    magnitude: FLOAT,       // Operator strength
    objects: LIST<STRING>,  // Typical objects this verb operates on

    // Phase-shifted properties (Pirate Insight)
    i: LIST<FLOAT>,         // Centered j-vector = j - global_mean
                            // Represents INTRINSIC action type
    delta_tau: FLOAT,       // Δτ: abstraction effect
                            //   > 0: ascends (abstract)
                            //   < 0: descends (concrete)
    delta_g: FLOAT,         // Δg: moral push direction
                            //   > 0: pushes toward good
                            //   < 0: pushes toward evil
    transition_count: INT,  // Number of VIA edges for this verb
    phase_shifted: BOOL     // Flag: new properties computed
})

// GlobalMean - stores the bias for phase shifting
(:GlobalMean {
    name: STRING,           // 'verb_j_mean'
    j_mean: LIST<FLOAT>,    // Global mean j-vector
    updated_at: DATETIME
})
```

### Relationships

```cypher
// VIA - verb-mediated transition (the core innovation)
(:Concept)-[:VIA {
    verb: STRING,           // The connecting verb
    weight: FLOAT,          // Transition strength [0.1, 1.0]
    count: INT,             // Raw occurrence count
    source: STRING          // "svo" | "corpus" | "conversation"
}]->(:Concept)

// Example:
(dream)-[:VIA {verb: "have", weight: 0.8}]->(meaning)
(dream)-[:VIA {verb: "make", weight: 0.7}]->(sense)
(love)-[:VIA {verb: "find", weight: 0.9}]->(way)

// OPERATES_ON - what verbs typically act upon
(:VerbOperator)-[:OPERATES_ON {
    weight: FLOAT,
    count: INT
}]->(:Concept)

// Example:
(understand:VerbOperator)-[:OPERATES_ON]->(meaning)
(understand:VerbOperator)-[:OPERATES_ON]->(word)
(understand:VerbOperator)-[:OPERATES_ON]->(need)
```

## Intent-Driven Queries

### 1. Find intent-aligned paths

```cypher
// User query: "help me understand my dream"
// Intent verbs: [understand, help]
// Root noun: dream

MATCH (root:Concept {word: $root})
      -[r:VIA]->
      (target:Concept)
WHERE r.verb IN $intent_verbs
   OR EXISTS {
       MATCH (v:VerbOperator)-[:OPERATES_ON]->(target)
       WHERE v.verb IN $intent_verbs
   }
RETURN target.word, r.verb, r.weight
ORDER BY r.weight DESC
```

### 2. Collapse to intent-relevant meanings

```cypher
// Find paths where verb matches intent OR target is operated by intent verb
MATCH path = (root:Concept {word: $root})
             -[:VIA*1..3]->
             (end:Concept)
WITH path,
     [r IN relationships(path) | r.verb] as verbs,
     end
WHERE ANY(v IN verbs WHERE v IN $intent_verbs)
   OR EXISTS {
       MATCH (op:VerbOperator)-[:OPERATES_ON]->(end)
       WHERE op.verb IN $intent_verbs
   }
RETURN path, end.word, end.g
ORDER BY end.g DESC
```

### 3. Build meaning tree with intent collapse

```cypher
// Recursive tree building with intent filtering
MATCH (root:Concept {word: $root})
CALL {
    WITH root
    MATCH (root)-[r:VIA]->(child:Concept)
    WHERE r.verb IN $intent_verbs
       OR child.word IN $intent_targets
    RETURN child, r.verb as verb, r.weight as weight
    ORDER BY weight DESC
    LIMIT $max_children
}
RETURN root, collect({
    word: child.word,
    verb: verb,
    weight: weight,
    g: child.g
}) as children
```

## Data Sources

### 1. SVO Patterns (PostgreSQL/CSV)

```
Source: hyp_svo_triads table / svo_patterns.csv
Format: subject -> [(verb, object), ...]

Example:
  dream -> [(make, sense), (have, meaning), (take, place)]
  love  -> [(find, way), (have, power), (know, bound)]
```

### 2. Verb Objects (PostgreSQL/CSV)

```
Source: verb_objects.csv
Format: verb -> [object1, object2, ...]

Example:
  understand -> [word, meaning, need, language]
  help -> [matter, situation, case]
```

### 3. Verb Operators (CSV)

```
Source: verb_operators.json / verb_operators.csv
Format: verb -> {j-vector, magnitude}

Example:
  understand -> {vector: [0.2, 0.3, ...], magnitude: 1.5}
```

## Loading Strategy

```python
# 1. Load concepts from main semantic space
for word in wholeness.states:
    if state.tau > 1.5:  # Skip function words
        create_concept(word, state.g, state.tau, state.j)

# 2. Load SVO patterns as VIA relationships
for subject, patterns in svo_patterns.items():
    for verb, object in patterns:
        create_via(subject, object, verb)

# 3. Load verb operators
for verb, data in verb_operators.items():
    create_verb_operator(verb, data['vector'], data['magnitude'])

# 4. Load verb-object relationships
for verb, objects in verb_objects.items():
    for obj in objects:
        create_operates_on(verb, obj)
```

## Docker Configuration

```yaml
# meaning_chain/config/docker-compose.yml
services:
  neo4j-meaning:
    image: neo4j:5.26.0
    container_name: neo4j-meaning-chain
    ports:
      - "7688:7687"    # Different port from main (7687)
      - "7475:7474"    # Different port from main (7474)
    environment:
      NEO4J_AUTH: neo4j/meaning123
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - meaning_data:/data
      - meaning_logs:/logs

volumes:
  meaning_data:
  meaning_logs:
```

## Integration with Tree Builder

```python
class IntentTreeBuilder:
    """Build meaning trees using intent-driven collapse."""

    def __init__(self, graph: MeaningGraph):
        self.graph = graph
        self.intent_verbs = set()
        self.intent_targets = set()

    def set_intent(self, verbs: List[str]):
        """Set intent verbs that drive navigation."""
        self.intent_verbs = set(verbs)
        # Get what these verbs operate on
        self.intent_targets = self.graph.get_verb_targets(verbs)

    def build_tree(self, root: str, depth: int = 3) -> MeaningTree:
        """Build tree collapsed to intent-relevant paths."""
        return self._build_node(root, depth, visited=set())

    def _build_node(self, word: str, depth: int, visited: Set[str]):
        """Recursively build with intent filtering."""
        if depth == 0 or word in visited:
            return MeaningNode(word)

        # Get intent-collapsed transitions
        transitions = self.graph.get_intent_transitions(
            word,
            self.intent_verbs,
            self.intent_targets
        )

        # Build children
        children = []
        for verb, target, weight in transitions[:4]:
            child = self._build_node(target, depth-1, visited | {word})
            child.verb_from_parent = verb
            children.append(child)

        return MeaningNode(word, children=children)
```

## Example Flow

```
User: "help me understand my dream"

1. Decompose:
   - Nouns: [dream]
   - Verbs: [help, understand]

2. Set Intent:
   - intent_verbs = {help, understand}
   - intent_targets = {meaning, word, need, matter, situation}

3. Query: dream with intent collapse

   MATCH (dream)-[r:VIA]->(target)
   WHERE r.verb IN ['help', 'understand']
      OR target.word IN ['meaning', 'word', 'need', 'matter', 'situation']

   Results:
   - dream -[have]-> meaning  (target in intent_targets!)
   - dream -[make]-> sense    (not directly, but semantic)

4. Build Tree:
   dream (root)
   └── [have] → meaning (intent-aligned!)
       └── [find] → truth
       └── [give] → life

5. Render to LLM with intent context
```

## Verb Dual Structure (The Pirate Insight)

Verbs have a parallel structure to nouns:

```
NOUN                    VERB
----                    ----
τ (tau)                 Δτ (delta_tau) - ascend/descend effect
g (good/evil)           Δg (delta_g) - moral push direction
j (5D position)         i (centered j) - intrinsic action type
                        j (raw) - effect direction
```

### The Phase Shift

Raw verb j-vectors are biased toward a global mean, making opposite verbs appear similar (99% cosine similarity). The "pirate insight" centers them:

```python
i_vector = j_vector - global_mean  # The "phase shift"
```

**Results**:
- create/destroy: 0.99 → 0.04 (now ORTHOGONAL)
- rise/fall: 0.97 → -0.25 (now OPPOSITE)
- love: i-dominant = +life (life-affirming action)
- love: Δτ = -0.41 (GROUNDS abstract→concrete)

### Key Verbs

| Verb | i-dominant | Δτ | Δg | Meaning |
|------|-----------|-----|-----|---------|
| love | +life | -0.41 | -0.64 | Life-affirming, grounds |
| create | +sacred | +0.07 | +0.03 | Sacred, stable level |
| destroy | +good | -0.08 | +0.69 | Moral, targets evil |
| find | +love | -0.13 | -0.36 | Connection-seeking |
| learn | -love | -0.08 | -0.74 | Knowledge over sentiment |

### Queries

```cypher
// Find grounding verbs (Δτ < -0.2)
MATCH (v:VerbOperator)
WHERE v.delta_tau < -0.2
RETURN v.verb, v.delta_tau, v.i
ORDER BY v.delta_tau ASC

// Find life-affirming verbs
MATCH (v:VerbOperator)
WHERE v.i[1] > 0.1  // life dimension is index 1
RETURN v.verb, v.i[1] as life_score
ORDER BY life_score DESC
```

See `docs/VERB_DUAL_STRUCTURE.md` for full documentation.

## Success Criteria

1. **Isolation**: No changes to experience_knowledge
2. **Intent-driven**: Verbs actually filter navigation
3. **Semantic**: Paths make semantic sense
4. **Testable**: Clear before/after comparison
5. **Phase-shifted**: Verb similarities are meaningful
