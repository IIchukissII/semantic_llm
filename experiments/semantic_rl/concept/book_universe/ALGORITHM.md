# Book Universe Creation Algorithm

## Overview

Creating a Book Universe transforms raw text into a navigable semantic graph.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   RAW    │───▶│ SEMANTIC │───▶│  GRAPH   │───▶│  BOOK    │
│   TEXT   │    │EXTRACTION│    │BUILDING  │    │ UNIVERSE │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## Pre-computed Semantic Space

**Critical**: We use a **pre-computed, normalized semantic space** from QuantumCore:

```
┌─────────────────────────────────────────────────────────────┐
│              QUANTUMCORE SEMANTIC SPACE                     │
│                    (19,055 words)                           │
├─────────────────────────────────────────────────────────────┤
│  Each word has FIXED, INTRINSIC semantic coordinates:       │
│                                                             │
│    τ (tau)      = word's abstraction level (unique per word)│
│    g (goodness) = word's moral direction   (unique per word)│
│    j (direction)= word's 5D semantic vector(unique per word)│
│                                                             │
│  Examples:                                                  │
│    "love"    → τ=2.2, g=+0.8, j=[...]                       │
│    "stone"   → τ=0.3, g=+0.1, j=[...]                       │
│    "fear"    → τ=1.5, g=-0.4, j=[...]                       │
│    "wisdom"  → τ=2.7, g=+0.6, j=[...]                       │
│                                                             │
│  NO computation needed:                                     │
│    ✗ TF-IDF weights                                         │
│    ✗ Document frequency                                     │
│    ✗ Word embeddings                                        │
│    ✗ Sentiment analysis                                     │
│                                                             │
│  τ, g, j are INTRINSIC to each word - not derived from      │
│  corpus statistics. Simply LOOKUP, never compute.           │
└─────────────────────────────────────────────────────────────┘
```

This means:
- **Concept filtering**: Check if word exists in semantic space (membership test)
- **Property lookup**: Direct O(1) lookup - each word has its own fixed τ, g, j
- **Semantic distance**: Computed from j-vector cosine similarity
- **No weighting**: Word properties are intrinsic, not corpus-dependent

---

## Phase 1: Semantic Extraction

### 1.1 Text Preprocessing

```python
def preprocess(text: str) -> List[str]:
    """
    Clean and tokenize text.

    Steps:
    1. Remove Gutenberg headers/footers
    2. Normalize whitespace
    3. Sentence segmentation
    4. Word tokenization
    5. Lowercase normalization
    """
    # Remove metadata
    text = remove_gutenberg_metadata(text)

    # Segment into sentences (preserve structure)
    sentences = sent_tokenize(text)

    # Tokenize words
    words = []
    for sent in sentences:
        words.extend(word_tokenize(sent.lower()))

    return words, sentences
```

### 1.2 Concept Extraction

```python
def extract_concepts(words: List[str],
                     semantic_space: Dict[str, SemanticState],
                     min_frequency: int = 3) -> List[str]:
    """
    Extract meaningful concepts from text.

    Criteria:
    1. Exists in semantic space (has τ, g, j)
    2. Appears at least min_frequency times
    3. Not a stopword
    4. Length >= 3 characters
    """
    # Count frequencies
    freq = Counter(words)

    # Filter to semantic concepts
    concepts = []
    for word, count in freq.items():
        if (count >= min_frequency and
            word in semantic_space and
            word not in STOPWORDS and
            len(word) >= 3):
            concepts.append(word)

    return concepts
```

### 1.3 Semantic Property Assignment

```python
def assign_properties(concepts: List[str],
                      semantic_space: Dict) -> Dict[str, ConceptNode]:
    """
    Assign semantic properties to each concept.

    Properties from semantic space:
    - τ (tau): abstraction level
    - g (goodness): moral/aesthetic direction
    - j (direction): 5D semantic orientation

    Derived properties:
    - altitude: τ (for visualization)
    - luminance: (g + 1) / 2 (for coloring)
    - mass: 2.0 - τ (concrete = heavy)
    """
    nodes = {}

    for concept in concepts:
        state = semantic_space[concept]

        nodes[concept] = ConceptNode(
            word=concept,
            tau=state.tau,
            goodness=state.goodness,
            j_vector=state.j,
            # Derived
            altitude=state.tau,
            luminance=(state.goodness + 1) / 2,
            mass=2.0 - state.tau
        )

    return nodes
```

---

## Phase 2: Graph Construction

### 2.1 Narrative Connection Detection

```python
def build_narrative_edges(text: str,
                          concepts: Set[str],
                          window_size: int = 30) -> List[Edge]:
    """
    Build edges based on narrative proximity.

    Two concepts are connected if they appear
    within window_size words of each other.

    Edge weight = Σ (1 / distance) for all co-occurrences
    """
    words = tokenize(text)
    cooccurrence = defaultdict(float)

    # Sliding window
    for i, word1 in enumerate(words):
        if word1 not in concepts:
            continue

        for j in range(i + 1, min(i + window_size, len(words))):
            word2 = words[j]
            if word2 in concepts and word2 != word1:
                # Weight by proximity
                weight = 1.0 / (j - i)
                cooccurrence[(word1, word2)] += weight

    # Create edges
    edges = []
    for (w1, w2), weight in cooccurrence.items():
        if weight >= THRESHOLD:
            edges.append(Edge(
                from_node=w1,
                to_node=w2,
                weight=weight,
                verb=infer_verb(w1, w2)  # See 2.2
            ))

    return edges
```

### 2.2 Verb Inference

```python
def infer_verb(from_word: str, to_word: str,
               context_sentences: List[str]) -> str:
    """
    Infer the verb connecting two concepts.

    Methods:
    1. Direct: Find sentences with both words, extract verb
    2. Semantic: Based on delta_g direction
    3. LLM: Ask LLM for appropriate connecting verb
    """
    # Method 1: Direct extraction
    for sentence in context_sentences:
        if from_word in sentence and to_word in sentence:
            verb = extract_verb_between(sentence, from_word, to_word)
            if verb:
                return verb

    # Method 2: Semantic inference
    delta_g = nodes[to_word].goodness - nodes[from_word].goodness

    if delta_g > 0.3:
        return random.choice(["discover", "find", "embrace", "achieve"])
    elif delta_g < -0.3:
        return random.choice(["face", "confront", "endure", "suffer"])
    else:
        return random.choice(["experience", "encounter", "witness", "know"])

    # Method 3: LLM inference (optional, expensive)
    # return llm.infer_verb(from_word, to_word, book_context)
```

### 2.3 Graph Assembly

```python
def assemble_graph(nodes: Dict[str, ConceptNode],
                   edges: List[Edge]) -> BookGraph:
    """
    Assemble the final book graph.

    Post-processing:
    1. Add reverse edges (bidirectional navigation)
    2. Compute thematic clusters
    3. Identify key waypoints
    4. Link to text passages
    """
    graph = BookGraph()

    # Add nodes
    for concept, node in nodes.items():
        graph.add_node(node)

    # Add edges (with reverse)
    for edge in edges:
        graph.add_edge(edge)
        graph.add_edge(edge.reverse())

    # Compute clusters
    graph.clusters = detect_thematic_clusters(graph)

    # Find key waypoints
    graph.waypoints = find_key_waypoints(graph)

    # Link passages
    graph.passages = link_passages(graph, original_text)

    return graph
```

---

## Phase 3: Passage Linking

### 3.1 Concept-to-Passage Mapping

```python
def link_passages(graph: BookGraph,
                  text: str,
                  context_window: int = 200) -> Dict[str, List[Passage]]:
    """
    Link each concept to relevant text passages.

    For each concept, find sentences where it appears
    and extract surrounding context.
    """
    sentences = sent_tokenize(text)
    passages = defaultdict(list)

    for concept in graph.nodes:
        for i, sentence in enumerate(sentences):
            if concept in sentence.lower():
                # Get context
                start = max(0, i - 2)
                end = min(len(sentences), i + 3)
                context = ' '.join(sentences[start:end])

                passages[concept].append(Passage(
                    text=context,
                    sentence_index=i,
                    relevance=compute_relevance(concept, context)
                ))

    # Keep top N passages per concept
    for concept in passages:
        passages[concept] = sorted(
            passages[concept],
            key=lambda p: p.relevance,
            reverse=True
        )[:MAX_PASSAGES]

    return passages
```

### 3.2 Passage Relevance Scoring

```python
def compute_relevance(concept: str, passage: str) -> float:
    """
    Score how relevant a passage is to a concept.

    Factors:
    1. Frequency of concept in passage
    2. Presence of related concepts
    3. Narrative importance (proper nouns, dialogue)
    4. Semantic density
    """
    score = 0.0

    # Frequency
    freq = passage.lower().count(concept)
    score += freq * 0.3

    # Related concepts
    related = get_neighbors(concept)
    for r in related:
        if r in passage.lower():
            score += 0.2

    # Narrative markers
    if '"' in passage:  # Dialogue
        score += 0.3
    if any(c.isupper() for c in passage.split()[0]):  # Proper noun
        score += 0.1

    return score
```

---

## Phase 4: Universe Finalization

### 4.1 Journey Endpoint Detection

```python
def detect_journey_endpoints(graph: BookGraph) -> Tuple[str, str]:
    """
    Detect natural start and goal for the book journey.

    Start: Low goodness concept with good connectivity
    Goal: High goodness concept (often abstract)
    """
    # Preferred archetypes
    START_ARCHETYPES = [
        "darkness", "fear", "confusion", "ignorance",
        "poverty", "isolation", "doubt", "chaos"
    ]

    GOAL_ARCHETYPES = [
        "wisdom", "love", "truth", "redemption",
        "freedom", "peace", "understanding", "light"
    ]

    # Find best start
    start = None
    for archetype in START_ARCHETYPES:
        if archetype in graph.nodes:
            if len(graph.get_neighbors(archetype)) > 0:
                start = archetype
                break

    # Fallback: most connected low-g node
    if not start:
        start = min(graph.nodes,
                    key=lambda n: graph.nodes[n].goodness)

    # Find best goal
    goal = None
    for archetype in GOAL_ARCHETYPES:
        if archetype in graph.nodes:
            goal = archetype
            break

    # Fallback: highest-g reachable node
    if not goal:
        goal = max(graph.nodes,
                   key=lambda n: graph.nodes[n].goodness)

    return start, goal
```

### 4.2 Thematic Cluster Detection

```python
def detect_thematic_clusters(graph: BookGraph) -> List[Cluster]:
    """
    Detect thematic clusters using community detection.

    Uses Louvain algorithm on semantic similarity graph.
    """
    # Build similarity matrix
    n = len(graph.nodes)
    nodes = list(graph.nodes.keys())
    similarity = np.zeros((n, n))

    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i < j:
                sim = cosine_similarity(
                    graph.nodes[n1].j_vector,
                    graph.nodes[n2].j_vector
                )
                similarity[i, j] = similarity[j, i] = sim

    # Community detection
    communities = louvain_communities(similarity)

    # Build clusters
    clusters = []
    for community in communities:
        cluster = Cluster(
            nodes=[nodes[i] for i in community],
            center=find_cluster_center(community, graph),
            theme=infer_theme(community, graph)
        )
        clusters.append(cluster)

    return clusters
```

### 4.3 Final Export

```python
def export_universe(graph: BookGraph,
                    metadata: BookMetadata,
                    output_path: str):
    """
    Export the complete book universe.

    Format: JSON with all data needed for exploration.
    """
    universe = {
        "metadata": {
            "title": metadata.title,
            "author": metadata.author,
            "created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "nodes": {
            word: {
                "tau": node.tau,
                "goodness": node.goodness,
                "j_vector": node.j_vector.tolist(),
                "passages": [p.to_dict() for p in node.passages]
            }
            for word, node in graph.nodes.items()
        },
        "edges": [
            {
                "from": e.from_node,
                "to": e.to_node,
                "verb": e.verb,
                "weight": e.weight
            }
            for e in graph.edges
        ],
        "clusters": [c.to_dict() for c in graph.clusters],
        "journey": {
            "start": graph.start,
            "goal": graph.goal,
            "waypoints": graph.waypoints
        }
    }

    with open(output_path, 'w') as f:
        json.dump(universe, f, indent=2)
```

---

## Complete Pipeline

```python
def create_book_universe(book_path: str,
                         semantic_space: Dict,
                         output_path: str) -> BookGraph:
    """
    Complete pipeline to create a book universe.
    """
    # Phase 1: Extraction
    print("Phase 1: Semantic Extraction")
    text = load_text(book_path)
    words, sentences = preprocess(text)
    concepts = extract_concepts(words, semantic_space)
    nodes = assign_properties(concepts, semantic_space)
    print(f"  Extracted {len(concepts)} concepts")

    # Phase 2: Graph Construction
    print("Phase 2: Graph Construction")
    edges = build_narrative_edges(text, set(concepts))
    graph = assemble_graph(nodes, edges)
    print(f"  Built {len(edges)} connections")

    # Phase 3: Passage Linking
    print("Phase 3: Passage Linking")
    graph.passages = link_passages(graph, text)
    print(f"  Linked {sum(len(p) for p in graph.passages.values())} passages")

    # Phase 4: Finalization
    print("Phase 4: Finalization")
    graph.start, graph.goal = detect_journey_endpoints(graph)
    graph.clusters = detect_thematic_clusters(graph)
    print(f"  Journey: {graph.start} → {graph.goal}")
    print(f"  Clusters: {len(graph.clusters)}")

    # Export
    metadata = extract_metadata(book_path)
    export_universe(graph, metadata, output_path)
    print(f"  Exported to: {output_path}")

    return graph
```

---

## Optimization Considerations

### Computational Complexity

| Phase | Complexity | Optimization |
|-------|------------|--------------|
| Preprocessing | O(n) | Streaming |
| Concept Extraction | O(n) | Bloom filter |
| Edge Building | O(n × w) | Inverted index |
| Clustering | O(n²) | Approximate methods |
| Passage Linking | O(n × s) | Pre-indexing |

Where n = text length, w = window size, s = sentences

### Memory Optimization

```python
# Stream processing for large books
def stream_process(book_path: str, chunk_size: int = 10000):
    """Process book in chunks to limit memory."""
    with open(book_path) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield process_chunk(chunk)
```

### Caching Strategy

```python
# Cache semantic lookups
@lru_cache(maxsize=20000)
def get_semantic_properties(word: str) -> Optional[SemanticState]:
    return semantic_space.get(word)
```

---

## Quality Metrics

### Graph Quality

```python
def evaluate_graph_quality(graph: BookGraph) -> Dict[str, float]:
    """Evaluate the quality of the generated graph."""
    return {
        "connectivity": avg_degree(graph) / len(graph.nodes),
        "coverage": len(graph.nodes) / total_content_words,
        "cluster_coherence": avg_intra_cluster_similarity(graph),
        "path_diversity": count_unique_paths(graph.start, graph.goal),
        "passage_relevance": avg_passage_relevance(graph)
    }
```

### Recommended Thresholds

| Metric | Minimum | Target |
|--------|---------|--------|
| Connectivity | 0.05 | 0.15 |
| Coverage | 0.3 | 0.5 |
| Cluster Coherence | 0.6 | 0.8 |
| Path Diversity | 10 | 100+ |
| Passage Relevance | 0.5 | 0.7 |
