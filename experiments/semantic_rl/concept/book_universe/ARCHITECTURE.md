# Book Universe: System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BOOK UNIVERSE PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │   READER    │   │    LLM      │   │  UNIVERSE   │   │   SOCIAL    │ │
│  │  INTERFACE  │◄─▶│   GUIDE     │◄─▶│   ENGINE    │◄─▶│   LAYER     │ │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘ │
│         │                 │                 │                 │         │
│         └─────────────────┴────────┬────────┴─────────────────┘         │
│                                    │                                    │
│                           ┌────────┴────────┐                           │
│                           │   CORE ENGINE   │                           │
│                           └────────┬────────┘                           │
│                                    │                                    │
│         ┌──────────────────────────┼──────────────────────────┐        │
│         │                          │                          │        │
│  ┌──────┴──────┐           ┌───────┴───────┐          ┌───────┴──────┐ │
│  │  SEMANTIC   │           │    GRAPH      │          │   PASSAGE    │ │
│  │   SPACE     │           │   DATABASE    │          │    INDEX     │ │
│  │  (19K words)│           │   (per book)  │          │  (per book)  │ │
│  └─────────────┘           └───────────────┘          └──────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Reader Interface

The entry point for human interaction with the book universe.

```
┌─────────────────────────────────────────────────────────────────┐
│                      READER INTERFACE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   UNIVERSE VIEW                          │   │
│  │     3D/2D visualization of semantic landscape           │   │
│  │     - Concepts as glowing nodes                         │   │
│  │     - Paths as connecting threads                       │   │
│  │     - Reader avatar showing current position            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  CURRENT STATE   │  │   AVAILABLE      │  │   JOURNEY    │  │
│  │                  │  │   ACTIONS        │  │   LOG        │  │
│  │  Concept: FEAR   │  │                  │  │              │  │
│  │  τ: 1.5          │  │  → face COURAGE  │  │  darkness    │  │
│  │  g: -0.2         │  │  → embrace HOPE  │  │  ↓           │  │
│  │  believe: 0.7    │  │  ⚡ tunnel       │  │  fear ←      │  │
│  │                  │  │                  │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    PASSAGE PANEL                         │   │
│  │                                                          │   │
│  │  "He felt the fear rising in his chest, but knew        │   │
│  │   that to face it was the only way forward..."          │   │
│  │                                    - Chapter 3, p.47    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Interface Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Explore** | Free navigation | Discovery |
| **Guided** | LLM suggests paths | Learning |
| **Challenge** | Find path to goal | Gamified |
| **Social** | See others' paths | Community |
| **Study** | Academic analysis | Research |

### 2. LLM Guide

The AI companion that narrates and assists exploration.

```python
class LLMGuide:
    """
    AI guide for book universe exploration.

    Responsibilities:
    1. Narrate transitions poetically
    2. Suggest meaningful paths
    3. Explain concept connections
    4. Read relevant passages aloud
    5. Answer questions about the book
    """

    def narrate_arrival(self, concept: str, journey: Journey) -> str:
        """Generate narrative for arriving at a concept."""
        prompt = f"""
        The reader has arrived at '{concept}' in {self.book_title}.
        Journey so far: {journey.path}

        Generate a brief, evocative description (2-3 sentences)
        of what it means to arrive at this concept in the story.
        """
        return self.llm.generate(prompt)

    def suggest_path(self, current: str, goal: str,
                     visited: Set[str]) -> List[str]:
        """Suggest meaningful next steps."""
        prompt = f"""
        In {self.book_title}, the reader is at '{current}'.
        Goal: reach '{goal}'
        Already visited: {visited}

        Suggest 3 meaningful next concepts to explore,
        with brief reasons for each.
        """
        return self.parse_suggestions(self.llm.generate(prompt))

    def explain_tunnel(self, from_c: str, to_c: str) -> str:
        """Explain a tunneling insight."""
        prompt = f"""
        The reader just had an insight, jumping from '{from_c}'
        directly to '{to_c}' in {self.book_title}.

        Explain this moment of understanding poetically,
        as if a sudden connection became clear.
        """
        return self.llm.generate(prompt)

    def read_passage(self, concept: str) -> str:
        """Read a relevant passage with context."""
        passage = self.get_best_passage(concept)
        return self.llm.enhance_reading(passage, concept)
```

### 3. Universe Engine

The core simulation engine managing the semantic world.

```python
class UniverseEngine:
    """
    Core engine for book universe simulation.

    Manages:
    - Reader state (position, believe, knowledge)
    - World state (graph, passages, clusters)
    - Physics (movement costs, tunneling)
    - Events (arrivals, insights, discoveries)
    """

    def __init__(self, universe_path: str):
        self.graph = load_graph(universe_path)
        self.passages = load_passages(universe_path)
        self.physics = SemanticPhysics()

    def move_thermal(self, reader: Reader,
                     verb: str, target: str) -> MoveResult:
        """Execute thermal (gradual) movement."""
        # Verify connection exists
        if not self.graph.has_edge(reader.position, target, verb):
            return MoveResult(success=False, reason="no_path")

        # Compute cost
        cost = self.physics.movement_cost(
            self.graph.nodes[reader.position],
            self.graph.nodes[target]
        )

        # Check energy
        if reader.energy < cost:
            return MoveResult(success=False, reason="insufficient_energy")

        # Execute move
        reader.energy -= cost
        reader.position = target
        reader.visited.add(target)
        reader.path.append((verb, target))

        # Update believe
        delta_g = self.graph.nodes[target].goodness
        reader.believe = self.physics.update_believe(reader.believe, delta_g)

        return MoveResult(
            success=True,
            new_position=target,
            cost=cost,
            passages=self.passages.get(target, [])
        )

    def attempt_tunnel(self, reader: Reader,
                       target: str) -> TunnelResult:
        """Attempt quantum tunneling."""
        # Check if target is reachable (lived knowledge)
        if not self.is_tunnel_reachable(target, reader.visited):
            return TunnelResult(
                success=False,
                reason="not_in_lived_knowledge"
            )

        # Compute probability
        prob = self.physics.tunnel_probability(
            self.graph.nodes[reader.position],
            self.graph.nodes[target],
            reader.believe
        )

        # Attempt
        if random.random() < prob:
            # Success!
            reader.position = target
            reader.visited.add(target)
            reader.path.append(("⚡tunnel", target))
            reader.insights.append(target)
            reader.believe = min(1.0, reader.believe + 0.1)

            return TunnelResult(
                success=True,
                probability=prob,
                new_position=target,
                passages=self.passages.get(target, [])
            )
        else:
            # Failed
            reader.believe = max(0.1, reader.believe - 0.05)
            return TunnelResult(
                success=False,
                probability=prob,
                reason="probability_failed"
            )

    def is_tunnel_reachable(self, target: str,
                            visited: Set[str]) -> bool:
        """
        Check if target is reachable via tunneling.

        A concept is reachable if:
        1. It has been visited before, OR
        2. It is directly connected to a visited concept
        """
        if target in visited:
            return True

        for v in visited:
            if self.graph.has_any_edge(v, target):
                return True

        return False
```

### 4. Social Layer

Multi-reader features and community interaction.

```python
class SocialLayer:
    """
    Social features for shared exploration.

    Features:
    1. Path sharing - see where others have been
    2. Insights - leave notes at concepts
    3. Challenges - race to goals
    4. Book clubs - explore together
    """

    def get_heatmap(self, book_id: str) -> Dict[str, float]:
        """Get visitation heatmap for a book."""
        visits = self.db.aggregate_visits(book_id)
        max_visits = max(visits.values())
        return {c: v / max_visits for c, v in visits.items()}

    def get_popular_paths(self, book_id: str,
                          from_c: str, to_c: str,
                          limit: int = 5) -> List[Path]:
        """Get most popular paths between concepts."""
        return self.db.query_paths(
            book_id, from_c, to_c,
            order_by="popularity",
            limit=limit
        )

    def leave_insight(self, reader: Reader,
                      concept: str, text: str):
        """Leave an insight for others to find."""
        insight = Insight(
            reader_id=reader.id,
            concept=concept,
            text=text,
            timestamp=now(),
            upvotes=0
        )
        self.db.save_insight(insight)

    def get_insights(self, concept: str,
                     limit: int = 10) -> List[Insight]:
        """Get insights left at a concept."""
        return self.db.query_insights(
            concept=concept,
            order_by="upvotes",
            limit=limit
        )

    def create_book_club_session(self,
                                 book_id: str,
                                 readers: List[Reader]) -> Session:
        """Create a shared exploration session."""
        session = Session(
            book_id=book_id,
            readers=readers,
            start_time=now(),
            shared_visited=set(),
            chat_history=[]
        )
        return session
```

---

## Data Models

### Reader State

```python
@dataclass
class Reader:
    id: str
    position: str              # Current concept
    believe: float            # 0.0 - 1.0
    energy: float             # Movement resource
    visited: Set[str]         # Lived knowledge
    path: List[Tuple[str, str]]  # (verb, concept) history
    insights: List[str]       # Tunnel destinations
    start_time: datetime
```

### Book Universe

```python
@dataclass
class BookUniverse:
    id: str
    title: str
    author: str
    graph: SemanticGraph
    passages: Dict[str, List[Passage]]
    clusters: List[Cluster]
    journey_start: str
    journey_goal: str
    metadata: Dict
```

### Exploration Event

```python
@dataclass
class ExplorationEvent:
    event_type: str  # "move", "tunnel", "read_passage", "insight"
    reader_id: str
    book_id: str
    from_concept: Optional[str]
    to_concept: str
    verb: Optional[str]
    timestamp: datetime
    metadata: Dict
```

---

## API Design

### REST API

```yaml
# Book Universe API

/api/v1/universes:
  GET:
    description: List available book universes
    response: List[UniverseSummary]

  POST:
    description: Create universe from book
    body: { book_path: str, options: CreateOptions }
    response: Universe

/api/v1/universes/{id}:
  GET:
    description: Get universe details
    response: Universe

/api/v1/universes/{id}/explore:
  POST:
    description: Start exploration session
    body: { reader_id: str, start_concept?: str }
    response: Session

/api/v1/sessions/{id}/move:
  POST:
    description: Execute movement
    body: { verb: str, target: str } | { tunnel: true, target: str }
    response: MoveResult

/api/v1/sessions/{id}/state:
  GET:
    description: Get current session state
    response: SessionState

/api/v1/universes/{id}/passages/{concept}:
  GET:
    description: Get passages for concept
    response: List[Passage]

/api/v1/universes/{id}/social/heatmap:
  GET:
    description: Get exploration heatmap
    response: Dict[str, float]
```

### WebSocket Events

```yaml
# Real-time events

client -> server:
  move: { verb: str, target: str }
  tunnel: { target: str }
  request_narration: { concept: str }
  leave_insight: { concept: str, text: str }

server -> client:
  position_update: { concept: str, passages: List[Passage] }
  tunnel_result: { success: bool, narrative: str }
  narration: { text: str, audio_url?: str }
  other_reader_arrived: { reader_id: str, concept: str }
  new_insight: { reader_id: str, concept: str, text: str }
```

---

## Visualization Subsystem

### 3D Universe View

```
    τ (altitude)
    │
    │       ★ WISDOM (goal)
    │      ╱│╲
    │     ╱ │ ╲
    │    ╱  │  ╲
    │   ○───○───○   (high concepts)
    │    ╲ │ ╱
    │     ╲│╱
    │      ○       (mid concepts)
    │     ╱│╲
    │    ╱ │ ╲
    │   ○──○──○    (low concepts)
    │      │
    │      ◉ ← YOU ARE HERE
    │      │
    └──────┴────────────── g (goodness)
          │
         -1    0    +1

Legend:
  ★ = Goal
  ◉ = Reader position
  ○ = Concepts
  ─ = Connections
```

### Visual Elements

| Element | Representation | Property Mapped |
|---------|---------------|-----------------|
| Node size | Radius | Importance (degree) |
| Node color | Gradient | Goodness (g) |
| Node height | Y-position | Abstraction (τ) |
| Node glow | Intensity | Visit frequency |
| Edge thickness | Width | Connection strength |
| Edge color | Gradient | Delta-g |
| Path trail | Glowing line | Reader's journey |
| Tunnel | Lightning bolt | Insight moment |

---

## Performance Considerations

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    CACHING LAYERS                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L1: In-Memory (per session)                                │
│      - Current graph region                                 │
│      - Recent passages                                      │
│      - Reader state                                         │
│                                                             │
│  L2: Redis (shared)                                         │
│      - Popular paths                                        │
│      - Heatmaps                                             │
│      - LLM response cache                                   │
│                                                             │
│  L3: Database (persistent)                                  │
│      - Full universe data                                   │
│      - All passages                                         │
│      - Historical analytics                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Scalability

| Component | Strategy |
|-----------|----------|
| Universe Engine | Stateless, horizontal scaling |
| LLM Guide | Queue-based, dedicated GPU nodes |
| Graph Database | Neo4j cluster |
| Passage Index | Elasticsearch |
| Real-time | Redis pub/sub |
| Storage | S3 for large assets |
