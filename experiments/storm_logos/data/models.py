"""Data Models: Core data structures for Storm-Logos.

All dataclasses for bonds, coordinates, trajectories, and states.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


# ============================================================================
# WORD COORDINATES
# ============================================================================

@dataclass
class WordCoordinates:
    """Coordinates for a single word in semantic space."""
    word: str
    A: float = 0.0       # Affirmation axis (-1 to +1)
    S: float = 0.0       # Sacred axis (-1 to +1)
    tau: float = 2.5     # Abstraction level (0.5 to 4.5)
    source: str = 'unknown'  # Where coordinates came from

    @property
    def coords(self) -> Tuple[float, float, float]:
        """Return (A, S, τ) tuple."""
        return (self.A, self.S, self.tau)

    def as_array(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.A, self.S, self.tau])


# ============================================================================
# BOND
# ============================================================================

@dataclass
class Bond:
    """A noun-adjective bond with coordinates."""
    noun: str
    adj: Optional[str] = None
    variety: int = 0     # Frequency count in corpus
    A: float = 0.0
    S: float = 0.0
    tau: float = 2.5

    @property
    def text(self) -> str:
        """Human-readable bond text."""
        if self.adj:
            return f"{self.adj} {self.noun}"
        return self.noun

    @property
    def coords(self) -> Tuple[float, float, float]:
        """Return (A, S, τ) tuple."""
        return (self.A, self.S, self.tau)

    def as_array(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return np.array([self.A, self.S, self.tau])

    def distance_to(self, other: 'Bond') -> float:
        """Euclidean distance to another bond."""
        return np.linalg.norm(self.as_array() - other.as_array())

    def __repr__(self) -> str:
        return f"Bond({self.text}, A={self.A:+.2f}, S={self.S:+.2f}, τ={self.tau:.2f})"


# ============================================================================
# SEMANTIC STATE
# ============================================================================

@dataclass
class SemanticState:
    """Position in semantic space with psychological markers.

    This represents the current "charge" on the RC capacitor.
    """
    A: float = 0.0      # Affirmation axis (-1 to +1)
    S: float = 0.0      # Sacred axis (-1 to +1)
    tau: float = 2.5    # Abstraction level (0.5 to 4.5)

    # Detected markers
    irony: float = 0.0       # 0-1 irony strength
    sarcasm: float = 0.0     # 0-1 sarcasm strength
    emotion: str = "neutral"  # Primary emotion
    intensity: float = 0.0   # Emotional intensity 0-1

    # Extended markers
    vulnerability: float = 0.0
    deflection: float = 0.0
    self_deprecation: float = 0.0
    minimization: float = 0.0
    projection: float = 0.0
    rationalization: float = 0.0
    humor_defense: float = 0.0

    def copy(self) -> 'SemanticState':
        """Create a copy of this state."""
        return SemanticState(
            A=self.A, S=self.S, tau=self.tau,
            irony=self.irony, sarcasm=self.sarcasm,
            emotion=self.emotion, intensity=self.intensity,
            vulnerability=self.vulnerability,
            deflection=self.deflection,
            self_deprecation=self.self_deprecation,
            minimization=self.minimization,
            projection=self.projection,
            rationalization=self.rationalization,
            humor_defense=self.humor_defense,
        )

    @property
    def coords(self) -> Tuple[float, float, float]:
        """Return (A, S, τ) tuple."""
        return (self.A, self.S, self.tau)

    def as_array(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return np.array([self.A, self.S, self.tau])

    def distance_to(self, other: 'SemanticState') -> float:
        """Euclidean distance to another state."""
        return (
            (self.A - other.A)**2 +
            (self.S - other.S)**2 +
            (self.tau - other.tau)**2
        ) ** 0.5

    def __repr__(self) -> str:
        return f"State(A={self.A:+.2f}, S={self.S:+.2f}, τ={self.tau:.2f})"


# ============================================================================
# TRAJECTORY
# ============================================================================

@dataclass
class Trajectory:
    """A sequence of bonds forming a semantic path.

    Used for:
    - Book analysis (author's walked path)
    - Generation (output skeleton)
    - Conversation history
    """
    bonds: List[Bond] = field(default_factory=list)
    states: List[SemanticState] = field(default_factory=list)
    sentence_boundaries: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.bonds)

    def add(self, bond: Bond, state: Optional[SemanticState] = None):
        """Add a bond (and optionally its state) to trajectory."""
        self.bonds.append(bond)
        if state:
            self.states.append(state)

    @property
    def current(self) -> Optional[Bond]:
        """Most recent bond."""
        return self.bonds[-1] if self.bonds else None

    @property
    def current_state(self) -> Optional[SemanticState]:
        """Most recent state."""
        return self.states[-1] if self.states else None

    def get_coords(self) -> np.ndarray:
        """Get all coordinates as (N, 3) array."""
        if not self.bonds:
            return np.array([]).reshape(0, 3)
        return np.array([b.as_array() for b in self.bonds])

    def get_window(self, size: int = 10) -> List[Bond]:
        """Get last N bonds."""
        return self.bonds[-size:]

    def clear(self):
        """Reset trajectory."""
        self.bonds.clear()
        self.states.clear()
        self.sentence_boundaries.clear()


# ============================================================================
# CONVERSATION TRAJECTORY
# ============================================================================

@dataclass
class ConversationTrajectory:
    """Track conversation path through semantic space.

    Implements RC dynamics: state accumulates like charge on capacitor.
    """
    history: List[SemanticState] = field(default_factory=list)
    window_size: int = 10  # RC memory window

    def add(self, state: SemanticState):
        """Add new state to trajectory."""
        self.history.append(state.copy())

    @property
    def current(self) -> Optional[SemanticState]:
        """Most recent state."""
        return self.history[-1] if self.history else None

    @property
    def previous(self) -> Optional[SemanticState]:
        """Previous state."""
        return self.history[-2] if len(self.history) >= 2 else None

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Semantic velocity: direction of movement (dA, dS, dτ)."""
        if len(self.history) < 2:
            return (0.0, 0.0, 0.0)
        prev, curr = self.history[-2], self.history[-1]
        return (
            curr.A - prev.A,
            curr.S - prev.S,
            curr.tau - prev.tau
        )

    @property
    def mean_state(self) -> SemanticState:
        """Average state over recent window (RC charge)."""
        if not self.history:
            return SemanticState()
        window = self.history[-self.window_size:]
        n = len(window)
        return SemanticState(
            A=sum(s.A for s in window) / n,
            S=sum(s.S for s in window) / n,
            tau=sum(s.tau for s in window) / n,
            irony=sum(s.irony for s in window) / n,
            sarcasm=sum(s.sarcasm for s in window) / n,
        )

    @property
    def n_turns(self) -> int:
        """Number of turns in conversation."""
        return len(self.history)

    def clear(self):
        """Reset trajectory."""
        self.history.clear()


# ============================================================================
# GENERATION RESULT
# ============================================================================

@dataclass
class GenerationResult:
    """Result of a single generation step."""
    bond: Bond
    new_state: SemanticState
    candidates_count: int
    filtered_count: int
    winner_score: float
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class Metrics:
    """Metrics output from the Metrics Engine."""
    # Core metrics
    irony: float = 0.0
    coherence: float = 0.0
    tau_mean: float = 0.0
    tau_variance: float = 0.0
    tau_slope: float = 0.0
    tension_score: float = 0.0

    # Position metrics
    A_position: float = 0.0
    S_position: float = 0.0

    # Defense metrics
    defenses: List[str] = field(default_factory=list)

    # Derived metrics
    noise_ratio: float = 0.0
    boundary_detected: bool = False

    def as_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'irony': self.irony,
            'coherence': self.coherence,
            'tau_mean': self.tau_mean,
            'tau_variance': self.tau_variance,
            'tau_slope': self.tau_slope,
            'tension_score': self.tension_score,
            'A_position': self.A_position,
            'S_position': self.S_position,
            'defenses': self.defenses,
            'noise_ratio': self.noise_ratio,
            'boundary_detected': self.boundary_detected,
        }


# ============================================================================
# ERRORS
# ============================================================================

@dataclass
class Errors:
    """Error signals from the Feedback Engine."""
    coherence_error: float = 0.0
    irony_error: float = 0.0
    tension_error: float = 0.0
    tau_slope_error: float = 0.0
    tau_variance_error: float = 0.0
    noise_ratio_error: float = 0.0

    # Accumulated errors (for I term)
    integral: Dict[str, float] = field(default_factory=dict)

    # Rate of change (for D term if needed)
    derivative: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'coherence_error': self.coherence_error,
            'irony_error': self.irony_error,
            'tension_error': self.tension_error,
            'tau_slope_error': self.tau_slope_error,
            'tau_variance_error': self.tau_variance_error,
            'noise_ratio_error': self.noise_ratio_error,
        }


# ============================================================================
# PARAMETERS
# ============================================================================

@dataclass
class Parameters:
    """Adaptive parameters controlled by the PI controller."""
    storm_radius: float = 1.0
    dialectic_tension: float = 0.5
    chain_decay: float = 0.85
    gravity_strength: float = 0.5
    mirror_depth: float = 0.5
    antithesis_weight: float = 0.5
    coherence_threshold: float = 0.3

    def as_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'storm_radius': self.storm_radius,
            'dialectic_tension': self.dialectic_tension,
            'chain_decay': self.chain_decay,
            'gravity_strength': self.gravity_strength,
            'mirror_depth': self.mirror_depth,
            'antithesis_weight': self.antithesis_weight,
            'coherence_threshold': self.coherence_threshold,
        }

    def copy(self) -> 'Parameters':
        """Create a copy."""
        return Parameters(**self.as_dict())


# ============================================================================
# DREAM ANALYSIS MODELS
# ============================================================================

@dataclass
class DreamSymbol:
    """A symbol extracted from a dream with archetypal analysis."""
    bond: Bond
    raw_text: str
    archetype: str = ""          # shadow, anima, self, mother, father, hero, trickster
    archetype_score: float = 0.0  # Confidence in archetype identification
    interpretation: str = ""      # Symbol-specific interpretation
    corpus_sources: List[str] = field(default_factory=list)  # Books where symbol appears

    def as_dict(self) -> Dict:
        return {
            "text": self.raw_text,
            "bond": self.bond.text,
            "A": self.bond.A,
            "S": self.bond.S,
            "tau": self.bond.tau,
            "archetype": self.archetype,
            "archetype_score": self.archetype_score,
            "interpretation": self.interpretation,
            "corpus_sources": self.corpus_sources,
        }


@dataclass
class DreamState:
    """State of a dream analysis session.

    Tracks emotional trajectory and archetypal patterns through dream exploration.
    """
    # Semantic coordinates (averaged from symbols)
    A: float = 0.0
    S: float = 0.0
    tau: float = 2.5

    # Archetypal scores (0-1 presence strength)
    shadow: float = 0.0
    anima_animus: float = 0.0
    self_archetype: float = 0.0
    mother: float = 0.0
    father: float = 0.0
    hero: float = 0.0
    trickster: float = 0.0
    death_rebirth: float = 0.0

    # Dream-specific markers
    lucidity: float = 0.0        # Awareness within dream
    transformation: float = 0.0   # Presence of change/metamorphosis
    journey: float = 0.0          # Quest/travel elements
    confrontation: float = 0.0    # Conflict/facing something

    def dominant_archetype(self) -> Tuple[str, float]:
        """Return the strongest archetype and its score."""
        archetypes = {
            "shadow": self.shadow,
            "anima_animus": self.anima_animus,
            "self": self.self_archetype,
            "mother": self.mother,
            "father": self.father,
            "hero": self.hero,
            "trickster": self.trickster,
            "death_rebirth": self.death_rebirth,
        }
        if not any(archetypes.values()):
            return ("unknown", 0.0)
        best = max(archetypes.items(), key=lambda x: x[1])
        return best

    def as_dict(self) -> Dict:
        return {
            "coordinates": {"A": self.A, "S": self.S, "tau": self.tau},
            "archetypes": {
                "shadow": self.shadow,
                "anima_animus": self.anima_animus,
                "self": self.self_archetype,
                "mother": self.mother,
                "father": self.father,
                "hero": self.hero,
                "trickster": self.trickster,
                "death_rebirth": self.death_rebirth,
            },
            "markers": {
                "lucidity": self.lucidity,
                "transformation": self.transformation,
                "journey": self.journey,
                "confrontation": self.confrontation,
            },
            "dominant_archetype": self.dominant_archetype()[0],
        }


@dataclass
class DreamAnalysis:
    """Complete analysis of a dream."""
    dream_text: str
    symbols: List[DreamSymbol] = field(default_factory=list)
    state: DreamState = field(default_factory=DreamState)
    interpretation: str = ""
    corpus_resonances: List[Dict] = field(default_factory=list)
    timestamp: str = ""

    def as_dict(self) -> Dict:
        return {
            "dream_text": self.dream_text,
            "symbols": [s.as_dict() for s in self.symbols],
            "state": self.state.as_dict(),
            "interpretation": self.interpretation,
            "corpus_resonances": self.corpus_resonances,
            "timestamp": self.timestamp,
        }
