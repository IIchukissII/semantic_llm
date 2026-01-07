"""Storm-Logos Configuration.

Central configuration for all system parameters.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path


# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================

# Boltzmann temperature: sets scale of semantic fluctuations
KT = math.exp(-1/5)  # ≈ 0.819

# RC dynamics constants
Q_MAX = 2.0           # Saturation limit
DECAY = 0.05          # State decay rate
DT = 0.5              # Time step

# Gravity constants
LAMBDA = 0.5          # Pull toward concrete (lower τ)
MU = 0.5              # Pull toward good (higher A)

# Zipf law constants
ALPHA_0 = 2.5         # Baseline Zipf exponent
ALPHA_1 = -1.4        # τ-dependent Zipf: α(τ) = α₀ + α₁×τ


# ============================================================================
# HOMEOSTATIC TARGETS
# ============================================================================

HOMEOSTATIC_TARGETS = {
    'coherence': 0.70,
    'irony': 0.15,
    'tension': 0.60,
    'tau_variance': 0.80,
    'noise_ratio': 0.20,
    'tau_slope': -0.10,
}


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    host: str = 'localhost'
    database: str = 'bonds'
    user: str = 'bonds'
    password: str = 'bonds_secret'
    port: int = 5432

    def as_dict(self) -> Dict:
        return {
            'host': self.host,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'port': self.port,
        }


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'neo4j_secret'


# ============================================================================
# SEMANTIC LAYER CONFIGURATION
# ============================================================================

@dataclass
class StormConfig:
    """Storm algorithm configuration."""
    radius: float = 1.0              # Search radius in (A, S, τ) space
    max_candidates: int = 100        # Maximum candidates to return
    min_variety: int = 3             # Minimum bond frequency
    sources_weight: Dict[str, float] = field(default_factory=lambda: {
        'follows': 0.4,              # Neo4j FOLLOWS edges
        'spatial': 0.4,              # Spatial neighbors
        'gravity': 0.2,              # Gravity direction
    })


@dataclass
class DialecticConfig:
    """Dialectic engine configuration."""
    tension_weight: float = 0.5      # Weight for tension scoring
    antithesis_weight: float = 0.5   # Weight for antithesis pull
    coherence_threshold: float = 0.3 # Minimum coherence


@dataclass
class ChainConfig:
    """Chain reaction configuration."""
    decay: float = 0.85              # Resonance decay factor
    threshold: float = 0.7           # Lasing threshold
    history_length: int = 10         # Bond history for resonance


# ============================================================================
# GENRE PARAMETERS
# ============================================================================

@dataclass
class GenreParams:
    """Parameters for genre-specific generation."""
    name: str
    R_storm: float              # Radius for candidate explosion
    coh_threshold: float        # Minimum coherence to pass
    boundary_A_jump: float      # A jump magnitude at boundaries
    boundary_S_jump: float      # S jump magnitude at boundaries
    S_decay: float              # S decay factor at boundaries (1.0 = no decay)
    bonds_per_sentence: int     # Target bonds per sentence


GENRE_PRESETS = {
    'dramatic': GenreParams(
        name='dramatic',
        R_storm=0.6,
        coh_threshold=0.2,
        boundary_A_jump=0.5,
        boundary_S_jump=0.4,
        S_decay=1.0,
        bonds_per_sentence=4,
    ),
    'ironic': GenreParams(
        name='ironic',
        R_storm=0.5,
        coh_threshold=0.3,
        boundary_A_jump=0.4,
        boundary_S_jump=0.0,
        S_decay=0.5,
        bonds_per_sentence=4,
    ),
    'balanced': GenreParams(
        name='balanced',
        R_storm=0.4,
        coh_threshold=0.4,
        boundary_A_jump=0.15,
        boundary_S_jump=0.15,
        S_decay=1.0,
        bonds_per_sentence=4,
    ),
}


# ============================================================================
# HEALTH TARGET
# ============================================================================

@dataclass
class HealthTarget:
    """Therapeutic health target in semantic space."""
    A: float = 0.3      # Positive, affirming
    S: float = 0.2      # Meaningful
    tau: float = 2.0    # Grounded but not too concrete


HEALTH = HealthTarget()


# ============================================================================
# MAIN CONFIG CLASS
# ============================================================================

@dataclass
class Config:
    """Main configuration container."""
    # Database
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)

    # Semantic layer
    storm: StormConfig = field(default_factory=StormConfig)
    dialectic: DialecticConfig = field(default_factory=DialecticConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)

    # Health target
    health: HealthTarget = field(default_factory=HealthTarget)

    # Paths
    data_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "meaning_chain/data")

    # Constants (exposed for convenience)
    kT: float = KT
    Q_max: float = Q_MAX
    decay: float = DECAY
    dt: float = DT
    lambda_gravity: float = LAMBDA
    mu_gravity: float = MU

    def get_genre(self, name: str) -> GenreParams:
        """Get genre parameters by name."""
        return GENRE_PRESETS.get(name, GENRE_PRESETS['balanced'])


# ============================================================================
# SINGLETON
# ============================================================================

_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
