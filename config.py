"""
Configuration for Quantum Semantic Architecture
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
CORE_DIR = PROJECT_ROOT / "core"
VALIDATION_DIR = PROJECT_ROOT / "validation"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Database
DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}

DATABASE_URL = "postgresql://bonds:bonds_secret@localhost:5432/bonds"

# Semantic Space Dimensions
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']  # 5D transcendentals
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']  # 11D surface
VERB_DIMS = ['beauty', 'life', 'sacred', 'good', 'love', 'truth']  # 6D verb operators
ALL_DIMS = J_DIMS + I_DIMS  # 16D total

# Theory Constants
TAU_RANGE = (1, 6)  # Abstraction level range
ONE_BIT_CONSTANT = 1.08  # H_adj - H_verb ≈ 1.08 bits
EULER_CONSTANT = 0.3679  # 1/e ≈ 0.3679

# Model Configuration
MODEL_CONFIG = {
    "vocab_size": 30000,
    "embed_dim": 128,
    "hidden_dim": 128,
    "n_basis_adjectives": 100,
    "ortho_weight": 4.7,
    "learning_rate": 0.009
}

# Training Configuration
TRAINING_CONFIG = {
    "n_samples": 15000,
    "seq_len": 64,
    "epochs": 50,
    "batch_size": 64,
    "semantic_weight": 0.1,
    "separation_weight": 0.05
}
