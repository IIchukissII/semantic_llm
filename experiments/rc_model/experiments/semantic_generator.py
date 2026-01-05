"""Semantic Generator: Genre-guided text skeleton generation.

Uses (τ, A, S) boundary patterns to generate semantic skeletons
that can be used as LLM prompts for style-controlled text generation.

WORKFLOW:
  1. Select genre pattern (dramatic, ironic, balanced)
  2. Generate semantic skeleton (sequence of bonds)
  3. Use skeleton as LLM prompt
  4. LLM generates grammatical text in the desired style

PATTERNS:
  DRAMATIC: Big A+S jumps → emotional + philosophical shifts
  IRONIC:   Big A, decaying S → emotion in mundane setting
  BALANCED: Small jumps → controlled, measured discourse
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import psycopg2

from ..core.coord_loader import get_loader, CoordLoader
from ..core.semantic_rc_v2 import KT


@dataclass
class GenrePattern:
    """Pattern for genre-specific generation."""
    name: str
    description: str

    # Boundary jump magnitudes
    A_jump: float      # Affirmation jump at boundaries
    S_jump: float      # Sacred jump at boundaries
    S_decay: float     # S decay factor (1.0 = no decay, 0.5 = halve)

    # Within-sentence dynamics
    smoothing: float   # How much to smooth (0.6 = fast, 0.8 = slow)


# Pre-defined genre patterns
PATTERNS = {
    'dramatic': GenrePattern(
        name='dramatic',
        description='Gothic/Dostoevsky: high tension, emotional + philosophical',
        A_jump=0.5,
        S_jump=0.4,
        S_decay=1.0,
        smoothing=0.6,
    ),
    'ironic': GenrePattern(
        name='ironic',
        description='Kafka/Poe: tension + emotion, but mundane concepts',
        A_jump=0.4,
        S_jump=0.0,
        S_decay=0.5,  # Decay S toward mundane
        smoothing=0.6,
    ),
    'balanced': GenrePattern(
        name='balanced',
        description='Austen/Plato: controlled, measured discourse',
        A_jump=0.12,
        S_jump=0.12,
        S_decay=1.0,
        smoothing=0.7,
    ),
}


class SemanticGenerator:
    """Generate semantic skeletons for genre-controlled text."""

    def __init__(
        self,
        loader: Optional[CoordLoader] = None,
        db_config: Optional[dict] = None,
        min_freq: int = 5,
    ):
        """Initialize generator.

        Args:
            loader: Coordinate loader
            db_config: PostgreSQL config (host, database, user, password)
            min_freq: Minimum bond frequency to include
        """
        self.loader = loader or get_loader()
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'bonds',
            'user': 'bonds',
            'password': 'bonds_secret',
        }
        self.min_freq = min_freq

        # Vocabulary
        self._bonds = []
        self._coords = None
        self._freqs = None
        self._loaded = False

    def load_vocabulary(self) -> int:
        """Load vocabulary from database.

        Returns:
            Number of bonds loaded
        """
        if self._loaded:
            return len(self._bonds)

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        # Get frequent bonds
        cur.execute('''
            SELECT n.noun_text, a.adj_text, COUNT(*) as cnt
            FROM text_activations ta
            JOIN nouns n ON ta.noun_id = n.noun_id
            JOIN adjectives a ON ta.adj_id = a.adj_id
            GROUP BY n.noun_text, a.adj_text
            HAVING COUNT(*) >= %s
            ORDER BY cnt DESC
        ''', (self.min_freq,))

        bonds = []
        coords = []
        freqs = []

        for noun, adj, cnt in cur.fetchall():
            noun_l, adj_l = noun.lower(), adj.lower()
            if self.loader.has(noun_l):
                nc = self.loader.get(noun_l)
                bonds.append(f'{adj} {noun}')
                coords.append([nc.A, nc.S, nc.tau])
                freqs.append(cnt)

        conn.close()

        self._bonds = bonds
        self._coords = np.array(coords)
        self._freqs = np.array(freqs)
        self._loaded = True

        return len(bonds)

    def generate_skeleton(
        self,
        pattern: str = 'balanced',
        n_bonds: int = 12,
        bonds_per_sentence: int = 4,
        seed: Optional[int] = None,
        temperature: float = 0.6,
    ) -> list[list[str]]:
        """Generate semantic skeleton for a genre.

        Args:
            pattern: Genre pattern name
            n_bonds: Total number of bonds
            bonds_per_sentence: Bonds per "sentence"
            seed: Random seed
            temperature: Sampling temperature

        Returns:
            List of sentences, each a list of bonds
        """
        if not self._loaded:
            self.load_vocabulary()

        if seed is not None:
            np.random.seed(seed)

        pat = PATTERNS.get(pattern, PATTERNS['balanced'])

        # Initial state
        Q_A, Q_S, Q_tau = 0.0, 0.0, 4.5

        sentences = []
        current_sentence = []
        bonds_generated = 0

        while bonds_generated < n_bonds:
            # Check for sentence boundary
            if len(current_sentence) >= bonds_per_sentence:
                sentences.append(current_sentence)
                current_sentence = []

                # Apply boundary pattern
                Q_A += np.random.randn() * pat.A_jump
                Q_S = Q_S * pat.S_decay + np.random.randn() * pat.S_jump

            # Score bonds
            delta_A = self._coords[:, 0] - Q_A
            delta_S = self._coords[:, 1] - Q_S
            delta_tau = np.abs(self._coords[:, 2] - Q_tau)

            log_p = -delta_tau / KT - delta_A**2 / 0.3**2 - delta_S**2 / 0.3**2
            log_p += np.log(self._freqs + 1) * 0.2  # Frequency boost
            log_p = log_p / temperature

            # Sample from top-k
            top_k = 50
            top_idx = np.argsort(log_p)[-top_k:]
            weights = np.exp(log_p[top_idx] - log_p[top_idx].max())
            weights /= weights.sum()

            chosen = np.random.choice(top_idx, p=weights)
            current_sentence.append(self._bonds[chosen])

            # Update state
            s = pat.smoothing
            Q_A = Q_A * s + self._coords[chosen, 0] * (1 - s)
            Q_S = Q_S * s + self._coords[chosen, 1] * (1 - s)
            Q_tau = Q_tau * s + self._coords[chosen, 2] * (1 - s)

            bonds_generated += 1

        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    def generate_prompt(
        self,
        pattern: str = 'balanced',
        n_bonds: int = 12,
        seed: Optional[int] = None,
    ) -> str:
        """Generate LLM prompt from semantic skeleton.

        Args:
            pattern: Genre pattern
            n_bonds: Number of bonds
            seed: Random seed

        Returns:
            Formatted prompt for LLM
        """
        sentences = self.generate_skeleton(pattern, n_bonds, seed=seed)
        all_bonds = [b for sent in sentences for b in sent]

        pat = PATTERNS.get(pattern, PATTERNS['balanced'])

        prompt = f"""Write a {pattern} paragraph in the style of {pat.description}.

Incorporate these semantic concepts naturally:
{', '.join(all_bonds)}

The text should feel {pattern}:"""

        if pattern == 'dramatic':
            prompt += "\n- High emotional intensity"
            prompt += "\n- Philosophical or sacred themes"
            prompt += "\n- Strong atmosphere and tension"
        elif pattern == 'ironic':
            prompt += "\n- Underlying tension or unease"
            prompt += "\n- Mundane, everyday setting"
            prompt += "\n- Horror or absurdity in the ordinary"
        elif pattern == 'balanced':
            prompt += "\n- Measured, controlled tone"
            prompt += "\n- Clear reasoning"
            prompt += "\n- Calm and analytical"

        return prompt


def quick_test():
    """Quick test of semantic generator."""
    gen = SemanticGenerator()
    n = gen.load_vocabulary()
    print(f"Loaded {n:,} bonds")
    print()

    for pattern in ['dramatic', 'ironic', 'balanced']:
        print(f"=== {pattern.upper()} ===")
        sentences = gen.generate_skeleton(pattern, n_bonds=12, seed=42)
        for i, sent in enumerate(sentences):
            prefix = "   " if i == 0 else " → "
            print(f"{prefix}{' | '.join(sent)}")
        print()

        print("LLM Prompt:")
        print(gen.generate_prompt(pattern, n_bonds=8, seed=42))
        print()
        print("-" * 60)


if __name__ == "__main__":
    quick_test()
