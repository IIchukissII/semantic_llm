#!/usr/bin/env python3
"""
PHASE 4: Semantic Core Architecture
====================================

16D semantic space as the core of language understanding.

Architecture:
    ┌─────────────────────────────────────────┐
    │              Text Input                 │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │         Semantic Projector              │
    │         768D → 16D                      │
    │                                         │
    │   ┌─────┐  ┌─────────┐  ┌───────────┐  │
    │   │  τ  │  │ i-space │  │  j-space  │  │
    │   │ (1) │  │  (11)   │  │    (5)    │  │
    │   └─────┘  └─────────┘  └───────────┘  │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │         Semantic Controller             │
    │                                         │
    │   • Measure current position            │
    │   • Compare with target                 │
    │   • Steer generation                    │
    └─────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Dimension names
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']
ALL_DIMS = J_DIMS + I_DIMS

# Verb dimensions (j-space + truth from i-space)
VERB_DIMS = ['beauty', 'life', 'sacred', 'good', 'love', 'truth']


@dataclass
class SemanticCoords:
    """16D semantic coordinates + tau."""
    tau: float          # Abstraction level [1, 6]
    j: np.ndarray       # 5D transcendental space
    i: np.ndarray       # 11D surface space

    @property
    def vector(self) -> np.ndarray:
        """Full 16D vector."""
        return np.concatenate([self.j, self.i])

    @property
    def j_magnitude(self) -> float:
        """Transcendental depth."""
        return float(np.linalg.norm(self.j))

    def to_dict(self) -> Dict:
        return {
            'tau': self.tau,
            'j': dict(zip(J_DIMS, self.j.tolist())),
            'i': dict(zip(I_DIMS, self.i.tolist())),
            'j_magnitude': self.j_magnitude
        }


@dataclass
class VerbVector:
    """6D verb transition vector (j-space + truth)."""
    vector: np.ndarray  # 6D: [beauty, life, sacred, good, love, truth]

    @property
    def j_component(self) -> np.ndarray:
        """5D j-space component."""
        return self.vector[:5]

    @property
    def truth_component(self) -> float:
        """Truth (i-space) component."""
        return float(self.vector[5])

    @property
    def magnitude(self) -> float:
        """Transition strength."""
        return float(np.linalg.norm(self.vector))

    def to_dict(self) -> Dict:
        return {
            'vector': dict(zip(VERB_DIMS, self.vector.tolist())),
            'j': dict(zip(J_DIMS, self.j_component.tolist())),
            'truth': self.truth_component,
            'magnitude': self.magnitude
        }


class VerbTransition:
    """
    Model verbs as transition operators in semantic space.

    Verbs transform noun states: verb(noun1) -> noun2
    The 6D verb vector indicates how the verb shifts meaning
    along transcendental dimensions + truth.
    """

    def __init__(self, verb: str, vector: np.ndarray):
        self.verb = verb
        self.vector = VerbVector(vector=vector)

    def apply(self, coords: SemanticCoords, strength: float = 1.0) -> SemanticCoords:
        """
        Apply verb transition to semantic coordinates.

        The verb modifies:
        - j-space (transcendentals): shifted by verb's j-component
        - i-space truth: shifted by verb's truth component
        """
        # Modify j-space
        new_j = coords.j + strength * self.vector.j_component
        new_j = np.clip(new_j, -1, 1)  # Keep in bounds

        # Modify truth in i-space
        new_i = coords.i.copy()
        truth_idx = I_DIMS.index('truth')
        new_i[truth_idx] += strength * self.vector.truth_component
        new_i = np.clip(new_i, -1, 1)

        return SemanticCoords(
            tau=coords.tau,  # Verbs don't change abstraction level
            j=new_j,
            i=new_i
        )

    def get_transition_vector(self) -> np.ndarray:
        """Get the full 16D transition vector (for training)."""
        # Expand 6D verb vector to 16D
        full_vec = np.zeros(16)
        full_vec[:5] = self.vector.j_component  # j-space
        full_vec[5] = self.vector.truth_component  # truth in i-space
        return full_vec


class SemanticProjector(nn.Module):
    """
    Project embeddings to 16D semantic space.

    Can be trained supervised (word -> coords) or
    self-supervised (reconstruction).
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()

        # Projection layers
        self.to_j = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

        self.to_i = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 11)
        )

        self.to_tau = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project embeddings to semantic space.

        Args:
            embeddings: [batch, embed_dim] or [batch, seq, embed_dim]

        Returns:
            tau: [batch] or [batch, seq] - abstraction level [1, 6]
            i: [batch, 11] or [batch, seq, 11] - surface space [-1, 1]
            j: [batch, 5] or [batch, seq, 5] - transcendental space [-1, 1]
        """
        j = torch.tanh(self.to_j(embeddings))
        i = torch.tanh(self.to_i(embeddings))
        tau = 1 + 5 * torch.sigmoid(self.to_tau(embeddings).squeeze(-1))  # [1, 6]

        return tau, i, j

    def get_coords(self, embeddings: torch.Tensor) -> SemanticCoords:
        """Get coords as dataclass (for single embedding)."""
        tau, i, j = self.forward(embeddings)
        return SemanticCoords(
            tau=tau.item(),
            j=j.detach().numpy().flatten(),
            i=i.detach().numpy().flatten()
        )


class SemanticAutoencoder(nn.Module):
    """
    Compress text through 16D semantic bottleneck with verb transitions.

    Structure matches the theory:
        - j-space (5D): Transcendentals (Beauty, Life, Sacred, Good, Love)
        - i-space (11D): Surface axes (Truth, Freedom, Meaning, ...)
        - τ: Abstraction level [1, 6]
        - v-space (6D): Verb transition (j-space + truth)

    Total: 16D semantic vector + τ + 6D verb transition
    """

    def __init__(self, vocab_size: int = 50000, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()

        # Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Bottleneck: Separate j-space, i-space, and verb transition
        # j-space: 5D transcendentals (Beauty, Life, Sacred, Good, Love)
        self.to_j = nn.Linear(hidden_dim * 2, 5)
        # i-space: 11D surface (Truth, Freedom, Meaning, ...)
        self.to_i = nn.Linear(hidden_dim * 2, 11)
        # τ: abstraction level
        self.to_tau = nn.Linear(hidden_dim * 2, 1)
        # v-space: 6D verb transition (j-space + truth)
        self.to_verb = nn.Linear(hidden_dim * 2, 6)

        # Decoder: 16D + τ + 6D verb = 23D -> hidden
        self.expand = nn.Linear(23, hidden_dim)  # 5 + 11 + 1 + 6 = 23
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode tokens to structured semantic space with verb transition.

        Returns:
            j: [B, 5] - transcendental space (Beauty, Life, Sacred, Good, Love)
            i: [B, 11] - surface space (Truth, Freedom, ...)
            tau: [B, 1] - abstraction level [1, 6]
            semantic: [B, 16] - combined j+i vector
            verb: [B, 6] - verb transition vector
        """
        emb = self.embedding(tokens)  # [B, seq, embed]
        _, (h, _) = self.encoder(emb)  # [2, B, hidden]
        h = torch.cat([h[0], h[1]], dim=-1)  # [B, 2*hidden]

        # Separate spaces (matching the theory)
        j = torch.tanh(self.to_j(h))  # [B, 5] - j-space
        i = torch.tanh(self.to_i(h))  # [B, 11] - i-space
        tau = 1 + 5 * torch.sigmoid(self.to_tau(h))  # [B, 1] - τ in [1, 6]
        verb = torch.tanh(self.to_verb(h))  # [B, 6] - verb transition

        # Combined 16D vector (j first, then i - matching vocabulary structure)
        semantic = torch.cat([j, i], dim=-1)  # [B, 16]

        return j, i, tau, semantic, verb

    def decode(self, j: torch.Tensor, i: torch.Tensor, tau: torch.Tensor,
               verb: torch.Tensor, length: int) -> torch.Tensor:
        """Decode structured semantic space with verb transition to token logits."""
        # Combine all components including verb
        combined = torch.cat([j, i, tau, verb], dim=-1)  # [B, 23]
        expanded = self.expand(combined)  # [B, hidden]

        # Repeat for sequence
        expanded = expanded.unsqueeze(1).repeat(1, length, 1)  # [B, seq, hidden]
        decoded, _ = self.decoder(expanded)  # [B, seq, hidden]
        logits = self.output(decoded)  # [B, seq, vocab]

        return logits

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full encode-decode pass.

        Returns:
            logits: [B, seq, vocab] - token predictions
            j: [B, 5] - j-space (transcendentals)
            i: [B, 11] - i-space (surface)
            tau: [B, 1] - abstraction level
            semantic: [B, 16] - combined vector
            verb: [B, 6] - verb transition
        """
        j, i, tau, semantic, verb = self.encode(tokens)
        logits = self.decode(j, i, tau, verb, tokens.size(1))
        return logits, j, i, tau, semantic, verb

    def get_j_magnitude(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get transcendental depth (||j||)."""
        j, _, _, _, _ = self.encode(tokens)
        return torch.norm(j, dim=-1)

    def get_verb_magnitude(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get verb transition strength (||verb||)."""
        _, _, _, _, verb = self.encode(tokens)
        return torch.norm(verb, dim=-1)

    @staticmethod
    def orthogonality_loss(j: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality loss to enforce j ⊥ i.

        Theory: j-space (transcendentals) and i-space (surface) should be
        independent/orthogonal. This loss penalizes correlation between them.

        Loss = mean(|correlation(j, i)|)
        """
        # Normalize to unit vectors
        j_norm = F.normalize(j, dim=-1)  # [B, 5]
        i_norm = F.normalize(i, dim=-1)  # [B, 11]

        # Compute cross-correlation matrix: [B, 5, 11]
        # Each element is correlation between j_dim and i_dim
        corr = torch.bmm(j_norm.unsqueeze(-1), i_norm.unsqueeze(1))  # [B, 5, 11]

        # Penalize any correlation (should all be ~0)
        return torch.mean(torch.abs(corr))

    @staticmethod
    def j_magnitude_loss(j: torch.Tensor, target_mag: float = 0.05) -> torch.Tensor:
        """
        Encourage j-space to have meaningful magnitude.

        Without this, the model might collapse j to zero and put everything in i.
        Target magnitude ~0.05 matches our vocabulary statistics.
        """
        j_mag = torch.norm(j, dim=-1)  # [B]
        return F.mse_loss(j_mag, torch.full_like(j_mag, target_mag))

    @staticmethod
    def separation_loss(j: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        Combined loss to enforce j/i separation:
        1. Orthogonality: j ⊥ i
        2. J-magnitude: ||j|| should be meaningful
        3. Variance: both j and i should have variance (not collapsed)
        """
        # Orthogonality
        ortho = SemanticAutoencoder.orthogonality_loss(j, i)

        # J should have meaningful magnitude
        j_mag = SemanticAutoencoder.j_magnitude_loss(j, target_mag=0.05)

        # Both should have variance (prevent collapse)
        j_var = torch.mean(torch.var(j, dim=0))  # Variance across batch for each dim
        i_var = torch.mean(torch.var(i, dim=0))
        var_loss = 1.0 / (j_var + 0.01) + 1.0 / (i_var + 0.01)  # Penalize low variance

        return ortho + 0.1 * j_mag + 0.01 * var_loss


class SemanticVocabulary:
    """
    Pre-computed 16D coordinates for vocabulary.

    Built from our corpus analysis.
    Includes both nouns (16D) and verbs (6D transition operators).
    """

    def __init__(self, data_path: Path = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data/semantic_vectors"

        # Load pre-computed noun vectors (16D)
        with open(data_path / "noun_vectors_16d.json") as f:
            noun_data = json.load(f)
        with open(data_path / "tau_levels.json") as f:
            tau_data = json.load(f)

        self.dimensions = noun_data['dimensions']
        self.vectors = noun_data['vectors']
        self.noun_tau = tau_data['nouns']
        self.verb_tau = tau_data.get('verbs', {})

        # Load verb vectors (6D transition operators)
        verb_path = data_path / "verb_vectors_6d.json"
        if verb_path.exists():
            with open(verb_path) as f:
                verb_data = json.load(f)
            self.verb_dimensions = verb_data['dimensions']
            self.verb_vectors = verb_data['vectors']
            self._verbs = list(self.verb_vectors.keys())
            self._verb_matrix = np.array([self.verb_vectors[v] for v in self._verbs])
        else:
            self.verb_dimensions = VERB_DIMS
            self.verb_vectors = {}
            self._verbs = []
            self._verb_matrix = np.array([])

        # Build numpy arrays for fast lookup
        self._words = list(self.vectors.keys())
        self._matrix = np.array([self.vectors[w] for w in self._words])
        self._tau = np.array([self.noun_tau.get(w, 4.0) for w in self._words])

    def get_coords(self, word: str) -> Optional[SemanticCoords]:
        """Get 16D coords for a noun."""
        word = word.lower()
        if word not in self.vectors:
            return None

        vec = np.array(self.vectors[word])
        tau = self.noun_tau.get(word, self.verb_tau.get(word, 4.0))

        return SemanticCoords(
            tau=tau,
            j=vec[:5],
            i=vec[5:]
        )

    def get_verb(self, verb: str) -> Optional[VerbTransition]:
        """Get verb as transition operator."""
        verb = verb.lower()
        if verb not in self.verb_vectors:
            return None

        vec = np.array(self.verb_vectors[verb])
        return VerbTransition(verb=verb, vector=vec)

    def get_verb_vector(self, verb: str) -> Optional[VerbVector]:
        """Get raw 6D verb vector."""
        verb = verb.lower()
        if verb not in self.verb_vectors:
            return None

        vec = np.array(self.verb_vectors[verb])
        return VerbVector(vector=vec)

    def is_verb(self, word: str) -> bool:
        """Check if word is a known verb."""
        return word.lower() in self.verb_vectors

    def is_noun(self, word: str) -> bool:
        """Check if word is a known noun."""
        return word.lower() in self.vectors

    def get_text_coords(self, text: str, include_verbs: bool = True) -> SemanticCoords:
        """
        Get aggregated coords for text.

        If include_verbs=True, verbs contribute their 6D transition vector
        (expanded to 16D with zeros for non-truth i-space dimensions).
        """
        import re
        words = re.findall(r'\b[a-z]+\b', text.lower())

        vectors = []
        taus = []

        for word in words:
            # Check nouns first (16D)
            if word in self.vectors:
                vectors.append(self.vectors[word])
            # Then check verbs (6D -> expanded to 16D)
            elif include_verbs and word in self.verb_vectors:
                verb_6d = np.array(self.verb_vectors[word])
                # Expand 6D to 16D: [j0-j4, truth, 0, 0, ..., 0]
                full_vec = np.zeros(16)
                full_vec[:5] = verb_6d[:5]  # j-space
                full_vec[5] = verb_6d[5]  # truth (first i-space dim)
                vectors.append(full_vec)

            # Get tau
            if word in self.noun_tau:
                taus.append(self.noun_tau[word])
            elif word in self.verb_tau:
                taus.append(self.verb_tau[word])

        if not vectors:
            return SemanticCoords(tau=4.0, j=np.zeros(5), i=np.zeros(11))

        mean_vec = np.mean(vectors, axis=0)
        mean_tau = np.mean(taus) if taus else 4.0

        return SemanticCoords(
            tau=mean_tau,
            j=mean_vec[:5],
            i=mean_vec[5:]
        )

    def analyze_text(self, text: str) -> Dict:
        """
        Full semantic analysis of text with noun/verb breakdown.

        Returns dict with:
        - coords: aggregated 16D coordinates
        - nouns: list of (noun, coords) tuples
        - verbs: list of (verb, transition) tuples
        - transitions: cumulative verb effect
        """
        import re
        words = re.findall(r'\b[a-z]+\b', text.lower())

        nouns = []
        verbs = []

        for word in words:
            if word in self.vectors:
                coords = self.get_coords(word)
                if coords:
                    nouns.append((word, coords))
            elif word in self.verb_vectors:
                trans = self.get_verb(word)
                if trans:
                    verbs.append((word, trans))

        # Compute aggregated coordinates
        coords = self.get_text_coords(text)

        # Compute cumulative verb transition
        if verbs:
            trans_vectors = [t.get_transition_vector() for _, t in verbs]
            cumulative = np.mean(trans_vectors, axis=0)
            transition = SemanticCoords(
                tau=4.0,
                j=cumulative[:5],
                i=cumulative[5:]
            )
        else:
            transition = SemanticCoords(tau=4.0, j=np.zeros(5), i=np.zeros(11))

        return {
            'coords': coords,
            'nouns': nouns,
            'verbs': verbs,
            'transition': transition,
            'noun_count': len(nouns),
            'verb_count': len(verbs)
        }

    def find_nearest(self, coords: SemanticCoords, n: int = 10,
                     max_tau: float = None) -> List[Tuple[str, float]]:
        """Find words nearest to coords."""
        target = coords.vector

        # Compute distances
        distances = np.linalg.norm(self._matrix - target, axis=1)

        # Apply tau filter
        if max_tau is not None:
            mask = self._tau <= max_tau
            distances[~mask] = np.inf

        # Get top n
        indices = np.argsort(distances)[:n]

        return [(self._words[i], float(distances[i])) for i in indices]


class SemanticController:
    """
    Control text generation using 16D semantic space.

    Can work with any LLM as post-processing or steering.
    """

    def __init__(self):
        self.vocab = SemanticVocabulary()
        self.target: Optional[SemanticCoords] = None
        self.history: List[SemanticCoords] = []

    def set_target(self,
                   j: Dict[str, float] = None,
                   i: Dict[str, float] = None,
                   tau: float = None):
        """Set target semantic coordinates."""
        j_vec = np.zeros(5)
        i_vec = np.zeros(11)

        if j:
            for dim, val in j.items():
                if dim in J_DIMS:
                    j_vec[J_DIMS.index(dim)] = val

        if i:
            for dim, val in i.items():
                if dim in I_DIMS:
                    i_vec[I_DIMS.index(dim)] = val

        self.target = SemanticCoords(
            tau=tau or 3.5,
            j=j_vec,
            i=i_vec
        )

    def measure(self, text: str) -> SemanticCoords:
        """Measure semantic coords of text."""
        coords = self.vocab.get_text_coords(text)
        self.history.append(coords)
        return coords

    def get_distance(self, text: str) -> float:
        """Get distance from target."""
        if self.target is None:
            return 0.0

        coords = self.measure(text)
        return float(np.linalg.norm(coords.vector - self.target.vector))

    def get_steering_words(self, n: int = 20, max_tau: float = 4) -> List[str]:
        """Get words that steer toward target."""
        if self.target is None:
            return []

        # Current position (average of history or zero)
        if self.history:
            current = np.mean([c.vector for c in self.history], axis=0)
        else:
            current = np.zeros(16)

        # Direction to target
        direction = self.target.vector - current

        # Score all words
        scores = np.dot(self.vocab._matrix, direction)

        # Apply tau filter
        mask = self.vocab._tau <= max_tau
        scores[~mask] = -np.inf

        # Get top words
        indices = np.argsort(scores)[-n:][::-1]
        return [self.vocab._words[i] for i in indices]

    def should_regenerate(self, text: str, max_distance: float = 1.0) -> bool:
        """Check if text is too far from target."""
        return self.get_distance(text) > max_distance

    def filter_response(self, response: str,
                        min_j_magnitude: float = 0.02,
                        fallback: str = "Let me think more deeply...") -> str:
        """Filter responses by j-magnitude (transcendental depth)."""
        coords = self.measure(response)

        if coords.j_magnitude < min_j_magnitude:
            return fallback

        return response


class SemanticLoss(nn.Module):
    """
    Loss function for semantic alignment.

    Can be added to any LLM training.
    """

    def __init__(self, projector: SemanticProjector,
                 j_weight: float = 1.0,
                 tau_weight: float = 0.5):
        super().__init__()
        self.projector = projector
        self.j_weight = j_weight
        self.tau_weight = tau_weight

    def forward(self, embeddings: torch.Tensor,
                target_coords: torch.Tensor,
                target_tau: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic alignment loss.

        Args:
            embeddings: [batch, embed_dim]
            target_coords: [batch, 16] target in 16D space
            target_tau: [batch] target abstraction level
        """
        tau, i, j = self.projector(embeddings)
        current = torch.cat([j, i], dim=-1)  # [batch, 16]

        # Coordinate loss
        coord_loss = F.mse_loss(current, target_coords)

        # Tau loss
        tau_loss = F.mse_loss(tau, target_tau)

        return coord_loss + self.tau_weight * tau_loss


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC CORE ARCHITECTURE TEST")
    print("=" * 60)

    # Test vocabulary
    print("\n[1] Testing SemanticVocabulary...")
    vocab = SemanticVocabulary()

    test_words = ['love', 'heart', 'god', 'war', 'beauty', 'death']
    for word in test_words:
        coords = vocab.get_coords(word)
        if coords:
            print(f"  {word:<10}: tau={coords.tau:.1f}, j_mag={coords.j_magnitude:.3f}")

    # Test text coords
    print("\n[2] Testing text coordinates...")
    texts = [
        "Love and beauty fill the heart with joy",
        "War brings death and destruction",
        "God watches over heaven with sacred light",
    ]

    for text in texts:
        coords = vocab.get_text_coords(text)
        top_j = J_DIMS[np.argmax(np.abs(coords.j))]
        print(f"  '{text[:40]}...'")
        print(f"    tau={coords.tau:.2f}, j_mag={coords.j_magnitude:.3f}, top_j={top_j}")

    # Test controller
    print("\n[3] Testing SemanticController...")
    controller = SemanticController()

    controller.set_target(j={'love': 0.5, 'beauty': 0.3}, tau=2.5)

    print(f"  Target: j={dict(zip(J_DIMS, controller.target.j))}")
    print(f"  Steering words: {controller.get_steering_words(10)}")

    test_response = "The sunset painted colors across the sky"
    coords = controller.measure(test_response)
    dist = controller.get_distance(test_response)
    print(f"\n  Response: '{test_response}'")
    print(f"  Coords: tau={coords.tau:.2f}, j={dict(zip(J_DIMS, coords.j.round(3)))}")
    print(f"  Distance from target: {dist:.3f}")

    # Test projector (random init)
    print("\n[4] Testing SemanticProjector...")
    projector = SemanticProjector(embed_dim=768)

    dummy_emb = torch.randn(2, 768)
    tau, i, j = projector(dummy_emb)

    print(f"  Input: [2, 768]")
    print(f"  tau: {tau.shape} = {tau.detach().numpy().round(2)}")
    print(f"  j: {j.shape}")
    print(f"  i: {i.shape}")

    # Test autoencoder (structure only)
    print("\n[5] Testing SemanticAutoencoder structure...")
    autoencoder = SemanticAutoencoder(vocab_size=10000)

    dummy_tokens = torch.randint(0, 10000, (2, 20))
    logits, j_out, i_out, tau_out, semantic, verb = autoencoder(dummy_tokens)

    print(f"  Input tokens: {dummy_tokens.shape}")
    print(f"  j-space (5D): {j_out.shape}")
    print(f"  i-space (11D): {i_out.shape}")
    print(f"  tau: {tau_out.shape}")
    print(f"  semantic (16D): {semantic.shape}")
    print(f"  verb transition (6D): {verb.shape}")
    print(f"  Output logits: {logits.shape}")

    total_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test verb functionality
    print("\n[6] Testing verb transitions...")
    test_verbs = ['love', 'hate', 'create', 'destroy', 'give', 'take']
    for verb_word in test_verbs:
        verb_trans = vocab.get_verb(verb_word)
        if verb_trans:
            print(f"  {verb_word:<10}: mag={verb_trans.vector.magnitude:.4f}, truth={verb_trans.vector.truth_component:.4f}")
        else:
            print(f"  {verb_word:<10}: not found in verb vocabulary")

    # Test text analysis with verbs
    print("\n[7] Testing text analysis with verbs...")
    test_texts = [
        "Love conquers all hatred",
        "The creator destroys his creation",
        "She gives and he takes",
    ]
    for text in test_texts:
        analysis = vocab.analyze_text(text)
        print(f"  '{text}'")
        print(f"    nouns: {analysis['noun_count']}, verbs: {analysis['verb_count']}")
        print(f"    j_mag: {analysis['coords'].j_magnitude:.4f}")
        print(f"    transition j_mag: {analysis['transition'].j_magnitude:.4f}")

    print("\n" + "=" * 60)
    print("SEMANTIC CORE READY")
    print("=" * 60)
    print("""
Architecture components:
1. SemanticVocabulary - pre-computed 16D coords for nouns + 6D verbs
2. SemanticProjector - learn 768D -> 16D projection
3. SemanticAutoencoder - compress text through 16D + 6D verb bottleneck
4. SemanticController - steer generation with 16D target
5. SemanticLoss - training loss for semantic alignment
6. VerbTransition - model verbs as semantic state transitions

Key insight:
  - Nouns are points in 16D semantic space
  - Verbs are transition operators (6D: j-space + truth)
  - Together they model meaning as state + transition
  This enables:
    - Explicit understanding (interpretable coords)
    - Directed generation (semantic compass)
    - Dynamic semantics (verb transitions)
""")
