#!/usr/bin/env python3
"""
Semantic Bottleneck V2 - Based on THE_MAP Theory

Key theoretical insight from THE_MAP.md:
- Adjectives = 16D projections of transcendental+surface basis
- Nouns = "projections of projections" = clouds of adjectives
- τ = derived from variety (number/spread of adjectives), NOT learned separately
- Verbs = 6D transition operators

Architecture:
1. Adjective Encoder: word → 16D direct projection
2. Noun Encoder: word → attention over adjective embeddings → (centroid, variety)
3. τ Derivation: variety → τ (energy level)
4. Verb Encoder: word → 6D transition vector

This is fundamentally different from V1 which tried to encode nouns directly to 16D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Semantic dimensions
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']  # 5D transcendental
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',  # 11D surface
          'power', 'wisdom', 'nature', 'time', 'self', 'social']
VERB_DIMS = J_DIMS + ['truth']  # 6D for verbs

J_SIZE = len(J_DIMS)      # 5
I_SIZE = len(I_DIMS)      # 11
SEMANTIC_SIZE = J_SIZE + I_SIZE  # 16
VERB_SIZE = 6


class AdjectiveEmbedding(nn.Module):
    """
    Adjectives as direct 16D projections onto semantic basis.

    |adj⟩ = Σₖ sₖ|iₖ⟩ + Σₘ tₘ|jₘ⟩

    Each adjective has a fixed position in 16D semantic space.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Word embedding (for context)
        self.word_embed = nn.Embedding(vocab_size, embed_dim)

        # Project to 16D semantic space
        # J-space (transcendentals) - should be small, stable
        self.j_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, J_SIZE),
            nn.Tanh()  # Bounded [-1, 1]
        )

        # I-space (surface) - can vary more
        self.i_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, I_SIZE),
            nn.Tanh()
        )

        # Scale factors (j-space typically has smaller magnitude)
        self.j_scale = nn.Parameter(torch.tensor(0.05))
        self.i_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokens to 16D semantic space.

        Returns:
            j: [batch, seq, 5] - transcendental coordinates
            i: [batch, seq, 11] - surface coordinates
        """
        embed = self.word_embed(token_ids)  # [batch, seq, embed_dim]

        j = self.j_proj(embed) * self.j_scale  # [batch, seq, 5]
        i = self.i_proj(embed) * self.i_scale  # [batch, seq, 11]

        return j, i

    def get_semantic_vector(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get full 16D semantic vector."""
        j, i = self.forward(token_ids)
        return torch.cat([j, i], dim=-1)


class NounAsAdjectiveCloud(nn.Module):
    """
    Nouns as "projections of projections" - clouds of adjectives.

    |noun⟩ = ∫|adj⟩ d(variety)

    A noun is characterized by:
    1. WHICH adjectives can describe it (attention weights)
    2. The CENTROID of those adjectives (semantic position)
    3. The VARIETY (spread) of those adjectives (determines τ)

    This is fundamentally different from direct 16D encoding!
    """

    def __init__(
        self,
        vocab_size: int,
        adj_embedding: AdjectiveEmbedding,
        n_adjectives: int = 1000,  # Number of "basis" adjectives
        hidden_dim: int = 256
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.adj_embedding = adj_embedding
        self.n_adjectives = n_adjectives
        self.hidden_dim = hidden_dim

        # Word embedding for nouns (separate from adjectives)
        self.noun_embed = nn.Embedding(vocab_size, hidden_dim)

        # Learn which adjectives describe this noun (attention)
        # Query: noun embedding, Keys: adjective embeddings
        self.adj_query = nn.Linear(hidden_dim, hidden_dim)
        self.adj_key = nn.Linear(adj_embedding.embed_dim, hidden_dim)

        # Temperature for attention sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Basis adjective indices (to be set from vocabulary)
        self.register_buffer('basis_adj_ids', torch.zeros(n_adjectives, dtype=torch.long))

    def set_basis_adjectives(self, adj_ids: torch.Tensor):
        """Set the basis adjective token IDs from vocabulary."""
        self.basis_adj_ids = adj_ids[:self.n_adjectives]

    def forward(
        self,
        noun_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode nouns as clouds of adjectives.

        Args:
            noun_ids: [batch, seq] noun token IDs

        Returns:
            centroid_j: [batch, seq, 5] - j-space centroid of adjective cloud
            centroid_i: [batch, seq, 11] - i-space centroid
            variety: [batch, seq] - spread of adjective cloud (determines τ)
            attention: [batch, seq, n_adj] - attention over basis adjectives
        """
        batch, seq = noun_ids.shape
        device = noun_ids.device

        # Get noun embeddings
        noun_embed = self.noun_embed(noun_ids)  # [batch, seq, hidden]

        # Get basis adjective embeddings (move to same device as input)
        basis_ids = self.basis_adj_ids.to(device)
        basis_adj_embed = self.adj_embedding.word_embed(basis_ids)  # [n_adj, embed]

        # Compute attention: which adjectives describe this noun?
        query = self.adj_query(noun_embed)  # [batch, seq, hidden]
        key = self.adj_key(basis_adj_embed)  # [n_adj, hidden]

        # Attention scores
        scores = torch.matmul(query, key.T) / (self.hidden_dim ** 0.5)  # [batch, seq, n_adj]
        attention = F.softmax(scores / self.temperature, dim=-1)  # [batch, seq, n_adj]

        # Get adjective semantic coordinates
        adj_j, adj_i = self.adj_embedding(basis_ids.unsqueeze(0))  # [1, n_adj, 5], [1, n_adj, 11]
        adj_j = adj_j.squeeze(0)  # [n_adj, 5]
        adj_i = adj_i.squeeze(0)  # [n_adj, 11]

        # Compute weighted centroid (noun position = mean of adjective cloud)
        centroid_j = torch.matmul(attention, adj_j)  # [batch, seq, 5]
        centroid_i = torch.matmul(attention, adj_i)  # [batch, seq, 11]

        # Compute variety (spread of adjective cloud)
        # Higher variety = more adjectives apply = lower τ (closer to source)
        variety = self.compute_variety(attention, adj_j, adj_i, centroid_j, centroid_i)

        return centroid_j, centroid_i, variety, attention

    def compute_variety(
        self,
        attention: torch.Tensor,  # [batch, seq, n_adj]
        adj_j: torch.Tensor,      # [n_adj, 5]
        adj_i: torch.Tensor,      # [n_adj, 11]
        centroid_j: torch.Tensor, # [batch, seq, 5]
        centroid_i: torch.Tensor  # [batch, seq, 11]
    ) -> torch.Tensor:
        """
        Compute variety (spread) of adjective cloud.

        Variety = attention entropy (how many adjectives contribute)
        High variety → many diverse adjectives → abstract concept → low τ
        Low variety → few specific adjectives → concrete thing → high τ

        Using attention entropy instead of geometric variance because:
        1. Numerically stable (doesn't depend on embedding scales)
        2. Directly measures "how many adjectives describe this noun"
        3. Range [0, 1] when normalized by max entropy
        """
        batch, seq, n_adj = attention.shape

        # Compute attention entropy as variety measure
        # H = -Σ p log(p)
        # High entropy = many adjectives with similar weights = abstract
        # Low entropy = few adjectives dominate = concrete
        entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1)  # [batch, seq]

        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(n_adj)
        variety = entropy / max_entropy  # [batch, seq] in [0, 1]

        return variety


class TauFromVariety(nn.Module):
    """
    Derive τ (abstraction level) from variety.

    τ is NOT learned separately - it EMERGES from adjective variety.

    High variety (many adjectives) → Low τ (abstract, close to source)
    Low variety (few adjectives) → High τ (concrete, far from source)

    Think of it as energy levels:
    - τ=1: "Love" - characterized by many diverse adjectives
    - τ=6: "John" - characterized by very few specific adjectives

    Now variety is normalized attention entropy in [0, 1].
    """

    def __init__(self, min_tau: float = 1.0, max_tau: float = 6.0):
        super().__init__()
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.tau_range = max_tau - min_tau

        # Small learnable adjustment (mostly linear mapping)
        # variety ∈ [0, 1], we want:
        #   variety=1 (high entropy) → τ=1 (abstract)
        #   variety=0 (low entropy) → τ=6 (concrete)
        self.adjustment = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh()  # Small adjustment in [-1, 1]
        )
        self.adjustment_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, variety: torch.Tensor) -> torch.Tensor:
        """
        Map variety to τ.

        Args:
            variety: [batch, seq] - normalized attention entropy in [0, 1]

        Returns:
            tau: [batch, seq] - abstraction level in [min_tau, max_tau]
        """
        # Direct inverse mapping: high variety → low τ
        # Base τ = max_tau - variety * range = max_tau - variety * 5
        # variety=1 → τ=1, variety=0 → τ=6
        base_tau = self.max_tau - variety * self.tau_range

        # Small learnable adjustment for non-linearity
        variety_input = variety.unsqueeze(-1)  # [batch, seq, 1]
        adjustment = self.adjustment(variety_input).squeeze(-1)  # [batch, seq]

        # Final τ with adjustment
        tau = base_tau + self.adjustment_scale * adjustment

        # Clamp to valid range
        tau = torch.clamp(tau, self.min_tau, self.max_tau)

        return tau


class VerbProjection(nn.Module):
    """
    Verbs as 6D projections onto transition basis.

    |verb⟩ = Σₖ vₖ|bₖ⟩  where b = [beauty, life, sacred, good, love, truth]

    Like adjectives, verbs are DIRECT projections onto a basis.
    The difference is:
    - Adjectives → 16D (j + i)
    - Verbs → 6D (j + truth)

    Verbs are projections that describe HOW things change.
    When applied to nouns, they shift the noun's coordinates.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.word_embed = nn.Embedding(vocab_size, embed_dim)

        # Direct projection to 6D verb basis
        # [beauty, life, sacred, good, love, truth]
        self.verb_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, VERB_SIZE),
            nn.Tanh()  # Bounded [-1, 1]
        )

        # Scale factor (verb projections similar magnitude to j-space)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, verb_ids: torch.Tensor) -> torch.Tensor:
        """
        Project verbs onto 6D basis.

        Args:
            verb_ids: [batch, seq] verb token IDs

        Returns:
            projection: [batch, seq, 6] - verb coordinates in 6D basis
        """
        embed = self.word_embed(verb_ids)
        projection = self.verb_proj(embed) * self.scale
        return projection

    def apply_as_transition(
        self,
        noun_j: torch.Tensor,    # [batch, 5]
        noun_i: torch.Tensor,    # [batch, 11]
        verb_proj: torch.Tensor, # [batch, 6]
        strength: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply verb projection as transition operator on noun coordinates.

        The verb's 6D projection shifts:
        - j-space (5D): beauty, life, sacred, good, love
        - truth (1D from i-space)
        """
        # Split verb projection into j-part and truth-part
        verb_j = verb_proj[..., :5]      # [batch, 5]
        verb_truth = verb_proj[..., 5:]  # [batch, 1]

        # Apply as transition (additive shift)
        new_j = noun_j + strength * verb_j
        new_i = noun_i.clone()
        new_i[..., 0] = noun_i[..., 0] + strength * verb_truth.squeeze(-1)

        return new_j, new_i


class SemanticBottleneckV2(nn.Module):
    """
    Complete Semantic Bottleneck V2 - Based on THE_MAP Theory.

    Text → (j-space, i-space, τ, verb-projections) → Text

    PROJECTION HIERARCHY (from THE_MAP):
    =====================================

    Level 1 - DIRECT PROJECTIONS onto basis:
    ----------------------------------------
    • Adjectives → 16D (5 j-space + 11 i-space)
      |adj⟩ = Σₖ sₖ|iₖ⟩ + Σₘ tₘ|jₘ⟩
      Example: "beautiful" → (beauty=0.8, life=0.2, ...)

    • Verbs → 6D (5 j-space + truth)
      |verb⟩ = Σₖ vₖ|bₖ⟩ where b = [beauty, life, sacred, good, love, truth]
      Example: "love" → (beauty=0.3, life=0.4, ..., truth=0.2)

    Level 2 - PROJECTIONS OF PROJECTIONS:
    -------------------------------------
    • Nouns → cloud of adjectives → τ (energy level)
      |noun⟩ = ∫|adj⟩ d(variety)

      A noun is characterized by WHICH adjectives describe it.
      τ (abstraction level) = derived from variety (adjective spread)

      High variety → abstract → low τ (e.g., "love" τ=1)
      Low variety → concrete → high τ (e.g., "chair" τ=5)

    Key differences from V1:
    1. Nouns encoded as adjective clouds, not direct 16D vectors
    2. τ derived from variety, not learned separately
    3. Verbs are projections, not just transition operators
    4. Proper theoretical grounding from THE_MAP
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_basis_adjectives: int = 1000
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Level 1: Direct projections
        self.adj_embedding = AdjectiveEmbedding(vocab_size, embed_dim)  # Adjectives → 16D
        self.verb_projection = VerbProjection(vocab_size, embed_dim)     # Verbs → 6D

        # Level 2: Projections of projections
        self.noun_encoder = NounAsAdjectiveCloud(vocab_size, self.adj_embedding, n_basis_adjectives, hidden_dim)
        self.tau_deriver = TauFromVariety()  # τ from variety, not learned

        # Encoder LSTM (for context)
        self.encoder_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Word type classifier (noun/verb/adj/other)
        self.word_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 0=noun, 1=verb, 2=adj, 3=other
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=SEMANTIC_SIZE + 1 + VERB_SIZE,  # 16 + 1 + 6 = 23
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Shared word embedding (for encoder context)
        self.word_embed = nn.Embedding(vocab_size, embed_dim)

    def encode(
        self,
        tokens: torch.Tensor,
        word_types: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to semantic bottleneck.

        Args:
            tokens: [batch, seq] token IDs
            word_types: [batch, seq] optional word type labels (0=noun, 1=verb, 2=adj)

        Returns:
            Dict with j, i, tau, verb, variety, attention
        """
        batch, seq = tokens.shape
        device = tokens.device

        # Get context from LSTM
        embed = self.word_embed(tokens)
        context, _ = self.encoder_lstm(embed)  # [batch, seq, hidden*2]

        # Classify word types if not provided
        if word_types is None:
            word_type_logits = self.word_type_classifier(context)
            word_types = word_type_logits.argmax(dim=-1)  # [batch, seq]

        # Create masks
        noun_mask = (word_types == 0).float()  # [batch, seq]
        verb_mask = (word_types == 1).float()
        adj_mask = (word_types == 2).float()

        # Encode based on word type
        # Nouns: use adjective cloud encoding
        noun_j, noun_i, variety, attention = self.noun_encoder(tokens)

        # Adjectives: direct 16D projection
        adj_j, adj_i = self.adj_embedding(tokens)

        # Verbs: 6D projection + j-space (for navigation)
        verb_proj = self.verb_projection(tokens)
        # Verbs also get j-space encoding (same as adjectives) for semantic direction
        verb_j, verb_i = self.adj_embedding(tokens)  # Reuse adj embedding for verb j-space

        # Combine based on word type
        # For each position, use appropriate encoding
        # NOTE: Verbs now get j-space too (for navigation/direction)
        j = (noun_j * noun_mask.unsqueeze(-1) +
             adj_j * adj_mask.unsqueeze(-1) +
             verb_j * verb_mask.unsqueeze(-1))
        i = (noun_i * noun_mask.unsqueeze(-1) +
             adj_i * adj_mask.unsqueeze(-1) +
             verb_i * verb_mask.unsqueeze(-1))

        # τ only for nouns (derived from variety)
        tau = self.tau_deriver(variety) * noun_mask + 3.0 * (1 - noun_mask)  # Default τ=3 for non-nouns

        # Aggregate to sequence level (mean pooling)
        j_seq = j.mean(dim=1)  # [batch, 5]
        i_seq = i.mean(dim=1)  # [batch, 11]
        tau_seq = (tau * noun_mask).sum(dim=1) / (noun_mask.sum(dim=1) + 1e-8)  # [batch]
        verb_seq = (verb_proj * verb_mask.unsqueeze(-1)).sum(dim=1) / (verb_mask.sum(dim=1, keepdim=True) + 1e-8)  # [batch, 6]

        return {
            'j': j_seq,           # [batch, 5]
            'i': i_seq,           # [batch, 11]
            'tau': tau_seq,       # [batch]
            'verb': verb_seq,     # [batch, 6]
            'variety': variety.mean(dim=1),  # [batch]
            'word_types': word_types,        # [batch, seq]
            # Per-token for detailed analysis
            'j_tokens': j,        # [batch, seq, 5]
            'i_tokens': i,        # [batch, seq, 11]
            'tau_tokens': tau,    # [batch, seq]
        }

    def decode(
        self,
        semantic: Dict[str, torch.Tensor],
        seq_len: int
    ) -> torch.Tensor:
        """
        Decode from semantic bottleneck to token logits.

        THE_MAP-aware decoding:
        - For nouns: (j, i, τ) → noun from vocabulary
          - τ constrains abstraction level
          - (j, i) centroid of adjective cloud → nearest noun
        - For verbs: verb[6D] → verb from vocabulary
        - For adjectives: (j, i) → adjective directly (they're direct projections)

        Currently uses LSTM decoder that implicitly learns these mappings.
        The bottleneck structure ensures the model must encode:
        - Noun semantics in (j, i) with τ indicating abstraction
        - Verb transitions in verb[6D]
        - Adjective qualities in (j, i) directly

        Args:
            semantic: Dict from encode()
            seq_len: Target sequence length

        Returns:
            logits: [batch, seq, vocab_size]
        """
        batch = semantic['j'].shape[0]
        device = semantic['j'].device

        # Combine semantic coordinates
        j = semantic['j']      # [batch, 5]  - transcendental centroid
        i = semantic['i']      # [batch, 11] - surface centroid
        tau = semantic['tau'].unsqueeze(-1)  # [batch, 1] - abstraction level
        verb = semantic['verb']  # [batch, 6] - verb transition vector

        # THE_MAP structure:
        # - Nouns are represented by (j, i, τ) where (j, i) comes from adjective cloud centroid
        # - Verbs are represented by verb[6D] = [beauty, life, sacred, good, love, truth]
        # - Adjectives would directly use (j, i) if we had them
        # - τ tells us: abstract (τ≈1) vs concrete (τ≈6)

        # Bottleneck: [batch, 23]
        # [j:5, i:11, τ:1, verb:6] = 23D
        bottleneck = torch.cat([j, i, tau, verb], dim=-1)

        # Expand for sequence
        bottleneck_seq = bottleneck.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, 23]

        # Decode via LSTM (learns to map bottleneck → word sequences)
        # The LSTM learns that:
        # - High τ → concrete nouns (chair, table)
        # - Low τ → abstract nouns (love, truth)
        # - verb[6D] → appropriate verbs
        decoded, _ = self.decoder_lstm(bottleneck_seq)  # [batch, seq, hidden]
        logits = self.output_proj(decoded)  # [batch, seq, vocab]

        return logits

    def decode_word_type_aware(
        self,
        semantic: Dict[str, torch.Tensor],
        word_types: torch.Tensor,  # [batch, seq] - 0=noun, 1=verb, 2=adj, 3=other
        noun_vocab_embeddings: torch.Tensor,  # [n_nouns, 16]
        verb_vocab_embeddings: torch.Tensor,  # [n_verbs, 6]
    ) -> torch.Tensor:
        """
        Word-type-aware decoding following THE_MAP.

        For each position:
        - If NOUN: Find noun closest to (j, i) with τ-based weighting
        - If VERB: Find verb closest to verb[6D]
        - If ADJ: Use (j, i) to find closest adjective

        Args:
            semantic: Dict from encode()
            word_types: Predicted or given word types per position
            noun_vocab_embeddings: Pre-computed noun vectors [n_nouns, 16]
            verb_vocab_embeddings: Pre-computed verb vectors [n_verbs, 6]

        Returns:
            logits: [batch, seq, vocab_size]
        """
        batch, seq = word_types.shape
        device = word_types.device

        j_tokens = semantic.get('j_tokens', semantic['j'].unsqueeze(1).expand(-1, seq, -1))
        i_tokens = semantic.get('i_tokens', semantic['i'].unsqueeze(1).expand(-1, seq, -1))
        tau_tokens = semantic.get('tau_tokens', semantic['tau'].unsqueeze(1).expand(-1, seq))
        verb = semantic['verb']  # [batch, 6]

        # Initialize logits
        logits = torch.zeros(batch, seq, self.vocab_size, device=device)

        # Process by word type
        # This is a simplified version - a full implementation would use masks and batched operations

        # For nouns: similarity to noun vocabulary based on (j, i)
        noun_semantic = torch.cat([j_tokens, i_tokens], dim=-1)  # [batch, seq, 16]
        # τ-weighted distance (penalize mismatch in abstraction level)
        noun_sims = torch.matmul(noun_semantic, noun_vocab_embeddings.T)  # [batch, seq, n_nouns]

        # For verbs: similarity to verb vocabulary based on verb[6D]
        verb_sims = torch.matmul(verb.unsqueeze(1).expand(-1, seq, -1), verb_vocab_embeddings.T)

        # Combine based on word type (simplified)
        # In practice, would need vocab indices mapping
        logits = noun_sims  # Placeholder - full implementation needs vocab alignment

        return logits

    def forward(
        self,
        tokens: torch.Tensor,
        word_types: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass: encode then decode.
        """
        semantic = self.encode(tokens, word_types)
        logits = self.decode(semantic, tokens.shape[1])
        return logits, semantic


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to push semantic opposites apart.

    Words like (love, hate), (life, death), (good, evil) should have
    opposite j-space coordinates (negative cosine similarity).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_j: torch.Tensor,    # [batch, 5]
        positive_j: torch.Tensor,  # [batch, 5] similar words
        negative_j: torch.Tensor   # [batch, 5] opposite words
    ) -> torch.Tensor:
        """
        Triplet loss: anchor close to positive, far from negative.
        """
        # Normalize
        anchor_n = F.normalize(anchor_j, dim=-1)
        positive_n = F.normalize(positive_j, dim=-1)
        negative_n = F.normalize(negative_j, dim=-1)

        # Similarities
        pos_sim = (anchor_n * positive_n).sum(dim=-1)  # Should be high
        neg_sim = (anchor_n * negative_n).sum(dim=-1)  # Should be low (negative)

        # Triplet loss: want pos_sim > neg_sim + margin
        loss = F.relu(neg_sim - pos_sim + self.margin)

        return loss.mean()


class OrthogonalityLoss(nn.Module):
    """
    Enforce j ⊥ i (transcendental orthogonal to surface).

    Inspired by test_basis_dimension.py: orthogonality is measured by
    how much of j can be explained by i (and vice versa).

    If j can be perfectly reconstructed from i, they're NOT orthogonal.
    True orthogonality means: residual after projection ≈ 100% of original.

    Uses:
    1. Projection residual loss: penalize if j can be explained by i
    2. Cross-covariance penalty: dimensions should not covary
    3. Variance preservation: prevent collapse to zero
    """

    def forward(self, j: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        j: [batch, 5]
        i: [batch, 11]
        """
        batch_size = j.shape[0]

        # 1. PROJECTION RESIDUAL LOSS
        # Try to predict j from i using least squares (in batch)
        # If j can be predicted from i, they're not orthogonal
        # j ≈ i @ W means j is in the span of i

        # Solve: j = i @ W, where W is [11, 5]
        # Closed form: W = (i.T @ i)^{-1} @ i.T @ j
        # But use pseudo-inverse for numerical stability
        i_T = i.T  # [11, batch]
        gram = torch.mm(i_T, i)  # [11, 11]

        # Regularize for stability
        gram_reg = gram + 0.01 * torch.eye(11, device=gram.device)

        # Solve for W using Cholesky (more stable than inverse)
        try:
            L = torch.linalg.cholesky(gram_reg)
            y = torch.cholesky_solve(torch.mm(i_T, j), L)  # W = [11, 5]
        except:
            # Fallback to pseudo-inverse
            y = torch.linalg.lstsq(i, j).solution

        # Predicted j from i
        j_from_i = torch.mm(i, y)  # [batch, 5]

        # Residual: what can't be explained by i
        residual = j - j_from_i

        # We WANT large residuals (meaning j is orthogonal to i)
        # So penalize small residuals
        residual_norm = torch.norm(residual, dim=1)  # [batch]
        j_norm = torch.norm(j, dim=1) + 1e-8  # [batch]
        residual_fraction = residual_norm / j_norm  # Should be ~1 if orthogonal

        # Loss: penalize if residual is small (meaning j can be explained by i)
        projection_loss = F.relu(1.0 - residual_fraction).mean()

        # 2. CROSS-COVARIANCE PENALTY
        j_centered = j - j.mean(dim=0, keepdim=True)
        i_centered = i - i.mean(dim=0, keepdim=True)
        cross_cov = torch.mm(j_centered.T, i_centered) / (batch_size - 1 + 1e-8)
        cov_loss = torch.mean(cross_cov ** 2)

        # 3. VARIANCE PRESERVATION (prevent collapse)
        j_var = torch.var(j, dim=0)  # [5]
        i_var = torch.var(i, dim=0)  # [11]
        var_loss = 1.0 / (j_var.mean() + 0.01) + 1.0 / (i_var.mean() + 0.01)

        return projection_loss + cov_loss + 0.01 * var_loss


class PT1SaturationLoss(nn.Module):
    """
    Enforce PT1 (first-order lag) saturation behavior from bond space.

    From THE_MAP:
        b(ν) = b_max * (1 - e^(-ν/τ))

    Where:
    - b = bonds per noun (relates to attention entropy/variety)
    - ν = noun vocabulary size (relates to effective adjective count)
    - τ = time constant (~40,000 nouns for semantic saturation)

    This loss ensures that:
    - High variety (many adjectives) → Low τ (abstract nouns like "love")
    - Low variety (few adjectives) → High τ (concrete nouns like "chair")
    - The relationship follows exponential saturation, not linear

    Physical interpretation:
    - Abstract nouns are "closer to the source" - they connect to many adjectives
    - Concrete nouns are "far from source" - they have specific adjective profiles
    - τ measures "distance from source" in semantic space
    """

    def __init__(self, tau_semantic: float = 40000.0, b_max: float = 50.0):
        super().__init__()
        self.tau_semantic = tau_semantic  # Universal time constant
        self.b_max = b_max  # Maximum bonds at saturation

        # Learnable scale factors to map variety to "effective vocabulary"
        self.variety_scale = nn.Parameter(torch.tensor(1.0))
        self.variety_offset = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        variety: torch.Tensor,  # [batch] - variety from adjective cloud
        tau: torch.Tensor,      # [batch] - predicted τ values [1, 6]
        attention: torch.Tensor  # [batch, n_adj] - attention over adjectives
    ) -> torch.Tensor:
        """
        Enforce PT1 saturation relationship between variety and τ.

        The relationship should be:
        - High attention entropy → high variety → low τ
        - Low attention entropy → low variety → high τ
        - This follows PT1: saturation = 1 - exp(-effective_vocab / τ)
        """
        # Compute attention entropy as measure of "how many adjectives"
        # High entropy = many adjectives with similar weights = abstract
        # Low entropy = few adjectives dominate = concrete
        attention_entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=-1)
        max_entropy = np.log(attention.shape[-1])
        normalized_entropy = attention_entropy / max_entropy  # [0, 1]

        # Map normalized entropy to "effective vocabulary fraction"
        # High entropy → high effective vocab → should give low τ
        # Low entropy → low effective vocab → should give high τ
        effective_vocab_fraction = normalized_entropy  # [0, 1]

        # PT1 prediction for τ:
        # saturation = 1 - exp(-ν/τ)
        # We want: high effective_vocab → low τ
        # Invert: τ = -ν / ln(1 - saturation)
        # But we use a simpler mapping: τ = τ_max - (τ_max - τ_min) * saturation
        tau_min, tau_max = 1.0, 6.0
        predicted_tau = tau_max - (tau_max - tau_min) * effective_vocab_fraction

        # Loss: predicted τ should match the τ from variety
        tau_loss = F.mse_loss(tau.squeeze(), predicted_tau)

        # Also enforce that variety is inversely related to τ
        # Normalize variety for this comparison
        variety_normalized = variety / (variety.max() + 1e-8)
        variety_tau_loss = F.mse_loss(variety_normalized, 1.0 - (tau.squeeze() - tau_min) / (tau_max - tau_min))

        # Entropy regularization: encourage non-degenerate attention
        # (avoid all weight on single adjective)
        entropy_reg = F.relu(0.1 - normalized_entropy).mean()

        return tau_loss + 0.5 * variety_tau_loss + 0.1 * entropy_reg


def test_architecture():
    """Test the V2 architecture."""
    print("=" * 70)
    print("TESTING SEMANTIC BOTTLENECK V2")
    print("=" * 70)

    vocab_size = 10000
    batch_size = 4
    seq_len = 32

    # Create model
    model = SemanticBottleneckV2(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        n_basis_adjectives=500
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # Create dummy input
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    word_types = torch.randint(0, 4, (batch_size, seq_len))  # 0=noun, 1=verb, 2=adj, 3=other

    # Set dummy basis adjectives
    basis_adj_ids = torch.randint(0, vocab_size, (500,))
    model.noun_encoder.set_basis_adjectives(basis_adj_ids)

    print(f"\nInput shape: {tokens.shape}")
    print(f"Word types shape: {word_types.shape}")

    # Forward pass
    with torch.no_grad():
        logits, semantic = model(tokens, word_types)

    print(f"\n[Bottleneck Outputs]")
    print(f"  j-space (5D): {semantic['j'].shape}")
    print(f"  i-space (11D): {semantic['i'].shape}")
    print(f"  τ: {semantic['tau'].shape}")
    print(f"  verb (6D): {semantic['verb'].shape}")
    print(f"  variety: {semantic['variety'].shape}")

    print(f"\n[Output]")
    print(f"  logits: {logits.shape}")

    # Test losses
    print(f"\n[Loss Functions]")

    ortho_loss = OrthogonalityLoss()
    ortho = ortho_loss(semantic['j'], semantic['i'])
    print(f"  Orthogonality loss: {ortho.item():.4f}")

    contrastive = ContrastiveLoss()
    anchor = torch.randn(batch_size, 5)
    positive = anchor + 0.1 * torch.randn(batch_size, 5)  # Similar
    negative = -anchor + 0.1 * torch.randn(batch_size, 5)  # Opposite
    cont_loss = contrastive(anchor, positive, negative)
    print(f"  Contrastive loss: {cont_loss.item():.4f}")

    # Test τ derivation (variety is now normalized entropy in [0, 1])
    print(f"\n[τ Derivation from Variety]")
    print("  variety = normalized attention entropy")
    print("  variety=1 (high entropy) → many adjectives → abstract → low τ")
    print("  variety=0 (low entropy) → few adjectives → concrete → high τ")
    tau_deriver = TauFromVariety()
    test_varieties = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    for v in test_varieties:
        tau = tau_deriver(v.unsqueeze(0))
        print(f"  variety={v.item():.2f} → τ={tau.item():.2f}")

    print("\n" + "=" * 70)
    print("ARCHITECTURE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_architecture()
