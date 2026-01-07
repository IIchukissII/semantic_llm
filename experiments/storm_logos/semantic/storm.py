"""Storm: Candidate Explosion Algorithm.

Generates candidates around current position Q from multiple sources:
1. Neo4j FOLLOWS edges - where authors went next
2. Spatial neighbors - bonds within radius in (A, S, τ) space
3. Gravity direction - candidates in gravity well direction

Like neocortical activation burst: explode all possibilities,
then filter via Logos.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
import math

from ..data.models import Bond, SemanticState
from ..data.postgres import PostgresData, get_data
from ..data.neo4j import Neo4jData, get_neo4j
from ..config import get_config, StormConfig


class Storm:
    """Storm algorithm: explode candidates around current Q.

    Sources (with configurable weights):
        - FOLLOWS: Neo4j graph edges (where authors went)
        - SPATIAL: Nearby bonds in coordinate space
        - GRAVITY: Direction toward gravity well

    Returns unfiltered candidates for Dialectic filtering.
    """

    def __init__(self,
                 data: Optional[PostgresData] = None,
                 neo4j: Optional[Neo4jData] = None,
                 config: Optional[StormConfig] = None):
        self.data = data or get_data(load_bonds=True)
        self.neo4j = neo4j or get_neo4j()
        self.config = config or get_config().storm

        # Build spatial index
        self._tree: Optional[cKDTree] = None
        self._indexed_bonds: List[Bond] = []
        self._build_spatial_index()

    def _build_spatial_index(self):
        """Build KD-tree for spatial queries."""
        if not self.data.bonds:
            return

        coords = []
        self._indexed_bonds = []

        for bond in self.data.bonds:
            coords.append([bond.A, bond.S, bond.tau])
            self._indexed_bonds.append(bond)

        if coords:
            self._tree = cKDTree(np.array(coords))

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================

    def explode(self, Q: SemanticState,
                radius: Optional[float] = None,
                max_candidates: Optional[int] = None) -> List[Bond]:
        """Explode candidates around Q.

        Combines all sources:
            - FOLLOWS edges from Neo4j
            - Spatial neighbors from coordinate space
            - Gravity-directed candidates

        Args:
            Q: Current state
            radius: Search radius (uses config default if None)
            max_candidates: Maximum candidates (uses config default if None)

        Returns:
            List of candidate bonds (unfiltered)
        """
        radius = radius or self.config.radius
        max_candidates = max_candidates or self.config.max_candidates

        candidates = []
        weights = self.config.sources_weight

        # Source 1: FOLLOWS edges (if Neo4j connected)
        if weights.get('follows', 0) > 0:
            follows = self._get_follows_candidates(Q)
            candidates.extend(follows)

        # Source 2: Spatial neighbors
        if weights.get('spatial', 0) > 0:
            spatial = self._get_spatial_candidates(Q, radius)
            candidates.extend(spatial)

        # Source 3: Gravity direction
        if weights.get('gravity', 0) > 0:
            gravity = self._get_gravity_candidates(Q, radius)
            candidates.extend(gravity)

        # Deduplicate by bond text
        seen = set()
        unique = []
        for bond in candidates:
            key = (bond.adj, bond.noun)
            if key not in seen:
                seen.add(key)
                unique.append(bond)

        # Limit to max candidates
        if len(unique) > max_candidates:
            # Sort by variety (frequency) and take top
            unique.sort(key=lambda b: b.variety, reverse=True)
            unique = unique[:max_candidates]

        return unique

    # ========================================================================
    # SOURCE: FOLLOWS
    # ========================================================================

    def _get_follows_candidates(self, Q: SemanticState) -> List[Bond]:
        """Get candidates from Neo4j FOLLOWS edges.

        Queries: where did authors go from similar positions?
        """
        candidates = []

        # Find current position's nearest bond
        if not self._tree or not self._indexed_bonds:
            return candidates

        # Query Neo4j for followers
        # For now, we use spatial nearest as seed
        nearest = self._get_nearest_bond(Q)
        if nearest:
            bond_id = f"{nearest.adj}_{nearest.noun}" if nearest.adj else nearest.noun
            followers = self.neo4j.get_followers(bond_id)
            candidates.extend(followers)

        return candidates

    def _get_nearest_bond(self, Q: SemanticState) -> Optional[Bond]:
        """Get nearest bond to Q in coordinate space."""
        if not self._tree:
            return None

        query = np.array([Q.A, Q.S, Q.tau])
        dist, idx = self._tree.query(query, k=1)

        if idx < len(self._indexed_bonds):
            return self._indexed_bonds[idx]
        return None

    # ========================================================================
    # SOURCE: SPATIAL
    # ========================================================================

    def _get_spatial_candidates(self, Q: SemanticState,
                                radius: float) -> List[Bond]:
        """Get candidates within radius in (A, S, τ) space."""
        if not self._tree:
            return []

        query = np.array([Q.A, Q.S, Q.tau])
        indices = self._tree.query_ball_point(query, radius)

        candidates = []
        for idx in indices:
            if idx < len(self._indexed_bonds):
                candidates.append(self._indexed_bonds[idx])

        return candidates

    # ========================================================================
    # SOURCE: GRAVITY
    # ========================================================================

    def _get_gravity_candidates(self, Q: SemanticState,
                                radius: float) -> List[Bond]:
        """Get candidates in gravity direction.

        Gravity pulls toward:
            - Lower τ (more concrete)
            - Higher A (more affirming)
        """
        if not self._tree:
            return []

        # Gravity target: shift Q toward concrete/good
        from ..config import LAMBDA, MU

        target_A = Q.A + MU * 0.3  # Move toward +A
        target_tau = Q.tau - LAMBDA * 0.3  # Move toward lower τ

        target = np.array([target_A, Q.S, target_tau])
        indices = self._tree.query_ball_point(target, radius)

        candidates = []
        for idx in indices:
            if idx < len(self._indexed_bonds):
                candidates.append(self._indexed_bonds[idx])

        return candidates

    # ========================================================================
    # UTILITY
    # ========================================================================

    def get_candidates_by_coords(self, A: float, S: float, tau: float,
                                 radius: float = 0.5) -> List[Bond]:
        """Get candidates around specific coordinates."""
        if not self._tree:
            return []

        query = np.array([A, S, tau])
        indices = self._tree.query_ball_point(query, radius)

        candidates = []
        for idx in indices:
            if idx < len(self._indexed_bonds):
                candidates.append(self._indexed_bonds[idx])

        return candidates

    def stats(self) -> Dict:
        """Return storm statistics."""
        return {
            'n_indexed_bonds': len(self._indexed_bonds),
            'tree_size': self._tree.n if self._tree else 0,
            'config': {
                'radius': self.config.radius,
                'max_candidates': self.config.max_candidates,
                'sources_weight': self.config.sources_weight,
            }
        }


# ============================================================================
# SINGLETON
# ============================================================================

_storm_instance: Optional[Storm] = None


def get_storm() -> Storm:
    """Get singleton Storm instance."""
    global _storm_instance
    if _storm_instance is None:
        _storm_instance = Storm()
    return _storm_instance
