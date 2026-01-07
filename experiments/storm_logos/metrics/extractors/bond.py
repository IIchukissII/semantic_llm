"""Bond Extractor: Bonds -> Coordinates.

Looks up coordinates for bonds from the data layer.
"""

from typing import List, Optional, Tuple

from ...data.models import Bond, WordCoordinates
from ...data.postgres import PostgresData, get_data


class BondExtractor:
    """Extract coordinates for bonds."""

    def __init__(self, data: Optional[PostgresData] = None):
        self.data = data or get_data()

    def get_coordinates(self, bond: Bond) -> Tuple[float, float, float]:
        """Get (A, S, τ) coordinates for a bond.

        Uses noun coordinates, potentially modified by adjective.

        Args:
            bond: Bond to look up

        Returns:
            (A, S, τ) tuple
        """
        # Look up noun first (if available)
        if bond.noun:
            coords = self.data.get(bond.noun)
            if coords:
                return (coords.A, coords.S, coords.tau)

        # Try adjective as fallback
        if bond.adj:
            coords = self.data.get(bond.adj)
            if coords:
                return (coords.A, coords.S, coords.tau)

        # Fallback to default
        return (0.0, 0.0, 2.5)

    def enrich_bond(self, bond: Bond) -> Bond:
        """Add coordinates to a bond.

        Args:
            bond: Bond to enrich

        Returns:
            Bond with coordinates set
        """
        A, S, tau = self.get_coordinates(bond)
        return Bond(
            noun=bond.noun,
            adj=bond.adj,
            variety=bond.variety,
            A=A,
            S=S,
            tau=tau,
        )

    def enrich_bonds(self, bonds: List[Bond]) -> List[Bond]:
        """Enrich multiple bonds.

        Args:
            bonds: List of bonds

        Returns:
            List of bonds with coordinates
        """
        return [self.enrich_bond(b) for b in bonds]

    def compute_mean_coords(self, bonds: List[Bond]) -> Tuple[float, float, float]:
        """Compute mean coordinates over bonds.

        Args:
            bonds: List of bonds

        Returns:
            Mean (A, S, τ)
        """
        if not bonds:
            return (0.0, 0.0, 2.5)

        A_sum, S_sum, tau_sum = 0.0, 0.0, 0.0
        count = 0

        for bond in bonds:
            if bond.A != 0 or bond.S != 0 or bond.tau != 2.5:
                A_sum += bond.A
                S_sum += bond.S
                tau_sum += bond.tau
                count += 1
            else:
                # Try to look up
                A, S, tau = self.get_coordinates(bond)
                A_sum += A
                S_sum += S
                tau_sum += tau
                count += 1

        if count == 0:
            return (0.0, 0.0, 2.5)

        return (A_sum / count, S_sum / count, tau_sum / count)

    def lookup_word(self, word: str) -> Optional[WordCoordinates]:
        """Look up coordinates for a single word.

        Args:
            word: Word to look up

        Returns:
            WordCoordinates or None
        """
        return self.data.get(word)
