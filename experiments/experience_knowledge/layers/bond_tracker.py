"""
Bond Tracker: User-specific path tracking

Tracks walked paths per user as a lightweight map on edges.
Does NOT change navigation behavior - just records data.

Architecture:
- Edge has base weight (corpus/common)
- Edge has bonds map: {user_id: {w, n, last}}
- BondTracker reads/writes bonds, reuses WeightDynamics

Edge structure:
    -[:TRANSITION {
        weight: 0.7,                    # base (corpus)
        bonds: {                        # user bonds (optional)
            "user_123": {w: 0.9, n: 5, last: "2025-12-23"},
            ...
        }
    }]->

"The signature is not yours alone - it's the path we walked together."
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import json

from .dynamics import WeightDynamics, WeightConfig


@dataclass
class Bond:
    """A user's bond to a specific edge."""
    user_id: str
    weight: float
    walks: int
    last_walked: str  # ISO timestamp

    def to_dict(self) -> dict:
        return {
            "w": self.weight,
            "n": self.walks,
            "last": self.last_walked
        }

    @classmethod
    def from_dict(cls, user_id: str, data: dict) -> 'Bond':
        return cls(
            user_id=user_id,
            weight=data.get("w", 0.1),
            walks=data.get("n", 0),
            last_walked=data.get("last", "")
        )


class BondTracker:
    """
    Track user bonds on edges.

    Reuses WeightDynamics for learn/forget.
    Stores bonds as map property on TRANSITION edges.

    Usage:
        tracker = BondTracker(driver)
        tracker.record_walk("love", "truth", "user_123")
        bond = tracker.get_bond("love", "truth", "user_123")
    """

    def __init__(self, driver, dynamics: WeightDynamics = None):
        """
        Args:
            driver: Neo4j driver instance
            dynamics: WeightDynamics instance (uses default if None)
        """
        self.driver = driver
        self.dynamics = dynamics or WeightDynamics()

    def _parse_bonds(self, bonds_json: str) -> dict:
        """Parse bonds JSON string to dict."""
        if not bonds_json:
            return {}
        try:
            return json.loads(bonds_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _serialize_bonds(self, bonds: dict) -> str:
        """Serialize bonds dict to JSON string."""
        return json.dumps(bonds)

    def get_bond(self, from_word: str, to_word: str, user_id: str) -> Optional[Bond]:
        """
        Get user's bond on an edge.

        Returns None if edge doesn't exist or user has no bond.
        """
        if not self.driver:
            return None

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:SemanticState {word: $from_word})
                      -[t:TRANSITION]->
                      (b:SemanticState {word: $to_word})
                RETURN t.bonds as bonds
            """, from_word=from_word, to_word=to_word)

            record = result.single()
            if not record:
                return None

            bonds = self._parse_bonds(record["bonds"])
            if not bonds or user_id not in bonds:
                return None

            return Bond.from_dict(user_id, bonds[user_id])

    def record_walk(self, from_word: str, to_word: str, user_id: str) -> Optional[Bond]:
        """
        Record a walk for a user on an edge.

        - Creates bond if first walk
        - Applies learning dynamics if existing
        - Updates timestamp

        Returns updated bond, or None if edge doesn't exist.
        """
        if not self.driver or not user_id:
            return None

        now = datetime.now().isoformat()

        with self.driver.session() as session:
            # Get current bond state
            result = session.run("""
                MATCH (a:SemanticState {word: $from_word})
                      -[t:TRANSITION]->
                      (b:SemanticState {word: $to_word})
                RETURN t.bonds as bonds
            """, from_word=from_word, to_word=to_word)

            record = result.single()
            if not record:
                return None

            bonds = self._parse_bonds(record["bonds"])

            # Calculate new bond
            if user_id in bonds:
                current = Bond.from_dict(user_id, bonds[user_id])
                new_weight = self.dynamics.learn(current.weight, reinforcements=1)
                new_walks = current.walks + 1
            else:
                # First walk - start with low weight
                new_weight = self.dynamics.config.w_min * 2  # conversation source
                new_walks = 1

            # Update bond in map
            new_bond = Bond(
                user_id=user_id,
                weight=new_weight,
                walks=new_walks,
                last_walked=now
            )
            bonds[user_id] = new_bond.to_dict()

            # Write back to edge as JSON string
            session.run("""
                MATCH (a:SemanticState {word: $from_word})
                      -[t:TRANSITION]->
                      (b:SemanticState {word: $to_word})
                SET t.bonds = $bonds
            """, from_word=from_word, to_word=to_word, bonds=self._serialize_bonds(bonds))

            return new_bond

    def decay_user_bonds(self, user_id: str, days_elapsed: float) -> Dict:
        """
        Apply forgetting to all bonds for a user.

        Called during sleep or after inactivity.

        Returns:
            Statistics about decay applied
        """
        if not self.driver or not user_id:
            return {"error": "No driver or user_id"}

        stats = {"bonds_processed": 0, "bonds_decayed": 0, "total_decay": 0.0}

        with self.driver.session() as session:
            # Find all edges with bonds (check if user exists in parsed JSON)
            result = session.run("""
                MATCH (a:SemanticState)-[t:TRANSITION]->(b:SemanticState)
                WHERE t.bonds IS NOT NULL
                RETURN a.word as from_word, b.word as to_word, t.bonds as bonds
            """)

            for record in result:
                from_word = record["from_word"]
                to_word = record["to_word"]
                bonds = self._parse_bonds(record["bonds"])

                if user_id not in bonds:
                    continue

                stats["bonds_processed"] += 1

                current = Bond.from_dict(user_id, bonds[user_id])
                new_weight = self.dynamics.forget(current.weight, days_elapsed)

                # Only update if weight actually changed
                if abs(new_weight - current.weight) > 0.01:
                    decay_amount = current.weight - new_weight
                    stats["bonds_decayed"] += 1
                    stats["total_decay"] += decay_amount

                    # Update bond
                    bonds[user_id]["w"] = new_weight

                    session.run("""
                        MATCH (a:SemanticState {word: $from_word})
                              -[t:TRANSITION]->
                              (b:SemanticState {word: $to_word})
                        SET t.bonds = $bonds
                    """, from_word=from_word, to_word=to_word, bonds=self._serialize_bonds(bonds))

        return stats

    def get_user_walks(self, user_id: str, limit: int = 100) -> List[Dict]:
        """
        Get all walked edges for a user.

        Returns list of {from, to, weight, walks, last} sorted by walks desc.
        """
        if not self.driver or not user_id:
            return []

        walks = []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:SemanticState)-[t:TRANSITION]->(b:SemanticState)
                WHERE t.bonds IS NOT NULL
                RETURN a.word as from_word, b.word as to_word, t.bonds as bonds
            """)

            for record in result:
                bonds = self._parse_bonds(record["bonds"])
                if user_id in bonds:
                    bond_data = bonds[user_id]
                    walks.append({
                        "from": record["from_word"],
                        "to": record["to_word"],
                        "weight": bond_data.get("w", 0),
                        "walks": bond_data.get("n", 0),
                        "last": bond_data.get("last", "")
                    })

        # Sort by walks descending
        walks.sort(key=lambda x: -x["walks"])
        return walks[:limit]

    def get_user_stats(self, user_id: str) -> Dict:
        """
        Get statistics for a user's bonds.
        """
        if not self.driver or not user_id:
            return {}

        # Since bonds are stored as JSON string, we need to parse in Python
        total_edges = 0
        total_walks = 0
        total_weight = 0.0
        max_walks = 0

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:SemanticState)-[t:TRANSITION]->(b:SemanticState)
                WHERE t.bonds IS NOT NULL
                RETURN t.bonds as bonds
            """)

            for record in result:
                bonds = self._parse_bonds(record["bonds"])
                if user_id in bonds:
                    bond = bonds[user_id]
                    total_edges += 1
                    walks = bond.get("n", 0)
                    total_walks += walks
                    total_weight += bond.get("w", 0)
                    if walks > max_walks:
                        max_walks = walks

        return {
            "total_edges": total_edges,
            "total_walks": total_walks,
            "avg_weight": total_weight / total_edges if total_edges > 0 else 0,
            "max_walks": max_walks
        }


# Convenience function for quick access
def create_tracker(driver) -> BondTracker:
    """Create a BondTracker with default dynamics."""
    return BondTracker(driver, WeightDynamics())
