#!/usr/bin/env python3
"""Sync learned bonds from PostgreSQL to Neo4j.

This script syncs bonds and trajectories learned during conversations
from PostgreSQL (primary storage) to Neo4j (graph traversal).

Usage:
    python -m storm_logos.scripts.sync_bonds --all
    python -m storm_logos.scripts.sync_bonds --since "2024-01-01"
    python -m storm_logos.scripts.sync_bonds --min-use 3
    python -m storm_logos.scripts.sync_bonds --dry-run
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from storm_logos.data.postgres import PostgresData, get_data
from storm_logos.data.neo4j import Neo4jData, get_neo4j
from storm_logos.data.models import Bond


class BondSyncer:
    """Syncs learned bonds from PostgreSQL to Neo4j."""

    def __init__(self, pg: PostgresData = None, neo4j: Neo4jData = None):
        self.pg = pg or get_data()
        self.neo4j = neo4j

    def connect(self) -> bool:
        """Connect to Neo4j."""
        if self.neo4j is None:
            self.neo4j = get_neo4j()
        return self.neo4j.connect()

    def get_learned_bonds(self,
                          since: datetime = None,
                          min_use_count: int = 1,
                          limit: int = 10000) -> List[Tuple[Bond, dict]]:
        """Get learned bonds from PostgreSQL with metadata.

        Args:
            since: Only get bonds learned after this date
            min_use_count: Minimum use count filter
            limit: Maximum bonds to return

        Returns:
            List of (Bond, metadata) tuples
        """
        import psycopg2

        try:
            conn = psycopg2.connect(**self.pg.config.as_dict())
            cur = conn.cursor()

            if since:
                cur.execute('''
                    SELECT adj, noun, A, S, tau, use_count, source, confidence,
                           created_at, last_used
                    FROM learned_bonds
                    WHERE created_at >= %s AND use_count >= %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ''', (since, min_use_count, limit))
            else:
                cur.execute('''
                    SELECT adj, noun, A, S, tau, use_count, source, confidence,
                           created_at, last_used
                    FROM learned_bonds
                    WHERE use_count >= %s
                    ORDER BY use_count DESC, last_used DESC
                    LIMIT %s
                ''', (min_use_count, limit))

            results = []
            for row in cur.fetchall():
                bond = Bond(
                    adj=row[0],
                    noun=row[1],
                    A=row[2],
                    S=row[3],
                    tau=row[4],
                    variety=row[5],  # use_count
                )
                metadata = {
                    'source': row[6],
                    'confidence': row[7],
                    'created_at': row[8],
                    'last_used': row[9],
                }
                results.append((bond, metadata))

            conn.close()
            return results

        except Exception as e:
            print(f"Error getting learned bonds: {e}")
            return []

    def get_bond_trajectories(self, limit: int = 1000) -> List[List[Bond]]:
        """Get conversation trajectories from PostgreSQL.

        Trajectories are sequences of bonds that occurred together
        in conversations.

        Returns:
            List of bond sequences (trajectories)
        """
        import psycopg2

        # Check if we have a trajectories table
        try:
            conn = psycopg2.connect(**self.pg.config.as_dict())
            cur = conn.cursor()

            # Check if conversation_trajectories table exists
            cur.execute('''
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'conversation_trajectories'
                )
            ''')

            if not cur.fetchone()[0]:
                conn.close()
                return []

            # Get trajectories
            cur.execute('''
                SELECT session_id, bond_sequence, created_at
                FROM conversation_trajectories
                ORDER BY created_at DESC
                LIMIT %s
            ''', (limit,))

            trajectories = []
            for row in cur.fetchall():
                # Parse bond sequence (stored as JSON array)
                import json
                sequence = json.loads(row[1]) if isinstance(row[1], str) else row[1]

                bonds = []
                for item in sequence:
                    if isinstance(item, dict):
                        bonds.append(Bond(
                            adj=item.get('adj', ''),
                            noun=item.get('noun', ''),
                            A=item.get('A', 0.0),
                            S=item.get('S', 0.0),
                            tau=item.get('tau', 2.5),
                        ))

                if bonds:
                    trajectories.append(bonds)

            conn.close()
            return trajectories

        except Exception as e:
            print(f"Error getting trajectories: {e}")
            return []

    def sync_bond(self, bond: Bond, metadata: dict = None) -> bool:
        """Sync a single bond to Neo4j.

        Args:
            bond: Bond to sync
            metadata: Optional metadata (source, confidence, etc.)

        Returns:
            True if synced successfully
        """
        if not self.neo4j or not self.neo4j._connected:
            return False

        # Add bond node
        bond_id = self.neo4j.add_bond(bond)
        return bool(bond_id)

    def sync_trajectory(self, trajectory: List[Bond],
                        session_id: str = 'conversation') -> int:
        """Sync a conversation trajectory to Neo4j.

        Creates bonds and FOLLOWS edges.

        Args:
            trajectory: List of bonds in sequence
            session_id: Session identifier for the trajectory

        Returns:
            Number of edges created
        """
        if not self.neo4j or not self.neo4j._connected:
            return 0

        if len(trajectory) < 2:
            return 0

        edges_created = 0

        for i, bond in enumerate(trajectory):
            # Add bond node
            self.neo4j.add_bond(bond)

            # Add FOLLOWS edge to previous
            if i > 0:
                success = self.neo4j.add_follows(
                    source=trajectory[i-1],
                    target=bond,
                    book_id=session_id,
                    chapter=0,
                    sentence=i,
                    position=i,
                    weight=0.2,  # Conversation weight
                    edge_source='conversation',
                )
                if success:
                    edges_created += 1

        return edges_created

    def sync_all(self,
                 since: datetime = None,
                 min_use_count: int = 1,
                 dry_run: bool = False) -> dict:
        """Sync all learned bonds and trajectories.

        Args:
            since: Only sync bonds learned after this date
            min_use_count: Minimum use count filter
            dry_run: If True, only report what would be synced

        Returns:
            Statistics dict
        """
        stats = {
            'bonds_found': 0,
            'bonds_synced': 0,
            'trajectories_found': 0,
            'edges_created': 0,
            'errors': 0,
        }

        # Get learned bonds
        print("Fetching learned bonds from PostgreSQL...")
        bonds_with_meta = self.get_learned_bonds(
            since=since,
            min_use_count=min_use_count
        )
        stats['bonds_found'] = len(bonds_with_meta)
        print(f"  Found {len(bonds_with_meta)} learned bonds")

        if dry_run:
            print("\n[DRY RUN] Would sync:")
            for bond, meta in bonds_with_meta[:10]:
                print(f"  - {bond.adj} {bond.noun} "
                      f"(A={bond.A:.2f}, S={bond.S:.2f}, tau={bond.tau:.2f}) "
                      f"[uses: {bond.variety}]")
            if len(bonds_with_meta) > 10:
                print(f"  ... and {len(bonds_with_meta) - 10} more")
            return stats

        # Sync bonds
        print("\nSyncing bonds to Neo4j...")
        for bond, meta in bonds_with_meta:
            try:
                if self.sync_bond(bond, meta):
                    stats['bonds_synced'] += 1
                else:
                    stats['errors'] += 1
            except Exception as e:
                print(f"  Error syncing {bond.adj}_{bond.noun}: {e}")
                stats['errors'] += 1

        print(f"  Synced {stats['bonds_synced']}/{stats['bonds_found']} bonds")

        # Get and sync trajectories
        print("\nFetching conversation trajectories...")
        trajectories = self.get_bond_trajectories()
        stats['trajectories_found'] = len(trajectories)
        print(f"  Found {len(trajectories)} trajectories")

        if trajectories:
            print("\nSyncing trajectories...")
            for i, traj in enumerate(trajectories):
                try:
                    edges = self.sync_trajectory(traj, f"conversation_{i}")
                    stats['edges_created'] += edges
                except Exception as e:
                    print(f"  Error syncing trajectory {i}: {e}")
                    stats['errors'] += 1

            print(f"  Created {stats['edges_created']} FOLLOWS edges")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Sync learned bonds from PostgreSQL to Neo4j',
        prog='sync_bonds',
    )

    parser.add_argument('--all', action='store_true',
                        help='Sync all learned bonds')
    parser.add_argument('--since', type=str, default=None,
                        help='Only sync bonds learned after this date (YYYY-MM-DD)')
    parser.add_argument('--min-use', type=int, default=1,
                        help='Minimum use count to sync (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be synced without syncing')

    args = parser.parse_args()

    # Parse since date
    since = None
    if args.since:
        try:
            since = datetime.strptime(args.since, '%Y-%m-%d')
        except ValueError:
            print(f"ERROR: Invalid date format: {args.since}")
            print("Use YYYY-MM-DD format")
            sys.exit(1)

    print("=" * 60)
    print("PostgreSQL → Neo4j Bond Sync")
    print("=" * 60)
    print()

    syncer = BondSyncer()

    # Connect to Neo4j
    if not args.dry_run:
        print("Connecting to Neo4j...")
        if not syncer.connect():
            print("ERROR: Could not connect to Neo4j")
            sys.exit(1)
        print("Connected!")
        print()

    # Run sync
    stats = syncer.sync_all(
        since=since,
        min_use_count=args.min_use,
        dry_run=args.dry_run,
    )

    # Summary
    print()
    print("=" * 60)
    print("SYNC COMPLETE" if not args.dry_run else "DRY RUN COMPLETE")
    print("=" * 60)
    print(f"  Bonds found:      {stats['bonds_found']}")
    print(f"  Bonds synced:     {stats['bonds_synced']}")
    print(f"  Trajectories:     {stats['trajectories_found']}")
    print(f"  Edges created:    {stats['edges_created']}")
    print(f"  Errors:           {stats['errors']}")


if __name__ == '__main__':
    main()
