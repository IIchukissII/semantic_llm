#!/usr/bin/env python3
"""
Nightly Decay Job - Apply forgetting to FOLLOWS edge weights.

This script implements the "forgetting" part of the learning/forgetting dynamics:

    dw/dt = lambda_forget * (w_min - w)

    Discrete form:
    w(t+dt) = w_min + (w - w_min) * e^(-lambda_forget * dt)

Parameters:
    w_min = 0.1             Floor - never fully forgotten
    lambda_forget = 0.05    Forgetting rate (per day)

Run modes:
    1. Simple nightly decay (default):
       python nightly_decay.py
       Applies 1 day of decay to all edges

    2. Custom days elapsed:
       python nightly_decay.py --days 7
       Applies 7 days of decay (e.g., after a week offline)

    3. Timestamp-based decay:
       python nightly_decay.py --timestamp-based
       Each edge decays based on its own last_used timestamp

    4. Dry run (preview):
       python nightly_decay.py --dry-run
       Shows what would happen without applying changes

    5. Initialize weights:
       python nightly_decay.py --init-weights
       Sets weight=1.0 on edges that don't have it

Cron setup:
    # Run every night at 3 AM
    0 3 * * * /path/to/python /path/to/nightly_decay.py >> /path/to/decay.log 2>&1

Usage:
    python -m storm_logos.scripts.nightly_decay [options]

"Knowledge is never lost. It only becomes dormant."
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Setup path for imports
_THIS_FILE = Path(__file__).resolve()
_STORM_LOGOS = _THIS_FILE.parent.parent
sys.path.insert(0, str(_STORM_LOGOS.parent))

from storm_logos.data.neo4j import Neo4jData, get_neo4j
from storm_logos.data.weight_dynamics import get_dynamics_info, W_MIN, LAMBDA_FORGET


def format_timestamp():
    """Get formatted timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"[{format_timestamp()}] {title}")
    print('=' * 60)


def run_simple_decay(neo4j: Neo4jData, days: float, dry_run: bool) -> dict:
    """
    Run simple decay for a fixed number of days.

    Args:
        neo4j: Neo4jData instance
        days: Number of days of decay to apply
        dry_run: If True, only preview changes

    Returns:
        Decay statistics
    """
    print(f"\nApplying {days} day(s) of decay...")
    print(f"  Formula: w(t+dt) = {W_MIN} + (w - {W_MIN}) * e^(-{LAMBDA_FORGET} * {days})")

    result = neo4j.apply_decay(days_elapsed=days, dry_run=dry_run)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return result

    print(f"\nResults {'(DRY RUN)' if dry_run else ''}:")
    print(f"  Edges affected: {result.get('edges_affected', 0)}")

    if result.get('total_weight_before'):
        print(f"  Total weight before: {result['total_weight_before']:.2f}")
        print(f"  Total weight after:  {result['total_weight_after']:.2f}")
        print(f"  Total decay:         {result.get('total_decay', 0):.4f}")
        print(f"  Newly dormant edges: {result.get('newly_dormant', 0)}")

    if not dry_run and result.get('applied_at'):
        print(f"  Applied at: {result['applied_at']}")

    return result


def run_timestamp_decay(neo4j: Neo4jData, dry_run: bool) -> dict:
    """
    Run decay based on individual edge timestamps.

    Args:
        neo4j: Neo4jData instance
        dry_run: If True, only preview changes

    Returns:
        Decay statistics
    """
    print("\nApplying timestamp-based decay...")
    print("  Each edge decays based on its own last_used timestamp")

    result = neo4j.apply_decay_since_last_use(dry_run=dry_run)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return result

    if result.get('edges_with_timestamp') == 0:
        print(f"  NOTE: {result.get('note', 'No edges with timestamps')}")
        print("  Run with --init-weights first to set weights and timestamps")
        return result

    print(f"\nResults {'(DRY RUN)' if dry_run else ''}:")
    print(f"  Edges affected: {result.get('edges_affected', 0)}")

    if result.get('total_weight_before'):
        print(f"  Total weight before: {result['total_weight_before']:.2f}")
        print(f"  Total weight after:  {result['total_weight_after']:.2f}")
        if result.get('avg_days_since_use'):
            print(f"  Avg days since use:  {result['avg_days_since_use']:.1f}")
            print(f"  Max days since use:  {result['max_days_since_use']:.1f}")
        print(f"  Newly dormant edges: {result.get('newly_dormant', 0)}")

    if not dry_run and result.get('applied_at'):
        print(f"  Applied at: {result['applied_at']}")

    return result


def initialize_weights(neo4j: Neo4jData) -> dict:
    """
    Initialize weights on edges that don't have them.

    Args:
        neo4j: Neo4jData instance

    Returns:
        Initialization statistics
    """
    print("\nInitializing edge weights...")

    result = neo4j.initialize_weights(default_weight=1.0)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return result

    print("\nResults:")
    print(f"  Edges initialized: {result.get('initialized', 0)}")
    print(f"  Default weight:    {result.get('default_weight', 1.0)}")

    return result


def show_stats(neo4j: Neo4jData):
    """Show current decay statistics."""
    print("\nCurrent Weight Statistics:")

    stats = neo4j.get_decay_stats()
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"  Total edges:        {stats['total_edges']}")
    print(f"  Average weight:     {stats['avg_weight']:.4f}")
    print(f"  Min weight:         {stats['min_weight']:.4f}")
    print(f"  Max weight:         {stats['max_weight']:.4f}")
    print(f"  Active edges:       {stats['active_count']} ({100 - stats['dormant_percentage']:.1f}%)")
    print(f"  Dormant edges:      {stats['dormant_count']} ({stats['dormant_percentage']:.1f}%)")
    print(f"  Saturated (w>=0.9): {stats['saturated_count']}")
    print(f"  With weight set:    {stats['edges_with_weight']}")
    print(f"  With timestamp:     {stats['edges_with_timestamp']}")

    # Show weight distribution
    print("\nWeight Distribution:")
    dist = neo4j.get_weight_distribution(buckets=5)
    total = stats['total_edges']
    for bucket in dist:
        bar_len = int(bucket['count'] / max(1, total) * 40) if total > 0 else 0
        bar = '#' * bar_len
        print(f"  [{bucket['range_start']:.1f}-{bucket['range_end']:.1f}): {bucket['count']:6d} {bar}")


def show_dynamics_info():
    """Show information about the weight dynamics parameters."""
    info = get_dynamics_info()

    print("\nWeight Dynamics Parameters:")
    for key, value in info['parameters'].items():
        print(f"  {key}: {value}")

    print("\nWeight Sources (initial weights):")
    for source, weight in info['weight_sources'].items():
        print(f"  {source}: {weight}")

    print("\nHalf-lives:")
    print(f"  Forgetting: {info['half_lives']['forgetting_days']:.1f} days")
    print(f"  Learning:   {info['half_lives']['learning_events']:.1f} events")

    print(f"\nKey insight: {info['key_insight']}")


def main():
    parser = argparse.ArgumentParser(
        description='Nightly decay job for FOLLOWS edge weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Apply 1 day of decay
  %(prog)s --days 7           # Apply 7 days of decay
  %(prog)s --dry-run          # Preview without applying
  %(prog)s --timestamp-based  # Decay based on last_used
  %(prog)s --init-weights     # Initialize edge weights
  %(prog)s --stats            # Show current statistics
  %(prog)s --info             # Show dynamics parameters
        """
    )

    parser.add_argument(
        '--days', '-d',
        type=float,
        default=1.0,
        help='Days of decay to apply (default: 1.0)'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview changes without applying'
    )

    parser.add_argument(
        '--timestamp-based', '-t',
        action='store_true',
        help='Use per-edge timestamps for decay calculation'
    )

    parser.add_argument(
        '--init-weights', '-i',
        action='store_true',
        help='Initialize weights on edges that do not have them'
    )

    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show current weight statistics'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show weight dynamics parameters'
    )

    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (for cron)'
    )

    args = parser.parse_args()

    # Show info and exit
    if args.info:
        show_dynamics_info()
        return 0

    # Connect to Neo4j
    if not args.quiet:
        print_header("Storm-Logos Nightly Decay Job")

    neo4j = Neo4jData()
    if not neo4j.connect():
        print("ERROR: Could not connect to Neo4j")
        print("  Ensure Neo4j is running at bolt://localhost:7687")
        return 1

    results = {}

    try:
        # Show stats mode
        if args.stats:
            show_stats(neo4j)
            neo4j.close()
            return 0

        # Initialize weights mode
        if args.init_weights:
            results = initialize_weights(neo4j)
            if args.json:
                print(json.dumps(results, indent=2))
            neo4j.close()
            return 0 if 'error' not in results else 1

        # Main decay operation
        if args.timestamp_based:
            results = run_timestamp_decay(neo4j, args.dry_run)
        else:
            results = run_simple_decay(neo4j, args.days, args.dry_run)

        # Show stats after decay
        if not args.quiet and not args.dry_run and 'error' not in results:
            show_stats(neo4j)

        if args.json:
            print(json.dumps(results, indent=2))

        # Log summary for cron
        if args.quiet and 'error' not in results:
            print(f"[{format_timestamp()}] Decay applied: "
                  f"{results.get('edges_affected', 0)} edges, "
                  f"decay={results.get('total_decay', 0):.4f}, "
                  f"newly_dormant={results.get('newly_dormant', 0)}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        neo4j.close()

    return 0 if 'error' not in results else 1


if __name__ == "__main__":
    sys.exit(main())
