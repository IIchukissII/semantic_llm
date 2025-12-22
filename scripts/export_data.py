#!/usr/bin/env python3
"""
Data Export Script
==================

Exports semantic data from PostgreSQL to JSON and CSV formats.

Exports:
    - word_vectors.json/csv: All 16D vectors (nouns, adjectives) with τ
    - verb_operators.json/csv: Verb 6D transition operators
    - entropy_stats.json/csv: Shannon entropy statistics (H_adj, H_verb)
    - spin_pairs.json/csv: Prefix spin operator pairs
    - bond_statistics.json/csv: Adjective-noun bond frequencies

Usage:
    python export_data.py --all           # Export everything
    python export_data.py --vectors       # Export word vectors only
    python export_data.py --entropy       # Export entropy stats only
"""

import argparse
import json
import csv
import numpy as np
import psycopg2
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}

OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Dimension names (from semantic_core.py)
J_DIMS = ['beauty', 'life', 'sacred', 'good', 'love']
I_DIMS = ['truth', 'freedom', 'meaning', 'order', 'peace',
          'power', 'nature', 'time', 'knowledge', 'self', 'society']
VERB_DIMS = ['beauty', 'life', 'sacred', 'good', 'love', 'truth']


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy H = -Σ p·log₂(p)"""
    if not counts:
        return 0.0
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return entropy


def normalized_entropy(counts: Dict[str, int]) -> float:
    """Normalized entropy H / H_max"""
    if not counts or len(counts) <= 1:
        return 0.0
    h = shannon_entropy(counts)
    h_max = np.log2(len(counts))
    return h / h_max if h_max > 0 else 0.0


def export_word_vectors():
    """Export word vectors from hyp_semantic_index."""
    print("Exporting word vectors...")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT word, word_type, j, i, tau, variety, verb, count, n_books
        FROM hyp_semantic_index
        ORDER BY count DESC
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("  No data in hyp_semantic_index. Run semantic pipeline first.")
        return

    # Prepare JSON structure
    data = {
        "exported_at": datetime.now().isoformat(),
        "description": "16D semantic word vectors with τ (entropy-based abstraction)",
        "dimensions": {
            "j_space": J_DIMS,
            "i_space": I_DIMS
        },
        "word_types": {0: "noun", 1: "verb", 2: "adjective", 3: "other"},
        "words": {}
    }

    # Prepare CSV rows
    csv_header = ['word', 'word_type', 'tau', 'variety', 'count', 'n_books'] + \
                 [f'j_{d}' for d in J_DIMS] + \
                 [f'i_{d}' for d in I_DIMS] + \
                 [f'verb_{d}' for d in VERB_DIMS]
    csv_rows = []

    for word, wtype, j, i, tau, variety, verb, count, n_books in rows:
        # JSON entry
        entry = {
            "word_type": wtype,
            "tau": tau,
            "variety": variety,
            "count": count,
            "n_books": n_books,
            "j": dict(zip(J_DIMS, j)) if j else None,
            "i": dict(zip(I_DIMS, i)) if i else None,
        }
        if verb:
            entry["verb"] = dict(zip(VERB_DIMS, verb))
        data["words"][word] = entry

        # CSV row
        row = [word, wtype, tau, variety, count, n_books]
        row.extend(j if j else [None]*5)
        row.extend(i if i else [None]*11)
        row.extend(verb if verb else [None]*6)
        csv_rows.append(row)

    # Write JSON
    json_path = OUTPUT_DIR / "json" / "word_vectors.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Written {len(data['words'])} words to {json_path}")

    # Write CSV
    csv_path = OUTPUT_DIR / "csv" / "word_vectors.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"  Written {len(csv_rows)} rows to {csv_path}")


def export_entropy_stats():
    """Export Shannon entropy statistics for nouns."""
    print("Exporting entropy statistics...")

    conn = get_connection()
    cur = conn.cursor()

    # Load adjective profiles
    cur.execute('''
        SELECT bond, total_count
        FROM hyp_bond_vocab
        WHERE total_count >= 2
    ''')

    noun_adj = defaultdict(lambda: defaultdict(int))
    for bond, count in cur.fetchall():
        parts = bond.split('|')
        if len(parts) == 2:
            adj, noun = parts
            noun_adj[noun][adj.lower()] += count

    # Load verb profiles
    cur.execute('''
        SELECT verb, object, SUM(total_count) as count
        FROM hyp_svo_triads
        WHERE total_count >= 1
        GROUP BY verb, object
    ''')

    noun_verb = defaultdict(lambda: defaultdict(int))
    for verb, noun, count in cur.fetchall():
        noun_verb[noun][verb] += count

    cur.close()
    conn.close()

    # Compute entropy profiles
    common_nouns = set(noun_adj.keys()) & set(noun_verb.keys())

    data = {
        "exported_at": datetime.now().isoformat(),
        "description": "Shannon entropy statistics for nouns",
        "theory": {
            "tau_formula": "τ = 1 + 5 × (1 - H_norm)",
            "one_bit_law": "H_adj - H_verb ≈ 1.08 bits",
            "euler_law": "ln(H_adj/H_verb) ≈ 1/e"
        },
        "nouns": {}
    }

    csv_header = ['noun', 'h_adj', 'h_verb', 'h_adj_norm', 'h_verb_norm',
                  'delta', 'ratio', 'tau_entropy', 'variety_adj', 'variety_verb']
    csv_rows = []

    for noun in common_nouns:
        adj_counts = noun_adj[noun]
        verb_counts = noun_verb[noun]

        h_adj = shannon_entropy(adj_counts)
        h_verb = shannon_entropy(verb_counts)
        h_adj_norm = normalized_entropy(adj_counts)
        h_verb_norm = normalized_entropy(verb_counts)

        delta = h_adj - h_verb
        ratio = h_adj / h_verb if h_verb > 0.1 else None
        tau_entropy = 1 + 5 * (1 - h_adj_norm)

        entry = {
            "h_adj": round(h_adj, 4),
            "h_verb": round(h_verb, 4),
            "h_adj_norm": round(h_adj_norm, 4),
            "h_verb_norm": round(h_verb_norm, 4),
            "delta": round(delta, 4),
            "ratio": round(ratio, 4) if ratio else None,
            "tau_entropy": round(tau_entropy, 2),
            "variety_adj": len(adj_counts),
            "variety_verb": len(verb_counts)
        }
        data["nouns"][noun] = entry

        csv_rows.append([
            noun, h_adj, h_verb, h_adj_norm, h_verb_norm,
            delta, ratio, tau_entropy, len(adj_counts), len(verb_counts)
        ])

    # Compute aggregate statistics
    deltas = [data["nouns"][n]["delta"] for n in data["nouns"]]
    ratios = [data["nouns"][n]["ratio"] for n in data["nouns"] if data["nouns"][n]["ratio"]]

    data["aggregate"] = {
        "n_nouns": len(common_nouns),
        "mean_delta": float(round(np.mean(deltas), 4)),
        "mean_ratio": float(round(np.mean(ratios), 4)),
        "one_bit_law_confirmed": bool(abs(np.mean(deltas) - 1.0) < 0.2)
    }

    # Write JSON
    json_path = OUTPUT_DIR / "json" / "entropy_stats.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Written {len(data['nouns'])} nouns to {json_path}")

    # Write CSV
    csv_path = OUTPUT_DIR / "csv" / "entropy_stats.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"  Written {len(csv_rows)} rows to {csv_path}")


def export_verb_operators():
    """Export verb transition operators."""
    print("Exporting verb operators...")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT word, verb, count, n_books
        FROM hyp_semantic_index
        WHERE word_type = 1 AND verb IS NOT NULL
        ORDER BY count DESC
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("  No verb data found.")
        return

    data = {
        "exported_at": datetime.now().isoformat(),
        "description": "Verb 6D transition operators",
        "dimensions": VERB_DIMS,
        "verbs": {}
    }

    csv_header = ['verb', 'count', 'n_books'] + [f'{d}' for d in VERB_DIMS] + ['magnitude']
    csv_rows = []

    for word, verb, count, n_books in rows:
        if verb:
            magnitude = float(np.linalg.norm(verb))
            data["verbs"][word] = {
                "vector": dict(zip(VERB_DIMS, verb)),
                "magnitude": round(magnitude, 4),
                "count": count,
                "n_books": n_books
            }

            row = [word, count, n_books] + list(verb) + [magnitude]
            csv_rows.append(row)

    # Write JSON
    json_path = OUTPUT_DIR / "json" / "verb_operators.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Written {len(data['verbs'])} verbs to {json_path}")

    # Write CSV
    csv_path = OUTPUT_DIR / "csv" / "verb_operators.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"  Written {len(csv_rows)} rows to {csv_path}")


def export_spin_pairs():
    """Export spin (prefix) pairs from existing analysis."""
    print("Exporting spin pairs...")

    # Try to load from existing spin_prefix_results.json
    spin_file = Path(__file__).parent.parent / "archive" / "phase4_architecture" / "spin_prefix_results.json"

    if not spin_file.exists():
        print(f"  Spin results not found at {spin_file}")
        return

    with open(spin_file) as f:
        spin_data = json.load(f)

    # Extract pairs from tau_conservation array
    raw_pairs = spin_data.get("tau_conservation", spin_data.get("pairs", []))

    # Restructure
    pairs = []
    for pair in raw_pairs:
        delta_tau = pair.get("delta_tau", 0)
        tau_conserved = abs(delta_tau) < 0.5  # τ conserved if change < 0.5
        pairs.append({
            "base": pair.get("base"),
            "prefixed": pair.get("prefixed"),
            "delta_tau": delta_tau,
            "tau_conserved": tau_conserved
        })

    data = {
        "exported_at": datetime.now().isoformat(),
        "description": "Prefix spin operator pairs (τ-conserving direction flips)",
        "theory": {
            "tau_conservation": "Prefixes conserve τ (abstraction level)",
            "direction_flip": "Prefixes flip direction in j-space (semantic conjugation)"
        },
        "pairs": pairs
    }

    csv_header = ['base', 'prefixed', 'delta_tau', 'tau_conserved']
    csv_rows = []

    for pair in pairs:
        csv_rows.append([
            pair.get("base"),
            pair.get("prefixed"),
            pair.get("delta_tau"),
            pair.get("tau_conserved")
        ])

    # Write JSON
    json_path = OUTPUT_DIR / "json" / "spin_pairs.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Written {len(data['pairs'])} pairs to {json_path}")

    # Write CSV
    csv_path = OUTPUT_DIR / "csv" / "spin_pairs.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"  Written {len(csv_rows)} rows to {csv_path}")


def export_bond_statistics():
    """Export adjective-noun bond frequency statistics."""
    print("Exporting bond statistics...")

    conn = get_connection()
    cur = conn.cursor()

    # Get top bonds by frequency
    cur.execute('''
        SELECT bond, total_count
        FROM hyp_bond_vocab
        WHERE total_count >= 10
        ORDER BY total_count DESC
        LIMIT 10000
    ''')

    rows = cur.fetchall()
    cur.close()
    conn.close()

    data = {
        "exported_at": datetime.now().isoformat(),
        "description": "Top adjective|noun bonds by frequency",
        "bonds": []
    }

    csv_header = ['adj', 'noun', 'count']
    csv_rows = []

    for bond, count in rows:
        parts = bond.split('|')
        if len(parts) == 2:
            adj, noun = parts
            data["bonds"].append({
                "adj": adj,
                "noun": noun,
                "count": count
            })
            csv_rows.append([adj, noun, count])

    # Write JSON
    json_path = OUTPUT_DIR / "json" / "bond_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Written {len(data['bonds'])} bonds to {json_path}")

    # Write CSV
    csv_path = OUTPUT_DIR / "csv" / "bond_statistics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"  Written {len(csv_rows)} rows to {csv_path}")


def export_noun_adj_profiles():
    """
    Export noun adjective profiles for NounCloud construction.

    Each noun's profile contains:
    - adj_profile: {adj: weight} (top adjectives with probabilities)
    - variety: number of distinct adjectives
    - h_adj: Shannon entropy
    - h_adj_norm: normalized entropy [0,1]
    - tau: derived abstraction level (1 + 5*(1-h_adj_norm))
    - j_centroid: 5D j-space centroid
    - i_centroid: 11D i-space centroid
    """
    print("Exporting noun adjective profiles...")

    conn = get_connection()
    cur = conn.cursor()

    # Load all adjective-noun bonds
    print("  Loading adjective-noun bonds...")
    cur.execute('''
        SELECT bond, total_count
        FROM hyp_bond_vocab
        WHERE total_count >= 2
        ORDER BY total_count DESC
    ''')

    noun_adj = defaultdict(lambda: defaultdict(int))
    for bond, count in cur.fetchall():
        parts = bond.split('|')
        if len(parts) == 2:
            adj, noun = parts
            noun_adj[noun][adj.lower()] += count

    # Load adjective vectors for centroid computation
    # Note: Adjectives may not be classified as word_type=2, so we get vectors
    # for all words that appear as adjectives in the bond vocabulary
    print("  Loading adjective vectors...")

    # First, get unique adjectives from bond vocabulary
    unique_adjs = set()
    for adj_counts in noun_adj.values():
        unique_adjs.update(adj_counts.keys())

    # Then get vectors for those adjectives
    cur.execute("""
        SELECT word, j, i
        FROM hyp_semantic_index
        WHERE j IS NOT NULL
    """)

    adj_vectors = {}
    for word, j, i in cur.fetchall():
        if j and i and word in unique_adjs:  # Only keep words that appear as adjectives
            adj_vectors[word] = {
                'j': np.array(j),
                'i': np.array(i)
            }

    cur.close()
    conn.close()

    print(f"  Found {len(noun_adj)} nouns, {len(adj_vectors)} adjective vectors")

    # Build profiles
    data = {
        "exported_at": datetime.now().isoformat(),
        "description": "Noun adjective profiles for NounCloud (projections of projections)",
        "theory": {
            "principle": "Nouns are clouds of adjectives, not direct 16D vectors",
            "tau_formula": "τ = 1 + 5 × (1 - H_norm)",
            "centroid": "j/i centroids are weighted sums of adjective vectors"
        },
        "dimensions": {
            "j_space": J_DIMS,
            "i_space": I_DIMS
        },
        "nouns": {}
    }

    # CSV for raw profiles
    csv_header = ['noun', 'variety', 'h_adj', 'h_adj_norm', 'tau', 'total_count',
                  'top_adjs', 'is_cloud'] + \
                 [f'j_{d}' for d in J_DIMS] + [f'i_{d}' for d in I_DIMS]
    csv_rows = []

    min_adjectives = 5  # Minimum for cloud construction

    for noun, adj_counts in noun_adj.items():
        variety = len(adj_counts)
        total_count = sum(adj_counts.values())

        # Compute entropy
        h_adj = shannon_entropy(adj_counts)
        h_adj_norm = normalized_entropy(adj_counts)
        tau = 1 + 5 * (1 - h_adj_norm)

        # Normalize to probabilities (keep top 100)
        top_adjs = sorted(adj_counts.items(), key=lambda x: -x[1])[:100]
        probs = {adj: count / total_count for adj, count in top_adjs}

        # Compute centroids
        j_centroid = np.zeros(5)
        i_centroid = np.zeros(11)
        weight_sum = 0.0

        for adj, weight in probs.items():
            if adj in adj_vectors:
                j_centroid += weight * adj_vectors[adj]['j']
                i_centroid += weight * adj_vectors[adj]['i']
                weight_sum += weight

        if weight_sum > 0:
            j_centroid /= weight_sum
            i_centroid /= weight_sum

        is_cloud = variety >= min_adjectives

        # JSON entry
        entry = {
            "adj_profile": {adj: round(p, 6) for adj, p in probs.items()},
            "variety": variety,
            "h_adj": round(h_adj, 4),
            "h_adj_norm": round(h_adj_norm, 4),
            "tau": round(tau, 2),
            "total_count": total_count,
            "is_cloud": is_cloud,
            "j_centroid": dict(zip(J_DIMS, [round(x, 6) for x in j_centroid])),
            "i_centroid": dict(zip(I_DIMS, [round(x, 6) for x in i_centroid]))
        }
        data["nouns"][noun] = entry

        # CSV row (top 5 adjectives as string)
        top5_str = ','.join(f"{adj}:{p:.3f}" for adj, p in list(probs.items())[:5])
        row = [noun, variety, round(h_adj, 4), round(h_adj_norm, 4), round(tau, 2),
               total_count, top5_str, is_cloud]
        row.extend([round(x, 6) for x in j_centroid])
        row.extend([round(x, 6) for x in i_centroid])
        csv_rows.append(row)

    # Statistics
    cloud_count = sum(1 for n, d in data["nouns"].items() if d["is_cloud"])
    fallback_count = len(data["nouns"]) - cloud_count

    data["statistics"] = {
        "total_nouns": len(data["nouns"]),
        "cloud_nouns": cloud_count,
        "fallback_nouns": fallback_count,
        "min_adjectives_for_cloud": min_adjectives,
        "mean_variety": round(np.mean([d["variety"] for d in data["nouns"].values()]), 1),
        "mean_tau": round(np.mean([d["tau"] for d in data["nouns"].values()]), 2)
    }

    # Write JSON
    json_path = OUTPUT_DIR / "json" / "noun_adj_profiles.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Written {len(data['nouns'])} noun profiles to {json_path}")
    print(f"    Cloud nouns: {cloud_count}, Fallback nouns: {fallback_count}")

    # Write CSV
    csv_path = OUTPUT_DIR / "csv" / "noun_adj_profiles.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"  Written {len(csv_rows)} rows to {csv_path}")


def export_svo_triads():
    """Export subject-verb-object triads for navigation."""
    print("Exporting SVO triads...")

    conn = get_connection()
    cur = conn.cursor()

    # Get verb-object pairs (aggregated across subjects)
    cur.execute('''
        SELECT s.verb, s.object, SUM(s.total_count) as cnt
        FROM hyp_svo_triads s
        JOIN hyp_semantic_index i ON s.object = i.word
        WHERE s.total_count >= 2
          AND LENGTH(s.verb) >= 2
          AND LENGTH(s.object) >= 2
          AND i.j IS NOT NULL
        GROUP BY s.verb, s.object
        HAVING SUM(s.total_count) >= 3
        ORDER BY s.verb, cnt DESC
    ''')

    verb_objects = cur.fetchall()

    # Get subject-specific patterns
    cur.execute('''
        SELECT subject, verb, object, SUM(total_count) as cnt
        FROM hyp_svo_triads
        WHERE total_count >= 5 AND LENGTH(verb) >= 3 AND LENGTH(object) >= 3
        GROUP BY subject, verb, object
        HAVING SUM(total_count) >= 10
        ORDER BY subject, cnt DESC
    ''')

    svo_patterns = cur.fetchall()

    cur.close()
    conn.close()

    # Write verb-object CSV
    csv_path = OUTPUT_DIR / "csv" / "verb_objects.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['verb', 'object', 'count'])
        writer.writerows(verb_objects)
    print(f"  Written {len(verb_objects)} verb-object pairs to {csv_path}")

    # Write SVO patterns CSV
    csv_path = OUTPUT_DIR / "csv" / "svo_patterns.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'verb', 'object', 'count'])
        writer.writerows(svo_patterns)
    print(f"  Written {len(svo_patterns)} SVO patterns to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Export semantic data to JSON/CSV")
    parser.add_argument('--all', action='store_true', help='Export everything')
    parser.add_argument('--vectors', action='store_true', help='Export word vectors')
    parser.add_argument('--entropy', action='store_true', help='Export entropy statistics')
    parser.add_argument('--verbs', action='store_true', help='Export verb operators')
    parser.add_argument('--spin', action='store_true', help='Export spin pairs')
    parser.add_argument('--bonds', action='store_true', help='Export bond statistics')
    parser.add_argument('--svo', action='store_true', help='Export SVO triads')
    parser.add_argument('--profiles', action='store_true', help='Export noun adjective profiles (NounCloud)')
    args = parser.parse_args()

    # Ensure output directories exist
    (OUTPUT_DIR / "json").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SEMANTIC DATA EXPORT")
    print("=" * 60)

    if args.all or not any([args.vectors, args.entropy, args.verbs, args.spin, args.bonds, args.svo, args.profiles]):
        export_word_vectors()
        export_entropy_stats()
        export_verb_operators()
        export_spin_pairs()
        export_bond_statistics()
        export_svo_triads()
        export_noun_adj_profiles()
    else:
        if args.vectors:
            export_word_vectors()
        if args.entropy:
            export_entropy_stats()
        if args.verbs:
            export_verb_operators()
        if args.spin:
            export_spin_pairs()
        if args.bonds:
            export_bond_statistics()
        if args.svo:
            export_svo_triads()
        if args.profiles:
            export_noun_adj_profiles()

    print("\nExport complete!")


if __name__ == "__main__":
    main()
