#!/usr/bin/env python3
"""
Navigator V2 - Corrected Compass with NounCloud Support
========================================================

Uses actual good→evil direction from data, not assumed [1,1,1,1,1].
Now uses NounCloud (projections of projections) for theory-consistent nouns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

# Import DataLoader and NounCloud
from core.data_loader import DataLoader, NounCloud

# Optional: database fallback
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

DB_CONFIG = {
    "dbname": "bonds",
    "user": "bonds",
    "password": "bonds_secret",
    "host": "localhost",
    "port": 5432
}


@dataclass
class NounState:
    word: str
    j: np.ndarray
    tau: float
    goodness: float  # projection onto good direction

    # NounCloud metadata
    is_cloud: bool = False  # True if derived from adjective cloud
    variety: int = 0        # Number of adjectives (if is_cloud)
    h_adj_norm: float = 0.0 # Normalized entropy

    def __repr__(self):
        cloud_marker = "☁" if self.is_cloud else ""
        return f"|{self.word}{cloud_marker}, τ={self.tau:.2f}, g={self.goodness:+.2f}⟩"


@dataclass
class Transition:
    verb: str
    from_state: NounState
    to_state: NounState
    delta_g: float  # goodness change


class SemanticSpace:
    """
    Semantic space using NounCloud (projections of projections).

    Uses DataLoader to load data from CSV/JSON files or database.
    Nouns are loaded as NounCloud (adjective centroids with derived τ).
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.loader = data_loader or DataLoader()
        self.nouns: Dict[str, NounState] = {}
        self.j_good = None  # Will be computed from data
        self._load_space()

    def _load_space(self):
        """Load semantic space using NounCloud (projections of projections)."""
        print("Loading semantic space with NounCloud support...")

        vectors = self.loader.load_word_vectors()
        j_dims = ['beauty', 'life', 'sacred', 'good', 'love']

        # Load NounClouds first (to compute goodness from cloud centroids)
        noun_clouds = self.loader.load_noun_clouds()

        # Compute goodness direction from NounCloud centroids (not raw adjective vectors)
        # This ensures consistency: both goodness direction and noun positions
        # are in the same "cloud centroid" space
        special = {}
        for word in ['good', 'evil', 'love', 'hate', 'peace', 'war']:
            if word in noun_clouds:
                special[word] = noun_clouds[word].j

        # Compute goodness direction from cloud-based pairs
        directions = []
        for pos, neg in [('good', 'evil'), ('love', 'hate'), ('peace', 'war')]:
            if pos in special and neg in special:
                d = special[pos] - special[neg]
                norm = np.linalg.norm(d)
                if norm > 0:
                    directions.append(d / norm)

        if directions:
            # Average the directions
            self.j_good = np.mean(directions, axis=0)
            self.j_good = self.j_good / np.linalg.norm(self.j_good)
        else:
            self.j_good = np.array([1, 1, 1, 1, 1]) / np.sqrt(5)

        print(f"  Goodness direction (cloud-based): [{', '.join(f'{x:.2f}' for x in self.j_good)}]")
        cloud_count = 0
        fallback_count = 0

        # Load nouns from NounClouds
        for word, cloud in noun_clouds.items():
            j_arr = cloud.j
            goodness = float(np.dot(j_arr, self.j_good))
            self.nouns[word] = NounState(
                word=word,
                j=j_arr,
                tau=cloud.tau,
                goodness=goodness,
                is_cloud=cloud.is_cloud,
                variety=cloud.variety,
                h_adj_norm=cloud.h_adj_norm
            )
            if cloud.is_cloud:
                cloud_count += 1
            else:
                fallback_count += 1

        # Add remaining words from vectors
        for word, v in vectors.items():
            if word not in self.nouns and v.get('j') and v.get('tau') and v['tau'] > 0:
                j_arr = np.array([v['j'].get(d, 0) for d in j_dims])
                goodness = float(np.dot(j_arr, self.j_good))
                self.nouns[word] = NounState(
                    word=word, j=j_arr, tau=v['tau'], goodness=goodness,
                    is_cloud=False
                )

        print(f"  Loaded {len(self.nouns)} nouns")
        print(f"    NounCloud (theory-consistent): {cloud_count}")
        print(f"    Fallback (sparse): {fallback_count}")

        # Show calibration
        print("\n  Goodness calibration:")
        for w in ['good', 'evil', 'love', 'hate', 'beauty', 'ugly', 'life', 'death', 'god', 'devil']:
            if w in self.nouns:
                print(f"    {w:<10}: g={self.nouns[w].goodness:+.2f}")

    def get_state(self, word: str) -> Optional[NounState]:
        return self.nouns.get(word)


class Navigator:
    def __init__(self, space: SemanticSpace):
        self.space = space
        self.verb_objects = self._load_verbs()

    def _load_verbs(self) -> Dict[str, List[str]]:
        """Load verb transitions using DataLoader."""
        print("\nLoading verb transitions...")

        raw_verb_objects = self.space.loader.load_verb_objects()

        # Filter to verbs with valid objects (present in semantic space)
        verb_objects = {}
        for verb, objs in raw_verb_objects.items():
            valid_objs = [obj for obj in objs if obj in self.space.nouns]
            if len(valid_objs) >= 3:  # At least 3 valid objects
                verb_objects[verb] = valid_objs[:10]

        print(f"  Loaded {len(verb_objects)} meaningful verbs")
        return verb_objects

    def get_transitions(self, state: NounState, top_k: int = 20) -> List[Transition]:
        """Get transitions sorted by goodness change."""
        transitions = []

        for verb, objects in self.verb_objects.items():
            for obj in objects:
                to_state = self.space.get_state(obj)
                if to_state and to_state.word != state.word:
                    delta_g = to_state.goodness - state.goodness
                    transitions.append(Transition(
                        verb=verb,
                        from_state=state,
                        to_state=to_state,
                        delta_g=delta_g
                    ))

        return transitions

    def toward_good(self, state: NounState, n: int = 5) -> List[Transition]:
        """Get transitions that maximize goodness."""
        trans = self.get_transitions(state)
        return sorted(trans, key=lambda t: t.delta_g, reverse=True)[:n]

    def toward_evil(self, state: NounState, n: int = 5) -> List[Transition]:
        """Get transitions that minimize goodness."""
        trans = self.get_transitions(state)
        return sorted(trans, key=lambda t: t.delta_g)[:n]

    def navigate(self, start: str, goal: str = "good", steps: int = 3) -> List[Transition]:
        """Navigate multi-step trajectory."""
        state = self.space.get_state(start)
        if not state:
            return []

        trajectory = []
        visited = {start}

        for _ in range(steps):
            if goal == "good":
                options = self.toward_good(state)
            else:
                options = self.toward_evil(state)

            # Filter visited
            options = [t for t in options if t.to_state.word not in visited]

            if not options:
                break

            best = options[0]
            trajectory.append(best)
            visited.add(best.to_state.word)
            state = best.to_state

        return trajectory


def demo():
    print("=" * 70)
    print("NAVIGATOR V2 - NounCloud Support (Projections of Projections)")
    print("=" * 70)

    space = SemanticSpace()
    nav = Navigator(space)

    # Test navigation
    print("\n" + "=" * 70)
    print("NAVIGATION TOWARD GOOD")
    print("=" * 70)

    for start in ['war', 'death', 'hate', 'evil', 'enemy']:
        state = space.get_state(start)
        if not state:
            continue

        print(f"\n--- From: {state} ---")
        traj = nav.navigate(start, "good", 3)

        for i, t in enumerate(traj):
            print(f"  {i+1}. [{t.verb}] → {t.to_state} (Δg={t.delta_g:+.2f})")

    print("\n" + "=" * 70)
    print("NAVIGATION TOWARD EVIL")
    print("=" * 70)

    for start in ['love', 'peace', 'good', 'friend', 'beauty']:
        state = space.get_state(start)
        if not state:
            continue

        print(f"\n--- From: {state} ---")
        traj = nav.navigate(start, "evil", 3)

        for i, t in enumerate(traj):
            print(f"  {i+1}. [{t.verb}] → {t.to_state} (Δg={t.delta_g:+.2f})")

    # Show verb semantics
    print("\n" + "=" * 70)
    print("VERB COMPASS - From 'man'")
    print("=" * 70)

    state = space.get_state('man')
    if state:
        print(f"\nFrom: {state}")

        print("\nTop 10 GOOD transitions:")
        for t in nav.toward_good(state, 10):
            print(f"  {t.verb:<12} → {t.to_state.word:<12} Δg={t.delta_g:+.3f}")

        print("\nTop 10 EVIL transitions:")
        for t in nav.toward_evil(state, 10):
            print(f"  {t.verb:<12} → {t.to_state.word:<12} Δg={t.delta_g:+.3f}")


if __name__ == "__main__":
    demo()
