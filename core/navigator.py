#!/usr/bin/env python3
"""
Navigator V2 - Corrected Compass
================================

Uses actual good→evil direction from data, not assumed [1,1,1,1,1].
"""

import numpy as np
import psycopg2
from dataclasses import dataclass
from typing import List, Dict, Optional

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

    def __repr__(self):
        return f"|{self.word}, τ={self.tau:.2f}, g={self.goodness:+.2f}⟩"


@dataclass
class Transition:
    verb: str
    from_state: NounState
    to_state: NounState
    delta_g: float  # goodness change


class SemanticSpace:
    def __init__(self):
        self.nouns: Dict[str, NounState] = {}
        self.j_good = None  # Will be computed from data
        self._load_from_db()

    def _load_from_db(self):
        print("Loading semantic space...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # First, get the good and evil vectors to find direction
        cur.execute("""
            SELECT word, j FROM hyp_semantic_index
            WHERE word IN ('good', 'evil', 'love', 'hate', 'beauty', 'ugly')
        """)
        special = {row[0]: np.array(row[1]) for row in cur.fetchall()}

        # Compute goodness direction from multiple pairs
        directions = []
        if 'good' in special and 'evil' in special:
            d = special['good'] - special['evil']
            directions.append(d / np.linalg.norm(d))
        if 'love' in special and 'hate' in special:
            d = special['love'] - special['hate']
            directions.append(d / np.linalg.norm(d))
        if 'beauty' in special and 'ugly' in special:
            d = special['beauty'] - special['ugly']
            directions.append(d / np.linalg.norm(d))

        if directions:
            # Average the directions
            self.j_good = np.mean(directions, axis=0)
            self.j_good = self.j_good / np.linalg.norm(self.j_good)
        else:
            self.j_good = np.array([1, 1, 1, 1, 1]) / np.sqrt(5)

        print(f"  Goodness direction: [{', '.join(f'{x:.2f}' for x in self.j_good)}]")

        # Now load all nouns
        cur.execute("""
            SELECT word, j, tau_entropy
            FROM hyp_semantic_index
            WHERE j IS NOT NULL AND tau_entropy IS NOT NULL AND tau_entropy > 0
        """)

        for word, j, tau in cur.fetchall():
            j_arr = np.array(j)
            goodness = float(np.dot(j_arr, self.j_good))
            self.nouns[word] = NounState(
                word=word,
                j=j_arr,
                tau=tau,
                goodness=goodness
            )

        conn.close()
        print(f"  Loaded {len(self.nouns)} nouns")

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
        print("\nLoading verb transitions...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Get semantically meaningful verbs
        cur.execute("""
            SELECT verb, object, SUM(total_count) as cnt
            FROM hyp_svo_triads
            WHERE total_count >= 10
              AND LENGTH(verb) >= 3
              AND LENGTH(object) >= 3
            GROUP BY verb, object
            HAVING SUM(total_count) >= 20
            ORDER BY verb, cnt DESC
        """)

        verb_objects = {}
        for verb, obj, cnt in cur.fetchall():
            if verb not in verb_objects:
                verb_objects[verb] = []
            if len(verb_objects[verb]) < 10 and obj in self.space.nouns:
                verb_objects[verb].append(obj)

        # Filter verbs that have at least 3 valid objects
        verb_objects = {v: objs for v, objs in verb_objects.items() if len(objs) >= 3}

        conn.close()
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
    print("NAVIGATOR V2 - Corrected Compass")
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
