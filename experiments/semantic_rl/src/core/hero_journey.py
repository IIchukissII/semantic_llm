"""
Hero Journey Extractor: Extract the protagonist's path through semantic space.

The book is a UNIVERSE (map of all concepts).
The hero is a MOVING POSITION through that space.

Key insight: The hero's position is NOT individual words, but the
CENTROID of recent semantic content - a moving window through j-space.

"Only believe what was lived is knowledge"
"""

import re
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM_PATH = _THIS_FILE.parent.parent.parent.parent.parent
if str(_SEMANTIC_LLM_PATH) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_LLM_PATH))


@dataclass
class SemanticNode:
    """A node in the book's semantic universe."""
    word: str
    goodness: float
    tau: float
    j: np.ndarray
    occurrences: int = 0

    def __hash__(self):
        return hash(self.word)


@dataclass
class HeroPosition:
    """Hero's position at a point in the narrative."""
    position: float          # 0-1 in book
    goodness: float          # g of current state
    tau: float              # τ of current state
    j: np.ndarray           # 5D position in j-space
    words: List[str]        # words contributing to this position
    context: str = ""       # text snippet


@dataclass
class BookUniverse:
    """The semantic universe of a book - all concepts as a map."""
    title: str
    nodes: Dict[str, SemanticNode]

    def centroid(self, words: List[str]) -> Tuple[float, float, np.ndarray]:
        """Compute centroid (g, τ, j) of a set of words."""
        valid = [self.nodes[w] for w in words if w in self.nodes]
        if not valid:
            return 0.0, 2.0, np.zeros(5)

        g = np.mean([n.goodness for n in valid])
        tau = np.mean([n.tau for n in valid])
        j = np.mean([n.j for n in valid], axis=0)
        return g, tau, j


@dataclass
class HeroPath:
    """The hero's trajectory through semantic space."""
    positions: List[HeroPosition]
    universe: BookUniverse

    @property
    def goodness_arc(self) -> np.ndarray:
        return np.array([p.goodness for p in self.positions])

    @property
    def tau_arc(self) -> np.ndarray:
        return np.array([p.tau for p in self.positions])

    def delta_g(self) -> float:
        if len(self.positions) < 2:
            return 0
        return self.positions[-1].goodness - self.positions[0].goodness


class HeroJourneyExtractor:
    """
    Extract hero's journey as a trajectory through semantic space.

    The hero's position at any point is the CENTROID of recent
    semantic content - not individual word jumps.
    """

    def __init__(self):
        self.core = None
        self._load_core()

    def _load_core(self):
        """Load QuantumCore."""
        try:
            import importlib.util

            hybrid_llm_path = _SEMANTIC_LLM_PATH / "core" / "hybrid_llm.py"
            data_loader_path = _SEMANTIC_LLM_PATH / "core" / "data_loader.py"

            spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
            data_loader_module = importlib.util.module_from_spec(spec)
            sys.modules['core.data_loader'] = data_loader_module
            spec.loader.exec_module(data_loader_module)

            spec = importlib.util.spec_from_file_location("hybrid_llm", hybrid_llm_path)
            hybrid_llm_module = importlib.util.module_from_spec(spec)
            sys.modules['core.hybrid_llm'] = hybrid_llm_module
            spec.loader.exec_module(hybrid_llm_module)

            print("Loading QuantumCore...")
            self.core = hybrid_llm_module.QuantumCore()
            print(f"  Loaded {len(self.core.states)} states")

        except Exception as e:
            print(f"Warning: Could not load QuantumCore: {e}")
            self.core = None

    def load_book(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def build_universe(self, text: str, title: str = "Book") -> BookUniverse:
        """
        Build universe: ALL words in book that exist in semantic space.
        No filtering - the universe is the complete map.
        """
        if self.core is None:
            return BookUniverse(title=title, nodes={})

        print(f"\nBuilding universe for '{title}'...")

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        counts = Counter(words)

        nodes = {}
        for word, count in counts.items():
            state = self.core.states.get(word)
            if state:
                nodes[word] = SemanticNode(
                    word=word,
                    goodness=state.goodness,
                    tau=state.tau,
                    j=state.j,
                    occurrences=count
                )

        print(f"  Universe: {len(nodes)} concepts")
        return BookUniverse(title=title, nodes=nodes)

    def extract_hero_path(self,
                          text: str,
                          universe: BookUniverse,
                          window_words: int = 100,
                          step_words: int = 50) -> HeroPath:
        """
        Extract hero's path as moving centroid through semantic space.

        The hero's position at each point is the CENTROID of the
        surrounding window of text - a smooth trajectory through j-space.

        Args:
            text: Book text
            universe: The semantic universe
            window_words: Size of moving window (words)
            step_words: Step size between positions
        """
        if self.core is None or not universe.nodes:
            return HeroPath(positions=[], universe=universe)

        print(f"\nExtracting hero trajectory (window={window_words}, step={step_words})...")

        # Skip header/footer
        text_len = len(text)
        text = text[int(text_len * 0.05):int(text_len * 0.95)]

        # Tokenize
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        total_words = len(words)

        positions = []

        for i in range(0, total_words - window_words, step_words):
            window = words[i:i + window_words]
            position = i / total_words

            # Get semantic words in window
            sem_words = [w for w in window if w in universe.nodes]

            if not sem_words:
                continue

            # Compute centroid
            g, tau, j = universe.centroid(sem_words)

            # Context: first few words
            context = " ".join(window[:15]) + "..."

            positions.append(HeroPosition(
                position=position,
                goodness=g,
                tau=tau,
                j=j,
                words=list(set(sem_words)),
                context=context
            ))

        print(f"  Trajectory: {len(positions)} positions")
        return HeroPath(positions=positions, universe=universe)

    def extract_journey(self, filepath: str, title: str = None,
                        window: int = 100, step: int = 50) -> Tuple[BookUniverse, HeroPath]:
        """Extract complete journey."""
        if title is None:
            title = Path(filepath).stem
            if " - " in title:
                title = title.split(" - ", 1)[1]

        text = self.load_book(filepath)
        universe = self.build_universe(text, title)
        path = self.extract_hero_path(text, universe, window, step)

        return universe, path


def visualize_journey(universe: BookUniverse, path: HeroPath, save_path: str = None):
    """Visualize the hero's trajectory through semantic space."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    positions = [p.position * 100 for p in path.positions]
    g_arc = path.goodness_arc
    tau_arc = path.tau_arc

    # 1. Goodness arc (the main narrative arc)
    ax1 = axes[0, 0]
    ax1.fill_between(positions, g_arc, alpha=0.3, color='green')
    ax1.plot(positions, g_arc, '-', color='darkgreen', lw=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Position in book (%)')
    ax1.set_ylabel('Goodness (g)')
    ax1.set_title(f"Hero's Goodness Arc - {universe.title}")
    ax1.grid(True, alpha=0.3)

    # Mark key points
    min_idx = np.argmin(g_arc)
    max_idx = np.argmax(g_arc)
    ax1.scatter([positions[min_idx]], [g_arc[min_idx]], s=100, c='red', zorder=5, label=f'Nadir ({g_arc[min_idx]:.2f})')
    ax1.scatter([positions[max_idx]], [g_arc[max_idx]], s=100, c='gold', zorder=5, label=f'Zenith ({g_arc[max_idx]:.2f})')
    ax1.legend()

    # 2. Tau arc (abstraction level)
    ax2 = axes[0, 1]
    ax2.plot(positions, tau_arc, '-', color='blue', lw=2, alpha=0.7)
    ax2.set_xlabel('Position in book (%)')
    ax2.set_ylabel('Abstraction (τ)')
    ax2.set_title("Hero's Abstraction Level")
    ax2.grid(True, alpha=0.3)

    # 3. g-τ phase space trajectory
    ax3 = axes[1, 0]

    # Plot universe as background
    node_g = [n.goodness for n in universe.nodes.values()]
    node_tau = [n.tau for n in universe.nodes.values()]
    ax3.scatter(node_g, node_tau, s=5, alpha=0.1, c='gray')

    # Plot trajectory with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(g_arc)))
    for i in range(len(g_arc) - 1):
        ax3.plot(g_arc[i:i+2], tau_arc[i:i+2], '-', color=colors[i], lw=2, alpha=0.7)

    ax3.scatter([g_arc[0]], [tau_arc[0]], s=200, c='cyan', marker='o',
               edgecolors='black', linewidths=2, zorder=10, label='Start')
    ax3.scatter([g_arc[-1]], [tau_arc[-1]], s=200, c='gold', marker='*',
               edgecolors='black', linewidths=2, zorder=10, label='End')

    ax3.set_xlabel('Goodness (g)')
    ax3.set_ylabel('Abstraction (τ)')
    ax3.set_title('Hero Trajectory in g-τ Space')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
    {universe.title}
    {'='*40}

    Universe: {len(universe.nodes)} concepts

    Hero's Journey:
      Positions: {len(path.positions)}
      Δg (start→end): {path.delta_g():+.3f}

      Start g: {g_arc[0]:+.3f}
      End g:   {g_arc[-1]:+.3f}
      Min g:   {np.min(g_arc):+.3f} at {positions[min_idx]:.0f}%
      Max g:   {np.max(g_arc):+.3f} at {positions[max_idx]:.0f}%

      g variance: {np.var(g_arc):.4f}
      τ mean: {np.mean(tau_arc):.2f}

    Arc Classification:
      {'DESCENT-ASCENT (U-shape)' if min_idx > 10 and min_idx < len(g_arc) - 10 else
       'ASCENT' if g_arc[-1] > g_arc[0] else 'DESCENT'}
    """

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract hero's journey")
    parser.add_argument("book", help="Path to book")
    parser.add_argument("--window", type=int, default=100, help="Window size")
    parser.add_argument("--step", type=int, default=50, help="Step size")
    parser.add_argument("--plot", action="store_true", help="Plot")

    args = parser.parse_args()

    extractor = HeroJourneyExtractor()
    universe, path = extractor.extract_journey(args.book, window=args.window, step=args.step)

    print("\n" + "=" * 60)
    print(f"Universe: {len(universe.nodes)} concepts")
    print(f"Path: {len(path.positions)} positions")
    print(f"Δg: {path.delta_g():+.3f}")

    if args.plot:
        out = f"visualizations/{universe.title.replace(' ', '_')}_trajectory.png"
        visualize_journey(universe, path, save_path=out)
