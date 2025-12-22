"""
Book Path Extractor: Extract the hero's path from a book.

Simple principle:
- Book = region in semantic space
- Hero's path = sequence of concepts in reading order
- The path is ALREADY in the book - we just extract it

The semantic-LLM observes these paths as experience.
"""

import re
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator

_THIS_FILE = Path(__file__).resolve()
_SEMANTIC_LLM_PATH = _THIS_FILE.parent.parent.parent.parent.parent
if str(_SEMANTIC_LLM_PATH) not in sys.path:
    sys.path.insert(0, str(_SEMANTIC_LLM_PATH))


@dataclass
class PathPoint:
    """A point on the hero's path."""
    word: str
    position: int       # word index in book
    goodness: float
    tau: float
    j: np.ndarray


@dataclass
class BookRegion:
    """A region in semantic space defined by a book."""
    title: str
    concepts: Dict[str, 'PathPoint']  # unique concepts in this region

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Bounding box of this region in g-τ space."""
        if not self.concepts:
            return {}
        g_vals = [c.goodness for c in self.concepts.values()]
        tau_vals = [c.tau for c in self.concepts.values()]
        return {
            'g': (min(g_vals), max(g_vals)),
            'tau': (min(tau_vals), max(tau_vals))
        }


@dataclass
class HeroPath:
    """The hero's path through a book's region."""
    region: BookRegion
    path: List[PathPoint]  # sequence in reading order

    def __len__(self):
        return len(self.path)

    def __iter__(self) -> Iterator[PathPoint]:
        return iter(self.path)

    @property
    def g_sequence(self) -> np.ndarray:
        """Goodness sequence."""
        return np.array([p.goodness for p in self.path])

    @property
    def tau_sequence(self) -> np.ndarray:
        """Tau sequence."""
        return np.array([p.tau for p in self.path])

    @property
    def j_sequence(self) -> np.ndarray:
        """J-vector sequence (N x 5)."""
        return np.array([p.j for p in self.path])

    def delta_g(self, window: int = 1) -> np.ndarray:
        """Change in goodness at each step."""
        g = self.g_sequence
        return np.diff(g, n=window)

    def smoothed_g(self, window: int = 50) -> np.ndarray:
        """Smoothed goodness trajectory."""
        g = self.g_sequence
        if len(g) < window:
            return g
        kernel = np.ones(window) / window
        return np.convolve(g, kernel, mode='valid')


class BookPathExtractor:
    """
    Extract hero's path from books.

    The path is simply the sequence of semantic concepts
    in reading order. No filtering, no averaging.
    """

    def __init__(self):
        self.core = None
        self._load_core()

    def _load_core(self):
        """Load QuantumCore."""
        try:
            import importlib.util

            data_loader_path = _SEMANTIC_LLM_PATH / "core" / "data_loader.py"
            hybrid_llm_path = _SEMANTIC_LLM_PATH / "core" / "hybrid_llm.py"

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
            print(f"  {len(self.core.states)} states in semantic space")

        except Exception as e:
            print(f"Error loading QuantumCore: {e}")
            self.core = None

    def extract(self, filepath: str, title: str = None) -> HeroPath:
        """
        Extract hero's path from a book.

        Simply: sequence of semantic words in reading order.
        """
        if self.core is None:
            raise RuntimeError("QuantumCore not loaded")

        if title is None:
            title = Path(filepath).stem
            if " - " in title:
                title = title.split(" - ", 1)[1]

        print(f"\nExtracting path from '{title}'...")

        # Load text
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Skip header/footer (Gutenberg)
        text_len = len(text)
        text = text[int(text_len * 0.05):int(text_len * 0.95)]

        # Extract words in order
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        print(f"  Total words: {len(words)}")

        # Build path: sequence of semantic words
        path = []
        concepts = {}

        for i, word in enumerate(words):
            state = self.core.states.get(word)
            if state is None:
                continue

            point = PathPoint(
                word=word,
                position=i,
                goodness=state.goodness,
                tau=state.tau,
                j=state.j
            )
            path.append(point)

            if word not in concepts:
                concepts[word] = point

        region = BookRegion(title=title, concepts=concepts)
        hero_path = HeroPath(region=region, path=path)

        print(f"  Path length: {len(path)} semantic words")
        print(f"  Region size: {len(concepts)} unique concepts")
        print(f"  Region bounds: g=[{region.bounds['g'][0]:.2f}, {region.bounds['g'][1]:.2f}]")

        return hero_path


def visualize_path(hero_path: HeroPath, save_path: str = None):
    """Visualize the hero's path."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw goodness sequence (sampled for visibility)
    ax1 = axes[0, 0]
    g = hero_path.g_sequence
    sample_rate = max(1, len(g) // 1000)
    g_sampled = g[::sample_rate]
    x = np.arange(len(g_sampled)) * sample_rate / len(g) * 100

    ax1.plot(x, g_sampled, '-', color='green', alpha=0.5, lw=0.5)
    ax1.set_xlabel('Position in book (%)')
    ax1.set_ylabel('Goodness (g)')
    ax1.set_title(f"Raw Path: {hero_path.region.title}")
    ax1.grid(True, alpha=0.3)

    # 2. Smoothed goodness (the narrative arc)
    ax2 = axes[0, 1]
    window = max(50, len(g) // 100)
    g_smooth = hero_path.smoothed_g(window)
    x_smooth = np.linspace(0, 100, len(g_smooth))

    ax2.fill_between(x_smooth, g_smooth, alpha=0.3, color='green')
    ax2.plot(x_smooth, g_smooth, '-', color='darkgreen', lw=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Position in book (%)')
    ax2.set_ylabel('Goodness (g)')
    ax2.set_title(f"Smoothed Arc (window={window})")
    ax2.grid(True, alpha=0.3)

    # Mark nadir and zenith
    min_idx = np.argmin(g_smooth)
    max_idx = np.argmax(g_smooth)
    ax2.scatter([x_smooth[min_idx]], [g_smooth[min_idx]], s=100, c='red', zorder=5)
    ax2.scatter([x_smooth[max_idx]], [g_smooth[max_idx]], s=100, c='gold', zorder=5)

    # 3. Region map (g vs τ)
    ax3 = axes[1, 0]
    concepts = list(hero_path.region.concepts.values())
    g_vals = [c.goodness for c in concepts]
    tau_vals = [c.tau for c in concepts]

    ax3.scatter(g_vals, tau_vals, s=10, alpha=0.5, c='blue')
    ax3.set_xlabel('Goodness (g)')
    ax3.set_ylabel('Abstraction (τ)')
    ax3.set_title(f"Region: {len(concepts)} concepts")
    ax3.grid(True, alpha=0.3)

    # 4. Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats = f"""
    {hero_path.region.title}
    {'='*40}

    Region:
      Concepts: {len(hero_path.region.concepts)}
      g range: [{hero_path.region.bounds['g'][0]:.2f}, {hero_path.region.bounds['g'][1]:.2f}]
      τ range: [{hero_path.region.bounds['tau'][0]:.2f}, {hero_path.region.bounds['tau'][1]:.2f}]

    Path:
      Length: {len(hero_path)} semantic words
      Start g: {g[0]:.3f}
      End g: {g[-1]:.3f}
      Δg: {g[-1] - g[0]:+.3f}

    Arc:
      Min g: {np.min(g_smooth):.3f} at {x_smooth[min_idx]:.0f}%
      Max g: {np.max(g_smooth):.3f} at {x_smooth[max_idx]:.0f}%
      Variance: {np.var(g_smooth):.4f}
    """

    ax4.text(0.05, 0.95, stats, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("book", help="Path to book")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    extractor = BookPathExtractor()
    path = extractor.extract(args.book)

    if args.plot:
        title = path.region.title.replace(' ', '_')
        visualize_path(path, f"visualizations/{title}_path.png")
