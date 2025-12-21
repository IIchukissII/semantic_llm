"""
Journey Visualization: Visualize semantic journeys through books.

Provides:
- Graph visualization with semantic properties as visual attributes
- Path highlighting showing the agent's journey
- Animation of the journey through semantic space
- Narrative arc plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Default output directory for visualizations
VIZ_OUTPUT_DIR = Path(__file__).parent.parent.parent / "visualizations"
VIZ_OUTPUT_DIR.mkdir(exist_ok=True)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Some visualizations limited.")


@dataclass
class JourneyData:
    """Data from a journey for visualization."""
    path: List[str]
    rewards: List[float]
    believe_history: List[float]
    tunnel_events: List[Dict]
    states_info: Dict[str, Dict]  # word -> {tau, goodness, ...}


class JourneyVisualizer:
    """
    Visualize semantic journeys through graphs.

    Creates beautiful visualizations of:
    - The semantic landscape
    - Agent's path through it
    - Tunneling events
    - Narrative arc
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        self.figsize = figsize

        # Color schemes
        self.goodness_cmap = self._create_goodness_cmap()
        self.path_color = '#FF6B35'  # Orange for path
        self.tunnel_color = '#9B5DE5'  # Purple for tunnels
        self.start_color = '#00F5D4'  # Cyan for start
        self.goal_color = '#FFD166'   # Gold for goal

    def _create_goodness_cmap(self):
        """Create colormap from evil (red) to good (green)."""
        colors = ['#D62828', '#F77F00', '#FCBF49', '#90BE6D', '#43AA8B']
        return LinearSegmentedColormap.from_list('goodness', colors)

    def plot_semantic_graph(self,
                           graph,
                           path: List[str] = None,
                           tunnel_events: List[Dict] = None,
                           title: str = "Semantic Landscape",
                           save_path: str = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot the semantic graph with optional journey path.

        Args:
            graph: SemanticGraph object
            path: List of words representing the journey
            tunnel_events: List of tunnel event dicts
            title: Plot title
            save_path: Path to save figure
            show: Whether to display the figure
        """
        if not HAS_NETWORKX:
            return self._plot_simple_graph(graph, path, title, save_path, show)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Build networkx graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for word, state in graph.states.items():
            G.add_node(word,
                      tau=state.tau,
                      goodness=state.goodness,
                      in_path=path and word in path)

        # Add edges
        for from_word, transitions in graph.transitions.items():
            for trans in transitions:
                G.add_edge(from_word, trans.to_state,
                          verb=trans.verb,
                          delta_g=trans.delta_g)

        # Layout - use spring layout with goodness influencing vertical position
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Adjust y-position based on tau (abstraction = higher)
        for node in G.nodes():
            tau = G.nodes[node].get('tau', 1.0)
            pos[node] = (pos[node][0], pos[node][1] + tau * 0.3)

        # Node colors based on goodness
        node_colors = []
        for node in G.nodes():
            g = G.nodes[node].get('goodness', 0)
            # Normalize to 0-1
            g_norm = (g + 1) / 2
            node_colors.append(self.goodness_cmap(g_norm))

        # Node sizes based on tau
        node_sizes = []
        for node in G.nodes():
            tau = G.nodes[node].get('tau', 1.0)
            base_size = 300
            node_sizes.append(base_size + tau * 200)

        # Draw base graph
        nx.draw_networkx_edges(G, pos, ax=ax,
                              edge_color='lightgray',
                              alpha=0.3,
                              arrows=True,
                              arrowsize=10,
                              connectionstyle='arc3,rad=0.1')

        nx.draw_networkx_nodes(G, pos, ax=ax,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.7)

        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax,
                               font_size=8,
                               font_weight='bold')

        # Highlight path if provided
        if path and len(path) > 1:
            self._draw_path(ax, G, pos, path, tunnel_events)

        # Mark start and goal
        if path:
            start_word = path[0]
            if start_word in pos:
                ax.scatter(*pos[start_word], s=800, c=self.start_color,
                          marker='o', zorder=5, edgecolors='black', linewidths=2)

            # Find if goal was reached (last unique word)
            goal_word = path[-1]
            if goal_word in pos and goal_word != start_word:
                ax.scatter(*pos[goal_word], s=800, c=self.goal_color,
                          marker='*', zorder=5, edgecolors='black', linewidths=2)

        # Legend
        legend_elements = [
            mpatches.Patch(color=self.goodness_cmap(0.0), label='Low Goodness'),
            mpatches.Patch(color=self.goodness_cmap(1.0), label='High Goodness'),
            plt.Line2D([0], [0], color=self.path_color, linewidth=3, label='Journey Path'),
            plt.Line2D([0], [0], color=self.tunnel_color, linewidth=3,
                      linestyle='--', label='Tunnel'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.start_color,
                      markersize=15, label='Start'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=self.goal_color,
                      markersize=15, label='Goal'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def _draw_path(self, ax, G, pos, path: List[str], tunnel_events: List[Dict] = None):
        """Draw the journey path on the graph."""
        tunnel_pairs = set()
        if tunnel_events:
            for event in tunnel_events:
                tunnel_pairs.add((event.get('from'), event.get('to')))

        # Draw path edges
        for i in range(len(path) - 1):
            from_word = path[i]
            to_word = path[i + 1]

            if from_word not in pos or to_word not in pos:
                continue

            is_tunnel = (from_word, to_word) in tunnel_pairs

            x1, y1 = pos[from_word]
            x2, y2 = pos[to_word]

            if is_tunnel:
                # Dashed purple line for tunnel
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=self.tunnel_color,
                                         lw=3, ls='--'))
            else:
                # Solid orange line for regular transition
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=self.path_color,
                                         lw=2.5))

    def _plot_simple_graph(self, graph, path, title, save_path, show):
        """Simple visualization without networkx."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Position nodes in a circle
        words = list(graph.states.keys())
        n = len(words)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)

        pos = {}
        for i, word in enumerate(words):
            state = graph.states[word]
            r = 1 + state.tau * 0.3  # Radius varies with tau
            pos[word] = (r * np.cos(angles[i]), r * np.sin(angles[i]))

        # Draw nodes
        for word, (x, y) in pos.items():
            state = graph.states[word]
            g_norm = (state.goodness + 1) / 2
            color = self.goodness_cmap(g_norm)
            size = 100 + state.tau * 50

            ax.scatter(x, y, s=size, c=[color], alpha=0.7)
            ax.annotate(word, (x, y), fontsize=7, ha='center')

        # Draw path
        if path:
            for i in range(len(path) - 1):
                if path[i] in pos and path[i+1] in pos:
                    x1, y1 = pos[path[i]]
                    x2, y2 = pos[path[i+1]]
                    ax.plot([x1, x2], [y1, y2], color=self.path_color, lw=2, alpha=0.7)

        ax.set_title(title)
        ax.axis('equal')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()

        return fig

    def plot_narrative_arc(self,
                          path: List[str],
                          states_info: Dict[str, Dict],
                          believe_history: List[float] = None,
                          title: str = "Narrative Arc",
                          save_path: str = None,
                          show: bool = True) -> plt.Figure:
        """
        Plot the narrative arc - goodness over the journey.

        Shows how the agent moves through semantic space
        from a goodness perspective.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Get goodness values along path
        unique_path = []
        goodness_values = []
        tau_values = []

        for word in path:
            if word in states_info:
                unique_path.append(word)
                goodness_values.append(states_info[word].get('goodness', 0))
                tau_values.append(states_info[word].get('tau', 1))

        steps = range(len(unique_path))

        # Plot 1: Goodness arc
        ax1 = axes[0]
        colors = [self.goodness_cmap((g + 1) / 2) for g in goodness_values]

        ax1.fill_between(steps, goodness_values, alpha=0.3, color='green')
        ax1.plot(steps, goodness_values, 'o-', color='darkgreen', lw=2, markersize=8)

        # Color points by goodness
        for i, (s, g) in enumerate(zip(steps, goodness_values)):
            ax1.scatter(s, g, c=[colors[i]], s=100, zorder=5, edgecolors='black')

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Goodness (g)', fontsize=12)
        ax1.set_title(f'{title} - Goodness Through Journey', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Annotate key points
        if len(unique_path) > 0:
            # Start
            ax1.annotate(unique_path[0], (0, goodness_values[0]),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9, fontweight='bold')
            # End
            ax1.annotate(unique_path[-1], (len(unique_path)-1, goodness_values[-1]),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9, fontweight='bold')
            # Min/Max
            min_idx = np.argmin(goodness_values)
            max_idx = np.argmax(goodness_values)
            if min_idx not in [0, len(unique_path)-1]:
                ax1.annotate(f'↓{unique_path[min_idx]}', (min_idx, goodness_values[min_idx]),
                            textcoords="offset points", xytext=(0, -15),
                            ha='center', fontsize=8, color='red')
            if max_idx not in [0, len(unique_path)-1]:
                ax1.annotate(f'↑{unique_path[max_idx]}', (max_idx, goodness_values[max_idx]),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=8, color='green')

        # Plot 2: Believe and Tau
        ax2 = axes[1]

        ax2.plot(steps, tau_values, 's-', color='blue', lw=2, markersize=6,
                label='Abstraction (τ)', alpha=0.7)

        if believe_history and len(believe_history) >= len(steps):
            believe_subset = believe_history[:len(steps)]
            ax2.plot(steps, believe_subset, '^-', color='purple', lw=2, markersize=6,
                    label='Believe', alpha=0.7)

        ax2.set_xlabel('Journey Step', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # X-axis labels with word names
        if len(unique_path) <= 20:
            ax2.set_xticks(steps)
            ax2.set_xticklabels(unique_path, rotation=45, ha='right', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_journey_summary(self,
                            journey_data: JourneyData,
                            graph = None,
                            title: str = "Semantic Journey",
                            save_path: str = None,
                            show: bool = True) -> plt.Figure:
        """
        Create a comprehensive journey summary visualization.

        Combines graph view and narrative arc.
        """
        fig = plt.figure(figsize=(16, 10))

        # Create grid
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.5, 1])

        # Top left: Graph (if available)
        ax_graph = fig.add_subplot(gs[0, 0])
        if graph and HAS_NETWORKX:
            self._draw_mini_graph(ax_graph, graph, journey_data.path, journey_data.tunnel_events)
        else:
            ax_graph.text(0.5, 0.5, 'Graph visualization\nrequires networkx',
                         ha='center', va='center', fontsize=12)
            ax_graph.axis('off')
        ax_graph.set_title('Semantic Landscape', fontsize=12, fontweight='bold')

        # Top right: Journey stats
        ax_stats = fig.add_subplot(gs[0, 1])
        self._draw_stats(ax_stats, journey_data)
        ax_stats.set_title('Journey Statistics', fontsize=12, fontweight='bold')

        # Bottom: Narrative arc
        ax_arc = fig.add_subplot(gs[1, :])
        self._draw_arc_mini(ax_arc, journey_data)
        ax_arc.set_title('Narrative Arc', fontsize=12, fontweight='bold')

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def _draw_mini_graph(self, ax, graph, path, tunnel_events):
        """Draw a smaller version of the graph."""
        G = nx.DiGraph()

        # Only include states in or near the path
        path_set = set(path) if path else set()

        for word, state in graph.states.items():
            G.add_node(word, tau=state.tau, goodness=state.goodness)

        for from_word, transitions in graph.transitions.items():
            for trans in transitions:
                G.add_edge(from_word, trans.to_state)

        pos = nx.spring_layout(G, k=1.5, iterations=30, seed=42)

        # Node colors
        node_colors = []
        for node in G.nodes():
            g = G.nodes[node].get('goodness', 0)
            node_colors.append(self.goodness_cmap((g + 1) / 2))

        # Draw
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', alpha=0.2, arrows=False)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=100, alpha=0.6)

        # Highlight path
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)
                         if path[i] in G.nodes() and path[i+1] in G.nodes()]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges,
                                  edge_color=self.path_color, width=2, alpha=0.8)

            # Start and end
            if path[0] in pos:
                ax.scatter(*pos[path[0]], s=200, c=self.start_color, marker='o', zorder=5)
            if path[-1] in pos:
                ax.scatter(*pos[path[-1]], s=200, c=self.goal_color, marker='*', zorder=5)

        ax.axis('off')

    def _draw_stats(self, ax, journey_data: JourneyData):
        """Draw journey statistics."""
        ax.axis('off')

        path = journey_data.path
        unique_states = len(set(path))
        total_steps = len(path)
        n_tunnels = len(journey_data.tunnel_events)
        total_reward = sum(journey_data.rewards) if journey_data.rewards else 0
        final_believe = journey_data.believe_history[-1] if journey_data.believe_history else 0

        stats_text = f"""
        Journey Statistics
        ─────────────────────

        Total Steps: {total_steps}
        Unique States: {unique_states}
        Tunnel Events: {n_tunnels}

        Total Reward: {total_reward:+.2f}
        Final Believe: {final_believe:.2f}

        Path Preview:
        {' → '.join(path[:5])}
        {'...' if len(path) > 5 else ''}
        """

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _draw_arc_mini(self, ax, journey_data: JourneyData):
        """Draw a mini narrative arc."""
        path = journey_data.path
        goodness = [journey_data.states_info.get(w, {}).get('goodness', 0) for w in path]

        steps = range(len(path))
        colors = [self.goodness_cmap((g + 1) / 2) for g in goodness]

        ax.fill_between(steps, goodness, alpha=0.3, color='green')
        ax.scatter(steps, goodness, c=colors, s=50, edgecolors='black', linewidths=0.5)
        ax.plot(steps, goodness, '-', color='darkgreen', alpha=0.5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Goodness')
        ax.grid(True, alpha=0.3)


def plot_journey(graph,
                path: List[str],
                tunnel_events: List[Dict] = None,
                title: str = "Semantic Journey",
                save_path: str = None) -> plt.Figure:
    """
    Quick function to plot a journey.

    Args:
        graph: SemanticGraph object
        path: List of words in the journey
        tunnel_events: Optional list of tunnel event dicts
        title: Plot title
        save_path: Optional path to save figure
    """
    viz = JourneyVisualizer()
    return viz.plot_semantic_graph(graph, path, tunnel_events, title, save_path)


def animate_journey(graph,
                   path: List[str],
                   states_info: Dict[str, Dict],
                   interval: int = 500,
                   save_path: str = None) -> FuncAnimation:
    """
    Create an animation of the journey through semantic space.

    Args:
        graph: SemanticGraph object
        path: List of words in the journey
        states_info: Dict of word -> {tau, goodness, ...}
        interval: Milliseconds between frames
        save_path: Optional path to save animation (requires ffmpeg)
    """
    if not HAS_NETWORKX:
        print("Animation requires networkx")
        return None

    fig, (ax_graph, ax_arc) = plt.subplots(1, 2, figsize=(16, 8))

    viz = JourneyVisualizer()

    # Build graph
    G = nx.DiGraph()
    for word, state in graph.states.items():
        G.add_node(word, tau=state.tau, goodness=state.goodness)
    for from_word, transitions in graph.transitions.items():
        for trans in transitions:
            G.add_edge(from_word, trans.to_state)

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Adjust positions by tau
    for node in G.nodes():
        tau = G.nodes[node].get('tau', 1.0)
        pos[node] = (pos[node][0], pos[node][1] + tau * 0.3)

    def update(frame):
        ax_graph.clear()
        ax_arc.clear()

        current_path = path[:frame+1]

        # Draw graph
        node_colors = [viz.goodness_cmap((G.nodes[n].get('goodness', 0) + 1) / 2)
                      for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='lightgray', alpha=0.2)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_colors,
                              node_size=200, alpha=0.5)

        # Highlight current path
        for i in range(len(current_path) - 1):
            if current_path[i] in pos and current_path[i+1] in pos:
                x1, y1 = pos[current_path[i]]
                x2, y2 = pos[current_path[i+1]]
                ax_graph.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                 arrowprops=dict(arrowstyle='->', color=viz.path_color, lw=2))

        # Current position
        if current_path[-1] in pos:
            ax_graph.scatter(*pos[current_path[-1]], s=500, c='red', marker='o',
                           zorder=10, edgecolors='black', linewidths=2)
            ax_graph.annotate(current_path[-1], pos[current_path[-1]],
                            textcoords="offset points", xytext=(0, 15),
                            ha='center', fontsize=12, fontweight='bold')

        ax_graph.set_title(f'Step {frame+1}: {current_path[-1]}', fontsize=14)
        ax_graph.axis('off')

        # Draw arc so far
        goodness = [states_info.get(w, {}).get('goodness', 0) for w in current_path]
        steps = range(len(current_path))

        ax_arc.fill_between(steps, goodness, alpha=0.3, color='green')
        ax_arc.plot(steps, goodness, 'o-', color='darkgreen', lw=2)
        ax_arc.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_arc.set_xlim(0, len(path))
        ax_arc.set_ylim(-1.5, 1.5)
        ax_arc.set_xlabel('Step')
        ax_arc.set_ylabel('Goodness')
        ax_arc.set_title('Narrative Arc', fontsize=14)
        ax_arc.grid(True, alpha=0.3)

        return ax_graph, ax_arc

    anim = FuncAnimation(fig, update, frames=len(path), interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=2)
        print(f"Saved animation: {save_path}")

    plt.tight_layout()
    return anim


if __name__ == "__main__":
    # Demo with sample data
    print("Journey Visualization Demo")
    print("=" * 50)

    # Create sample journey data
    sample_path = ["darkness", "fear", "struggle", "hope", "courage", "wisdom"]
    sample_states = {
        "darkness": {"tau": 0.5, "goodness": -0.3},
        "fear": {"tau": 0.6, "goodness": -0.2},
        "struggle": {"tau": 0.8, "goodness": 0.1},
        "hope": {"tau": 1.0, "goodness": 0.5},
        "courage": {"tau": 1.2, "goodness": 0.7},
        "wisdom": {"tau": 1.5, "goodness": 0.9},
    }

    viz = JourneyVisualizer()
    viz.plot_narrative_arc(sample_path, sample_states,
                          title="Sample Hero's Journey")
