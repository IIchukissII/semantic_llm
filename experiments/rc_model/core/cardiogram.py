"""Cardiogram visualization for RC-Model.

Semantic trajectory as "EKG of meaning".
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional

from .semantic_rc import Trajectory


def plot_cardiogram(
    trajectory: Trajectory,
    title: str = "Semantic Cardiogram",
    figsize: tuple[int, int] = (14, 5),
    show_boundaries: bool = True,
    show_coherence: bool = False,
    mode: str = 'quantum',
) -> Figure:
    """Plot semantic trajectory as cardiogram.

    Args:
        trajectory: RC model trajectory
        title: Plot title
        figsize: Figure size
        show_boundaries: Show sentence boundary markers
        show_coherence: Also plot coherence/bond strength
        mode: 'quantum' for (n, θ, r) or 'cartesian' for (A, S, τ)

    Returns:
        Matplotlib figure
    """
    Q = trajectory.Q_array
    if len(Q) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig

    n_plots = 2 if show_coherence else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots))
    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    x = np.arange(len(Q))

    if mode == 'quantum':
        # Quantum numbers (n, θ, r)
        ax.plot(x, trajectory.Q_n, 'r-', label='n (orbital)', alpha=0.8, linewidth=1.5)
        ax.plot(x, trajectory.Q_theta, 'b-', label='θ (phase)', alpha=0.8, linewidth=1.5)
        ax.plot(x, trajectory.Q_r, 'g-', label='r (magnitude)', alpha=0.8, linewidth=1.5)
        ax.set_ylabel('Quantum Numbers')
    else:
        # Cartesian (A, S, τ)
        ax.plot(x, trajectory.A, 'r-', label='A (Affirmation)', alpha=0.8, linewidth=1.5)
        ax.plot(x, trajectory.S, 'b-', label='S (Sacred)', alpha=0.8, linewidth=1.5)
        ax.plot(x, trajectory.tau, 'g-', label='τ (Abstraction)', alpha=0.8, linewidth=1.5)
        ax.set_ylabel('Cartesian Coordinates')

    # Sentence boundaries
    if show_boundaries and trajectory.sentence_boundaries:
        for b in trajectory.sentence_boundaries[:-1]:
            ax.axvline(x=b, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Bond index')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Coherence / Bond strength plot
    if show_coherence:
        ax2 = axes[1]
        coherence = trajectory.coherence
        bond_str = trajectory.bond_strength

        ax2.plot(x, coherence, 'orange', linewidth=1.5, label='Coherence cos(Δθ)', alpha=0.8)
        ax2.plot(x, bond_str, 'purple', linewidth=1.5, label='Bond Strength', alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        if show_boundaries and trajectory.sentence_boundaries:
            for b in trajectory.sentence_boundaries[:-1]:
                ax2.axvline(x=b, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

        ax2.set_xlabel('Bond index')
        ax2.set_ylabel('Coherence / Bond Strength')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(
    trajectories: dict[str, Trajectory],
    title: str = "Trajectory Comparison",
    figsize: tuple[int, int] = (14, 8),
    component: str = 'potential',
) -> Figure:
    """Compare multiple trajectories.

    Args:
        trajectories: Dict of name -> trajectory
        title: Plot title
        figsize: Figure size
        component: Which component to plot ('potential', 'A', 'S', 'tau', 'all')

    Returns:
        Matplotlib figure
    """
    n_traj = len(trajectories)
    if n_traj == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No trajectories", ha='center', va='center')
        return fig

    if component == 'all':
        fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1] * 1.5))
        components = [('A', 0, 'red'), ('S', 1, 'blue'), ('τ', 2, 'green')]

        for ax, (name, idx, color) in zip(axes, components):
            for traj_name, traj in trajectories.items():
                Q = traj.Q_array
                if len(Q) > 0:
                    ax.plot(Q[:, idx], label=traj_name, alpha=0.7)
            ax.set_ylabel(f'Q_{name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Bond index')
        axes[0].set_title(title)

    else:
        fig, ax = plt.subplots(figsize=figsize)

        for traj_name, traj in trajectories.items():
            if len(traj) == 0:
                continue

            if component == 'potential':
                y = traj.potential
                ylabel = 'Semantic Potential Φ'
            elif component == 'A':
                y = traj.Q_A
                ylabel = 'Q_A (Affirmation)'
            elif component == 'S':
                y = traj.Q_S
                ylabel = 'Q_S (Sacred)'
            elif component == 'tau':
                y = traj.Q_tau
                ylabel = 'Q_τ (Abstraction)'
            else:
                raise ValueError(f"Unknown component: {component}")

            ax.plot(y, label=traj_name, alpha=0.7)

        ax.set_xlabel('Bond index')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_phase_space(
    trajectory: Trajectory,
    title: str = "Semantic Phase Space",
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """Plot trajectory in (A, S) phase space.

    Args:
        trajectory: RC model trajectory
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    Q = trajectory.Q_array
    if len(Q) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # Color by time (bond index)
    colors = np.linspace(0, 1, len(Q))
    scatter = ax.scatter(Q[:, 0], Q[:, 1], c=colors, cmap='viridis', s=20, alpha=0.6)

    # Connect points with line
    ax.plot(Q[:, 0], Q[:, 1], 'gray', alpha=0.3, linewidth=0.5)

    # Mark start and end
    ax.scatter([Q[0, 0]], [Q[0, 1]], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([Q[-1, 0]], [Q[-1, 1]], c='red', s=100, marker='s', label='End', zorder=5)

    ax.set_xlabel('Q_A (Affirmation)')
    ax.set_ylabel('Q_S (Sacred)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Time (bond index)')
    plt.tight_layout()
    return fig
