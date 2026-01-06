#!/usr/bin/env python3
"""Visualize conversation trajectory through semantic space."""

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path


def visualize_trajectory(json_path: str, output_path: str = None):
    """Create 3D trajectory visualization from conversation JSON."""

    with open(json_path) as f:
        data = json.load(f)

    conversation = data['conversation']

    # Extract coordinates
    turns = [c['turn'] for c in conversation]
    A = [c['state']['A'] for c in conversation]
    S = [c['state']['S'] for c in conversation]
    tau = [c['state']['tau'] for c in conversation]
    irony = [c['state']['irony'] for c in conversation]

    # Health target
    health = {'A': 0.3, 'S': 0.2, 'tau': 2.0}

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot trajectory
    ax1.plot(A, S, tau, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax1.scatter(A, S, tau, c=irony, cmap='Reds', s=100, edgecolors='black',
                label='States', vmin=0, vmax=1)

    # Mark start and end
    ax1.scatter([A[0]], [S[0]], [tau[0]], c='green', s=200, marker='^', label='Start')
    ax1.scatter([A[-1]], [S[-1]], [tau[-1]], c='blue', s=200, marker='s', label='End')

    # Health target
    ax1.scatter([health['A']], [health['S']], [health['tau']],
                c='gold', s=300, marker='*', label='Health Target')

    ax1.set_xlabel('A (Affirmation)')
    ax1.set_ylabel('S (Sacred)')
    ax1.set_zlabel('τ (Abstraction)')
    ax1.set_title('Semantic Trajectory (3D)')
    ax1.legend(loc='upper left')

    # 2. A-S projection (top view)
    ax2 = fig.add_subplot(2, 2, 2)

    for i in range(len(A)-1):
        ax2.annotate('', xy=(A[i+1], S[i+1]), xytext=(A[i], S[i]),
                     arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))

    scatter = ax2.scatter(A, S, c=irony, cmap='Reds', s=150, edgecolors='black', vmin=0, vmax=1)
    ax2.scatter([health['A']], [health['S']], c='gold', s=300, marker='*', zorder=5)

    # Add turn labels
    for i, turn in enumerate(turns):
        ax2.annotate(str(turn), (A[i], S[i]), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel('A (Affirmation)')
    ax2.set_ylabel('S (Sacred)')
    ax2.set_title('A-S Projection (Irony intensity = color)')
    plt.colorbar(scatter, ax=ax2, label='Irony')

    # 3. Time series
    ax3 = fig.add_subplot(2, 2, 3)

    ax3.plot(turns, A, 'g-o', label='A (Affirmation)', linewidth=2)
    ax3.plot(turns, S, 'b-s', label='S (Sacred)', linewidth=2)
    ax3.plot(turns, [t/4 for t in tau], 'r-^', label='τ/4 (Abstraction)', linewidth=2)
    ax3.plot(turns, irony, 'm--', label='Irony', linewidth=2, alpha=0.7)

    ax3.axhline(y=health['A'], color='g', linestyle=':', alpha=0.5, label='Health A')
    ax3.axhline(y=health['S'], color='b', linestyle=':', alpha=0.5, label='Health S')

    ax3.set_xlabel('Turn')
    ax3.set_ylabel('Value')
    ax3.set_title('Semantic Coordinates Over Time')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4. Irony and defenses
    ax4 = fig.add_subplot(2, 2, 4)

    colors = ['red' if c.get('defenses') else 'steelblue' for c in conversation]
    bars = ax4.bar(turns, irony, color=colors, edgecolor='black', alpha=0.7)

    ax4.axhline(y=0.3, color='red', linestyle='--', label='Defense threshold')
    ax4.set_xlabel('Turn')
    ax4.set_ylabel('Irony Level')
    ax4.set_title('Irony Detection (red = defense detected)')
    ax4.legend()

    # Add defense annotations
    for i, c in enumerate(conversation):
        if c.get('defenses'):
            ax4.annotate('\n'.join(c['defenses']), (turns[i], irony[i]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha='center', color='red')

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = json_path.replace('.json', '_trajectory.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    plt.close()
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Find most recent conversation
        results_dir = Path(__file__).parent.parent / 'results'
        json_files = sorted(results_dir.glob('conversation_*.json'))
        if json_files:
            json_path = str(json_files[-1])
        else:
            print("No conversation files found")
            sys.exit(1)

    visualize_trajectory(json_path)
