"""Boundary Analyzer: Detect sentence and narrative boundaries."""

from typing import List, Optional, Tuple
import numpy as np

from ...data.models import Bond, Trajectory


class BoundaryAnalyzer:
    """Detect sentence and narrative boundaries.

    Boundaries are marked by:
    - Large jumps in τ
    - Large jumps in A or S
    - Combined multi-axis jumps
    """

    def __init__(self, tau_threshold: float = 0.5,
                 A_threshold: float = 0.4,
                 S_threshold: float = 0.3):
        self.tau_threshold = tau_threshold
        self.A_threshold = A_threshold
        self.S_threshold = S_threshold

    def detect(self, prev: Bond, curr: Bond) -> bool:
        """Detect if there's a boundary between bonds.

        Args:
            prev: Previous bond
            curr: Current bond

        Returns:
            True if boundary detected
        """
        delta_tau = abs(curr.tau - prev.tau)
        delta_A = abs(curr.A - prev.A)
        delta_S = abs(curr.S - prev.S)

        # Any single large jump
        if delta_tau > self.tau_threshold:
            return True
        if delta_A > self.A_threshold:
            return True
        if delta_S > self.S_threshold:
            return True

        # Combined moderate jumps
        total = delta_tau / self.tau_threshold + \
                delta_A / self.A_threshold + \
                delta_S / self.S_threshold
        if total > 1.5:
            return True

        return False

    def find_boundaries(self, trajectory: Trajectory) -> List[int]:
        """Find all boundary positions in trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            List of boundary indices
        """
        boundaries = []

        for i in range(1, len(trajectory.bonds)):
            if self.detect(trajectory.bonds[i-1], trajectory.bonds[i]):
                boundaries.append(i)

        return boundaries

    def segment(self, trajectory: Trajectory) -> List[List[Bond]]:
        """Segment trajectory at boundaries.

        Args:
            trajectory: Trajectory to segment

        Returns:
            List of segments (each segment is a list of bonds)
        """
        boundaries = self.find_boundaries(trajectory)

        segments = []
        prev_idx = 0

        for idx in boundaries:
            if idx > prev_idx:
                segments.append(trajectory.bonds[prev_idx:idx])
            prev_idx = idx

        # Add final segment
        if prev_idx < len(trajectory.bonds):
            segments.append(trajectory.bonds[prev_idx:])

        return segments

    def compute_boundary_jumps(self, trajectory: Trajectory) -> dict:
        """Compute statistics about boundary jumps.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dictionary with boundary statistics
        """
        boundaries = self.find_boundaries(trajectory)

        if not boundaries:
            return {
                'n_boundaries': 0,
                'mean_A_jump': 0.0,
                'mean_S_jump': 0.0,
                'mean_tau_jump': 0.0,
            }

        A_jumps = []
        S_jumps = []
        tau_jumps = []

        for idx in boundaries:
            prev = trajectory.bonds[idx-1]
            curr = trajectory.bonds[idx]
            A_jumps.append(abs(curr.A - prev.A))
            S_jumps.append(abs(curr.S - prev.S))
            tau_jumps.append(abs(curr.tau - prev.tau))

        return {
            'n_boundaries': len(boundaries),
            'mean_A_jump': np.mean(A_jumps),
            'mean_S_jump': np.mean(S_jumps),
            'mean_tau_jump': np.mean(tau_jumps),
            'max_A_jump': np.max(A_jumps),
            'max_S_jump': np.max(S_jumps),
            'max_tau_jump': np.max(tau_jumps),
        }

    def compute_within_segment_stats(self, trajectory: Trajectory) -> dict:
        """Compute statistics within segments (between boundaries).

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Dictionary with within-segment statistics
        """
        segments = self.segment(trajectory)

        if not segments:
            return {
                'mean_within_A': 0.0,
                'mean_within_S': 0.0,
                'mean_within_tau': 0.0,
            }

        within_A = []
        within_S = []
        within_tau = []

        for segment in segments:
            for i in range(1, len(segment)):
                within_A.append(abs(segment[i].A - segment[i-1].A))
                within_S.append(abs(segment[i].S - segment[i-1].S))
                within_tau.append(abs(segment[i].tau - segment[i-1].tau))

        return {
            'mean_within_A': np.mean(within_A) if within_A else 0.0,
            'mean_within_S': np.mean(within_S) if within_S else 0.0,
            'mean_within_tau': np.mean(within_tau) if within_tau else 0.0,
        }
