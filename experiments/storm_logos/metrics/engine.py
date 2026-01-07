"""Metrics Engine: Orchestrates all measurement.

Central hub for measuring semantic state, trajectory, and text.
"""

from typing import Optional, List

from ..data.models import Metrics, SemanticState, Trajectory, Bond
from .extractors.text import TextExtractor
from .extractors.bond import BondExtractor
from .extractors.state import StateExtractor
from .analyzers.irony import IronyAnalyzer
from .analyzers.coherence import CoherenceAnalyzer
from .analyzers.tau import TauAnalyzer
from .analyzers.tension import TensionAnalyzer
from .analyzers.defense import DefenseAnalyzer
from .analyzers.boundary import BoundaryAnalyzer


class MetricsEngine:
    """Central orchestrator for all semantic measurements.

    Usage:
        engine = MetricsEngine()
        metrics = engine.measure(text="...")
        metrics = engine.measure_trajectory(trajectory)
        metrics = engine.measure_state(state, history)
    """

    def __init__(self):
        # Extractors
        self.text_extractor = TextExtractor()
        self.bond_extractor = BondExtractor()
        self.state_extractor = StateExtractor()

        # Analyzers
        self.irony = IronyAnalyzer()
        self.coherence = CoherenceAnalyzer()
        self.tau = TauAnalyzer()
        self.tension = TensionAnalyzer()
        self.defense = DefenseAnalyzer()
        self.boundary = BoundaryAnalyzer()

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================

    def measure(self, text: str = None,
                trajectory: Trajectory = None,
                state: SemanticState = None,
                history: List[SemanticState] = None) -> Metrics:
        """Unified measurement interface.

        Provide any combination of inputs; all relevant metrics computed.

        Args:
            text: Text to analyze
            trajectory: Bond trajectory
            state: Current semantic state
            history: History of states

        Returns:
            Metrics object with all measurements
        """
        metrics = Metrics()

        # Extract from text if provided
        if text:
            extracted = self.text_extractor.extract(text)
            bonds = extracted.all_bonds

            # Enrich bonds with coordinates
            bonds = self.bond_extractor.enrich_bonds(bonds)

            # Build trajectory from extracted bonds
            if not trajectory:
                trajectory = Trajectory(bonds=bonds)

            # Measure irony
            metrics.irony = self.irony.analyze(text=text, state=state)

            # Detect defenses
            metrics.defenses = self.defense.analyze(text=text, state=state)

        # Measure trajectory
        if trajectory:
            metrics.coherence = self.coherence.analyze(trajectory=trajectory)
            metrics.noise_ratio = self.coherence.compute_noise_ratio(trajectory)

            tau_metrics = self.tau.analyze(trajectory=trajectory)
            metrics.tau_mean = tau_metrics.get('tau_mean', 2.5)
            metrics.tau_variance = tau_metrics.get('tau_variance', 0.0)
            metrics.tau_slope = tau_metrics.get('tau_slope', 0.0)

            metrics.tension_score = self.tension.analyze(trajectory=trajectory)

            # Compute A and S position from trajectory bonds
            if trajectory.bonds:
                valid_bonds = [b for b in trajectory.bonds if b.A != 0 or b.S != 0]
                if valid_bonds:
                    metrics.A_position = sum(b.A for b in valid_bonds) / len(valid_bonds)
                    metrics.S_position = sum(b.S for b in valid_bonds) / len(valid_bonds)

                # Check boundaries
                boundary_stats = self.boundary.compute_boundary_jumps(trajectory)
                metrics.boundary_detected = boundary_stats.get('n_boundaries', 0) > 0

        # Measure state
        if state:
            metrics.A_position = state.A
            metrics.S_position = state.S
            metrics.irony = max(metrics.irony, state.irony)

            if not metrics.defenses:
                metrics.defenses = self.defense.analyze(state=state)

            metrics.tension_score = self.tension.analyze(state=state)

        return metrics

    def measure_text(self, text: str) -> Metrics:
        """Measure text only.

        Args:
            text: Text to analyze

        Returns:
            Metrics
        """
        return self.measure(text=text)

    def measure_trajectory(self, trajectory: Trajectory) -> Metrics:
        """Measure trajectory only.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            Metrics
        """
        return self.measure(trajectory=trajectory)

    def measure_state(self, state: SemanticState,
                      history: List[SemanticState] = None) -> Metrics:
        """Measure state with optional history.

        Args:
            state: Current state
            history: History of states

        Returns:
            Metrics
        """
        metrics = self.measure(state=state)

        if history:
            # Add history-based metrics
            state_metrics = self.state_extractor.extract_from_state(state, history)
            metrics.tau_mean = state_metrics.get('tau_mean', metrics.tau_mean)
            metrics.tau_variance = state_metrics.get('tau_variance', metrics.tau_variance)
            metrics.tau_slope = state_metrics.get('tau_slope', metrics.tau_slope)

        return metrics

    # ========================================================================
    # COMPARISON
    # ========================================================================

    def compare(self, metrics: Metrics, targets: dict) -> dict:
        """Compare metrics against targets.

        Args:
            metrics: Current metrics
            targets: Target values (homeostatic)

        Returns:
            Dictionary of errors (target - current)
        """
        errors = {}
        metrics_dict = metrics.as_dict()

        for key, target in targets.items():
            if key in metrics_dict:
                current = metrics_dict[key]
                if isinstance(current, (int, float)):
                    errors[f'{key}_error'] = target - current

        return errors


# ============================================================================
# SINGLETON
# ============================================================================

_engine_instance: Optional[MetricsEngine] = None


def get_metrics_engine() -> MetricsEngine:
    """Get singleton MetricsEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MetricsEngine()
    return _engine_instance
