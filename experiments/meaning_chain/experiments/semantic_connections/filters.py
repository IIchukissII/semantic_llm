"""
Semantic Filters: τ-Based Frequency Filtering
==============================================

Goals as filters:
  LOW-PASS (GROUNDED):   Pass τ < cutoff (concrete)
  HIGH-PASS (DEEP):      Pass τ > cutoff (abstract)
  BAND-PASS (FOCUSED):   Pass τ_low < τ < τ_high
  BAND-STOP (AVOID):     Block τ_low < τ < τ_high

Transfer function: H(τ) determines how much of each τ-level passes.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class FilteredConcept:
    """A concept after filtering"""
    concept: str
    tau: float
    gain: float  # How much it passes through [0, 1]
    passed: bool  # Whether it passed the filter


@dataclass
class FilterResponse:
    """Complete filter response"""
    filter_type: str
    input_concepts: List[str]
    output_concepts: List[FilteredConcept]
    passed_count: int
    blocked_count: int
    avg_gain: float

    @property
    def passed(self) -> List[str]:
        """Concepts that passed"""
        return [c.concept for c in self.output_concepts if c.passed]

    @property
    def blocked(self) -> List[str]:
        """Concepts that were blocked"""
        return [c.concept for c in self.output_concepts if not c.passed]

    @property
    def pass_rate(self) -> float:
        """Fraction that passed"""
        total = len(self.output_concepts)
        if total == 0:
            return 0.0
        return self.passed_count / total


class SemanticFilter(ABC):
    """Abstract base class for semantic filters"""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Minimum gain to pass [0, 1]
        """
        self.threshold = threshold

    @abstractmethod
    def transfer_function(self, tau: float) -> float:
        """
        Compute filter gain H(τ) at given τ

        Args:
            tau: Abstraction level

        Returns:
            Gain [0, 1] where 1 = full pass, 0 = full block
        """
        pass

    @property
    @abstractmethod
    def filter_type(self) -> str:
        """Filter type name"""
        pass

    def filter_concept(self, concept: str, tau: float) -> FilteredConcept:
        """Filter a single concept"""
        gain = self.transfer_function(tau)
        return FilteredConcept(
            concept=concept,
            tau=tau,
            gain=gain,
            passed=gain >= self.threshold,
        )

    def filter_concepts(
        self,
        concepts: List[Tuple[str, float]]
    ) -> FilterResponse:
        """
        Filter a list of concepts

        Args:
            concepts: List of (concept, tau) pairs

        Returns:
            FilterResponse with all filtering results
        """
        output = []
        for concept, tau in concepts:
            filtered = self.filter_concept(concept, tau)
            output.append(filtered)

        passed = [c for c in output if c.passed]
        blocked = [c for c in output if not c.passed]

        return FilterResponse(
            filter_type=self.filter_type,
            input_concepts=[c for c, _ in concepts],
            output_concepts=output,
            passed_count=len(passed),
            blocked_count=len(blocked),
            avg_gain=np.mean([c.gain for c in output]) if output else 0.0,
        )


class LowPassFilter(SemanticFilter):
    """
    Low-pass filter: Pass τ < cutoff (concrete concepts)

    H(τ) = 1 / (1 + (τ/τ_c)^(2n))

    Use for: GROUNDED goal
    """

    def __init__(
        self,
        cutoff: float = 1.74,  # n=2 orbital by default
        order: int = 2,
        threshold: float = 0.5
    ):
        """
        Args:
            cutoff: τ cutoff frequency (default: n=2 orbital)
            order: Filter order (sharpness)
            threshold: Pass threshold
        """
        super().__init__(threshold)
        self.cutoff = cutoff
        self.order = order

    @property
    def filter_type(self) -> str:
        return f"low-pass(τ_c={self.cutoff:.2f})"

    def transfer_function(self, tau: float) -> float:
        """Low-pass: H(τ) = 1 / (1 + (τ/τ_c)^(2n))"""
        if self.cutoff <= 0:
            return 0.0
        return 1.0 / (1.0 + (tau / self.cutoff) ** (2 * self.order))


class HighPassFilter(SemanticFilter):
    """
    High-pass filter: Pass τ > cutoff (abstract concepts)

    H(τ) = (τ/τ_c)^(2n) / (1 + (τ/τ_c)^(2n))

    Use for: DEEP goal
    """

    def __init__(
        self,
        cutoff: float = 2.1,  # n=3 orbital by default
        order: int = 2,
        threshold: float = 0.5
    ):
        """
        Args:
            cutoff: τ cutoff frequency (default: n=3 orbital)
            order: Filter order (sharpness)
            threshold: Pass threshold
        """
        super().__init__(threshold)
        self.cutoff = cutoff
        self.order = order

    @property
    def filter_type(self) -> str:
        return f"high-pass(τ_c={self.cutoff:.2f})"

    def transfer_function(self, tau: float) -> float:
        """High-pass: H(τ) = (τ/τ_c)^(2n) / (1 + (τ/τ_c)^(2n))"""
        if self.cutoff <= 0:
            return 1.0
        ratio = (tau / self.cutoff) ** (2 * self.order)
        return ratio / (1.0 + ratio)


class BandPassFilter(SemanticFilter):
    """
    Band-pass filter: Pass τ_low < τ < τ_high

    H(τ) = H_low(τ) × H_high(τ)

    Use for: WISDOM, ACCURATE, POWERFUL goals
    """

    def __init__(
        self,
        center: float = 1.45,
        bandwidth: float = 0.3,
        order: int = 2,
        threshold: float = 0.5
    ):
        """
        Args:
            center: Center τ of passband
            bandwidth: Width of passband (τ_high - τ_low)
            order: Filter order (sharpness)
            threshold: Pass threshold
        """
        super().__init__(threshold)
        self.center = center
        self.bandwidth = bandwidth
        self.order = order
        self.tau_low = center - bandwidth / 2
        self.tau_high = center + bandwidth / 2

    @property
    def filter_type(self) -> str:
        return f"band-pass({self.tau_low:.2f}<τ<{self.tau_high:.2f})"

    def transfer_function(self, tau: float) -> float:
        """Band-pass: Gaussian-like around center"""
        # Use Gaussian shape for smooth band-pass
        sigma = self.bandwidth / 2.355  # FWHM to sigma
        return np.exp(-((tau - self.center) ** 2) / (2 * sigma ** 2))


class BandStopFilter(SemanticFilter):
    """
    Band-stop (notch) filter: Block τ_low < τ < τ_high

    H(τ) = 1 - H_bandpass(τ)

    Use for: Avoiding specific τ ranges
    """

    def __init__(
        self,
        center: float = 2.718,  # The Veil
        bandwidth: float = 0.5,
        order: int = 2,
        threshold: float = 0.5
    ):
        """
        Args:
            center: Center τ of stopband
            bandwidth: Width of stopband
            order: Filter order
            threshold: Pass threshold
        """
        super().__init__(threshold)
        self.bandpass = BandPassFilter(center, bandwidth, order, 0.0)

    @property
    def filter_type(self) -> str:
        bp = self.bandpass
        return f"band-stop({bp.tau_low:.2f}<τ<{bp.tau_high:.2f})"

    def transfer_function(self, tau: float) -> float:
        """Band-stop: 1 - bandpass"""
        return 1.0 - self.bandpass.transfer_function(tau)


class GoalFilter:
    """
    Goal-to-filter mapping

    Maps navigation goals to appropriate τ filters
    """

    # Goal configurations
    GOAL_CONFIGS = {
        "grounded": {"type": "low-pass", "cutoff": 1.74, "order": 2},
        "deep": {"type": "high-pass", "cutoff": 2.1, "order": 2},
        "wisdom": {"type": "band-pass", "center": 1.45, "bandwidth": 0.3},
        "powerful": {"type": "band-pass", "center": 2.6, "bandwidth": 0.4},
        "accurate": {"type": "band-pass", "center": 1.37, "bandwidth": 0.2},
        "balanced": {"type": "band-pass", "center": 1.74, "bandwidth": 0.8},
        "stable": {"type": "low-pass", "cutoff": 2.0, "order": 1},
        "exploratory": None,  # No filter (pass all)
    }

    @classmethod
    def for_goal(cls, goal: str) -> Optional[SemanticFilter]:
        """
        Get appropriate filter for a navigation goal

        Args:
            goal: Goal name (grounded, deep, wisdom, etc.)

        Returns:
            SemanticFilter or None if no filtering
        """
        config = cls.GOAL_CONFIGS.get(goal.lower())

        if config is None:
            return None

        filter_type = config["type"]

        if filter_type == "low-pass":
            return LowPassFilter(
                cutoff=config["cutoff"],
                order=config.get("order", 2),
            )
        elif filter_type == "high-pass":
            return HighPassFilter(
                cutoff=config["cutoff"],
                order=config.get("order", 2),
            )
        elif filter_type == "band-pass":
            return BandPassFilter(
                center=config["center"],
                bandwidth=config["bandwidth"],
                order=config.get("order", 2),
            )
        elif filter_type == "band-stop":
            return BandStopFilter(
                center=config["center"],
                bandwidth=config["bandwidth"],
            )

        return None

    @classmethod
    def filter_for_goal(
        cls,
        goal: str,
        concepts: List[Tuple[str, float]]
    ) -> FilterResponse:
        """
        Filter concepts for a specific goal

        Args:
            goal: Navigation goal
            concepts: List of (concept, tau) pairs

        Returns:
            FilterResponse
        """
        filter_obj = cls.for_goal(goal)

        if filter_obj is None:
            # No filtering, pass everything
            output = [
                FilteredConcept(concept=c, tau=t, gain=1.0, passed=True)
                for c, t in concepts
            ]
            return FilterResponse(
                filter_type="none",
                input_concepts=[c for c, _ in concepts],
                output_concepts=output,
                passed_count=len(output),
                blocked_count=0,
                avg_gain=1.0,
            )

        return filter_obj.filter_concepts(concepts)


@dataclass
class FilterBank:
    """
    Bank of filters for multi-band analysis
    """

    filters: List[SemanticFilter]

    def analyze(
        self,
        concepts: List[Tuple[str, float]]
    ) -> Dict[str, FilterResponse]:
        """
        Apply all filters and return responses

        Args:
            concepts: List of (concept, tau) pairs

        Returns:
            Dict mapping filter_type to FilterResponse
        """
        return {
            f.filter_type: f.filter_concepts(concepts)
            for f in self.filters
        }

    @classmethod
    def standard_bank(cls) -> 'FilterBank':
        """Create standard filter bank covering τ spectrum"""
        return cls(filters=[
            LowPassFilter(cutoff=1.5),      # Ground level
            BandPassFilter(center=1.74, bandwidth=0.4),  # Everyday
            BandPassFilter(center=2.1, bandwidth=0.4),   # Meaningful
            BandPassFilter(center=2.5, bandwidth=0.4),   # Important
            HighPassFilter(cutoff=2.7),     # Transcendental
        ])

    @classmethod
    def goal_bank(cls) -> 'FilterBank':
        """Create filter bank for all goals"""
        filters = []
        for goal in ["grounded", "deep", "wisdom", "powerful", "accurate", "balanced"]:
            f = GoalFilter.for_goal(goal)
            if f:
                filters.append(f)
        return cls(filters=filters)
