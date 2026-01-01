"""
Semantic Oscillators: Resonance in Meaning Space
=================================================

Based on PT1 (First-Order Lag) Dynamics
----------------------------------------

The fundamental equation (capacitor charging / semantic saturation):

    b/ν = (b/ν)_max × (1 - e^(-ν/τ_ν))

This is identical to:
    V(t) = V_max × (1 - e^(-t/RC))

Key insight: Semantic space saturates like a capacitor charges.

RC Circuit ↔ Semantic Space Mapping
------------------------------------

| Circuit       | Semantic                        | Formula              |
|---------------|--------------------------------|----------------------|
| Voltage V     | Meaning saturation             | b/ν                  |
| V_max         | Maximum meaning capacity       | (b/ν)_max            |
| Time t        | Observations processed         | ν (nouns)            |
| Time const RC | Saturation constant            | τ_ν = 42,921 nouns   |
| Resistance R  | τ (abstraction resistance)     | τ ∈ [1, 6]           |
| Capacitance C | Semantic capacity (degree)     | degree/max_degree    |
| Inductance L  | Semantic inertia (1/variety)   | 1/variety            |
| Current I     | Meaning flow                   | transition_weight    |

Oscillation Formula
-------------------

ω₀ = 1/√(LC)

Where:
  L = semantic inertia (resistance to τ change) = 1/variety
  C = semantic capacity (connection density) = degree/max_degree

The 63.21% Rule (Euler's constant)
-----------------------------------

At one time constant τ:
  - Saturation = 1 - 1/e = 63.21%
  - Each subsequent τ captures 63.21% of REMAINING

Cascade:
  τ=1: 63.2% captured
  τ=2: 86.5% captured  (63.2% + 23.3%)
  τ=3: 95.1% captured  (86.5% + 8.6%)
  τ=4: 98.2% captured
  τ=5: 99.4% captured

Concepts have natural resonance frequencies.
Questions can "resonate" with matching concepts.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

# Euler's number - fundamental constant
E = 2.718281828459045

# PT1 time constant from empirical data (16,500 books, 6M bonds)
TAU_SATURATION = 42921  # nouns
MAX_BOND_RATIO = 40.5   # bonds per noun (asymptote)

# Semantic temperature from Euler-Temperature Law (H1)
KT_NATURAL = np.exp(-1/5)  # = e^(-1/5) ≈ 0.8187


@dataclass
class PT1Dynamics:
    """
    First-Order Lag (PT1) Dynamics for Semantic Saturation

    Models how semantic space "fills up" like a capacitor charges.

    Differential equation:
        τ × dV/dt + V = V_max

    Solution:
        V(t) = V_max × (1 - e^(-t/τ))

    Semantic interpretation:
        - V = current saturation level
        - V_max = maximum capacity
        - τ = time constant (how fast we learn)
        - t = observations processed
    """

    V_max: float = 1.0        # Maximum capacity (normalized)
    tau: float = 1.0          # Time constant
    V_current: float = 0.0    # Current saturation level

    def saturation_at(self, t: float) -> float:
        """
        Saturation level at time t

        V(t) = V_max × (1 - e^(-t/τ))
        """
        return self.V_max * (1 - np.exp(-t / self.tau))

    def saturation_fraction(self, t: float) -> float:
        """Fraction of maximum: V(t) / V_max"""
        return 1 - np.exp(-t / self.tau)

    def remaining_capacity(self, t: float) -> float:
        """How much capacity remains: V_max - V(t)"""
        return self.V_max * np.exp(-t / self.tau)

    def rate_of_change(self, t: float) -> float:
        """
        Instantaneous rate: dV/dt = (V_max - V) / τ

        Rate is proportional to remaining capacity!
        """
        V = self.saturation_at(t)
        return (self.V_max - V) / self.tau

    def time_to_fraction(self, fraction: float) -> float:
        """
        Time to reach given fraction of V_max

        t = -τ × ln(1 - fraction)
        """
        if fraction >= 1.0:
            return float('inf')
        if fraction <= 0.0:
            return 0.0
        return -self.tau * np.log(1 - fraction)

    @property
    def time_to_63_percent(self) -> float:
        """Time to reach 63.21% (one time constant)"""
        return self.tau

    @property
    def time_to_95_percent(self) -> float:
        """Time to reach 95% (three time constants)"""
        return 3 * self.tau

    @property
    def time_to_99_percent(self) -> float:
        """Time to reach 99% (about 4.6 time constants)"""
        return -self.tau * np.log(0.01)

    def cascade_fractions(self, n_steps: int = 5) -> List[float]:
        """
        Cascade capture fractions

        Each step captures 63.21% of remaining.
        Returns cumulative fractions.
        """
        fractions = []
        for i in range(1, n_steps + 1):
            # At t = i×τ, cumulative fraction is 1 - (1/e)^i
            cumulative = 1 - np.exp(-i)
            fractions.append(cumulative)
        return fractions


@dataclass
class RCCircuit:
    """
    RC Circuit model for semantic navigation

    Represents a concept as an RC element with:
    - R = τ (abstraction level as resistance)
    - C = capacity (degree-based)

    Time constant: τ_RC = R × C
    """

    R: float  # Resistance (τ, abstraction)
    C: float  # Capacitance (connectivity)

    @property
    def time_constant(self) -> float:
        """τ = RC"""
        return self.R * self.C

    @property
    def impedance_at_dc(self) -> float:
        """At DC (ω=0): Z = ∞ (capacitor blocks DC)"""
        return float('inf')

    def impedance_at(self, omega: float) -> complex:
        """
        Complex impedance at frequency ω

        Z = R + 1/(jωC) = R - j/(ωC)
        """
        if omega <= 0:
            return complex(self.R, float('inf'))
        return complex(self.R, -1 / (omega * self.C))

    def transfer_function(self, omega: float) -> complex:
        """
        Transfer function H(jω) = 1 / (1 + jωRC)

        This is the frequency response of a low-pass filter.
        """
        if omega <= 0:
            return complex(1, 0)
        j_omega_rc = complex(0, omega * self.time_constant)
        return 1 / (1 + j_omega_rc)

    def gain_at(self, omega: float) -> float:
        """Magnitude of transfer function |H(jω)|"""
        return abs(self.transfer_function(omega))

    def phase_at(self, omega: float) -> float:
        """Phase of transfer function arg(H(jω)) in radians"""
        return np.angle(self.transfer_function(omega))

    def cutoff_frequency(self) -> float:
        """
        -3dB cutoff frequency: ω_c = 1/(RC)

        At this frequency, gain = 1/√2 ≈ 0.707
        """
        if self.time_constant <= 0:
            return float('inf')
        return 1 / self.time_constant

    def is_low_pass(self) -> bool:
        """RC circuit is inherently a low-pass filter"""
        return True


@dataclass
class RLCCircuit:
    """
    RLC Circuit model for semantic resonance

    R = τ (abstraction resistance)
    L = 1/variety (semantic inertia)
    C = degree/max_degree (semantic capacity)

    Resonance: ω₀ = 1/√(LC)
    Damping: ζ = R/(2√(L/C))
    """

    R: float  # Resistance
    L: float  # Inductance
    C: float  # Capacitance

    @property
    def resonance_frequency(self) -> float:
        """Natural frequency: ω₀ = 1/√(LC)"""
        if self.L <= 0 or self.C <= 0:
            return 0.0
        return 1.0 / np.sqrt(self.L * self.C)

    @property
    def damping_ratio(self) -> float:
        """Damping ratio: ζ = R/(2√(L/C))"""
        if self.L <= 0 or self.C <= 0:
            return 0.0
        return self.R / (2 * np.sqrt(self.L / self.C))

    @property
    def quality_factor(self) -> float:
        """Q = 1/(2ζ) = √(L/C) / R"""
        if self.R <= 0:
            return float('inf')
        return np.sqrt(self.L / self.C) / self.R

    @property
    def is_underdamped(self) -> bool:
        """ζ < 1: oscillates"""
        return self.damping_ratio < 1.0

    @property
    def is_critically_damped(self) -> bool:
        """ζ = 1: fastest non-oscillating response"""
        return abs(self.damping_ratio - 1.0) < 0.01

    @property
    def is_overdamped(self) -> bool:
        """ζ > 1: slow approach, no oscillation"""
        return self.damping_ratio > 1.0

    def impedance_at(self, omega: float) -> complex:
        """
        Complex impedance: Z = R + jωL + 1/(jωC)
                            = R + j(ωL - 1/(ωC))
        """
        if omega <= 0:
            return complex(self.R, float('inf'))

        X_L = omega * self.L           # Inductive reactance
        X_C = 1 / (omega * self.C)     # Capacitive reactance

        return complex(self.R, X_L - X_C)

    def impedance_at_resonance(self) -> float:
        """At ω₀: Z = R (purely resistive)"""
        return self.R

    def damped_frequency(self) -> float:
        """
        Damped natural frequency: ω_d = ω₀ × √(1 - ζ²)

        Only meaningful if underdamped (ζ < 1)
        """
        if self.damping_ratio >= 1.0:
            return 0.0
        return self.resonance_frequency * np.sqrt(1 - self.damping_ratio ** 2)

    def response_amplitude(self, omega: float) -> float:
        """Response amplitude at driving frequency"""
        omega_0 = self.resonance_frequency
        if omega_0 <= 0:
            return 0.0

        r = omega / omega_0
        zeta = self.damping_ratio

        denominator = np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
        if denominator < 1e-10:
            return 1e10

        return 1.0 / denominator


@dataclass
class SemanticOscillator:
    """
    A concept as a resonant oscillator

    Properties:
      L: Inductance (semantic inertia)
      C: Capacitance (semantic capacity)
      ω₀: Natural resonance frequency
      Q: Quality factor (sharpness of resonance)
    """

    concept: str
    L: float           # Inductance (inertia)
    C: float           # Capacitance (connectivity)
    tau: float         # Abstraction level
    variety: int       # Adjective variety
    degree: int        # Connection degree

    @property
    def omega_0(self) -> float:
        """Natural resonance frequency ω₀ = 1/√(LC)"""
        if self.L <= 0 or self.C <= 0:
            return 0.0
        return 1.0 / np.sqrt(self.L * self.C)

    @property
    def frequency(self) -> float:
        """Frequency in Hz (normalized): f = ω₀/(2π)"""
        return self.omega_0 / (2 * np.pi)

    @property
    def period(self) -> float:
        """Period T = 1/f = 2π/ω₀"""
        if self.omega_0 <= 0:
            return float('inf')
        return 2 * np.pi / self.omega_0

    @property
    def Q_factor(self) -> float:
        """
        Quality factor: sharpness of resonance

        Q = √(L/C)

        High Q: Sharp resonance, narrow bandwidth
        Low Q: Broad resonance, wide bandwidth
        """
        if self.C <= 0:
            return float('inf')
        return np.sqrt(self.L / self.C)

    @property
    def bandwidth(self) -> float:
        """
        Resonance bandwidth Δω = ω₀/Q

        Narrow bandwidth = selective (responds to specific frequencies)
        Wide bandwidth = responsive (responds to many frequencies)
        """
        if self.Q_factor <= 0:
            return float('inf')
        return self.omega_0 / self.Q_factor

    @property
    def frequency_class(self) -> str:
        """Classify by frequency range"""
        omega = self.omega_0
        if omega < 0.5:
            return "deep"      # Slow, responds to philosophical
        elif omega < 1.0:
            return "medium"    # Moderate, general questions
        else:
            return "surface"   # Fast, specific questions

    def response_at(self, omega: float, damping: float = 0.1) -> float:
        """
        Response amplitude at driving frequency ω

        H(ω) = 1 / √[(1 - (ω/ω₀)²)² + (2ζω/ω₀)²]

        Args:
            omega: Driving frequency (from query)
            damping: Damping ratio ζ

        Returns:
            Response amplitude [0, ∞)
        """
        if self.omega_0 <= 0:
            return 0.0

        r = omega / self.omega_0
        denominator = np.sqrt((1 - r**2)**2 + (2 * damping * r)**2)

        if denominator < 1e-6:
            return 1.0 / 1e-6  # Large but finite

        return 1.0 / denominator

    def resonance_with(self, query_omega: float) -> float:
        """
        Resonance quality with query [0, 1]

        1.0 = perfect resonance (ω = ω₀)
        0.0 = no resonance
        """
        response = self.response_at(query_omega)
        max_response = self.response_at(self.omega_0)

        if max_response <= 0:
            return 0.0

        return min(1.0, response / max_response)

    def __repr__(self):
        return (f"Oscillator({self.concept}: ω₀={self.omega_0:.2f}, "
                f"Q={self.Q_factor:.2f}, class={self.frequency_class})")


def compute_inductance(variety: int, max_variety: int = 100) -> float:
    """
    Compute semantic inductance L

    L = 1/variety (normalized)

    High variety → Low L → Easy to change (low inertia)
    Low variety → High L → Resistant to change (high inertia)

    Args:
        variety: Adjective variety count
        max_variety: Maximum expected variety

    Returns:
        Inductance L ∈ (0, 1]
    """
    if variety <= 0:
        return 1.0  # Maximum inertia

    # Normalize: L = 1 - variety/max_variety
    # But invert so low variety = high L
    normalized = min(variety, max_variety) / max_variety
    return max(0.01, 1.0 - normalized + 0.01)  # Avoid zero


def compute_capacitance(degree: int, max_degree: int = 500) -> float:
    """
    Compute semantic capacitance C

    C = degree/max_degree (normalized)

    High degree → High C → Can hold many connections
    Low degree → Low C → Limited connections

    Args:
        degree: Number of connections
        max_degree: Maximum expected connections

    Returns:
        Capacitance C ∈ (0, 1]
    """
    if degree <= 0:
        return 0.01  # Minimum capacity

    normalized = min(degree, max_degree) / max_degree
    return max(0.01, normalized)


def compute_resonance_frequency(
    variety: int,
    degree: int,
    max_variety: int = 100,
    max_degree: int = 500
) -> float:
    """
    Compute natural resonance frequency ω₀ = 1/√(LC)

    Args:
        variety: Adjective variety (→ L)
        degree: Connection degree (→ C)

    Returns:
        Resonance frequency ω₀
    """
    L = compute_inductance(variety, max_variety)
    C = compute_capacitance(degree, max_degree)

    return 1.0 / np.sqrt(L * C)


def create_oscillator(
    concept: str,
    tau: float,
    variety: int,
    degree: int,
    max_variety: int = 100,
    max_degree: int = 500
) -> SemanticOscillator:
    """
    Create a semantic oscillator for a concept

    Args:
        concept: Concept name
        tau: Abstraction level
        variety: Adjective variety
        degree: Connection degree

    Returns:
        SemanticOscillator with computed L, C, ω₀
    """
    L = compute_inductance(variety, max_variety)
    C = compute_capacitance(degree, max_degree)

    return SemanticOscillator(
        concept=concept,
        L=L,
        C=C,
        tau=tau,
        variety=variety,
        degree=degree,
    )


@dataclass
class ResonanceSpectrum:
    """
    Resonance spectrum of a concept collection
    """

    oscillators: List[SemanticOscillator]
    omega_min: float
    omega_max: float
    omega_mean: float
    omega_std: float

    @property
    def frequency_distribution(self) -> Dict[str, List[SemanticOscillator]]:
        """Group oscillators by frequency class"""
        groups = {"deep": [], "medium": [], "surface": []}
        for osc in self.oscillators:
            groups[osc.frequency_class].append(osc)
        return groups

    def oscillators_at_frequency(
        self,
        omega: float,
        threshold: float = 0.5
    ) -> List[Tuple[SemanticOscillator, float]]:
        """
        Find oscillators that resonate at given frequency

        Args:
            omega: Query frequency
            threshold: Minimum resonance quality

        Returns:
            List of (oscillator, resonance_quality) pairs
        """
        resonators = []
        for osc in self.oscillators:
            quality = osc.resonance_with(omega)
            if quality >= threshold:
                resonators.append((osc, quality))

        return sorted(resonators, key=lambda x: x[1], reverse=True)


def compute_spectrum(oscillators: List[SemanticOscillator]) -> ResonanceSpectrum:
    """
    Compute resonance spectrum for a collection of oscillators
    """
    if not oscillators:
        return ResonanceSpectrum(
            oscillators=[],
            omega_min=0,
            omega_max=0,
            omega_mean=0,
            omega_std=0,
        )

    frequencies = [osc.omega_0 for osc in oscillators]

    return ResonanceSpectrum(
        oscillators=oscillators,
        omega_min=min(frequencies),
        omega_max=max(frequencies),
        omega_mean=float(np.mean(frequencies)),
        omega_std=float(np.std(frequencies)),
    )


def estimate_query_frequency(
    intent_verbs: List[str],
    verb_frequencies: Optional[Dict[str, float]] = None
) -> float:
    """
    Estimate query frequency from intent verbs

    Different verbs suggest different "speeds" of thinking:
    - "understand", "contemplate" → low ω (deep, slow)
    - "find", "get", "know" → medium ω
    - "see", "check", "look" → high ω (surface, fast)

    Args:
        intent_verbs: Verbs from query
        verb_frequencies: Optional mapping verb → ω

    Returns:
        Estimated query frequency
    """
    default_frequencies = {
        # Deep (low ω)
        "understand": 0.2,
        "contemplate": 0.1,
        "ponder": 0.15,
        "meditate": 0.1,
        "realize": 0.25,

        # Medium
        "know": 0.5,
        "learn": 0.4,
        "discover": 0.45,
        "find": 0.6,
        "think": 0.5,
        "feel": 0.4,
        "believe": 0.35,

        # Surface (high ω)
        "see": 0.9,
        "look": 0.85,
        "check": 0.95,
        "get": 0.8,
        "have": 0.75,
        "use": 0.8,
        "do": 0.7,
    }

    if verb_frequencies:
        frequencies = verb_frequencies
    else:
        frequencies = default_frequencies

    if not intent_verbs:
        return 0.5  # Default medium

    omega_sum = 0.0
    count = 0

    for verb in intent_verbs:
        verb_lower = verb.lower()
        if verb_lower in frequencies:
            omega_sum += frequencies[verb_lower]
            count += 1

    if count == 0:
        return 0.5  # Default medium

    return omega_sum / count


class OscillatorBank:
    """
    Bank of semantic oscillators for resonance matching
    """

    def __init__(self):
        self.oscillators: Dict[str, SemanticOscillator] = {}

    def add(self, oscillator: SemanticOscillator):
        """Add an oscillator to the bank"""
        self.oscillators[oscillator.concept] = oscillator

    def get(self, concept: str) -> Optional[SemanticOscillator]:
        """Get oscillator for a concept"""
        return self.oscillators.get(concept)

    def find_resonant(
        self,
        query_omega: float,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[SemanticOscillator, float]]:
        """
        Find concepts that resonate with query frequency

        Args:
            query_omega: Query frequency
            threshold: Minimum resonance quality
            limit: Maximum results

        Returns:
            List of (oscillator, resonance) pairs
        """
        resonators = []

        for osc in self.oscillators.values():
            quality = osc.resonance_with(query_omega)
            if quality >= threshold:
                resonators.append((osc, quality))

        resonators.sort(key=lambda x: x[1], reverse=True)
        return resonators[:limit]

    def spectrum(self) -> ResonanceSpectrum:
        """Compute spectrum of all oscillators"""
        return compute_spectrum(list(self.oscillators.values()))
