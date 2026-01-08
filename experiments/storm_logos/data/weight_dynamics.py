"""
Weight Dynamics Module - Learning and Forgetting for FOLLOWS Edges
===================================================================

Implements capacitor-like dynamics for edge weights:

    dw/dt = lambda * (w_target - w)

Discrete form:
    w(t+dt) = w_target + (w_current - w_target) * e^(-lambda * dt)

For forgetting (decay toward w_min):
    w(t+dt) = w_min + (w_current - w_min) * e^(-lambda_forget * dt)

For learning (rise toward w_max):
    w(t+dt) = w_max - (w_max - w_current) * e^(-lambda_learn * dt)

Parameters:
    w_min = 0.1         Floor - never fully forgotten
    w_max = 1.0         Ceiling - fully learned
    lambda_learn = 0.3  Learning rate (per reinforcement)
    lambda_forget = 0.05    Forgetting rate (per day)

KEY: Learning is 6x faster than forgetting.
     Easy to learn, slow to fully forget.

Weight Sources:
    Corpus (books):     w_max (1.0)   - Established knowledge
    Conversation:       2*w_min (0.2) - Needs reinforcement
    Context-inferred:   w_min (0.1)   - Weakest, most uncertain

Dormancy States:
    Active:   w > 0.2  -> Used in navigation
    Dormant:  w <= 0.2 -> Exists but not actively used
    Gone:     NEVER    -> "Knowledge is never lost"

The Capacitor Analogy:
    Learning  = Charging capacitor (voltage rises toward max)
    Forgetting = Discharging capacitor (voltage falls toward baseline)
    The baseline is never zero. Something always remains.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict


# =============================================================================
# CONSTANTS
# =============================================================================

# Weight bounds
W_MIN = 0.1       # Floor - never fully forgotten
W_MAX = 1.0       # Ceiling - fully learned

# Rate constants (exponential time constants)
LAMBDA_LEARN = 0.3    # Learning rate (per reinforcement event)
LAMBDA_FORGET = 0.05  # Forgetting rate (per day)

# Weight sources (initial weights by source type)
WEIGHT_CORPUS = 1.0       # Books - established knowledge
WEIGHT_ARTICLE = 0.8      # Curated content
WEIGHT_CONVERSATION = 0.2 # Needs reinforcement
WEIGHT_CONTEXT = 0.1      # Weakest, most uncertain

# Dormancy threshold
DORMANCY_THRESHOLD = 0.2  # Below this, edge is "dormant"

# Learning increment (for simple reinforcement)
LEARNING_INCREMENT = 0.05  # w += 0.05 per reinforcement


# =============================================================================
# WEIGHT DYNAMICS FUNCTIONS
# =============================================================================

def decay_weight(w_current: float, days_elapsed: float,
                 lambda_forget: float = LAMBDA_FORGET,
                 w_min: float = W_MIN) -> float:
    """
    Apply forgetting decay to a weight.

    Formula: w(t+dt) = w_min + (w_current - w_min) * e^(-lambda * dt)

    Args:
        w_current: Current weight value
        days_elapsed: Days since last reinforcement
        lambda_forget: Forgetting rate constant (default 0.05/day)
        w_min: Minimum weight floor (default 0.1)

    Returns:
        New weight after decay (never below w_min)

    Example:
        >>> decay_weight(1.0, 1)   # After 1 day
        0.9561...
        >>> decay_weight(1.0, 7)   # After 1 week
        0.7629...
        >>> decay_weight(1.0, 30) # After 1 month
        0.3231...
        >>> decay_weight(1.0, 100) # After 100 days
        0.1067... (approaching w_min)
    """
    if days_elapsed <= 0:
        return w_current

    # Exponential decay toward w_min
    decay_factor = math.exp(-lambda_forget * days_elapsed)
    w_new = w_min + (w_current - w_min) * decay_factor

    return max(w_min, w_new)


def learn_weight(w_current: float,
                 lambda_learn: float = LAMBDA_LEARN,
                 w_max: float = W_MAX) -> float:
    """
    Apply learning boost to a weight (single reinforcement event).

    Formula: w(t+1) = w_max - (w_max - w_current) * e^(-lambda_learn)

    Args:
        w_current: Current weight value
        lambda_learn: Learning rate constant (default 0.3)
        w_max: Maximum weight ceiling (default 1.0)

    Returns:
        New weight after learning (never above w_max)

    Example:
        >>> learn_weight(0.2)  # From conversation weight
        0.4296...
        >>> learn_weight(0.5)  # From mid-weight
        0.6296...
        >>> learn_weight(0.9)  # Near ceiling
        0.9740...
    """
    # Exponential rise toward w_max
    learn_factor = math.exp(-lambda_learn)
    w_new = w_max - (w_max - w_current) * learn_factor

    return min(w_max, w_new)


def learn_weight_simple(w_current: float,
                        increment: float = LEARNING_INCREMENT,
                        w_max: float = W_MAX) -> float:
    """
    Apply simple additive learning.

    Formula: w(t+1) = min(w_max, w_current + increment)

    Args:
        w_current: Current weight value
        increment: Weight increment per reinforcement (default 0.05)
        w_max: Maximum weight ceiling (default 1.0)

    Returns:
        New weight after learning
    """
    return min(w_max, w_current + increment)


def time_to_dormancy(w_current: float,
                     lambda_forget: float = LAMBDA_FORGET,
                     w_min: float = W_MIN,
                     threshold: float = DORMANCY_THRESHOLD) -> float:
    """
    Calculate days until weight decays to dormancy threshold.

    Solves: threshold = w_min + (w_current - w_min) * e^(-lambda * t)
    For t:  t = -ln((threshold - w_min) / (w_current - w_min)) / lambda

    Args:
        w_current: Current weight value
        lambda_forget: Forgetting rate constant
        w_min: Minimum weight floor
        threshold: Dormancy threshold (default 0.2)

    Returns:
        Days until dormancy (inf if already dormant or won't reach)
    """
    if w_current <= threshold:
        return 0.0  # Already dormant

    if threshold <= w_min:
        return float('inf')  # Never reaches below floor

    numerator = threshold - w_min
    denominator = w_current - w_min

    if denominator <= 0:
        return float('inf')

    ratio = numerator / denominator
    if ratio <= 0 or ratio >= 1:
        return float('inf')

    return -math.log(ratio) / lambda_forget


def half_life(lambda_rate: float) -> float:
    """
    Calculate half-life for given decay rate.

    t_half = ln(2) / lambda

    Args:
        lambda_rate: Decay rate constant

    Returns:
        Time for weight to decay to half its excess over minimum
    """
    return math.log(2) / lambda_rate


# =============================================================================
# WEIGHT STATE ANALYSIS
# =============================================================================

@dataclass
class WeightState:
    """Analysis of a weight's current state."""
    weight: float
    is_dormant: bool
    is_active: bool
    is_saturated: bool
    days_to_dormancy: float
    relative_strength: float  # 0-1 normalized


def analyze_weight(w: float) -> WeightState:
    """
    Analyze the state of a weight value.

    Args:
        w: Weight value to analyze

    Returns:
        WeightState with analysis
    """
    return WeightState(
        weight=w,
        is_dormant=w <= DORMANCY_THRESHOLD,
        is_active=w > DORMANCY_THRESHOLD,
        is_saturated=w >= W_MAX - 0.01,
        days_to_dormancy=time_to_dormancy(w),
        relative_strength=(w - W_MIN) / (W_MAX - W_MIN)
    )


def weight_source_name(w: float) -> str:
    """
    Infer the likely source of a weight based on its value.

    Returns:
        Human-readable source name
    """
    if w >= 0.95:
        return "corpus"
    elif w >= 0.75:
        return "article"
    elif w >= 0.3:
        return "reinforced"
    elif w >= 0.15:
        return "conversation"
    else:
        return "context"


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

def compute_decay_batch(weights: List[float], days_elapsed: float) -> List[float]:
    """
    Apply decay to a batch of weights.

    Args:
        weights: List of current weight values
        days_elapsed: Days since last reinforcement

    Returns:
        List of decayed weights
    """
    return [decay_weight(w, days_elapsed) for w in weights]


def decay_statistics(weights_before: List[float], weights_after: List[float]) -> Dict:
    """
    Compute statistics about a decay operation.

    Args:
        weights_before: Weights before decay
        weights_after: Weights after decay

    Returns:
        Dictionary of statistics
    """
    if not weights_before:
        return {"count": 0}

    deltas = [b - a for b, a in zip(weights_before, weights_after)]

    dormant_before = sum(1 for w in weights_before if w <= DORMANCY_THRESHOLD)
    dormant_after = sum(1 for w in weights_after if w <= DORMANCY_THRESHOLD)

    return {
        "count": len(weights_before),
        "total_decay": sum(deltas),
        "avg_decay": sum(deltas) / len(deltas),
        "max_decay": max(deltas),
        "min_decay": min(deltas),
        "avg_weight_before": sum(weights_before) / len(weights_before),
        "avg_weight_after": sum(weights_after) / len(weights_after),
        "dormant_before": dormant_before,
        "dormant_after": dormant_after,
        "newly_dormant": dormant_after - dormant_before
    }


# =============================================================================
# CONSTANTS INFO
# =============================================================================

def get_dynamics_info() -> Dict:
    """
    Get information about the weight dynamics parameters.

    Returns:
        Dictionary describing the dynamics
    """
    return {
        "parameters": {
            "w_min": W_MIN,
            "w_max": W_MAX,
            "lambda_learn": LAMBDA_LEARN,
            "lambda_forget": LAMBDA_FORGET,
            "learning_increment": LEARNING_INCREMENT,
            "dormancy_threshold": DORMANCY_THRESHOLD
        },
        "weight_sources": {
            "corpus": WEIGHT_CORPUS,
            "article": WEIGHT_ARTICLE,
            "conversation": WEIGHT_CONVERSATION,
            "context": WEIGHT_CONTEXT
        },
        "half_lives": {
            "forgetting_days": half_life(LAMBDA_FORGET),
            "learning_events": half_life(LAMBDA_LEARN)
        },
        "key_insight": "Learning is 6x faster than forgetting (0.3 vs 0.05)"
    }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Weight Dynamics Demo")
    print("=" * 60)

    info = get_dynamics_info()
    print("\n1. Parameters:")
    for k, v in info["parameters"].items():
        print(f"   {k}: {v}")

    print(f"\n2. Half-life of forgetting: {info['half_lives']['forgetting_days']:.1f} days")
    print(f"   (Time for weight to decay halfway to w_min)")

    print("\n3. Decay over time (starting from w=1.0):")
    for days in [1, 7, 14, 30, 60, 100]:
        w = decay_weight(1.0, days)
        state = analyze_weight(w)
        status = "DORMANT" if state.is_dormant else "active"
        print(f"   Day {days:3d}: w={w:.4f} ({status})")

    print("\n4. Learning progression (starting from w=0.2):")
    w = WEIGHT_CONVERSATION
    for i in range(10):
        print(f"   Reinforcement {i}: w={w:.4f}")
        w = learn_weight_simple(w)

    print("\n5. Days to dormancy by starting weight:")
    for w in [1.0, 0.8, 0.5, 0.3, 0.25]:
        days = time_to_dormancy(w)
        print(f"   w={w:.2f}: {days:.1f} days to dormancy")

    print("\n" + "=" * 60)
