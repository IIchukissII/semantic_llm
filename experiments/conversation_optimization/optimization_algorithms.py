#!/usr/bin/env python3
"""
Optimization Algorithms for Semantic Space Navigation
======================================================

Implements classic optimization algorithms to navigate semantic space:
1. Hill Climbing (greedy ascent)
2. Random Local Search (with restarts)
3. Simulated Annealing (probabilistic acceptance)

Objective functions:
- Maximize goodness (g) - projection onto "good" direction
- Minimize τ (abstraction) - prefer concrete words
- Maximize efficiency - Δg per step

Author: Quantum Semantic Research
"""

import sys
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # semantic_llm
sys.path.insert(0, str(Path(__file__).parent))  # conversation_optimization

from core.hybrid_llm import QuantumCore, OllamaRenderer, Trajectory, Transition, SemanticState


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    algorithm: str
    start_word: str
    end_word: str
    objective: str

    # Metrics
    start_score: float
    end_score: float
    improvement: float
    steps: int
    efficiency: float  # improvement / steps

    # Path
    path: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    # Algorithm-specific
    restarts: int = 0
    accepted_worse: int = 0

    def __repr__(self):
        return (f"{self.algorithm}: {self.start_word} → {self.end_word} | "
                f"Δ={self.improvement:+.3f} in {self.steps} steps | "
                f"eff={self.efficiency:.4f}")


class SemanticOptimizer:
    """
    Optimization algorithms for semantic space navigation.

    The semantic space is a landscape where:
    - States = words (positions in 16D space)
    - Neighbors = words reachable via verb transitions
    - Objective = function of (g, τ, entropy, etc.)
    """

    def __init__(self, core: Optional[QuantumCore] = None, max_edges: int = 100000,
                 dynamic_graph: bool = True, use_spin: bool = True):
        print("Initializing SemanticOptimizer...")
        self.core = core or QuantumCore()
        self.max_edges = max_edges
        self.dynamic_graph = dynamic_graph
        self.use_spin = use_spin  # Enable spin transitions (quantum tunneling)

        # Add timeout for Ollama to prevent hanging
        try:
            self.renderer = OllamaRenderer(model="qwen2.5:1.5b")
        except Exception as e:
            print(f"  Warning: Ollama renderer unavailable: {e}")
            self.renderer = None

        # Precompute j_norms for efficiency (cache expensive computation)
        print("Precomputing j_norms...")
        self.j_norms = {}
        for word, state in self.core.states.items():
            self.j_norms[word] = np.linalg.norm(state.j)
        print(f"  Cached {len(self.j_norms)} j_norm values")

        # Build verb index for dynamic lookup
        print("Building verb index...")
        self._build_verb_index()

        # Build spin index for quantum tunneling
        if use_spin:
            print("Building spin index (quantum tunneling)...")
            self._build_spin_index()

        if dynamic_graph:
            print(f"  Using DYNAMIC graph (on-demand neighbor computation)")
            print(f"  Available: {len(self.core.verb_objects)} verbs, {len(self.verb_to_objects)} indexed")
            self.neighbors = {}  # Will be computed dynamically
        else:
            # Precompute neighbor graph for efficiency
            print("Building static neighbor graph...")
            self.neighbors = self._build_neighbor_graph()
            print(f"  Graph: {len(self.neighbors)} nodes, "
                  f"{sum(len(v) for v in self.neighbors.values())} edges")

    def _build_spin_index(self):
        """
        Build spin pair index for quantum tunneling transitions.

        QUANTUM TUNNELING MODEL:
        ========================
        P(tunnel) = e^(-2κd)

        where:
          d = semantic distance between states (barrier width)
          κ = opacity (how "different" the states are)

        In our implementation:
          d = |Δτ| (abstraction distance)
          κ = (1 - j_cosine) / 2  (direction difference, 0=same, 1=opposite)

        Spin operator: word ↔ prefixed_word
        - τ approximately conserved (small d)
        - j inverted (high κ, but that's the point)
        - Tunneling probability computed from quantum formula
        """
        self.spin_index = {}  # word → (partner, delta_g, tunnel_prob, ...)
        spin_count = 0

        for word, state in self.core.states.items():
            if word in self.core.spin_pairs:
                pair = self.core.spin_pairs[word]
                # Determine partner
                partner = pair.prefixed if word == pair.base else pair.base

                if partner in self.core.states:
                    partner_state = self.core.states[partner]
                    delta_g = partner_state.goodness - state.goodness

                    # QUANTUM TUNNELING PROBABILITY
                    # d = barrier width (τ difference)
                    d = abs(pair.delta_tau)
                    # κ = opacity (direction difference)
                    # j_cosine: 1 = same direction, -1 = opposite
                    # κ: 0 = transparent, 1 = opaque
                    kappa = (1 - pair.j_cosine) / 2

                    # P(tunnel) = e^(-2κd)
                    # For spin pairs: low d (τ conserved), high κ (j flipped)
                    # This gives moderate tunneling probability
                    tunnel_prob = math.exp(-2 * kappa * d) if d > 0 else 1.0

                    self.spin_index[word] = {
                        'partner': partner,
                        'delta_g': delta_g,
                        'j_cosine': pair.j_cosine,
                        'delta_tau': pair.delta_tau,
                        'prefix': pair.prefix,
                        'tunnel_prob': tunnel_prob,
                        'd': d,
                        'kappa': kappa
                    }
                    spin_count += 1

        print(f"  Indexed {spin_count} spin transitions (quantum tunneling enabled)")
        print(f"  Tunneling formula: P = e^(-2κd)")

        # Show examples with tunneling probability
        examples = list(self.spin_index.items())[:3]
        for word, info in examples:
            print(f"    {word} ↔ {info['partner']} (Δg={info['delta_g']:+.2f}, "
                  f"P_tunnel={info['tunnel_prob']:.3f})")

    def tunnel_probability(self, word1: str, word2: str) -> float:
        """
        Calculate quantum tunneling probability between two words.

        P(tunnel) = e^(-2κd)

        where:
          d = |τ₁ - τ₂| (abstraction barrier width)
          κ = (1 - cos(j₁, j₂)) / 2 (semantic opacity)

        Returns probability in [0, 1].
        """
        if word1 not in self.core.states or word2 not in self.core.states:
            return 0.0

        s1 = self.core.states[word1]
        s2 = self.core.states[word2]

        # d = barrier width (τ difference)
        d = abs(s1.tau - s2.tau)

        # κ = opacity (direction difference in j-space)
        j1_norm = np.linalg.norm(s1.j)
        j2_norm = np.linalg.norm(s2.j)
        if j1_norm > 0 and j2_norm > 0:
            j_cos = float(np.dot(s1.j, s2.j) / (j1_norm * j2_norm))
        else:
            j_cos = 0

        kappa = (1 - j_cos) / 2  # 0 = same direction, 1 = opposite

        # Quantum tunneling probability
        tunnel_prob = math.exp(-2 * kappa * d) if d > 0 else 1.0

        return tunnel_prob

    def _build_verb_index(self):
        """Build reverse index: object → [(verb, count), ...] for fast lookup."""
        self.object_to_verbs = defaultdict(list)
        self.verb_to_objects = {}

        for verb, objects in self.core.verb_objects.items():
            valid_objects = [obj for obj in objects if obj in self.core.states]
            if valid_objects:
                self.verb_to_objects[verb] = valid_objects
                for obj in valid_objects:
                    self.object_to_verbs[obj].append(verb)

        print(f"  Indexed {len(self.verb_to_objects)} verbs -> {len(self.object_to_verbs)} objects")

    def _build_neighbor_graph(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """Build adjacency list: word → [(neighbor, verb, delta_g), ...]

        Memory safeguard: limits total edges to max_edges to prevent OOM.
        Uses stratified sampling to ensure verb diversity.
        """
        graph = defaultdict(list)
        edge_count = 0

        # CRITICAL FIX: Shuffle verbs to avoid alphabetical bias (abandon, absorb, etc.)
        verb_items = list(self.core.verb_objects.items())
        random.shuffle(verb_items)

        # Calculate edges per verb to ensure diversity
        num_verbs = len(verb_items)
        edges_per_verb = max(10, self.max_edges // num_verbs)

        print(f"  Building diverse graph: {num_verbs} verbs, ~{edges_per_verb} edges/verb")

        for verb, objects in verb_items:
            verb_edge_count = 0
            # Shuffle objects too for diversity
            objects_shuffled = list(objects)
            random.shuffle(objects_shuffled)

            for obj in objects_shuffled:
                if obj in self.core.states:
                    obj_state = self.core.states[obj]

                    # Sample words instead of iterating all
                    word_list = list(self.core.states.keys())
                    random.shuffle(word_list)

                    for word in word_list[:50]:  # Sample up to 50 source words per object
                        if word != obj:
                            # Memory safeguard: check limits
                            if edge_count >= self.max_edges:
                                print(f"  Warning: Edge limit ({self.max_edges}) reached")
                                return dict(graph)
                            if verb_edge_count >= edges_per_verb:
                                break  # Move to next verb for diversity

                            state = self.core.states[word]
                            delta_g = obj_state.goodness - state.goodness
                            graph[word].append((obj, verb, delta_g))
                            edge_count += 1
                            verb_edge_count += 1

                if verb_edge_count >= edges_per_verb:
                    break

        # Also add subject-specific transitions (high priority, no limit)
        for subj, patterns in self.core.subject_verbs.items():
            if subj in self.core.states:
                subj_state = self.core.states[subj]
                for verb, obj in patterns:
                    if obj in self.core.states:
                        if edge_count >= self.max_edges * 1.1:  # Allow 10% extra for SVO
                            break
                        obj_state = self.core.states[obj]
                        delta_g = obj_state.goodness - subj_state.goodness
                        graph[subj].append((obj, verb, delta_g))
                        edge_count += 1

        # Report verb diversity
        verbs_used = set()
        for word, edges in graph.items():
            for _, verb, _ in edges:
                verbs_used.add(verb)
        print(f"  Verb diversity: {len(verbs_used)} unique verbs in graph")

        return dict(graph)

    def get_neighbors(self, word: str, include_spin: bool = True
                       ) -> List[Tuple[str, str, float, bool]]:
        """Get neighbors of a word: [(neighbor, verb, delta_g, is_spin), ...]

        If dynamic_graph=True, computes neighbors on-demand using ALL verbs.
        If include_spin=True, adds spin partner as special "tunneling" transition.

        Returns 4-tuple: (neighbor, verb, delta_g, is_spin_transition)
        """
        if not self.dynamic_graph:
            # Add is_spin=False to static neighbors
            return [(n, v, dg, False) for n, v, dg in self.neighbors.get(word, [])]

        # DYNAMIC: Compute neighbors on-the-fly
        if word not in self.core.states:
            return []

        state = self.core.states[word]
        neighbors = []

        # 0. SPIN TRANSITION (quantum tunneling) - HIGHEST PRIORITY
        if self.use_spin and include_spin and word in self.spin_index:
            spin_info = self.spin_index[word]
            partner = spin_info['partner']
            delta_g = spin_info['delta_g']
            # Verb for spin is "become" + prefix effect
            spin_verb = f"⟲{spin_info['prefix']}"  # Special marker for spin
            neighbors.append((partner, spin_verb, delta_g, True))  # is_spin=True

        # 1. Subject-specific transitions (high priority)
        if word in self.core.subject_verbs:
            for verb, obj in self.core.subject_verbs[word]:
                if obj in self.core.states:
                    obj_state = self.core.states[obj]
                    delta_g = obj_state.goodness - state.goodness
                    neighbors.append((obj, verb, delta_g, False))

        # 2. All verb-object pairs (use ALL 2444 verbs!)
        seen = set((obj for obj, _, _, _ in neighbors))  # Avoid duplicates

        # Sample verbs for diversity (use all if under limit)
        verb_list = list(self.verb_to_objects.keys())
        if len(verb_list) > 200:
            # Sample 200 verbs randomly for diversity
            sampled_verbs = random.sample(verb_list, 200)
        else:
            sampled_verbs = verb_list

        for verb in sampled_verbs:
            objects = self.verb_to_objects.get(verb, [])
            # Sample objects if too many
            if len(objects) > 10:
                objects = random.sample(objects, 10)

            for obj in objects:
                if obj not in seen and obj in self.core.states:
                    obj_state = self.core.states[obj]
                    delta_g = obj_state.goodness - state.goodness
                    neighbors.append((obj, verb, delta_g, False))
                    seen.add(obj)

        return neighbors

    # =========================================================================
    # OBJECTIVE FUNCTIONS
    # =========================================================================

    def objective_goodness(self, word: str) -> float:
        """Maximize goodness (g)."""
        state = self.core.states.get(word)
        return state.goodness if state else -999

    def objective_neg_tau(self, word: str) -> float:
        """Minimize τ (prefer concrete words). Return negative for maximization."""
        state = self.core.states.get(word)
        return -state.tau if state else -999

    def objective_combined(self, word: str, g_weight: float = 1.0,
                           tau_weight: float = 0.3) -> float:
        """Combined: maximize g, minimize τ."""
        state = self.core.states.get(word)
        if not state:
            return -999
        return g_weight * state.goodness - tau_weight * state.tau

    def objective_energy(self, word: str) -> float:
        """
        Energy function: E = g² + ||j||²
        Maximize "semantic energy" - magnitude of semantic content.
        Uses cached j_norms for efficiency.
        """
        state = self.core.states.get(word)
        if not state:
            return -999
        # Use cached j_norm instead of computing on every call
        j_norm = self.j_norms.get(word, 0.0)
        return state.goodness**2 + j_norm**2

    # =========================================================================
    # HILL CLIMBING
    # =========================================================================

    def hill_climbing(self, start: str, objective: Callable[[str], float],
                      max_steps: int = 50, verbose: bool = True) -> OptimizationResult:
        """
        Steepest ascent hill climbing with spin transitions.

        Always moves to the best neighbor if it improves the objective.
        Stops at local maximum.
        Spin transitions (tunneling) are considered as valid moves.
        """
        if start not in self.core.states:
            raise ValueError(f"Unknown start word: {start}")

        current = start
        path = [current]
        scores = [objective(current)]
        visited = {current}
        spin_transitions = 0

        for step in range(max_steps):
            neighbors = self.get_neighbors(current)
            # Filter visited - handle 4-tuple format
            neighbors = [(n, v, dg, is_spin) for n, v, dg, is_spin in neighbors
                         if n not in visited]

            if not neighbors:
                if verbose:
                    print(f"  Step {step}: No unvisited neighbors")
                break

            # Find best neighbor (including spin transitions)
            best_neighbor = None
            best_score = scores[-1]
            best_verb = None
            best_is_spin = False

            for neighbor, verb, _, is_spin in neighbors:
                score = objective(neighbor)
                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor
                    best_verb = verb
                    best_is_spin = is_spin

            if best_neighbor is None:
                if verbose:
                    print(f"  Step {step}: Local maximum reached at '{current}'")
                break

            if verbose:
                spin_marker = " ⟲SPIN" if best_is_spin else ""
                print(f"  Step {step}: {current} --{best_verb}--> {best_neighbor} "
                      f"(score: {scores[-1]:.3f} → {best_score:.3f}){spin_marker}")

            if best_is_spin:
                spin_transitions += 1

            current = best_neighbor
            path.append(current)
            scores.append(best_score)
            visited.add(current)

        result = OptimizationResult(
            algorithm="Hill Climbing",
            start_word=start,
            end_word=current,
            objective="custom",
            start_score=scores[0],
            end_score=scores[-1],
            improvement=scores[-1] - scores[0],
            steps=len(path) - 1,
            efficiency=(scores[-1] - scores[0]) / max(1, len(path) - 1),
            path=path,
            scores=scores
        )
        result.spin_transitions = spin_transitions
        return result

    # =========================================================================
    # RANDOM LOCAL SEARCH
    # =========================================================================

    def random_local_search(self, start: str, objective: Callable[[str], float],
                            max_steps: int = 50, restarts: int = 5,
                            verbose: bool = True) -> OptimizationResult:
        """
        Random local search with restarts and spin transitions.

        Randomly picks neighbors and accepts improvements.
        Restarts from random positions to escape local maxima.
        """
        best_result = None
        all_paths = []

        for restart in range(restarts):
            # Start from original or random word
            if restart == 0:
                current = start
            else:
                current = random.choice(list(self.core.states.keys()))

            path = [current]
            scores = [objective(current)]
            visited = {current}
            spin_count = 0

            steps_per_restart = max_steps // restarts

            for step in range(steps_per_restart):
                neighbors = self.get_neighbors(current)
                neighbors = [(n, v, dg, is_spin) for n, v, dg, is_spin in neighbors
                             if n not in visited]

                if not neighbors:
                    break

                # Random selection
                neighbor, verb, _, is_spin = random.choice(neighbors)
                score = objective(neighbor)

                # Accept if better
                if score > scores[-1]:
                    current = neighbor
                    path.append(current)
                    scores.append(score)
                    visited.add(current)
                    if is_spin:
                        spin_count += 1

                    if verbose and restart == 0:
                        spin_marker = " ⟲SPIN" if is_spin else ""
                        print(f"  Step {step}: {path[-2]} --{verb}--> {current} "
                              f"(score: {scores[-2]:.3f} → {score:.3f}){spin_marker}")

            result = OptimizationResult(
                algorithm="Random Local Search",
                start_word=start,
                end_word=path[-1],
                objective="custom",
                start_score=objective(start),
                end_score=scores[-1],
                improvement=scores[-1] - objective(start),
                steps=len(path) - 1,
                efficiency=(scores[-1] - objective(start)) / max(1, len(path) - 1),
                path=path,
                scores=scores,
                restarts=restart + 1
            )
            result.spin_transitions = spin_count

            if best_result is None or result.end_score > best_result.end_score:
                best_result = result

        return best_result

    # =========================================================================
    # SIMULATED ANNEALING
    # =========================================================================

    def simulated_annealing(self, start: str, objective: Callable[[str], float],
                            max_steps: int = 100, initial_temp: float = 1.0,
                            cooling_rate: float = 0.95, spin_bonus: float = 0.3,
                            verbose: bool = True) -> OptimizationResult:
        """
        Simulated annealing with quantum spin transitions.

        Accepts worse moves with probability exp(-ΔE/T).
        Temperature decreases over time, reducing randomness.

        QUANTUM EXTENSION:
        - Spin transitions (tunneling) have boosted acceptance probability
        - Spin allows "jumping" to opposite semantic state
        - P(spin) = exp(Δg/T) * (1 + spin_bonus)
        """
        if start not in self.core.states:
            raise ValueError(f"Unknown start word: {start}")

        current = start
        path = [current]
        scores = [objective(current)]
        visited = {current}

        best_word = current
        best_score = scores[0]

        temperature = initial_temp
        accepted_worse = 0
        spin_transitions = 0  # Count quantum tunneling events

        for step in range(max_steps):
            neighbors = self.get_neighbors(current)
            neighbors = [(n, v, dg, is_spin) for n, v, dg, is_spin in neighbors
                         if n not in visited]

            if not neighbors:
                break

            # Random neighbor (with spin preference at high T)
            if temperature > 0.5 and self.use_spin:
                # At high T, prefer spin transitions (exploration)
                spin_neighbors = [n for n in neighbors if n[3]]  # is_spin=True
                if spin_neighbors and random.random() < 0.3:  # 30% chance to try spin
                    neighbor, verb, _, is_spin = random.choice(spin_neighbors)
                else:
                    neighbor, verb, _, is_spin = random.choice(neighbors)
            else:
                neighbor, verb, _, is_spin = random.choice(neighbors)

            score = objective(neighbor)
            delta = score - scores[-1]

            # Accept?
            accept = False
            if delta > 0:
                accept = True
            else:
                # Probability of accepting worse move
                # QUANTUM: Spin transitions get bonus (tunneling through barrier)
                if temperature > 0.001:
                    base_prob = math.exp(delta / temperature)
                    if is_spin:
                        # Spin transitions can "tunnel" - higher acceptance
                        prob = min(1.0, base_prob * (1 + spin_bonus))
                    else:
                        prob = base_prob
                else:
                    prob = 0

                if random.random() < prob:
                    accept = True
                    accepted_worse += 1

            if accept:
                if verbose:
                    marker = "↑" if delta > 0 else "↓"
                    spin_marker = " ⟲SPIN" if is_spin else ""
                    print(f"  Step {step} (T={temperature:.3f}): {current} --{verb}--> "
                          f"{neighbor} {marker} Δ={delta:+.3f}{spin_marker}")

                if is_spin:
                    spin_transitions += 1

                current = neighbor
                path.append(current)
                scores.append(score)
                visited.add(current)

                if score > best_score:
                    best_score = score
                    best_word = current

            # Cool down
            temperature *= cooling_rate

        result = OptimizationResult(
            algorithm="Simulated Annealing",
            start_word=start,
            end_word=best_word,
            objective="custom",
            start_score=scores[0],
            end_score=best_score,
            improvement=best_score - scores[0],
            steps=len(path) - 1,
            efficiency=(best_score - scores[0]) / max(1, len(path) - 1),
            path=path,
            scores=scores,
            accepted_worse=accepted_worse
        )
        # Add spin count to result
        result.spin_transitions = spin_transitions
        return result

    # =========================================================================
    # QUANTUM ANNEALING (Thermal + Tunneling)
    # =========================================================================

    def quantum_annealing(self, start: str, objective: Callable[[str], float],
                          max_steps: int = 100, initial_temp: float = 1.0,
                          cooling_rate: float = 0.95, tunnel_threshold: float = 0.3,
                          believe: float = 1.0, verbose: bool = True) -> OptimizationResult:
        """
        Quantum Annealing: combines thermal annealing with quantum tunneling.

        TWO FORMULAS:
        1. Thermal acceptance: P(accept) = e^(Δg/T)  — smooth transitions
        2. Quantum tunneling:  P(tunnel) = believe × e^(-2κd)  — insight/breakthrough

        BEHAVIOR (PRIORITY ORDER):
        1. FIRST: Try tunneling (if believe is high enough)
        2. FALLBACK: Use thermal exploration until tunneling possible
        - Tunneling = "insight" (jump to semantically different state)

        BELIEVE PARAMETER:
        - believe = 1.0: strong belief → tunnel often, breakthrough easily
        - believe = 0.1: weak belief → stay stuck, rare breakthroughs
        - Models belief in possibility of change/breakthrough

        This models human cognition:
        - Normal thinking = thermal exploration (gradual)
        - Insight = quantum tunneling through semantic barrier
        - Belief = capacity to attempt breakthrough
        """
        if start not in self.core.states:
            raise ValueError(f"Unknown start word: {start}")

        current = start
        path = [current]
        scores = [objective(current)]
        visited = {current}

        best_word = current
        best_score = scores[0]

        temperature = initial_temp
        accepted_worse = 0
        tunnel_events = []  # Track tunneling "insights"
        stuck_count = 0

        for step in range(max_steps):
            neighbors = self.get_neighbors(current)
            neighbors = [(n, v, dg, is_spin) for n, v, dg, is_spin in neighbors
                         if n not in visited]

            # PRIORITY 1: Try tunneling first (if believe is high enough)
            # believe parameter determines how aggressively we seek tunneling
            tunnel_attempted = False
            if self.use_spin and random.random() < believe:
                # Look for tunneling candidates
                tunnel_candidates = []
                for word in random.sample(list(self.core.states.keys()), min(300, len(self.core.states))):
                    if word not in visited and word != current:
                        p = self.tunnel_probability(current, word)
                        score = objective(word)
                        # Only tunnel if it improves our position
                        if p > tunnel_threshold and score > scores[-1]:
                            tunnel_candidates.append((word, p, score))

                if tunnel_candidates:
                    # Choose best by score, weighted by tunnel probability
                    tunnel_candidates.sort(key=lambda x: x[2] * x[1], reverse=True)
                    target, prob, new_score = tunnel_candidates[0]

                    effective_prob = min(1.0, believe * prob)
                    if random.random() < effective_prob:
                        if verbose:
                            delta = new_score - scores[-1]
                            print(f"  Step {step} ⚡TUNNEL: {current} ══════> {target} "
                                  f"(P={prob:.3f}, Δ={delta:+.3f}) [INSIGHT!]")
                        tunnel_events.append({
                            'step': step,
                            'from': current,
                            'to': target,
                            'probability': prob,
                            'delta_score': new_score - scores[-1],
                            'type': 'belief_tunnel'
                        })
                        current = target
                        path.append(current)
                        scores.append(new_score)
                        visited.add(current)
                        stuck_count = 0

                        if new_score > best_score:
                            best_score = new_score
                            best_word = current
                        tunnel_attempted = True

            if tunnel_attempted:
                temperature *= cooling_rate
                continue

            if not neighbors:
                # No neighbors and no tunneling - try spin partner escape
                if self.use_spin and current in self.spin_index:
                    spin_info = self.spin_index[current]
                    partner = spin_info['partner']
                    if partner not in visited:
                        tunnel_prob = spin_info['tunnel_prob']
                        effective_prob = min(1.0, believe * tunnel_prob)
                        if random.random() < effective_prob:
                            if verbose:
                                print(f"  Step {step} ⚡TUNNEL: {current} ════> {partner} "
                                      f"(P={tunnel_prob:.3f}) [ESCAPE]")
                            tunnel_events.append({
                                'step': step,
                                'from': current,
                                'to': partner,
                                'probability': tunnel_prob,
                                'type': 'escape'
                            })
                            current = partner
                            path.append(current)
                            scores.append(objective(current))
                            visited.add(current)
                            stuck_count = 0
                            continue
                break

            # Check if we're stuck at local maximum
            current_score = scores[-1]
            best_neighbor_score = max(objective(n) for n, _, _, _ in neighbors)

            if best_neighbor_score <= current_score:
                stuck_count += 1
            else:
                stuck_count = 0

            # If stuck, try quantum tunneling
            # believe parameter affects how quickly we attempt tunneling:
            #   believe=1.0 → stuck_threshold=1 (try immediately)
            #   believe=0.1 → stuck_threshold=30 (need to be very stuck)
            stuck_threshold = max(1, int(3 / max(0.1, believe)))
            if stuck_count >= stuck_threshold and self.use_spin:
                # Find any word we can tunnel to (not just spin partner)
                tunnel_candidates = []
                for word in list(self.core.states.keys())[:500]:  # Sample
                    if word not in visited and word != current:
                        p = self.tunnel_probability(current, word)
                        if p > tunnel_threshold:
                            tunnel_candidates.append((word, p))

                if tunnel_candidates:
                    # Choose based on tunneling probability
                    tunnel_candidates.sort(key=lambda x: -x[1])
                    target, prob = tunnel_candidates[0]

                    # Effective probability = believe × base probability
                    effective_prob = min(1.0, believe * prob)
                    if random.random() < effective_prob:
                        new_score = objective(target)
                        if verbose:
                            delta = new_score - current_score
                            print(f"  Step {step} ⚡TUNNEL: {current} ══════> {target} "
                                  f"(P={prob:.3f}, Δ={delta:+.3f}) [INSIGHT!]")
                        tunnel_events.append({
                            'step': step,
                            'from': current,
                            'to': target,
                            'probability': prob,
                            'delta_score': new_score - current_score,
                            'type': 'insight'
                        })
                        current = target
                        path.append(current)
                        scores.append(new_score)
                        visited.add(current)
                        stuck_count = 0

                        if new_score > best_score:
                            best_score = new_score
                            best_word = current
                        continue

            # Normal thermal annealing move
            neighbor, verb, _, is_spin = random.choice(neighbors)
            score = objective(neighbor)
            delta = score - scores[-1]

            accept = False
            if delta > 0:
                accept = True
            else:
                if temperature > 0.001:
                    prob = math.exp(delta / temperature)
                    if random.random() < prob:
                        accept = True
                        accepted_worse += 1

            if accept:
                if verbose:
                    marker = "↑" if delta > 0 else "↓"
                    spin_marker = " ⟲" if is_spin else ""
                    print(f"  Step {step} (T={temperature:.3f}): {current} --{verb}--> "
                          f"{neighbor} {marker} Δ={delta:+.3f}{spin_marker}")

                current = neighbor
                path.append(current)
                scores.append(score)
                visited.add(current)

                if score > best_score:
                    best_score = score
                    best_word = current

            temperature *= cooling_rate

        result = OptimizationResult(
            algorithm="Quantum Annealing",
            start_word=start,
            end_word=best_word,
            objective="custom",
            start_score=scores[0],
            end_score=best_score,
            improvement=best_score - scores[0],
            steps=len(path) - 1,
            efficiency=(best_score - scores[0]) / max(1, len(path) - 1),
            path=path,
            scores=scores,
            accepted_worse=accepted_worse
        )
        result.tunnel_events = tunnel_events
        result.spin_transitions = len(tunnel_events)

        if verbose and tunnel_events:
            print(f"\n  ⚡ INSIGHTS DETECTED: {len(tunnel_events)}")
            for event in tunnel_events:
                print(f"     {event['from']} ══> {event['to']} (P={event['probability']:.3f})")

        return result

    # =========================================================================
    # COMPARISON
    # =========================================================================

    def compare_algorithms(self, start: str, objective_name: str = "goodness",
                           max_steps: int = 30) -> Dict[str, OptimizationResult]:
        """Compare all algorithms on the same start word and objective."""

        # Select objective function
        objectives = {
            "goodness": self.objective_goodness,
            "neg_tau": self.objective_neg_tau,
            "combined": self.objective_combined,
            "energy": self.objective_energy
        }
        objective = objectives.get(objective_name, self.objective_goodness)

        print(f"\n{'='*70}")
        print(f"COMPARING ALGORITHMS")
        print(f"Start: {start}, Objective: {objective_name}, Max steps: {max_steps}")
        print(f"{'='*70}")

        results = {}

        # Hill Climbing
        print(f"\n--- Hill Climbing ---")
        results["hill_climbing"] = self.hill_climbing(
            start, objective, max_steps, verbose=True
        )

        # Random Local Search
        print(f"\n--- Random Local Search (5 restarts) ---")
        results["random_local"] = self.random_local_search(
            start, objective, max_steps, restarts=5, verbose=True
        )

        # Simulated Annealing
        print(f"\n--- Simulated Annealing ---")
        results["simulated_annealing"] = self.simulated_annealing(
            start, objective, max_steps * 2, initial_temp=1.0,
            cooling_rate=0.95, verbose=True
        )

        # Summary
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Algorithm':<25} {'End Word':<15} {'Start':<8} {'End':<8} "
              f"{'Δ':<8} {'Steps':<6} {'Eff':<8}")
        print("-" * 70)

        for name, result in results.items():
            print(f"{result.algorithm:<25} {result.end_word:<15} "
                  f"{result.start_score:<8.3f} {result.end_score:<8.3f} "
                  f"{result.improvement:<+8.3f} {result.steps:<6} "
                  f"{result.efficiency:<8.4f}")

        return results

    def visualize_path(self, result: OptimizationResult):
        """Visualize optimization path."""
        print(f"\n{'─'*60}")
        print(f"Path: {result.algorithm}")
        print(f"{'─'*60}")

        # ASCII score plot
        min_score = min(result.scores)
        max_score = max(result.scores)
        range_score = max_score - min_score if max_score > min_score else 1

        print("\nScore progression:")
        for i, (word, score) in enumerate(zip(result.path, result.scores)):
            # Normalize to 0-40 chars
            bar_len = int((score - min_score) / range_score * 40)
            bar = "█" * bar_len
            print(f"  {i:2d}. {word:<15} {score:+.3f} |{bar}")

        # Render with LLM (safely handle None renderer)
        if self.renderer is not None and hasattr(self.renderer, 'available') and self.renderer.available and len(result.path) > 2:
            print(f"\nLLM Narrative:")
            # Create mini-trajectory for rendering
            text = " → ".join(result.path[:10])  # Limit for LLM
            print(f"  Path: {text}")


def main():
    """Run optimization algorithm comparison."""
    optimizer = SemanticOptimizer()

    # Test words representing different starting points
    test_words = ["war", "fear", "darkness", "chaos"]

    # Test different objectives
    objectives = ["goodness", "combined", "energy"]

    for start_word in test_words:
        for obj in objectives:
            results = optimizer.compare_algorithms(
                start_word,
                objective_name=obj,
                max_steps=20
            )

            # Visualize best result
            best = max(results.values(), key=lambda r: r.improvement)
            optimizer.visualize_path(best)

            print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
