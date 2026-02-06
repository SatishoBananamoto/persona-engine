"""
Determinism Manager

Provides seeded randomness for reproducible persona behavior.
All "random" persona decisions go through this manager.
"""

import random
from collections.abc import Sequence as SequenceT
from typing import Any


class DeterminismManager:
    """
    Manages deterministic randomness for persona behavior.

    ALL persona randomness should go through this manager to ensure:
    - Reproducible test scenarios (with same seed)
    - Controlled variance (humans aren't perfectly predictable)
    - Debuggable behavior (can replay exact scenario)
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize determinism manager.

        Args:
            seed: Random seed (None = non-deterministic)
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.call_count = 0  # For debugging

    def set_seed(self, seed: int) -> None:
        """Update seed (e.g., for each turn in conversation)"""
        self.seed = seed
        self.rng = random.Random(seed)
        self.call_count = 0

    # ========================================================================
    # Core Random Methods
    # ========================================================================

    def random(self) -> float:
        """Random float in [0, 1)"""
        self.call_count += 1
        return self.rng.random()

    def randint(self, a: int, b: int) -> int:
        """Random integer in [a, b] inclusive"""
        self.call_count += 1
        return self.rng.randint(a, b)

    def choice(self, sequence: SequenceT[Any]) -> Any:
        """Choose random element from sequence"""
        self.call_count += 1
        return self.rng.choice(sequence)

    def uniform(self, a: float, b: float) -> float:
        """Random float in [a, b]"""
        self.call_count += 1
        return self.rng.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Gaussian (normal) distribution"""
        self.call_count += 1
        return self.rng.gauss(mu, sigma)

    # ========================================================================
    # Persona-Specific Methods
    # ========================================================================

    def add_noise(
        self,
        value: float,
        noise_budget: float,
        distribution: str = "uniform"
    ) -> float:
        """
        Add controlled random noise to a value.

        Used for subtle human unpredictability without breaking character.

        Args:
            value: Base value
            noise_budget: Max deviation (e.g., 0.05 = ±5%)
            distribution: "uniform" or "gaussian"

        Returns:
            Value with noise added
        """
        if distribution == "gaussian":
            noise = self.gauss(0, noise_budget / 2)
        else:  # uniform
            noise = self.uniform(-noise_budget, noise_budget)

        return value + noise

    def should_trigger(self, probability: float) -> bool:
        """
        Bernoulli trial - should event trigger?

        Args:
            probability: 0-1, chance of triggering

        Returns:
            True if triggered
        """
        return self.random() < probability

    def weighted_choice(self, options: dict[str, float]) -> str:
        """
        Choose from weighted options deterministically.

        CRITICAL: Sorts keys before iteration to ensure determinism
        regardless of dict insertion order.

        Args:
            options: Dict of option -> weight

        Returns:
            Selected option
        """
        # Normalize weights
        total = sum(options.values())
        if total == 0:
            # All weights zero - return first alphabetically
            return sorted(options.keys())[0]

        normalized = {k: v/total for k, v in options.items()}

        # CRITICAL: Sort keys for deterministic iteration
        sorted_items = sorted(normalized.items(), key=lambda x: x[0])

        # Choose
        rand = self.random()
        cumulative = 0.0

        for option, weight in sorted_items:
            cumulative += weight
            if rand < cumulative:
                return option

        # Fallback (shouldn't reach here except for floating point rounding)
        return sorted_items[-1][0]

    def jitter_value(
        self,
        target: float,
        min_val: float,
        max_val: float,
        jitter_amount: float = 0.1
    ) -> float:
        """
        Add slight random jitter to a target value within bounds.

        Useful for trait-based calculations that shouldn't be perfectly deterministic.

        Args:
            target: Target value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            jitter_amount: How much to jitter (0-1, fraction of range)

        Returns:
            Jittered value within bounds
        """
        value_range = max_val - min_val
        jitter = self.uniform(-jitter_amount * value_range, jitter_amount * value_range)

        result = target + jitter
        return max(min_val, min(max_val, result))

    # ========================================================================
    # State & Debugging
    # ========================================================================

    def get_state(self) -> dict:
        """Get current state (for debugging/logging)"""
        return {
            "seed": self.seed,
            "call_count": self.call_count,
            "deterministic": self.seed is not None
        }

    def reset_call_count(self) -> None:
        """Reset call counter (e.g., at start of new turn)"""
        self.call_count = 0
