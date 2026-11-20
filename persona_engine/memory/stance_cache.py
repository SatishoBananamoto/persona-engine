"""
Stance Cache

Maintains stance consistency across conversation turns to prevent flip-flopping.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class CachedStance:
    """
    Stored stance on a topic with metadata for decay/reconsideration.
    """
    stance: str
    rationale_seeds: str
    elasticity: float
    confidence: float
    created_turn: int
    decay_rate: float = 0.1  # Decays over ~10 turns

    def is_expired(self, current_turn: int) -> bool:
        """Check if cached stance has decayed"""
        age = current_turn - self.created_turn
        strength = 1.0 - (age * self.decay_rate)
        return strength <= 0.0

    def get_strength(self, current_turn: int) -> float:
        """Get current strength of cached stance (0-1)"""
        age = current_turn - self.created_turn
        strength = 1.0 - (age * self.decay_rate)
        return max(0.0, strength)


class StanceCache:
    """
    Remembers stances on topics to maintain consistency.

    Stance persists unless:
    - New evidence contradicts it
    - Elasticity allows reconsideration
    - Significant time has passed (decay)
    """

    def __init__(self) -> None:
        self.cache: dict[str, CachedStance] = {}

    def _make_key(self, topic_signature: str, interaction_mode: str) -> str:
        """Create cache key from topic and mode"""
        return f"{topic_signature}::{interaction_mode}"

    def get_stance(
        self,
        topic_signature: str,
        interaction_mode: str,
        current_turn: int
    ) -> CachedStance | None:
        """
        Retrieve cached stance if exists and not expired.

        Args:
            topic_signature: Topic identifier (hash/key)
            interaction_mode: Interaction context
            current_turn: Current turn number

        Returns:
            CachedStance if found and valid, else None
        """
        key = self._make_key(topic_signature, interaction_mode)
        cached = self.cache.get(key)

        if cached and not cached.is_expired(current_turn):
            return cached
        elif cached and cached.is_expired(current_turn):
            # Clean up expired entry
            del self.cache[key]

        return None

    def store_stance(
        self,
        topic_signature: str,
        interaction_mode: str,
        stance: str,
        rationale: str,
        elasticity: float,
        confidence: float,
        turn_number: int,
        decay_rate: float = 0.1
    ) -> None:
        """
        Store stance for future consistency.

        Args:
            topic_signature: Topic identifier
            interaction_mode: Interaction context
            stance: The stance taken
            rationale: Reasoning for stance
            elasticity: How flexible this stance is
            confidence: How confident in this stance
            turn_number: When stance was formed
            decay_rate: How quickly stance decays per turn
        """
        key = self._make_key(topic_signature, interaction_mode)
        self.cache[key] = CachedStance(
            stance=stance,
            rationale_seeds=rationale,
            elasticity=elasticity,
            confidence=confidence,
            created_turn=turn_number,
            decay_rate=decay_rate
        )

    def should_reconsider(
        self,
        cached: CachedStance,
        new_evidence_strength: float,
        current_elasticity: float,
        current_turn: int
    ) -> bool:
        """
        Decide if cached stance should be reconsidered.

        High elasticity + strong new evidence → reconsider
        Low elasticity + high confidence → stick with stance
        Decayed stance → more likely to reconsider

        Args:
            cached: Cached stance data
            new_evidence_strength: Strength of contradictory evidence (0-1)
            current_elasticity: Current elasticity (may have changed)
            current_turn: Current turn number

        Returns:
            True if should reconsider stance
        """
        # Decay weakens commitment to cached stance
        cache_strength = cached.get_strength(current_turn)

        # Calculate reconsideration probability
        reconsider_prob = (
            (new_evidence_strength * 0.5) +     # New evidence pushes reconsideration
            (current_elasticity * 0.3) +        # High elasticity → open to change
            ((1.0 - cache_strength) * 0.3) -    # Decayed stance → reconsider
            (cached.confidence * 0.2)           # High confidence → resist change
        )

        return reconsider_prob > 0.5

    def invalidate_topic(self, topic_signature: str) -> None:
        """
        Invalidate all stances on a topic (e.g., after major new information).

        Args:
            topic_signature: Topic to invalidate
        """
        keys_to_remove = [
            k for k in self.cache.keys()
            if k.startswith(topic_signature + "::")
        ]
        for key in keys_to_remove:
            del self.cache[key]

    def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for debugging"""
        return {
            "total_cached_stances": len(self.cache),
            "topics": list(set(k.split("::")[0] for k in self.cache.keys()))
        }
