"""
Relationship Store — tracks trust and rapport dynamics.

Maintains running trust and rapport scores that evolve based on
conversation events (agreement, challenge, disclosure, etc.).

Capacity-bounded: when events exceed max_capacity, oldest events are
evicted and their deltas are folded into the base trust/rapport values
to preserve accuracy.
"""

from __future__ import annotations

from persona_engine.memory.models import MemorySource, MemoryType, RelationshipMemory


class RelationshipStore:
    """
    Tracks the evolving relationship between persona and user.

    Running scores:
    - trust: How much the persona trusts the user's claims/expertise (0-1)
    - rapport: How comfortable/connected the conversation feels (0-1)

    Each RelationshipMemory records a trust/rapport delta event.
    Running scores = base + sum of event deltas.

    When events are evicted, their deltas are folded into the base values
    to preserve the accumulated relationship state. Trust and rapport
    access is O(1) via cached running totals.
    """

    def __init__(
        self,
        initial_trust: float = 0.5,
        initial_rapport: float = 0.3,
        max_capacity: int = 50,
    ) -> None:
        self._events: list[RelationshipMemory] = []
        self._base_trust = initial_trust
        self._base_rapport = initial_rapport
        self._max_capacity = max_capacity
        # Cached running totals for O(1) access
        self._cached_trust_delta: float = 0.0
        self._cached_rapport_delta: float = 0.0

    def record_event(self, event: RelationshipMemory) -> None:
        """Record a relationship-affecting event, evicting oldest if at capacity."""
        if len(self._events) >= self._max_capacity:
            self._evict_oldest()
        self._events.append(event)
        self._cached_trust_delta += event.trust_delta
        self._cached_rapport_delta += event.rapport_delta

    def _evict_oldest(self) -> None:
        """Evict the oldest event, folding its deltas into base values."""
        if not self._events:
            return
        oldest = self._events.pop(0)
        # Fold the evicted event's deltas into base values
        self._base_trust += oldest.trust_delta
        self._base_rapport += oldest.rapport_delta
        # Update cached deltas (the evicted event is no longer in the sum)
        self._cached_trust_delta -= oldest.trust_delta
        self._cached_rapport_delta -= oldest.rapport_delta

    @property
    def trust(self) -> float:
        """Current trust level (0-1), clamped. O(1) via cached totals."""
        total = self._base_trust + self._cached_trust_delta
        return max(0.0, min(1.0, total))

    @property
    def rapport(self) -> float:
        """Current rapport level (0-1), clamped. O(1) via cached totals."""
        total = self._base_rapport + self._cached_rapport_delta
        return max(0.0, min(1.0, total))

    def recent_events(self, n: int = 5) -> list[RelationshipMemory]:
        """Get the N most recent relationship events."""
        return self._events[-n:] if self._events else []

    def trust_trend(self, window: int = 5) -> float:
        """
        Trust trend over recent events.

        Returns:
            Positive = trust increasing, negative = decreasing, 0 = stable
        """
        recent = self._events[-window:] if len(self._events) >= window else self._events
        if not recent:
            return 0.0
        return sum(e.trust_delta for e in recent)

    def rapport_trend(self, window: int = 5) -> float:
        """Rapport trend over recent events."""
        recent = self._events[-window:] if len(self._events) >= window else self._events
        if not recent:
            return 0.0
        return sum(e.rapport_delta for e in recent)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    def summary(self) -> dict[str, float]:
        """Get relationship summary."""
        return {
            "trust": self.trust,
            "rapport": self.rapport,
            "trust_trend": self.trust_trend(),
            "rapport_trend": self.rapport_trend(),
            "events": self.event_count,
        }
