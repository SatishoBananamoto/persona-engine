"""
Preference Store — tracks learned user behavioral patterns.

Preferences strengthen or weaken over time based on repeated signals.
Capacity-bounded by unique preference count.
"""

from __future__ import annotations

from persona_engine.memory.models import MemorySource, MemoryType, Preference


class PreferenceStore:
    """
    Tracks user preferences inferred from conversation behavior.

    Supports:
    - Store preferences with confidence and strength
    - Reinforce preferences when same signal observed again
    - Retrieve active preferences above strength threshold
    - Decay weak preferences over time
    - Capacity-bounded by unique preference keys
    """

    def __init__(self, max_capacity: int = 100) -> None:
        self._preferences: dict[str, list[Preference]] = {}  # keyed by content hash
        self._max_capacity = max_capacity

    def store(self, pref: Preference) -> None:
        """Store a new preference observation, evicting weakest if at capacity."""
        key = pref.content.lower().strip()
        if key not in self._preferences:
            # New unique preference — check capacity
            if len(self._preferences) >= self._max_capacity:
                self._evict_weakest()
            self._preferences[key] = []
        self._preferences[key].append(pref)

    def _evict_weakest(self) -> None:
        """Remove the unique preference with lowest aggregated strength."""
        if not self._preferences:
            return
        weakest_key = min(
            self._preferences.keys(),
            key=lambda k: self._aggregate_strength(k),
        )
        del self._preferences[weakest_key]

    def _aggregate_strength(self, key: str) -> float:
        """Compute aggregated strength for a preference key."""
        observations = self._preferences.get(key, [])
        if not observations:
            return 0.0
        latest = observations[-1]
        return min(1.0, latest.strength + (len(observations) - 1) * 0.1)

    def get_active(
        self,
        current_turn: int = 0,
        min_strength: float = 0.3,
    ) -> list[Preference]:
        """
        Get all active preferences above strength threshold.

        For each preference key, returns the most recent observation.
        Aggregated strength considers how many times observed.

        Args:
            current_turn: Current turn for confidence decay
            min_strength: Minimum strength to include

        Returns:
            Active preferences, strongest first
        """
        results = []
        for key, observations in self._preferences.items():
            if not observations:
                continue
            latest = observations[-1]
            agg_strength = self._aggregate_strength(key)
            if agg_strength >= min_strength:
                results.append(latest)
        return sorted(results, key=lambda p: p.strength, reverse=True)

    def get_by_tag(self, tag: str) -> list[Preference]:
        """Get preferences that have a specific tag."""
        results = []
        for observations in self._preferences.values():
            for pref in observations:
                if tag in pref.tags:
                    results.append(pref)
        return results

    def reinforcement_count(self, content: str) -> int:
        """How many times has this preference been observed?"""
        key = content.lower().strip()
        return len(self._preferences.get(key, []))

    def search(self, query: str) -> list[Preference]:
        """Search preferences by keyword."""
        query_lower = query.lower()
        results = []
        for key, observations in self._preferences.items():
            if query_lower in key:
                if observations:
                    results.append(observations[-1])
        return results

    @property
    def count(self) -> int:
        return sum(len(v) for v in self._preferences.values())

    @property
    def unique_count(self) -> int:
        return len(self._preferences)

    @property
    def max_capacity(self) -> int:
        return self._max_capacity
