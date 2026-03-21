"""
Fact Store — stores and retrieves concrete user information.

Facts are immutable records. Updates create new records with higher
confidence, superseding older ones for the same category.

Capacity-bounded with LRU eviction.
"""

from __future__ import annotations

from persona_engine.memory.models import Fact, MemorySource, MemoryType


class FactStore:
    """
    Stores factual information about the user.

    Supports:
    - Store facts with confidence and privacy levels
    - Retrieve facts by category or keyword
    - Supersede outdated facts (new record wins if higher confidence)
    - Respect privacy thresholds when retrieving
    - Capacity-bounded with LRU eviction (oldest facts evicted first)
    """

    def __init__(self, max_capacity: int = 500) -> None:
        self._facts: list[Fact] = []
        self._by_category: dict[str, list[Fact]] = {}
        self._max_capacity = max_capacity

    def store(self, fact: Fact) -> None:
        """Store a new fact, evicting oldest if at capacity."""
        if len(self._facts) >= self._max_capacity:
            self._evict_oldest()
        self._facts.append(fact)
        cat = fact.category.lower()
        if cat not in self._by_category:
            self._by_category[cat] = []
        self._by_category[cat].append(fact)

    def _evict_oldest(self) -> None:
        """Remove the oldest fact (lowest turn_created)."""
        if not self._facts:
            return
        # Find oldest by turn_created
        oldest = min(self._facts, key=lambda f: f.turn_created)
        self._facts.remove(oldest)
        # Clean up category index
        cat = oldest.category.lower()
        if cat in self._by_category:
            cat_list = self._by_category[cat]
            if oldest in cat_list:
                cat_list.remove(oldest)
            if not cat_list:
                del self._by_category[cat]

    def get_by_category(
        self,
        category: str,
        current_turn: int = 0,
        min_confidence: float = 0.0,
        max_privacy: float = 1.0,
    ) -> list[Fact]:
        """
        Retrieve facts by category.

        Args:
            category: Fact category (e.g. "occupation", "location")
            current_turn: Current turn for confidence decay
            min_confidence: Minimum decayed confidence threshold
            max_privacy: Maximum privacy level to return

        Returns:
            Matching facts, most recent first
        """
        candidates = self._by_category.get(category.lower(), [])
        results = []
        for fact in candidates:
            if fact.privacy_level > max_privacy:
                continue
            if current_turn > 0 and fact.decayed_confidence(current_turn) < min_confidence:
                continue
            results.append(fact)
        return sorted(results, key=lambda f: f.turn_created, reverse=True)

    def search(
        self,
        query: str,
        current_turn: int = 0,
        min_confidence: float = 0.0,
        max_privacy: float = 1.0,
    ) -> list[Fact]:
        """
        Search facts by keyword match against content.

        Args:
            query: Keyword to search for
            current_turn: Current turn for confidence decay
            min_confidence: Minimum decayed confidence
            max_privacy: Max privacy level

        Returns:
            Matching facts, highest confidence first
        """
        query_lower = query.lower()
        results = []
        for fact in self._facts:
            if fact.privacy_level > max_privacy:
                continue
            if current_turn > 0 and fact.decayed_confidence(current_turn) < min_confidence:
                continue
            if query_lower in fact.content.lower() or query_lower in fact.category.lower():
                results.append(fact)
        return sorted(
            results,
            key=lambda f: f.decayed_confidence(current_turn) if current_turn > 0 else f.confidence,
            reverse=True,
        )

    def get_best_fact(self, category: str, current_turn: int = 0) -> Fact | None:
        """Get the most confident, most recent fact for a category."""
        facts = self.get_by_category(category, current_turn=current_turn)
        return facts[0] if facts else None

    @property
    def count(self) -> int:
        return len(self._facts)

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    @property
    def categories(self) -> list[str]:
        return list(self._by_category.keys())

    def all_facts(self) -> list[Fact]:
        """Return all stored facts."""
        return list(self._facts)
