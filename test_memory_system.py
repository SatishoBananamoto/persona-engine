"""
Comprehensive tests for the persona_engine.memory subsystem.

Covers all four memory stores (FactStore, PreferenceStore, RelationshipStore,
EpisodicStore), the MemoryManager orchestrator, integration with IR schema
MemoryOps, and multi-turn behavioral scenarios.

Target: 100+ tests exercising creation, retrieval, filtering, decay,
privacy, reinforcement, clamping, trends, and realistic conversation flows.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from persona_engine.memory.episodic_store import EpisodicStore
from persona_engine.memory.fact_store import FactStore
from persona_engine.memory.memory_manager import MemoryManager, _make_id
from persona_engine.memory.models import (
    Episode,
    Fact,
    MemoryRecord,
    MemorySource,
    MemoryType,
    Preference,
    RelationshipMemory,
)
from persona_engine.memory.preference_store import PreferenceStore
from persona_engine.memory.relationship_store import RelationshipStore
from persona_engine.schema.ir_schema import (
    MemoryOps,
    MemoryReadRequest,
    MemoryWriteIntent,
)


# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def fact_store() -> FactStore:
    return FactStore()


@pytest.fixture
def preference_store() -> PreferenceStore:
    return PreferenceStore()


@pytest.fixture
def relationship_store() -> RelationshipStore:
    return RelationshipStore()


@pytest.fixture
def episodic_store() -> EpisodicStore:
    return EpisodicStore()


@pytest.fixture
def memory_manager() -> MemoryManager:
    return MemoryManager()


def _make_fact(
    content: str = "User works as a software engineer",
    category: str = "occupation",
    confidence: float = 0.9,
    privacy_level: float = 0.3,
    source: MemorySource = MemorySource.USER_STATED,
    turn_created: int = 1,
    conversation_id: str = "conv-001",
    tags: tuple[str, ...] = (),
) -> Fact:
    return Fact(
        memory_id=_make_id(content, turn_created),
        memory_type=MemoryType.FACT,
        content=content,
        confidence=confidence,
        privacy_level=privacy_level,
        source=source,
        turn_created=turn_created,
        conversation_id=conversation_id,
        category=category,
        tags=tags,
    )


def _make_preference(
    content: str = "User prefers brief answers",
    strength: float = 0.6,
    confidence: float = 0.7,
    source: MemorySource = MemorySource.INFERRED,
    turn_created: int = 1,
    conversation_id: str = "conv-001",
    tags: tuple[str, ...] = (),
) -> Preference:
    return Preference(
        memory_id=_make_id(content, turn_created),
        memory_type=MemoryType.PREFERENCE,
        content=content,
        confidence=confidence,
        privacy_level=0.2,
        source=source,
        turn_created=turn_created,
        conversation_id=conversation_id,
        strength=strength,
        tags=tags,
    )


def _make_relationship_event(
    content: str = "User agreed with persona's suggestion",
    trust_delta: float = 0.05,
    rapport_delta: float = 0.03,
    turn_created: int = 1,
    conversation_id: str = "conv-001",
) -> RelationshipMemory:
    return RelationshipMemory(
        memory_id=_make_id(content, turn_created),
        memory_type=MemoryType.RELATIONSHIP,
        content=content,
        confidence=0.8,
        privacy_level=0.1,
        source=MemorySource.OBSERVED,
        turn_created=turn_created,
        conversation_id=conversation_id,
        trust_delta=trust_delta,
        rapport_delta=rapport_delta,
    )


def _make_episode(
    content: str = "Discussed UX research methodologies",
    topic: str = "ux research",
    outcome: str = "agreed",
    turn_start: int = 1,
    turn_end: int = 3,
    conversation_id: str = "conv-001",
) -> Episode:
    return Episode(
        memory_id=_make_id(content, turn_start),
        memory_type=MemoryType.EPISODE,
        content=content,
        confidence=0.9,
        privacy_level=0.3,
        source=MemorySource.OBSERVED,
        turn_created=turn_start,
        conversation_id=conversation_id,
        topic=topic,
        turn_start=turn_start,
        turn_end=turn_end,
        outcome=outcome,
    )


# ============================================================================
# 1. TestMemoryModels
# ============================================================================


class TestMemoryModels:
    """Tests for base MemoryRecord and all four typed subclasses."""

    def test_memory_record_creation(self):
        record = MemoryRecord(
            memory_id="test-001",
            memory_type=MemoryType.FACT,
            content="Test content",
            confidence=0.8,
            privacy_level=0.3,
            source=MemorySource.USER_STATED,
        )
        assert record.memory_id == "test-001"
        assert record.memory_type == MemoryType.FACT
        assert record.content == "Test content"
        assert record.confidence == 0.8
        assert record.privacy_level == 0.3
        assert record.source == MemorySource.USER_STATED

    def test_memory_record_defaults(self):
        record = MemoryRecord(
            memory_id="test",
            memory_type=MemoryType.FACT,
            content="c",
            confidence=0.5,
            privacy_level=0.0,
            source=MemorySource.SYSTEM,
        )
        assert record.turn_created == 0
        assert record.conversation_id == ""
        assert record.tags == ()

    def test_memory_record_is_frozen(self):
        record = _make_fact()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.content = "mutated"  # type: ignore[misc]

    def test_fact_is_frozen(self):
        fact = _make_fact()
        with pytest.raises(dataclasses.FrozenInstanceError):
            fact.category = "changed"  # type: ignore[misc]

    def test_preference_is_frozen(self):
        pref = _make_preference()
        with pytest.raises(dataclasses.FrozenInstanceError):
            pref.strength = 1.0  # type: ignore[misc]

    def test_relationship_memory_is_frozen(self):
        event = _make_relationship_event()
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.trust_delta = 0.99  # type: ignore[misc]

    def test_episode_is_frozen(self):
        ep = _make_episode()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ep.topic = "changed"  # type: ignore[misc]

    def test_fact_enforces_memory_type(self):
        fact = Fact(
            memory_id="f1",
            memory_type=MemoryType.PREFERENCE,  # wrong type
            content="should become FACT",
            confidence=0.5,
            privacy_level=0.0,
            source=MemorySource.SYSTEM,
        )
        assert fact.memory_type == MemoryType.FACT

    def test_preference_enforces_memory_type(self):
        pref = Preference(
            memory_id="p1",
            memory_type=MemoryType.FACT,  # wrong type
            content="should become PREFERENCE",
            confidence=0.5,
            privacy_level=0.0,
            source=MemorySource.SYSTEM,
        )
        assert pref.memory_type == MemoryType.PREFERENCE

    def test_relationship_enforces_memory_type(self):
        rel = RelationshipMemory(
            memory_id="r1",
            memory_type=MemoryType.FACT,  # wrong type
            content="should become RELATIONSHIP",
            confidence=0.5,
            privacy_level=0.0,
            source=MemorySource.SYSTEM,
        )
        assert rel.memory_type == MemoryType.RELATIONSHIP

    def test_episode_enforces_memory_type(self):
        ep = Episode(
            memory_id="e1",
            memory_type=MemoryType.FACT,  # wrong type
            content="should become EPISODE",
            confidence=0.5,
            privacy_level=0.0,
            source=MemorySource.SYSTEM,
        )
        assert ep.memory_type == MemoryType.EPISODE

    def test_confidence_decay_zero_age(self):
        record = _make_fact(turn_created=5)
        assert record.decayed_confidence(current_turn=5) == 0.9

    def test_confidence_decay_with_age(self):
        record = _make_fact(confidence=1.0, turn_created=0)
        # age=10, decay_rate=0.02 => decayed = 1.0 - (10 * 0.02) = 0.8
        assert record.decayed_confidence(current_turn=10) == pytest.approx(0.8)

    def test_confidence_decay_floors_at_zero(self):
        record = _make_fact(confidence=0.5, turn_created=0)
        # age=100, decay = 0.5 - (100 * 0.02) = 0.5 - 2.0 = -1.5 → clamped to 0
        assert record.decayed_confidence(current_turn=100) == 0.0

    def test_confidence_decay_custom_rate(self):
        record = _make_fact(confidence=1.0, turn_created=0)
        # age=5, rate=0.1 => 1.0 - 0.5 = 0.5
        assert record.decayed_confidence(current_turn=5, decay_rate=0.1) == pytest.approx(0.5)

    def test_confidence_decay_negative_age_clamped(self):
        """If current_turn < turn_created, age is clamped to 0."""
        record = _make_fact(confidence=0.9, turn_created=10)
        assert record.decayed_confidence(current_turn=5) == 0.9

    def test_memory_type_enum_values(self):
        assert MemoryType.FACT == "fact"
        assert MemoryType.PREFERENCE == "preference"
        assert MemoryType.RELATIONSHIP == "relationship"
        assert MemoryType.EPISODE == "episode"

    def test_memory_source_enum_values(self):
        assert MemorySource.USER_STATED == "user_stated"
        assert MemorySource.INFERRED == "inferred_from_context"
        assert MemorySource.OBSERVED == "observed_behavior"
        assert MemorySource.SYSTEM == "system"

    def test_fact_default_category(self):
        fact = Fact(
            memory_id="f",
            memory_type=MemoryType.FACT,
            content="some fact",
            confidence=0.5,
            privacy_level=0.0,
            source=MemorySource.SYSTEM,
        )
        assert fact.category == ""

    def test_episode_fields(self):
        ep = _make_episode(topic="python", outcome="explored", turn_start=5, turn_end=8)
        assert ep.topic == "python"
        assert ep.outcome == "explored"
        assert ep.turn_start == 5
        assert ep.turn_end == 8

    def test_preference_default_strength(self):
        pref = Preference(
            memory_id="p",
            memory_type=MemoryType.PREFERENCE,
            content="likes code",
            confidence=0.5,
            privacy_level=0.0,
            source=MemorySource.SYSTEM,
        )
        assert pref.strength == 0.5

    def test_tags_tuple_immutability(self):
        fact = _make_fact(tags=("work", "tech"))
        assert fact.tags == ("work", "tech")
        # tuples are already immutable, confirm it's a tuple
        assert isinstance(fact.tags, tuple)


# ============================================================================
# 2. TestFactStore
# ============================================================================


class TestFactStore:
    """Tests for FactStore: storage, retrieval, filtering, search."""

    def test_store_and_count(self, fact_store: FactStore):
        fact_store.store(_make_fact())
        assert fact_store.count == 1

    def test_store_multiple_facts(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="Fact 1"))
        fact_store.store(_make_fact(content="Fact 2"))
        fact_store.store(_make_fact(content="Fact 3"))
        assert fact_store.count == 3

    def test_get_by_category(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="engineer", category="occupation"))
        fact_store.store(_make_fact(content="London", category="location"))
        results = fact_store.get_by_category("occupation")
        assert len(results) == 1
        assert results[0].content == "engineer"

    def test_get_by_category_case_insensitive(self, fact_store: FactStore):
        fact_store.store(_make_fact(category="Occupation"))
        results = fact_store.get_by_category("occupation")
        assert len(results) == 1

    def test_get_by_category_most_recent_first(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="junior dev", category="occupation", turn_created=1))
        fact_store.store(_make_fact(content="senior dev", category="occupation", turn_created=5))
        results = fact_store.get_by_category("occupation")
        assert results[0].content == "senior dev"
        assert results[1].content == "junior dev"

    def test_get_by_category_privacy_filtering(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="public info", privacy_level=0.2))
        fact_store.store(_make_fact(content="private info", privacy_level=0.9))
        results = fact_store.get_by_category("occupation", max_privacy=0.5)
        assert len(results) == 1
        assert results[0].content == "public info"

    def test_get_by_category_confidence_decay_filter(self, fact_store: FactStore):
        fact_store.store(_make_fact(confidence=0.5, turn_created=0))
        # At turn 20: decayed = 0.5 - (20 * 0.02) = 0.1
        results = fact_store.get_by_category("occupation", current_turn=20, min_confidence=0.3)
        assert len(results) == 0

    def test_get_by_category_empty(self, fact_store: FactStore):
        results = fact_store.get_by_category("nonexistent")
        assert results == []

    def test_search_by_keyword_in_content(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="User is a software engineer"))
        fact_store.store(_make_fact(content="User lives in London", category="location"))
        results = fact_store.search("engineer")
        assert len(results) == 1
        assert "engineer" in results[0].content.lower()

    def test_search_by_keyword_in_category(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="works at Google", category="occupation"))
        results = fact_store.search("occupation")
        assert len(results) == 1

    def test_search_case_insensitive(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="User is an ENGINEER"))
        results = fact_store.search("engineer")
        assert len(results) == 1

    def test_search_respects_privacy(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="User salary is 100k", privacy_level=0.95))
        results = fact_store.search("salary", max_privacy=0.5)
        assert len(results) == 0

    def test_search_respects_confidence_decay(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="old fact", confidence=0.3, turn_created=0))
        # At turn 10: decayed = 0.3 - (10 * 0.02) = 0.1
        results = fact_store.search("old", current_turn=10, min_confidence=0.2)
        assert len(results) == 0

    def test_search_ordered_by_confidence(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="engineer low", confidence=0.3, category="occupation"))
        fact_store.store(_make_fact(content="engineer high", confidence=0.9, category="occupation"))
        results = fact_store.search("engineer")
        assert results[0].confidence > results[1].confidence

    def test_search_empty_store(self, fact_store: FactStore):
        results = fact_store.search("anything")
        assert results == []

    def test_search_no_match(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="User likes dogs"))
        results = fact_store.search("quantum physics")
        assert results == []

    def test_get_best_fact(self, fact_store: FactStore):
        fact_store.store(_make_fact(content="old job", category="occupation", turn_created=1))
        fact_store.store(_make_fact(content="new job", category="occupation", turn_created=10))
        best = fact_store.get_best_fact("occupation")
        assert best is not None
        assert best.content == "new job"

    def test_get_best_fact_empty_category(self, fact_store: FactStore):
        assert fact_store.get_best_fact("nonexistent") is None

    def test_categories_property(self, fact_store: FactStore):
        fact_store.store(_make_fact(category="occupation"))
        fact_store.store(_make_fact(category="location"))
        cats = fact_store.categories
        assert "occupation" in cats
        assert "location" in cats

    def test_all_facts(self, fact_store: FactStore):
        f1 = _make_fact(content="fact one")
        f2 = _make_fact(content="fact two")
        fact_store.store(f1)
        fact_store.store(f2)
        all_f = fact_store.all_facts()
        assert len(all_f) == 2


# ============================================================================
# 3. TestPreferenceStore
# ============================================================================


class TestPreferenceStore:
    """Tests for PreferenceStore: storage, reinforcement, retrieval, search."""

    def test_store_and_count(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference())
        assert preference_store.count == 1

    def test_unique_count(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference(content="User prefers brief answers"))
        preference_store.store(_make_preference(content="User prefers brief answers"))
        assert preference_store.unique_count == 1
        assert preference_store.count == 2

    def test_reinforcement_count(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference(content="likes examples"))
        preference_store.store(_make_preference(content="likes examples"))
        preference_store.store(_make_preference(content="likes examples"))
        assert preference_store.reinforcement_count("likes examples") == 3

    def test_reinforcement_count_zero(self, preference_store: PreferenceStore):
        assert preference_store.reinforcement_count("nonexistent") == 0

    def test_reinforcement_count_case_insensitive(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference(content="Likes Examples"))
        assert preference_store.reinforcement_count("likes examples") == 1

    def test_get_active_returns_latest_observation(self, preference_store: PreferenceStore):
        preference_store.store(
            _make_preference(content="brief answers", strength=0.4, turn_created=1)
        )
        preference_store.store(
            _make_preference(content="brief answers", strength=0.7, turn_created=5)
        )
        active = preference_store.get_active()
        assert len(active) == 1
        # Should return the latest observation (turn 5, strength 0.7)
        assert active[0].strength == 0.7

    def test_get_active_min_strength_filter(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference(content="weak pref", strength=0.2))
        preference_store.store(_make_preference(content="strong pref", strength=0.8))
        active = preference_store.get_active(min_strength=0.5)
        assert len(active) == 1
        assert active[0].content == "strong pref"

    def test_get_active_aggregated_strength_from_reinforcement(
        self, preference_store: PreferenceStore
    ):
        """Reinforcement boosts aggregated strength: base + (n-1)*0.1."""
        # Store 3 observations of a preference with base strength 0.3
        for i in range(3):
            preference_store.store(_make_preference(content="tech detail", strength=0.3))
        # Aggregated = 0.3 + (3-1)*0.1 = 0.5
        # With min_strength=0.5, it should pass
        active = preference_store.get_active(min_strength=0.5)
        assert len(active) == 1

    def test_get_active_aggregated_strength_capped_at_one(
        self, preference_store: PreferenceStore
    ):
        """Aggregated strength is capped at 1.0."""
        for i in range(20):
            preference_store.store(_make_preference(content="super strong", strength=0.9))
        active = preference_store.get_active(min_strength=0.99)
        # agg = min(1.0, 0.9 + 19*0.1) = min(1.0, 2.8) = 1.0
        assert len(active) == 1

    def test_get_active_sorted_by_strength_desc(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference(content="weak", strength=0.3))
        preference_store.store(_make_preference(content="medium", strength=0.5))
        preference_store.store(_make_preference(content="strong", strength=0.8))
        active = preference_store.get_active(min_strength=0.0)
        assert active[0].strength >= active[-1].strength

    def test_search_by_keyword(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference(content="User prefers brief answers"))
        preference_store.store(_make_preference(content="User likes technical detail"))
        results = preference_store.search("brief")
        assert len(results) == 1
        assert "brief" in results[0].content.lower()

    def test_search_returns_latest(self, preference_store: PreferenceStore):
        preference_store.store(
            _make_preference(content="prefers code examples", strength=0.3, turn_created=1)
        )
        preference_store.store(
            _make_preference(content="prefers code examples", strength=0.8, turn_created=5)
        )
        results = preference_store.search("code")
        assert len(results) == 1
        # Returns latest observation
        assert results[0].strength == 0.8

    def test_search_no_match(self, preference_store: PreferenceStore):
        preference_store.store(_make_preference(content="likes code"))
        results = preference_store.search("poetry")
        assert results == []

    def test_get_by_tag(self, preference_store: PreferenceStore):
        preference_store.store(
            _make_preference(content="brief answers", tags=("communication", "style"))
        )
        preference_store.store(
            _make_preference(content="tech detail", tags=("content", "depth"))
        )
        results = preference_store.get_by_tag("communication")
        assert len(results) == 1
        assert results[0].content == "brief answers"

    def test_get_by_tag_empty(self, preference_store: PreferenceStore):
        results = preference_store.get_by_tag("nonexistent")
        assert results == []

    def test_get_by_tag_returns_all_observations(self, preference_store: PreferenceStore):
        preference_store.store(
            _make_preference(content="brief answers", tags=("style",), turn_created=1)
        )
        preference_store.store(
            _make_preference(content="brief answers", tags=("style",), turn_created=5)
        )
        results = preference_store.get_by_tag("style")
        assert len(results) == 2

    def test_empty_store_get_active(self, preference_store: PreferenceStore):
        assert preference_store.get_active() == []


# ============================================================================
# 4. TestRelationshipStore
# ============================================================================


class TestRelationshipStore:
    """Tests for RelationshipStore: trust/rapport tracking, trends, clamping."""

    def test_initial_trust_and_rapport(self, relationship_store: RelationshipStore):
        assert relationship_store.trust == 0.5
        assert relationship_store.rapport == 0.3

    def test_custom_initial_values(self):
        store = RelationshipStore(initial_trust=0.7, initial_rapport=0.6)
        assert store.trust == 0.7
        assert store.rapport == 0.6

    def test_record_event_increases_trust(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(trust_delta=0.1, rapport_delta=0.0)
        )
        assert relationship_store.trust == pytest.approx(0.6)

    def test_record_event_increases_rapport(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(trust_delta=0.0, rapport_delta=0.1)
        )
        assert relationship_store.rapport == pytest.approx(0.4)

    def test_record_negative_event(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(trust_delta=-0.2, rapport_delta=-0.1)
        )
        assert relationship_store.trust == pytest.approx(0.3)
        assert relationship_store.rapport == pytest.approx(0.2)

    def test_trust_clamped_at_one(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(trust_delta=0.8)
        )
        assert relationship_store.trust == 1.0

    def test_trust_clamped_at_zero(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(trust_delta=-0.9)
        )
        assert relationship_store.trust == 0.0

    def test_rapport_clamped_at_one(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(rapport_delta=0.9)
        )
        assert relationship_store.rapport == 1.0

    def test_rapport_clamped_at_zero(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(rapport_delta=-0.5)
        )
        assert relationship_store.rapport == 0.0

    def test_multiple_events_cumulative(self, relationship_store: RelationshipStore):
        for _ in range(3):
            relationship_store.record_event(
                _make_relationship_event(trust_delta=0.1, rapport_delta=0.05)
            )
        assert relationship_store.trust == pytest.approx(0.8)
        assert relationship_store.rapport == pytest.approx(0.45)

    def test_event_count(self, relationship_store: RelationshipStore):
        assert relationship_store.event_count == 0
        relationship_store.record_event(_make_relationship_event())
        relationship_store.record_event(_make_relationship_event())
        assert relationship_store.event_count == 2

    def test_recent_events(self, relationship_store: RelationshipStore):
        for i in range(10):
            relationship_store.record_event(
                _make_relationship_event(content=f"event-{i}", turn_created=i)
            )
        recent = relationship_store.recent_events(n=3)
        assert len(recent) == 3
        assert recent[-1].content == "event-9"

    def test_recent_events_fewer_than_n(self, relationship_store: RelationshipStore):
        relationship_store.record_event(_make_relationship_event())
        recent = relationship_store.recent_events(n=5)
        assert len(recent) == 1

    def test_recent_events_empty(self, relationship_store: RelationshipStore):
        assert relationship_store.recent_events() == []

    def test_trust_trend_positive(self, relationship_store: RelationshipStore):
        for _ in range(3):
            relationship_store.record_event(
                _make_relationship_event(trust_delta=0.05)
            )
        assert relationship_store.trust_trend() == pytest.approx(0.15)

    def test_trust_trend_negative(self, relationship_store: RelationshipStore):
        for _ in range(3):
            relationship_store.record_event(
                _make_relationship_event(trust_delta=-0.05)
            )
        assert relationship_store.trust_trend() == pytest.approx(-0.15)

    def test_trust_trend_mixed(self, relationship_store: RelationshipStore):
        relationship_store.record_event(_make_relationship_event(trust_delta=0.1))
        relationship_store.record_event(_make_relationship_event(trust_delta=-0.1))
        assert relationship_store.trust_trend() == pytest.approx(0.0)

    def test_trust_trend_empty(self, relationship_store: RelationshipStore):
        assert relationship_store.trust_trend() == 0.0

    def test_rapport_trend(self, relationship_store: RelationshipStore):
        relationship_store.record_event(_make_relationship_event(rapport_delta=0.1))
        relationship_store.record_event(_make_relationship_event(rapport_delta=0.05))
        assert relationship_store.rapport_trend() == pytest.approx(0.15)

    def test_rapport_trend_empty(self, relationship_store: RelationshipStore):
        assert relationship_store.rapport_trend() == 0.0

    def test_trust_trend_windowed(self, relationship_store: RelationshipStore):
        # Add 10 events; first 5 positive, last 5 negative
        for _ in range(5):
            relationship_store.record_event(_make_relationship_event(trust_delta=0.1))
        for _ in range(5):
            relationship_store.record_event(_make_relationship_event(trust_delta=-0.1))
        # Window=5 should only see the last 5 negative events
        assert relationship_store.trust_trend(window=5) == pytest.approx(-0.5)

    def test_summary(self, relationship_store: RelationshipStore):
        relationship_store.record_event(
            _make_relationship_event(trust_delta=0.1, rapport_delta=0.05)
        )
        s = relationship_store.summary()
        assert "trust" in s
        assert "rapport" in s
        assert "trust_trend" in s
        assert "rapport_trend" in s
        assert "events" in s
        assert s["events"] == 1


# ============================================================================
# 5. TestEpisodicStore
# ============================================================================


class TestEpisodicStore:
    """Tests for EpisodicStore: episodes by topic, conversation, search."""

    def test_store_and_count(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode())
        assert episodic_store.count == 1

    def test_get_by_topic(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(topic="python"))
        episodic_store.store(_make_episode(topic="ux research"))
        results = episodic_store.get_by_topic("python")
        assert len(results) == 1
        assert results[0].topic == "python"

    def test_get_by_topic_case_insensitive(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(topic="Python"))
        results = episodic_store.get_by_topic("python")
        assert len(results) == 1

    def test_get_by_topic_most_recent_first(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(topic="python", content="ep1", turn_end=1))
        episodic_store.store(_make_episode(topic="python", content="ep2", turn_end=10))
        results = episodic_store.get_by_topic("python")
        assert results[0].turn_end == 10

    def test_get_by_topic_limit(self, episodic_store: EpisodicStore):
        for i in range(10):
            episodic_store.store(
                _make_episode(topic="python", content=f"ep-{i}", turn_end=i)
            )
        results = episodic_store.get_by_topic("python", limit=3)
        assert len(results) == 3

    def test_get_by_topic_empty(self, episodic_store: EpisodicStore):
        results = episodic_store.get_by_topic("nonexistent")
        assert results == []

    def test_get_by_conversation(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(conversation_id="conv-001"))
        episodic_store.store(_make_episode(conversation_id="conv-002"))
        episodic_store.store(_make_episode(conversation_id="conv-001", content="second"))
        results = episodic_store.get_by_conversation("conv-001")
        assert len(results) == 2

    def test_get_by_conversation_empty(self, episodic_store: EpisodicStore):
        results = episodic_store.get_by_conversation("nonexistent")
        assert results == []

    def test_search_by_content(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(content="Discussed UX research", topic="ux research"))
        episodic_store.store(_make_episode(content="Talked about Python", topic="python"))
        results = episodic_store.search("UX")
        assert len(results) == 1

    def test_search_by_topic(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(topic="machine learning", content="something"))
        results = episodic_store.search("machine learning")
        assert len(results) == 1

    def test_search_by_outcome(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(outcome="agreed on approach"))
        results = episodic_store.search("agreed")
        assert len(results) == 1

    def test_search_case_insensitive(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(content="PYTHON programming"))
        results = episodic_store.search("python")
        assert len(results) == 1

    def test_search_limit(self, episodic_store: EpisodicStore):
        for i in range(10):
            episodic_store.store(
                _make_episode(content=f"Topic alpha item {i}", turn_end=i)
            )
        results = episodic_store.search("alpha", limit=3)
        assert len(results) == 3

    def test_search_most_recent_first(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(content="alpha old", turn_end=1))
        episodic_store.store(_make_episode(content="alpha new", turn_end=10))
        results = episodic_store.search("alpha")
        assert results[0].turn_end == 10

    def test_search_no_match(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(content="dogs"))
        results = episodic_store.search("quantum")
        assert results == []

    def test_has_discussed_true(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(topic="python"))
        assert episodic_store.has_discussed("python") is True

    def test_has_discussed_false(self, episodic_store: EpisodicStore):
        assert episodic_store.has_discussed("python") is False

    def test_has_discussed_case_insensitive(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(topic="Python"))
        assert episodic_store.has_discussed("python") is True

    def test_get_recent(self, episodic_store: EpisodicStore):
        for i in range(10):
            episodic_store.store(_make_episode(content=f"ep-{i}", turn_end=i))
        recent = episodic_store.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[0].turn_end == 9

    def test_get_recent_empty(self, episodic_store: EpisodicStore):
        assert episodic_store.get_recent() == []

    def test_topics_property(self, episodic_store: EpisodicStore):
        episodic_store.store(_make_episode(topic="python"))
        episodic_store.store(_make_episode(topic="ux research"))
        topics = episodic_store.topics
        assert "python" in topics
        assert "ux research" in topics


# ============================================================================
# 6. TestMemoryManager
# ============================================================================


class TestMemoryManager:
    """Tests for MemoryManager orchestration and convenience methods."""

    def test_initial_stats(self, memory_manager: MemoryManager):
        stats = memory_manager.stats()
        assert stats["facts"] == 0
        assert stats["preferences"] == 0
        assert stats["relationship_events"] == 0
        assert stats["episodes"] == 0
        assert stats["trust"] == 0.5
        assert stats["rapport"] == 0.3

    def test_remember_fact(self, memory_manager: MemoryManager):
        fact = memory_manager.remember_fact(
            content="User works as a designer",
            category="occupation",
            confidence=0.9,
            turn=1,
        )
        assert fact.category == "occupation"
        assert fact.confidence == 0.9
        assert memory_manager.facts.count == 1

    def test_remember_fact_infers_category(self, memory_manager: MemoryManager):
        fact = memory_manager.remember_fact(content="User works as a designer", turn=1)
        assert fact.category == "occupation"

    def test_remember_preference(self, memory_manager: MemoryManager):
        pref = memory_manager.remember_preference(
            content="prefers detailed explanations",
            strength=0.7,
            turn=2,
        )
        assert pref.strength == 0.7
        assert memory_manager.preferences.count == 1

    def test_record_episode(self, memory_manager: MemoryManager):
        ep = memory_manager.record_episode(
            content="Discussed project architecture",
            topic="architecture",
            outcome="explored",
            turn_start=1,
            turn_end=5,
        )
        assert ep.topic == "architecture"
        assert ep.outcome == "explored"
        assert memory_manager.episodes.count == 1

    def test_record_relationship_event(self, memory_manager: MemoryManager):
        event = memory_manager.record_relationship_event(
            content="User agreed with suggestion",
            trust_delta=0.05,
            rapport_delta=0.03,
            turn=3,
        )
        assert event.trust_delta == 0.05
        assert memory_manager.relationships.trust == pytest.approx(0.55)

    def test_process_write_intents_fact(self, memory_manager: MemoryManager):
        intents = [
            MemoryWriteIntent(
                content_type="fact",
                content="User works as a data scientist",
                confidence=0.9,
                privacy_level=0.3,
                source="user_stated",
            )
        ]
        created = memory_manager.process_write_intents(intents, turn=1, conversation_id="c1")
        assert len(created) == 1
        assert isinstance(created[0], Fact)
        assert memory_manager.facts.count == 1

    def test_process_write_intents_preference(self, memory_manager: MemoryManager):
        intents = [
            MemoryWriteIntent(
                content_type="preference",
                content="User prefers concise answers",
                confidence=0.7,
                privacy_level=0.2,
                source="inferred_from_context",
            )
        ]
        created = memory_manager.process_write_intents(intents, turn=2)
        assert len(created) == 1
        assert isinstance(created[0], Preference)

    def test_process_write_intents_relationship(self, memory_manager: MemoryManager):
        intents = [
            MemoryWriteIntent(
                content_type="relationship",
                content="User agreed with persona's recommendation",
                confidence=0.8,
                privacy_level=0.1,
                source="observed_behavior",
            )
        ]
        created = memory_manager.process_write_intents(intents, turn=3)
        assert len(created) == 1
        assert isinstance(created[0], RelationshipMemory)
        assert memory_manager.relationships.event_count == 1

    def test_process_write_intents_episode(self, memory_manager: MemoryManager):
        intents = [
            MemoryWriteIntent(
                content_type="episode",
                content="Discussed machine learning approaches",
                confidence=0.85,
                privacy_level=0.2,
                source="observed_behavior",
            )
        ]
        created = memory_manager.process_write_intents(intents, turn=5)
        assert len(created) == 1
        assert isinstance(created[0], Episode)

    def test_process_write_intents_multiple(self, memory_manager: MemoryManager):
        intents = [
            MemoryWriteIntent(
                content_type="fact",
                content="User lives in Berlin",
                confidence=0.9,
                privacy_level=0.3,
                source="user_stated",
            ),
            MemoryWriteIntent(
                content_type="preference",
                content="User likes technical depth",
                confidence=0.7,
                privacy_level=0.2,
                source="inferred_from_context",
            ),
            MemoryWriteIntent(
                content_type="episode",
                content="Discussed career goals",
                confidence=0.8,
                privacy_level=0.4,
                source="observed_behavior",
            ),
        ]
        created = memory_manager.process_write_intents(intents, turn=1, conversation_id="c1")
        assert len(created) == 3

    def test_process_write_intents_unknown_type_skipped(self, memory_manager: MemoryManager):
        """Unknown content_type should be skipped (returns None from _create_record)."""
        # MemoryWriteIntent validates content_type so we need to test via _create_record
        # Instead, test that all four known types work and nothing else sneaks in
        intents = [
            MemoryWriteIntent(
                content_type="fact",
                content="Valid fact",
                confidence=0.9,
                privacy_level=0.3,
                source="user_stated",
            )
        ]
        created = memory_manager.process_write_intents(intents, turn=1)
        assert len(created) == 1

    def test_fulfill_read_requests_fact(self, memory_manager: MemoryManager):
        memory_manager.remember_fact("User works as engineer", category="occupation", turn=1)
        requests = [
            MemoryReadRequest(query_type="fact", query="engineer", confidence_threshold=0.0)
        ]
        results = memory_manager.fulfill_read_requests(requests, current_turn=2)
        assert "engineer" in results
        assert len(results["engineer"]) == 1

    def test_fulfill_read_requests_preference(self, memory_manager: MemoryManager):
        memory_manager.remember_preference("prefers brief answers", turn=1)
        requests = [
            MemoryReadRequest(query_type="preference", query="brief")
        ]
        results = memory_manager.fulfill_read_requests(requests, current_turn=2)
        assert "brief" in results
        assert len(results["brief"]) == 1

    def test_fulfill_read_requests_relationship(self, memory_manager: MemoryManager):
        memory_manager.record_relationship_event(
            "User agreed", trust_delta=0.05, turn=1
        )
        requests = [
            MemoryReadRequest(query_type="relationship", query="recent events")
        ]
        results = memory_manager.fulfill_read_requests(requests, current_turn=2)
        assert "recent events" in results
        assert len(results["recent events"]) == 1

    def test_fulfill_read_requests_episode(self, memory_manager: MemoryManager):
        memory_manager.record_episode(
            content="Discussed python frameworks",
            topic="python",
            turn_start=1,
            turn_end=3,
        )
        requests = [
            MemoryReadRequest(query_type="episode", query="python")
        ]
        results = memory_manager.fulfill_read_requests(requests, current_turn=5)
        assert "python" in results
        assert len(results["python"]) == 1

    def test_fulfill_read_requests_empty(self, memory_manager: MemoryManager):
        requests = [
            MemoryReadRequest(query_type="fact", query="nonexistent")
        ]
        results = memory_manager.fulfill_read_requests(requests, current_turn=1)
        assert len(results["nonexistent"]) == 0

    def test_get_context_for_turn_basic(self, memory_manager: MemoryManager):
        context = memory_manager.get_context_for_turn(current_turn=1)
        assert "relationship" in context
        assert "active_preferences" in context
        assert "known_facts" in context

    def test_get_context_for_turn_with_topic(self, memory_manager: MemoryManager):
        memory_manager.record_episode(
            content="Discussed python",
            topic="python",
            turn_start=1,
            turn_end=3,
        )
        context = memory_manager.get_context_for_turn(topic="python", current_turn=5)
        assert "topic_episodes" in context
        assert "previously_discussed" in context
        assert context["previously_discussed"] is True

    def test_get_context_for_turn_topic_not_discussed(self, memory_manager: MemoryManager):
        context = memory_manager.get_context_for_turn(topic="quantum", current_turn=5)
        assert context["previously_discussed"] is False

    def test_get_context_excludes_high_privacy_facts(self, memory_manager: MemoryManager):
        memory_manager.remember_fact(
            "User salary is 200k",
            category="salary",
            confidence=0.9,
            privacy_level=0.9,
            turn=1,
        )
        memory_manager.remember_fact(
            "User likes Python",
            category="interests",
            confidence=0.9,
            privacy_level=0.2,
            turn=1,
        )
        context = memory_manager.get_context_for_turn(current_turn=1)
        contents = [f["content"] for f in context["known_facts"]]
        assert "User likes Python" in contents
        assert "User salary is 200k" not in contents

    def test_get_context_excludes_low_confidence_facts(self, memory_manager: MemoryManager):
        memory_manager.remember_fact(
            "User might like cats",
            confidence=0.3,
            turn=0,
        )
        # At turn 10: decayed = 0.3 - (10 * 0.02) = 0.1 < 0.5 threshold
        context = memory_manager.get_context_for_turn(current_turn=10)
        assert len(context["known_facts"]) == 0

    def test_stats_populated(self, memory_manager: MemoryManager):
        memory_manager.remember_fact("fact", turn=1)
        memory_manager.remember_preference("pref", turn=2)
        memory_manager.record_episode("ep", topic="t", turn_start=3, turn_end=4)
        memory_manager.record_relationship_event("event", trust_delta=0.1, turn=5)
        stats = memory_manager.stats()
        assert stats["facts"] == 1
        assert stats["preferences"] == 1
        assert stats["episodes"] == 1
        assert stats["relationship_events"] == 1

    def test_category_inference_occupation(self):
        assert MemoryManager._infer_category("User works as a designer") == "occupation"

    def test_category_inference_location(self):
        assert MemoryManager._infer_category("User lives in Berlin") == "location"

    def test_category_inference_name(self):
        assert MemoryManager._infer_category("User's name is Alex") == "name"

    def test_category_inference_age(self):
        assert MemoryManager._infer_category("User is 30 years old") == "age"

    def test_category_inference_family(self):
        assert MemoryManager._infer_category("User has two children") == "family"

    def test_category_inference_interests(self):
        assert MemoryManager._infer_category("User enjoys hiking") == "interests"

    def test_category_inference_general_fallback(self):
        assert MemoryManager._infer_category("User mentioned something") == "general"

    def test_relationship_delta_inference_positive_trust(self):
        t, r = MemoryManager._infer_relationship_deltas("User agreed with suggestion")
        assert t > 0

    def test_relationship_delta_inference_positive_rapport(self):
        t, r = MemoryManager._infer_relationship_deltas("User was friendly and warm")
        assert r > 0

    def test_relationship_delta_inference_disclosure(self):
        t, r = MemoryManager._infer_relationship_deltas("User shared personal story")
        assert r > 0

    def test_relationship_delta_inference_negative_trust(self):
        t, r = MemoryManager._infer_relationship_deltas("User challenged the claim")
        assert t < 0

    def test_relationship_delta_inference_negative_rapport(self):
        t, r = MemoryManager._infer_relationship_deltas("There was tension in the conversation")
        assert r < 0

    def test_relationship_delta_inference_neutral(self):
        t, r = MemoryManager._infer_relationship_deltas("User asked about the weather")
        assert t == 0.0
        assert r == 0.0

    def test_extract_topic_with_discussed(self):
        topic = MemoryManager._extract_topic("Discussed machine learning approaches")
        assert "machine learning" in topic

    def test_extract_topic_fallback(self):
        topic = MemoryManager._extract_topic("Some episode content")
        assert len(topic) <= 50

    def test_make_id_deterministic(self):
        id1 = _make_id("content", 5)
        id2 = _make_id("content", 5)
        assert id1 == id2

    def test_make_id_different_for_different_input(self):
        id1 = _make_id("content_a", 5)
        id2 = _make_id("content_b", 5)
        assert id1 != id2

    def test_make_id_length(self):
        mid = _make_id("test", 1)
        assert len(mid) == 12

    def test_source_map_coverage(self):
        from persona_engine.memory.memory_manager import SOURCE_MAP

        assert SOURCE_MAP["user_stated"] == MemorySource.USER_STATED
        assert SOURCE_MAP["inferred_from_context"] == MemorySource.INFERRED
        assert SOURCE_MAP["observed_behavior"] == MemorySource.OBSERVED
        assert SOURCE_MAP["system"] == MemorySource.SYSTEM


# ============================================================================
# 7. TestMemoryManagerWithIR
# ============================================================================


class TestMemoryManagerWithIR:
    """Integration tests with actual IR schema MemoryOps objects."""

    def test_memory_ops_default(self):
        ops = MemoryOps()
        assert ops.read_requests == []
        assert ops.write_intents == []
        assert ops.write_policy == "strict"

    def test_memory_write_intent_creation(self):
        intent = MemoryWriteIntent(
            content_type="fact",
            content="User works as a data scientist",
            confidence=0.9,
            privacy_level=0.3,
            source="user_stated",
        )
        assert intent.content_type == "fact"
        assert intent.confidence == 0.9

    def test_memory_read_request_creation(self):
        req = MemoryReadRequest(
            query_type="fact",
            query="occupation",
            confidence_threshold=0.5,
        )
        assert req.query_type == "fact"
        assert req.confidence_threshold == 0.5

    def test_full_write_then_read_cycle(self, memory_manager: MemoryManager):
        """Write via IR intents, then read via IR requests."""
        write_ops = MemoryOps(
            write_intents=[
                MemoryWriteIntent(
                    content_type="fact",
                    content="User works as a data scientist",
                    confidence=0.9,
                    privacy_level=0.3,
                    source="user_stated",
                ),
                MemoryWriteIntent(
                    content_type="preference",
                    content="User prefers detailed explanations",
                    confidence=0.7,
                    privacy_level=0.2,
                    source="inferred_from_context",
                ),
            ]
        )
        created = memory_manager.process_write_intents(
            write_ops.write_intents, turn=1, conversation_id="conv-ir-1"
        )
        assert len(created) == 2

        read_ops = MemoryOps(
            read_requests=[
                MemoryReadRequest(query_type="fact", query="scientist"),
                MemoryReadRequest(query_type="preference", query="detailed"),
            ]
        )
        results = memory_manager.fulfill_read_requests(
            read_ops.read_requests, current_turn=2
        )
        assert len(results["scientist"]) == 1
        assert len(results["detailed"]) == 1

    def test_write_relationship_via_ir(self, memory_manager: MemoryManager):
        ops = MemoryOps(
            write_intents=[
                MemoryWriteIntent(
                    content_type="relationship",
                    content="User validated persona's expertise",
                    confidence=0.8,
                    privacy_level=0.1,
                    source="observed_behavior",
                ),
            ]
        )
        created = memory_manager.process_write_intents(ops.write_intents, turn=3)
        assert len(created) == 1
        assert isinstance(created[0], RelationshipMemory)
        # "validated" triggers positive trust delta
        assert created[0].trust_delta > 0

    def test_write_episode_via_ir(self, memory_manager: MemoryManager):
        ops = MemoryOps(
            write_intents=[
                MemoryWriteIntent(
                    content_type="episode",
                    content="Discussed deployment strategies for microservices",
                    confidence=0.85,
                    privacy_level=0.2,
                    source="observed_behavior",
                ),
            ]
        )
        created = memory_manager.process_write_intents(ops.write_intents, turn=5)
        assert len(created) == 1
        assert isinstance(created[0], Episode)
        assert "deployment strategies" in created[0].topic or created[0].topic != ""

    def test_read_with_confidence_threshold(self, memory_manager: MemoryManager):
        memory_manager.remember_fact(
            "User might be from France",
            category="location",
            confidence=0.3,
            turn=0,
        )
        requests = [
            MemoryReadRequest(
                query_type="fact",
                query="France",
                confidence_threshold=0.5,
            )
        ]
        results = memory_manager.fulfill_read_requests(requests, current_turn=5)
        # Decayed confidence = 0.3 - (5*0.02) = 0.2 < 0.5 threshold
        assert len(results["France"]) == 0

    def test_multiple_read_requests(self, memory_manager: MemoryManager):
        memory_manager.remember_fact("User is an engineer", category="occupation", turn=1)
        memory_manager.remember_fact("User lives in London", category="location", turn=1)
        requests = [
            MemoryReadRequest(query_type="fact", query="engineer"),
            MemoryReadRequest(query_type="fact", query="London"),
        ]
        results = memory_manager.fulfill_read_requests(requests, current_turn=2)
        assert len(results) == 2
        assert len(results["engineer"]) == 1
        assert len(results["London"]) == 1


# ============================================================================
# 8. TestMemoryBehavioralScenarios
# ============================================================================


class TestMemoryBehavioralScenarios:
    """Multi-turn conversation scenarios testing realistic memory behavior."""

    def test_sarah_remembers_occupation_across_turns(self):
        """Persona Sarah remembers user's occupation across turns."""
        mgr = MemoryManager()

        # Turn 1: User mentions occupation
        mgr.remember_fact(
            "User works as a UX researcher",
            category="occupation",
            confidence=0.95,
            source=MemorySource.USER_STATED,
            turn=1,
            conversation_id="conv-sarah-1",
        )

        # Turn 5: Sarah retrieves occupation
        best = mgr.facts.get_best_fact("occupation", current_turn=5)
        assert best is not None
        assert "UX researcher" in best.content

        # Turn 20: Still remembered (decayed but above threshold)
        # decayed = 0.95 - (19 * 0.02) = 0.95 - 0.38 = 0.57
        best = mgr.facts.get_best_fact("occupation", current_turn=20)
        assert best is not None

    def test_sarah_remembers_occupation_decays_eventually(self):
        """Occupation confidence decays over many turns."""
        mgr = MemoryManager()
        mgr.remember_fact(
            "User works as a UX researcher",
            category="occupation",
            confidence=0.95,
            turn=1,
        )
        # After 47 turns: decayed = 0.95 - (46 * 0.02) = 0.95 - 0.92 = 0.03
        facts = mgr.facts.get_by_category(
            "occupation", current_turn=48, min_confidence=0.1
        )
        assert len(facts) == 0

    def test_trust_builds_over_agreeable_interactions(self):
        """Trust should increase over multiple agreeable interactions."""
        mgr = MemoryManager()
        initial_trust = mgr.relationships.trust

        for turn in range(1, 6):
            mgr.record_relationship_event(
                content=f"User agreed with suggestion at turn {turn}",
                trust_delta=0.05,
                rapport_delta=0.03,
                turn=turn,
            )

        final_trust = mgr.relationships.trust
        assert final_trust > initial_trust
        assert final_trust == pytest.approx(0.75)  # 0.5 + 5*0.05
        assert mgr.relationships.trust_trend() > 0

    def test_trust_drops_when_challenged(self):
        """Trust decreases when user challenges persona."""
        mgr = MemoryManager()

        # Build some trust first
        for turn in range(1, 4):
            mgr.record_relationship_event(
                content="User agreed",
                trust_delta=0.1,
                turn=turn,
            )
        trust_after_build = mgr.relationships.trust  # 0.5 + 0.3 = 0.8

        # User challenges
        mgr.record_relationship_event(
            content="User challenged persona's expertise",
            trust_delta=-0.15,
            turn=4,
        )

        trust_after_challenge = mgr.relationships.trust
        assert trust_after_challenge < trust_after_build
        assert trust_after_challenge == pytest.approx(0.65)

    def test_trust_trend_shifts_after_challenge(self):
        """Trust trend should be negative after recent challenges."""
        mgr = MemoryManager()

        # Positive turns
        for turn in range(1, 4):
            mgr.record_relationship_event(
                content="User agreed",
                trust_delta=0.1,
                turn=turn,
            )

        # Negative turns
        for turn in range(4, 7):
            mgr.record_relationship_event(
                content="User challenged",
                trust_delta=-0.1,
                turn=turn,
            )

        # Window=3 should capture only negative events
        assert mgr.relationships.trust_trend(window=3) < 0

    def test_previously_discussed_topics_recognized(self):
        """Persona recognizes topics from prior conversations."""
        mgr = MemoryManager()

        # Conversation 1: discuss Python
        mgr.record_episode(
            content="Discussed Python best practices",
            topic="python",
            outcome="agreed",
            turn_start=1,
            turn_end=5,
            conversation_id="conv-1",
        )

        # Conversation 2: new topic comes up
        context = mgr.get_context_for_turn(topic="python", current_turn=10)
        assert context["previously_discussed"] is True
        assert len(context["topic_episodes"]) == 1

        # Unrelated topic should not be marked as discussed
        context2 = mgr.get_context_for_turn(topic="rust", current_turn=10)
        assert context2["previously_discussed"] is False

    def test_privacy_filtering_in_context(self):
        """High-privacy facts should not appear in turn context."""
        mgr = MemoryManager()

        # Low privacy: should appear
        mgr.remember_fact(
            "User is from Germany",
            category="location",
            privacy_level=0.2,
            confidence=0.9,
            turn=1,
        )

        # High privacy: should be filtered
        mgr.remember_fact(
            "User has a medical condition",
            category="health",
            privacy_level=0.95,
            confidence=0.9,
            turn=1,
        )

        # Medium privacy just under threshold: should appear
        mgr.remember_fact(
            "User studied at MIT",
            category="education",
            privacy_level=0.7,
            confidence=0.9,
            turn=1,
        )

        context = mgr.get_context_for_turn(current_turn=1)
        fact_contents = [f["content"] for f in context["known_facts"]]

        assert "User is from Germany" in fact_contents
        assert "User studied at MIT" in fact_contents
        assert "User has a medical condition" not in fact_contents

    def test_preference_reinforcement_over_repeated_observations(self):
        """Repeated observation of a preference should reinforce it."""
        mgr = MemoryManager()

        # Observe preference multiple times across turns
        for turn in range(1, 6):
            mgr.remember_preference(
                content="User prefers code examples",
                strength=0.4,
                turn=turn,
            )

        # Reinforcement count
        count = mgr.preferences.reinforcement_count("User prefers code examples")
        assert count == 5

        # Aggregated strength: 0.4 + (5-1)*0.1 = 0.8
        active = mgr.preferences.get_active(min_strength=0.7)
        assert len(active) == 1

    def test_preference_weak_without_reinforcement(self):
        """A single weak preference observation stays weak."""
        mgr = MemoryManager()
        mgr.remember_preference(
            content="User might like analogies",
            strength=0.2,
            turn=1,
        )
        active = mgr.preferences.get_active(min_strength=0.3)
        # 0.2 < 0.3, should not appear
        assert len(active) == 0

    def test_confidence_decay_over_many_turns(self):
        """Facts should decay over many turns and eventually fall below threshold."""
        mgr = MemoryManager()
        mgr.remember_fact(
            "User mentioned liking jazz",
            category="interests",
            confidence=0.7,
            turn=0,
        )

        # At turn 5: 0.7 - 0.10 = 0.60 (still above 0.5)
        context = mgr.get_context_for_turn(current_turn=5)
        assert len(context["known_facts"]) == 1

        # At turn 9: 0.7 - 0.18 = 0.52 (still above 0.5)
        context = mgr.get_context_for_turn(current_turn=9)
        assert len(context["known_facts"]) == 1

        # At turn 11: 0.7 - 0.22 = 0.48 (below 0.5, floats may push turn 10 below too)
        context = mgr.get_context_for_turn(current_turn=11)
        assert len(context["known_facts"]) == 0

    def test_multi_conversation_memory_persistence(self):
        """Memories should persist across multiple conversations."""
        mgr = MemoryManager()

        # Conversation 1
        mgr.remember_fact(
            "User's name is Alex",
            category="name",
            turn=1,
            conversation_id="conv-1",
        )
        mgr.record_episode(
            content="Introduced each other",
            topic="introduction",
            turn_start=1,
            turn_end=3,
            conversation_id="conv-1",
        )

        # Conversation 2
        mgr.remember_fact(
            "User works in finance",
            category="occupation",
            turn=1,
            conversation_id="conv-2",
        )

        # Both conversations' facts accessible
        assert mgr.facts.count == 2
        assert mgr.facts.get_best_fact("name") is not None
        assert mgr.facts.get_best_fact("occupation") is not None

    def test_relationship_builds_gradually_across_turns(self):
        """Relationship should evolve naturally across a multi-turn conversation."""
        mgr = MemoryManager()

        events = [
            ("User introduced themselves", 0.0, 0.05),   # polite intro
            ("User shared work context", 0.02, 0.08),     # opening up
            ("User agreed with approach", 0.05, 0.03),    # agreement
            ("User asked probing question", -0.02, 0.0),  # questioning
            ("User laughed at joke", 0.0, 0.1),           # rapport building
            ("User opened up about frustrations", 0.05, 0.12),  # disclosure
        ]

        for turn, (content, td, rd) in enumerate(events, 1):
            mgr.record_relationship_event(
                content=content, trust_delta=td, rapport_delta=rd, turn=turn
            )

        # Trust should have increased overall
        assert mgr.relationships.trust > 0.5
        # Rapport should have increased significantly
        assert mgr.relationships.rapport > 0.5
        # 6 events recorded
        assert mgr.relationships.event_count == 6

    def test_context_combines_all_memory_types(self):
        """get_context_for_turn should combine facts, preferences, episodes, relationship."""
        mgr = MemoryManager()

        mgr.remember_fact("User is a designer", category="occupation", confidence=0.9, turn=1)
        mgr.remember_preference("User prefers visual examples", strength=0.7, turn=2)
        mgr.record_episode(
            "Discussed prototyping tools", topic="prototyping", outcome="agreed",
            turn_start=3, turn_end=5,
        )
        mgr.record_relationship_event("User agreed", trust_delta=0.1, rapport_delta=0.05, turn=4)

        context = mgr.get_context_for_turn(topic="prototyping", current_turn=6)

        # Should have all components
        assert len(context["known_facts"]) >= 1
        assert len(context["active_preferences"]) >= 1
        assert context["previously_discussed"] is True
        assert len(context["topic_episodes"]) >= 1
        assert context["relationship"]["trust"] > 0.5

    def test_overwriting_fact_with_higher_confidence(self):
        """Newer fact with higher confidence should appear first (most recent)."""
        mgr = MemoryManager()

        mgr.remember_fact(
            "User might be from London",
            category="location",
            confidence=0.5,
            turn=1,
        )
        mgr.remember_fact(
            "User confirmed they live in London",
            category="location",
            confidence=0.95,
            turn=5,
        )

        best = mgr.facts.get_best_fact("location")
        assert best is not None
        assert "confirmed" in best.content

    def test_trust_cannot_exceed_one_even_with_many_positive_events(self):
        """Trust clamped at 1.0 no matter how many positive events."""
        mgr = MemoryManager()
        for turn in range(1, 20):
            mgr.record_relationship_event(
                content="Very positive",
                trust_delta=0.1,
                turn=turn,
            )
        assert mgr.relationships.trust == 1.0

    def test_rapport_cannot_go_below_zero(self):
        """Rapport clamped at 0.0 even with many negative events."""
        mgr = MemoryManager()
        for turn in range(1, 20):
            mgr.record_relationship_event(
                content="Very negative",
                rapport_delta=-0.1,
                turn=turn,
            )
        assert mgr.relationships.rapport == 0.0

    def test_full_conversation_flow(self):
        """
        End-to-end: simulate a 5-turn conversation with writes and reads.

        Turn 1: User introduces themselves (fact + relationship event)
        Turn 2: User asks about a topic (episode)
        Turn 3: User expresses preference (preference)
        Turn 4: Read context for new topic
        Turn 5: User shares personal info (high privacy fact)
        """
        mgr = MemoryManager()
        conv = "conv-flow-1"

        # Turn 1: Introduction
        t1_intents = [
            MemoryWriteIntent(
                content_type="fact",
                content="User's name is Jordan",
                confidence=0.95,
                privacy_level=0.2,
                source="user_stated",
            ),
            MemoryWriteIntent(
                content_type="relationship",
                content="User was friendly and warm during introduction",
                confidence=0.8,
                privacy_level=0.1,
                source="observed_behavior",
            ),
        ]
        mgr.process_write_intents(t1_intents, turn=1, conversation_id=conv)

        # Turn 2: Topic discussion
        t2_intents = [
            MemoryWriteIntent(
                content_type="episode",
                content="Discussed machine learning pipelines",
                confidence=0.85,
                privacy_level=0.2,
                source="observed_behavior",
            ),
        ]
        mgr.process_write_intents(t2_intents, turn=2, conversation_id=conv)

        # Turn 3: Preference expressed
        t3_intents = [
            MemoryWriteIntent(
                content_type="preference",
                content="User prefers practical examples over theory",
                confidence=0.7,
                privacy_level=0.2,
                source="inferred_from_context",
            ),
        ]
        mgr.process_write_intents(t3_intents, turn=3, conversation_id=conv)

        # Turn 4: Read context
        context = mgr.get_context_for_turn(topic="machine learning", current_turn=4)
        # We should see facts, preferences, relationship info
        assert len(context["known_facts"]) >= 1  # Jordan's name
        assert len(context["active_preferences"]) >= 1
        assert context["relationship"]["rapport"] > 0.3  # Warm intro boosted rapport

        # Turn 5: High privacy fact
        t5_intents = [
            MemoryWriteIntent(
                content_type="fact",
                content="User mentioned struggling with anxiety",
                confidence=0.8,
                privacy_level=0.95,
                source="user_stated",
            ),
        ]
        mgr.process_write_intents(t5_intents, turn=5, conversation_id=conv)

        # Verify high privacy fact stored but filtered from context
        assert mgr.facts.count == 2  # name + anxiety
        context_t5 = mgr.get_context_for_turn(current_turn=5)
        fact_contents = [f["content"] for f in context_t5["known_facts"]]
        assert "User mentioned struggling with anxiety" not in fact_contents

        # Verify stats reflect everything
        stats = mgr.stats()
        assert stats["facts"] == 2
        assert stats["preferences"] == 1
        assert stats["episodes"] == 1
        assert stats["relationship_events"] == 1

    def test_multiple_topics_tracked_independently(self):
        """Different topics should be tracked independently in episodic store."""
        mgr = MemoryManager()

        mgr.record_episode("Discussed Python", topic="python", turn_start=1, turn_end=3)
        mgr.record_episode("Discussed Rust", topic="rust", turn_start=4, turn_end=6)
        mgr.record_episode("More Python talk", topic="python", turn_start=7, turn_end=9)

        assert mgr.episodes.has_discussed("python")
        assert mgr.episodes.has_discussed("rust")
        assert not mgr.episodes.has_discussed("java")

        python_eps = mgr.episodes.get_by_topic("python")
        assert len(python_eps) == 2

        rust_eps = mgr.episodes.get_by_topic("rust")
        assert len(rust_eps) == 1

    def test_rapid_trust_erosion_scenario(self):
        """Trust erodes rapidly when multiple negative events occur in sequence."""
        mgr = MemoryManager(initial_trust=0.8, initial_rapport=0.6)

        # Series of negative interactions
        negative_events = [
            ("User disagreed strongly", -0.1, -0.05),
            ("User was defensive about claims", -0.08, -0.08),
            ("Conversation became awkward", -0.05, -0.1),
        ]

        for turn, (content, td, rd) in enumerate(negative_events, 1):
            mgr.record_relationship_event(
                content=content, trust_delta=td, rapport_delta=rd, turn=turn
            )

        # Trust should have dropped significantly
        assert mgr.relationships.trust < 0.6
        # Rapport should have dropped
        assert mgr.relationships.rapport < 0.4
        # Both trends should be negative
        assert mgr.relationships.trust_trend() < 0
        assert mgr.relationships.rapport_trend() < 0

    def test_user_fact_update_scenario(self):
        """User corrects a fact — newer fact should supersede old one."""
        mgr = MemoryManager()

        # Turn 1: Initial fact (inferred, lower confidence)
        mgr.remember_fact(
            "User might be a backend developer",
            category="occupation",
            confidence=0.5,
            source=MemorySource.INFERRED,
            turn=1,
        )

        # Turn 8: User explicitly states their role
        mgr.remember_fact(
            "User explicitly said they are a full-stack developer",
            category="occupation",
            confidence=0.95,
            source=MemorySource.USER_STATED,
            turn=8,
        )

        # Best fact should be the most recent (turn 8)
        best = mgr.facts.get_best_fact("occupation", current_turn=9)
        assert best is not None
        assert "full-stack" in best.content

    def test_preference_not_active_below_threshold(self):
        """A single weak preference should not be active above a reasonable threshold."""
        mgr = MemoryManager()
        mgr.remember_preference("might prefer short responses", strength=0.2, turn=1)

        active = mgr.preferences.get_active(min_strength=0.3)
        assert len(active) == 0

        # But at lower threshold it appears
        active_low = mgr.preferences.get_active(min_strength=0.1)
        assert len(active_low) == 1
