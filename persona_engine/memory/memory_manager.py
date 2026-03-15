"""
Memory Manager — orchestrates all memory stores.

Central point for reading and writing memories. Integrates with
TurnPlanner via MemoryOps from the IR schema.

Key responsibilities:
- Execute MemoryWriteIntents from IR
- Fulfill MemoryReadRequests for IR generation
- Provide conversation context to TurnPlanner
- Prevent persona drift (no verbatim replay, confidence decay)
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)
from datetime import datetime
from typing import Any, Sequence

from persona_engine.memory.episodic_store import EpisodicStore
from persona_engine.memory.fact_store import FactStore
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
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.schema.ir_schema import (
    MemoryOps,
    MemoryReadRequest,
    MemoryWriteIntent,
)


def _make_id(content: str, turn: int) -> str:
    """Generate a short deterministic memory ID."""
    raw = f"{content}:{turn}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


SOURCE_MAP: dict[str, MemorySource] = {
    "user_stated": MemorySource.USER_STATED,
    "inferred_from_context": MemorySource.INFERRED,
    "observed_behavior": MemorySource.OBSERVED,
    "system": MemorySource.SYSTEM,
}


class MemoryManager:
    """
    Orchestrates all four memory stores + stance cache.

    Usage:
        manager = MemoryManager()

        # After TurnPlanner generates IR:
        manager.process_write_intents(ir.memory_ops.write_intents, turn=3, conv_id="abc")

        # Before TurnPlanner generates IR (provide context):
        context = manager.get_context_for_turn(topic="ux_research", current_turn=5)

        # Fulfill read requests:
        results = manager.fulfill_read_requests(ir.memory_ops.read_requests, current_turn=5)
    """

    def __init__(
        self,
        initial_trust: float = 0.5,
        initial_rapport: float = 0.3,
        stance_cache: StanceCache | None = None,
    ) -> None:
        self.facts = FactStore()
        self.preferences = PreferenceStore()
        self.relationships = RelationshipStore(
            initial_trust=initial_trust,
            initial_rapport=initial_rapport,
        )
        self.episodes = EpisodicStore()
        # Accept injected StanceCache to avoid duplicate instances.
        # The engine owns the canonical cache and passes it here.
        self.stance_cache = stance_cache or StanceCache()

    # ========================================================================
    # Write: process MemoryWriteIntents from IR
    # ========================================================================

    # Minimum confidence for a write intent to be accepted in strict mode.
    STRICT_CONFIDENCE_THRESHOLD: float = 0.7

    def process_write_intents(
        self,
        intents: list[MemoryWriteIntent],
        turn: int,
        conversation_id: str = "",
        write_policy: str = "strict",
    ) -> list[MemoryRecord]:
        """
        Execute memory write intents from IR.

        When *write_policy* is ``"strict"`` (the default), intents whose
        confidence falls below ``STRICT_CONFIDENCE_THRESHOLD`` are silently
        skipped.  ``"lenient"`` writes all intents regardless of confidence.

        Args:
            intents: Write intents from ir.memory_ops.write_intents
            turn: Current turn number
            conversation_id: Current conversation ID
            write_policy: ``"strict"`` or ``"lenient"`` (from ``MemoryOps.write_policy``)

        Returns:
            List of created memory records
        """
        logger.debug(
            "Processing write intents",
            extra={"intent_count": len(intents), "turn": turn, "write_policy": write_policy},
        )
        created: list[MemoryRecord] = []
        for intent in intents:
            if write_policy == "strict" and intent.confidence < self.STRICT_CONFIDENCE_THRESHOLD:
                logger.debug(
                    "Skipping low-confidence intent (strict mode)",
                    extra={"content": intent.content[:50], "confidence": intent.confidence},
                )
                continue
            record = self._create_record(intent, turn, conversation_id)
            if record:
                self._store_record(record)
                created.append(record)
        return created

    def _create_record(
        self,
        intent: MemoryWriteIntent,
        turn: int,
        conversation_id: str,
    ) -> MemoryRecord | None:
        """Convert a MemoryWriteIntent to a typed MemoryRecord."""
        memory_id = _make_id(intent.content, turn)
        source = SOURCE_MAP.get(intent.source, MemorySource.INFERRED)

        if intent.content_type == "fact":
            return Fact(
                memory_id=memory_id,
                memory_type=MemoryType.FACT,
                content=intent.content,
                confidence=intent.confidence,
                privacy_level=intent.privacy_level,
                source=source,
                turn_created=turn,
                conversation_id=conversation_id,
                category=self._infer_category(intent.content),
            )
        elif intent.content_type == "preference":
            return Preference(
                memory_id=memory_id,
                memory_type=MemoryType.PREFERENCE,
                content=intent.content,
                confidence=intent.confidence,
                privacy_level=intent.privacy_level,
                source=source,
                turn_created=turn,
                conversation_id=conversation_id,
                strength=intent.confidence,  # Initial strength = confidence
            )
        elif intent.content_type == "relationship":
            # Infer trust/rapport deltas from content
            trust_d, rapport_d = self._infer_relationship_deltas(intent.content)
            return RelationshipMemory(
                memory_id=memory_id,
                memory_type=MemoryType.RELATIONSHIP,
                content=intent.content,
                confidence=intent.confidence,
                privacy_level=intent.privacy_level,
                source=source,
                turn_created=turn,
                conversation_id=conversation_id,
                trust_delta=trust_d,
                rapport_delta=rapport_d,
            )
        elif intent.content_type == "episode":
            return Episode(
                memory_id=memory_id,
                memory_type=MemoryType.EPISODE,
                content=intent.content,
                confidence=intent.confidence,
                privacy_level=intent.privacy_level,
                source=source,
                turn_created=turn,
                conversation_id=conversation_id,
                topic=self._extract_topic(intent.content),
                turn_start=turn,
                turn_end=turn,
            )
        return None

    def _store_record(self, record: MemoryRecord) -> None:
        """Route a record to the appropriate store."""
        if isinstance(record, Fact):
            self.facts.store(record)
        elif isinstance(record, Preference):
            self.preferences.store(record)
        elif isinstance(record, RelationshipMemory):
            self.relationships.record_event(record)
        elif isinstance(record, Episode):
            self.episodes.store(record)

    # ========================================================================
    # Read: fulfill MemoryReadRequests
    # ========================================================================

    def fulfill_read_requests(
        self,
        requests: list[MemoryReadRequest],
        current_turn: int = 0,
    ) -> dict[str, Sequence[MemoryRecord]]:
        """
        Fulfill memory read requests from IR.

        Args:
            requests: Read requests from ir.memory_ops.read_requests
            current_turn: Current turn for confidence decay

        Returns:
            Dict mapping query → matching records
        """
        results: dict[str, Sequence[MemoryRecord]] = {}
        for req in requests:
            records = self._execute_read(req, current_turn)
            results[req.query] = records
        return results

    def _execute_read(
        self,
        request: MemoryReadRequest,
        current_turn: int,
    ) -> Sequence[MemoryRecord]:
        """Execute a single read request."""
        min_conf = request.confidence_threshold

        if request.query_type == "fact":
            return self.facts.search(
                request.query,
                current_turn=current_turn,
                min_confidence=min_conf,
            )
        elif request.query_type == "preference":
            return self.preferences.search(request.query)
        elif request.query_type == "relationship":
            return self.relationships.recent_events()
        elif request.query_type == "episode":
            return self.episodes.search(request.query)
        return []

    # ========================================================================
    # Context: provide conversation context for TurnPlanner
    # ========================================================================

    def get_context_for_turn(
        self,
        topic: str = "",
        current_turn: int = 0,
    ) -> dict[str, Any]:
        """
        Gather memory context relevant to the current turn.

        Used by TurnPlanner to inform IR generation with memory.

        Args:
            topic: Current topic being discussed
            current_turn: Current turn number

        Returns:
            Dict with relevant memories for the turn
        """
        context: dict[str, Any] = {
            "relationship": self.relationships.summary(),
            "active_preferences": [
                {"content": p.content, "strength": p.strength}
                for p in self.preferences.get_active(current_turn)
            ],
        }

        if topic:
            context["topic_episodes"] = [
                {"content": e.content, "outcome": e.outcome}
                for e in self.episodes.get_by_topic(topic, limit=3)
            ]
            context["previously_discussed"] = self.episodes.has_discussed(topic)

        # Include high-confidence facts (respect privacy)
        context["known_facts"] = [
            {"content": f.content, "category": f.category, "confidence": f.confidence}
            for f in self.facts.all_facts()
            if f.decayed_confidence(current_turn) >= 0.5 and f.privacy_level < 0.8
        ]

        logger.debug(
            "Context assembled for turn",
            extra={
                "fact_count": len(context["known_facts"]),
                "preference_count": len(context["active_preferences"]),
                "previously_discussed": context.get("previously_discussed", False),
            },
        )

        return context

    # ========================================================================
    # Convenience: direct store methods
    # ========================================================================

    def remember_fact(
        self,
        content: str,
        category: str = "",
        confidence: float = 0.8,
        privacy_level: float = 0.3,
        source: MemorySource = MemorySource.USER_STATED,
        turn: int = 0,
        conversation_id: str = "",
    ) -> Fact:
        """Convenience: store a fact directly."""
        fact = Fact(
            memory_id=_make_id(content, turn),
            memory_type=MemoryType.FACT,
            content=content,
            confidence=confidence,
            privacy_level=privacy_level,
            source=source,
            turn_created=turn,
            conversation_id=conversation_id,
            category=category or self._infer_category(content),
        )
        self.facts.store(fact)
        return fact

    def remember_preference(
        self,
        content: str,
        strength: float = 0.5,
        confidence: float = 0.7,
        source: MemorySource = MemorySource.INFERRED,
        turn: int = 0,
        conversation_id: str = "",
    ) -> Preference:
        """Convenience: store a preference directly."""
        pref = Preference(
            memory_id=_make_id(content, turn),
            memory_type=MemoryType.PREFERENCE,
            content=content,
            confidence=confidence,
            privacy_level=0.2,
            source=source,
            turn_created=turn,
            conversation_id=conversation_id,
            strength=strength,
        )
        self.preferences.store(pref)
        return pref

    def record_episode(
        self,
        content: str,
        topic: str,
        outcome: str = "",
        turn_start: int = 0,
        turn_end: int = 0,
        conversation_id: str = "",
    ) -> Episode:
        """Convenience: store an episode directly."""
        episode = Episode(
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
        self.episodes.store(episode)
        return episode

    def record_relationship_event(
        self,
        content: str,
        trust_delta: float = 0.0,
        rapport_delta: float = 0.0,
        turn: int = 0,
        conversation_id: str = "",
    ) -> RelationshipMemory:
        """Convenience: record a relationship event directly."""
        event = RelationshipMemory(
            memory_id=_make_id(content, turn),
            memory_type=MemoryType.RELATIONSHIP,
            content=content,
            confidence=0.8,
            privacy_level=0.1,
            source=MemorySource.OBSERVED,
            turn_created=turn,
            conversation_id=conversation_id,
            trust_delta=trust_delta,
            rapport_delta=rapport_delta,
        )
        self.relationships.record_event(event)
        return event

    # ========================================================================
    # Stats
    # ========================================================================

    def stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            "facts": self.facts.count,
            "fact_categories": self.facts.categories,
            "preferences": self.preferences.unique_count,
            "preference_observations": self.preferences.count,
            "relationship_events": self.relationships.event_count,
            "trust": self.relationships.trust,
            "rapport": self.relationships.rapport,
            "episodes": self.episodes.count,
            "episode_topics": self.episodes.topics,
            "cached_stances": len(self.stance_cache.cache),
        }

    # ========================================================================
    # Internal helpers
    # ========================================================================

    @staticmethod
    def _infer_category(content: str) -> str:
        """Simple keyword-based category inference."""
        lower = content.lower()
        if any(w in lower for w in ["works as", "job", "occupation", "profession", "career"]):
            return "occupation"
        if any(w in lower for w in ["lives in", "from", "located", "city", "country"]):
            return "location"
        if any(w in lower for w in ["name is", "called", "named"]):
            return "name"
        if any(w in lower for w in ["age", "years old", "born"]):
            return "age"
        if any(w in lower for w in ["family", "children", "married", "partner", "spouse"]):
            return "family"
        if any(w in lower for w in ["hobby", "hobbies", "enjoys", "likes to", "interested in"]):
            return "interests"
        return "general"

    @staticmethod
    def _extract_topic(content: str) -> str:
        """Extract topic from episode content (simple heuristic)."""
        lower = content.lower()
        if "discussed" in lower:
            parts = lower.split("discussed")
            if len(parts) > 1:
                # Take first segment, split on common separators
                raw = parts[1].strip()
                for sep in (":", ".", ",", " — ", " - "):
                    raw = raw.split(sep)[0]
                return raw.strip()
        return lower[:50]

    @staticmethod
    def _infer_relationship_deltas(content: str) -> tuple[float, float]:
        """Infer trust/rapport deltas from content description."""
        lower = content.lower()
        trust_d = 0.0
        rapport_d = 0.0

        # Positive trust signals
        if any(w in lower for w in [
            "agreed", "trust", "validated", "confirmed",
            "expertise", "knowledgeable", "helpful", "accurate",
            "substantive", "insightful",
        ]):
            trust_d += 0.05

        # Positive rapport signals
        if any(w in lower for w in [
            "rapport", "friendly", "connected", "warm", "laughed",
            "exchange", "explored", "enjoyed", "casual",
        ]):
            rapport_d += 0.05
        if any(w in lower for w in ["shared personal", "opened up", "disclosed"]):
            rapport_d += 0.08

        # Baseline engagement signal — any engagement is mild rapport
        if "engaged" in lower:
            rapport_d += 0.02

        # Negative signals
        if any(w in lower for w in ["challenged", "disagreed", "questioned expertise"]):
            trust_d -= 0.05
        if any(w in lower for w in ["tension", "awkward", "defensive", "hostile"]):
            rapport_d -= 0.05

        return trust_d, rapport_d
