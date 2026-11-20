"""
Memory data models.

All four memory types use typed, immutable records with confidence,
privacy, recency, and provenance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    """The four typed memory categories."""

    FACT = "fact"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    EPISODE = "episode"


class MemorySource(StrEnum):
    """How the memory was acquired."""

    USER_STATED = "user_stated"          # User explicitly said it
    INFERRED = "inferred_from_context"   # Persona inferred from conversation
    OBSERVED = "observed_behavior"       # Observed from user behavior patterns
    SYSTEM = "system"                    # Injected by system/setup


@dataclass(frozen=True)
class MemoryRecord:
    """
    Base record for all memory types.

    Immutable by design — updates create new records rather than
    mutating existing ones. This prevents drift and provides audit trail.
    """

    memory_id: str
    memory_type: MemoryType
    content: str
    confidence: float          # 0.0 = uncertain guess, 1.0 = explicitly stated
    privacy_level: float       # 0.0 = public, 1.0 = highly private
    source: MemorySource
    created_at: datetime = field(default_factory=datetime.now)
    turn_created: int = 0
    conversation_id: str = ""
    tags: tuple[str, ...] = ()  # Frozen tuple for hashability

    def decayed_confidence(self, current_turn: int, decay_rate: float = 0.02) -> float:
        """Confidence decays with turn distance. Slower than stance (0.02 vs 0.1)."""
        age = max(0, current_turn - self.turn_created)
        return max(0.0, self.confidence - (age * decay_rate))


# ============================================================================
# Fact: concrete user information
# ============================================================================


@dataclass(frozen=True)
class Fact(MemoryRecord):
    """
    Concrete factual information about the user.

    Examples: "User works as a software engineer", "User lives in London",
    "User has two children", "User's name is Alex".
    """

    category: str = ""  # e.g. "occupation", "location", "family", "name"

    def __post_init__(self) -> None:
        # Enforce type
        if self.memory_type != MemoryType.FACT:
            object.__setattr__(self, "memory_type", MemoryType.FACT)


# ============================================================================
# Preference: learned behavioral patterns
# ============================================================================


@dataclass(frozen=True)
class Preference(MemoryRecord):
    """
    Learned user preference or behavioral pattern.

    Examples: "User prefers brief answers", "User likes technical detail",
    "User dislikes small talk", "User responds well to examples".
    """

    strength: float = 0.5  # How strong the preference is (0=weak, 1=strong)

    def __post_init__(self) -> None:
        if self.memory_type != MemoryType.PREFERENCE:
            object.__setattr__(self, "memory_type", MemoryType.PREFERENCE)


# ============================================================================
# Relationship: trust and rapport state
# ============================================================================


@dataclass(frozen=True)
class RelationshipMemory(MemoryRecord):
    """
    Tracks relationship dynamics with the user.

    Examples: "User trusts persona on UX topics", "Rapport is building",
    "User challenged persona's expertise — trust dipped".
    """

    trust_delta: float = 0.0     # How this event changed trust (-1 to +1)
    rapport_delta: float = 0.0   # How this event changed rapport (-1 to +1)

    def __post_init__(self) -> None:
        if self.memory_type != MemoryType.RELATIONSHIP:
            object.__setattr__(self, "memory_type", MemoryType.RELATIONSHIP)


# ============================================================================
# Episode: compressed conversation summary
# ============================================================================


@dataclass(frozen=True)
class Episode(MemoryRecord):
    """
    Compressed summary of a conversation segment.

    NOT verbatim transcripts — compressed semantic summaries to prevent
    persona drift from exact memory replay.
    """

    topic: str = ""
    turn_start: int = 0
    turn_end: int = 0
    outcome: str = ""  # e.g. "agreed", "disagreed", "explored", "deferred"

    def __post_init__(self) -> None:
        if self.memory_type != MemoryType.EPISODE:
            object.__setattr__(self, "memory_type", MemoryType.EPISODE)
