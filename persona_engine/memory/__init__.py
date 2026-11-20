"""Memory module for persona state persistence.

Four typed memory stores:
- FactStore: concrete user information (name, occupation, location)
- PreferenceStore: learned behavioral patterns (prefers brief answers)
- RelationshipStore: trust and rapport dynamics
- EpisodicStore: compressed conversation summaries

Plus StanceCache for within-conversation stance consistency.
MemoryManager orchestrates all stores.
"""

from persona_engine.memory.episodic_store import EpisodicStore
from persona_engine.memory.fact_store import FactStore
from persona_engine.memory.memory_manager import MemoryManager
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
from persona_engine.memory.stance_cache import CachedStance, StanceCache

__all__ = [
    # Manager
    "MemoryManager",
    # Stores
    "FactStore",
    "PreferenceStore",
    "RelationshipStore",
    "EpisodicStore",
    "StanceCache",
    # Models
    "MemoryRecord",
    "Fact",
    "Preference",
    "RelationshipMemory",
    "Episode",
    "CachedStance",
    "MemoryType",
    "MemorySource",
]
