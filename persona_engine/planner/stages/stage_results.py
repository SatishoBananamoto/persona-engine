"""
Typed dataclasses for stage return values.

Replaces ``dict[str, Any]`` returns with structured, self-documenting types
so that downstream stages and the orchestrator get static type checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from persona_engine.schema.ir_schema import (
    KnowledgeClaimType,
    Tone,
    UncertaintyAction,
    Verbosity,
)


@dataclass
class InterpretationResult:
    """Output of Stage 2 (Interpretation)."""
    topic_relevance: float
    persona_domains: list[dict]
    domain: str
    proficiency: float
    expert_allowed: bool
    user_intent: str
    needs_clarification: bool
    policy_modifications: dict[str, Any]
    # Injected by orchestrator after foundation stage
    memory_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehavioralMetricsResult:
    """Output of Stage 3 (Behavioral Metrics)."""
    elasticity: float
    stance: str
    rationale: str
    confidence: float
    competence: float
    tone: Tone
    verbosity: Verbosity
    formality: float
    directness: float
    trait_guidance: Any = None   # TraitGuidance (avoids circular import)
    cognitive_guidance: Any = None  # CognitiveGuidance
    adaptation: Any = None       # AdaptationDirectives | None
    schema_effect: Any = None    # SchemaEffect | None


@dataclass
class KnowledgeSafetyResult:
    """Output of Stage 4 (Knowledge & Safety)."""
    disclosure_level: float
    uncertainty_action: UncertaintyAction
    claim_enum: KnowledgeClaimType
