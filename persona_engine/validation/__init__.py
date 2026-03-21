"""
Validation Module — end-to-end coherence checks for the persona engine.

Five validation layers:
1. IR Coherence: Internal field consistency
2. Persona Compliance: IR vs persona profile
3. Cross-Turn Consistency: Multi-turn behavior coherence
4. IR Validator: Pre-generation consistency and constraint checks
5. Style Drift Detection: Multi-turn behavioral drift tracking
6. Knowledge Boundary Enforcer: Expertise limit enforcement
"""

from persona_engine.validation.cross_turn import CrossTurnTracker, TurnSnapshot
from persona_engine.validation.ir_coherence import validate_ir_coherence
from persona_engine.validation.ir_validator import (
    Severity,
    ValidationIssue,
    has_errors,
    validate_ir,
)
from persona_engine.validation.knowledge_boundary import (
    BoundaryReport,
    BoundaryViolation,
    KnowledgeBoundaryEnforcer,
)
from persona_engine.validation.persona_compliance import validate_persona_compliance
from persona_engine.validation.pipeline_validator import PipelineValidator
from persona_engine.validation.style_drift import (
    DriftReport,
    StyleDriftDetector,
    TurnMetrics,
)
from persona_engine.validation.trait_scorer import (
    TraitMarkerScorer,
    TraitScorerResult,
    TraitScore,
)

__all__ = [
    "PipelineValidator",
    "CrossTurnTracker",
    "TurnSnapshot",
    "validate_ir_coherence",
    "validate_persona_compliance",
    # IR Validator
    "validate_ir",
    "has_errors",
    "Severity",
    "ValidationIssue",
    # Style Drift
    "StyleDriftDetector",
    "DriftReport",
    "TurnMetrics",
    # Knowledge Boundary
    "KnowledgeBoundaryEnforcer",
    "BoundaryReport",
    "BoundaryViolation",
    # Trait Scoring
    "TraitMarkerScorer",
    "TraitScorerResult",
    "TraitScore",
]
