"""
Intermediate Representation (IR) Schema - Pydantic v2
Persona Engine MVP

This schema defines the structured output from the Turn Planner that makes
persona behavior testable, debuggable, and deterministic.

Production-hardened with:
- Delta-based citations (numeric and non-numeric)
- Memory operations channel
- Safety plan for constraint transparency
- Deterministic normalization utilities
"""

import json
from enum import StrEnum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Enums for Controlled Vocabularies
# ============================================================================

class InteractionMode(StrEnum):
    """Type of conversation context"""
    CASUAL_CHAT = "casual_chat"
    INTERVIEW = "interview"
    CUSTOMER_SUPPORT = "customer_support"
    SURVEY = "survey"
    COACHING = "coaching"
    DEBATE = "debate"
    SMALL_TALK = "small_talk"
    BRAINSTORM = "brainstorm"


class ConversationGoal(StrEnum):
    """What the conversation is trying to achieve"""
    GATHER_INFO = "gather_info"
    RESOLVE_ISSUE = "resolve_issue"
    BUILD_RAPPORT = "build_rapport"
    PERSUADE = "persuade"
    EDUCATE = "educate"
    ENTERTAIN = "entertain"
    EXPLORE_IDEAS = "explore_ideas"


class Verbosity(StrEnum):
    """Response length target"""
    BRIEF = "brief"          # 1-2 sentences
    MEDIUM = "medium"        # 3-5 sentences
    DETAILED = "detailed"    # 6+ sentences


class UncertaintyAction(StrEnum):
    """How to handle uncertainty"""
    ANSWER = "answer"                    # Provide confident answer
    HEDGE = "hedge"                      # Answer with hedging language
    ASK_CLARIFYING = "ask_clarifying"   # Ask for more info
    REFUSE = "refuse"                    # Politely decline to answer


class KnowledgeClaimType(StrEnum):
    """Type of knowledge claim being made"""
    PERSONAL_EXPERIENCE = "personal_experience"
    COMMON_KNOWLEDGE = "general_common_knowledge"
    DOMAIN_EXPERT = "domain_expert"
    SPECULATIVE = "speculative"
    NONE = "none"  # No knowledge claim (e.g., just asking questions)


class Tone(StrEnum):
    """Curated emotional tone vocabulary (from valence/arousal mapping)"""
    # Positive valence, high arousal
    WARM_ENTHUSIASTIC = "warm_enthusiastic"
    EXCITED_ENGAGED = "excited_engaged"

    # Positive valence, moderate arousal
    THOUGHTFUL_ENGAGED = "thoughtful_engaged"
    WARM_CONFIDENT = "warm_confident"
    FRIENDLY_RELAXED = "friendly_relaxed"

    # Positive valence, low arousal
    CONTENT_CALM = "content_calm"
    SATISFIED_PEACEFUL = "satisfied_peaceful"

    # Neutral valence
    NEUTRAL_CALM = "neutral_calm"
    PROFESSIONAL_COMPOSED = "professional_composed"
    MATTER_OF_FACT = "matter_of_fact"

    # Negative valence, high arousal
    FRUSTRATED_TENSE = "frustrated_tense"
    ANXIOUS_STRESSED = "anxious_stressed"
    DEFENSIVE_AGITATED = "defensive_agitated"

    # Negative valence, moderate arousal
    CONCERNED_EMPATHETIC = "concerned_empathetic"
    DISAPPOINTED_RESIGNED = "disappointed_resigned"

    # Negative valence, low arousal
    SAD_SUBDUED = "sad_subdued"
    TIRED_WITHDRAWN = "tired_withdrawn"


# ============================================================================
# Sub-models for IR Components
# ============================================================================

class ConversationFrame(BaseModel):
    """Context and goals for the current interaction"""

    interaction_mode: InteractionMode = Field(
        description="Type of conversation (casual, interview, support, etc.)"
    )

    goal: ConversationGoal = Field(
        description="Primary objective of this conversation"
    )

    success_criteria: list[str] | None = Field(
        default=None,
        description="Optional specific objectives to achieve",
        examples=[["Understand user's main concern", "Offer 2-3 solutions", "Confirm next steps"]]
    )


class ResponseStructure(BaseModel):
    """Core content and reasoning of the response"""

    intent: str = Field(
        description="What the persona wants to communicate",
        examples=["Share personal experience with UX research", "Ask clarifying question about requirements"]
    )

    stance: str | None = Field(
        default=None,
        description="Persona's opinion/position on the topic (if applicable)",
        examples=["Supports remote work flexibility", "Skeptical of AI replacing designers"]
    )

    rationale: str | None = Field(
        default=None,
        description="Why this stance (values + experience + uncertainty)",
        examples=["Based on 5 years remote work (experience) + work-life balance value (values)"]
    )

    elasticity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Openness to changing mind (0=rigid, 1=fully flexible). Tied to Openness + cognitive_complexity"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Certainty in the response (0=very uncertain, 1=very certain)"
    )

    competence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How equipped the persona is to engage with this topic "
            "(0=completely out of depth, 1=deep expert). "
            "Distinct from confidence: a knowledgeable persona can feel uncertain "
            "(low confidence, high competence), and vice versa."
        ),
    )


class CommunicationStyle(BaseModel):
    """How the message should be delivered"""

    tone: Tone = Field(
        description="Emotional affect (from valence/arousal mapping)"
    )

    verbosity: Verbosity = Field(
        description="Target response length"
    )

    formality: float = Field(
        ge=0.0,
        le=1.0,
        description="Formality level (0=casual, 1=very formal). Context-adjusted from social role mode"
    )

    directness: float = Field(
        ge=0.0,
        le=1.0,
        description="How direct vs indirect (0=very indirect/diplomatic, 1=very direct/blunt)"
    )


class KnowledgeAndDisclosure(BaseModel):
    """What to reveal and how to handle knowledge boundaries"""

    disclosure_level: float = Field(
        ge=0.0,
        le=1.0,
        description="How much personal info to share (from disclosure function: topic × trust × context × mood)"
    )

    uncertainty_action: UncertaintyAction = Field(
        description="How to handle uncertain knowledge"
    )

    knowledge_claim_type: KnowledgeClaimType = Field(
        description="Type of knowledge being claimed"
    )

    @field_validator('knowledge_claim_type')
    @classmethod
    def validate_claim_type_with_confidence(cls, v: "KnowledgeClaimType", info: Any) -> "KnowledgeClaimType":
        """Ensure claim type aligns with confidence"""
        # Note: Full validation happens in IR validator with persona proficiency
        return v


# Type alias for JSON-serializable scalar values (for delta tracking)
JsonScalar = Union[float, int, str, bool, None]


class Citation(BaseModel):
    """
    Delta-based citation for complete traceability.

    Tracks not just "what influenced" but "how and by how much".
    Supports both numeric (directness, confidence) and non-numeric
    (tone, verbosity) field changes.
    """

    source_type: Literal["base", "trait", "value", "goal", "rule", "state", "memory", "constraint"] = Field(
        description="What influenced this decision (added 'base' for initialization, 'constraint' for clamps)"
    )

    source_id: str = Field(
        description="Specific trait/value/rule/constraint identifier",
        examples=["conscientiousness", "self_direction", "work_social_role", "mood_positive", "privacy_filter"]
    )

    effect: str = Field(
        description="Human-readable description of the influence",
        examples=[
            "High conscientiousness → planned, structured response",
            "Self-direction value → emphasizes autonomy in stance",
            "Work social role → increased formality",
            "Positive mood → warmer tone"
        ]
    )

    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Strength of this influence (0-1)",
        default=1.0
    )

    # NEW: Delta tracking fields
    target_field: str | None = Field(
        default=None,
        description="IR field being modified (dot notation)",
        examples=["communication_style.directness", "response_structure.confidence", "communication_style.tone"]
    )

    operation: Literal["set", "add", "multiply", "clamp", "override", "blend"] | None = Field(
        default=None,
        description="Type of modification applied"
    )

    value_before: JsonScalar | None = Field(
        default=None,
        description="Value before modification (float, int, str, bool for enums)"
    )

    value_after: JsonScalar | None = Field(
        default=None,
        description="Value after modification"
    )

    delta: float | None = Field(
        default=None,
        description="Numeric delta (value_after - value_before), auto-computed for numeric values"
    )

    reason: str | None = Field(
        default=None,
        description="Short explanation of why this modification was made"
    )

    @model_validator(mode="after")
    def validate_and_compute_delta(self) -> "Citation":
        """Validate consistency and auto-compute delta for numeric values"""
        # If operation is specified, target_field should be too
        if self.operation and not self.target_field:
            raise ValueError("operation requires target_field")

        # Auto-compute delta for numeric values
        if isinstance(self.value_before, (int, float)) and isinstance(self.value_after, (int, float)):
            computed_delta = float(self.value_after) - float(self.value_before)

            if self.delta is None:
                self.delta = round(computed_delta, 6)  # Auto-fill
            else:
                # Validate manually specified delta matches
                if abs(self.delta - computed_delta) > 1e-5:
                    raise ValueError(
                        f"delta mismatch: specified {self.delta} vs computed {computed_delta} (tolerance 1e-5)"
                    )

        return self


# ============================================================================
# Memory Operations (Phase 4 Prep)
# ============================================================================

class MemoryReadRequest(BaseModel):
    """Request to read from memory (future: fact/preference lookup)"""

    query_type: Literal["fact", "preference", "relationship", "episode"] = Field(
        description="Type of memory to retrieve"
    )

    query: str = Field(
        description="What to look up",
        examples=["User's job title", "Preferred communication style", "Previous conversation about AI"]
    )

    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for returned memories"
    )


class MemoryWriteIntent(BaseModel):
    """Intent to write to memory (not immediate execution - queued for Phase 4)"""

    content_type: Literal["fact", "preference", "relationship", "episode"] = Field(
        description="Type of memory to store"
    )

    content: str = Field(
        description="What to remember",
        examples=[
            "User prefers remote work",
            "User seems stressed about deadlines",
            "Discussed UX research methodologies"
        ]
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this memory (0=uncertain inference, 1=explicitly stated)"
    )

    privacy_level: float = Field(
        ge=0.0,
        le=1.0,
        description="Sensitivity of this information (0=public, 1=highly private)"
    )

    source: str = Field(
        description="Origin of this memory",
        examples=["user_stated", "inferred_from_context", "observed_behavior"]
    )


class MemoryOps(BaseModel):
    """
    Memory operations channel (separated from response generation).

    Phase 4 will use this to prevent leaky memory or forgotten facts.
    For now, Turn Planner populates this but it's not executed.
    """

    read_requests: list[MemoryReadRequest] = Field(
        default_factory=list,
        description="Memory lookups needed for this turn"
    )

    write_intents: list[MemoryWriteIntent] = Field(
        default_factory=list,
        description="Memories to store after this turn"
    )

    write_policy: Literal["strict", "lenient"] = Field(
        default="strict",
        description="Strict = only write high-confidence memories, Lenient = write all"
    )


# ============================================================================
# Safety & Constraints
# ============================================================================

class ClampRecord(BaseModel):
    """Record of a value being clamped by constraints"""

    proposed: float = Field(description="Value before clamping")
    actual: float = Field(description="Value after clamping")
    minimum: float | None = Field(default=None, description="Lower bound (if applicable)")
    maximum: float | None = Field(default=None, description="Upper bound (if applicable)")
    reason: str | None = Field(
        default=None,
        description="Why clamped",
        examples=["Privacy filter", "Bounds [0,1]", "Claim policy"]
    )


class SafetyPlan(BaseModel):
    """
    Active safety constraints and blocks for this turn.

    Makes constraint enforcement auditable and transparent.
    """

    active_constraints: list[str] = Field(
        default_factory=list,
        description="Which constraints are active",
        examples=[["privacy_filter", "claim_policy", "must_avoid"]]
    )

    blocked_topics: list[str] = Field(
        default_factory=list,
        description="Topics from must_avoid that were matched",
        examples=[["employer_name", "participant_data"]]
    )

    clamped_fields: dict[str, list[ClampRecord]] = Field(
        default_factory=dict,
        description="Fields that were clamped (supports multiple clamps per field)"
    )

    pattern_blocks: list[str] = Field(
        default_factory=list,
        description="Response patterns that were blocked and why",
        examples=[["Pattern 'share_work_story' blocked: mentions must_avoid 'employer_name'"]]
    )


# ============================================================================
# Main IR Model
# ============================================================================

class IntermediateRepresentation(BaseModel):
    """
    Complete structured plan for a persona's response (production-hardened).

    This is the output of the Turn Planner and input to the Text Renderer.
    All behavior should be testable by examining this structure.

    Production features:
    - Delta-based citations for debuggability
    - Safety plan for constraint transparency
    - Memory ops channel for Phase 4 preparation
    - Deterministic normalization for golden tests
    """

    # Conversation context
    conversation_frame: ConversationFrame

    # Core response content
    response_structure: ResponseStructure

    # Communication style
    communication_style: CommunicationStyle

    # Knowledge & disclosure
    knowledge_disclosure: KnowledgeAndDisclosure

    # Traceability (enhanced with deltas)
    citations: list[Citation] = Field(
        default_factory=list,
        description="Delta-based citations showing all modifications"
    )

    # NEW: Safety & constraints
    safety_plan: SafetyPlan = Field(
        default_factory=SafetyPlan,
        description="Active constraints, blocks, and clamps"
    )

    # NEW: Memory operations (Phase 4 prep)
    memory_ops: MemoryOps = Field(
        default_factory=MemoryOps,
        description="Memory read/write intents for this turn"
    )

    # Metadata
    turn_id: str | None = Field(
        default=None,
        description="Unique identifier for this turn (for logging/debugging)"
    )

    seed: int | None = Field(
        default=None,
        description="Random seed used for this turn (for deterministic reproduction)"
    )

    def normalize(self, ndigits: int = 3) -> "IntermediateRepresentation":
        """Return normalized copy for deterministic comparison (test/QA only)"""
        return normalize_ir(self, ndigits=ndigits)

    def to_json_deterministic(self, ndigits: int = 3) -> str:
        """
        Convert to deterministic JSON string for golden tests.

        Features:
        - Quantized floats (default 3 decimals)
        - Sorted keys
        - Sorted citations and safety plan lists
        - Removes runtime IDs (turn_id)
        """
        return ir_to_deterministic_json(self, ndigits=ndigits)

    model_config = {"json_schema_extra": {
        "examples": [{
            "conversation_frame": {
                "interaction_mode": "casual_chat",
                "goal": "explore_ideas",
                "success_criteria": None
            },
            "response_structure": {
                "intent": "Share perspective on AI in UX research",
                "stance": "AI is useful tool but can't replace human empathy",
                "rationale": "Based on 8 years UX research experience + benevolence value + moderate openness",
                "elasticity": 0.6,
                "confidence": 0.75
            },
            "communication_style": {
                "tone": "thoughtful_engaged",
                "verbosity": "medium",
                "formality": 0.4,
                "directness": 0.7
            },
            "knowledge_disclosure": {
                "disclosure_level": 0.6,
                "uncertainty_action": "answer",
                "knowledge_claim_type": "domain_expert"
            },
            "citations": [
                {
                    "source_type": "trait",
                    "source_id": "openness",
                    "effect": "Moderate openness (0.75) → interested in AI but with critical perspective",
                    "weight": 0.8
                },
                {
                    "source_type": "value",
                    "source_id": "benevolence",
                    "effect": "High benevolence (0.78) → emphasizes human empathy in reasoning",
                    "weight": 0.9
                },
                {
                    "source_type": "state",
                    "source_id": "mood_positive",
                    "effect": "Positive mood → thoughtful/engaged tone",
                    "weight": 0.5
                }
            ],
            "turn_id": "conv_abc123_turn_05",
            "seed": 42
        }]
    }}


# ============================================================================
# Validation Result Models
# ============================================================================

class ValidationViolation(BaseModel):
    """A single validation failure"""

    violation_type: str = Field(
        description="Category of violation",
        examples=["invariant_contradiction", "knowledge_boundary_exceeded", "style_out_of_bounds"]
    )

    severity: Literal["error", "warning"] = Field(
        description="How serious is this violation"
    )

    message: str = Field(
        description="Human-readable explanation"
    )

    field_path: str | None = Field(
        default=None,
        description="IR field that caused the violation (dot notation)",
        examples=["knowledge_disclosure.knowledge_claim_type", "response_structure.stance"]
    )

    suggested_fix: str | None = Field(
        default=None,
        description="How to repair this violation"
    )


class IRValidationResult(BaseModel):
    """Result of validating an IR against persona profile"""

    passed: bool = Field(
        description="Whether IR passed all checks"
    )

    violations: list[ValidationViolation] = Field(
        default_factory=list,
        description="List of validation failures (if any)"
    )

    checked_invariants: list[str] = Field(
        description="Which invariants were checked",
        examples=[["identity_facts", "cannot_claim", "must_avoid", "knowledge_boundaries"]]
    )

    timestamp: str | None = Field(
        default=None,
        description="When validation occurred (ISO format)"
    )


# ============================================================================
# NOTE: Example usage has been moved to examples/ir_usage_example.py
# to keep schema module focused on data structures only.
# ============================================================================


# ============================================================================
# Normalization Utilities (for deterministic testing)
# ============================================================================

from collections.abc import Mapping, Sequence


def _quantize(obj: Any, ndigits: int = 3) -> Any:
    """
    Recursively quantize floats in a data structure.

    Prevents float rounding differences from breaking golden tests.
    """
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, Mapping):
        # Sort keys for deterministic structure
        return {k: _quantize(obj[k], ndigits) for k in sorted(obj.keys())}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_quantize(x, ndigits) for x in obj]
    return obj


def normalize_ir(
    ir: "IntermediateRepresentation",
    ndigits: int = 3
) -> "IntermediateRepresentation":
    """
    Normalize IR for deterministic comparison.

    Prevents test flakiness from:
    - Float rounding differences across platforms
    - Dict iteration order
    - Unsorted collections

    Args:
        ir: IR to normalize
        ndigits: Decimal places for float rounding

    Returns:
        Normalized copy of IR
    """
    # Get python dict representation
    data = ir.model_dump(mode="python")

    # Sort citations by key fields for deterministic order
    if "citations" in data and isinstance(data["citations"], list):
        data["citations"] = sorted(
            data["citations"],
            key=lambda c: (
                c.get("target_field") or "",
                c.get("operation") or "",
                c.get("source_type") or "",
                c.get("source_id") or "",
                c.get("effect") or "",
            ),
        )

    # Sort safety_plan lists
    sp = data.get("safety_plan")
    if isinstance(sp, dict):
        for k in ("active_constraints", "blocked_topics", "pattern_blocks"):
            if isinstance(sp.get(k), list):
                sp[k] = sorted(sp[k])

    # Recursive quantization of all floats
    data = _quantize(data, ndigits=ndigits)

    # Reconstruct IR from normalized data
    return IntermediateRepresentation.model_validate(data)


def ir_to_deterministic_json(
    ir: "IntermediateRepresentation",
    ndigits: int = 3
) -> str:
    """
    Convert IR to byte-identical JSON for golden tests.

    Pydantic v2 compatible (uses json.dumps, not model_dump_json(sort_keys=...)).

    Args:
        ir: IR to serialize
        ndigits: Decimal places for float rounding

    Returns:
        Deterministic JSON string

    Usage:
        ir1_json = ir_to_deterministic_json(ir1)
        ir2_json = ir_to_deterministic_json(ir2)
        assert ir1_json == ir2_json  # Exact byte-for-byte match
    """
    # Normalize first
    normalized = normalize_ir(ir, ndigits=ndigits)

    # Get JSON-serializable dict
    payload: dict[str, Any] = normalized.model_dump(mode="json")

    # Remove runtime IDs from golden tests (optional)
    payload.pop("turn_id", None)

    # Serialize with sorted keys and consistent separators
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        separators=(",", ":")  # Removes whitespace variability
    )


# ============================================================================
# Helper Functions (for Turn Planner ergonomics)
# ============================================================================

def apply_numeric_modifier(
    *,
    citations: list[Citation],
    source_type: str,
    source_id: str,
    target_field: str,
    operation: Literal["set", "add", "multiply", "clamp", "override", "blend"],
    before: float,
    after: float,
    effect: str,
    weight: float,
    reason: str | None = None,
) -> float:
    """
    Apply a numeric modifier and auto-generate citation.

    Prevents "forgot to add citation" bugs.

    Args:
        citations: List to append citation to
        source_type: "trait", "value", "rule", "state", "constraint", etc.
        source_id: Specific identifier (e.g., "agreeableness")
        target_field: IR field in dot notation (e.g., "communication_style.directness")
        operation: Type of modification
        before: Value before modification
        after: Value after modification
        effect: Human-readable description
        weight: Importance (0-1)
        reason: Optional detailed explanation

    Returns:
        The 'after' value (for chaining)

    Example:
        directness = apply_numeric_modifier(
            citations=citations,
            source_type="trait",
            source_id="agreeableness",
            target_field="communication_style.directness",
            operation="add",
            before=0.680,
            after=0.547,
            effect="High agreeableness reduces directness",
            weight=0.8,
            reason="Inverse correlation: A=0.72 → modifier=-0.133"
        )
    """
    citations.append(Citation(
        source_type=source_type,  # type: ignore
        source_id=source_id,
        target_field=target_field,
        operation=operation,
        value_before=before,
        value_after=after,
        effect=effect,
        weight=weight,
        reason=reason,
    ))
    return after


def apply_enum_modifier(
    *,
    citations: list[Citation],
    source_type: str,
    source_id: str,
    target_field: str,
    operation: Literal["set", "override"],
    before: str,
    after: str,
    effect: str,
    weight: float,
    reason: str | None = None,
) -> str:
    """
    Apply an enum/string modifier and auto-generate citation.

    Use for tone, verbosity, uncertainty_action, etc.

    Args:
        Same as apply_numeric_modifier but before/after are strings

    Returns:
        The 'after' value (for chaining)

    Example:
        tone = apply_enum_modifier(
            citations=citations,
            source_type="state",
            source_id="mood",
            target_field="communication_style.tone",
            operation="set",
            before="neutral_calm",
            after="warm_enthusiastic",
            effect="Positive mood (v=0.6, a=0.7) → warm tone",
            weight=1.0
        )
    """
    citations.append(Citation(
        source_type=source_type,  # type: ignore
        source_id=source_id,
        target_field=target_field,
        operation=operation,
        value_before=before,
        value_after=after,
        effect=effect,
        weight=weight,
        reason=reason,
        delta=None  # No numeric delta for enums
    ))
    return after
