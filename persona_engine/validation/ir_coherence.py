"""
IR Coherence Validator — internal consistency checks.

Validates that an IR's fields are logically consistent with each other,
independent of the persona that produced it.

Rules:
- Non-expert claim type requires low-to-moderate confidence
- High confidence with hedge uncertainty action is contradictory
- Disclosure level should be consistent with uncertainty action
- Tone should be plausible given confidence and uncertainty
- Citation trail should cover key fields
"""

from __future__ import annotations

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    KnowledgeClaimType,
    Tone,
    UncertaintyAction,
    ValidationViolation,
    Verbosity,
)

# Tones considered "negative"
_NEGATIVE_TONES = frozenset({
    Tone.FRUSTRATED_TENSE,
    Tone.ANXIOUS_STRESSED,
    Tone.DEFENSIVE_AGITATED,
    Tone.DISAPPOINTED_RESIGNED,
    Tone.SAD_SUBDUED,
    Tone.TIRED_WITHDRAWN,
})

# Tones considered "high energy positive"
_POSITIVE_TONES = frozenset({
    Tone.WARM_ENTHUSIASTIC,
    Tone.EXCITED_ENGAGED,
    Tone.WARM_CONFIDENT,
})


def validate_ir_coherence(ir: IntermediateRepresentation) -> list[ValidationViolation]:
    """
    Check that IR fields are internally consistent.

    Returns:
        List of violations (empty = coherent)
    """
    violations: list[ValidationViolation] = []

    confidence = ir.response_structure.confidence
    claim = ir.knowledge_disclosure.knowledge_claim_type
    uncertainty = ir.knowledge_disclosure.uncertainty_action
    disclosure = ir.knowledge_disclosure.disclosure_level
    tone = ir.communication_style.tone
    verbosity = ir.communication_style.verbosity

    # Rule 1: Non-expert claim types shouldn't have very high confidence
    if claim in (KnowledgeClaimType.SPECULATIVE, KnowledgeClaimType.NONE):
        if confidence > 0.85:
            violations.append(ValidationViolation(
                violation_type="confidence_claim_mismatch",
                severity="warning",
                message=(
                    f"Confidence {confidence:.2f} is very high for "
                    f"'{claim.value}' claim type — expected < 0.85"
                ),
                field_path="response_structure.confidence",
                suggested_fix="Lower confidence or upgrade claim type",
            ))

    # Rule 2: Domain expert with very low confidence is suspicious
    if claim == KnowledgeClaimType.DOMAIN_EXPERT and confidence < 0.4:
        violations.append(ValidationViolation(
            violation_type="expert_low_confidence",
            severity="warning",
            message=(
                f"Domain expert claim with confidence {confidence:.2f} — "
                "experts should typically have confidence >= 0.4"
            ),
            field_path="knowledge_disclosure.knowledge_claim_type",
            suggested_fix="Downgrade claim type or raise confidence",
        ))

    # Rule 3: High confidence + hedge is contradictory
    if confidence > 0.8 and uncertainty == UncertaintyAction.HEDGE:
        violations.append(ValidationViolation(
            violation_type="confidence_uncertainty_contradiction",
            severity="warning",
            message=(
                f"Confidence {confidence:.2f} with hedge action — "
                "high confidence typically maps to answer, not hedge"
            ),
            field_path="knowledge_disclosure.uncertainty_action",
            suggested_fix="Use 'answer' for high confidence, or lower confidence",
        ))

    # Rule 4: Very low confidence + direct answer is contradictory
    if confidence < 0.3 and uncertainty == UncertaintyAction.ANSWER:
        violations.append(ValidationViolation(
            violation_type="low_confidence_direct_answer",
            severity="warning",
            message=(
                f"Confidence {confidence:.2f} with direct answer — "
                "low confidence should typically hedge or ask for clarification"
            ),
            field_path="knowledge_disclosure.uncertainty_action",
            suggested_fix="Use 'hedge' or 'ask_clarifying' for low confidence",
        ))

    # Rule 5: Refuse action but high disclosure is contradictory
    if uncertainty == UncertaintyAction.REFUSE and disclosure > 0.7:
        violations.append(ValidationViolation(
            violation_type="refuse_high_disclosure",
            severity="warning",
            message=(
                f"Refusing to answer but disclosure level is {disclosure:.2f} — "
                "a refusal should correspond to low disclosure"
            ),
            field_path="knowledge_disclosure.disclosure_level",
            suggested_fix="Lower disclosure level when refusing to answer",
        ))

    # Rule 6: Negative tone with very high confidence is unusual
    if tone in _NEGATIVE_TONES and confidence > 0.9:
        violations.append(ValidationViolation(
            violation_type="negative_tone_high_confidence",
            severity="warning",
            message=(
                f"Negative tone '{tone.value}' with confidence {confidence:.2f} — "
                "negative emotional states usually reduce certainty"
            ),
            field_path="communication_style.tone",
            suggested_fix="Consider if stress/anxiety should lower confidence",
        ))

    # Rule 7: Elasticity and confidence should have some tension
    elasticity = ir.response_structure.elasticity
    if elasticity is not None and elasticity < 0.15 and confidence < 0.3:
        violations.append(ValidationViolation(
            violation_type="rigid_uncertain",
            severity="warning",
            message=(
                f"Very rigid (elasticity {elasticity:.2f}) but very uncertain "
                f"(confidence {confidence:.2f}) — rigid personas are typically "
                "more confident in their positions"
            ),
            field_path="response_structure.elasticity",
            suggested_fix="Raise confidence or increase elasticity",
        ))

    # Rule 8: Citation completeness — key fields should have at least one citation
    _check_citation_completeness(ir, violations)

    return violations


def _check_citation_completeness(
    ir: IntermediateRepresentation,
    violations: list[ValidationViolation],
) -> None:
    """Check that key IR fields have supporting citations."""
    cited_fields = {c.target_field for c in ir.citations if c.target_field}

    key_fields = [
        "communication_style.formality",
        "communication_style.directness",
        "response_structure.confidence",
    ]

    missing = [f for f in key_fields if f not in cited_fields]
    if missing:
        violations.append(ValidationViolation(
            violation_type="incomplete_citations",
            severity="warning",
            message=f"Missing citations for key fields: {', '.join(missing)}",
            field_path="citations",
            suggested_fix="Ensure TraceContext records citations for all computed fields",
        ))
