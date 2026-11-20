"""
IR Validator — Pre-generation validation of Intermediate Representations.

Checks the IR for internal consistency, range validity, citation completeness,
and constraint compliance before it's sent to the response generator.

Severity levels:
- ERROR: Blocks generation (incoherent IR would produce bad output)
- WARNING: Logged but generation proceeds (minor inconsistency)
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    KnowledgeClaimType,
    UncertaintyAction,
)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single validation issue found in the IR."""
    severity: Severity
    field: str
    message: str
    expected: Any = None
    actual: Any = None


def validate_ir(ir: IntermediateRepresentation) -> list[ValidationIssue]:
    """
    Validate an IR for internal consistency and correctness.

    Args:
        ir: The IntermediateRepresentation to validate.

    Returns:
        List of ValidationIssue objects. Empty list = clean IR.
    """
    issues: list[ValidationIssue] = []

    issues.extend(_check_confidence_uncertainty_alignment(ir))
    issues.extend(_check_range_validity(ir))
    issues.extend(_check_citation_completeness(ir))
    issues.extend(_check_constraint_compliance(ir))
    issues.extend(_check_knowledge_claim_coherence(ir))

    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    """Check if any issues are ERROR severity."""
    return any(i.severity == Severity.ERROR for i in issues)


def _check_confidence_uncertainty_alignment(ir: IntermediateRepresentation) -> list[ValidationIssue]:
    """Check that confidence and uncertainty_action are consistent."""
    issues = []
    confidence = ir.response_structure.confidence
    action = ir.knowledge_disclosure.uncertainty_action

    # High confidence + REFUSE is contradictory
    if confidence > 0.8 and action == UncertaintyAction.REFUSE:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            field="knowledge_disclosure.uncertainty_action",
            message=f"High confidence ({confidence:.2f}) with REFUSE action is contradictory",
            expected="ANSWER or HEDGE",
            actual=action.value,
        ))

    # Very low confidence + ANSWER without hedging is risky
    if confidence < 0.3 and action == UncertaintyAction.ANSWER:
        claim = ir.knowledge_disclosure.knowledge_claim_type
        if claim not in (KnowledgeClaimType.PERSONAL_EXPERIENCE, KnowledgeClaimType.COMMON_KNOWLEDGE):
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field="knowledge_disclosure.uncertainty_action",
                message=f"Low confidence ({confidence:.2f}) with ANSWER action for {claim.value} claim",
                expected="HEDGE or ASK_CLARIFYING",
                actual=action.value,
            ))

    return issues


def _check_range_validity(ir: IntermediateRepresentation) -> list[ValidationIssue]:
    """Defense-in-depth range checks on all 0-1 float fields."""
    issues = []

    checks = [
        ("response_structure.confidence", ir.response_structure.confidence),
        ("response_structure.competence", ir.response_structure.competence),
        ("communication_style.formality", ir.communication_style.formality),
        ("communication_style.directness", ir.communication_style.directness),
        ("knowledge_disclosure.disclosure_level", ir.knowledge_disclosure.disclosure_level),
    ]

    if ir.response_structure.elasticity is not None:
        checks.append(("response_structure.elasticity", ir.response_structure.elasticity))

    for field_name, value in checks:
        if value < 0.0 or value > 1.0:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field=field_name,
                message=f"Value {value:.4f} is outside valid range [0, 1]",
                expected="[0.0, 1.0]",
                actual=value,
            ))

    return issues


def _check_citation_completeness(ir: IntermediateRepresentation) -> list[ValidationIssue]:
    """Check that behavioral floats have citations explaining them."""
    issues = []

    citation_sources = {c.source_id for c in ir.citations}

    # Key behavioral fields that should have at least one citation
    behavioral_fields = [
        ("confidence", ir.response_structure.confidence, 0.5),
        ("disclosure_level", ir.knowledge_disclosure.disclosure_level, 0.5),
    ]

    for field_name, value, default in behavioral_fields:
        # Only flag if the value is non-default (i.e., something computed it)
        if abs(value - default) > 0.05 and not ir.citations:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field=field_name,
                message=f"Non-default value ({value:.2f}) with no citations",
                expected="At least one citation",
                actual="0 citations",
            ))

    return issues


def _check_constraint_compliance(ir: IntermediateRepresentation) -> list[ValidationIssue]:
    """Check that must_avoid topics don't appear in stance."""
    issues = []

    must_avoid = ir.safety_plan.must_avoid
    cannot_claim = ir.safety_plan.cannot_claim

    stance = ir.response_structure.stance or ""
    stance_lower = stance.lower()

    for topic in must_avoid:
        if topic.lower() in stance_lower:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field="response_structure.stance",
                message=f"Stance contains must_avoid topic: '{topic}'",
                expected=f"No mention of '{topic}'",
                actual=stance[:100],
            ))

    for claim in cannot_claim:
        if claim.lower() in stance_lower:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                field="response_structure.stance",
                message=f"Stance contains cannot_claim role: '{claim}'",
                expected=f"No mention of '{claim}'",
                actual=stance[:100],
            ))

    return issues


def _check_knowledge_claim_coherence(ir: IntermediateRepresentation) -> list[ValidationIssue]:
    """Check knowledge claim type consistency."""
    issues = []

    claim = ir.knowledge_disclosure.knowledge_claim_type
    confidence = ir.response_structure.confidence
    competence = ir.response_structure.competence

    # Domain expert claim with low competence
    if claim == KnowledgeClaimType.DOMAIN_EXPERT and competence < 0.5:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            field="knowledge_disclosure.knowledge_claim_type",
            message=f"DOMAIN_EXPERT claim with low competence ({competence:.2f})",
            expected="competence >= 0.5 for expert claims",
            actual=f"competence={competence:.2f}",
        ))

    # NONE claim type but answering confidently
    if claim == KnowledgeClaimType.NONE and confidence > 0.7:
        action = ir.knowledge_disclosure.uncertainty_action
        if action == UncertaintyAction.ANSWER:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                field="knowledge_disclosure.knowledge_claim_type",
                message=f"No knowledge claim but answering with high confidence ({confidence:.2f})",
                expected="Non-NONE claim type if answering confidently",
                actual=claim.value,
            ))

    return issues
