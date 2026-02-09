"""
Persona Compliance Validator — checks IR against persona profile.

Validates that the IR respects the persona's psychological boundaries,
knowledge domains, and safety constraints.

Rules:
- Knowledge claim type must align with persona's domain proficiency
- Confidence should be plausible given proficiency for detected domain
- Directness should be in a plausible range given agreeableness
- Disclosure level should respect privacy constraints
- Stance must not violate invariants (cannot_claim, must_avoid)
- Formality should be plausible given communication preferences
"""

from __future__ import annotations

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    KnowledgeClaimType,
    ValidationViolation,
)
from persona_engine.schema.persona_schema import Persona

# Expert threshold matches turn_planner.py
_EXPERT_THRESHOLD = 0.7


def validate_persona_compliance(
    ir: IntermediateRepresentation,
    persona: Persona,
) -> list[ValidationViolation]:
    """
    Check that IR is consistent with the persona that produced it.

    Args:
        ir: The generated Intermediate Representation
        persona: The persona profile that should constrain the IR

    Returns:
        List of violations (empty = compliant)
    """
    violations: list[ValidationViolation] = []

    _check_knowledge_boundaries(ir, persona, violations)
    _check_trait_plausibility(ir, persona, violations)
    _check_invariant_compliance(ir, persona, violations)
    _check_disclosure_constraints(ir, persona, violations)
    _check_communication_prefs(ir, persona, violations)

    return violations


def _check_knowledge_boundaries(
    ir: IntermediateRepresentation,
    persona: Persona,
    violations: list[ValidationViolation],
) -> None:
    """Verify knowledge claims align with persona's expertise."""
    claim = ir.knowledge_disclosure.knowledge_claim_type
    confidence = ir.response_structure.confidence
    competence = ir.response_structure.competence

    # Find the best matching domain proficiency from citations
    domain_proficiency = _get_domain_proficiency(ir, persona)

    # Rule 1: Domain expert claim requires sufficient proficiency
    if claim == KnowledgeClaimType.DOMAIN_EXPERT:
        if domain_proficiency is not None and domain_proficiency < _EXPERT_THRESHOLD:
            violations.append(ValidationViolation(
                violation_type="knowledge_boundary_exceeded",
                severity="error",
                message=(
                    f"Domain expert claim with proficiency {domain_proficiency:.2f} "
                    f"— requires >= {_EXPERT_THRESHOLD:.2f}"
                ),
                field_path="knowledge_disclosure.knowledge_claim_type",
                suggested_fix="Downgrade to personal_experience or speculative",
            ))

    # Rule 2: Confidence should be plausible given proficiency
    if domain_proficiency is not None:
        # A persona with 0.3 proficiency having 0.95 confidence is suspicious
        max_plausible = min(1.0, domain_proficiency + 0.3)
        if confidence > max_plausible:
            violations.append(ValidationViolation(
                violation_type="confidence_exceeds_proficiency",
                severity="warning",
                message=(
                    f"Confidence {confidence:.2f} seems high for proficiency "
                    f"{domain_proficiency:.2f} — plausible max ~{max_plausible:.2f}"
                ),
                field_path="response_structure.confidence",
                suggested_fix="Lower confidence to align with domain proficiency",
            ))

    # Rule 3: Dunning-Kruger check — low competence + high confidence is an error
    # unless bias_simulator intentionally produces it for specific trait profiles
    if competence < 0.3 and confidence > 0.7:
        violations.append(ValidationViolation(
            violation_type="competence_confidence_mismatch",
            severity="error",
            message=(
                f"Low competence ({competence:.2f}) with high confidence "
                f"({confidence:.2f}) — persona claims certainty in a topic "
                "they are not equipped to discuss"
            ),
            field_path="response_structure.competence",
            suggested_fix="Lower confidence to align with competence, or investigate bias",
        ))

    # Rule 3: Check cannot_claim against claim type
    if persona.invariants and persona.invariants.cannot_claim:
        stance = ir.response_structure.stance or ""
        rationale = ir.response_structure.rationale or ""
        combined = f"{stance} {rationale}".lower()

        for forbidden in persona.invariants.cannot_claim:
            if forbidden.lower() in combined:
                violations.append(ValidationViolation(
                    violation_type="invariant_contradiction",
                    severity="error",
                    message=f"Stance/rationale contains forbidden claim: '{forbidden}'",
                    field_path="response_structure.stance",
                    suggested_fix=f"Remove reference to '{forbidden}' from stance/rationale",
                ))


def _check_trait_plausibility(
    ir: IntermediateRepresentation,
    persona: Persona,
    violations: list[ValidationViolation],
) -> None:
    """Check that style parameters are plausible given Big Five traits."""
    traits = persona.psychology.big_five
    directness = ir.communication_style.directness

    # Rule: High agreeableness (> 0.7) should correlate with lower directness
    # Allow a wide range but flag extreme contradictions
    if traits.agreeableness > 0.8 and directness > 0.85:
        violations.append(ValidationViolation(
            violation_type="trait_style_mismatch",
            severity="warning",
            message=(
                f"Directness {directness:.2f} is very high for agreeableness "
                f"{traits.agreeableness:.2f} — highly agreeable personas are "
                "typically more diplomatic"
            ),
            field_path="communication_style.directness",
            suggested_fix="Lower directness to reflect high agreeableness",
        ))

    # Rule: Low agreeableness (< 0.3) with very low directness
    if traits.agreeableness < 0.25 and directness < 0.2:
        violations.append(ValidationViolation(
            violation_type="trait_style_mismatch",
            severity="warning",
            message=(
                f"Directness {directness:.2f} is very low for agreeableness "
                f"{traits.agreeableness:.2f} — low-agreeable personas are "
                "typically more direct"
            ),
            field_path="communication_style.directness",
            suggested_fix="Raise directness to reflect low agreeableness",
        ))

    # Rule: High neuroticism should not produce very high confidence
    if traits.neuroticism > 0.8:
        confidence = ir.response_structure.confidence
        if confidence > 0.95:
            violations.append(ValidationViolation(
                violation_type="trait_confidence_mismatch",
                severity="warning",
                message=(
                    f"Confidence {confidence:.2f} is very high for neuroticism "
                    f"{traits.neuroticism:.2f} — high neuroticism typically "
                    "reduces confidence"
                ),
                field_path="response_structure.confidence",
                suggested_fix="Neuroticism modifier should reduce confidence",
            ))


def _check_invariant_compliance(
    ir: IntermediateRepresentation,
    persona: Persona,
    violations: list[ValidationViolation],
) -> None:
    """Check that safety plan reflects persona's invariants."""
    if not persona.invariants:
        return

    # Rule: must_avoid topics should appear in safety_plan if relevant
    must_avoid = persona.invariants.must_avoid or []
    stance = (ir.response_structure.stance or "").lower()
    rationale = (ir.response_structure.rationale or "").lower()
    intent = ir.response_structure.intent.lower()

    for topic in must_avoid:
        topic_lower = topic.lower()
        if topic_lower in stance or topic_lower in rationale or topic_lower in intent:
            # Topic appears in IR content but wasn't blocked
            if topic not in ir.safety_plan.blocked_topics:
                violations.append(ValidationViolation(
                    violation_type="must_avoid_leak",
                    severity="error",
                    message=(
                        f"Must-avoid topic '{topic}' appears in IR content "
                        "but wasn't recorded in safety_plan.blocked_topics"
                    ),
                    field_path="safety_plan.blocked_topics",
                    suggested_fix=f"Block topic '{topic}' in safety plan",
                ))


def _check_disclosure_constraints(
    ir: IntermediateRepresentation,
    persona: Persona,
    violations: list[ValidationViolation],
) -> None:
    """Check that disclosure respects privacy boundaries."""
    disclosure = ir.knowledge_disclosure.disclosure_level

    # Rule: Disclosure should not exceed persona's maximum openness
    # Base openness + extraversion boost + state = max ~0.9 in most cases
    base_openness = persona.disclosure_policy.base_openness if persona.disclosure_policy else 0.5
    max_plausible = min(1.0, base_openness + 0.4)

    if disclosure > max_plausible:
        violations.append(ValidationViolation(
            violation_type="disclosure_exceeds_policy",
            severity="warning",
            message=(
                f"Disclosure {disclosure:.2f} exceeds plausible max "
                f"{max_plausible:.2f} (base_openness={base_openness:.2f})"
            ),
            field_path="knowledge_disclosure.disclosure_level",
            suggested_fix="Clamp disclosure to privacy policy bounds",
        ))


def _check_communication_prefs(
    ir: IntermediateRepresentation,
    persona: Persona,
    violations: list[ValidationViolation],
) -> None:
    """Check style parameters against base communication preferences."""
    prefs = persona.psychology.communication
    formality = ir.communication_style.formality

    # Rule: Formality shouldn't deviate wildly from base (> 0.5 delta)
    base_formality = prefs.formality
    delta = abs(formality - base_formality)
    if delta > 0.5:
        violations.append(ValidationViolation(
            violation_type="formality_deviation",
            severity="warning",
            message=(
                f"Formality {formality:.2f} deviates {delta:.2f} from base "
                f"{base_formality:.2f} — social role blend typically stays "
                "within +/-0.5"
            ),
            field_path="communication_style.formality",
            suggested_fix="Check social role blend weights",
        ))


def _get_domain_proficiency(
    ir: IntermediateRepresentation,
    persona: Persona,
) -> float | None:
    """Extract domain proficiency from citations or persona domains."""
    # Look for proficiency in citations
    for citation in ir.citations:
        if citation.source_id == "domain_proficiency" and citation.value_after is not None:
            return float(citation.value_after)

    # Fallback: check if any domain was detected in citations
    for citation in ir.citations:
        if citation.source_type == "base" and "proficiency" in (citation.effect or "").lower():
            if citation.value_after is not None:
                return float(citation.value_after)

    return None
