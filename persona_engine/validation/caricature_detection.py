"""
Caricature Detection — Cross-module coherence check.

Detects when individually-bounded modifiers stack to create extreme
composite profiles that no longer resemble plausible human behavior.

The Research Scientist review identified this risk: a High-N, Low-A,
Low-C, Low-E persona gets compounding negative modifiers that push
confidence near floor, maximize hedging, elevate arousal, AND increase
directness — but genuinely anxious people withdraw rather than confront.

This module catches such contradictions by checking the IR after all
modifiers are applied.
"""

from __future__ import annotations

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    ValidationViolation,
)
from persona_engine.schema.persona_schema import Persona

# Thresholds for extreme values (floor/ceiling detection)
_FLOOR = 0.15
_CEILING = 0.85

# Maximum number of simultaneously extreme fields before flagging
_MAX_SIMULTANEOUS_EXTREMES = 4


def validate_caricature(
    ir: IntermediateRepresentation,
    persona: Persona,
) -> list[ValidationViolation]:
    """Check for caricature accumulation across stacked modifier modules.

    Runs three checks:
    1. Extreme value accumulation — too many fields pinned at floor/ceiling
    2. Anxiety-directness contradiction — high neuroticism + high directness
    3. Confidence-disclosure incoherence — very low confidence + very high disclosure

    Returns a list of ValidationViolation warnings.
    """
    violations: list[ValidationViolation] = []

    rs = ir.response_structure
    cs = ir.communication_style
    kd = ir.knowledge_disclosure

    # ----------------------------------------------------------------
    # Check 1: Extreme value accumulation
    # ----------------------------------------------------------------
    extreme_fields: list[str] = []

    field_values = {
        "confidence": rs.confidence,
        "competence": rs.competence,
        "elasticity": rs.elasticity,
        "formality": cs.formality,
        "directness": cs.directness,
        "disclosure_level": kd.disclosure_level,
    }

    for name, val in field_values.items():
        if val <= _FLOOR or val >= _CEILING:
            extreme_fields.append(f"{name}={val:.2f}")

    if len(extreme_fields) >= _MAX_SIMULTANEOUS_EXTREMES:
        violations.append(ValidationViolation(
            violation_type="caricature_accumulation",
            severity="warning",
            message=(
                f"Caricature risk: {len(extreme_fields)} fields at extreme values "
                f"({', '.join(extreme_fields)}). Stacked modifiers may produce "
                f"implausible behavior. Consider dampening cross-module effects."
            ),
            field="response_structure",
            expected_range="At most 3 simultaneous extreme fields",
            actual_value=str(len(extreme_fields)),
        ))

    # ----------------------------------------------------------------
    # Check 2: Anxiety-directness contradiction
    # ----------------------------------------------------------------
    # Genuinely anxious people (High-N) tend to withdraw, not confront.
    # High directness + low confidence + high neuroticism is implausible.
    neuroticism = persona.psychology.big_five.neuroticism

    if (neuroticism > 0.7
            and rs.confidence < 0.3
            and cs.directness > 0.7):
        violations.append(ValidationViolation(
            violation_type="caricature_contradiction",
            severity="warning",
            message=(
                f"Anxiety-directness contradiction: neuroticism={neuroticism:.2f} "
                f"with confidence={rs.confidence:.2f} and directness={cs.directness:.2f}. "
                f"High-N individuals with low confidence typically reduce directness "
                f"(withdrawal response), not increase it."
            ),
            field="communication_style.directness",
            expected_range="directness < 0.5 when neuroticism > 0.7 and confidence < 0.3",
            actual_value=f"{cs.directness:.2f}",
            suggested_fix=(
                "Dampen directness when both neuroticism and low-confidence are active. "
                "Consider a cross-module cap: directness <= 1.0 - neuroticism * 0.3 "
                "when confidence < 0.3."
            ),
        ))

    # ----------------------------------------------------------------
    # Check 3: Confidence-disclosure incoherence
    # ----------------------------------------------------------------
    # Very low confidence + very high disclosure is psychologically unusual.
    # People unsure of themselves tend to guard rather than over-share.
    if rs.confidence < 0.2 and kd.disclosure_level > 0.8:
        violations.append(ValidationViolation(
            violation_type="caricature_contradiction",
            severity="warning",
            message=(
                f"Confidence-disclosure incoherence: confidence={rs.confidence:.2f} "
                f"but disclosure_level={kd.disclosure_level:.2f}. "
                f"Low-confidence individuals typically guard information rather than "
                f"over-disclose."
            ),
            field="knowledge_disclosure.disclosure_level",
            expected_range="disclosure < 0.6 when confidence < 0.2",
            actual_value=f"{kd.disclosure_level:.2f}",
        ))

    return violations
