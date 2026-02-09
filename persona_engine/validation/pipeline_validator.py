"""
Pipeline Validator — orchestrates all validation checks.

Combines IR coherence, persona compliance, and cross-turn consistency
into a single validation pass. Produces IRValidationResult.

Usage:
    validator = PipelineValidator(persona)

    # Single turn:
    result = validator.validate(ir)

    # Multi-turn:
    for each turn:
        ir = planner.generate_ir(ctx)
        result = validator.validate(ir, turn_number=n, topic="ux")
        if not result.passed:
            handle_violations(result.violations)
"""

from __future__ import annotations

from datetime import datetime, timezone

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    IRValidationResult,
    ValidationViolation,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.validation.cross_turn import CrossTurnTracker
from persona_engine.validation.ir_coherence import validate_ir_coherence
from persona_engine.validation.persona_compliance import validate_persona_compliance


class PipelineValidator:
    """
    Orchestrates all validation checks for the persona engine pipeline.

    Runs three validation layers:
    1. IR Coherence — internal field consistency
    2. Persona Compliance — IR vs persona profile
    3. Cross-Turn Consistency — multi-turn coherence

    Errors cause validation to fail. Warnings are informational.
    """

    def __init__(
        self,
        persona: Persona,
        fail_on_warnings: bool = False,
    ) -> None:
        """
        Args:
            persona: The persona profile to validate against
            fail_on_warnings: If True, warnings also cause validation to fail
        """
        self.persona = persona
        self.fail_on_warnings = fail_on_warnings
        self.cross_turn = CrossTurnTracker()

    def validate(
        self,
        ir: IntermediateRepresentation,
        turn_number: int = 0,
        topic: str = "",
    ) -> IRValidationResult:
        """
        Run all validation checks on an IR.

        Args:
            ir: The IR to validate
            turn_number: Current turn number (for cross-turn checks)
            topic: Current topic signature (for cross-turn checks)

        Returns:
            IRValidationResult with pass/fail and any violations
        """
        all_violations: list[ValidationViolation] = []
        checked: list[str] = []

        # Layer 1: IR Coherence
        coherence_violations = validate_ir_coherence(ir)
        all_violations.extend(coherence_violations)
        checked.extend([
            "confidence_claim_alignment",
            "confidence_uncertainty_alignment",
            "disclosure_consistency",
            "tone_confidence_plausibility",
            "citation_completeness",
        ])

        # Layer 2: Persona Compliance
        compliance_violations = validate_persona_compliance(ir, self.persona)
        all_violations.extend(compliance_violations)
        checked.extend([
            "knowledge_boundaries",
            "trait_plausibility",
            "invariant_compliance",
            "disclosure_constraints",
            "communication_preferences",
        ])

        # Layer 3: Cross-Turn Consistency
        cross_violations = self.cross_turn.validate_turn(
            ir, turn_number=turn_number, topic=topic
        )
        all_violations.extend(cross_violations)
        checked.extend([
            "parameter_swing",
            "expertise_consistency",
            "stance_consistency",
        ])

        # Determine pass/fail
        errors = [v for v in all_violations if v.severity == "error"]
        warnings = [v for v in all_violations if v.severity == "warning"]

        if self.fail_on_warnings:
            passed = len(all_violations) == 0
        else:
            passed = len(errors) == 0

        return IRValidationResult(
            passed=passed,
            violations=all_violations,
            checked_invariants=checked,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def validate_single(
        self,
        ir: IntermediateRepresentation,
    ) -> IRValidationResult:
        """
        Validate a single IR without cross-turn tracking.

        Convenience method for one-shot validation (no history).
        """
        violations: list[ValidationViolation] = []
        checked: list[str] = []

        coherence_violations = validate_ir_coherence(ir)
        violations.extend(coherence_violations)
        checked.extend([
            "confidence_claim_alignment",
            "confidence_uncertainty_alignment",
            "disclosure_consistency",
            "tone_confidence_plausibility",
            "citation_completeness",
        ])

        compliance_violations = validate_persona_compliance(ir, self.persona)
        violations.extend(compliance_violations)
        checked.extend([
            "knowledge_boundaries",
            "trait_plausibility",
            "invariant_compliance",
            "disclosure_constraints",
            "communication_preferences",
        ])

        errors = [v for v in violations if v.severity == "error"]

        if self.fail_on_warnings:
            passed = len(violations) == 0
        else:
            passed = len(errors) == 0

        return IRValidationResult(
            passed=passed,
            violations=violations,
            checked_invariants=checked,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def reset(self) -> None:
        """Reset cross-turn history for a new conversation."""
        self.cross_turn.reset()

    @property
    def turn_count(self) -> int:
        """Number of turns validated so far."""
        return len(self.cross_turn.history)

    def summary(self, result: IRValidationResult) -> str:
        """Human-readable summary of validation result."""
        status = "PASSED" if result.passed else "FAILED"
        errors = [v for v in result.violations if v.severity == "error"]
        warnings = [v for v in result.violations if v.severity == "warning"]

        lines = [f"Validation: {status}"]
        lines.append(f"Checked: {len(result.checked_invariants)} invariants")

        if errors:
            lines.append(f"ERRORS ({len(errors)}):")
            for v in errors:
                lines.append(f"  [{v.violation_type}] {v.message}")
                if v.suggested_fix:
                    lines.append(f"    Fix: {v.suggested_fix}")

        if warnings:
            lines.append(f"WARNINGS ({len(warnings)}):")
            for v in warnings:
                lines.append(f"  [{v.violation_type}] {v.message}")

        if not errors and not warnings:
            lines.append("No violations detected.")

        return "\n".join(lines)
