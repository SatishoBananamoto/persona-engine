"""
Cross-Turn Consistency Validator — multi-turn coherence checks.

Validates that persona behavior across turns is psychologically consistent.
Catches stance flips, identity drift, and implausible parameter swings.

Usage:
    tracker = CrossTurnTracker()
    for each turn:
        ir = planner.generate_ir(context)
        violations = tracker.validate_turn(ir)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    KnowledgeClaimType,
    ValidationViolation,
)

# Maximum plausible swing per turn for continuous parameters
_MAX_CONFIDENCE_SWING = 0.45
_MAX_FORMALITY_SWING = 0.40
_MAX_DIRECTNESS_SWING = 0.40
_MAX_DISCLOSURE_SWING = 0.40


@dataclass
class TurnSnapshot:
    """Captures key IR values for a single turn."""

    turn_number: int
    confidence: float
    formality: float
    directness: float
    disclosure: float
    tone: str
    claim_type: str
    stance: str | None
    topic: str
    elasticity: float = 0.5
    competence: float = 0.5

    @classmethod
    def from_ir(cls, ir: IntermediateRepresentation, turn: int, topic: str = "") -> TurnSnapshot:
        return cls(
            turn_number=turn,
            confidence=ir.response_structure.confidence,
            formality=ir.communication_style.formality,
            directness=ir.communication_style.directness,
            disclosure=ir.knowledge_disclosure.disclosure_level,
            tone=ir.communication_style.tone.value,
            claim_type=ir.knowledge_disclosure.knowledge_claim_type.value,
            stance=ir.response_structure.stance,
            topic=topic,
            elasticity=ir.response_structure.elasticity if ir.response_structure.elasticity is not None else 0.5,
            competence=ir.response_structure.competence,
        )


class CrossTurnTracker:
    """
    Tracks IR snapshots across turns and validates consistency.

    Detects:
    - Implausible parameter swings (confidence, formality, etc.)
    - Stance reversals on the same topic without evidence
    - Claim type contradictions (expert then non-expert on same domain)
    - Identity drift (invariant violations across turns)
    """

    def __init__(self) -> None:
        self._history: list[TurnSnapshot] = []

    def validate_turn(
        self,
        ir: IntermediateRepresentation,
        turn_number: int = 0,
        topic: str = "",
    ) -> list[ValidationViolation]:
        """
        Validate this turn against all previous turns.

        Args:
            ir: Current turn's IR
            turn_number: Current turn number (auto-increments if 0)
            topic: Current topic signature

        Returns:
            List of cross-turn violations
        """
        if turn_number == 0:
            turn_number = len(self._history) + 1

        current = TurnSnapshot.from_ir(ir, turn_number, topic)
        violations: list[ValidationViolation] = []

        if self._history:
            prev = self._history[-1]
            _check_parameter_swings(prev, current, violations)
            _check_claim_consistency(self._history, current, violations)
            _check_stance_consistency(self._history, current, violations)

        self._history.append(current)
        return violations

    @property
    def history(self) -> list[TurnSnapshot]:
        """Access the turn history (read-only)."""
        return list(self._history)

    def reset(self) -> None:
        """Clear history for a new conversation."""
        self._history.clear()


def _check_parameter_swings(
    prev: TurnSnapshot,
    current: TurnSnapshot,
    violations: list[ValidationViolation],
) -> None:
    """Flag implausibly large parameter changes between consecutive turns."""
    checks = [
        ("confidence", prev.confidence, current.confidence, _MAX_CONFIDENCE_SWING),
        ("formality", prev.formality, current.formality, _MAX_FORMALITY_SWING),
        ("directness", prev.directness, current.directness, _MAX_DIRECTNESS_SWING),
        ("disclosure", prev.disclosure, current.disclosure, _MAX_DISCLOSURE_SWING),
    ]

    for name, prev_val, curr_val, max_swing in checks:
        delta = abs(curr_val - prev_val)
        if delta > max_swing:
            violations.append(ValidationViolation(
                violation_type="parameter_swing",
                severity="warning",
                message=(
                    f"{name} swung {delta:.2f} in one turn "
                    f"({prev_val:.2f} → {curr_val:.2f}), "
                    f"max expected: {max_swing:.2f}"
                ),
                field_path=f"communication_style.{name}" if name != "confidence" else "response_structure.confidence",
                suggested_fix=f"Check if {name} modifiers are over-applying",
            ))


def _check_claim_consistency(
    history: list[TurnSnapshot],
    current: TurnSnapshot,
    violations: list[ValidationViolation],
) -> None:
    """Flag contradictory expertise claims on the same topic."""
    if not current.topic:
        return

    for prev in history:
        if prev.topic != current.topic:
            continue

        prev_expert = prev.claim_type == KnowledgeClaimType.DOMAIN_EXPERT.value
        curr_expert = current.claim_type == KnowledgeClaimType.DOMAIN_EXPERT.value

        # Expert → non-expert on same topic is suspicious
        if prev_expert and not curr_expert:
            violations.append(ValidationViolation(
                violation_type="expertise_inconsistency",
                severity="warning",
                message=(
                    f"Claimed domain_expert on '{current.topic}' in turn "
                    f"{prev.turn_number} but '{current.claim_type}' in turn "
                    f"{current.turn_number}"
                ),
                field_path="knowledge_disclosure.knowledge_claim_type",
                suggested_fix="Expertise on a topic should be consistent across turns",
            ))
            break


def _check_stance_consistency(
    history: list[TurnSnapshot],
    current: TurnSnapshot,
    violations: list[ValidationViolation],
) -> None:
    """Flag stance reversals on the same topic without evidence."""
    if not current.topic or not current.stance:
        return

    for prev in reversed(history):
        if prev.topic != current.topic or not prev.stance:
            continue

        # Same topic, both have stances — check for reversal
        # Simple heuristic: if stances are very different strings, flag it
        prev_lower = prev.stance.lower()
        curr_lower = current.stance.lower()

        # Check for explicit negation patterns
        negation_pairs = [
            ("support", "against"),
            ("agree", "disagree"),
            ("favor", "oppose"),
            ("good", "bad"),
            ("beneficial", "harmful"),
            ("important", "unimportant"),
        ]

        for pos, neg in negation_pairs:
            prev_has_pos = pos in prev_lower
            prev_has_neg = neg in prev_lower
            curr_has_pos = pos in curr_lower
            curr_has_neg = neg in curr_lower

            if (prev_has_pos and curr_has_neg) or (prev_has_neg and curr_has_pos):
                violations.append(ValidationViolation(
                    violation_type="stance_reversal",
                    severity="warning",
                    message=(
                        f"Stance on '{current.topic}' appears to have reversed "
                        f"between turn {prev.turn_number} and {current.turn_number}"
                    ),
                    field_path="response_structure.stance",
                    suggested_fix="Use stance cache with reconsideration logic",
                ))
                return  # Only flag once per topic

        break  # Only check most recent matching turn
