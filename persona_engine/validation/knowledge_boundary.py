"""
Knowledge Boundary Enforcer — Validates that personas respect their expertise limits.

Tracks claims made per domain per turn and flags when a non-expert persona
makes expert-level claims (high confidence + domain-specific + low proficiency).

Called after IR generation, before response rendering.
"""

from dataclasses import dataclass, field
from typing import Any

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    KnowledgeClaimType,
    UncertaintyAction,
)


@dataclass
class BoundaryViolation:
    """A knowledge boundary violation."""
    turn_number: int
    domain: str
    proficiency: float
    confidence: float
    claim_type: str
    message: str


@dataclass
class BoundaryReport:
    """Report from the knowledge boundary enforcer."""
    violations: list[BoundaryViolation] = field(default_factory=list)
    claims_by_domain: dict[str, int] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0


class KnowledgeBoundaryEnforcer:
    """
    Track and enforce knowledge boundary constraints across turns.

    Usage:
        enforcer = KnowledgeBoundaryEnforcer(domain_proficiencies)
        enforcer.check_turn(ir, turn_number)
        report = enforcer.get_report()
    """

    def __init__(
        self,
        domain_proficiencies: dict[str, float],
        expert_threshold: float = 0.7,
    ):
        """
        Args:
            domain_proficiencies: Mapping of domain name -> proficiency (0-1)
            expert_threshold: Minimum proficiency to claim expert status
        """
        self.domain_proficiencies = domain_proficiencies
        self.expert_threshold = expert_threshold
        self._violations: list[BoundaryViolation] = []
        self._claims_by_domain: dict[str, int] = {}

    def check_turn(
        self,
        ir: IntermediateRepresentation,
        turn_number: int,
        detected_domain: str = "unknown",
    ) -> list[BoundaryViolation]:
        """
        Check a single turn's IR for knowledge boundary violations.

        Args:
            ir: The IntermediateRepresentation to check
            turn_number: Current turn number
            detected_domain: The domain detected for this turn

        Returns:
            List of violations found in this turn (also stored internally)
        """
        turn_violations = []

        claim_type = ir.knowledge_disclosure.knowledge_claim_type
        confidence = ir.response_structure.confidence
        competence = ir.response_structure.competence
        action = ir.knowledge_disclosure.uncertainty_action

        # Track claims per domain
        self._claims_by_domain[detected_domain] = self._claims_by_domain.get(detected_domain, 0) + 1

        proficiency = self.domain_proficiencies.get(detected_domain, 0.0)

        # Check: non-expert making expert claims
        if (
            claim_type == KnowledgeClaimType.DOMAIN_EXPERT
            and proficiency < self.expert_threshold
        ):
            v = BoundaryViolation(
                turn_number=turn_number,
                domain=detected_domain,
                proficiency=proficiency,
                confidence=confidence,
                claim_type=claim_type.value,
                message=(
                    f"Expert claim in '{detected_domain}' with proficiency "
                    f"{proficiency:.2f} (threshold: {self.expert_threshold})"
                ),
            )
            turn_violations.append(v)

        # Check: high confidence + ANSWER in non-expert domain
        if (
            proficiency < self.expert_threshold
            and confidence > 0.8
            and action == UncertaintyAction.ANSWER
            and claim_type != KnowledgeClaimType.PERSONAL_EXPERIENCE
            and claim_type != KnowledgeClaimType.COMMON_KNOWLEDGE
        ):
            v = BoundaryViolation(
                turn_number=turn_number,
                domain=detected_domain,
                proficiency=proficiency,
                confidence=confidence,
                claim_type=claim_type.value,
                message=(
                    f"High confidence answer ({confidence:.2f}) in non-expert "
                    f"domain '{detected_domain}' (proficiency: {proficiency:.2f})"
                ),
            )
            turn_violations.append(v)

        self._violations.extend(turn_violations)
        return turn_violations

    def get_report(self) -> BoundaryReport:
        """Get the cumulative boundary enforcement report."""
        return BoundaryReport(
            violations=list(self._violations),
            claims_by_domain=dict(self._claims_by_domain),
        )
