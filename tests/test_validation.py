"""
Validation Layer Tests — Phase 6.

Tests all three validation layers:
1. IR Coherence: Internal field consistency
2. Persona Compliance: IR vs persona profile
3. Cross-Turn Consistency: Multi-turn behavior coherence
4. Pipeline Validator: Orchestrator end-to-end
5. Real Pipeline: Validation on actual TurnPlanner output
"""

import pytest
import yaml

from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.schema.ir_schema import (
    Citation,
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    KnowledgeClaimType,
    ResponseStructure,
    SafetyPlan,
    Tone,
    UncertaintyAction,
    ValidationViolation,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager
from persona_engine.validation import (
    CrossTurnTracker,
    PipelineValidator,
    TurnSnapshot,
    validate_ir_coherence,
    validate_persona_compliance,
)


# =============================================================================
# Helpers
# =============================================================================


def load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def make_ir(
    confidence: float = 0.7,
    tone: Tone = Tone.THOUGHTFUL_ENGAGED,
    claim: KnowledgeClaimType = KnowledgeClaimType.PERSONAL_EXPERIENCE,
    uncertainty: UncertaintyAction = UncertaintyAction.ANSWER,
    disclosure: float = 0.5,
    formality: float = 0.4,
    directness: float = 0.5,
    verbosity: Verbosity = Verbosity.MEDIUM,
    stance: str | None = None,
    rationale: str | None = None,
    elasticity: float | None = 0.5,
    citations: list[Citation] | None = None,
    safety_plan: SafetyPlan | None = None,
) -> IntermediateRepresentation:
    """Build a minimal IR for testing."""
    return IntermediateRepresentation(
        conversation_frame=ConversationFrame(
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        ),
        response_structure=ResponseStructure(
            intent="Share perspective",
            stance=stance,
            rationale=rationale,
            elasticity=elasticity,
            confidence=confidence,
        ),
        communication_style=CommunicationStyle(
            tone=tone,
            verbosity=verbosity,
            formality=formality,
            directness=directness,
        ),
        knowledge_disclosure=KnowledgeAndDisclosure(
            disclosure_level=disclosure,
            uncertainty_action=uncertainty,
            knowledge_claim_type=claim,
        ),
        citations=citations if citations is not None else [
            Citation(source_type="base", source_id="base_profile", effect="Base formality", target_field="communication_style.formality"),
            Citation(source_type="base", source_id="base_profile", effect="Base directness", target_field="communication_style.directness"),
            Citation(source_type="base", source_id="base_profile", effect="Base confidence", target_field="response_structure.confidence"),
        ],
        safety_plan=safety_plan or SafetyPlan(),
    )


def make_context(
    user_input: str,
    topic: str = "general",
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    turn: int = 1,
    stance_cache: StanceCache | None = None,
) -> ConversationContext:
    return ConversationContext(
        conversation_id="test_validation",
        turn_number=turn,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=stance_cache or StanceCache(),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def persona() -> Persona:
    return load_persona()


@pytest.fixture
def validator(persona: Persona) -> PipelineValidator:
    return PipelineValidator(persona)


# =============================================================================
# 1. IR Coherence Validator
# =============================================================================


class TestIRCoherence:
    """Tests internal field consistency."""

    def test_coherent_ir_passes(self):
        """A well-formed IR should produce no violations."""
        ir = make_ir(confidence=0.7, claim=KnowledgeClaimType.PERSONAL_EXPERIENCE)
        violations = validate_ir_coherence(ir)
        assert len(violations) == 0

    def test_speculative_high_confidence(self):
        """Speculative claim with high confidence is flagged."""
        ir = make_ir(confidence=0.9, claim=KnowledgeClaimType.SPECULATIVE)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "confidence_claim_mismatch" in types

    def test_none_claim_high_confidence(self):
        """None claim with high confidence is flagged."""
        ir = make_ir(confidence=0.95, claim=KnowledgeClaimType.NONE)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "confidence_claim_mismatch" in types

    def test_expert_low_confidence(self):
        """Domain expert with very low confidence is flagged."""
        ir = make_ir(confidence=0.2, claim=KnowledgeClaimType.DOMAIN_EXPERT)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "expert_low_confidence" in types

    def test_high_confidence_hedge(self):
        """High confidence + hedge uncertainty is contradictory."""
        ir = make_ir(confidence=0.85, uncertainty=UncertaintyAction.HEDGE)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "confidence_uncertainty_contradiction" in types

    def test_low_confidence_answer(self):
        """Low confidence + direct answer is flagged."""
        ir = make_ir(confidence=0.2, uncertainty=UncertaintyAction.ANSWER)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "low_confidence_direct_answer" in types

    def test_refuse_high_disclosure(self):
        """Refusing to answer with high disclosure is contradictory."""
        ir = make_ir(uncertainty=UncertaintyAction.REFUSE, disclosure=0.8)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "refuse_high_disclosure" in types

    def test_negative_tone_high_confidence(self):
        """Negative tone with very high confidence is unusual."""
        ir = make_ir(confidence=0.95, tone=Tone.ANXIOUS_STRESSED)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "negative_tone_high_confidence" in types

    def test_rigid_uncertain(self):
        """Very rigid + very uncertain is contradictory."""
        ir = make_ir(confidence=0.1, elasticity=0.1)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "rigid_uncertain" in types

    def test_moderate_values_no_violations(self):
        """Moderate, balanced values should have no violations."""
        ir = make_ir(
            confidence=0.6,
            claim=KnowledgeClaimType.PERSONAL_EXPERIENCE,
            uncertainty=UncertaintyAction.ANSWER,
            disclosure=0.5,
            tone=Tone.THOUGHTFUL_ENGAGED,
            elasticity=0.5,
        )
        violations = validate_ir_coherence(ir)
        assert len(violations) == 0

    def test_citation_completeness_flagged(self):
        """Missing citations for key fields is flagged."""
        ir = make_ir(citations=[])  # No citations
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "incomplete_citations" in types

    def test_citation_completeness_passes_with_citations(self):
        """IR with proper citations passes completeness check."""
        citations = [
            Citation(
                source_type="base",
                source_id="base",
                target_field="communication_style.formality",
                operation="set",
                value_before=0.0,
                value_after=0.4,
                effect="Base formality",
                weight=1.0,
            ),
            Citation(
                source_type="base",
                source_id="base",
                target_field="communication_style.directness",
                operation="set",
                value_before=0.0,
                value_after=0.5,
                effect="Base directness",
                weight=1.0,
            ),
            Citation(
                source_type="base",
                source_id="base",
                target_field="response_structure.confidence",
                operation="set",
                value_before=0.0,
                value_after=0.7,
                effect="Base confidence",
                weight=1.0,
            ),
        ]
        ir = make_ir(citations=citations)
        violations = validate_ir_coherence(ir)
        types = [v.violation_type for v in violations]
        assert "incomplete_citations" not in types

    def test_all_violations_have_required_fields(self):
        """All violations have violation_type, severity, and message."""
        ir = make_ir(confidence=0.95, claim=KnowledgeClaimType.SPECULATIVE)
        violations = validate_ir_coherence(ir)
        for v in violations:
            assert v.violation_type
            assert v.severity in ("error", "warning")
            assert v.message


# =============================================================================
# 2. Persona Compliance Validator
# =============================================================================


class TestPersonaCompliance:
    """Tests IR against persona profile."""

    def test_compliant_ir_passes(self, persona: Persona):
        """IR that respects persona boundaries passes."""
        ir = make_ir(
            confidence=0.7,
            claim=KnowledgeClaimType.PERSONAL_EXPERIENCE,
            directness=0.5,
            formality=0.4,
        )
        violations = validate_persona_compliance(ir, persona)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_forbidden_claim_in_stance(self, persona: Persona):
        """Stance mentioning cannot_claim triggers error."""
        # Get a forbidden claim from persona
        if persona.invariants and persona.invariants.cannot_claim:
            forbidden = persona.invariants.cannot_claim[0]
            ir = make_ir(
                stance=f"As a {forbidden}, I believe this is correct",
                rationale="Based on my expertise",
            )
            violations = validate_persona_compliance(ir, persona)
            types = [v.violation_type for v in violations]
            assert "invariant_contradiction" in types

    def test_must_avoid_topic_leak(self, persona: Persona):
        """Must-avoid topic in stance without safety plan is an error."""
        if persona.invariants and persona.invariants.must_avoid:
            avoided = persona.invariants.must_avoid[0]
            ir = make_ir(
                stance=f"When I was at {avoided}, we did great work",
            )
            violations = validate_persona_compliance(ir, persona)
            types = [v.violation_type for v in violations]
            assert "must_avoid_leak" in types

    def test_formality_deviation_flagged(self, persona: Persona):
        """Large formality deviation from base is flagged."""
        base = persona.psychology.communication.formality
        extreme = 1.0 if base < 0.5 else 0.0  # opposite extreme
        ir = make_ir(formality=extreme)
        violations = validate_persona_compliance(ir, persona)
        types = [v.violation_type for v in violations]
        assert "formality_deviation" in types

    def test_formality_within_range_passes(self, persona: Persona):
        """Formality within plausible range passes."""
        base = persona.psychology.communication.formality
        ir = make_ir(formality=base)  # Exactly at base
        violations = validate_persona_compliance(ir, persona)
        formality_violations = [v for v in violations if v.violation_type == "formality_deviation"]
        assert len(formality_violations) == 0

    def test_disclosure_exceeds_policy(self, persona: Persona):
        """Disclosure way beyond base_openness is flagged."""
        ir = make_ir(disclosure=0.99)
        violations = validate_persona_compliance(ir, persona)
        types = [v.violation_type for v in violations]
        # May or may not be flagged depending on base_openness
        # If base_openness is low, 0.99 should be flagged
        if persona.disclosure_policy.base_openness < 0.6:
            assert "disclosure_exceeds_policy" in types


# =============================================================================
# 3. Cross-Turn Consistency Validator
# =============================================================================


class TestCrossTurnConsistency:
    """Tests multi-turn behavior coherence."""

    def test_first_turn_no_violations(self):
        """First turn has no cross-turn violations."""
        tracker = CrossTurnTracker()
        ir = make_ir(confidence=0.7)
        violations = tracker.validate_turn(ir, turn_number=1)
        assert len(violations) == 0

    def test_stable_parameters_no_violations(self):
        """Stable parameters across turns produce no violations."""
        tracker = CrossTurnTracker()
        for turn in range(1, 4):
            ir = make_ir(confidence=0.7, formality=0.4, directness=0.5)
            violations = tracker.validate_turn(ir, turn_number=turn)
            assert len(violations) == 0

    def test_confidence_swing_flagged(self):
        """Large confidence swing between turns is flagged."""
        tracker = CrossTurnTracker()
        ir1 = make_ir(confidence=0.9)
        tracker.validate_turn(ir1, turn_number=1)

        ir2 = make_ir(confidence=0.2)
        violations = tracker.validate_turn(ir2, turn_number=2)
        types = [v.violation_type for v in violations]
        assert "parameter_swing" in types

    def test_formality_swing_flagged(self):
        """Large formality swing between turns is flagged."""
        tracker = CrossTurnTracker()
        ir1 = make_ir(formality=0.1)
        tracker.validate_turn(ir1, turn_number=1)

        ir2 = make_ir(formality=0.9)
        violations = tracker.validate_turn(ir2, turn_number=2)
        types = [v.violation_type for v in violations]
        assert "parameter_swing" in types

    def test_directness_swing_flagged(self):
        """Large directness swing between turns is flagged."""
        tracker = CrossTurnTracker()
        ir1 = make_ir(directness=0.1)
        tracker.validate_turn(ir1, turn_number=1)

        ir2 = make_ir(directness=0.9)
        violations = tracker.validate_turn(ir2, turn_number=2)
        types = [v.violation_type for v in violations]
        assert "parameter_swing" in types

    def test_expertise_inconsistency_flagged(self):
        """Expert then non-expert on same topic is flagged."""
        tracker = CrossTurnTracker()
        ir1 = make_ir(claim=KnowledgeClaimType.DOMAIN_EXPERT)
        tracker.validate_turn(ir1, turn_number=1, topic="ux_research")

        ir2 = make_ir(claim=KnowledgeClaimType.SPECULATIVE)
        violations = tracker.validate_turn(ir2, turn_number=2, topic="ux_research")
        types = [v.violation_type for v in violations]
        assert "expertise_inconsistency" in types

    def test_expertise_different_topics_ok(self):
        """Expert on one topic, non-expert on another is fine."""
        tracker = CrossTurnTracker()
        ir1 = make_ir(claim=KnowledgeClaimType.DOMAIN_EXPERT)
        tracker.validate_turn(ir1, turn_number=1, topic="ux_research")

        ir2 = make_ir(claim=KnowledgeClaimType.SPECULATIVE)
        violations = tracker.validate_turn(ir2, turn_number=2, topic="quantum_physics")
        types = [v.violation_type for v in violations]
        assert "expertise_inconsistency" not in types

    def test_stance_reversal_flagged(self):
        """Stance reversal on same topic is flagged."""
        tracker = CrossTurnTracker()
        ir1 = make_ir(stance="I support remote work flexibility")
        tracker.validate_turn(ir1, turn_number=1, topic="remote_work")

        ir2 = make_ir(stance="I am against remote work flexibility")
        violations = tracker.validate_turn(ir2, turn_number=2, topic="remote_work")
        types = [v.violation_type for v in violations]
        assert "stance_reversal" in types

    def test_stance_evolution_different_topics_ok(self):
        """Different stances on different topics is fine."""
        tracker = CrossTurnTracker()
        ir1 = make_ir(stance="I support this approach")
        tracker.validate_turn(ir1, turn_number=1, topic="topic_a")

        ir2 = make_ir(stance="I disagree with this idea")
        violations = tracker.validate_turn(ir2, turn_number=2, topic="topic_b")
        types = [v.violation_type for v in violations]
        assert "stance_reversal" not in types

    def test_tracker_history_accumulates(self):
        """Tracker maintains turn history."""
        tracker = CrossTurnTracker()
        for turn in range(1, 5):
            ir = make_ir(confidence=0.7)
            tracker.validate_turn(ir, turn_number=turn)
        assert len(tracker.history) == 4

    def test_tracker_reset_clears_history(self):
        """Reset clears all history."""
        tracker = CrossTurnTracker()
        ir = make_ir()
        tracker.validate_turn(ir, turn_number=1)
        assert len(tracker.history) == 1
        tracker.reset()
        assert len(tracker.history) == 0

    def test_turn_snapshot_from_ir(self):
        """TurnSnapshot correctly captures IR values."""
        ir = make_ir(
            confidence=0.8,
            formality=0.3,
            directness=0.6,
            disclosure=0.5,
            tone=Tone.WARM_CONFIDENT,
            claim=KnowledgeClaimType.DOMAIN_EXPERT,
            stance="I support this",
        )
        snap = TurnSnapshot.from_ir(ir, turn=3, topic="test")
        assert snap.confidence == 0.8
        assert snap.formality == 0.3
        assert snap.directness == 0.6
        assert snap.disclosure == 0.5
        assert snap.tone == "warm_confident"
        assert snap.claim_type == "domain_expert"
        assert snap.stance == "I support this"
        assert snap.turn_number == 3
        assert snap.topic == "test"


# =============================================================================
# 4. Pipeline Validator (Orchestrator)
# =============================================================================


class TestPipelineValidator:
    """Tests the orchestrator that combines all validation layers."""

    def test_clean_ir_passes(self, validator: PipelineValidator):
        """A well-formed IR passes all checks."""
        ir = make_ir(
            confidence=0.7,
            claim=KnowledgeClaimType.PERSONAL_EXPERIENCE,
            formality=0.4,
            directness=0.5,
        )
        result = validator.validate(ir, turn_number=1)
        errors = [v for v in result.violations if v.severity == "error"]
        assert len(errors) == 0

    def test_result_has_checked_invariants(self, validator: PipelineValidator):
        """Result lists all checked invariants."""
        ir = make_ir()
        result = validator.validate(ir, turn_number=1)
        assert len(result.checked_invariants) > 0
        assert "knowledge_boundaries" in result.checked_invariants
        assert "parameter_swing" in result.checked_invariants

    def test_result_has_timestamp(self, validator: PipelineValidator):
        """Result includes a timestamp."""
        ir = make_ir()
        result = validator.validate(ir, turn_number=1)
        assert result.timestamp is not None

    def test_errors_cause_failure(self, validator: PipelineValidator, persona: Persona):
        """Errors cause the validation to fail."""
        if persona.invariants and persona.invariants.cannot_claim:
            forbidden = persona.invariants.cannot_claim[0]
            ir = make_ir(stance=f"As a {forbidden}, I know this")
            result = validator.validate(ir, turn_number=1)
            assert not result.passed

    def test_warnings_dont_fail_by_default(self, validator: PipelineValidator):
        """Warnings don't cause failure by default."""
        ir = make_ir(confidence=0.9, claim=KnowledgeClaimType.SPECULATIVE)
        result = validator.validate(ir, turn_number=1)
        # Has warnings but should still pass (no errors)
        warnings = [v for v in result.violations if v.severity == "warning"]
        assert len(warnings) > 0
        assert result.passed

    def test_fail_on_warnings_mode(self, persona: Persona):
        """With fail_on_warnings=True, warnings also cause failure."""
        strict = PipelineValidator(persona, fail_on_warnings=True)
        ir = make_ir(confidence=0.9, claim=KnowledgeClaimType.SPECULATIVE)
        result = strict.validate(ir, turn_number=1)
        warnings = [v for v in result.violations if v.severity == "warning"]
        if warnings:
            assert not result.passed

    def test_validate_single_no_cross_turn(self, validator: PipelineValidator):
        """validate_single() works without cross-turn history."""
        ir = make_ir(confidence=0.7)
        result = validator.validate_single(ir)
        assert result is not None
        assert len(result.checked_invariants) > 0
        # Should not include cross-turn checks
        assert "parameter_swing" not in result.checked_invariants

    def test_multi_turn_validation(self, validator: PipelineValidator):
        """Multi-turn validation accumulates history."""
        for turn in range(1, 4):
            ir = make_ir(confidence=0.7)
            validator.validate(ir, turn_number=turn)
        assert validator.turn_count == 3

    def test_reset_clears_state(self, validator: PipelineValidator):
        """Reset clears cross-turn history."""
        ir = make_ir()
        validator.validate(ir, turn_number=1)
        assert validator.turn_count == 1
        validator.reset()
        assert validator.turn_count == 0

    def test_summary_format(self, validator: PipelineValidator):
        """Summary produces readable output."""
        ir = make_ir(confidence=0.9, claim=KnowledgeClaimType.SPECULATIVE)
        result = validator.validate(ir, turn_number=1)
        summary = validator.summary(result)
        assert "Validation:" in summary
        assert "Checked:" in summary

    def test_summary_clean_ir(self, validator: PipelineValidator):
        """Clean IR summary says no violations."""
        ir = make_ir(confidence=0.6, claim=KnowledgeClaimType.PERSONAL_EXPERIENCE, formality=0.4)
        result = validator.validate(ir, turn_number=1)
        if result.passed and not result.violations:
            summary = validator.summary(result)
            assert "No violations" in summary


# =============================================================================
# 5. Real Pipeline Validation
# =============================================================================


class TestRealPipelineValidation:
    """Run validation on actual TurnPlanner output."""

    def test_expert_domain_passes_validation(self, persona: Persona):
        """TurnPlanner output for expert domain passes validation."""
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        validator = PipelineValidator(persona)

        ctx = make_context(
            "Tell me about usability testing methods",
            topic="ux_research",
            turn=1,
        )
        ir = planner.generate_ir(ctx)
        result = validator.validate(ir, turn_number=1, topic="ux_research")

        errors = [v for v in result.violations if v.severity == "error"]
        assert len(errors) == 0, f"Errors: {[e.message for e in errors]}"

    def test_non_expert_domain_passes_validation(self, persona: Persona):
        """TurnPlanner output for non-expert domain passes validation."""
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        validator = PipelineValidator(persona)

        ctx = make_context(
            "Can you explain quantum computing?",
            topic="quantum_physics",
            turn=1,
        )
        ir = planner.generate_ir(ctx)
        result = validator.validate(ir, turn_number=1, topic="quantum_physics")

        errors = [v for v in result.violations if v.severity == "error"]
        assert len(errors) == 0, f"Errors: {[e.message for e in errors]}"

    def test_multi_turn_pipeline_passes(self, persona: Persona):
        """Multi-turn real pipeline passes cross-turn validation."""
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        validator = PipelineValidator(persona)
        cache = StanceCache()

        turns = [
            ("What's your experience with UX?", "ux_research"),
            ("Tell me more about that", "ux_research"),
            ("What about quantum physics?", "quantum_physics"),
            ("Going back to UX, any tips?", "ux_research"),
        ]

        all_errors: list[ValidationViolation] = []
        for turn_num, (user_input, topic) in enumerate(turns, 1):
            ctx = make_context(
                user_input, topic=topic, turn=turn_num, stance_cache=cache
            )
            ir = planner.generate_ir(ctx)
            result = validator.validate(ir, turn_number=turn_num, topic=topic)
            errors = [v for v in result.violations if v.severity == "error"]
            all_errors.extend(errors)

        assert len(all_errors) == 0, f"Errors: {[e.message for e in all_errors]}"

    def test_expert_domain_has_high_confidence(self, persona: Persona):
        """Validation confirms expert domain produces high confidence."""
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        ctx = make_context(
            "Tell me about user interview techniques",
            topic="ux_research",
            turn=1,
        )
        ir = planner.generate_ir(ctx)
        assert ir.response_structure.confidence > 0.5
        assert ir.knowledge_disclosure.knowledge_claim_type == KnowledgeClaimType.DOMAIN_EXPERT

    def test_non_expert_domain_hedges(self, persona: Persona):
        """Validation confirms non-expert domain produces hedging."""
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        ctx = make_context(
            "Explain quantum entanglement",
            topic="quantum_physics",
            turn=1,
        )
        ir = planner.generate_ir(ctx)
        assert ir.response_structure.confidence < 0.5
        assert ir.knowledge_disclosure.knowledge_claim_type != KnowledgeClaimType.DOMAIN_EXPERT

    def test_five_turn_conversation_zero_errors(self, persona: Persona):
        """A realistic 5-turn conversation produces zero validation errors."""
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        validator = PipelineValidator(persona)
        cache = StanceCache()

        turns = [
            ("Hi, what do you do?", "introduction"),
            ("Tell me about UX research", "ux_research"),
            ("How do you handle stakeholder pushback?", "stakeholder_management"),
            ("What about AI in research?", "ai_research"),
            ("Thanks, that was helpful!", "conclusion"),
        ]

        total_warnings = 0
        total_errors = 0
        for turn_num, (user_input, topic) in enumerate(turns, 1):
            ctx = make_context(
                user_input, topic=topic, turn=turn_num, stance_cache=cache
            )
            ir = planner.generate_ir(ctx)
            result = validator.validate(ir, turn_number=turn_num, topic=topic)
            total_errors += len([v for v in result.violations if v.severity == "error"])
            total_warnings += len([v for v in result.violations if v.severity == "warning"])

        assert total_errors == 0, "Real pipeline should produce zero errors"
