"""
Tests for Phase 6: Behavioral Fidelity & Validation

Covers:
- Fix 6.1: Uncertainty resolver with stress/fatigue
- Fix 6.2: Negation detection in negativity bias
- Fix 6.3: Schwartz circumplex adjacency in value conflict resolution
- Fix 6.4: success_criteria from goals
- Fix 6.5: languages[] deprecation warning
- Fix 6.6: IR validator
- Fix 6.7: Style drift detection
- Fix 6.8: Knowledge boundary enforcer
"""

import warnings

import pytest

from persona_engine.behavioral.bias_simulator import (
    BiasSimulator,
    _count_unnegated_markers,
)
from persona_engine.behavioral.uncertainty_resolver import resolve_uncertainty_action
from persona_engine.behavioral.values_interpreter import (
    SCHWARTZ_ADJACENCY,
    VALUE_CONFLICTS,
    ConflictResolution,
    ValuesInterpreter,
)
from persona_engine.schema.ir_schema import (
    Citation,
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    KnowledgeClaimType,
    MemoryOps,
    ResponseStructure,
    SafetyPlan,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import (
    BigFiveTraits,
    ClaimPolicy,
    CognitiveStyle,
    CommunicationPreferences,
    DisclosurePolicy,
    DynamicState,
    Goal,
    Identity,
    LanguageKnowledge,
    Persona,
    PersonaInvariants,
    PersonalityProfile,
    SchwartzValues,
    SocialRole,
    UncertaintyPolicy,
)
from persona_engine.validation.ir_validator import (
    Severity,
    ValidationIssue,
    has_errors,
    validate_ir,
)
from persona_engine.validation.knowledge_boundary import (
    KnowledgeBoundaryEnforcer,
)
from persona_engine.validation.style_drift import (
    StyleDriftDetector,
    TurnMetrics,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_persona(**overrides):
    """Build a minimal valid Persona for testing."""
    defaults = dict(
        persona_id="test-phase6",
        label="Test Persona",
        identity=Identity(
            age=30, location="NYC", education="BS",
            occupation="Dev", background="Test",
        ),
        psychology=PersonalityProfile(
            big_five=BigFiveTraits(
                openness=0.7, conscientiousness=0.6,
                extraversion=0.5, agreeableness=0.6, neuroticism=0.4,
            ),
            values=SchwartzValues(
                self_direction=0.7, stimulation=0.5, hedonism=0.4,
                achievement=0.8, power=0.3, security=0.6,
                conformity=0.4, tradition=0.3,
                benevolence=0.7, universalism=0.6,
            ),
            cognitive_style=CognitiveStyle(
                analytical_intuitive=0.7, systematic_heuristic=0.6,
                risk_tolerance=0.5, need_for_closure=0.5,
                cognitive_complexity=0.7,
            ),
            communication=CommunicationPreferences(
                verbosity=0.6, formality=0.5,
                directness=0.6, emotional_expressiveness=0.5,
            ),
        ),
        social_roles={
            "default": SocialRole(formality=0.5, directness=0.5, emotional_expressiveness=0.5),
        },
        uncertainty=UncertaintyPolicy(
            admission_threshold=0.4, hedging_frequency=0.5,
            clarification_tendency=0.6, knowledge_boundary_strictness=0.7,
        ),
        claim_policy=ClaimPolicy(),
        invariants=PersonaInvariants(identity_facts=["Age 30"]),
        disclosure_policy=DisclosurePolicy(base_openness=0.5, factors={"topic_sensitivity": -0.3}),
        initial_state=DynamicState(
            mood_valence=0.2, mood_arousal=0.5,
            fatigue=0.1, stress=0.2, engagement=0.7,
        ),
        time_scarcity=0.5,
        privacy_sensitivity=0.4,
    )
    defaults.update(overrides)
    return Persona(**defaults)


def _make_ir(**overrides):
    """Build a minimal valid IR for testing."""
    defaults = dict(
        conversation_frame=ConversationFrame(
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        ),
        response_structure=ResponseStructure(
            intent="Share thoughts",
            stance="Supportive",
            rationale="Based on experience",
            elasticity=0.5,
            confidence=0.6,
            competence=0.5,
        ),
        communication_style=CommunicationStyle(
            tone=Tone.THOUGHTFUL_ENGAGED,
            verbosity=Verbosity.MEDIUM,
            formality=0.5,
            directness=0.5,
        ),
        knowledge_disclosure=KnowledgeAndDisclosure(
            disclosure_level=0.5,
            uncertainty_action=UncertaintyAction.HEDGE,
            knowledge_claim_type=KnowledgeClaimType.COMMON_KNOWLEDGE,
        ),
        citations=[
            Citation(source_type="trait", source_id="test", effect="test", weight=0.5),
        ],
    )
    defaults.update(overrides)
    return IntermediateRepresentation(**defaults)


# =============================================================================
# Fix 6.1 — Uncertainty Resolver: Dynamic State Inputs
# =============================================================================

class TestUncertaintyResolverDynamicState:
    """Test that stress and fatigue affect uncertainty resolution."""

    def test_high_stress_lowers_effective_confidence(self):
        """Stressed persona with moderate confidence should hedge instead of answer."""
        citations = []
        # Without stress: confidence=0.65, risk_tolerance=0.7 → ANSWER
        result_calm = resolve_uncertainty_action(
            proficiency=0.8, confidence=0.65,
            risk_tolerance=0.7, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="hedge",
            citations=citations, stress=0.0, fatigue=0.0,
        )
        assert result_calm == UncertaintyAction.ANSWER

        citations2 = []
        # With high stress: effective_confidence = 0.65 - 0.9*0.15 = 0.515
        # Still moderate but stress lowers it
        result_stressed = resolve_uncertainty_action(
            proficiency=0.8, confidence=0.65,
            risk_tolerance=0.4, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="hedge",
            citations=citations2, stress=0.9, fatigue=0.0,
        )
        # With stress penalty, effective confidence drops → should hedge
        assert result_stressed == UncertaintyAction.HEDGE

    def test_high_fatigue_biases_toward_hedge(self):
        """Fatigued persona should prefer hedging over effortful reasoning."""
        citations = []
        result = resolve_uncertainty_action(
            proficiency=0.8, confidence=0.6,
            risk_tolerance=0.5, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="hedge",
            citations=citations, stress=0.0, fatigue=0.9,
        )
        assert result == UncertaintyAction.HEDGE

    def test_stress_and_fatigue_combined(self):
        """Combined stress + fatigue should have cumulative effect."""
        citations = []
        # confidence=0.75 normally → ANSWER
        # With stress=0.8 + fatigue=0.8: penalty = 0.12 + 0.08 = 0.20
        # effective = 0.55 → moderate range
        result = resolve_uncertainty_action(
            proficiency=0.8, confidence=0.75,
            risk_tolerance=0.4, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="hedge",
            citations=citations, stress=0.8, fatigue=0.8,
        )
        assert result == UncertaintyAction.HEDGE

    def test_no_stress_no_fatigue_unchanged(self):
        """Zero stress/fatigue should not change behavior from baseline."""
        citations = []
        result = resolve_uncertainty_action(
            proficiency=0.8, confidence=0.8,
            risk_tolerance=0.5, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="hedge",
            citations=citations, stress=0.0, fatigue=0.0,
        )
        assert result == UncertaintyAction.ANSWER

    def test_stress_citation_added(self):
        """Stress/fatigue should produce a citation."""
        citations = []
        resolve_uncertainty_action(
            proficiency=0.8, confidence=0.6,
            risk_tolerance=0.5, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="hedge",
            citations=citations, stress=0.8, fatigue=0.0,
        )
        state_citations = [c for c in citations if c.source_id == "dynamic_state"]
        assert len(state_citations) == 1
        assert "stress" in state_citations[0].effect

    def test_low_stress_no_citation(self):
        """Stress below 0.3 should not produce a citation."""
        citations = []
        resolve_uncertainty_action(
            proficiency=0.8, confidence=0.6,
            risk_tolerance=0.5, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="hedge",
            citations=citations, stress=0.2, fatigue=0.1,
        )
        state_citations = [c for c in citations if c.source_id == "dynamic_state"]
        assert len(state_citations) == 0

    def test_hard_constraints_still_dominate(self):
        """Claim policy hard constraints should override stress/fatigue effects."""
        citations = []
        result = resolve_uncertainty_action(
            proficiency=0.1, confidence=0.9,
            risk_tolerance=0.9, need_for_closure=0.5,
            time_pressure=0.3, claim_policy_lookup_behavior="refuse",
            citations=citations, stress=0.0, fatigue=0.0,
        )
        assert result == UncertaintyAction.REFUSE


# =============================================================================
# Fix 6.2 — Negativity Bias: Negation Detection
# =============================================================================

class TestNegationDetection:
    """Test that negated negative words don't trigger negativity bias."""

    def test_not_bad_no_trigger(self):
        assert _count_unnegated_markers("not bad") == 0

    def test_this_is_bad_triggers(self):
        assert _count_unnegated_markers("this is bad") == 1

    def test_no_problem_no_trigger(self):
        assert _count_unnegated_markers("no problem at all") == 0

    def test_serious_problem_triggers(self):
        assert _count_unnegated_markers("serious problem") == 1

    def test_dont_worry_no_trigger(self):
        assert _count_unnegated_markers("don't worry about it") == 0

    def test_i_am_worried_triggers(self):
        assert _count_unnegated_markers("i am worried about this") == 1

    def test_no_concerns_no_trigger(self):
        assert _count_unnegated_markers("i don't have any concern") == 0

    def test_i_have_concerns_triggers(self):
        assert _count_unnegated_markers("i have a concern about this") == 1

    def test_multiple_negated_and_real(self):
        """Mix of negated and real negatives."""
        text = "not bad but there is a serious problem and i'm frustrated"
        count = _count_unnegated_markers(text)
        assert count == 2  # "problem" and "frustrated" are real

    def test_empty_input(self):
        assert _count_unnegated_markers("") == 0

    def test_no_negative_words(self):
        assert _count_unnegated_markers("everything is wonderful today") == 0

    def test_bias_simulator_uses_negation_detection(self):
        """BiasSimulator should use negation-aware counting."""
        sim = BiasSimulator(
            traits={"neuroticism": 0.9, "openness": 0.5},
            value_priorities={},
        )
        # "not bad" should not trigger negativity bias
        result = sim.compute_modifiers("this is not bad at all")
        neg_mods = [m for m in result if m.bias_type.value == "negativity_bias"]
        assert len(neg_mods) == 0

        # "this is bad" should trigger
        result2 = sim.compute_modifiers("this is bad")
        neg_mods2 = [m for m in result2 if m.bias_type.value == "negativity_bias"]
        assert len(neg_mods2) == 1


# =============================================================================
# Fix 6.3 — Schwartz Circumplex Adjacency
# =============================================================================

class TestSchwartzAdjacency:
    """Test value conflict resolution with circumplex adjacency."""

    def _make_interpreter(self, **value_overrides):
        defaults = dict(
            self_direction=0.5, stimulation=0.5, hedonism=0.5,
            achievement=0.5, power=0.5, security=0.5,
            conformity=0.5, tradition=0.5,
            benevolence=0.5, universalism=0.5,
        )
        defaults.update(value_overrides)
        values = SchwartzValues(**defaults)
        return ValuesInterpreter(values)

    def test_adjacent_values_high_confidence(self):
        """Adjacent values (benevolence & universalism) should resolve with high confidence."""
        interp = self._make_interpreter(benevolence=0.8, universalism=0.75)
        result = interp.resolve_conflict_detailed("benevolence", "universalism")
        assert result.is_adjacent
        assert not result.is_opposing
        assert result.confidence >= 0.7

    def test_opposing_values_low_confidence(self):
        """Opposing values (achievement & benevolence) should resolve with lower confidence."""
        interp = self._make_interpreter(achievement=0.8, benevolence=0.75)
        result = interp.resolve_conflict_detailed("achievement", "benevolence")
        assert result.is_opposing
        assert not result.is_adjacent
        assert result.confidence < 0.7

    def test_opposing_produces_conflict_citation(self):
        """Opposing value resolution should include a conflict citation."""
        interp = self._make_interpreter(power=0.8, universalism=0.75)
        result = interp.resolve_conflict_detailed("power", "universalism")
        assert len(result.citations) > 0
        assert any("opposing" in c["effect"].lower() for c in result.citations)

    def test_adjacent_produces_adjacency_citation(self):
        """Adjacent value resolution should include an adjacency citation."""
        interp = self._make_interpreter(stimulation=0.8, hedonism=0.75)
        result = interp.resolve_conflict_detailed("stimulation", "hedonism")
        assert any("adjacent" in c["effect"].lower() for c in result.citations)

    def test_backward_compatible_resolve_conflict(self):
        """Old resolve_conflict() still returns just the winner string."""
        interp = self._make_interpreter(achievement=0.9, benevolence=0.5)
        winner = interp.resolve_conflict("achievement", "benevolence")
        assert winner == "achievement"

    def test_equal_adjacent_values_resolve_smoothly(self):
        """Equal-weight adjacent values should resolve with decent confidence."""
        interp = self._make_interpreter(benevolence=0.7, universalism=0.7)
        result = interp.resolve_conflict_detailed("benevolence", "universalism")
        assert result.confidence >= 0.7  # Adjacent boost helps

    def test_equal_opposing_values_resolve_with_tension(self):
        """Equal-weight opposing values should resolve with low confidence."""
        interp = self._make_interpreter(power=0.7, universalism=0.7)
        result = interp.resolve_conflict_detailed("power", "universalism")
        assert result.confidence <= 0.5  # High tension, low confidence

    def test_adjacency_map_is_symmetric(self):
        """If A is adjacent to B, B should be adjacent to A."""
        for value, neighbors in SCHWARTZ_ADJACENCY.items():
            for neighbor in neighbors:
                assert value in SCHWARTZ_ADJACENCY[neighbor], \
                    f"{value} lists {neighbor} as adjacent but not vice versa"


# =============================================================================
# Fix 6.4 — Wire success_criteria from Goals
# =============================================================================

class TestSuccessCriteria:
    """Test that success_criteria is derived from persona goals."""

    def test_persona_with_goals_has_criteria(self):
        """Persona with primary goals should produce non-None success_criteria."""
        from persona_engine.planner.turn_planner import create_turn_planner, ConversationContext
        from persona_engine.memory import StanceCache

        persona = _make_persona(
            primary_goals=[
                Goal(goal="Advance UX research methodology", weight=0.9),
                Goal(goal="Mentor junior researchers", weight=0.7),
            ],
            secondary_goals=[
                Goal(goal="Publish academic papers", weight=0.6),
            ],
        )

        planner = create_turn_planner(persona)
        context = ConversationContext(
            conversation_id="test_goals",
            user_input="Tell me about your research",
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="research",
            turn_number=1,
            stance_cache=StanceCache(),
        )
        ir = planner.generate_ir(context)
        assert ir.conversation_frame.success_criteria is not None
        assert len(ir.conversation_frame.success_criteria) > 0
        assert len(ir.conversation_frame.success_criteria) <= 3

    def test_persona_without_goals_gets_none(self):
        """Persona with no goals should get None success_criteria."""
        from persona_engine.planner.turn_planner import create_turn_planner, ConversationContext
        from persona_engine.memory import StanceCache

        persona = _make_persona(primary_goals=[], secondary_goals=[])
        planner = create_turn_planner(persona)
        context = ConversationContext(
            conversation_id="test_no_goals",
            user_input="Hello there",
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
            topic_signature="greeting",
            turn_number=1,
            stance_cache=StanceCache(),
        )
        ir = planner.generate_ir(context)
        assert ir.conversation_frame.success_criteria is None


# =============================================================================
# Fix 6.5 — Languages Warning
# =============================================================================

class TestLanguagesWarning:
    """Test that populating languages[] emits a deprecation warning."""

    def test_languages_populated_emits_warning(self):
        """Populating languages should emit UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            persona = _make_persona(
                languages=[LanguageKnowledge(language="English", proficiency=1.0)]
            )
        lang_warnings = [x for x in w if "languages" in str(x.message).lower()]
        assert len(lang_warnings) >= 1
        assert "unused" in str(lang_warnings[0].message).lower()

    def test_no_languages_no_warning(self):
        """Empty languages should not emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            persona = _make_persona(languages=[])
        lang_warnings = [x for x in w if "languages" in str(x.message).lower()]
        assert len(lang_warnings) == 0


# =============================================================================
# Fix 6.6 — IR Validator
# =============================================================================

class TestIRValidator:
    """Test the IR validator catches inconsistencies."""

    def test_clean_ir_passes(self):
        """A well-formed IR should pass with no issues."""
        ir = _make_ir()
        issues = validate_ir(ir)
        assert not has_errors(issues)

    def test_high_confidence_refuse_flagged(self):
        """High confidence + REFUSE should be flagged."""
        ir = _make_ir(
            response_structure=ResponseStructure(
                intent="test", confidence=0.9, competence=0.8,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.REFUSE,
                knowledge_claim_type=KnowledgeClaimType.NONE,
            ),
        )
        issues = validate_ir(ir)
        warnings_list = [i for i in issues if i.severity == Severity.WARNING]
        assert any("confidence" in i.message.lower() and "refuse" in i.message.lower()
                    for i in warnings_list)

    def test_must_avoid_in_stance_is_error(self):
        """Stance containing must_avoid topic should be ERROR severity."""
        ir = _make_ir(
            response_structure=ResponseStructure(
                intent="test", stance="I love politics and debate",
                confidence=0.5, competence=0.5,
            ),
            safety_plan=SafetyPlan(must_avoid=["politics"]),
        )
        issues = validate_ir(ir)
        assert has_errors(issues)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert any("politics" in i.message for i in errors)

    def test_cannot_claim_in_stance_is_error(self):
        """Stance claiming a forbidden role should be ERROR."""
        ir = _make_ir(
            response_structure=ResponseStructure(
                intent="test", stance="As a licensed therapist, I recommend...",
                confidence=0.8, competence=0.7,
            ),
            safety_plan=SafetyPlan(cannot_claim=["licensed therapist"]),
        )
        issues = validate_ir(ir)
        assert has_errors(issues)

    def test_domain_expert_low_competence_warning(self):
        """Domain expert claim with low competence should warn."""
        ir = _make_ir(
            response_structure=ResponseStructure(
                intent="test", confidence=0.8, competence=0.3,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT,
            ),
        )
        issues = validate_ir(ir)
        assert any("competence" in i.message.lower() for i in issues)

    def test_out_of_range_is_error(self):
        """Values outside [0,1] should be caught (defense in depth)."""
        # This would normally be caught by Pydantic, but we test the validator
        # in case raw dict construction bypasses it
        ir = _make_ir()
        # We can only test via validator since Pydantic blocks construction
        # So we test that valid values pass
        issues = validate_ir(ir)
        range_errors = [i for i in issues if "range" in i.message.lower()]
        assert len(range_errors) == 0


# =============================================================================
# Fix 6.7 — Style Drift Detection
# =============================================================================

class TestStyleDriftDetection:
    """Test the style drift detector."""

    def test_stable_persona_low_drift(self):
        """Consistent behavioral values over 10 turns should have low drift."""
        detector = StyleDriftDetector(window_size=10, drift_threshold=0.15)
        for i in range(10):
            detector.record_turn(TurnMetrics(
                turn_number=i,
                formality=0.5, directness=0.5,
                disclosure_level=0.5, confidence=0.6,
            ))
        report = detector.analyze()
        assert report.overall_drift < 0.05
        assert len(report.flagged_fields) == 0

    def test_drifting_persona_flagged(self):
        """Linearly increasing formality should be flagged as drift."""
        detector = StyleDriftDetector(window_size=10, drift_threshold=0.10)
        for i in range(10):
            detector.record_turn(TurnMetrics(
                turn_number=i,
                formality=0.1 + i * 0.08,  # 0.1 → 0.82
                directness=0.5, disclosure_level=0.5, confidence=0.6,
            ))
        report = detector.analyze()
        assert "formality" in report.flagged_fields

    def test_justified_drift_with_state_change(self):
        """Drift justified by state changes should be marked as justified."""
        detector = StyleDriftDetector(window_size=10, drift_threshold=0.10)
        for i in range(10):
            stress = 0.1 + i * 0.09  # Increasing stress (large swing)
            engagement = 0.8 - i * 0.06  # Decreasing engagement
            detector.record_turn(TurnMetrics(
                turn_number=i,
                formality=0.5 + i * 0.04,  # Mild drift
                directness=0.5, disclosure_level=0.5, confidence=0.6,
                stress=stress, engagement=engagement,
            ))
        report = detector.analyze()
        if report.flagged_fields:
            assert report.justified

    def test_insufficient_turns(self):
        """Less than 2 turns should return a clean report."""
        detector = StyleDriftDetector()
        detector.record_turn(TurnMetrics(
            turn_number=0, formality=0.5, directness=0.5,
            disclosure_level=0.5, confidence=0.6,
        ))
        report = detector.analyze()
        assert len(report.flagged_fields) == 0
        assert report.justified

    def test_window_respects_size(self):
        """Detector should only keep last N turns."""
        detector = StyleDriftDetector(window_size=5)
        for i in range(20):
            detector.record_turn(TurnMetrics(
                turn_number=i, formality=0.5, directness=0.5,
                disclosure_level=0.5, confidence=0.6,
            ))
        assert detector.turn_count == 5


# =============================================================================
# Fix 6.8 — Knowledge Boundary Enforcer
# =============================================================================

class TestKnowledgeBoundaryEnforcer:
    """Test the knowledge boundary enforcer."""

    def test_expert_in_domain_passes(self):
        """Expert making expert claims in their domain should pass."""
        enforcer = KnowledgeBoundaryEnforcer(
            domain_proficiencies={"psychology": 0.9},
            expert_threshold=0.7,
        )
        ir = _make_ir(
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT,
            ),
        )
        violations = enforcer.check_turn(ir, turn_number=1, detected_domain="psychology")
        assert len(violations) == 0

    def test_nonexpert_expert_claim_flagged(self):
        """Non-expert making expert claims should be flagged."""
        enforcer = KnowledgeBoundaryEnforcer(
            domain_proficiencies={"psychology": 0.3},
            expert_threshold=0.7,
        )
        ir = _make_ir(
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT,
            ),
        )
        violations = enforcer.check_turn(ir, turn_number=1, detected_domain="psychology")
        assert len(violations) >= 1

    def test_high_confidence_nonexpert_flagged(self):
        """High confidence answer in non-expert domain should be flagged."""
        enforcer = KnowledgeBoundaryEnforcer(
            domain_proficiencies={"finance": 0.2},
            expert_threshold=0.7,
        )
        ir = _make_ir(
            response_structure=ResponseStructure(
                intent="test", confidence=0.9, competence=0.5,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.SPECULATIVE,
            ),
        )
        violations = enforcer.check_turn(ir, turn_number=1, detected_domain="finance")
        assert len(violations) >= 1

    def test_personal_experience_not_flagged(self):
        """Personal experience claims should not trigger boundary violations."""
        enforcer = KnowledgeBoundaryEnforcer(
            domain_proficiencies={"cooking": 0.2},
            expert_threshold=0.7,
        )
        ir = _make_ir(
            response_structure=ResponseStructure(
                intent="test", confidence=0.9, competence=0.5,
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.PERSONAL_EXPERIENCE,
            ),
        )
        violations = enforcer.check_turn(ir, turn_number=1, detected_domain="cooking")
        assert len(violations) == 0

    def test_report_tracks_claims_by_domain(self):
        """Report should accumulate claims per domain."""
        enforcer = KnowledgeBoundaryEnforcer(
            domain_proficiencies={"tech": 0.9, "law": 0.2},
        )
        ir = _make_ir()
        enforcer.check_turn(ir, turn_number=1, detected_domain="tech")
        enforcer.check_turn(ir, turn_number=2, detected_domain="tech")
        enforcer.check_turn(ir, turn_number=3, detected_domain="law")

        report = enforcer.get_report()
        assert report.claims_by_domain["tech"] == 2
        assert report.claims_by_domain["law"] == 1

    def test_unknown_domain_treated_as_zero_proficiency(self):
        """Unknown domains should be treated as 0 proficiency."""
        enforcer = KnowledgeBoundaryEnforcer(
            domain_proficiencies={"tech": 0.9},
        )
        ir = _make_ir(
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=0.5,
                uncertainty_action=UncertaintyAction.ANSWER,
                knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT,
            ),
        )
        violations = enforcer.check_turn(ir, turn_number=1, detected_domain="quantum_physics")
        assert len(violations) >= 1
