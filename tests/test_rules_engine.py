"""
Comprehensive tests for persona_engine.behavioral.rules_engine

Targets: BehavioralRulesEngine (all methods, all branches) and
         create_behavioral_rules_engine factory function.

Goal: raise coverage from ~59 % to 95 %+.
"""

import pytest

from persona_engine.behavioral.rules_engine import (
    BehavioralRulesEngine,
    create_behavioral_rules_engine,
)
from persona_engine.schema.ir_schema import InteractionMode
from persona_engine.schema.persona_schema import (
    BigFiveTraits,
    CognitiveStyle,
    CommunicationPreferences,
    DecisionPolicy,
    DisclosurePolicy,
    DynamicState,
    Identity,
    Persona,
    PersonaInvariants,
    PersonalityProfile,
    ClaimPolicy,
    ResponsePattern,
    SchwartzValues,
    SocialRole,
    TopicSensitivity,
    UncertaintyPolicy,
)


# ---------------------------------------------------------------------------
# Helper: reusable persona factory
# ---------------------------------------------------------------------------

def make_test_persona(**overrides):
    """Build a minimal but complete Persona for testing."""
    defaults = dict(
        persona_id="test",
        label="Test Persona",
        identity=Identity(
            age=30,
            location="NYC",
            education="BS",
            occupation="Dev",
            background="Test",
        ),
        psychology=PersonalityProfile(
            big_five=BigFiveTraits(
                openness=0.7,
                conscientiousness=0.6,
                extraversion=0.5,
                agreeableness=0.6,
                neuroticism=0.4,
            ),
            values=SchwartzValues(
                self_direction=0.7,
                stimulation=0.5,
                hedonism=0.4,
                achievement=0.8,
                power=0.3,
                security=0.6,
                conformity=0.4,
                tradition=0.3,
                benevolence=0.7,
                universalism=0.6,
            ),
            cognitive_style=CognitiveStyle(
                analytical_intuitive=0.7,
                systematic_heuristic=0.6,
                risk_tolerance=0.5,
                need_for_closure=0.5,
                cognitive_complexity=0.7,
            ),
            communication=CommunicationPreferences(
                verbosity=0.6,
                formality=0.5,
                directness=0.6,
                emotional_expressiveness=0.5,
            ),
        ),
        social_roles={
            "default": SocialRole(
                formality=0.5, directness=0.5, emotional_expressiveness=0.5
            ),
            "at_work": SocialRole(
                formality=0.8, directness=0.6, emotional_expressiveness=0.3
            ),
            "friend": SocialRole(
                formality=0.2, directness=0.7, emotional_expressiveness=0.8
            ),
        },
        uncertainty=UncertaintyPolicy(
            admission_threshold=0.4,
            hedging_frequency=0.5,
            clarification_tendency=0.6,
            knowledge_boundary_strictness=0.7,
        ),
        claim_policy=ClaimPolicy(),
        invariants=PersonaInvariants(identity_facts=["Age 30", "Lives in NYC"]),
        disclosure_policy=DisclosurePolicy(
            base_openness=0.5, factors={"topic_sensitivity": -0.3}
        ),
        initial_state=DynamicState(
            mood_valence=0.2,
            mood_arousal=0.5,
            fatigue=0.1,
            stress=0.2,
            engagement=0.7,
        ),
        time_scarcity=0.5,
        privacy_sensitivity=0.4,
    )
    defaults.update(overrides)
    return Persona(**defaults)


@pytest.fixture
def persona():
    """A standard persona with default, at_work, and friend roles."""
    return make_test_persona()


@pytest.fixture
def engine(persona):
    return BehavioralRulesEngine(persona)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    """BehavioralRulesEngine.__init__"""

    def test_stores_persona(self, persona):
        engine = BehavioralRulesEngine(persona)
        assert engine.persona is persona

    def test_copies_social_roles(self, persona):
        engine = BehavioralRulesEngine(persona)
        assert engine.social_roles is persona.social_roles

    def test_copies_decision_policies(self, persona):
        engine = BehavioralRulesEngine(persona)
        assert engine.decision_policies is persona.decision_policies

    def test_copies_response_patterns(self, persona):
        engine = BehavioralRulesEngine(persona)
        assert engine.response_patterns is persona.response_patterns

    def test_copies_base_communication(self, persona):
        engine = BehavioralRulesEngine(persona)
        assert engine.base_communication is persona.psychology.communication


# ---------------------------------------------------------------------------
# get_social_role_mode
# ---------------------------------------------------------------------------

class TestGetSocialRoleMode:

    # --- explicit mapping entries ---

    def test_customer_support_maps_to_at_work(self, engine):
        assert engine.get_social_role_mode(InteractionMode.CUSTOMER_SUPPORT) == "at_work"

    def test_interview_maps_to_at_work(self, engine):
        assert engine.get_social_role_mode(InteractionMode.INTERVIEW) == "at_work"

    def test_casual_chat_maps_to_friend(self, engine):
        assert engine.get_social_role_mode(InteractionMode.CASUAL_CHAT) == "friend"

    def test_small_talk_maps_to_friend(self, engine):
        assert engine.get_social_role_mode(InteractionMode.SMALL_TALK) == "friend"

    def test_survey_maps_to_default(self, engine):
        assert engine.get_social_role_mode(InteractionMode.SURVEY) == "default"

    def test_coaching_maps_to_default(self, engine):
        assert engine.get_social_role_mode(InteractionMode.COACHING) == "default"

    def test_debate_maps_to_debate(self, engine):
        assert engine.get_social_role_mode(InteractionMode.DEBATE) == "debate"

    # --- BRAINSTORM is NOT in the mapping dict -> falls back to "default" ---

    def test_brainstorm_falls_back_to_default(self, engine):
        assert engine.get_social_role_mode(InteractionMode.BRAINSTORM) == "default"

    # --- mapped role doesn't exist in persona.social_roles -> fallback ---

    def test_fallback_when_mapped_role_missing(self):
        """If persona has no 'friend' role, CASUAL_CHAT should fall back to 'default'."""
        persona = make_test_persona(
            social_roles={
                "default": SocialRole(
                    formality=0.5, directness=0.5, emotional_expressiveness=0.5
                ),
                # no 'friend' or 'at_work'
            }
        )
        engine = BehavioralRulesEngine(persona)
        assert engine.get_social_role_mode(InteractionMode.CASUAL_CHAT) == "default"

    def test_fallback_when_at_work_missing(self):
        """CUSTOMER_SUPPORT maps to 'at_work'; missing -> 'default'."""
        persona = make_test_persona(
            social_roles={
                "default": SocialRole(
                    formality=0.5, directness=0.5, emotional_expressiveness=0.5
                ),
            }
        )
        engine = BehavioralRulesEngine(persona)
        assert engine.get_social_role_mode(InteractionMode.CUSTOMER_SUPPORT) == "default"


# ---------------------------------------------------------------------------
# apply_social_role_adjustments
# ---------------------------------------------------------------------------

class TestApplySocialRoleAdjustments:

    def _assert_blend(self, result, role, base_f, base_d, base_e):
        """Verify 70/30 blend arithmetic."""
        expected_f = role.formality * 0.7 + base_f * 0.3
        expected_d = role.directness * 0.7 + base_d * 0.3
        expected_e = role.emotional_expressiveness * 0.7 + base_e * 0.3
        assert result["formality"] == pytest.approx(expected_f)
        assert result["directness"] == pytest.approx(expected_d)
        assert result["emotional_expressiveness"] == pytest.approx(expected_e)

    def test_at_work_blend(self, engine, persona):
        base_f, base_d, base_e = 0.4, 0.5, 0.6
        result = engine.apply_social_role_adjustments(
            InteractionMode.INTERVIEW, base_f, base_d, base_e
        )
        self._assert_blend(result, persona.social_roles["at_work"], base_f, base_d, base_e)

    def test_friend_blend(self, engine, persona):
        base_f, base_d, base_e = 0.5, 0.5, 0.5
        result = engine.apply_social_role_adjustments(
            InteractionMode.CASUAL_CHAT, base_f, base_d, base_e
        )
        self._assert_blend(result, persona.social_roles["friend"], base_f, base_d, base_e)

    def test_default_blend(self, engine, persona):
        base_f, base_d, base_e = 0.3, 0.8, 0.2
        result = engine.apply_social_role_adjustments(
            InteractionMode.SURVEY, base_f, base_d, base_e
        )
        self._assert_blend(result, persona.social_roles["default"], base_f, base_d, base_e)

    def test_brainstorm_uses_default_blend(self, engine, persona):
        base_f, base_d, base_e = 0.6, 0.6, 0.6
        result = engine.apply_social_role_adjustments(
            InteractionMode.BRAINSTORM, base_f, base_d, base_e
        )
        self._assert_blend(result, persona.social_roles["default"], base_f, base_d, base_e)

    def test_returns_dict_with_three_keys(self, engine):
        result = engine.apply_social_role_adjustments(
            InteractionMode.DEBATE, 0.5, 0.5, 0.5
        )
        assert set(result.keys()) == {"formality", "directness", "emotional_expressiveness"}

    def test_extreme_base_values_zero(self, engine, persona):
        result = engine.apply_social_role_adjustments(
            InteractionMode.INTERVIEW, 0.0, 0.0, 0.0
        )
        role = persona.social_roles["at_work"]
        assert result["formality"] == pytest.approx(role.formality * 0.7)
        assert result["directness"] == pytest.approx(role.directness * 0.7)
        assert result["emotional_expressiveness"] == pytest.approx(
            role.emotional_expressiveness * 0.7
        )

    def test_extreme_base_values_one(self, engine, persona):
        result = engine.apply_social_role_adjustments(
            InteractionMode.INTERVIEW, 1.0, 1.0, 1.0
        )
        role = persona.social_roles["at_work"]
        assert result["formality"] == pytest.approx(role.formality * 0.7 + 0.3)
        assert result["directness"] == pytest.approx(role.directness * 0.7 + 0.3)
        assert result["emotional_expressiveness"] == pytest.approx(
            role.emotional_expressiveness * 0.7 + 0.3
        )

    def test_customer_support_blend(self, engine, persona):
        """CUSTOMER_SUPPORT also maps to at_work - verify separately."""
        base_f, base_d, base_e = 0.9, 0.1, 0.7
        result = engine.apply_social_role_adjustments(
            InteractionMode.CUSTOMER_SUPPORT, base_f, base_d, base_e
        )
        self._assert_blend(result, persona.social_roles["at_work"], base_f, base_d, base_e)

    def test_small_talk_blend(self, engine, persona):
        base_f, base_d, base_e = 0.2, 0.3, 0.4
        result = engine.apply_social_role_adjustments(
            InteractionMode.SMALL_TALK, base_f, base_d, base_e
        )
        self._assert_blend(result, persona.social_roles["friend"], base_f, base_d, base_e)


# ---------------------------------------------------------------------------
# check_decision_policy
# ---------------------------------------------------------------------------

class TestCheckDecisionPolicy:

    @pytest.fixture
    def engine_with_policies(self):
        persona = make_test_persona(
            decision_policies=[
                DecisionPolicy(
                    condition="budget",
                    approach="conservative",
                    time_needed="brief",
                ),
                DecisionPolicy(
                    condition="deadline",
                    approach="prioritize",
                    time_needed="immediate",
                ),
            ]
        )
        return BehavioralRulesEngine(persona)

    def test_match_first_policy(self, engine_with_policies):
        result = engine_with_policies.check_decision_policy(
            "We need to review the budget for Q3."
        )
        assert result is not None
        assert result.condition == "budget"

    def test_match_second_policy(self, engine_with_policies):
        result = engine_with_policies.check_decision_policy(
            "The deadline is approaching fast."
        )
        assert result is not None
        assert result.condition == "deadline"

    def test_case_insensitive_match(self, engine_with_policies):
        result = engine_with_policies.check_decision_policy("BUDGET review needed")
        assert result is not None
        assert result.condition == "budget"

    def test_no_match_returns_none(self, engine_with_policies):
        result = engine_with_policies.check_decision_policy("Let's discuss the weather.")
        assert result is None

    def test_empty_policies_returns_none(self, engine):
        """Default persona has no decision_policies."""
        result = engine.check_decision_policy("anything goes")
        assert result is None

    def test_returns_first_match_when_multiple_qualify(self):
        """Both policies match the situation; first one wins."""
        persona = make_test_persona(
            decision_policies=[
                DecisionPolicy(condition="project", approach="plan"),
                DecisionPolicy(condition="project", approach="delegate"),
            ]
        )
        eng = BehavioralRulesEngine(persona)
        result = eng.check_decision_policy("discuss the project timeline")
        assert result is not None
        assert result.approach == "plan"  # first match


# ---------------------------------------------------------------------------
# apply_decision_policy
# ---------------------------------------------------------------------------

class TestApplyDecisionPolicy:

    def test_without_time_needed(self, engine):
        policy = DecisionPolicy(condition="budget", approach="save money")
        result = engine.apply_decision_policy(policy)
        assert result["policy_triggered"] is True
        assert result["policy_name"] == "budget"
        assert result["suggested_approach"] == "save money"
        assert "decision_time" not in result

    def test_time_needed_immediate(self, engine):
        policy = DecisionPolicy(
            condition="fire", approach="evacuate", time_needed="immediate"
        )
        result = engine.apply_decision_policy(policy)
        assert result["decision_time"] == pytest.approx(0.1)

    def test_time_needed_brief(self, engine):
        policy = DecisionPolicy(
            condition="x", approach="y", time_needed="brief"
        )
        assert engine.apply_decision_policy(policy)["decision_time"] == pytest.approx(0.3)

    def test_time_needed_moderate(self, engine):
        policy = DecisionPolicy(
            condition="x", approach="y", time_needed="moderate"
        )
        assert engine.apply_decision_policy(policy)["decision_time"] == pytest.approx(0.5)

    def test_time_needed_extended(self, engine):
        policy = DecisionPolicy(
            condition="x", approach="y", time_needed="extended"
        )
        assert engine.apply_decision_policy(policy)["decision_time"] == pytest.approx(0.8)

    def test_time_needed_unknown_defaults_to_half(self, engine):
        policy = DecisionPolicy(
            condition="x", approach="y", time_needed="forever"
        )
        assert engine.apply_decision_policy(policy)["decision_time"] == pytest.approx(0.5)

    def test_output_keys_with_time(self, engine):
        policy = DecisionPolicy(
            condition="c", approach="a", time_needed="brief"
        )
        result = engine.apply_decision_policy(policy)
        assert set(result.keys()) == {
            "policy_triggered",
            "policy_name",
            "suggested_approach",
            "decision_time",
        }

    def test_output_keys_without_time(self, engine):
        policy = DecisionPolicy(condition="c", approach="a")
        result = engine.apply_decision_policy(policy)
        assert set(result.keys()) == {
            "policy_triggered",
            "policy_name",
            "suggested_approach",
        }


# ---------------------------------------------------------------------------
# check_response_pattern
# ---------------------------------------------------------------------------

class TestCheckResponsePattern:

    @pytest.fixture
    def engine_with_patterns(self):
        persona = make_test_persona(
            response_patterns=[
                ResponsePattern(
                    trigger="thank you",
                    response="You're welcome!",
                    emotionality=0.6,
                ),
                ResponsePattern(
                    trigger="sorry",
                    response="No worries!",
                    emotionality=0.4,
                ),
            ]
        )
        return BehavioralRulesEngine(persona)

    def test_match_first_pattern(self, engine_with_patterns):
        result = engine_with_patterns.check_response_pattern(
            "I just wanted to say thank you for helping."
        )
        assert result is not None
        assert result.trigger == "thank you"

    def test_match_second_pattern(self, engine_with_patterns):
        result = engine_with_patterns.check_response_pattern("I'm sorry about that")
        assert result is not None
        assert result.trigger == "sorry"

    def test_case_insensitive(self, engine_with_patterns):
        result = engine_with_patterns.check_response_pattern("THANK YOU so much!")
        assert result is not None
        assert result.trigger == "thank you"

    def test_no_match_returns_none(self, engine_with_patterns):
        result = engine_with_patterns.check_response_pattern("How is the weather?")
        assert result is None

    def test_empty_patterns_returns_none(self, engine):
        """Default persona has no response_patterns."""
        result = engine.check_response_pattern("anything")
        assert result is None

    def test_returns_first_match_when_multiple_qualify(self):
        persona = make_test_persona(
            response_patterns=[
                ResponsePattern(trigger="hello", response="Hi!", emotionality=0.3),
                ResponsePattern(trigger="hello", response="Hey!", emotionality=0.5),
            ]
        )
        eng = BehavioralRulesEngine(persona)
        result = eng.check_response_pattern("hello there")
        assert result is not None
        assert result.response == "Hi!"  # first match


# ---------------------------------------------------------------------------
# apply_response_pattern
# ---------------------------------------------------------------------------

class TestApplyResponsePattern:

    def test_basic_output(self, engine):
        pattern = ResponsePattern(
            trigger="thanks", response="No problem!", emotionality=0.8
        )
        result = engine.apply_response_pattern(pattern)
        assert result["pattern_triggered"] is True
        assert result["trigger"] == "thanks"
        assert result["suggested_response"] == "No problem!"
        assert result["emotional_intensity"] == pytest.approx(0.8)
        assert result["arousal_boost"] == pytest.approx(0.8 * 0.3)

    def test_zero_emotionality(self, engine):
        pattern = ResponsePattern(
            trigger="ok", response="Acknowledged.", emotionality=0.0
        )
        result = engine.apply_response_pattern(pattern)
        assert result["emotional_intensity"] == pytest.approx(0.0)
        assert result["arousal_boost"] == pytest.approx(0.0)

    def test_max_emotionality(self, engine):
        pattern = ResponsePattern(
            trigger="wow", response="Amazing!", emotionality=1.0
        )
        result = engine.apply_response_pattern(pattern)
        assert result["emotional_intensity"] == pytest.approx(1.0)
        assert result["arousal_boost"] == pytest.approx(0.3)

    def test_output_keys(self, engine):
        pattern = ResponsePattern(
            trigger="t", response="r", emotionality=0.5
        )
        result = engine.apply_response_pattern(pattern)
        assert set(result.keys()) == {
            "pattern_triggered",
            "trigger",
            "suggested_response",
            "emotional_intensity",
            "arousal_boost",
        }


# ---------------------------------------------------------------------------
# apply_all_rules
# ---------------------------------------------------------------------------

class TestApplyAllRules:

    def test_only_social_role_when_no_policies_or_patterns(self, engine):
        result = engine.apply_all_rules(
            InteractionMode.CASUAL_CHAT,
            "Let's chat about hiking.",
            {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5},
        )
        assert "social_role" in result
        assert "decision_policy" not in result
        assert "response_pattern" not in result

    def test_social_role_values_are_correct(self, engine, persona):
        base = {"formality": 0.4, "directness": 0.6, "emotional_expressiveness": 0.3}
        result = engine.apply_all_rules(
            InteractionMode.INTERVIEW, "some text", base
        )
        role = persona.social_roles["at_work"]
        sr = result["social_role"]
        assert sr["formality"] == pytest.approx(role.formality * 0.7 + 0.4 * 0.3)
        assert sr["directness"] == pytest.approx(role.directness * 0.7 + 0.6 * 0.3)
        assert sr["emotional_expressiveness"] == pytest.approx(
            role.emotional_expressiveness * 0.7 + 0.3 * 0.3
        )

    def test_includes_decision_policy_when_matched(self):
        persona = make_test_persona(
            decision_policies=[
                DecisionPolicy(condition="budget", approach="cut costs", time_needed="brief")
            ]
        )
        eng = BehavioralRulesEngine(persona)
        result = eng.apply_all_rules(
            InteractionMode.SURVEY,
            "Please review the budget now.",
            {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5},
        )
        assert "decision_policy" in result
        assert result["decision_policy"]["policy_triggered"] is True
        assert result["decision_policy"]["policy_name"] == "budget"
        assert result["decision_policy"]["decision_time"] == pytest.approx(0.3)

    def test_includes_response_pattern_when_matched(self):
        persona = make_test_persona(
            response_patterns=[
                ResponsePattern(trigger="thank you", response="You're welcome!", emotionality=0.7)
            ]
        )
        eng = BehavioralRulesEngine(persona)
        result = eng.apply_all_rules(
            InteractionMode.CASUAL_CHAT,
            "Thank you for everything!",
            {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5},
        )
        assert "response_pattern" in result
        assert result["response_pattern"]["pattern_triggered"] is True
        assert result["response_pattern"]["trigger"] == "thank you"
        assert result["response_pattern"]["arousal_boost"] == pytest.approx(0.7 * 0.3)

    def test_includes_both_policy_and_pattern(self):
        persona = make_test_persona(
            decision_policies=[
                DecisionPolicy(condition="budget", approach="save", time_needed="moderate")
            ],
            response_patterns=[
                ResponsePattern(trigger="budget", response="I'll handle it.", emotionality=0.5)
            ],
        )
        eng = BehavioralRulesEngine(persona)
        result = eng.apply_all_rules(
            InteractionMode.DEBATE,
            "Let's finalize the budget proposal.",
            {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5},
        )
        assert "social_role" in result
        assert "decision_policy" in result
        assert "response_pattern" in result

    def test_base_style_defaults_when_keys_missing(self):
        """If base_style dict is missing keys, 0.5 default is used."""
        persona = make_test_persona()
        eng = BehavioralRulesEngine(persona)
        result = eng.apply_all_rules(
            InteractionMode.SURVEY,
            "hello",
            {},  # empty base_style
        )
        role = persona.social_roles["default"]
        sr = result["social_role"]
        assert sr["formality"] == pytest.approx(role.formality * 0.7 + 0.5 * 0.3)
        assert sr["directness"] == pytest.approx(role.directness * 0.7 + 0.5 * 0.3)
        assert sr["emotional_expressiveness"] == pytest.approx(
            role.emotional_expressiveness * 0.7 + 0.5 * 0.3
        )


# ---------------------------------------------------------------------------
# get_privacy_filter_level
# ---------------------------------------------------------------------------

class TestGetPrivacyFilterLevel:

    @pytest.fixture
    def engine_with_sensitivities(self):
        persona = make_test_persona(
            topic_sensitivities=[
                TopicSensitivity(topic="health", sensitivity=0.9),
                TopicSensitivity(topic="salary", sensitivity=0.85),
            ],
            privacy_sensitivity=0.4,
        )
        return BehavioralRulesEngine(persona)

    def test_match_health(self, engine_with_sensitivities):
        assert engine_with_sensitivities.get_privacy_filter_level("my health issues") == pytest.approx(0.9)

    def test_match_salary(self, engine_with_sensitivities):
        assert engine_with_sensitivities.get_privacy_filter_level("what is your salary?") == pytest.approx(0.85)

    def test_case_insensitive_topic(self, engine_with_sensitivities):
        assert engine_with_sensitivities.get_privacy_filter_level("HEALTH problems") == pytest.approx(0.9)

    def test_no_match_falls_back_to_base(self, engine_with_sensitivities):
        assert engine_with_sensitivities.get_privacy_filter_level("weather forecast") == pytest.approx(0.4)

    def test_no_sensitivities_falls_back(self, engine):
        """Default persona has empty topic_sensitivities."""
        assert engine.get_privacy_filter_level("anything") == pytest.approx(0.4)

    def test_returns_first_matching_sensitivity(self):
        """If two sensitivities overlap, first match wins."""
        persona = make_test_persona(
            topic_sensitivities=[
                TopicSensitivity(topic="money", sensitivity=0.7),
                TopicSensitivity(topic="money", sensitivity=0.3),
            ],
            privacy_sensitivity=0.4,
        )
        eng = BehavioralRulesEngine(persona)
        assert eng.get_privacy_filter_level("money talk") == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# should_apply_time_constraint
# ---------------------------------------------------------------------------

class TestShouldApplyTimeConstraint:

    def test_below_threshold_returns_false(self):
        # time_scarcity=0.5 -> threshold = int(10/0.5) = 20
        persona = make_test_persona(time_scarcity=0.5)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(19) is False

    def test_at_threshold_returns_false(self):
        # threshold = 20, conversation_length == 20 -> NOT > 20
        persona = make_test_persona(time_scarcity=0.5)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(20) is False

    def test_above_threshold_returns_true(self):
        persona = make_test_persona(time_scarcity=0.5)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(21) is True

    def test_high_time_scarcity_low_threshold(self):
        # time_scarcity=1.0 -> threshold = int(10/1.0) = 10
        persona = make_test_persona(time_scarcity=1.0)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(10) is False
        assert eng.should_apply_time_constraint(11) is True

    def test_low_time_scarcity_high_threshold(self):
        # time_scarcity=0.1 -> threshold = int(10/0.1) = 100
        persona = make_test_persona(time_scarcity=0.1)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(100) is False
        assert eng.should_apply_time_constraint(101) is True

    def test_zero_time_scarcity_uses_floor(self):
        # time_scarcity=0.0 -> max(0.1, 0.0) = 0.1 -> threshold = 100
        persona = make_test_persona(time_scarcity=0.0)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(100) is False
        assert eng.should_apply_time_constraint(101) is True

    def test_very_high_conversation_length(self):
        persona = make_test_persona(time_scarcity=0.5)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(10000) is True

    def test_zero_conversation_length(self):
        persona = make_test_persona(time_scarcity=0.5)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(0) is False

    def test_fractional_time_scarcity(self):
        # time_scarcity=0.3 -> threshold = int(10/0.3) = int(33.33) = 33
        persona = make_test_persona(time_scarcity=0.3)
        eng = BehavioralRulesEngine(persona)
        assert eng.should_apply_time_constraint(33) is False
        assert eng.should_apply_time_constraint(34) is True


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestCreateBehavioralRulesEngine:

    def test_returns_engine_instance(self, persona):
        engine = create_behavioral_rules_engine(persona)
        assert isinstance(engine, BehavioralRulesEngine)

    def test_engine_uses_given_persona(self, persona):
        engine = create_behavioral_rules_engine(persona)
        assert engine.persona is persona

    def test_factory_with_policies_and_patterns(self):
        persona = make_test_persona(
            decision_policies=[
                DecisionPolicy(condition="test", approach="do it")
            ],
            response_patterns=[
                ResponsePattern(trigger="hi", response="hello", emotionality=0.5)
            ],
        )
        engine = create_behavioral_rules_engine(persona)
        assert len(engine.decision_policies) == 1
        assert len(engine.response_patterns) == 1


# ---------------------------------------------------------------------------
# Integration-style: round-trip / combined scenarios
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:

    def test_full_pipeline_with_all_triggers(self):
        """End-to-end: policy match + pattern match + role adjustment."""
        persona = make_test_persona(
            decision_policies=[
                DecisionPolicy(
                    condition="refund",
                    approach="empathize then process",
                    time_needed="immediate",
                )
            ],
            response_patterns=[
                ResponsePattern(
                    trigger="refund",
                    response="I understand your frustration.",
                    emotionality=0.9,
                )
            ],
        )
        eng = BehavioralRulesEngine(persona)
        result = eng.apply_all_rules(
            InteractionMode.CUSTOMER_SUPPORT,
            "I need a refund for my order.",
            {"formality": 0.6, "directness": 0.4, "emotional_expressiveness": 0.5},
        )
        # social_role: at_work blend
        role = persona.social_roles["at_work"]
        sr = result["social_role"]
        assert sr["formality"] == pytest.approx(role.formality * 0.7 + 0.6 * 0.3)

        # decision_policy
        assert result["decision_policy"]["policy_triggered"] is True
        assert result["decision_policy"]["decision_time"] == pytest.approx(0.1)

        # response_pattern
        assert result["response_pattern"]["pattern_triggered"] is True
        assert result["response_pattern"]["arousal_boost"] == pytest.approx(0.9 * 0.3)

    def test_privacy_filter_with_different_topics(self):
        persona = make_test_persona(
            topic_sensitivities=[
                TopicSensitivity(topic="politics", sensitivity=0.8),
                TopicSensitivity(topic="religion", sensitivity=0.95),
            ],
            privacy_sensitivity=0.3,
        )
        eng = BehavioralRulesEngine(persona)
        assert eng.get_privacy_filter_level("politics today") == pytest.approx(0.8)
        assert eng.get_privacy_filter_level("religion and culture") == pytest.approx(0.95)
        assert eng.get_privacy_filter_level("cooking recipes") == pytest.approx(0.3)

    def test_all_interaction_modes_produce_valid_output(self):
        """Every InteractionMode should produce a dict with 'social_role'."""
        persona = make_test_persona()
        eng = BehavioralRulesEngine(persona)
        for mode in InteractionMode:
            result = eng.apply_all_rules(
                mode,
                "generic input",
                {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5},
            )
            assert "social_role" in result
            sr = result["social_role"]
            assert 0.0 <= sr["formality"] <= 1.0
            assert 0.0 <= sr["directness"] <= 1.0
            assert 0.0 <= sr["emotional_expressiveness"] <= 1.0
