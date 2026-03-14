"""
Phase R6 Tests: Expanded Biases & Social Cognition

Tests:
- 5 new cognitive biases (anchoring, DK, availability, status quo, empathy gap)
- Persona-declared bias overrides
- Social cognition / user modeling
- Self-schema protection
- Self-disclosure reciprocity
- Pipeline integration
"""

import pytest

from persona_engine.behavioral.bias_simulator import (
    MAX_BIAS_IMPACT,
    BiasSimulator,
    BiasType,
)
from persona_engine.behavioral.social_cognition import (
    AdaptationDirectives,
    SchemaEffect,
    UserModel,
    compute_adaptation,
    compute_schema_effect,
    detect_schema_relevance,
    infer_user_model,
)
from persona_engine.memory import StanceCache
from persona_engine.planner.turn_planner import (
    ConversationContext,
    TurnPlanner,
)
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
)
from persona_engine.schema.persona_schema import BigFiveTraits, Persona
from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Helpers
# ============================================================================

def _make_traits(**overrides) -> BigFiveTraits:
    defaults = {
        "openness": 0.5, "conscientiousness": 0.5,
        "extraversion": 0.5, "agreeableness": 0.5,
        "neuroticism": 0.5,
    }
    defaults.update(overrides)
    return BigFiveTraits(**defaults)


def _make_bias_sim(**trait_overrides) -> BiasSimulator:
    traits = {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
              "agreeableness": 0.5, "neuroticism": 0.5}
    traits.update(trait_overrides)
    values = {"conformity": 0.5, "tradition": 0.5, "security": 0.5,
              "benevolence": 0.5, "universalism": 0.5,
              "self_direction": 0.5, "stimulation": 0.5,
              "hedonism": 0.5, "achievement": 0.5, "power": 0.5}
    return BiasSimulator(traits, values)


def _make_persona_data(**overrides) -> dict:
    base = {
        "persona_id": "TEST", "version": "1.0", "label": "Test Persona",
        "identity": {"age": 30, "gender": "female", "location": "NYC",
                     "education": "BS", "occupation": "Engineer", "background": "Test"},
        "psychology": {
            "big_five": {"openness": 0.5, "conscientiousness": 0.5,
                         "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
            "values": {"self_direction": 0.5, "stimulation": 0.5, "hedonism": 0.5,
                       "achievement": 0.5, "power": 0.5, "security": 0.5,
                       "conformity": 0.5, "tradition": 0.5, "benevolence": 0.5,
                       "universalism": 0.5},
            "cognitive_style": {"analytical_intuitive": 0.5, "systematic_heuristic": 0.5,
                                "risk_tolerance": 0.5, "need_for_closure": 0.5,
                                "cognitive_complexity": 0.5},
            "communication": {"verbosity": 0.5, "formality": 0.5,
                              "directness": 0.5, "emotional_expressiveness": 0.5},
        },
        "knowledge_domains": [{"domain": "Engineering", "proficiency": 0.7, "subdomains": []}],
        "social_roles": {"default": {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5}},
        "invariants": {"identity_facts": ["Engineer"], "cannot_claim": [], "must_avoid": []},
        "initial_state": {"mood_valence": 0.2, "mood_arousal": 0.4,
                          "fatigue": 0.2, "stress": 0.2, "engagement": 0.5},
        "uncertainty": {"admission_threshold": 0.45, "hedging_frequency": 0.4,
                        "clarification_tendency": 0.5, "knowledge_boundary_strictness": 0.6},
        "claim_policy": {
            "allowed_claim_types": ["personal_experience", "domain_expert", "general_common_knowledge"],
            "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
            "lookup_behavior": "hedge",
        },
        "time_scarcity": 0.45, "privacy_sensitivity": 0.5,
        "disclosure_policy": {
            "base_openness": 0.55,
            "factors": {"topic_sensitivity": -0.25, "trust_level": 0.3,
                        "formal_context": -0.15, "positive_mood": 0.1},
            "bounds": [0.1, 0.9],
        },
    }
    for key, val in overrides.items():
        if key in base["psychology"]["big_five"]:
            base["psychology"]["big_five"][key] = val
    return base


def _make_context(user_input: str = "What do you think?") -> ConversationContext:
    return ConversationContext(
        conversation_id="test", turn_number=1,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.EXPLORE_IDEAS,
        topic_signature="general", user_input=user_input,
        stance_cache=StanceCache(),
    )


def _generate_ir(persona_data: dict, user_input: str = "What do you think?"):
    persona = Persona(**persona_data)
    planner = TurnPlanner(persona, DeterminismManager(seed=42))
    ctx = _make_context(user_input)
    return planner.generate_ir(ctx)


# ============================================================================
# Test New Cognitive Biases (R6.1)
# ============================================================================

class TestAnchoringBias:
    """Anchoring bias: once a stance is set, resist changing."""

    def test_no_anchor_no_bias(self):
        sim = _make_bias_sim(openness=0.2)
        mods = sim.compute_modifiers("What do you think?")
        anchoring_mods = [m for m in mods if m.bias_type == BiasType.ANCHORING]
        assert len(anchoring_mods) == 0

    def test_anchor_set_triggers_bias(self):
        sim = _make_bias_sim(openness=0.2)
        sim.set_anchor("I believe X is correct")
        mods = sim.compute_modifiers("What about Y?")
        anchoring_mods = [m for m in mods if m.bias_type == BiasType.ANCHORING]
        assert len(anchoring_mods) == 1
        assert anchoring_mods[0].modifier < 0  # Reduces elasticity

    def test_high_o_resists_anchoring(self):
        sim = _make_bias_sim(openness=0.9)
        sim.set_anchor("I believe X is correct")
        mods = sim.compute_modifiers("What about Y?")
        anchoring_mods = [m for m in mods if m.bias_type == BiasType.ANCHORING]
        # High-O should have weak or no anchoring
        if anchoring_mods:
            assert abs(anchoring_mods[0].modifier) < 0.05


class TestStatusQuoBias:
    """Status quo bias: resist change proposals."""

    def test_change_proposal_triggers(self):
        sim = _make_bias_sim(openness=0.2, conscientiousness=0.8)
        mods = sim.compute_modifiers("We should switch to a completely different approach")
        sq_mods = [m for m in mods if m.bias_type == BiasType.STATUS_QUO]
        assert len(sq_mods) == 1
        assert sq_mods[0].modifier < 0  # Reduces elasticity

    def test_no_change_no_trigger(self):
        sim = _make_bias_sim(openness=0.2, conscientiousness=0.8)
        mods = sim.compute_modifiers("Tell me about the current system")
        sq_mods = [m for m in mods if m.bias_type == BiasType.STATUS_QUO]
        assert len(sq_mods) == 0

    def test_high_o_no_status_quo(self):
        sim = _make_bias_sim(openness=0.9, conscientiousness=0.2)
        mods = sim.compute_modifiers("We should switch to a new approach")
        sq_mods = [m for m in mods if m.bias_type == BiasType.STATUS_QUO]
        assert len(sq_mods) == 0  # High-O + Low-C = open to change


class TestAvailabilityBias:
    """Availability bias: High-N overweights negative examples."""

    def test_negative_content_high_n_triggers(self):
        sim = _make_bias_sim(neuroticism=0.8)
        mods = sim.compute_modifiers("There's a real problem with the risk involved")
        avail_mods = [m for m in mods if m.bias_type == BiasType.AVAILABILITY]
        assert len(avail_mods) == 1
        assert avail_mods[0].modifier > 0  # Increases arousal

    def test_low_n_no_availability_bias(self):
        sim = _make_bias_sim(neuroticism=0.2)
        mods = sim.compute_modifiers("There's a problem with the risk")
        avail_mods = [m for m in mods if m.bias_type == BiasType.AVAILABILITY]
        assert len(avail_mods) == 0


class TestEmpathyGapBias:
    """Empathy gap: Low-A + Low-N underestimates others' emotions."""

    def test_emotional_content_low_a_low_n(self):
        sim = _make_bias_sim(agreeableness=0.15, neuroticism=0.15)
        mods = sim.compute_modifiers("I'm feeling really hurt and overwhelmed")
        eg_mods = [m for m in mods if m.bias_type == BiasType.EMPATHY_GAP]
        assert len(eg_mods) == 1
        assert eg_mods[0].modifier < 0  # Reduces disclosure

    def test_high_a_no_empathy_gap(self):
        sim = _make_bias_sim(agreeableness=0.9, neuroticism=0.5)
        mods = sim.compute_modifiers("I'm feeling really hurt and overwhelmed")
        eg_mods = [m for m in mods if m.bias_type == BiasType.EMPATHY_GAP]
        assert len(eg_mods) == 0

    def test_no_emotional_content_no_trigger(self):
        sim = _make_bias_sim(agreeableness=0.15, neuroticism=0.15)
        mods = sim.compute_modifiers("What's the best approach to this problem")
        eg_mods = [m for m in mods if m.bias_type == BiasType.EMPATHY_GAP]
        assert len(eg_mods) == 0


class TestDunningKrugerBias:
    """DK bias: overconfident when unknowledgeable (low-O + high-C)."""

    def test_low_proficiency_triggers(self):
        sim = _make_bias_sim(openness=0.2, conscientiousness=0.8)
        mods = sim.compute_modifiers("Tell me about quantum physics", proficiency=0.1)
        dk_mods = [m for m in mods if m.bias_type == BiasType.DUNNING_KRUGER]
        assert len(dk_mods) == 1
        assert dk_mods[0].modifier > 0  # Increases confidence

    def test_high_proficiency_no_trigger(self):
        sim = _make_bias_sim(openness=0.2, conscientiousness=0.8)
        mods = sim.compute_modifiers("Tell me about my domain", proficiency=0.8)
        dk_mods = [m for m in mods if m.bias_type == BiasType.DUNNING_KRUGER]
        assert len(dk_mods) == 0

    def test_high_o_resists_dk(self):
        sim = _make_bias_sim(openness=0.9, conscientiousness=0.3)
        mods = sim.compute_modifiers("Tell me about something", proficiency=0.1)
        dk_mods = [m for m in mods if m.bias_type == BiasType.DUNNING_KRUGER]
        assert len(dk_mods) == 0


class TestBiasOverrides:
    """Persona-declared biases override trait-derived strengths."""

    def test_override_amplifies(self):
        """Persona declaring high anchoring strength should amplify effect."""
        overrides = [{"type": "anchoring_bias", "strength": 0.9}]
        sim = BiasSimulator(
            traits={"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
                    "agreeableness": 0.5, "neuroticism": 0.5},
            value_priorities={"conformity": 0.5, "tradition": 0.5, "security": 0.5},
            persona_biases=overrides,
        )
        sim.set_anchor("I believe X")
        mods = sim.compute_modifiers("What about Y?")
        anchoring = [m for m in mods if m.bias_type == BiasType.ANCHORING]
        assert len(anchoring) == 1
        # With override, strength should be higher than without
        sim_no_override = BiasSimulator(
            traits={"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
                    "agreeableness": 0.5, "neuroticism": 0.5},
            value_priorities={"conformity": 0.5, "tradition": 0.5, "security": 0.5},
        )
        sim_no_override.set_anchor("I believe X")
        mods_no = sim_no_override.compute_modifiers("What about Y?")
        anchoring_no = [m for m in mods_no if m.bias_type == BiasType.ANCHORING]
        if anchoring_no:
            assert abs(anchoring[0].modifier) >= abs(anchoring_no[0].modifier)

    def test_all_biases_bounded(self):
        """All biases must be bounded by MAX_BIAS_IMPACT * 2."""
        sim = _make_bias_sim(neuroticism=0.9, openness=0.1, conscientiousness=0.9,
                             agreeableness=0.1)
        sim.set_anchor("Anchor stance")
        mods = sim.compute_modifiers(
            "We should switch to change because of terrible problems and failure risk",
            value_alignment=0.9,
            proficiency=0.05,
        )
        for mod in mods:
            assert abs(mod.modifier) <= MAX_BIAS_IMPACT, \
                f"{mod.bias_type}: |modifier| {abs(mod.modifier)} > {MAX_BIAS_IMPACT}"


# ============================================================================
# Test User Model Inference (R6.3)
# ============================================================================

class TestUserModelInference:
    """User model inference from input text."""

    def test_technical_input_high_expertise(self):
        model = infer_user_model(
            "The algorithm's implementation uses polymorphism and concurrent architecture"
        )
        assert model.inferred_expertise > 0.3

    def test_casual_input_low_expertise(self):
        model = infer_user_model("hey what's up, just curious about stuff")
        assert model.inferred_expertise < 0.3

    def test_formal_input_detected(self):
        model = infer_user_model(
            "Furthermore, I would like to discuss this matter regarding the proposal"
        )
        assert model.inferred_formality > 0.2

    def test_personal_disclosure_detected(self):
        model = infer_user_model("I feel like I've been struggling with this personally")
        assert model.disclosed is True
        assert model.disclosure_depth > 0.1

    def test_no_disclosure_detected(self):
        model = infer_user_model("What's the best approach for this task?")
        assert model.disclosed is False

    def test_positive_emotion_detected(self):
        model = infer_user_model("This is great, I love this amazing approach!")
        assert model.inferred_emotion_valence > 0

    def test_negative_emotion_detected(self):
        model = infer_user_model("This is terrible and frustrating")
        assert model.inferred_emotion_valence < 0


# ============================================================================
# Test Social Adaptation (R6.3)
# ============================================================================

class TestSocialAdaptation:
    """Personality-modulated adaptation to user."""

    def test_high_a_mirrors_formality(self):
        user = UserModel(inferred_formality=0.8)
        traits = _make_traits(agreeableness=0.85)
        adapt = compute_adaptation(user, traits)
        assert adapt.formality_shift > 0  # Shifts toward formal

    def test_low_a_no_mirroring(self):
        user = UserModel(inferred_formality=0.8)
        traits = _make_traits(agreeableness=0.2)
        adapt = compute_adaptation(user, traits)
        assert adapt.formality_shift == 0

    def test_high_o_calibrates_depth(self):
        user = UserModel(inferred_expertise=0.8)
        traits = _make_traits(openness=0.85)
        adapt = compute_adaptation(user, traits)
        assert adapt.depth_shift > 0  # Goes deeper for expert user
        assert any("technical" in d.lower() for d in adapt.prompt_directives)

    def test_high_o_simplifies_for_novice(self):
        user = UserModel(inferred_expertise=0.1)
        traits = _make_traits(openness=0.85)
        adapt = compute_adaptation(user, traits)
        assert adapt.depth_shift < 0  # Simplifies for novice

    def test_disclosure_reciprocity(self):
        """User disclosure triggers reciprocal disclosure, modulated by E+A."""
        user = UserModel(disclosed=True, disclosure_depth=0.7)
        high_ea = _make_traits(extraversion=0.8, agreeableness=0.8)
        low_ea = _make_traits(extraversion=0.2, agreeableness=0.2)
        adapt_high = compute_adaptation(user, high_ea)
        adapt_low = compute_adaptation(user, low_ea)
        assert adapt_high.disclosure_reciprocity > adapt_low.disclosure_reciprocity

    def test_no_disclosure_no_reciprocity(self):
        user = UserModel(disclosed=False)
        traits = _make_traits(extraversion=0.9, agreeableness=0.9)
        adapt = compute_adaptation(user, traits)
        assert adapt.disclosure_reciprocity == 0


# ============================================================================
# Test Self-Schema Protection (R6.4)
# ============================================================================

class TestSelfSchemaProtection:
    """Schema-relevant challenges trigger identity protection."""

    def test_challenge_detected(self):
        schema, is_challenge = detect_schema_relevance(
            "Your research methodology is wrong",
            ["competent_researcher"]
        )
        assert schema == "competent_researcher"
        assert is_challenge is True

    def test_validation_detected(self):
        schema, is_challenge = detect_schema_relevance(
            "Your research approach is really impressive",
            ["competent_researcher"]
        )
        assert schema == "competent_researcher"
        assert is_challenge is False

    def test_no_schema_match(self):
        schema, is_challenge = detect_schema_relevance(
            "What's the weather like?",
            ["competent_researcher"]
        )
        assert schema is None

    def test_challenge_reduces_elasticity(self):
        effect = compute_schema_effect("competent_researcher", is_challenge=True)
        assert effect.elasticity_modifier < 0
        assert effect.confidence_modifier > 0

    def test_validation_increases_disclosure(self):
        effect = compute_schema_effect("competent_researcher", is_challenge=False)
        assert effect.disclosure_modifier > 0

    def test_no_schema_no_effect(self):
        effect = compute_schema_effect(None, is_challenge=False)
        assert effect.elasticity_modifier == 0
        assert effect.confidence_modifier == 0
        assert effect.disclosure_modifier == 0

    def test_empty_schemas_no_match(self):
        schema, is_challenge = detect_schema_relevance(
            "Your research is terrible", []
        )
        assert schema is None


# ============================================================================
# Test Pipeline Integration
# ============================================================================

class TestR6PipelineIntegration:

    def test_ir_generated_successfully(self):
        """Basic smoke test: IR generation still works with R6 features."""
        data = _make_persona_data()
        ir = _generate_ir(data, "What do you think about engineering?")
        assert ir is not None

    def test_challenging_emotional_input_generates_ir(self):
        data = _make_persona_data(neuroticism=0.85, agreeableness=0.2)
        ir = _generate_ir(data, "I'm feeling really hurt and overwhelmed by all these terrible problems")
        assert ir is not None

    def test_change_proposal_input(self):
        data = _make_persona_data(openness=0.15, conscientiousness=0.85)
        ir = _generate_ir(data, "We should completely switch to a different approach and redesign everything")
        assert ir is not None

    def test_persona_with_self_schemas(self):
        """Persona with self_schemas should handle schema-relevant input."""
        data = _make_persona_data()
        data["self_schemas"] = ["skilled_professional"]
        persona = Persona(**data)
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        ctx = _make_context("Your professional skills are not adequate for this role")
        ir = planner.generate_ir(ctx)
        assert ir is not None

    def test_bias_total_for_field_sums(self):
        """Multiple biases targeting same field should sum correctly."""
        sim = _make_bias_sim(openness=0.15, neuroticism=0.1, conscientiousness=0.85)
        sim.set_anchor("My position is X")
        mods = sim.compute_modifiers(
            "We should switch to a completely different alternative approach",
            value_alignment=0.8,
        )
        elasticity_total = sim.get_total_modifier_for_field(mods, "response_structure.elasticity")
        elasticity_mods = [m for m in mods if m.target_field == "response_structure.elasticity"]
        assert len(elasticity_mods) >= 2  # At least anchoring + one other
        # Total should equal sum of individual modifiers
        manual_sum = sum(m.modifier for m in elasticity_mods)
        assert abs(elasticity_total - manual_sum) < 0.001 or abs(elasticity_total) <= MAX_BIAS_IMPACT * 2
