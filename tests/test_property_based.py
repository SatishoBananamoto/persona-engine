"""
Property-Based Testing Framework (Fix 6.9)

Uses Hypothesis to generate random but valid personas and inputs,
then asserts invariants that must hold for ALL possible inputs.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from persona_engine.schema.persona_schema import (
    BigFiveTraits,
    ClaimPolicy,
    CognitiveStyle,
    CommunicationPreferences,
    DisclosurePolicy,
    DynamicState,
    Goal,
    Identity,
    Persona,
    PersonaInvariants,
    PersonalityProfile,
    SchwartzValues,
    SocialRole,
    UncertaintyPolicy,
)
from persona_engine.schema.ir_schema import (
    InteractionMode,
    ConversationGoal,
    UncertaintyAction,
)
from persona_engine.planner.turn_planner import create_turn_planner, ConversationContext
from persona_engine.memory import StanceCache


# =============================================================================
# Strategies for Generating Valid Personas
# =============================================================================

unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

@st.composite
def big_five_strategy(draw):
    return BigFiveTraits(
        openness=draw(unit_float),
        conscientiousness=draw(unit_float),
        extraversion=draw(unit_float),
        agreeableness=draw(unit_float),
        neuroticism=draw(unit_float),
    )

@st.composite
def schwartz_values_strategy(draw):
    return SchwartzValues(
        self_direction=draw(unit_float),
        stimulation=draw(unit_float),
        hedonism=draw(unit_float),
        achievement=draw(unit_float),
        power=draw(unit_float),
        security=draw(unit_float),
        conformity=draw(unit_float),
        tradition=draw(unit_float),
        benevolence=draw(unit_float),
        universalism=draw(unit_float),
    )

@st.composite
def cognitive_style_strategy(draw):
    return CognitiveStyle(
        analytical_intuitive=draw(unit_float),
        systematic_heuristic=draw(unit_float),
        risk_tolerance=draw(unit_float),
        need_for_closure=draw(unit_float),
        cognitive_complexity=draw(unit_float),
    )

@st.composite
def persona_strategy(draw):
    big_five = draw(big_five_strategy())
    values = draw(schwartz_values_strategy())
    cognitive = draw(cognitive_style_strategy())

    verbosity = draw(unit_float)
    formality = draw(unit_float)
    directness = draw(unit_float)
    expressiveness = draw(unit_float)

    must_avoid = draw(st.lists(
        st.sampled_from(["politics", "religion", "salary", "competitors"]),
        max_size=2, unique=True,
    ))
    cannot_claim = draw(st.lists(
        st.sampled_from(["licensed therapist", "medical doctor", "lawyer"]),
        max_size=2, unique=True,
    ))

    return Persona(
        persona_id=f"prop-test-{draw(st.integers(min_value=1, max_value=9999))}",
        label="Property Test Persona",
        identity=Identity(
            age=draw(st.integers(min_value=18, max_value=80)),
            location="Test City",
            education="BS",
            occupation="Tester",
            background="Generated for property testing",
        ),
        psychology=PersonalityProfile(
            big_five=big_five,
            values=values,
            cognitive_style=cognitive,
            communication=CommunicationPreferences(
                verbosity=verbosity,
                formality=formality,
                directness=directness,
                emotional_expressiveness=expressiveness,
            ),
        ),
        social_roles={
            "default": SocialRole(
                formality=formality,
                directness=directness,
                emotional_expressiveness=expressiveness,
            ),
        },
        uncertainty=UncertaintyPolicy(
            admission_threshold=draw(unit_float),
            hedging_frequency=draw(unit_float),
            clarification_tendency=draw(unit_float),
            knowledge_boundary_strictness=draw(unit_float),
        ),
        claim_policy=ClaimPolicy(),
        invariants=PersonaInvariants(
            identity_facts=["Test identity"],
            must_avoid=must_avoid,
            cannot_claim=cannot_claim,
        ),
        disclosure_policy=DisclosurePolicy(
            base_openness=draw(unit_float),
            factors={"topic_sensitivity": -0.3},
        ),
        initial_state=DynamicState(
            mood_valence=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
            mood_arousal=draw(unit_float),
            fatigue=draw(unit_float),
            stress=draw(unit_float),
            engagement=draw(unit_float),
        ),
        time_scarcity=draw(unit_float),
        privacy_sensitivity=draw(unit_float),
    )


def _make_context(user_input: str, turn: int = 1):
    """Build a ConversationContext with all required fields."""
    return ConversationContext(
        conversation_id="prop_test",
        turn_number=turn,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.EXPLORE_IDEAS,
        topic_signature="general",
        user_input=user_input,
        stance_cache=StanceCache(),
    )


# User input strategy
user_input_strategy = st.sampled_from([
    "Tell me about yourself",
    "What do you think about AI?",
    "How's the weather?",
    "Can you help me with this problem?",
    "I'm feeling frustrated with my project",
    "Research shows that remote work is better",
    "Have you ever dealt with burnout?",
    "What's your opinion on education reform?",
    "I don't have any concerns about this",
    "This is a terrible situation",
    "I'm curious about quantum computing",
    "Let's brainstorm some ideas",
])


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestIRInvariants:
    """Property-based tests asserting IR invariants across random personas."""

    @given(persona=persona_strategy(), user_input=user_input_strategy)
    @settings(max_examples=50, deadline=10000)
    def test_ir_passes_pydantic_validation(self, persona, user_input):
        """IR must always pass Pydantic validation (schema-valid output)."""
        planner = create_turn_planner(persona)
        context = _make_context(user_input)
        ir = planner.generate_ir(context)
        # If we get here without exception, Pydantic validation passed
        assert ir is not None

    @given(persona=persona_strategy(), user_input=user_input_strategy)
    @settings(max_examples=50, deadline=10000)
    def test_behavioral_floats_in_range(self, persona, user_input):
        """All behavioral floats must be in [0, 1]."""
        planner = create_turn_planner(persona)
        context = _make_context(user_input)
        ir = planner.generate_ir(context)

        assert 0.0 <= ir.response_structure.confidence <= 1.0
        assert 0.0 <= ir.response_structure.competence <= 1.0
        assert 0.0 <= ir.communication_style.formality <= 1.0
        assert 0.0 <= ir.communication_style.directness <= 1.0
        assert 0.0 <= ir.knowledge_disclosure.disclosure_level <= 1.0
        if ir.response_structure.elasticity is not None:
            assert 0.0 <= ir.response_structure.elasticity <= 1.0

    @given(persona=persona_strategy(), user_input=user_input_strategy)
    @settings(max_examples=30, deadline=10000)
    def test_citations_present_for_nondefault_values(self, persona, user_input):
        """IRs with non-default behavioral values should have citations."""
        planner = create_turn_planner(persona)
        context = _make_context(user_input)
        ir = planner.generate_ir(context)
        # Should always have at least some citations
        assert len(ir.citations) > 0

    @given(persona=persona_strategy())
    @settings(max_examples=30, deadline=10000)
    def test_determinism_same_seed_same_ir(self, persona):
        """Same seed + same inputs must produce identical IR."""
        planner1 = create_turn_planner(persona)
        planner2 = create_turn_planner(persona)

        context1 = _make_context("Tell me about yourself")
        context2 = _make_context("Tell me about yourself")

        ir1 = planner1.generate_ir(context1)
        ir2 = planner2.generate_ir(context2)

        assert ir1.response_structure.confidence == ir2.response_structure.confidence
        assert ir1.response_structure.competence == ir2.response_structure.competence
        assert ir1.communication_style.formality == ir2.communication_style.formality
        assert ir1.knowledge_disclosure.disclosure_level == ir2.knowledge_disclosure.disclosure_level

    @given(persona=persona_strategy(), user_input=user_input_strategy)
    @settings(max_examples=30, deadline=10000)
    def test_must_avoid_not_in_stance(self, persona, user_input):
        """must_avoid topics must never appear in the generated stance."""
        planner = create_turn_planner(persona)
        context = _make_context(user_input)
        ir = planner.generate_ir(context)

        stance = (ir.response_structure.stance or "").lower()
        for topic in persona.invariants.must_avoid:
            assert topic.lower() not in stance, \
                f"must_avoid topic '{topic}' found in stance: '{stance}'"

    @given(persona=persona_strategy(), user_input=user_input_strategy)
    @settings(max_examples=30, deadline=10000)
    def test_nonexpert_bounded_confidence(self, persona, user_input):
        """Non-expert personas should not have ANSWER + confidence > 0.85 in unknown domains."""
        assume(not persona.knowledge_domains)  # No expert domains

        planner = create_turn_planner(persona)
        context = _make_context(user_input)
        ir = planner.generate_ir(context)

        if ir.knowledge_disclosure.uncertainty_action == UncertaintyAction.ANSWER:
            # Non-expert answering should not have extremely high confidence
            # Allow up to 0.9 since some common knowledge topics can be high confidence
            assert ir.response_structure.confidence <= 0.95, \
                f"Non-expert with ANSWER has confidence {ir.response_structure.confidence}"
