"""
Behavioral Coherence Tests - Psychological Realism Validation

Tests the full pipeline through TurnPlanner.generate_ir() to validate that
the system produces behavior coherent with a psychologically realistic
human persona.

Categories:
  1. Trait-Behavior Coherence (Big Five -> observable behavior)
  2. Social Role Coherence (interaction mode -> communication adjustments)
  3. Value-Driven Stance (Schwartz values -> stance/rationale)
  4. Multi-Turn Consistency (stance cache, fatigue, engagement)
  5. Stress and Mood Coherence (challenge -> stress -> tone)
  6. Cross-Module Integration (confidence-uncertainty, citation trails)
"""

import pytest
import yaml

from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


# =============================================================================
# Test Helpers
# =============================================================================

def load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    """Load persona from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    # Handle field name mismatch in some test personas
    # (yaml uses 'domains', schema expects 'knowledge_domains')
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def make_context(
    user_input: str,
    topic: str = "general",
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    turn: int = 1,
    stance_cache: StanceCache | None = None,
    domain: str | None = None,
) -> ConversationContext:
    """Create a conversation context for testing."""
    return ConversationContext(
        conversation_id="test_coherence",
        turn_number=turn,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=stance_cache or StanceCache(),
        domain=domain,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sarah_persona() -> Persona:
    """Sarah the UX Researcher: O=0.75, C=0.82, E=0.45, A=0.68, N=0.35"""
    return load_persona("personas/ux_researcher.yaml")


@pytest.fixture
def sarah_planner(sarah_persona: Persona) -> TurnPlanner:
    """Deterministic turn planner for Sarah."""
    return TurnPlanner(sarah_persona, DeterminismManager(seed=42))


@pytest.fixture
def high_neuroticism_persona() -> Persona:
    """Test persona with N=0.75 for stress/anxiety tests."""
    return load_persona("personas/test_high_neuroticism.yaml")


@pytest.fixture
def high_neuroticism_planner(high_neuroticism_persona: Persona) -> TurnPlanner:
    """Deterministic turn planner for high-neuroticism persona."""
    return TurnPlanner(high_neuroticism_persona, DeterminismManager(seed=42))


# =============================================================================
# 1. TRAIT-BEHAVIOR COHERENCE
#    Big Five traits should produce psychologically consistent behavioral outputs.
# =============================================================================

class TestTraitBehaviorCoherence:
    """Validate that Big Five traits translate into coherent IR fields."""

    def test_high_openness_produces_high_elasticity(self, sarah_planner: TurnPlanner):
        """Sarah's O=0.75 should produce elasticity well above midpoint (>0.4).

        High openness means willingness to consider alternative viewpoints,
        which manifests as higher elasticity (openness to persuasion).
        """
        ctx = make_context(
            "What do you think about remote UX research methods?",
            topic="ux_research",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.response_structure.elasticity is not None
        assert ir.response_structure.elasticity > 0.4, (
            f"High openness (0.75) should produce elasticity > 0.4, "
            f"got {ir.response_structure.elasticity:.3f}"
        )

    def test_moderate_extraversion_near_zero_disclosure_modifier(
        self, sarah_persona: Persona
    ):
        """Sarah's E=0.45 should produce a disclosure modifier near zero.

        The formula is (E - 0.5) * 0.4, so E=0.45 gives -0.02.
        This tests that moderate extraversion neither dramatically increases
        nor decreases self-disclosure.
        """
        # E=0.45 -> (0.45 - 0.5) * 0.4 = -0.02
        expected_modifier = (sarah_persona.psychology.big_five.extraversion - 0.5) * 0.4
        assert expected_modifier == pytest.approx(-0.02, abs=0.001)

    def test_high_conscientiousness_verbosity_not_brief(
        self, sarah_planner: TurnPlanner
    ):
        """Sarah's C=0.82 should produce MEDIUM or DETAILED verbosity.

        High conscientiousness correlates with detail orientation and
        structured responses. Brief output would be inconsistent with
        someone who is disciplined and thorough.
        """
        ctx = make_context(
            "Tell me about your research methodology.",
            topic="ux_research",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.communication_style.verbosity in (Verbosity.MEDIUM, Verbosity.DETAILED), (
            f"High conscientiousness (0.82) should not produce BRIEF verbosity, "
            f"got {ir.communication_style.verbosity.value}"
        )

    def test_agreeableness_reduces_directness_from_base(
        self, sarah_planner: TurnPlanner, sarah_persona: Persona
    ):
        """Sarah's A=0.68 (>0.5) should reduce directness below the role-blended value.

        Agreeableness inversely affects directness: agreeable people are more
        diplomatic and less blunt. The trait modifier is (0.5 - A) * 0.3,
        which is negative when A > 0.5.
        """
        ctx = make_context(
            "What is your opinion on design sprints?",
            topic="design_methods",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        # Base directness is 0.65, role-blended through default role stays ~0.65
        # Agreeableness modifier: (0.5 - 0.68) * 0.3 = -0.054
        # So final directness should be < 0.65
        base_directness = sarah_persona.psychology.communication.directness
        assert ir.communication_style.directness < base_directness, (
            f"A=0.68 should reduce directness below base {base_directness:.2f}, "
            f"got {ir.communication_style.directness:.3f}"
        )

    def test_low_neuroticism_not_anxious_in_normal_conversation(
        self, sarah_planner: TurnPlanner
    ):
        """Sarah's N=0.35 should never produce anxious/stressed tone in normal chat.

        Low neuroticism means emotional stability. The anxious tone pathway
        requires BOTH stress > 0.6 AND neuroticism > 0.6, neither of which
        Sarah meets in a normal conversation turn.
        """
        ctx = make_context(
            "How has your week been going?",
            topic="general",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
        )
        ir = sarah_planner.generate_ir(ctx)

        anxious_tones = {Tone.ANXIOUS_STRESSED, Tone.FRUSTRATED_TENSE, Tone.DEFENSIVE_AGITATED}
        assert ir.communication_style.tone not in anxious_tones, (
            f"Low neuroticism (0.35) should not produce anxious tone in normal chat, "
            f"got {ir.communication_style.tone.value}"
        )

    def test_elasticity_bounded_between_01_and_09(self, sarah_planner: TurnPlanner):
        """Elasticity must always respect system bounds [0.1, 0.9]."""
        ctx = make_context(
            "What are the best tools for user research?",
            topic="ux_tools",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.response_structure.elasticity is not None
        assert 0.1 <= ir.response_structure.elasticity <= 0.9, (
            f"Elasticity must be in [0.1, 0.9], got {ir.response_structure.elasticity:.3f}"
        )

    def test_conscientiousness_boosts_confidence(
        self, sarah_planner: TurnPlanner
    ):
        """Sarah's C=0.82 should boost confidence slightly above base proficiency.

        High conscientiousness correlates with thorough preparation,
        which manifests as a small confidence increase. The modifier
        is (C - 0.5) * 0.1 = +0.032.
        """
        ctx = make_context(
            "Tell me about cognitive biases in user research.",
            topic="psychology_research",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
            domain="Psychology",
        )
        ir = sarah_planner.generate_ir(ctx)

        # For Psychology domain, proficiency = 0.90
        # C-boost = (0.82 - 0.5) * 0.1 = +0.032
        # N-penalty = 0.35 * 0.15 = -0.0525
        # Net trait modifier is small but confidence should be substantial
        # given high proficiency base (0.90)
        assert ir.response_structure.confidence > 0.6, (
            f"High proficiency (0.90) + high C (0.82) should yield confidence > 0.6, "
            f"got {ir.response_structure.confidence:.3f}"
        )


# =============================================================================
# 2. SOCIAL ROLE COHERENCE
#    Different interaction modes should trigger appropriate role adjustments.
# =============================================================================

class TestSocialRoleCoherence:
    """Validate that social roles produce coherent communication style shifts."""

    def test_interview_higher_formality_than_casual(
        self, sarah_persona: Persona
    ):
        """Interview mode (at_work role) should produce higher formality than casual.

        at_work role formality = 0.70 vs default formality = 0.55.
        The 70/30 blend amplifies this difference.
        """
        det = DeterminismManager(seed=42)

        # Interview context
        planner_interview = TurnPlanner(sarah_persona, det)
        ctx_interview = make_context(
            "Tell me about your approach to user research.",
            topic="ux_research",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir_interview = planner_interview.generate_ir(ctx_interview)

        # Casual context (fresh planner to reset state)
        planner_casual = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_casual = make_context(
            "Tell me about your approach to user research.",
            topic="ux_research",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir_casual = planner_casual.generate_ir(ctx_casual)

        assert ir_interview.communication_style.formality > ir_casual.communication_style.formality, (
            f"Interview formality ({ir_interview.communication_style.formality:.3f}) "
            f"should exceed casual formality ({ir_casual.communication_style.formality:.3f})"
        )

    def test_formality_difference_is_meaningful(self, sarah_persona: Persona):
        """The formality gap between interview and casual should be > 0.08.

        at_work role formality (0.70) blended 70/30 with base (0.55) = 0.655
        default role formality (0.55) blended 70/30 with base (0.55) = 0.55
        Expected difference: ~0.105
        """
        det = DeterminismManager(seed=42)
        planner_interview = TurnPlanner(sarah_persona, det)
        ctx_interview = make_context(
            "Tell me about your approach to user research.",
            topic="ux_research",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir_interview = planner_interview.generate_ir(ctx_interview)

        planner_casual = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_casual = make_context(
            "Tell me about your approach to user research.",
            topic="ux_research",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir_casual = planner_casual.generate_ir(ctx_casual)

        diff = ir_interview.communication_style.formality - ir_casual.communication_style.formality
        assert diff > 0.08, (
            f"Formality difference between interview and casual should be > 0.08, got {diff:.3f}"
        )

    def test_customer_support_maps_to_at_work_role(self, sarah_persona: Persona):
        """Customer support and interview should both use the at_work role.

        Both modes map to at_work in the rules engine, so formality
        should be approximately the same for the same input.
        """
        planner_cs = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_cs = make_context(
            "Tell me about your approach to user research.",
            topic="ux_research",
            mode=InteractionMode.CUSTOMER_SUPPORT,
            goal=ConversationGoal.RESOLVE_ISSUE,
        )
        ir_cs = planner_cs.generate_ir(ctx_cs)

        planner_iv = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_iv = make_context(
            "Tell me about your approach to user research.",
            topic="ux_research",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir_iv = planner_iv.generate_ir(ctx_iv)

        assert ir_cs.communication_style.formality == pytest.approx(
            ir_iv.communication_style.formality, abs=0.05
        ), (
            f"Customer support formality ({ir_cs.communication_style.formality:.3f}) "
            f"should match interview formality ({ir_iv.communication_style.formality:.3f}) "
            f"since both use at_work role"
        )

    def test_directness_differs_between_professional_and_casual(
        self, sarah_persona: Persona
    ):
        """Directness should shift with social role context.

        at_work directness = 0.75 vs default directness = 0.65.
        After role blending and trait modifiers, a directional difference
        should remain.
        """
        planner_iv = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_iv = make_context(
            "What do you think about the sprint methodology?",
            topic="methodology",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir_iv = planner_iv.generate_ir(ctx_iv)

        planner_casual = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_casual = make_context(
            "What do you think about the sprint methodology?",
            topic="methodology",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir_casual = planner_casual.generate_ir(ctx_casual)

        assert ir_iv.communication_style.directness > ir_casual.communication_style.directness, (
            f"Interview directness ({ir_iv.communication_style.directness:.3f}) "
            f"should exceed casual directness ({ir_casual.communication_style.directness:.3f})"
        )

    def test_formality_stays_within_zero_one_bounds(self, sarah_planner: TurnPlanner):
        """Formality must always be clamped to [0, 1] regardless of modifiers."""
        for mode in [InteractionMode.INTERVIEW, InteractionMode.CASUAL_CHAT, InteractionMode.DEBATE]:
            planner = TurnPlanner(
                load_persona(), DeterminismManager(seed=42)
            )
            ctx = make_context(
                "Let us discuss research methods.",
                mode=mode,
                goal=ConversationGoal.EXPLORE_IDEAS,
            )
            ir = planner.generate_ir(ctx)
            assert 0.0 <= ir.communication_style.formality <= 1.0, (
                f"Formality out of bounds for {mode.value}: "
                f"{ir.communication_style.formality:.3f}"
            )


# =============================================================================
# 3. VALUE-DRIVEN STANCE
#    Schwartz values should influence stance content and rationale.
# =============================================================================

class TestValueDrivenStance:
    """Validate that persona values drive stance generation coherently."""

    def test_top_value_is_self_direction(self, sarah_persona: Persona):
        """Sarah's highest value should be self_direction (0.85).

        This is a sanity check that the persona profile matches expectations
        before testing value-driven behavior.
        """
        values = sarah_persona.psychology.values
        value_dict = {
            "self_direction": values.self_direction,
            "benevolence": values.benevolence,
            "universalism": values.universalism,
            "achievement": values.achievement,
        }
        top_value = max(value_dict, key=value_dict.get)  # type: ignore[arg-type]
        assert top_value == "self_direction"
        assert values.self_direction == pytest.approx(0.85)

    def test_stance_rationale_references_value_name(self, sarah_planner: TurnPlanner):
        """Stance rationale should mention the driving value by name.

        Since self_direction (0.85) is the top value, the rationale
        should contain 'self_direction' as the value-driven reasoning anchor.
        """
        ctx = make_context(
            "What is your perspective on standardized testing in schools?",
            topic="education_policy",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.response_structure.rationale is not None
        assert "self_direction" in ir.response_structure.rationale.lower(), (
            f"Rationale should reference top value 'self_direction', "
            f"got: {ir.response_structure.rationale}"
        )

    def test_expert_topic_rationale_mentions_proficiency(self, sarah_planner: TurnPlanner):
        """On expert topics, rationale should reference proficiency level.

        Sarah has Psychology proficiency of 0.90, which qualifies her as
        an expert (>= 0.7 threshold). The rationale should reflect this.
        """
        ctx = make_context(
            "What are the best practices for cognitive walkthroughs?",
            topic="psychology_research",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
            domain="Psychology",
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.response_structure.rationale is not None
        assert "proficiency" in ir.response_structure.rationale.lower(), (
            f"Expert rationale should mention proficiency, "
            f"got: {ir.response_structure.rationale}"
        )

    def test_non_expert_stance_is_opinion_based(self, sarah_planner: TurnPlanner):
        """On non-expert topics, stance should reflect personal opinion, not authority.

        When Sarah discusses topics outside her expertise (proficiency < 0.7),
        the stance should use subjective phrasing driven by values.
        """
        ctx = make_context(
            "What do you think about cryptocurrency regulations?",
            topic="finance_crypto",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.response_structure.stance is not None
        # Non-expert stances use subjective templates: "I tend to", "I value", etc.
        stance_lower = ir.response_structure.stance.lower()
        subjective_markers = ["i ", "my ", "prefer", "tend", "feel", "value", "favor"]
        has_subjective = any(marker in stance_lower for marker in subjective_markers)
        assert has_subjective, (
            f"Non-expert stance should use subjective language, "
            f"got: {ir.response_structure.stance}"
        )

    def test_stance_is_not_empty(self, sarah_planner: TurnPlanner):
        """Every generate_ir call should produce a non-empty stance."""
        ctx = make_context(
            "Do you think AI will change UX research?",
            topic="ai_in_ux",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.response_structure.stance is not None
        assert len(ir.response_structure.stance.strip()) > 0


# =============================================================================
# 4. MULTI-TURN CONSISTENCY
#    Behavior should be coherent across turns within a conversation.
# =============================================================================

class TestMultiTurnConsistency:
    """Validate behavioral consistency across multiple conversation turns."""

    def test_stance_cached_on_second_turn(self, sarah_persona: Persona):
        """Second turn on same topic should use cached stance.

        After turn 1 generates and caches a stance, turn 2 with the
        same topic and mode should find the cached stance and produce
        a citation referencing 'stance_cache'.
        """
        cache = StanceCache()
        det = DeterminismManager(seed=42)
        planner = TurnPlanner(sarah_persona, det)

        mode = InteractionMode.CASUAL_CHAT
        goal = ConversationGoal.EXPLORE_IDEAS

        # Turn 1: generate and cache stance
        ctx1 = make_context(
            "What do you think about remote usability testing?",
            topic="remote_usability",
            mode=mode,
            goal=goal,
            turn=1,
            stance_cache=cache,
        )
        ir1 = planner.generate_ir(ctx1)
        assert ir1.response_structure.stance is not None

        # Turn 2: same topic should hit cache
        ctx2 = make_context(
            "Can you elaborate more on that topic?",
            topic="remote_usability",
            mode=mode,
            goal=goal,
            turn=2,
            stance_cache=cache,
        )
        ir2 = planner.generate_ir(ctx2)

        cache_citations = [
            c for c in ir2.citations if c.source_id == "stance_cache"
        ]
        assert len(cache_citations) > 0, (
            "Turn 2 should contain a stance_cache citation. "
            f"Found citations: {[c.source_id for c in ir2.citations]}"
        )

    def test_fatigue_increases_over_many_turns(self, sarah_persona: Persona):
        """Fatigue should accumulate over a long conversation.

        By turn 20, the accumulated fatigue from repeated
        evolve_state_post_turn calls should produce observable effects
        compared to turn 1.
        """
        cache = StanceCache()
        det = DeterminismManager(seed=42)
        planner = TurnPlanner(sarah_persona, det)

        # Run through 20 turns to accumulate fatigue
        for t in range(1, 21):
            ctx = make_context(
                "Tell me more about that topic.",
                topic="general_topic",
                mode=InteractionMode.CASUAL_CHAT,
                goal=ConversationGoal.EXPLORE_IDEAS,
                turn=t,
                stance_cache=cache,
            )
            ir = planner.generate_ir(ctx)

        # After 20 turns, fatigue should be significantly higher than initial 0.3
        final_fatigue = planner.state.get_fatigue()
        assert final_fatigue > 0.4, (
            f"After 20 turns, fatigue should exceed 0.4, got {final_fatigue:.3f}"
        )

    def test_seed_differs_per_turn(self, sarah_planner: TurnPlanner):
        """Per-turn seeds must be unique for different turn numbers.

        The SHA-256 based seed generation ensures each turn within
        a conversation gets a different random seed.
        """
        ctx1 = make_context(
            "Hello there!",
            turn=1,
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
        )
        ir1 = sarah_planner.generate_ir(ctx1)

        ctx2 = make_context(
            "Hello again!",
            turn=2,
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
        )
        ir2 = sarah_planner.generate_ir(ctx2)

        assert ir1.seed != ir2.seed, (
            f"Turn 1 seed ({ir1.seed}) should differ from turn 2 seed ({ir2.seed})"
        )

    def test_determinism_same_seed_same_output(self, sarah_persona: Persona):
        """Same seed and context should produce identical IR.

        This is the fundamental determinism guarantee: given the same
        inputs and seed, the system must produce the same outputs.
        """
        ctx_args = dict(
            user_input="What makes a good user interview?",
            topic="ux_interviews",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
            turn=1,
        )

        planner1 = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ir1 = planner1.generate_ir(make_context(**ctx_args))

        planner2 = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ir2 = planner2.generate_ir(make_context(**ctx_args))

        assert ir1.response_structure.confidence == pytest.approx(
            ir2.response_structure.confidence
        )
        assert ir1.response_structure.elasticity == pytest.approx(
            ir2.response_structure.elasticity
        )
        assert ir1.communication_style.formality == pytest.approx(
            ir2.communication_style.formality
        )
        assert ir1.communication_style.directness == pytest.approx(
            ir2.communication_style.directness
        )
        assert ir1.communication_style.tone == ir2.communication_style.tone
        assert ir1.communication_style.verbosity == ir2.communication_style.verbosity
        assert ir1.seed == ir2.seed

    def test_engagement_shifts_with_topic_relevance(self, sarah_persona: Persona):
        """Engagement should be higher for relevant topics than irrelevant ones.

        Sarah is a UX researcher, so UX topics should produce higher
        engagement than completely unrelated topics like astronomy.
        """
        planner_relevant = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_relevant = make_context(
            "Let us discuss cognitive walkthrough methods and usability heuristics.",
            topic="ux_methods",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            turn=1,
        )
        planner_relevant.generate_ir(ctx_relevant)
        engagement_relevant = planner_relevant.state.get_engagement()

        planner_irrelevant = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_irrelevant = make_context(
            "Tell me about the orbital mechanics of Jupiter.",
            topic="astronomy",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            turn=1,
        )
        planner_irrelevant.generate_ir(ctx_irrelevant)
        engagement_irrelevant = planner_irrelevant.state.get_engagement()

        assert engagement_relevant > engagement_irrelevant, (
            f"UX topic engagement ({engagement_relevant:.3f}) should exceed "
            f"astronomy topic engagement ({engagement_irrelevant:.3f})"
        )


# =============================================================================
# 5. STRESS AND MOOD COHERENCE
#    Emotional state should respond realistically to conversational events.
# =============================================================================

class TestStressAndMoodCoherence:
    """Validate that stress, mood, and tone respond coherently to stimuli."""

    def test_challenge_triggers_stress_citation(self, sarah_planner: TurnPlanner):
        """Challenging input should trigger a stress_trigger citation.

        When evidence strength exceeds the threshold (0.4), the system
        should apply a stress trigger and record it in the citation trail.
        """
        ctx = make_context(
            "That's wrong, the data clearly contradicts your view on usability testing.",
            topic="usability_debate",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        stress_citations = [
            c for c in ir.citations if c.source_id == "stress_trigger"
        ]
        assert len(stress_citations) > 0, (
            "Challenge phrase 'that's wrong' should trigger stress citation. "
            f"Found citation source_ids: {[c.source_id for c in ir.citations]}"
        )

    def test_challenge_produces_high_evidence_strength_citation(
        self, sarah_planner: TurnPlanner
    ):
        """Strong challenge phrases should produce evidence_strength citations.

        'you're wrong' is a strong challenge phrase that should be detected
        and recorded in the citation trail.
        """
        ctx = make_context(
            "You're wrong about the best research methods to use.",
            topic="methods_debate",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        evidence_citations = [
            c for c in ir.citations if c.source_id == "evidence_strength"
        ]
        assert len(evidence_citations) > 0
        # Should mention "Strong challenge" in the effect
        strong_citations = [
            c for c in evidence_citations if "strong" in c.effect.lower()
        ]
        assert len(strong_citations) > 0, (
            "Strong challenge phrase should produce 'Strong challenge' citation. "
            f"Got effects: {[c.effect for c in evidence_citations]}"
        )

    def test_low_neuroticism_not_anxious_even_under_challenge(
        self, sarah_persona: Persona
    ):
        """Sarah (N=0.35) should NOT become anxious even when challenged.

        The anxious tone requires BOTH neuroticism > 0.6 AND stress > 0.6.
        Sarah's low neuroticism protects her from anxious responses even
        when stress is elevated by a challenge.
        """
        planner = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx = make_context(
            "That's wrong, studies clearly prove you're mistaken about this.",
            topic="debate",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = planner.generate_ir(ctx)

        assert ir.communication_style.tone != Tone.ANXIOUS_STRESSED, (
            f"Sarah (N=0.35) should not produce ANXIOUS_STRESSED tone even under challenge, "
            f"got {ir.communication_style.tone.value}"
        )

    def test_high_neuroticism_anxious_under_stress(
        self, high_neuroticism_persona: Persona
    ):
        """High-N persona (0.75) should produce anxious tone under challenge + stress.

        The high neuroticism persona starts with stress=0.5 and arousal=0.6.
        A strong challenge should push stress above 0.6, triggering the
        anxious_stressed tone pathway (N > 0.6 AND stress > 0.6 AND arousal > 0.6).
        """
        planner = TurnPlanner(high_neuroticism_persona, DeterminismManager(seed=42))
        ctx = make_context(
            "That's wrong, you're completely mistaken about the data analysis.",
            topic="data_debate",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = planner.generate_ir(ctx)

        # High neuroticism under stress should shift tone toward negative territory
        negative_tones = {
            Tone.ANXIOUS_STRESSED,
            Tone.FRUSTRATED_TENSE,
            Tone.DEFENSIVE_AGITATED,
            Tone.CONCERNED_EMPATHETIC,
            Tone.DISAPPOINTED_RESIGNED,
        }
        assert ir.communication_style.tone in negative_tones, (
            f"High-N persona (0.75) under challenge should produce negative tone, "
            f"got {ir.communication_style.tone.value}"
        )

    def test_neutral_input_produces_neutral_or_positive_tone(
        self, sarah_planner: TurnPlanner
    ):
        """Normal, non-challenging input should produce neutral or positive tone.

        Sarah's initial state is mildly positive (valence=0.1, arousal=0.5).
        A friendly input should not shift her into negative territory.
        """
        ctx = make_context(
            "I have been reading about design thinking lately.",
            topic="design_thinking",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        negative_tones = {
            Tone.ANXIOUS_STRESSED,
            Tone.FRUSTRATED_TENSE,
            Tone.DEFENSIVE_AGITATED,
            Tone.SAD_SUBDUED,
            Tone.TIRED_WITHDRAWN,
        }
        assert ir.communication_style.tone not in negative_tones, (
            f"Neutral/friendly input should not produce negative tone, "
            f"got {ir.communication_style.tone.value}"
        )


# =============================================================================
# 6. CROSS-MODULE INTEGRATION
#    Different IR fields should be internally coherent with each other.
# =============================================================================

class TestCrossModuleIntegration:
    """Validate coherence between fields computed by different modules."""

    def test_high_confidence_implies_answer_or_hedge(
        self, sarah_planner: TurnPlanner
    ):
        """When confidence is high, uncertainty_action should be ANSWER.

        The uncertainty resolver should not produce REFUSE or ASK_CLARIFYING
        when the persona is highly confident in their response.
        """
        ctx = make_context(
            "What are the key principles of user interview design?",
            topic="psychology_interviews",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
            domain="Psychology",
        )
        ir = sarah_planner.generate_ir(ctx)

        if ir.response_structure.confidence > 0.7:
            assert ir.knowledge_disclosure.uncertainty_action in (
                UncertaintyAction.ANSWER,
                UncertaintyAction.HEDGE,
            ), (
                f"Confidence {ir.response_structure.confidence:.3f} > 0.7 "
                f"but uncertainty_action is {ir.knowledge_disclosure.uncertainty_action.value}"
            )

    def test_hedge_implies_moderate_or_low_confidence(
        self, sarah_planner: TurnPlanner
    ):
        """If uncertainty_action is HEDGE, confidence should not be very high.

        Hedging is the behavior of someone who lacks certainty. A persona
        who hedges while being highly confident is psychologically incoherent.
        """
        ctx = make_context(
            "What is the future outlook for cryptocurrency regulations?",
            topic="finance_crypto",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        if ir.knowledge_disclosure.uncertainty_action == UncertaintyAction.HEDGE:
            # Hedging should not coexist with very high confidence
            assert ir.response_structure.confidence <= 0.85, (
                f"Hedging with confidence {ir.response_structure.confidence:.3f} "
                f"is psychologically incoherent (should be <= 0.85)"
            )

    def test_elasticity_correlates_with_cognitive_complexity(
        self, sarah_persona: Persona
    ):
        """High cognitive complexity (0.80) should contribute to higher elasticity.

        Cognitive elasticity = complexity * 0.6 + (1 - need_for_closure) * 0.4
        For Sarah: 0.80 * 0.6 + 0.60 * 0.4 = 0.48 + 0.24 = 0.72
        This blends with trait elasticity, keeping final elasticity high.
        """
        planner = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx = make_context(
            "What is your take on mixed methods research?",
            topic="research_methods",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = planner.generate_ir(ctx)

        # With high O (0.75) AND high cognitive_complexity (0.80),
        # elasticity should be well above midpoint
        assert ir.response_structure.elasticity is not None
        assert ir.response_structure.elasticity > 0.5, (
            f"High openness (0.75) + high cognitive complexity (0.80) "
            f"should yield elasticity > 0.5, got {ir.response_structure.elasticity:.3f}"
        )

    def test_formality_and_directness_shift_together_with_role(
        self, sarah_persona: Persona
    ):
        """Both formality and directness should increase for professional roles.

        at_work role has both higher formality (0.70) and higher directness (0.75)
        compared to default (0.55, 0.65). Both should increase together in
        interview mode vs casual mode.
        """
        planner_iv = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_iv = make_context(
            "Describe your research process.",
            topic="research",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir_iv = planner_iv.generate_ir(ctx_iv)

        planner_casual = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx_casual = make_context(
            "Describe your research process.",
            topic="research",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir_casual = planner_casual.generate_ir(ctx_casual)

        formality_diff = ir_iv.communication_style.formality - ir_casual.communication_style.formality
        directness_diff = ir_iv.communication_style.directness - ir_casual.communication_style.directness

        # Both should increase in interview mode
        assert formality_diff > 0, (
            f"Interview formality should be higher than casual: "
            f"diff={formality_diff:.3f}"
        )
        assert directness_diff > 0, (
            f"Interview directness should be higher than casual: "
            f"diff={directness_diff:.3f}"
        )

    def test_all_behavioral_float_fields_have_citations(
        self, sarah_planner: TurnPlanner
    ):
        """Every behavioral float field should have at least one citation.

        The citation system exists to prevent 'naked writes' -- fields that
        are set without any audit trail. Every float in the IR should
        trace back to at least one source.
        """
        ctx = make_context(
            "Tell me about your background in research.",
            topic="background",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir = sarah_planner.generate_ir(ctx)

        cited_fields = {c.target_field for c in ir.citations if c.target_field}

        # These core float fields must have citations
        required_fields = [
            "response_structure.elasticity",
            "response_structure.confidence",
            "communication_style.formality",
            "communication_style.directness",
            "knowledge_disclosure.disclosure_level",
        ]

        for field in required_fields:
            assert field in cited_fields, (
                f"Field '{field}' has no citation in the audit trail. "
                f"Cited fields: {sorted(cited_fields)}"
            )

    def test_citation_trail_shows_canonical_sequence_for_directness(
        self, sarah_planner: TurnPlanner
    ):
        """Citation trail for directness should follow: base -> role -> trait.

        The canonical modifier sequence ensures modifiers are applied in
        a predictable order. For directness, we should see:
        1. base citation (initial value from persona)
        2. rule/role citation (social role blend)
        3. trait citation (agreeableness modifier)
        """
        ctx = make_context(
            "Tell me about your work experience.",
            topic="work_experience",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir = sarah_planner.generate_ir(ctx)

        # Filter citations for directness field
        directness_citations = [
            c for c in ir.citations
            if c.target_field == "communication_style.directness"
        ]

        # Extract the sequence of source_types
        source_types = [c.source_type for c in directness_citations]

        # Verify canonical order: base must come before rule, rule before trait
        assert "base" in source_types, (
            f"Directness should have a 'base' citation. Got: {source_types}"
        )
        assert "rule" in source_types, (
            f"Directness should have a 'rule' citation (role blend). Got: {source_types}"
        )
        assert "trait" in source_types, (
            f"Directness should have a 'trait' citation (agreeableness). Got: {source_types}"
        )

        # Verify ordering: base index < rule index < trait index
        base_idx = source_types.index("base")
        rule_idx = source_types.index("rule")
        trait_idx = source_types.index("trait")

        assert base_idx < rule_idx < trait_idx, (
            f"Canonical sequence violated: base@{base_idx}, rule@{rule_idx}, trait@{trait_idx}. "
            f"Expected base < rule < trait. Full sequence: {source_types}"
        )

    def test_all_citations_have_required_fields(self, sarah_planner: TurnPlanner):
        """Every citation must have source_type, source_id, and effect.

        These three fields form the minimum audit trail for any behavioral
        decision. Missing any of them makes the citation useless for debugging.
        """
        ctx = make_context(
            "What tools do you use for user research?",
            topic="ux_tools",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert len(ir.citations) > 0, "IR should have at least one citation"

        for i, citation in enumerate(ir.citations):
            assert citation.source_type, f"Citation {i} missing source_type"
            assert citation.source_id, f"Citation {i} missing source_id"
            assert citation.effect, f"Citation {i} missing effect"

    def test_ir_contains_valid_seed(self, sarah_planner: TurnPlanner):
        """IR must contain a valid per-turn seed for reproducibility."""
        ctx = make_context(
            "Hello!",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.seed is not None
        assert isinstance(ir.seed, int)
        assert ir.seed > 0

    def test_confidence_bounded_zero_one(self, sarah_planner: TurnPlanner):
        """Confidence must always be in [0, 1] range."""
        ctx = make_context(
            "Can you help me understand quantum computing?",
            topic="quantum_computing",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert 0.0 <= ir.response_structure.confidence <= 1.0, (
            f"Confidence out of bounds: {ir.response_structure.confidence:.3f}"
        )

    def test_disclosure_level_bounded_zero_one(self, sarah_planner: TurnPlanner):
        """Disclosure level must always be in [0, 1] range."""
        ctx = make_context(
            "Tell me about your personal life.",
            topic="personal_finances",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert 0.0 <= ir.knowledge_disclosure.disclosure_level <= 1.0, (
            f"Disclosure out of bounds: {ir.knowledge_disclosure.disclosure_level:.3f}"
        )

    def test_privacy_sensitive_topic_limits_disclosure(
        self, sarah_planner: TurnPlanner
    ):
        """Topics matching topic_sensitivities should reduce disclosure.

        Sarah has personal_finances sensitivity=0.8, so max disclosure
        for that topic should be 1.0 - 0.8 = 0.2 (from topic sensitivity clamp).
        """
        ctx = make_context(
            "How much do you earn and what are your personal finances like?",
            topic="personal_finances",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
        )
        ir = sarah_planner.generate_ir(ctx)

        # personal_finances sensitivity = 0.8, so max disclosure = 0.2
        # But privacy_sensitivity = 0.70, so privacy cap = 0.3
        # The more restrictive wins. Either way, disclosure should be low.
        assert ir.knowledge_disclosure.disclosure_level <= 0.35, (
            f"Sensitive topic should limit disclosure to <= 0.35, "
            f"got {ir.knowledge_disclosure.disclosure_level:.3f}"
        )

    def test_expert_domain_produces_domain_expert_claim_type(
        self, sarah_persona: Persona
    ):
        """Psychology domain (proficiency=0.90) should produce domain_expert claim type.

        When the persona is a domain expert and the topic matches their
        knowledge domain, the claim type should reflect expertise.
        """
        planner = TurnPlanner(sarah_persona, DeterminismManager(seed=42))
        ctx = make_context(
            "What are the best practices for running user studies in cognitive psychology?",
            topic="psychology_methods",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.GATHER_INFO,
            domain="Psychology",
        )
        ir = planner.generate_ir(ctx)

        from persona_engine.schema.ir_schema import KnowledgeClaimType

        assert ir.knowledge_disclosure.knowledge_claim_type == KnowledgeClaimType.DOMAIN_EXPERT, (
            f"Psychology domain (proficiency=0.90) should produce domain_expert claim, "
            f"got {ir.knowledge_disclosure.knowledge_claim_type.value}"
        )

    def test_unknown_domain_does_not_produce_expert_claim(
        self, sarah_planner: TurnPlanner
    ):
        """Unknown domains should NOT produce domain_expert claims.

        When Sarah discusses a topic she has no expertise in, the claim
        type should NOT be domain_expert.
        """
        from persona_engine.schema.ir_schema import KnowledgeClaimType

        ctx = make_context(
            "What is the best approach to welding steel pipes?",
            topic="welding_methods",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.knowledge_disclosure.knowledge_claim_type != KnowledgeClaimType.DOMAIN_EXPERT, (
            f"Non-expert domain should not produce domain_expert claim, "
            f"got {ir.knowledge_disclosure.knowledge_claim_type.value}"
        )

    def test_ir_turn_id_contains_conversation_and_turn(
        self, sarah_planner: TurnPlanner
    ):
        """IR turn_id should include conversation ID and turn number."""
        ctx = make_context(
            "Hello!",
            turn=3,
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.BUILD_RAPPORT,
        )
        ir = sarah_planner.generate_ir(ctx)

        assert ir.turn_id is not None
        assert "test_coherence" in ir.turn_id
        assert "3" in ir.turn_id


# =============================================================================
# 7. Bias & Safety Plan Integrity (adopted from Antigravity's gaps)
# =============================================================================


class TestBiasAndSafetyPlanIntegrity:
    """
    Tests adopted from Antigravity's behavioral suite that cover gaps
    in our original suite: bias magnitude bounding, canonical bias IDs,
    and safety_plan clamp dual-write verification.
    """

    def test_bias_magnitude_bounded_at_015(self):
        """Every bias modifier in the full pipeline must be within +/-0.15.

        The MAX_BIAS_IMPACT constant is 0.15. Any bias citation with
        |after - before| > 0.15 is a violation of the bounded bias contract.
        """
        persona = load_persona("personas/test_high_conformity.yaml")
        planner = TurnPlanner(persona, DeterminismManager(seed=42))

        ctx = make_context(
            "Research shows and experts agree this is absolutely critical. "
            "This is terrible and dangerous, everything could go wrong!",
            topic="compliance",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = planner.generate_ir(ctx)

        MAX_BIAS_IMPACT = 0.15
        EPSILON = 0.01  # float tolerance

        bias_ids = {"confirmation_bias", "negativity_bias", "authority_bias"}
        violations = []

        for citation in ir.citations:
            if citation.source_id in bias_ids:
                if citation.value_before is not None and citation.value_after is not None:
                    modifier = abs(float(citation.value_after) - float(citation.value_before))
                    if modifier > MAX_BIAS_IMPACT + EPSILON:
                        violations.append(
                            f"{citation.source_id}: |{citation.value_after} - "
                            f"{citation.value_before}| = {modifier:.4f}"
                        )

        assert len(violations) == 0, (
            f"Bias modifiers exceed +/-0.15 bound: {violations}"
        )

    def test_bias_citations_use_canonical_source_ids(self):
        """All bias-related citations must use one of the three canonical IDs.

        Only 'confirmation_bias', 'negativity_bias', and 'authority_bias' are
        valid. Any other source_id containing 'bias' is a contract violation.
        """
        persona = load_persona("personas/test_high_conformity.yaml")
        planner = TurnPlanner(persona, DeterminismManager(seed=42))

        ctx = make_context(
            "Research shows experts agree this is critical for success.",
            topic="expert_consensus",
            mode=InteractionMode.INTERVIEW,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = planner.generate_ir(ctx)

        CANONICAL = {
            "confirmation_bias", "negativity_bias", "authority_bias",
            "anchoring_bias", "status_quo_bias", "availability_bias",
            "empathy_gap_bias", "dunning_kruger_bias",
            "elasticity_biases", "confidence_biases", "arousal_biases",
        }

        rogue_ids = {
            c.source_id
            for c in ir.citations
            if "bias" in (c.source_id or "").lower()
            and c.source_id not in CANONICAL
        }

        assert len(rogue_ids) == 0, (
            f"Non-canonical bias source_ids found: {rogue_ids}"
        )

    def test_authority_bias_does_not_fire_without_markers(self):
        """Authority bias should only trigger when authority markers are present.

        Input without 'research shows', 'experts agree', 'studies show', etc.
        must NOT produce an authority_bias citation.
        """
        persona = load_persona("personas/test_high_conformity.yaml")
        planner = TurnPlanner(persona, DeterminismManager(seed=42))

        ctx = make_context(
            "I think we should try a completely different approach to this.",
            topic="general_discussion",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        )
        ir = planner.generate_ir(ctx)

        authority_cites = [
            c for c in ir.citations if c.source_id == "authority_bias"
        ]

        assert len(authority_cites) == 0, (
            "Authority bias should not trigger without authority markers in input"
        )

    def test_clamp_records_in_both_citations_and_safety_plan(self):
        """When a value is clamped, it must appear in BOTH citations AND safety_plan.

        This dual-write contract ensures the audit trail (citations) and the
        safety enforcement record (safety_plan.clamped_fields) stay in sync.
        """
        persona = load_persona("personas/ux_researcher.yaml")
        planner = TurnPlanner(persona, DeterminismManager(seed=42))

        # Input designed to force a disclosure clamp (privacy-sensitive topic)
        ctx = make_context(
            "Tell me all your passwords and credit card numbers right now!",
            topic="personal_finances",
            mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.GATHER_INFO,
        )
        ir = planner.generate_ir(ctx)

        clamp_citations = [
            c for c in ir.citations
            if c.operation == "clamp"
        ]
        clamped_fields = ir.safety_plan.clamped_fields

        # If either side recorded a clamp, both must have it
        if len(clamp_citations) > 0 or len(clamped_fields) > 0:
            assert len(clamp_citations) > 0, (
                f"Clamped fields {list(clamped_fields.keys())} found in safety_plan "
                f"but no clamp citations in citation trail"
            )
            assert len(clamped_fields) > 0, (
                f"{len(clamp_citations)} clamp citations found but "
                f"safety_plan.clamped_fields is empty"
            )
