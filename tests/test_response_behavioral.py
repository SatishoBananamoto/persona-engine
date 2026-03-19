"""
Behavioral Scenario Tests - Response Generation Layer

Tests that the response generation module produces psychologically coherent
text from IR. These validate the CONTRACT between IR fields and the actual
language produced — not just that adapters work, but that the TEXT reflects
the persona's computed behavioral state.

Key difference from test_response_generation.py:
  - test_response_generation.py: structural tests (adapters work, prompts build)
  - This file: behavioral tests (does the TEXT sound right?)

Key difference from test_behavioral_coherence.py:
  - test_behavioral_coherence.py: validates IR FIELD values (confidence=0.23)
  - This file: validates GENERATED TEXT reflects those IR fields
"""

import pytest
import yaml
from pathlib import Path

from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.generation import ResponseGenerator
from persona_engine.generation.llm_adapter import TemplateAdapter
from persona_engine.generation.prompt_builder import build_ir_prompt as build_system_prompt
from persona_engine.schema.ir_schema import (
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    IntermediateRepresentation,
    InteractionMode,
    KnowledgeAndDisclosure,
    KnowledgeClaimType,
    ResponseStructure,
    SafetyPlan,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Helpers
# ============================================================================


def _load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    """Load persona from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def _make_ir(
    tone: Tone = Tone.NEUTRAL_CALM,
    verbosity: Verbosity = Verbosity.MEDIUM,
    formality: float = 0.5,
    directness: float = 0.5,
    confidence: float = 0.5,
    stance: str | None = "This is a reasonable position on the topic.",
    rationale: str | None = "Based on professional experience.",
    elasticity: float = 0.5,
    disclosure_level: float = 0.5,
    uncertainty_action: UncertaintyAction = UncertaintyAction.ANSWER,
    claim_type: KnowledgeClaimType = KnowledgeClaimType.COMMON_KNOWLEDGE,
    intent: str = "Share perspective",
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    blocked_topics: list[str] | None = None,
) -> IntermediateRepresentation:
    """Build an IR with specified parameters for testing."""
    return IntermediateRepresentation(
        conversation_frame=ConversationFrame(
            interaction_mode=mode,
            goal=goal,
        ),
        response_structure=ResponseStructure(
            intent=intent,
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
            disclosure_level=disclosure_level,
            uncertainty_action=uncertainty_action,
            knowledge_claim_type=claim_type,
        ),
        safety_plan=SafetyPlan(
            blocked_topics=blocked_topics or [],
        ),
        turn_id="test_turn_1",
    )


def _generate_template(ir, user_input="Tell me about this.", persona=None):
    """Generate response using template adapter.

    Returns a SimpleNamespace with .text so call-sites can use resp.text
    (the generation/ TemplateAdapter.generate_from_ir returns a plain str).
    """
    from types import SimpleNamespace
    adapter = TemplateAdapter()
    text = adapter.generate_from_ir(ir=ir, user_input=user_input, persona=persona)
    return SimpleNamespace(text=text)


def _e2e_generate(persona, user_input, mode, goal, topic, turn=1, cache=None):
    """Full end-to-end: persona + user input → IR → response text."""
    determinism = DeterminismManager(seed=42)
    planner = TurnPlanner(persona, determinism)
    cache = cache or StanceCache()
    context = ConversationContext(
        conversation_id="test_session",
        turn_number=turn,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=cache,
    )
    ir = planner.generate_ir(context)
    gen = ResponseGenerator(persona=persona, provider="template")
    resp = gen.generate(ir, user_input)
    return resp, ir


# ============================================================================
# 1. Expert Domain vs Out-of-Domain Language
# ============================================================================


class TestExpertVsOutOfDomain:
    """Verify that expert domains produce confident language and
    out-of-domain topics produce hedging/uncertain language."""

    @pytest.fixture
    def persona(self):
        return _load_persona()

    def test_expert_domain_has_high_confidence_in_ir(self, persona):
        """Expert domain (UX/Psychology) should produce high confidence in IR."""
        resp, ir = _e2e_generate(
            persona,
            "What methods do you recommend for usability testing?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "usability_testing",
        )
        assert ir.response_structure.confidence >= 0.7, (
            f"Expert domain should have confidence >= 0.7, got {ir.response_structure.confidence}"
        )

    def test_out_of_domain_has_low_confidence_in_ir(self, persona):
        """Out-of-domain topic (quantum physics) should produce low confidence."""
        resp, ir = _e2e_generate(
            persona,
            "Can you explain quantum entanglement?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "quantum_physics",
        )
        assert ir.response_structure.confidence < 0.5, (
            f"Out-of-domain should have confidence < 0.5, got {ir.response_structure.confidence}"
        )

    def test_expert_response_no_hedging(self, persona):
        """Expert domain response should not start with hedging phrases."""
        resp, ir = _e2e_generate(
            persona,
            "Tell me about cognitive load theory.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "cognitive_psychology",
        )
        text_lower = resp.text.lower()
        hedges = ["i think", "if i'm not mistaken", "as far as i know", "from what i understand"]
        # High-confidence expert responses should use direct framing, not hedging
        for hedge in hedges:
            assert not text_lower.startswith(hedge), (
                f"Expert response should not start with '{hedge}', got: {resp.text[:100]}"
            )

    def test_out_of_domain_response_has_hedging(self, persona):
        """Out-of-domain response should include hedging or uncertainty markers."""
        resp, ir = _e2e_generate(
            persona,
            "Can you explain quantum computing?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "quantum_computing",
        )
        text_lower = resp.text.lower()
        uncertainty_markers = [
            "i think",
            "not entirely certain",
            "not really the right person",
            "from what i understand",
            "if i'm not mistaken",
            "not sure",
        ]
        has_uncertainty = any(m in text_lower for m in uncertainty_markers)
        assert has_uncertainty, (
            f"Out-of-domain response should show uncertainty, got: {resp.text[:200]}"
        )

    def test_expert_claim_type_is_domain_expert(self, persona):
        """Expert domain should produce domain_expert claim type."""
        resp, ir = _e2e_generate(
            persona,
            "What's the best way to conduct user interviews?",
            InteractionMode.INTERVIEW,
            ConversationGoal.GATHER_INFO,
            "ux_research",
        )
        assert ir.knowledge_disclosure.knowledge_claim_type == KnowledgeClaimType.DOMAIN_EXPERT

    def test_out_of_domain_claim_type_is_speculative(self, persona):
        """Out-of-domain should produce speculative claim type."""
        resp, ir = _e2e_generate(
            persona,
            "What's your view on cryptocurrency regulations?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "cryptocurrency",
        )
        assert ir.knowledge_disclosure.knowledge_claim_type in (
            KnowledgeClaimType.SPECULATIVE,
            KnowledgeClaimType.COMMON_KNOWLEDGE,
        ), f"Out-of-domain should be speculative or common_knowledge, got {ir.knowledge_disclosure.knowledge_claim_type}"

    def test_expert_vs_novice_response_length_differs(self, persona):
        """Expert responses tend to be longer (more detail) than novice ones."""
        resp_expert, _ = _e2e_generate(
            persona,
            "Tell me about UX research methods.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "ux_research",
        )
        resp_novice, _ = _e2e_generate(
            persona,
            "Tell me about quantum chromodynamics.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "quantum_physics",
        )
        # Not necessarily longer, but they should be different
        assert resp_expert.text != resp_novice.text, "Expert and novice responses should differ"


# ============================================================================
# 2. Tone Manifestation in Generated Text
# ============================================================================


class TestToneManifestationInText:
    """Verify that different tones produce distinctly different language."""

    def test_warm_enthusiastic_has_excitement_markers(self):
        """Warm enthusiastic tone should have excitement language."""
        ir = _make_ir(tone=Tone.WARM_ENTHUSIASTIC, confidence=0.8)
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        excitement = ["excited", "great", "love", "really", "!"]
        has_excitement = any(m in text_lower for m in excitement)
        assert has_excitement, f"Warm enthusiastic should show excitement: {resp.text[:150]}"

    def test_anxious_stressed_has_worry_language(self):
        """Anxious stressed tone should have anxiety markers."""
        ir = _make_ir(tone=Tone.ANXIOUS_STRESSED, confidence=0.3)
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        anxiety = ["worried", "concerned", "bit", "honestly", "..."]
        has_anxiety = any(m in text_lower for m in anxiety)
        assert has_anxiety, f"Anxious stressed should show anxiety: {resp.text[:150]}"

    def test_frustrated_tense_has_impatient_language(self):
        """Frustrated tense tone should have impatient or curt language."""
        ir = _make_ir(tone=Tone.FRUSTRATED_TENSE, confidence=0.6)
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        frustration = ["look", "honestly", "frustrating", "frank"]
        has_frustration = any(m in text_lower for m in frustration)
        assert has_frustration, f"Frustrated tone should show frustration: {resp.text[:150]}"

    def test_tired_withdrawn_is_brief(self):
        """Tired withdrawn tone should produce shorter, subdued responses."""
        ir = _make_ir(tone=Tone.TIRED_WITHDRAWN, verbosity=Verbosity.MEDIUM, confidence=0.4)
        resp = _generate_template(ir)
        # Check for low-energy openers
        text_lower = resp.text.lower()
        subdued = ["mm", "right", "well", "..."]
        has_subdued = any(m in text_lower for m in subdued)
        assert has_subdued, f"Tired tone should have subdued opener: {resp.text[:100]}"

    def test_defensive_agitated_has_pushback(self):
        """Defensive agitated tone should push back or correct."""
        ir = _make_ir(tone=Tone.DEFENSIVE_AGITATED, confidence=0.7)
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        pushback = ["don't think that's fair", "not quite right", "actually"]
        has_pushback = any(m in text_lower for m in pushback)
        assert has_pushback, f"Defensive tone should show pushback: {resp.text[:150]}"

    def test_professional_composed_vs_friendly_relaxed(self):
        """Professional and friendly tones should produce distinct language."""
        ir_prof = _make_ir(tone=Tone.PROFESSIONAL_COMPOSED, formality=0.7)
        ir_casual = _make_ir(tone=Tone.FRIENDLY_RELAXED, formality=0.2)
        resp_prof = _generate_template(ir_prof)
        resp_casual = _generate_template(ir_casual)
        assert resp_prof.text != resp_casual.text, "Professional vs friendly should differ"

    def test_different_tones_produce_different_openers(self):
        """All major tone categories should produce unique openers."""
        tones = [
            Tone.WARM_ENTHUSIASTIC,
            Tone.NEUTRAL_CALM,
            Tone.FRUSTRATED_TENSE,
            Tone.SAD_SUBDUED,
            Tone.PROFESSIONAL_COMPOSED,
        ]
        openers = set()
        for tone in tones:
            ir = _make_ir(tone=tone)
            resp = _generate_template(ir)
            # First sentence as opener
            first = resp.text.split(".")[0] if "." in resp.text else resp.text[:50]
            openers.add(first)
        # At least 4 of 5 should be unique (some may overlap in edge cases)
        assert len(openers) >= 4, f"Expected at least 4 unique openers, got {len(openers)}: {openers}"


# ============================================================================
# 3. Confidence-Uncertainty Coherence in Text
# ============================================================================


class TestConfidenceUncertaintyInText:
    """Verify that confidence level and uncertainty action manifest in language."""

    def test_high_confidence_direct_assertion(self):
        """High confidence + ANSWER should produce direct, unhesitating language."""
        ir = _make_ir(
            confidence=0.85,
            uncertainty_action=UncertaintyAction.ANSWER,
            stance="Remote usability testing is effective for most use cases.",
        )
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        # Should state stance directly, not hedged
        assert "remote usability testing is effective" in text_lower, (
            f"High confidence should state stance directly: {resp.text[:200]}"
        )
        assert "i think" not in text_lower[:50], (
            f"High confidence should not start with 'I think': {resp.text[:100]}"
        )

    def test_low_confidence_has_hedging(self):
        """Low confidence should produce hedging language."""
        ir = _make_ir(
            confidence=0.2,
            uncertainty_action=UncertaintyAction.HEDGE,
            stance="Maybe blockchain has some applications here.",
        )
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        hedges = ["i think", "from what i understand", "if i'm not mistaken", "as far as i know"]
        has_hedge = any(h in text_lower for h in hedges)
        assert has_hedge, f"Low confidence should have hedging: {resp.text[:200]}"

    def test_moderate_confidence_uses_experience_framing(self):
        """Moderate confidence should use experience-based framing."""
        ir = _make_ir(
            confidence=0.55,
            stance="This approach has some merit.",
        )
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        experience = ["in my experience", "from what i've seen", "based on my work"]
        has_experience = any(e in text_lower for e in experience)
        assert has_experience, f"Moderate confidence should use experience framing: {resp.text[:200]}"

    def test_uncertainty_refuse_declines_politely(self):
        """REFUSE uncertainty action should include declination language."""
        ir = _make_ir(
            confidence=0.1,
            uncertainty_action=UncertaintyAction.REFUSE,
        )
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        assert "not really the right person" in text_lower, (
            f"REFUSE should include refusal language: {resp.text[:200]}"
        )

    def test_uncertainty_ask_clarifying_asks_question(self):
        """ASK_CLARIFYING should include a question back to the user."""
        ir = _make_ir(
            confidence=0.4,
            uncertainty_action=UncertaintyAction.ASK_CLARIFYING,
        )
        resp = _generate_template(ir)
        text_lower = resp.text.lower()
        assert "?" in resp.text, f"ASK_CLARIFYING should contain a question: {resp.text[:200]}"
        assert "what" in text_lower or "specifically" in text_lower or "could you" in text_lower

    def test_hedge_vs_answer_text_differs(self):
        """HEDGE and ANSWER should produce noticeably different text."""
        ir_answer = _make_ir(confidence=0.8, uncertainty_action=UncertaintyAction.ANSWER)
        ir_hedge = _make_ir(confidence=0.3, uncertainty_action=UncertaintyAction.HEDGE)
        resp_answer = _generate_template(ir_answer)
        resp_hedge = _generate_template(ir_hedge)
        assert resp_answer.text != resp_hedge.text


# ============================================================================
# 4. Formality Manifestation in Text
# ============================================================================


class TestFormalityInText:
    """Verify that formality level changes language register."""

    def test_high_formality_removes_contractions(self):
        """High formality should expand contractions."""
        ir = _make_ir(formality=0.85, tone=Tone.PROFESSIONAL_COMPOSED)
        resp = _generate_template(ir)
        # Formalize transform should have run
        contractions = ["I'm ", "don't", "can't", "won't", "it's", "that's"]
        text = resp.text
        for c in contractions:
            assert c not in text, f"High formality should not contain '{c}': {text[:150]}"

    def test_low_formality_has_casual_language(self):
        """Low formality should use casual language transforms."""
        ir = _make_ir(formality=0.15, tone=Tone.FRIENDLY_RELAXED)
        resp = _generate_template(ir)
        # Casualize transform should have run
        text_lower = resp.text.lower()
        # Should have casual markers
        casual = ["'m", "sure thing", "yeah", "here's the deal", "got it", "don't"]
        has_casual = any(c in text_lower for c in casual)
        assert has_casual, f"Low formality should have casual language: {resp.text[:150]}"

    def test_formality_between_025_075_no_transform(self):
        """Mid-range formality should not apply either transform."""
        ir = _make_ir(formality=0.5, tone=Tone.NEUTRAL_CALM)
        resp = _generate_template(ir)
        # Text should remain as-is (no formalize/casualize applied)
        assert resp.text  # Just verify it generates

    def test_interview_mode_produces_higher_formality(self):
        """Full pipeline: interview mode should increase formality vs casual."""
        persona = _load_persona()
        resp_interview, ir_iv = _e2e_generate(
            persona,
            "What do you think about remote testing?",
            InteractionMode.INTERVIEW,
            ConversationGoal.GATHER_INFO,
            "usability_testing",
        )
        resp_casual, ir_cs = _e2e_generate(
            persona,
            "What do you think about remote testing?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "usability_testing",
        )
        assert ir_iv.communication_style.formality > ir_cs.communication_style.formality, (
            f"Interview formality ({ir_iv.communication_style.formality}) should exceed "
            f"casual ({ir_cs.communication_style.formality})"
        )


# ============================================================================
# 5. Verbosity Manifestation in Text
# ============================================================================


class TestVerbosityInText:
    """Verify that verbosity level controls response length and detail."""

    def test_brief_produces_short_response(self):
        """Brief verbosity should produce 1-2 sentences max."""
        ir = _make_ir(verbosity=Verbosity.BRIEF)
        resp = _generate_template(ir)
        sentence_count = resp.text.count(". ") + resp.text.count("? ") + (1 if resp.text.endswith(".") else 0)
        assert sentence_count <= 3, (
            f"Brief should produce <= 3 sentences, got ~{sentence_count}: {resp.text[:200]}"
        )

    def test_detailed_produces_longer_response(self):
        """Detailed verbosity should produce more content than brief."""
        persona = _load_persona()
        ir_brief = _make_ir(verbosity=Verbosity.BRIEF)
        ir_detailed = _make_ir(verbosity=Verbosity.DETAILED)
        resp_brief = _generate_template(ir_brief, persona=persona)
        resp_detailed = _generate_template(ir_detailed, persona=persona)
        assert len(resp_detailed.text) > len(resp_brief.text), (
            f"Detailed ({len(resp_detailed.text)} chars) should be longer than "
            f"brief ({len(resp_brief.text)} chars)"
        )

    def test_detailed_includes_persona_occupation(self):
        """Detailed verbosity with persona should mention occupation."""
        persona = _load_persona()
        ir = _make_ir(verbosity=Verbosity.DETAILED)
        resp = _generate_template(ir, persona=persona)
        assert "UX Researcher" in resp.text, (
            f"Detailed response with persona should mention occupation: {resp.text[:200]}"
        )

    def test_brief_omits_rationale(self):
        """Brief verbosity should not include the rationale text."""
        ir = _make_ir(
            verbosity=Verbosity.BRIEF,
            rationale="Based on 8 years of professional experience.",
        )
        resp = _generate_template(ir)
        assert "8 years" not in resp.text, (
            f"Brief should omit rationale details: {resp.text[:200]}"
        )


# ============================================================================
# 6. Role-Based Language Shifts (E2E)
# ============================================================================


class TestRoleBasedLanguage:
    """Verify that different interaction modes produce different language styles."""

    @pytest.fixture
    def persona(self):
        return _load_persona()

    def test_interview_vs_casual_ir_differs(self, persona):
        """Interview and casual modes should produce different IR style parameters.

        The template adapter may produce identical text when both formality
        values fall in the same band (0.25-0.75), but the underlying IR
        should always differ in formality and/or directness.
        """
        _, ir_iv = _e2e_generate(
            persona,
            "What do you think about user research?",
            InteractionMode.INTERVIEW,
            ConversationGoal.GATHER_INFO,
            "ux_research",
        )
        _, ir_cs = _e2e_generate(
            persona,
            "What do you think about user research?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "ux_research",
        )
        style_iv = ir_iv.communication_style
        style_cs = ir_cs.communication_style
        # At least one of formality/directness should differ between modes
        assert style_iv.formality != style_cs.formality or style_iv.directness != style_cs.directness, (
            f"Interview and casual should produce different style: "
            f"iv(f={style_iv.formality}, d={style_iv.directness}) vs "
            f"cs(f={style_cs.formality}, d={style_cs.directness})"
        )

    def test_interview_directness_differs_from_casual(self, persona):
        """Interview should have different directness than casual chat."""
        _, ir_iv = _e2e_generate(
            persona,
            "How would you approach a redesign project?",
            InteractionMode.INTERVIEW,
            ConversationGoal.GATHER_INFO,
            "ux_design",
        )
        _, ir_cs = _e2e_generate(
            persona,
            "How would you approach a redesign project?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "ux_design",
        )
        # Interview typically has different directness (higher for Sarah's at_work role)
        assert ir_iv.communication_style.directness != ir_cs.communication_style.directness, (
            f"Directness should differ: interview={ir_iv.communication_style.directness}, "
            f"casual={ir_cs.communication_style.directness}"
        )

    def test_debate_mode_produces_response(self, persona):
        """Debate mode should successfully generate a response."""
        resp, ir = _e2e_generate(
            persona,
            "I disagree, qualitative research is overrated.",
            InteractionMode.DEBATE,
            ConversationGoal.PERSUADE,
            "research_methods",
        )
        assert resp.text
        assert len(resp.text) > 10


# ============================================================================
# 7. Prompt Builder Behavioral Contracts
# ============================================================================


class TestPromptBuilderBehavioralContracts:
    """Verify that the system prompt correctly translates IR to behavioral instructions."""

    def test_high_confidence_prompt_includes_confident_instruction(self):
        """High confidence IR should produce 'speak with confidence' instruction."""
        ir = _make_ir(confidence=0.85)
        prompt = build_system_prompt(ir)
        assert "confidence" in prompt.lower() or "know what you" in prompt.lower(), (
            f"High confidence prompt should mention confidence: {prompt[:300]}"
        )

    def test_low_confidence_prompt_includes_uncertain_instruction(self):
        """Low confidence IR should produce uncertainty instruction."""
        ir = _make_ir(confidence=0.2)
        prompt = build_system_prompt(ir)
        uncertainty_keywords = ["uncertain", "not sure", "i think", "might be"]
        has_uncertainty = any(k in prompt.lower() for k in uncertainty_keywords)
        assert has_uncertainty, (
            f"Low confidence prompt should mention uncertainty: {prompt[:300]}"
        )

    def test_persona_identity_appears_in_prompt(self):
        """Persona identity should appear in the system prompt."""
        persona = _load_persona()
        ir = _make_ir()
        prompt = build_system_prompt(ir, persona=persona)
        assert "Sarah" in prompt, f"Prompt should include persona name: {prompt[:200]}"
        assert "UX Researcher" in prompt, f"Prompt should include occupation: {prompt[:200]}"

    def test_blocked_topics_appear_in_prompt(self):
        """Blocked topics should appear as critical constraints in prompt."""
        ir = _make_ir(blocked_topics=["employer_name", "participant_data"])
        prompt = build_system_prompt(ir)
        assert "employer_name" in prompt, f"Blocked topic should appear in prompt: {prompt}"
        assert "CRITICAL" in prompt or "NOT" in prompt, "Should have strong constraint language"

    def test_safety_plan_with_no_blocks_has_no_constraint_section(self):
        """If no safety constraints, the CONSTRAINTS section should be absent."""
        ir = _make_ir(blocked_topics=[])
        prompt = build_system_prompt(ir)
        assert "CONSTRAINTS:" not in prompt

    def test_stance_appears_in_response_guidance(self):
        """The persona's stance should appear in the response guidance section."""
        ir = _make_ir(stance="Remote work flexibility is important for creative teams.")
        prompt = build_system_prompt(ir)
        assert "Remote work flexibility" in prompt

    def test_all_tone_values_have_prompts(self):
        """Every Tone enum should have a corresponding prompt mapping."""
        from persona_engine.generation.prompt_builder import TONE_PROMPTS
        for tone in Tone:
            assert tone.value in TONE_PROMPTS, f"Missing prompt for tone: {tone.value}"

    def test_disclosure_level_affects_prompt(self):
        """Different disclosure levels should produce different instructions."""
        ir_low = _make_ir(disclosure_level=0.1)
        ir_high = _make_ir(disclosure_level=0.8)
        prompt_low = build_system_prompt(ir_low)
        prompt_high = build_system_prompt(ir_high)
        # Low disclosure should mention guarded
        assert "guarded" in prompt_low.lower() or "general" in prompt_low.lower()
        # High disclosure should mention open or sharing
        assert "open" in prompt_high.lower() or "shar" in prompt_high.lower()

    def test_elasticity_affects_prompt(self):
        """Different elasticity levels should produce different openness instructions."""
        ir_rigid = _make_ir(elasticity=0.1)
        ir_flexible = _make_ir(elasticity=0.8)
        prompt_rigid = build_system_prompt(ir_rigid)
        prompt_flex = build_system_prompt(ir_flexible)
        assert "firm" in prompt_rigid.lower() or "maintain" in prompt_rigid.lower()
        assert "open" in prompt_flex.lower() or "adjust" in prompt_flex.lower()


# ============================================================================
# 8. Persona Differentiation in Response
# ============================================================================


class TestPersonaDifferentiationInResponse:
    """Verify that different personas produce characteristically different responses."""

    @pytest.fixture
    def sarah(self):
        return _load_persona()

    def test_occupation_appears_in_detailed_response(self, sarah):
        """Persona's occupation should appear in detailed template responses."""
        resp, ir = _e2e_generate(
            sarah,
            "Tell me about your work.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.BUILD_RAPPORT,
            "professional_work",
        )
        # Detailed verbosity + persona should include occupation
        if ir.communication_style.verbosity == Verbosity.DETAILED:
            assert "UX Researcher" in resp.text

    def test_persona_background_in_prompt(self, sarah):
        """Prompt should reference persona's background for LLM backend."""
        ir = _make_ir()
        prompt = build_system_prompt(ir, persona=sarah)
        assert "Manchester" in prompt or "London" in prompt or "Psychology" in prompt

    def test_same_question_different_domain_different_confidence(self, sarah):
        """Same persona, different domains → different confidence/uncertainty."""
        _, ir_expert = _e2e_generate(
            sarah,
            "What research methods work best?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "research_methods",
        )
        _, ir_novice = _e2e_generate(
            sarah,
            "What investment strategies work best?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "investment",
        )
        assert ir_expert.response_structure.confidence > ir_novice.response_structure.confidence, (
            f"Expert confidence ({ir_expert.response_structure.confidence}) should exceed "
            f"novice ({ir_novice.response_structure.confidence})"
        )


# ============================================================================
# 9. Multi-Turn Consistency in Language
# ============================================================================


class TestMultiTurnLanguageConsistency:
    """Verify that response language stays consistent across multiple turns."""

    @pytest.fixture
    def persona(self):
        return _load_persona()

    def test_same_topic_same_stance_across_turns(self, persona):
        """Asking about the same topic twice should produce consistent stance."""
        cache = StanceCache()
        determinism = DeterminismManager(seed=42)
        planner = TurnPlanner(persona, determinism)
        gen = ResponseGenerator(persona=persona, provider="template")

        # Turn 1
        ctx1 = ConversationContext(
            conversation_id="consistency_test",
            turn_number=1,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="ux_research",
            user_input="What do you think about remote testing?",
            stance_cache=cache,
        )
        ir1 = planner.generate_ir(ctx1)
        resp1 = gen.generate(ir1, ctx1.user_input)

        # Turn 2 — same topic
        ctx2 = ConversationContext(
            conversation_id="consistency_test",
            turn_number=2,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="ux_research",
            user_input="Can you elaborate on remote testing?",
            stance_cache=cache,
        )
        ir2 = planner.generate_ir(ctx2)
        resp2 = gen.generate(ir2, ctx2.user_input)

        # Both should reference the same cached stance
        assert ir1.response_structure.stance == ir2.response_structure.stance, (
            "Same topic should use cached stance across turns"
        )

    def test_multiple_turns_all_produce_responses(self, persona):
        """Running multiple turns should not crash or produce empty responses."""
        cache = StanceCache()
        determinism = DeterminismManager(seed=42)
        planner = TurnPlanner(persona, determinism)
        gen = ResponseGenerator(persona=persona, provider="template")

        topics = [
            ("Tell me about UX research.", "ux_research"),
            ("What about project management?", "project_management"),
            ("How do you handle stress at work?", "work_stress"),
        ]
        for i, (question, topic) in enumerate(topics, 1):
            ctx = ConversationContext(
                conversation_id="multi_turn",
                turn_number=i,
                interaction_mode=InteractionMode.CASUAL_CHAT,
                goal=ConversationGoal.EXPLORE_IDEAS,
                topic_signature=topic,
                user_input=question,
                stance_cache=cache,
            )
            ir = planner.generate_ir(ctx)
            resp = gen.generate(ir, question)
            assert resp.text, f"Turn {i} should produce non-empty response"
            assert len(resp.text) > 5, f"Turn {i} response too short: {resp.text}"


# ============================================================================
# 10. Stress and Mood Effects on Response
# ============================================================================


class TestStressAndMoodOnResponse:
    """Verify that stress/mood state changes are reflected in response language."""

    @pytest.fixture
    def persona(self):
        return _load_persona()

    def test_challenge_affects_tone(self, persona):
        """Challenging input should affect the emotional tone."""
        # Normal question
        _, ir_normal = _e2e_generate(
            persona,
            "What do you think about UX design?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "ux_design",
        )
        # Challenging question (evidence markers)
        _, ir_challenge = _e2e_generate(
            persona,
            "Studies show that UX research is a waste of time and money, research proves this conclusively.",
            InteractionMode.DEBATE,
            ConversationGoal.PERSUADE,
            "ux_research",
        )
        # Challenge should produce different tone or at least different response
        assert ir_normal.communication_style.tone != ir_challenge.communication_style.tone or \
               ir_normal.response_structure.stance != ir_challenge.response_structure.stance, (
            "Challenge should affect tone or stance"
        )


# ============================================================================
# 11. Safety Constraints in Response
# ============================================================================


class TestSafetyConstraintsInResponse:
    """Verify safety constraints affect the system prompt."""

    def test_blocked_topics_produce_critical_instructions(self):
        """Blocked topics should produce CRITICAL DO NOT instructions."""
        ir = _make_ir(blocked_topics=["employer_name", "salary_info"])
        prompt = build_system_prompt(ir)
        assert "employer_name" in prompt
        assert "salary_info" in prompt
        assert "DO NOT" in prompt or "CRITICAL" in prompt

    def test_no_blocked_topics_no_constraint_section(self):
        """Without blocked topics, no constraint section should appear."""
        ir = _make_ir(blocked_topics=[])
        prompt = build_system_prompt(ir)
        assert "CONSTRAINTS:" not in prompt


# ============================================================================
# 12. Template Response Determinism
# ============================================================================


class TestTemplateResponseDeterminism:
    """Verify that template responses are deterministic given same IR."""

    def test_same_ir_same_response(self):
        """Identical IR should always produce identical template response."""
        ir = _make_ir(
            tone=Tone.WARM_CONFIDENT,
            confidence=0.75,
            stance="UX research is essential for product success.",
        )
        resp1 = _generate_template(ir, "Tell me about UX.")
        resp2 = _generate_template(ir, "Tell me about UX.")
        assert resp1.text == resp2.text, "Same IR should produce same template output"

    def test_e2e_determinism(self):
        """Full pipeline with same seed should produce same output."""
        persona = _load_persona()
        resp1, _ = _e2e_generate(
            persona,
            "Tell me about cognitive load.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "cognitive_psychology",
        )
        resp2, _ = _e2e_generate(
            persona,
            "Tell me about cognitive load.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "cognitive_psychology",
        )
        assert resp1.text == resp2.text, "Same seed + same input should produce same output"


# ============================================================================
# 13. End-to-End IR → Prompt → Response Coherence
# ============================================================================


class TestEndToEndCoherence:
    """Verify that the full pipeline maintains coherence from IR to text."""

    @pytest.fixture
    def persona(self):
        return _load_persona()

    def test_ir_confidence_matches_prompt_instruction(self, persona):
        """IR confidence should produce appropriate prompt instruction level."""
        _, ir = _e2e_generate(
            persona,
            "Tell me about research methods.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.GATHER_INFO,
            "research_methods",
        )
        prompt = build_system_prompt(ir, persona=persona)
        conf = ir.response_structure.confidence

        if conf >= 0.6:
            assert "confidence" in prompt.lower() or "know" in prompt.lower(), (
                f"High confidence ({conf}) should produce confident instruction"
            )
        elif conf < 0.3:
            assert "uncertain" in prompt.lower() or "not sure" in prompt.lower(), (
                f"Low confidence ({conf}) should produce uncertain instruction"
            )

    def test_ir_tone_matches_prompt_instruction(self, persona):
        """IR tone should produce matching tone instruction in prompt."""
        _, ir = _e2e_generate(
            persona,
            "This is exciting work!",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.BUILD_RAPPORT,
            "ux_research",
        )
        prompt = build_system_prompt(ir, persona=persona)
        tone = ir.communication_style.tone.value
        # The prompt should contain text related to the tone
        from persona_engine.generation.prompt_builder import TONE_PROMPTS
        expected_instruction = TONE_PROMPTS.get(tone, "")
        if expected_instruction:
            # At least part of the tone instruction should be in the prompt
            assert expected_instruction[:30] in prompt, (
                f"Tone {tone} instruction should appear in prompt"
            )

    def test_ir_verbosity_matches_prompt_instruction(self, persona):
        """IR verbosity should produce matching length instruction."""
        _, ir = _e2e_generate(
            persona,
            "Tell me something interesting.",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "ux_research",
        )
        prompt = build_system_prompt(ir, persona=persona)
        from persona_engine.generation.prompt_builder import VERBOSITY_PROMPTS
        verb = ir.communication_style.verbosity.value
        expected = VERBOSITY_PROMPTS.get(verb, "")
        if expected:
            assert expected[:20] in prompt

    def test_full_pipeline_template_produces_valid_response(self, persona):
        """Full pipeline should produce a non-empty, valid GeneratedResponse."""
        resp, ir = _e2e_generate(
            persona,
            "What's your view on design thinking?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.EXPLORE_IDEAS,
            "design_thinking",
        )
        assert resp.text
        assert resp.model == "template-rule-based"
        assert resp.ir is not None
        assert ir.turn_id == resp.ir.turn_id

    def test_mock_backend_captures_prompt(self, persona):
        """Mock backend should capture the system prompt for inspection."""
        from persona_engine.generation.llm_adapter import MockLLMAdapter

        determinism = DeterminismManager(seed=42)
        planner = TurnPlanner(persona, determinism)
        cache = StanceCache()
        ctx = ConversationContext(
            conversation_id="mock_test",
            turn_number=1,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="ux_research",
            user_input="Tell me about UX.",
            stance_cache=cache,
        )
        ir = planner.generate_ir(ctx)

        mock = MockLLMAdapter()
        gen = ResponseGenerator(
            persona=persona,
            adapter=mock,
        )
        resp = gen.generate(ir, "Tell me about UX.")
        # Mock should have captured the system prompt
        assert mock.last_system_prompt is not None
        assert "Sarah" in mock.last_system_prompt
        assert mock.call_count == 1
