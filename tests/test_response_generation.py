"""Tests for Phase 5: Response Generation.

Covers:
  1. Prompt builder (IR → system prompt)
  2. Float-to-instruction converters
  3. TemplateAdapter (rule-based text generation)
  4. Persona differentiation (different personas → different output)
  5. Full pipeline integration
  6. Strict mode
  7. LLM adapter error handling

All tests run without an API key.
All imports use the generation/ module (response/ is deprecated).
"""

import pytest
import yaml

from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.generation.llm_adapter import TemplateAdapter, AnthropicAdapter
from persona_engine.generation.response_generator import (
    ResponseGenerator,
    GeneratedResponse,
    GenerationBackend,
    ResponseConfig,
    create_response_generator,
)
from persona_engine.generation.prompt_builder import (
    CLAIM_TYPE_PROMPTS,
    TONE_PROMPTS,
    UNCERTAINTY_PROMPTS,
    VERBOSITY_PROMPTS,
    build_ir_prompt as build_system_prompt,
    confidence_instruction,
    directness_instruction,
    disclosure_instruction,
    elasticity_instruction,
    formality_instruction,
)
from persona_engine.schema.ir_schema import (
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    KnowledgeClaimType,
    ResponseStructure,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


# =============================================================================
# Fixtures
# =============================================================================


def load_persona(path: str = "personas/ux_researcher.yaml") -> Persona:
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def make_ir(
    tone: Tone = Tone.NEUTRAL_CALM,
    verbosity: Verbosity = Verbosity.MEDIUM,
    formality: float = 0.5,
    directness: float = 0.5,
    confidence: float = 0.5,
    elasticity: float = 0.5,
    disclosure_level: float = 0.5,
    uncertainty_action: UncertaintyAction = UncertaintyAction.ANSWER,
    knowledge_claim_type: KnowledgeClaimType = KnowledgeClaimType.COMMON_KNOWLEDGE,
    stance: str | None = "This approach has both strengths and weaknesses",
    rationale: str | None = "Based on experience and analysis of available evidence",
    intent: str = "Share perspective on the topic",
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    turn_id: str = "test_turn_1",
) -> IntermediateRepresentation:
    """Build an IR with controllable parameters for testing."""
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
            knowledge_claim_type=knowledge_claim_type,
        ),
        turn_id=turn_id,
        seed=42,
    )


def make_context(
    user_input: str,
    topic: str = "general",
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    turn: int = 1,
) -> ConversationContext:
    return ConversationContext(
        conversation_id="test_response",
        turn_number=turn,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=StanceCache(),
    )


# =============================================================================
# 1. Prompt Builder Tests
# =============================================================================


class TestPromptBuilder:
    """Test IR → system prompt conversion."""

    def test_all_tones_have_prompt(self):
        """Every Tone enum value must have a prompt mapping."""
        for tone in Tone:
            assert tone.value in TONE_PROMPTS, f"Missing prompt for tone: {tone.value}"

    def test_all_verbosities_have_prompt(self):
        for v in Verbosity:
            assert v.value in VERBOSITY_PROMPTS

    def test_all_uncertainty_actions_have_prompt(self):
        for ua in UncertaintyAction:
            assert ua.value in UNCERTAINTY_PROMPTS

    def test_all_claim_types_have_prompt(self):
        for ct in KnowledgeClaimType:
            assert ct.value in CLAIM_TYPE_PROMPTS

    def test_tone_appears_in_prompt(self):
        ir = make_ir(tone=Tone.ANXIOUS_STRESSED)
        prompt = build_system_prompt(ir)
        assert "anxiety" in prompt.lower() or "stress" in prompt.lower()

    def test_warm_tone_in_prompt(self):
        ir = make_ir(tone=Tone.WARM_ENTHUSIASTIC)
        prompt = build_system_prompt(ir)
        assert "warmth" in prompt.lower() or "enthusiasm" in prompt.lower()

    def test_verbosity_brief_in_prompt(self):
        ir = make_ir(verbosity=Verbosity.BRIEF)
        prompt = build_system_prompt(ir)
        assert "1-2 sentences" in prompt

    def test_verbosity_detailed_in_prompt(self):
        ir = make_ir(verbosity=Verbosity.DETAILED)
        prompt = build_system_prompt(ir)
        assert "6+" in prompt

    def test_high_formality_instruction(self):
        ir = make_ir(formality=0.85)
        prompt = build_system_prompt(ir)
        assert "formal" in prompt.lower()

    def test_low_formality_instruction(self):
        ir = make_ir(formality=0.15)
        prompt = build_system_prompt(ir)
        assert "casual" in prompt.lower()

    def test_high_confidence_instruction(self):
        ir = make_ir(confidence=0.8)
        prompt = build_system_prompt(ir)
        assert "confidence" in prompt.lower()

    def test_low_confidence_instruction(self):
        ir = make_ir(confidence=0.2)
        prompt = build_system_prompt(ir)
        assert "uncertain" in prompt.lower() or "not sure" in prompt.lower()

    def test_stance_appears_in_prompt(self):
        ir = make_ir(stance="AI should augment human creativity")
        prompt = build_system_prompt(ir)
        assert "AI should augment human creativity" in prompt

    def test_rationale_appears_in_prompt(self):
        ir = make_ir(rationale="8 years of UX research experience")
        prompt = build_system_prompt(ir)
        assert "8 years of UX research" in prompt

    def test_intent_appears_in_prompt(self):
        ir = make_ir(intent="Share expertise on UX methodology")
        prompt = build_system_prompt(ir)
        assert "Share expertise on UX methodology" in prompt

    def test_hedge_uncertainty_in_prompt(self):
        ir = make_ir(uncertainty_action=UncertaintyAction.HEDGE)
        prompt = build_system_prompt(ir)
        assert "hedging" in prompt.lower() or "limits" in prompt.lower()

    def test_refuse_uncertainty_in_prompt(self):
        ir = make_ir(uncertainty_action=UncertaintyAction.REFUSE)
        prompt = build_system_prompt(ir)
        assert "decline" in prompt.lower()

    def test_domain_expert_claim_in_prompt(self):
        ir = make_ir(knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT)
        prompt = build_system_prompt(ir)
        assert "professional expertise" in prompt.lower()

    def test_speculative_claim_in_prompt(self):
        ir = make_ir(knowledge_claim_type=KnowledgeClaimType.SPECULATIVE)
        prompt = build_system_prompt(ir)
        assert "speculation" in prompt.lower() or "wonder" in prompt.lower()

    def test_persona_identity_in_prompt(self):
        persona = load_persona()
        ir = make_ir()
        prompt = build_system_prompt(ir, persona=persona)
        assert "Sarah" in prompt
        assert "UX Researcher" in prompt

    def test_no_persona_still_works(self):
        ir = make_ir()
        prompt = build_system_prompt(ir, persona=None)
        assert "COMMUNICATION STYLE" in prompt
        assert "RESPONSE GUIDANCE" in prompt

    def test_safety_blocked_topics(self):
        ir = make_ir()
        ir.safety_plan.blocked_topics = ["employer_name", "participant_data"]
        prompt = build_system_prompt(ir)
        assert "employer_name" in prompt
        assert "participant_data" in prompt
        assert "CRITICAL" in prompt

    def test_different_irs_produce_different_prompts(self):
        ir1 = make_ir(tone=Tone.WARM_ENTHUSIASTIC, confidence=0.9)
        ir2 = make_ir(tone=Tone.ANXIOUS_STRESSED, confidence=0.2)
        prompt1 = build_system_prompt(ir1)
        prompt2 = build_system_prompt(ir2)
        assert prompt1 != prompt2

    def test_elasticity_firm_in_prompt(self):
        ir = make_ir(elasticity=0.1)
        prompt = build_system_prompt(ir)
        assert "firmly" in prompt.lower() or "firm" in prompt.lower()

    def test_elasticity_open_in_prompt(self):
        ir = make_ir(elasticity=0.8)
        prompt = build_system_prompt(ir)
        assert "open-minded" in prompt.lower() or "open" in prompt.lower()

    def test_disclosure_guarded_in_prompt(self):
        ir = make_ir(disclosure_level=0.1)
        prompt = build_system_prompt(ir)
        assert "guarded" in prompt.lower()

    def test_disclosure_open_in_prompt(self):
        ir = make_ir(disclosure_level=0.8)
        prompt = build_system_prompt(ir)
        assert "open" in prompt.lower()


# =============================================================================
# 2. Float-to-Instruction Tests
# =============================================================================


class TestFloatInstructions:
    """Test graduated float → instruction converters."""

    @pytest.mark.parametrize("val,keyword", [
        (0.1, "casual"),
        (0.3, "conversational"),
        (0.6, "professional"),
        (0.9, "formal"),
    ])
    def test_formality_levels(self, val, keyword):
        assert keyword in formality_instruction(val).lower()

    @pytest.mark.parametrize("val,keyword", [
        (0.1, "diplomatic"),
        (0.4, "tact"),
        (0.8, "direct"),
    ])
    def test_directness_levels(self, val, keyword):
        assert keyword in directness_instruction(val).lower()

    @pytest.mark.parametrize("val,keyword", [
        (0.1, "uncertainty"),
        (0.4, "moderate"),
        (0.8, "confidence"),
    ])
    def test_confidence_levels(self, val, keyword):
        assert keyword in confidence_instruction(val).lower()

    @pytest.mark.parametrize("val,keyword", [
        (0.1, "firmly"),
        (0.4, "open to adjusting"),
        (0.8, "open-minded"),
    ])
    def test_elasticity_levels(self, val, keyword):
        assert keyword in elasticity_instruction(val).lower()

    @pytest.mark.parametrize("val,keyword", [
        (0.1, "guarded"),
        (0.4, "relevant personal context"),
        (0.8, "open"),
    ])
    def test_disclosure_levels(self, val, keyword):
        assert keyword in disclosure_instruction(val).lower()


# =============================================================================
# 3. TemplateAdapter Tests
# =============================================================================


class TestTemplateAdapter:
    """Test rule-based text generation from IR.

    The generation/ TemplateAdapter.generate_from_ir() returns str (not
    GeneratedResponse), so assertions are on plain text.
    """

    def test_returns_string(self):
        adapter = TemplateAdapter()
        ir = make_ir()
        result = adapter.generate_from_ir(ir, "test input")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_brief_produces_short_text(self):
        adapter = TemplateAdapter()
        ir = make_ir(verbosity=Verbosity.BRIEF)
        text = adapter.generate_from_ir(ir, "test")
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        assert len(sentences) <= 3  # Allow some flexibility

    def test_detailed_longer_than_brief(self):
        adapter = TemplateAdapter()
        ir_brief = make_ir(verbosity=Verbosity.BRIEF)
        ir_detailed = make_ir(verbosity=Verbosity.DETAILED)
        text_brief = adapter.generate_from_ir(ir_brief, "test")
        text_detailed = adapter.generate_from_ir(ir_detailed, "test")
        assert len(text_detailed) > len(text_brief)

    def test_different_tones_different_openers(self):
        adapter = TemplateAdapter()
        ir_warm = make_ir(tone=Tone.WARM_ENTHUSIASTIC)
        ir_anxious = make_ir(tone=Tone.ANXIOUS_STRESSED)
        text_warm = adapter.generate_from_ir(ir_warm, "test")
        text_anxious = adapter.generate_from_ir(ir_anxious, "test")
        assert text_warm != text_anxious

    def test_low_confidence_adds_hedging(self):
        adapter = TemplateAdapter()
        ir = make_ir(confidence=0.2, stance="this could work")
        text = adapter.generate_from_ir(ir, "test")
        text_lower = text.lower()
        assert "i think" in text_lower or "from what" in text_lower

    def test_high_confidence_no_hedging(self):
        adapter = TemplateAdapter()
        ir = make_ir(confidence=0.9, stance="This approach works well")
        text = adapter.generate_from_ir(ir, "test")
        assert "I think this" not in text
        assert "From what I understand" not in text

    def test_ask_clarifying_adds_question(self):
        adapter = TemplateAdapter()
        ir = make_ir(uncertainty_action=UncertaintyAction.ASK_CLARIFYING)
        text = adapter.generate_from_ir(ir, "test")
        assert "?" in text

    def test_hedge_adds_uncertainty(self):
        adapter = TemplateAdapter()
        ir = make_ir(uncertainty_action=UncertaintyAction.HEDGE)
        text = adapter.generate_from_ir(ir, "test")
        assert "not entirely certain" in text.lower()

    def test_refuse_adds_declination(self):
        adapter = TemplateAdapter()
        ir = make_ir(uncertainty_action=UncertaintyAction.REFUSE)
        text = adapter.generate_from_ir(ir, "test")
        assert "not really the right person" in text.lower()

    def test_stance_appears_in_text(self):
        adapter = TemplateAdapter()
        ir = make_ir(
            stance="User research is essential for good design",
            confidence=0.8,
        )
        text = adapter.generate_from_ir(ir, "test")
        assert "user research" in text.lower()

    def test_high_formality_removes_contractions(self):
        adapter = TemplateAdapter()
        ir = make_ir(
            formality=0.9,
            tone=Tone.NEUTRAL_CALM,
            stance="I'm not sure about this",
        )
        text = adapter.generate_from_ir(ir, "test")
        # High formality should expand "I'm" to "I am"
        assert "I am" in text or "I'm" not in text

    def test_persona_adds_occupation_in_detailed(self):
        adapter = TemplateAdapter()
        persona = load_persona()
        ir = make_ir(verbosity=Verbosity.DETAILED)
        text = adapter.generate_from_ir(ir, "test", persona=persona)
        assert "ux researcher" in text.lower()


# =============================================================================
# 4. Persona Differentiation Tests
# =============================================================================


class TestPersonaDifferentiation:
    """Different personas/states MUST produce different responses."""

    def test_different_personas_different_template_output(self):
        """Two different personas → different IR → different template text."""
        persona1 = load_persona("personas/ux_researcher.yaml")
        persona2 = load_persona("personas/test_high_neuroticism.yaml")

        planner1 = TurnPlanner(persona1, DeterminismManager(seed=42))
        planner2 = TurnPlanner(persona2, DeterminismManager(seed=42))

        ctx1 = make_context("What do you think about remote work?")
        ctx2 = make_context("What do you think about remote work?")

        ir1 = planner1.generate_ir(ctx1)
        ir2 = planner2.generate_ir(ctx2)

        gen1 = ResponseGenerator(persona=persona1, provider="template")
        gen2 = ResponseGenerator(persona=persona2, provider="template")

        resp1 = gen1.generate(ir1, "What do you think about remote work?")
        resp2 = gen2.generate(ir2, "What do you think about remote work?")

        # Different personas should produce different text
        assert resp1.text != resp2.text, (
            "Different personas should produce different template responses"
        )

    def test_different_tones_different_character(self):
        """Same IR but different tones → different opening character."""
        adapter = TemplateAdapter()

        ir_warm = make_ir(tone=Tone.WARM_ENTHUSIASTIC, confidence=0.8)
        ir_tired = make_ir(tone=Tone.TIRED_WITHDRAWN, confidence=0.8)

        text_warm = adapter.generate_from_ir(ir_warm, "test")
        text_tired = adapter.generate_from_ir(ir_tired, "test")

        # Warm should start enthusiastically, tired should be subdued
        assert text_warm != text_tired
        assert "excited" in text_warm.lower() or "great" in text_warm.lower()

    def test_expert_vs_nonexpert_different_confidence(self):
        """Expert domain → high confidence text, unknown domain → hedging text."""
        persona = load_persona()
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        gen = ResponseGenerator(persona=persona, provider="template")

        # Expert domain (psychology)
        ctx_expert = make_context(
            "Tell me about cognitive load in UX design",
            topic="psychology",
        )
        ir_expert = planner.generate_ir(ctx_expert)
        resp_expert = gen.generate(ir_expert, ctx_expert.user_input)

        # Recreate planner to avoid state bleed
        planner2 = TurnPlanner(persona, DeterminismManager(seed=42))
        gen2 = ResponseGenerator(persona=persona, provider="template")

        # Unknown domain
        ctx_unknown = make_context(
            "Explain quantum chromodynamics",
            topic="particle_physics",
        )
        ir_unknown = planner2.generate_ir(ctx_unknown)
        resp_unknown = gen2.generate(ir_unknown, ctx_unknown.user_input)

        # Responses should differ
        assert resp_expert.text != resp_unknown.text

    def test_same_persona_different_moods_over_turns(self):
        """Same persona at turn 1 vs turn 15 should produce different output."""
        persona = load_persona()
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        gen = ResponseGenerator(persona=persona, provider="template")

        cache = StanceCache()

        # Turn 1
        ctx1 = ConversationContext(
            conversation_id="test_mood",
            turn_number=1,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="general",
            user_input="What's on your mind?",
            stance_cache=cache,
        )
        ir1 = planner.generate_ir(ctx1)
        resp1 = gen.generate(ir1, ctx1.user_input)

        # Simulate many turns to build fatigue
        for t in range(2, 16):
            ctx_mid = ConversationContext(
                conversation_id="test_mood",
                turn_number=t,
                interaction_mode=InteractionMode.CASUAL_CHAT,
                goal=ConversationGoal.EXPLORE_IDEAS,
                topic_signature="general",
                user_input="Tell me more.",
                stance_cache=cache,
            )
            planner.generate_ir(ctx_mid)

        # Turn 16
        ctx16 = ConversationContext(
            conversation_id="test_mood",
            turn_number=16,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="general",
            user_input="What's on your mind?",
            stance_cache=cache,
        )
        ir16 = planner.generate_ir(ctx16)
        resp16 = gen.generate(ir16, ctx16.user_input)

        # After 15 turns, fatigue should change the output
        assert resp1.text != resp16.text or ir1.communication_style.tone != ir16.communication_style.tone


# =============================================================================
# 5. Full Pipeline Integration Tests
# =============================================================================


class TestFullPipeline:
    """End-to-end: Persona → TurnPlanner → IR → ResponseGenerator → text."""

    def test_end_to_end_template(self):
        """Full pipeline with template backend produces non-empty text."""
        persona = load_persona()
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        gen = ResponseGenerator(persona=persona, provider="template")

        ctx = make_context("What do you think about AI in UX research?")
        ir = planner.generate_ir(ctx)
        resp = gen.generate(ir, ctx.user_input)

        assert isinstance(resp, GeneratedResponse)
        assert len(resp.text) > 10
        assert resp.model == "template-rule-based"

    def test_deterministic_template_output(self):
        """Same seed + same input → same template output."""
        persona = load_persona()

        planner1 = TurnPlanner(persona, DeterminismManager(seed=42))
        gen1 = ResponseGenerator(persona=persona, provider="template")
        ctx1 = make_context("What do you think about AI?")
        ir1 = planner1.generate_ir(ctx1)
        resp1 = gen1.generate(ir1, ctx1.user_input)

        planner2 = TurnPlanner(persona, DeterminismManager(seed=42))
        gen2 = ResponseGenerator(persona=persona, provider="template")
        ctx2 = make_context("What do you think about AI?")
        ir2 = planner2.generate_ir(ctx2)
        resp2 = gen2.generate(ir2, ctx2.user_input)

        assert resp1.text == resp2.text


# =============================================================================
# 6. Strict Mode (Phase A.2)
# =============================================================================


class TestStrictMode:
    """Strict mode forces TemplateAdapter for deterministic output."""

    def test_strict_mode_forces_template_adapter(self):
        """When strict_mode=True, adapter should be TemplateAdapter."""
        persona = load_persona()
        gen = ResponseGenerator(persona=persona, provider="mock", strict_mode=True)
        assert isinstance(gen.adapter, TemplateAdapter)

    def test_strict_mode_with_explicit_template_adapter(self):
        """Strict mode with explicit TemplateAdapter should keep it."""
        persona = load_persona()
        ta = TemplateAdapter()
        gen = ResponseGenerator(persona=persona, adapter=ta)
        assert gen.adapter is ta

    def test_strict_mode_produces_deterministic_output(self):
        """Same IR + strict mode → identical output every time."""
        persona = load_persona()
        ir = make_ir(confidence=0.8, tone=Tone.WARM_CONFIDENT)

        gen = ResponseGenerator(persona=persona, strict_mode=True)
        resp1 = gen.generate(ir, "Tell me about UX")
        resp2 = gen.generate(ir, "Tell me about UX")
        assert resp1.text == resp2.text

    def test_strict_mode_template_respects_ir_fields(self):
        """In strict mode, output should reflect IR confidence level."""
        persona = load_persona()
        gen = ResponseGenerator(persona=persona, strict_mode=True)

        # Low confidence → hedging language
        ir_low = make_ir(confidence=0.2, tone=Tone.NEUTRAL_CALM)
        resp_low = gen.generate(ir_low, "What do you think?")

        # High confidence → direct language
        ir_high = make_ir(confidence=0.9, tone=Tone.WARM_CONFIDENT)
        resp_high = gen.generate(ir_high, "What do you think?")

        # They should be different (different IR → different template output)
        assert resp_low.text != resp_high.text


# =============================================================================
# 7. LLM Adapter Error Handling (Phase B.1)
# =============================================================================


class TestLLMAdapterErrorHandling:
    """Verify that LLM adapters produce typed exceptions, not raw crashes."""

    def test_generation_adapter_empty_content_raises(self):
        """AnthropicAdapter from generation module: empty content → LLMResponseError."""
        from unittest.mock import MagicMock, patch
        from persona_engine.generation.llm_adapter import AnthropicAdapter
        from persona_engine.exceptions import LLMResponseError

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicAdapter(api_key="test-key")

        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = []  # Empty content
        mock_client.messages.create.return_value = mock_message
        adapter._client = mock_client

        with pytest.raises(LLMResponseError, match="empty response"):
            adapter.generate("system", "user")

    def test_generation_adapter_connection_error(self):
        """AnthropicAdapter from generation module: connection error → LLMConnectionError."""
        from unittest.mock import MagicMock
        from persona_engine.generation.llm_adapter import AnthropicAdapter
        from persona_engine.exceptions import LLMConnectionError

        adapter = AnthropicAdapter(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("network down")
        adapter._client = mock_client

        with pytest.raises(LLMConnectionError, match="connection failed"):
            adapter.generate("system", "user")

    def test_generation_adapter_generic_error(self):
        """AnthropicAdapter: unknown error → LLMResponseError."""
        from unittest.mock import MagicMock
        from persona_engine.generation.llm_adapter import AnthropicAdapter
        from persona_engine.exceptions import LLMResponseError

        adapter = AnthropicAdapter(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ValueError("bad value")
        adapter._client = mock_client

        with pytest.raises(LLMResponseError, match="ValueError"):
            adapter.generate("system", "user")

    def test_openai_adapter_empty_choices_raises(self):
        """OpenAIAdapter: empty choices → LLMResponseError."""
        from unittest.mock import MagicMock
        from persona_engine.generation.llm_adapter import OpenAIAdapter
        from persona_engine.exceptions import LLMResponseError

        adapter = OpenAIAdapter(api_key="test-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        adapter._client = mock_client

        with pytest.raises(LLMResponseError, match="empty response"):
            adapter.generate("system", "user")
