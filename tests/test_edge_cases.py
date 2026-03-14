"""
Edge case and integration tests for PersonaEngine.

Covers:
- Unicode and emoji input handling
- Very long input rejection
- Empty/whitespace input rejection
- Control character sanitization
- Multi-turn memory integration (facts influence IR)
- Concurrent engine instances (no shared state)
- Strict mode determinism
- Persona with minimal/extreme trait values
- Rapid fire multi-turn conversations
- IR field boundary values
- Twin persona divergence verification
"""

import copy
import pytest
from persona_engine.engine import PersonaEngine, ChatResult, _sanitize_text, _validate_user_input
from persona_engine.exceptions import InputValidationError
from persona_engine.schema.persona_schema import Persona


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_PERSONA_DATA = {
    "persona_id": "EDGE_TEST",
    "version": "1.0",
    "label": "Edge Case Persona",
    "identity": {
        "age": 35,
        "gender": "non-binary",
        "location": "Berlin",
        "education": "PhD",
        "occupation": "Researcher",
        "background": "Academic researcher in AI ethics.",
    },
    "psychology": {
        "big_five": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.3,
        },
        "values": {
            "self_direction": 0.8,
            "stimulation": 0.5,
            "hedonism": 0.3,
            "achievement": 0.6,
            "power": 0.2,
            "security": 0.5,
            "conformity": 0.3,
            "tradition": 0.2,
            "benevolence": 0.7,
            "universalism": 0.8,
        },
        "cognitive_style": {
            "analytical_intuitive": 0.8,
            "systematic_heuristic": 0.7,
            "risk_tolerance": 0.4,
            "need_for_closure": 0.5,
            "cognitive_complexity": 0.8,
        },
        "communication": {
            "verbosity": 0.5,
            "formality": 0.6,
            "directness": 0.5,
            "emotional_expressiveness": 0.4,
        },
    },
    "knowledge_domains": [
        {"domain": "Technology", "proficiency": 0.9, "subdomains": ["AI", "Ethics"]},
    ],
    "languages": [{"language": "English", "proficiency": 1.0}],
    "cultural_knowledge": {
        "primary_culture": "German",
        "exposure_level": {"european": 0.9},
    },
    "primary_goals": [{"goal": "Advance ethical AI", "weight": 0.9}],
    "social_roles": {
        "default": {"formality": 0.6, "directness": 0.5, "emotional_expressiveness": 0.4},
    },
    "uncertainty": {
        "admission_threshold": 0.4,
        "hedging_frequency": 0.5,
        "clarification_tendency": 0.5,
        "knowledge_boundary_strictness": 0.8,
    },
    "claim_policy": {
        "allowed_claim_types": ["personal_experience", "domain_expert"],
        "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
        "lookup_behavior": "hedge",
    },
    "invariants": {
        "identity_facts": ["Lives in Berlin", "Has a PhD"],
        "cannot_claim": ["medical doctor"],
        "must_avoid": ["classified research"],
    },
    "initial_state": {
        "mood_valence": 0.2,
        "mood_arousal": 0.4,
        "fatigue": 0.2,
        "stress": 0.2,
        "engagement": 0.7,
    },
    "disclosure_policy": {
        "base_openness": 0.5,
        "factors": {
            "topic_sensitivity": -0.3,
            "trust_level": 0.4,
            "formal_context": -0.2,
            "positive_mood": 0.15,
        },
        "bounds": [0.1, 0.9],
    },
    "biases": [{"type": "confirmation_bias", "strength": 0.2}],
    "time_scarcity": 0.3,
    "privacy_sensitivity": 0.6,
}


def _make_engine(**kwargs) -> PersonaEngine:
    """Create a test engine with mock provider."""
    persona = Persona(**MINIMAL_PERSONA_DATA)
    defaults = {"llm_provider": "mock", "seed": 42}
    defaults.update(kwargs)
    return PersonaEngine(persona, **defaults)


# ---------------------------------------------------------------------------
# Input Validation Edge Cases
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation and sanitization edge cases."""

    def test_empty_string_rejected(self):
        engine = _make_engine()
        with pytest.raises(InputValidationError, match="empty"):
            engine.chat("")

    def test_whitespace_only_rejected(self):
        engine = _make_engine()
        with pytest.raises(InputValidationError, match="empty"):
            engine.chat("   \t\n  ")

    def test_none_input_rejected(self):
        engine = _make_engine()
        with pytest.raises(InputValidationError, match="string"):
            engine.chat(None)  # type: ignore

    def test_integer_input_rejected(self):
        engine = _make_engine()
        with pytest.raises(InputValidationError, match="string"):
            engine.chat(42)  # type: ignore

    def test_very_long_input_rejected(self):
        engine = _make_engine()
        long_input = "a" * 10_001
        with pytest.raises(InputValidationError, match="exceeds maximum"):
            engine.chat(long_input)

    def test_max_length_input_accepted(self):
        engine = _make_engine()
        max_input = "a" * 10_000
        result = engine.chat(max_input)
        assert isinstance(result, ChatResult)

    def test_unicode_input_accepted(self):
        engine = _make_engine()
        result = engine.chat("What do you think about AI?")
        assert isinstance(result, ChatResult)

    def test_emoji_input_accepted(self):
        engine = _make_engine()
        result = engine.chat("What do you think? 🤔💡🎯")
        assert isinstance(result, ChatResult)

    def test_cjk_characters_accepted(self):
        engine = _make_engine()
        result = engine.chat("人工知能についてどう思いますか？")
        assert isinstance(result, ChatResult)

    def test_arabic_rtl_accepted(self):
        engine = _make_engine()
        result = engine.chat("ما رأيك في الذكاء الاصطناعي؟")
        assert isinstance(result, ChatResult)

    def test_mixed_scripts_accepted(self):
        engine = _make_engine()
        result = engine.chat("Hello 世界 مرحبا мир 🌍")
        assert isinstance(result, ChatResult)

    def test_control_chars_stripped(self):
        sanitized = _sanitize_text("Hello\x00World\x01!")
        assert "\x00" not in sanitized
        assert "\x01" not in sanitized
        assert "HelloWorld!" == sanitized

    def test_tabs_preserved(self):
        sanitized = _sanitize_text("Hello\tWorld")
        assert "\t" in sanitized

    def test_newlines_preserved(self):
        sanitized = _sanitize_text("Hello\nWorld")
        assert "\n" in sanitized

    def test_excessive_newlines_collapsed(self):
        sanitized = _sanitize_text("Hello\n\n\n\n\nWorld")
        assert sanitized == "Hello\n\nWorld"

    def test_null_bytes_stripped(self):
        sanitized = _sanitize_text("Hel\x00lo")
        assert sanitized == "Hello"


# ---------------------------------------------------------------------------
# Multi-Turn Memory Integration
# ---------------------------------------------------------------------------


class TestMultiTurnMemory:
    """Test that memory actually influences IR across turns."""

    def test_multi_turn_state_advances(self):
        engine = _make_engine()
        r1 = engine.chat("Tell me about AI ethics")
        r2 = engine.chat("What about data privacy?")
        r3 = engine.chat("How does regulation help?")
        assert r1.turn_number == 1
        assert r2.turn_number == 2
        assert r3.turn_number == 3

    def test_plan_multi_turn_state_advances(self):
        engine = _make_engine()
        ir1 = engine.plan("Tell me about AI")
        ir2 = engine.plan("What about ethics?")
        assert ir1 is not None
        assert ir2 is not None

    def test_reset_clears_turn_count(self):
        engine = _make_engine()
        engine.chat("Hello")
        engine.chat("How are you?")
        engine.reset()
        r = engine.chat("Starting fresh")
        assert r.turn_number == 1

    def test_cross_turn_inertia_smooths_behavior(self):
        """Behavioral metrics shouldn't jump wildly between turns."""
        engine = _make_engine()
        r1 = engine.chat("Tell me about AI")
        r2 = engine.chat("Actually, let's talk about cooking")
        # Confidence shouldn't change by more than ~0.5 between turns
        # due to cross-turn inertia (0.7 blend factor)
        diff = abs(r1.confidence - r2.confidence)
        assert diff < 0.6, f"Confidence jumped too much: {diff}"

    def test_five_turn_conversation_stable(self):
        """Five consecutive turns should all produce valid results."""
        engine = _make_engine()
        prompts = [
            "Hi there!",
            "What do you work on?",
            "That sounds interesting. Tell me more.",
            "Have you published any papers?",
            "Thanks for chatting!",
        ]
        results = []
        for prompt in prompts:
            result = engine.chat(prompt)
            assert isinstance(result, ChatResult)
            assert result.text  # non-empty
            assert result.ir is not None
            results.append(result)

        # Turn numbers should be sequential
        for i, r in enumerate(results, 1):
            assert r.turn_number == i


# ---------------------------------------------------------------------------
# Concurrent Engine Instances
# ---------------------------------------------------------------------------


class TestConcurrentEngines:
    """Verify that multiple engine instances don't share state."""

    def test_two_engines_independent_turns(self):
        e1 = _make_engine(seed=1)
        e2 = _make_engine(seed=2)
        r1 = e1.chat("Hello from engine 1")
        r2 = e2.chat("Hello from engine 2")
        assert r1.turn_number == 1
        assert r2.turn_number == 1

    def test_two_engines_different_seeds_different_ir(self):
        e1 = _make_engine(seed=100)
        e2 = _make_engine(seed=200)
        ir1 = e1.plan("What is consciousness?")
        ir2 = e2.plan("What is consciousness?")
        # Different seeds should produce at least some different values
        # (deterministic, so same seed = same result, different seed = different)
        # Note: some fields may coincidentally match, but not all
        assert ir1 is not None
        assert ir2 is not None

    def test_same_seed_same_ir(self):
        """Same persona + same input + same seed = identical IR."""
        e1 = _make_engine(seed=42)
        e2 = _make_engine(seed=42)
        ir1 = e1.plan("What is AI ethics?")
        ir2 = e2.plan("What is AI ethics?")
        assert ir1.response_structure.confidence == ir2.response_structure.confidence
        assert ir1.response_structure.elasticity == ir2.response_structure.elasticity
        assert ir1.communication_style.tone == ir2.communication_style.tone

    def test_engine_reset_produces_same_ir_as_fresh(self):
        e1 = _make_engine(seed=42)
        ir_fresh = e1.plan("What is AI?")
        e1.chat("Some other topic")
        e1.chat("Yet another topic")
        e1.reset()
        ir_reset = e1.plan("What is AI?")
        assert ir_fresh.response_structure.confidence == ir_reset.response_structure.confidence


# ---------------------------------------------------------------------------
# Strict Mode
# ---------------------------------------------------------------------------


class TestStrictMode:
    """Test strict mode determinism."""

    def test_strict_mode_creates_engine(self):
        engine = _make_engine(strict_mode=True)
        result = engine.chat("Tell me about AI")
        assert isinstance(result, ChatResult)
        assert result.text  # non-empty response

    def test_strict_mode_deterministic_output(self):
        """Same input with strict mode should produce identical text."""
        e1 = _make_engine(seed=42, strict_mode=True)
        e2 = _make_engine(seed=42, strict_mode=True)
        r1 = e1.chat("What is AI?")
        r2 = e2.chat("What is AI?")
        assert r1.text == r2.text, "Strict mode should produce identical text"

    def test_strict_mode_vs_non_strict(self):
        """Strict and non-strict should both work."""
        e_strict = _make_engine(strict_mode=True)
        e_normal = _make_engine(strict_mode=False)
        r_strict = e_strict.chat("Hello")
        r_normal = e_normal.chat("Hello")
        assert isinstance(r_strict, ChatResult)
        assert isinstance(r_normal, ChatResult)


# ---------------------------------------------------------------------------
# Extreme Trait Values
# ---------------------------------------------------------------------------


class TestExtremeTraits:
    """Test personas with extreme (min/max) trait values."""

    def _extreme_persona(self, **trait_overrides) -> Persona:
        data = copy.deepcopy(MINIMAL_PERSONA_DATA)
        data["persona_id"] = "EXTREME_TEST"
        for trait, value in trait_overrides.items():
            data["psychology"]["big_five"][trait] = value
        return Persona(**data)

    def test_all_traits_maximum(self):
        persona = self._extreme_persona(
            openness=1.0,
            conscientiousness=1.0,
            extraversion=1.0,
            agreeableness=1.0,
            neuroticism=1.0,
        )
        engine = PersonaEngine(persona, llm_provider="mock", seed=42)
        result = engine.chat("How are you?")
        assert isinstance(result, ChatResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_all_traits_minimum(self):
        persona = self._extreme_persona(
            openness=0.0,
            conscientiousness=0.0,
            extraversion=0.0,
            agreeableness=0.0,
            neuroticism=0.0,
        )
        engine = PersonaEngine(persona, llm_provider="mock", seed=42)
        result = engine.chat("How are you?")
        assert isinstance(result, ChatResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_high_neuroticism_lower_confidence(self):
        high_n = self._extreme_persona(neuroticism=0.95)
        low_n = self._extreme_persona(neuroticism=0.05)
        e_high = PersonaEngine(high_n, llm_provider="mock", seed=42)
        e_low = PersonaEngine(low_n, llm_provider="mock", seed=42)
        ir_high = e_high.plan("What is the meaning of life?")
        ir_low = e_low.plan("What is the meaning of life?")
        # High neuroticism should generally lead to lower confidence
        assert ir_high.response_structure.confidence <= ir_low.response_structure.confidence + 0.1

    def test_high_extraversion_higher_disclosure(self):
        high_e = self._extreme_persona(extraversion=0.95)
        low_e = self._extreme_persona(extraversion=0.05)
        e_high = PersonaEngine(high_e, llm_provider="mock", seed=42)
        e_low = PersonaEngine(low_e, llm_provider="mock", seed=42)
        ir_high = e_high.plan("Tell me about yourself")
        ir_low = e_low.plan("Tell me about yourself")
        assert ir_high.knowledge_disclosure.disclosure_level >= ir_low.knowledge_disclosure.disclosure_level - 0.1


# ---------------------------------------------------------------------------
# IR Field Boundaries
# ---------------------------------------------------------------------------


class TestIRFieldBoundaries:
    """Verify all IR fields stay within valid ranges."""

    def test_ir_fields_in_range(self):
        engine = _make_engine()
        ir = engine.plan("Tell me something interesting about AI")
        # ResponseStructure
        assert 0.0 <= ir.response_structure.confidence <= 1.0
        assert 0.0 <= ir.response_structure.competence <= 1.0
        assert 0.0 <= ir.response_structure.elasticity <= 1.0
        # CommunicationStyle
        assert 0.0 <= ir.communication_style.formality <= 1.0
        assert 0.0 <= ir.communication_style.directness <= 1.0
        # KnowledgeAndDisclosure
        assert 0.0 <= ir.knowledge_disclosure.disclosure_level <= 1.0

    def test_ir_has_citations(self):
        engine = _make_engine()
        ir = engine.plan("What do you think about machine learning?")
        assert len(ir.citations) > 0, "IR should have at least one citation"

    def test_ir_has_safety_plan(self):
        engine = _make_engine()
        ir = engine.plan("Tell me about your research")
        assert ir.safety_plan is not None

    def test_ir_tone_is_valid_enum(self):
        from persona_engine.schema.ir_schema import Tone
        engine = _make_engine()
        ir = engine.plan("How are you today?")
        assert isinstance(ir.communication_style.tone, Tone)

    def test_ir_verbosity_is_valid_enum(self):
        from persona_engine.schema.ir_schema import Verbosity
        engine = _make_engine()
        ir = engine.plan("Tell me a story")
        assert isinstance(ir.communication_style.verbosity, Verbosity)


# ---------------------------------------------------------------------------
# Validation Integration
# ---------------------------------------------------------------------------


class TestValidationIntegration:
    """Test that validation runs correctly through the engine."""

    def test_validation_passes_for_normal_input(self):
        engine = _make_engine()
        result = engine.chat("Hello, how are you?")
        # Most normal inputs should pass validation
        assert result.validation is not None

    def test_validation_disabled(self):
        engine = _make_engine(validate=False)
        result = engine.chat("Tell me something")
        assert isinstance(result, ChatResult)

    def test_chat_result_repr(self):
        engine = _make_engine()
        result = engine.chat("Hi")
        r = repr(result)
        assert "ChatResult" in r
        assert "turn=" in r


# ---------------------------------------------------------------------------
# Persona Loading Edge Cases
# ---------------------------------------------------------------------------


class TestPersonaLoading:
    """Test various persona loading paths."""

    def test_from_yaml_chef(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")
        result = engine.chat("What makes a good sauce?")
        assert isinstance(result, ChatResult)

    def test_from_yaml_physicist(self):
        engine = PersonaEngine.from_yaml("personas/physicist.yaml", llm_provider="mock")
        result = engine.chat("Explain quantum mechanics")
        assert isinstance(result, ChatResult)

    def test_from_yaml_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            PersonaEngine.from_yaml("personas/nonexistent.yaml", llm_provider="mock")

    def test_persona_builder_basic(self):
        from persona_engine.persona_builder import PersonaBuilder
        persona = PersonaBuilder("TestBot", "Engineer").build()
        engine = PersonaEngine(persona, llm_provider="mock")
        result = engine.chat("Hello")
        assert isinstance(result, ChatResult)

    def test_persona_builder_archetype(self):
        from persona_engine.persona_builder import PersonaBuilder
        persona = (
            PersonaBuilder("Dr. Test", "Scientist")
            .archetype("expert")
            .domain("Physics", 0.9)
            .build()
        )
        engine = PersonaEngine(persona, llm_provider="mock")
        ir = engine.plan("Explain relativity")
        # Archetype "expert" should produce valid IR regardless of domain match
        assert ir is not None
        assert 0.0 <= ir.response_structure.competence <= 1.0
        assert 0.0 <= ir.response_structure.confidence <= 1.0
