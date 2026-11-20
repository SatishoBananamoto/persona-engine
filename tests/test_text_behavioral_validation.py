"""
Text-Level Behavioral Validation Tests (Phase 8)

Validates that persona traits actually influence the generated text and prompts.
Complements test_response_behavioral.py by focusing on trait-to-text contract:

- TraitGuidance directives appear in the system prompt
- CognitiveGuidance directives appear in the system prompt
- Personality language directives appear in the system prompt
- Trait extremes produce measurably different prompts
- Behavioral directives are non-empty for extreme personas
"""

import pytest
import yaml
from pathlib import Path

from persona_engine.engine import PersonaEngine
from persona_engine.generation.llm_adapter import MockLLMAdapter, TemplateAdapter
from persona_engine.generation.prompt_builder import build_ir_prompt as build_system_prompt
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager

from conftest import make_persona_data


# ============================================================================
# Helpers
# ============================================================================


def _make_planner(persona: Persona, seed: int = 42) -> TurnPlanner:
    """Create a TurnPlanner with deterministic seed."""
    return TurnPlanner(persona, DeterminismManager(seed=seed))


def _generate_ir(planner: TurnPlanner, user_input: str, turn: int = 1):
    """Generate IR for a single turn."""
    ctx = ConversationContext(
        conversation_id="test-validation",
        turn_number=turn,
        interaction_mode=None,
        goal=None,
        topic_signature="test_topic",
        user_input=user_input,
        stance_cache=StanceCache(),
    )
    return planner.generate_ir(ctx)


def _load_persona(path: str) -> Persona:
    data = yaml.safe_load(Path(path).read_text())
    return Persona(**data)


# ============================================================================
# Trait-Driven Behavioral Directives in IR
# ============================================================================


class TestTraitDrivenDirectives:
    """Verify that trait extremes produce behavioral directives in the IR."""

    def test_high_agreeableness_produces_validation_directive(self):
        """High-A (>0.7) should generate 'Acknowledge the other person's point' directive."""
        data = make_persona_data(agreeableness=0.85)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "I think Python is terrible for data science")

        directives = ir.behavioral_directives
        has_validation = any("acknowledge" in d.lower() for d in directives)
        assert has_validation, (
            f"High-A persona should have validation directive. "
            f"Got directives: {directives}"
        )

    def test_low_agreeableness_no_validation_directive(self):
        """Low-A (<0.5) should NOT generate validation directive."""
        data = make_persona_data(agreeableness=0.3)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "I think Python is terrible for data science")

        directives = ir.behavioral_directives
        has_validation = any("acknowledge" in d.lower() for d in directives)
        assert not has_validation, (
            f"Low-A persona should not have validation directive. "
            f"Got: {directives}"
        )

    def test_high_openness_produces_abstract_directive(self):
        """High-O (>0.7) should generate abstract/metaphor directive."""
        data = make_persona_data(openness=0.85)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "What do you think about code quality?")

        directives = ir.behavioral_directives
        has_abstract = any(
            "metaphor" in d.lower() or "abstract" in d.lower() or "analogies" in d.lower()
            for d in directives
        )
        assert has_abstract, (
            f"High-O persona should have abstract reasoning directive. "
            f"Got: {directives}"
        )

    def test_low_openness_no_abstract_directive(self):
        """Low-O (<0.5) should NOT generate abstract/metaphor directive."""
        data = make_persona_data(openness=0.3)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "What do you think about code quality?")

        directives = ir.behavioral_directives
        has_abstract = any(
            "metaphor" in d.lower() or "abstract" in d.lower()
            for d in directives
        )
        assert not has_abstract

    def test_high_extraversion_produces_proactive_directive(self):
        """High-E (>0.7) should generate follow-up question directive."""
        data = make_persona_data(extraversion=0.85)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Tell me about agile methodologies")

        directives = ir.behavioral_directives
        has_followup = any("follow-up" in d.lower() or "follow up" in d.lower() for d in directives)
        assert has_followup, (
            f"High-E persona should have proactive follow-up directive. "
            f"Got: {directives}"
        )

    def test_low_extraversion_no_proactive_directive(self):
        """Low-E (<0.5) should NOT generate proactive follow-up directive."""
        data = make_persona_data(extraversion=0.3)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Tell me about agile methodologies")

        directives = ir.behavioral_directives
        has_followup = any("follow-up" in d.lower() or "follow up" in d.lower() for d in directives)
        assert not has_followup

    def test_high_hedging_produces_hedging_directive(self):
        """High-A (hedging_level > 0.4) should generate hedging directive."""
        data = make_persona_data(agreeableness=0.8)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Should we use microservices?")

        directives = ir.behavioral_directives
        has_hedging = any("hedging" in d.lower() or "I think" in d for d in directives)
        assert has_hedging, (
            f"High-A persona should have hedging directive. "
            f"Got: {directives}"
        )


# ============================================================================
# Cognitive Guidance in IR
# ============================================================================


class TestCognitiveDirectives:
    """Verify cognitive style extremes produce appropriate directives."""

    def test_analytical_style_produces_analytical_directive(self):
        """High analytical style should produce step-by-step reasoning directive."""
        data = make_persona_data(cog_analytical_intuitive=0.85)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "What's the best deployment strategy?")

        directives = ir.behavioral_directives
        has_analytical = any(
            "step by step" in d.lower() or "logical" in d.lower()
            for d in directives
        )
        assert has_analytical, (
            f"High-analytical persona should have step-by-step directive. "
            f"Got: {directives}"
        )

    def test_intuitive_style_produces_intuitive_directive(self):
        """Low analytical style should produce gut-feeling directive."""
        data = make_persona_data(cog_analytical_intuitive=0.15)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "What's the best deployment strategy?")

        directives = ir.behavioral_directives
        has_intuitive = any(
            "gut feeling" in d.lower() or "instinct" in d.lower()
            for d in directives
        )
        assert has_intuitive, (
            f"Low-analytical persona should have intuitive directive. "
            f"Got: {directives}"
        )

    def test_high_cognitive_complexity_produces_nuance_directive(self):
        """High cognitive complexity should produce nuance directive."""
        data = make_persona_data(cog_cognitive_complexity=0.85)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Is remote work better than office?")

        directives = ir.behavioral_directives
        has_nuance = any(
            "nuanced" in d.lower() or "multifaceted" in d.lower() or "edge cases" in d.lower()
            for d in directives
        )
        assert has_nuance, (
            f"High cognitive complexity persona should have nuance directive. "
            f"Got: {directives}"
        )

    def test_low_cognitive_complexity_produces_decisive_directive(self):
        """Low cognitive complexity should produce decisive directive."""
        data = make_persona_data(cog_cognitive_complexity=0.15)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Is remote work better than office?")

        directives = ir.behavioral_directives
        has_decisive = any(
            "decisive" in d.lower() or "clear" in d.lower() or "black-and-white" in d.lower()
            for d in directives
        )
        assert has_decisive, (
            f"Low cognitive complexity persona should have decisive directive. "
            f"Got: {directives}"
        )


# ============================================================================
# Directives Flow Into System Prompt
# ============================================================================


class TestDirectivesInIRAndPrompt:
    """Verify behavioral directives flow from traits into IR and generation prompt."""

    def test_behavioral_directives_populated_for_extreme_traits(self):
        """Extreme traits should produce non-empty behavioral_directives in IR."""
        data = make_persona_data(openness=0.9, agreeableness=0.9, extraversion=0.9)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Tell me about system design")

        assert len(ir.behavioral_directives) > 0, "Extreme traits should produce directives"

    def test_personality_language_populated_for_extreme_traits(self):
        """High-openness persona should have personality language directives in IR."""
        data = make_persona_data(openness=0.9)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Tell me about creativity")

        assert len(ir.personality_language) > 0, (
            "High-openness persona should have personality language directives"
        )

    def test_behavioral_directives_in_generation_prompt(self):
        """IR behavioral_directives should be included in the IRPromptBuilder prompt."""
        from persona_engine.generation.prompt_builder import IRPromptBuilder

        data = make_persona_data(openness=0.9, agreeableness=0.9, extraversion=0.9)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Tell me about system design")

        builder = IRPromptBuilder()
        prompt = builder.build_generation_prompt(
            ir=ir,
            user_input="Tell me about system design",
            persona=persona,
            behavioral_directives=ir.behavioral_directives,
        )

        found_any = any(d in prompt for d in ir.behavioral_directives)
        assert found_any, (
            f"Behavioral directives should appear in generation prompt. "
            f"Directives: {ir.behavioral_directives[:3]}"
        )

    def test_personality_language_in_generation_prompt(self):
        """IR personality_language should be included in the generation prompt."""
        from persona_engine.generation.prompt_builder import IRPromptBuilder

        data = make_persona_data(openness=0.9)
        persona = Persona(**data)
        planner = _make_planner(persona)
        ir = _generate_ir(planner, "Tell me about creativity")

        builder = IRPromptBuilder()
        prompt = builder.build_generation_prompt(
            ir=ir,
            user_input="Tell me about creativity",
            persona=persona,
        )

        found_any = any(pl in prompt for pl in ir.personality_language)
        assert found_any, (
            f"Personality language should appear in generation prompt. "
            f"Language: {ir.personality_language[:3]}"
        )


# ============================================================================
# Trait Contrast Tests (Counterfactual)
# ============================================================================


class TestTraitContrastInText:
    """Verify that different trait levels produce measurably different outputs."""

    def test_high_vs_low_openness_different_directives(self):
        """High-O and low-O should produce different behavioral directives."""
        high_o = Persona(**make_persona_data(openness=0.9))
        low_o = Persona(**make_persona_data(openness=0.1))

        ir_high = _generate_ir(_make_planner(high_o), "What do you think of modern art?")
        ir_low = _generate_ir(_make_planner(low_o), "What do you think of modern art?")

        assert ir_high.behavioral_directives != ir_low.behavioral_directives, (
            "High-O and low-O should produce different behavioral directives"
        )

    def test_high_vs_low_agreeableness_different_directness(self):
        """High-A should have lower directness than low-A."""
        high_a = Persona(**make_persona_data(agreeableness=0.9))
        low_a = Persona(**make_persona_data(agreeableness=0.1))

        ir_high = _generate_ir(_make_planner(high_a), "Your approach is completely wrong")
        ir_low = _generate_ir(_make_planner(low_a), "Your approach is completely wrong")

        assert ir_high.communication_style.directness < ir_low.communication_style.directness, (
            f"High-A directness ({ir_high.communication_style.directness:.2f}) "
            f"should be lower than low-A ({ir_low.communication_style.directness:.2f})"
        )

    def test_high_vs_low_extraversion_different_tone(self):
        """High-E and low-E should produce different tones on the same input."""
        high_e = Persona(**make_persona_data(extraversion=0.9))
        low_e = Persona(**make_persona_data(extraversion=0.1))

        ir_high = _generate_ir(_make_planner(high_e), "Tell me about your favorite project")
        ir_low = _generate_ir(_make_planner(low_e), "Tell me about your favorite project")

        # High-E should have a warmer/more enthusiastic tone
        warm_tones = {"warm_enthusiastic", "excited_engaged", "warm_confident", "friendly_relaxed"}
        assert ir_high.communication_style.tone.value in warm_tones or \
               ir_high.communication_style.tone != ir_low.communication_style.tone, (
            f"High-E tone ({ir_high.communication_style.tone}) should differ from "
            f"low-E tone ({ir_low.communication_style.tone})"
        )

    def test_high_vs_low_neuroticism_different_prompts(self):
        """High-N and low-N should produce different system prompts."""
        high_n = Persona(**make_persona_data(neuroticism=0.9))
        low_n = Persona(**make_persona_data(neuroticism=0.1))

        ir_high = _generate_ir(_make_planner(high_n), "Everything is going wrong at work")
        ir_low = _generate_ir(_make_planner(low_n), "Everything is going wrong at work")

        prompt_high = build_system_prompt(ir_high, high_n)
        prompt_low = build_system_prompt(ir_low, low_n)

        assert prompt_high != prompt_low, (
            "High-N and low-N should produce different system prompts"
        )


# ============================================================================
# End-to-End: PersonaEngine with Template Provider
# ============================================================================


class TestEndToEndTextValidation:
    """Full pipeline text validation using template provider."""

    @pytest.fixture
    def high_openness_engine(self) -> PersonaEngine:
        data = make_persona_data(openness=0.9, extraversion=0.8)
        persona = Persona(**data)
        return PersonaEngine(
            persona=persona,
            adapter=MockLLMAdapter(),
            seed=42,
        )

    @pytest.fixture
    def low_openness_engine(self) -> PersonaEngine:
        data = make_persona_data(openness=0.1, extraversion=0.2)
        persona = Persona(**data)
        return PersonaEngine(
            persona=persona,
            adapter=MockLLMAdapter(),
            seed=42,
        )

    def test_engines_produce_different_ir(self, high_openness_engine, low_openness_engine):
        """Two contrasting engines should produce different IR on same input."""
        ir_high = high_openness_engine.plan("What do you think about AI art?")
        ir_low = low_openness_engine.plan("What do you think about AI art?")

        # Behavioral directives should differ
        assert ir_high.behavioral_directives != ir_low.behavioral_directives

    def test_engine_ir_has_personality_language(self, high_openness_engine):
        """Engine-produced IR should include personality language directives."""
        ir = high_openness_engine.plan("Tell me about creativity in engineering")
        assert len(ir.personality_language) > 0, (
            "High-openness engine should produce personality language directives"
        )

    def test_engine_directives_count_scales_with_trait_extremity(self):
        """More extreme traits should produce more directives."""
        extreme = Persona(**make_persona_data(
            openness=0.95, agreeableness=0.95, extraversion=0.95
        ))
        moderate = Persona(**make_persona_data(
            openness=0.5, agreeableness=0.5, extraversion=0.5
        ))

        engine_ext = PersonaEngine(persona=extreme, adapter=MockLLMAdapter(), seed=42)
        engine_mod = PersonaEngine(persona=moderate, adapter=MockLLMAdapter(), seed=42)

        ir_ext = engine_ext.plan("What's your take on team management?")
        ir_mod = engine_mod.plan("What's your take on team management?")

        assert len(ir_ext.behavioral_directives) >= len(ir_mod.behavioral_directives), (
            f"Extreme persona should have >= directives ({len(ir_ext.behavioral_directives)}) "
            f"than moderate ({len(ir_mod.behavioral_directives)})"
        )
