"""
Phase R1 Tests: Activate Dead Psychology

Tests that all formerly-orphaned trait, cognitive, and value methods are now
wired into the pipeline and produce observable behavioral differences.

Target: 40+ tests covering:
- TraitGuidance computation and effects
- CognitiveGuidance computation and effects
- Value conflict resolution in stance generation
- Decision policy wiring
- Behavioral directives in IR and prompt
"""

import copy
import pytest
import yaml

from conftest import make_persona_data

from persona_engine.behavioral.trait_interpreter import TraitInterpreter
from persona_engine.behavioral.cognitive_interpreter import CognitiveStyleInterpreter
from persona_engine.behavioral.values_interpreter import ValuesInterpreter
from persona_engine.memory import StanceCache
from persona_engine.planner.stance_generator import generate_stance_safe, _value_short_description
from persona_engine.planner.trace_context import TraceContext
from persona_engine.planner.turn_planner import (
    ConversationContext,
    CognitiveGuidance,
    TraitGuidance,
    TurnPlanner,
)
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    Tone,
)
from persona_engine.schema.persona_schema import (
    BigFiveTraits,
    CognitiveStyle,
    Persona,
    SchwartzValues,
)
from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Helpers
# ============================================================================

def _load_persona(yaml_path: str) -> Persona:
    with open(yaml_path) as f:
        return Persona(**yaml.safe_load(f))


def _make_context(user_input: str = "What do you think?") -> ConversationContext:
    return ConversationContext(
        conversation_id="test",
        turn_number=1,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.EXPLORE_IDEAS,
        topic_signature="general",
        user_input=user_input,
        stance_cache=StanceCache(),
    )


def _generate_ir(persona_data: dict, user_input: str = "What do you think?"):
    persona = Persona(**persona_data)
    planner = TurnPlanner(persona, DeterminismManager(seed=42))
    ctx = _make_context(user_input)
    return planner.generate_ir(ctx)


# ============================================================================
# TraitGuidance Tests
# ============================================================================

class TestTraitGuidanceComputation:
    """Tests for _compute_trait_guidance."""

    def test_high_agreeableness_validates_first(self):
        """High-A persona should have should_validate_first=True."""
        ir = _generate_ir(make_persona_data(agreeableness=0.85))
        # Check behavioral directives contain validation language
        assert any("acknowledge" in d.lower() or "point" in d.lower()
                    for d in ir.behavioral_directives)

    def test_low_agreeableness_no_validation(self):
        """Low-A persona should NOT validate first."""
        ir = _generate_ir(make_persona_data(agreeableness=0.2))
        assert not any("acknowledge" in d.lower() for d in ir.behavioral_directives)

    def test_high_agreeableness_hedging(self):
        """High-A persona should have hedging directives."""
        ir = _generate_ir(make_persona_data(agreeableness=0.85))
        assert any("hedging" in d.lower() or "perhaps" in d.lower()
                    for d in ir.behavioral_directives)

    def test_low_agreeableness_no_hedging(self):
        """Low-A should not have hedging."""
        ir = _generate_ir(make_persona_data(agreeableness=0.2))
        assert not any("hedging" in d.lower() for d in ir.behavioral_directives)

    def test_high_extraversion_proactive_followup(self):
        """High-E persona should have proactive follow-up directive."""
        ir = _generate_ir(make_persona_data(extraversion=0.85))
        assert any("follow-up" in d.lower() or "question" in d.lower()
                    for d in ir.behavioral_directives)

    def test_low_extraversion_no_followup(self):
        """Low-E persona should NOT have proactive follow-up."""
        ir = _generate_ir(make_persona_data(extraversion=0.2))
        assert not any("follow-up" in d.lower() for d in ir.behavioral_directives)

    def test_high_openness_abstract_language(self):
        """High-O persona should prefer abstract/metaphorical language."""
        ir = _generate_ir(make_persona_data(openness=0.85))
        assert any("metaphor" in d.lower() or "abstract" in d.lower()
                    for d in ir.behavioral_directives)

    def test_low_openness_no_abstract(self):
        """Low-O persona should NOT have abstract language directive."""
        ir = _generate_ir(make_persona_data(openness=0.2))
        assert not any("metaphor" in d.lower() for d in ir.behavioral_directives)

    def test_high_openness_novelty(self):
        """High-O persona should prefer novelty."""
        ir = _generate_ir(make_persona_data(openness=0.85))
        assert any("creative" in d.lower() or "unconventional" in d.lower()
                    for d in ir.behavioral_directives)

    def test_trait_guidance_citation_exists(self):
        """TraitGuidance computation should generate a citation."""
        ir = _generate_ir(make_persona_data(agreeableness=0.85))
        trait_citations = [c for c in ir.citations
                          if c.source_id == "trait_guidance"]
        assert len(trait_citations) >= 1


class TestTraitGuidanceEffects:
    """Tests that trait guidance produces observable IR differences."""

    def test_enthusiasm_boost_affects_tone(self):
        """High-E persona should get arousal boost affecting tone selection."""
        ir_high_e = _generate_ir(make_persona_data(extraversion=0.9))
        ir_low_e = _generate_ir(make_persona_data(extraversion=0.15))
        # High-E should have warmer/more engaged tone OR same tone is fine
        # but there should be an enthusiasm citation
        high_e_cites = [c for c in ir_high_e.citations
                       if c.source_id == "extraversion_enthusiasm"]
        low_e_cites = [c for c in ir_low_e.citations
                      if c.source_id == "extraversion_enthusiasm"]
        assert len(high_e_cites) >= 1
        assert len(low_e_cites) == 0

    def test_conflict_avoidance_reduces_directness(self):
        """High-A persona facing contentious input should have reduced directness."""
        contentious = "Your approach is completely wrong and ridiculous"
        ir_high_a = _generate_ir(make_persona_data(agreeableness=0.9), contentious)
        ir_low_a = _generate_ir(make_persona_data(agreeableness=0.15), contentious)
        assert ir_high_a.communication_style.directness < ir_low_a.communication_style.directness

    def test_no_conflict_avoidance_on_neutral_input(self):
        """Neutral input should not trigger conflict avoidance."""
        neutral = "Tell me about your hobbies"
        ir = _generate_ir(make_persona_data(agreeableness=0.9), neutral)
        ca_cites = [c for c in ir.citations if c.source_id == "conflict_avoidance"]
        assert len(ca_cites) == 0

    def test_validation_test_high_vs_low_a(self):
        """Plan's 'Validation Test': High-A acknowledges, Low-A disagrees directly."""
        user_says = "I think remote work is terrible for productivity"
        ir_a = _generate_ir(make_persona_data(agreeableness=0.85), user_says)
        ir_b = _generate_ir(make_persona_data(agreeableness=0.2), user_says)
        # High-A should have validation directive
        assert any("acknowledge" in d.lower() for d in ir_a.behavioral_directives)
        assert not any("acknowledge" in d.lower() for d in ir_b.behavioral_directives)


# ============================================================================
# CognitiveGuidance Tests
# ============================================================================

class TestCognitiveGuidanceComputation:
    """Tests for _compute_cognitive_guidance."""

    def test_high_analytical_reasoning_style(self):
        """High analytical persona should have analytical reasoning directives."""
        ir = _generate_ir(make_persona_data(cog_analytical_intuitive=0.9))
        assert any("step by step" in d.lower() or "logical" in d.lower()
                    for d in ir.behavioral_directives)

    def test_intuitive_reasoning_style(self):
        """Low analytical persona should have intuitive reasoning directives."""
        ir = _generate_ir(make_persona_data(cog_analytical_intuitive=0.15))
        assert any("gut feeling" in d.lower() or "instinct" in d.lower()
                    for d in ir.behavioral_directives)

    def test_high_cognitive_complexity_tradeoffs(self):
        """High complexity persona should acknowledge tradeoffs."""
        ir = _generate_ir(make_persona_data(cog_cognitive_complexity=0.85))
        assert any("tradeoff" in d.lower() or "counterargument" in d.lower()
                    for d in ir.behavioral_directives)

    def test_low_cognitive_complexity_decisive(self):
        """Low complexity persona should be decisive, black-and-white."""
        ir = _generate_ir(make_persona_data(cog_cognitive_complexity=0.15))
        assert any("decisive" in d.lower() or "black-and-white" in d.lower()
                    for d in ir.behavioral_directives)

    def test_high_nuance_level(self):
        """High complexity should produce nuanced views."""
        ir = _generate_ir(make_persona_data(cog_cognitive_complexity=0.85))
        assert any("nuance" in d.lower() or "multifaceted" in d.lower()
                    for d in ir.behavioral_directives)

    def test_cognitive_guidance_citation_exists(self):
        """CognitiveGuidance should generate a citation."""
        ir = _generate_ir(make_persona_data(cog_analytical_intuitive=0.9))
        cog_citations = [c for c in ir.citations
                        if c.source_id == "cognitive_guidance"]
        assert len(cog_citations) >= 1

    def test_reasoning_style_test_analytical_vs_intuitive(self):
        """Plan's 'Reasoning Style Test': analytical vs intuitive on same prompt."""
        prompt = "Should we use microservices or monolith?"
        ir_analytical = _generate_ir(
            make_persona_data(cog_analytical_intuitive=0.9, cog_cognitive_complexity=0.85),
            prompt,
        )
        ir_intuitive = _generate_ir(
            make_persona_data(cog_analytical_intuitive=0.2, cog_cognitive_complexity=0.3),
            prompt,
        )
        # Analytical should have step-by-step, intuitive should have gut feeling
        assert any("step" in d.lower() for d in ir_analytical.behavioral_directives)
        assert any("gut" in d.lower() or "instinct" in d.lower()
                    for d in ir_intuitive.behavioral_directives)
        # Analytical should acknowledge tradeoffs
        assert any("tradeoff" in d.lower() for d in ir_analytical.behavioral_directives)
        assert not any("tradeoff" in d.lower() for d in ir_intuitive.behavioral_directives)


# ============================================================================
# Value Conflict Resolution Tests
# ============================================================================

class TestValueConflictResolution:
    """Tests that value conflicts are detected and expressed in stance."""

    def test_conflicting_values_expressed_in_stance(self):
        """Persona with high opposing values should express conflict in stance."""
        # self_direction and conformity are opposing
        data = make_persona_data(val_self_direction=0.85, val_conformity=0.8)
        ir = _generate_ir(data)
        # Stance should mention feeling conflicted
        if ir.response_structure.stance:
            assert "conflicted" in ir.response_structure.stance.lower() or \
                   "value both" in ir.response_structure.stance.lower()

    def test_no_conflict_with_aligned_values(self):
        """Persona without opposing high values should NOT express conflict."""
        # benevolence and universalism are adjacent, not opposing
        data = make_persona_data(val_benevolence=0.85, val_universalism=0.8)
        ir = _generate_ir(data)
        if ir.response_structure.stance:
            assert "conflicted" not in ir.response_structure.stance.lower()

    def test_conflict_citation_generated(self):
        """Value conflict should produce circumplex-related citations."""
        data = make_persona_data(val_self_direction=0.85, val_conformity=0.8)
        ir = _generate_ir(data)
        conflict_cites = [c for c in ir.citations
                         if "schwartz" in c.source_id.lower() or
                         "conflict" in c.effect.lower()]
        # Should have at least one conflict-related citation
        assert len(conflict_cites) >= 1

    def test_value_short_description(self):
        """Value short descriptions should be human-readable."""
        assert "autonomy" in _value_short_description("self_direction")
        assert "caring" in _value_short_description("benevolence")
        assert "safety" in _value_short_description("security")

    def test_internal_conflict_test_from_plan(self):
        """Plan's 'Internal Conflict Test': self_direction vs security on AI sharing."""
        data = make_persona_data(val_self_direction=0.85, val_security=0.8)
        ir = _generate_ir(data, "Should companies be forced to share their AI models openly?")
        # Should detect conflict (self_direction opposes security on circumplex)
        if ir.response_structure.stance:
            # Either expressed in stance or detected in citations
            has_conflict_cite = any(
                "conflict" in c.effect.lower() or "opposing" in c.effect.lower()
                for c in ir.citations
            )
            has_conflict_stance = "conflicted" in ir.response_structure.stance.lower()
            assert has_conflict_cite or has_conflict_stance


# ============================================================================
# Decision Policy Tests
# ============================================================================

class TestDecisionPolicyWiring:
    """Tests that decision policies are checked and cited in interpretation stage."""

    def test_policy_matched_generates_citation(self):
        """When decision policy condition matches, it should be cited."""
        data = make_persona_data()
        data["decision_policies"] = [
            {"condition": "high_stakes_decision", "approach": "analytical_systematic",
             "time_needed": "extended"},
        ]
        ir = _generate_ir(data, "This is a high_stakes_decision we need to make")
        policy_cites = [c for c in ir.citations if c.source_id == "decision_policy"]
        assert len(policy_cites) >= 1
        assert "analytical_systematic" in policy_cites[0].effect

    def test_no_policy_match_no_citation(self):
        """When no policy matches, no decision_policy citation should appear."""
        data = make_persona_data()
        data["decision_policies"] = [
            {"condition": "high_stakes_decision", "approach": "analytical_systematic"},
        ]
        ir = _generate_ir(data, "What's your favorite color?")
        policy_cites = [c for c in ir.citations if c.source_id == "decision_policy"]
        assert len(policy_cites) == 0


# ============================================================================
# Behavioral Directives in IR Tests
# ============================================================================

class TestBehavioralDirectivesInIR:
    """Tests that behavioral directives flow through the IR correctly."""

    def test_directives_present_in_ir(self):
        """IR should contain behavioral_directives field."""
        ir = _generate_ir(make_persona_data(agreeableness=0.85, openness=0.85))
        assert isinstance(ir.behavioral_directives, list)
        assert len(ir.behavioral_directives) > 0

    def test_moderate_traits_fewer_directives(self):
        """Moderate traits should produce fewer directives than extreme traits."""
        ir_moderate = _generate_ir(make_persona_data())
        ir_extreme = _generate_ir(make_persona_data(
            agreeableness=0.9, openness=0.9, extraversion=0.9,
            cog_analytical_intuitive=0.9, cog_cognitive_complexity=0.9,
        ))
        assert len(ir_extreme.behavioral_directives) > len(ir_moderate.behavioral_directives)

    def test_directives_are_strings(self):
        """All directives should be non-empty strings."""
        ir = _generate_ir(make_persona_data(openness=0.9))
        for d in ir.behavioral_directives:
            assert isinstance(d, str)
            assert len(d) > 10


# ============================================================================
# Prompt Builder Integration Tests
# ============================================================================

class TestPromptBuilderIntegration:
    """Tests that behavioral directives reach the prompt builder."""

    def test_prompt_contains_personality_section(self):
        """Generated prompt should contain personality-driven behavior section."""
        from persona_engine.generation.prompt_builder import IRPromptBuilder
        ir = _generate_ir(make_persona_data(openness=0.9, agreeableness=0.9))
        builder = IRPromptBuilder()
        prompt = builder.build_generation_prompt(
            ir=ir,
            user_input="What do you think?",
            behavioral_directives=ir.behavioral_directives,
        )
        assert "PERSONALITY-DRIVEN BEHAVIOR" in prompt

    def test_prompt_without_directives_no_section(self):
        """Prompt with no directives should NOT have personality section header."""
        from persona_engine.generation.prompt_builder import IRPromptBuilder
        ir = _generate_ir(make_persona_data())
        builder = IRPromptBuilder()
        # Force empty directives
        prompt = builder.build_generation_prompt(
            ir=ir,
            user_input="What do you think?",
            behavioral_directives=None,
        )
        assert "=== PERSONALITY-DRIVEN BEHAVIOR ===" not in prompt


# ============================================================================
# Twin Differentiation Tests (Observable Behavior)
# ============================================================================

class TestTwinDifferentiation:
    """Test that counterfactual twins produce different IR on the same prompt."""

    def test_openness_twins_differ_in_directives(self):
        """High-O vs Low-O should produce different behavioral directives."""
        ir_high = _generate_ir(make_persona_data(openness=0.9))
        ir_low = _generate_ir(make_persona_data(openness=0.15))
        assert set(ir_high.behavioral_directives) != set(ir_low.behavioral_directives)

    def test_agreeableness_twins_differ_in_directness(self):
        """High-A vs Low-A should produce different directness on contentious input."""
        contentious = "That idea is completely wrong"
        ir_high = _generate_ir(make_persona_data(agreeableness=0.9), contentious)
        ir_low = _generate_ir(make_persona_data(agreeableness=0.15), contentious)
        assert ir_high.communication_style.directness < ir_low.communication_style.directness

    def test_extraversion_twins_differ_in_directives(self):
        """High-E vs Low-E should differ in proactivity directives."""
        ir_high = _generate_ir(make_persona_data(extraversion=0.9))
        ir_low = _generate_ir(make_persona_data(extraversion=0.15))
        high_has_followup = any("follow-up" in d.lower() for d in ir_high.behavioral_directives)
        low_has_followup = any("follow-up" in d.lower() for d in ir_low.behavioral_directives)
        assert high_has_followup and not low_has_followup

    def test_cognitive_twins_differ_in_reasoning(self):
        """Analytical vs intuitive should produce different reasoning directives."""
        ir_analytical = _generate_ir(make_persona_data(cog_analytical_intuitive=0.9))
        ir_intuitive = _generate_ir(make_persona_data(cog_analytical_intuitive=0.15))
        analytical_dirs = " ".join(ir_analytical.behavioral_directives).lower()
        intuitive_dirs = " ".join(ir_intuitive.behavioral_directives).lower()
        assert "step by step" in analytical_dirs or "logical" in analytical_dirs
        assert "gut" in intuitive_dirs or "instinct" in intuitive_dirs

    def test_all_five_traits_produce_unique_directives(self):
        """Each Big Five extreme should produce at least one unique directive."""
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        directive_sets = {}
        for trait in traits:
            data = make_persona_data(**{trait: 0.9})
            ir = _generate_ir(data)
            directive_sets[trait] = set(ir.behavioral_directives)

        # Each trait's directives should not be identical to all others
        for i, t1 in enumerate(traits):
            matches = sum(
                1 for t2 in traits if t1 != t2 and directive_sets[t1] == directive_sets[t2]
            )
            # Should not match ALL other traits
            assert matches < len(traits) - 1, f"{t1} has identical directives to all others"


# ============================================================================
# Zero Dead Methods Checkpoint
# ============================================================================

class TestZeroDeadMethods:
    """Verify that formerly dead methods now influence pipeline output."""

    def test_get_validation_tendency_influences_output(self):
        """get_validation_tendency should influence behavioral directives."""
        ir = _generate_ir(make_persona_data(agreeableness=0.9))
        assert any("acknowledge" in d.lower() for d in ir.behavioral_directives)

    def test_get_conflict_avoidance_influences_directness(self):
        """get_conflict_avoidance should reduce directness on contentious input."""
        ir = _generate_ir(make_persona_data(agreeableness=0.9),
                         "This is completely wrong and stupid")
        ca_cites = [c for c in ir.citations if c.source_id == "conflict_avoidance"]
        assert len(ca_cites) >= 1

    def test_influences_hedging_frequency_produces_directive(self):
        """influences_hedging_frequency should produce hedging directive for high-A."""
        ir = _generate_ir(make_persona_data(agreeableness=0.85))
        assert any("hedging" in d.lower() for d in ir.behavioral_directives)

    def test_get_enthusiasm_baseline_produces_citation(self):
        """get_enthusiasm_baseline should produce enthusiasm citation for high-E."""
        ir = _generate_ir(make_persona_data(extraversion=0.9))
        e_cites = [c for c in ir.citations if c.source_id == "extraversion_enthusiasm"]
        assert len(e_cites) >= 1

    def test_get_negative_tone_bias_computed(self):
        """get_negative_tone_bias should be computed (non-zero for high-N)."""
        persona = Persona(**make_persona_data(neuroticism=0.9))
        planner = TurnPlanner(persona, DeterminismManager(seed=42))
        ctx = TraceContext()
        guidance = planner._behavioral.compute_trait_guidance(ctx, "test")
        assert guidance.negative_tone_weight > 0.4

    def test_influences_proactivity_produces_directive(self):
        """influences_proactivity should produce follow-up directive for high-E."""
        ir = _generate_ir(make_persona_data(extraversion=0.85))
        assert any("follow-up" in d.lower() for d in ir.behavioral_directives)

    def test_get_novelty_seeking_produces_directive(self):
        """get_novelty_seeking should produce novelty directive for high-O."""
        ir = _generate_ir(make_persona_data(openness=0.85))
        assert any("unconventional" in d.lower() or "creative" in d.lower()
                    for d in ir.behavioral_directives)

    def test_influences_abstract_reasoning_produces_directive(self):
        """influences_abstract_reasoning should produce abstract directive for high-O."""
        ir = _generate_ir(make_persona_data(openness=0.85))
        assert any("metaphor" in d.lower() or "abstract" in d.lower()
                    for d in ir.behavioral_directives)

    def test_get_reasoning_approach_influences_output(self):
        """get_reasoning_approach should influence cognitive directives."""
        ir = _generate_ir(make_persona_data(cog_analytical_intuitive=0.9))
        assert any("step" in d.lower() or "logical" in d.lower()
                    for d in ir.behavioral_directives)

    def test_should_acknowledge_tradeoffs_influences_output(self):
        """should_acknowledge_tradeoffs should produce tradeoff directive."""
        ir = _generate_ir(make_persona_data(cog_cognitive_complexity=0.85))
        assert any("tradeoff" in d.lower() for d in ir.behavioral_directives)

    def test_get_nuance_capacity_influences_output(self):
        """get_nuance_capacity should produce nuance directive for high complexity."""
        ir = _generate_ir(make_persona_data(cog_cognitive_complexity=0.85))
        assert any("nuance" in d.lower() or "multifaceted" in d.lower()
                    for d in ir.behavioral_directives)

    def test_value_conflict_resolution_called(self):
        """resolve_conflict_detailed should be called when values conflict."""
        data = make_persona_data(val_self_direction=0.85, val_conformity=0.8)
        ir = _generate_ir(data)
        conflict_cites = [c for c in ir.citations
                         if "schwartz" in c.source_id.lower() or
                         "opposition" in c.effect.lower() or
                         "conflict" in c.effect.lower()]
        assert len(conflict_cites) >= 1

    def test_decision_policy_checked(self):
        """check_decision_policy should be called during interpretation."""
        data = make_persona_data()
        data["decision_policies"] = [
            {"condition": "high_stakes", "approach": "systematic"},
        ]
        ir = _generate_ir(data, "This is a high_stakes question")
        policy_cites = [c for c in ir.citations if c.source_id == "decision_policy"]
        assert len(policy_cites) >= 1
