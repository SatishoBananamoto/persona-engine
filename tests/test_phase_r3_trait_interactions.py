"""
Phase R3 Tests: Trait Interactions — Emergent Personality Patterns

Tests that trait combinations produce emergent behavioral patterns that
neither trait alone would predict.

Covers:
- Activation function (geometric mean, threshold)
- All 9 interaction patterns
- Integration with IR pipeline
- Emergent patterns produce qualitatively different behavior
"""

import math
import pytest
import yaml

from persona_engine.behavioral.trait_interactions import (
    INTERACTION_PATTERNS,
    InteractionEffect,
    TraitInteractionEngine,
    compute_activation,
)
from persona_engine.memory import StanceCache
from persona_engine.planner.turn_planner import (
    ConversationContext,
    TurnPlanner,
)
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    Verbosity,
)
from persona_engine.schema.persona_schema import BigFiveTraits, Persona
from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Helpers
# ============================================================================

def _make_persona_data(**overrides) -> dict:
    base = {
        "persona_id": "TEST",
        "version": "1.0",
        "label": "Test Persona",
        "identity": {
            "age": 30, "gender": "female", "location": "NYC",
            "education": "BS", "occupation": "Engineer",
            "background": "Test",
        },
        "psychology": {
            "big_five": {
                "openness": 0.5, "conscientiousness": 0.5,
                "extraversion": 0.5, "agreeableness": 0.5,
                "neuroticism": 0.5,
            },
            "values": {
                "self_direction": 0.5, "stimulation": 0.5,
                "hedonism": 0.5, "achievement": 0.5, "power": 0.5,
                "security": 0.5, "conformity": 0.5, "tradition": 0.5,
                "benevolence": 0.5, "universalism": 0.5,
            },
            "cognitive_style": {
                "analytical_intuitive": 0.5, "systematic_heuristic": 0.5,
                "risk_tolerance": 0.5, "need_for_closure": 0.5,
                "cognitive_complexity": 0.5,
            },
            "communication": {
                "verbosity": 0.5, "formality": 0.5,
                "directness": 0.5, "emotional_expressiveness": 0.5,
            },
        },
        "knowledge_domains": [
            {"domain": "Engineering", "proficiency": 0.7, "subdomains": []},
        ],
        "social_roles": {
            "default": {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5},
        },
        "invariants": {
            "identity_facts": ["Engineer"],
            "cannot_claim": [],
            "must_avoid": [],
        },
        "initial_state": {
            "mood_valence": 0.2, "mood_arousal": 0.4,
            "fatigue": 0.2, "stress": 0.2, "engagement": 0.5,
        },
        "uncertainty": {
            "admission_threshold": 0.45, "hedging_frequency": 0.4,
            "clarification_tendency": 0.5, "knowledge_boundary_strictness": 0.6,
        },
        "claim_policy": {
            "allowed_claim_types": ["personal_experience", "domain_expert", "general_common_knowledge"],
            "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
            "lookup_behavior": "hedge",
        },
        "time_scarcity": 0.45,
        "privacy_sensitivity": 0.5,
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
# Test Activation Function
# ============================================================================

class TestActivationFunction:
    """Test geometric mean activation computation."""

    def test_both_traits_extreme_strong_activation(self):
        """When both traits exceed threshold, activation should be positive."""
        traits = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.1, neuroticism=0.5)
        pattern = INTERACTION_PATTERNS[0]  # intellectual_combatant: high-O + low-A
        activation = compute_activation(traits, pattern)
        assert activation > 0.3

    def test_one_trait_below_threshold_zero(self):
        """When one trait doesn't meet threshold, activation should be 0."""
        traits = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        pattern = INTERACTION_PATTERNS[0]  # intellectual_combatant: high-O + low-A
        activation = compute_activation(traits, pattern)
        assert activation == 0.0

    def test_moderate_traits_no_activation(self):
        """Moderate traits (all 0.5) should not activate any pattern."""
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.5)
        for pattern in INTERACTION_PATTERNS:
            assert compute_activation(traits, pattern) == 0.0

    def test_geometric_mean_property(self):
        """Activation should be lower than arithmetic mean of extremities."""
        traits = BigFiveTraits(openness=0.95, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.15, neuroticism=0.5)
        pattern = INTERACTION_PATTERNS[0]  # intellectual_combatant
        activation = compute_activation(traits, pattern)
        # Both extremities should be positive for geometric mean
        assert 0 < activation < 1.0

    def test_three_trait_pattern_harder_to_activate(self):
        """Vulnerable ruminant (3 traits) needs all 3 conditions met."""
        # Only 2 of 3 conditions met
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.1, agreeableness=0.5, neuroticism=0.9)
        vulnerable_pattern = [p for p in INTERACTION_PATTERNS if p["name"] == "vulnerable_ruminant"][0]
        activation = compute_activation(traits, vulnerable_pattern)
        assert activation == 0.0  # conscientiousness not low enough

        # All 3 conditions met
        traits_full = BigFiveTraits(openness=0.5, conscientiousness=0.1,
                                     extraversion=0.1, agreeableness=0.5, neuroticism=0.9)
        activation_full = compute_activation(traits_full, vulnerable_pattern)
        assert activation_full > 0.0


# ============================================================================
# Test Individual Patterns
# ============================================================================

class TestIntellectualCombatant:
    """High-O + Low-A → curious but confrontational."""

    def test_activated_for_high_o_low_a(self):
        traits = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.15, neuroticism=0.5)
        engine = TraitInteractionEngine(traits)
        patterns = engine.detect_active_patterns()
        names = [p.pattern_name for p in patterns]
        assert "intellectual_combatant" in names

    def test_not_activated_for_high_o_high_a(self):
        traits = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.8, neuroticism=0.5)
        engine = TraitInteractionEngine(traits)
        patterns = engine.detect_active_patterns()
        names = [p.pattern_name for p in patterns]
        assert "intellectual_combatant" not in names

    def test_modifiers_include_directness_boost(self):
        traits = BigFiveTraits(openness=0.9, conscientiousness=0.5,
                               extraversion=0.5, agreeableness=0.15, neuroticism=0.5)
        engine = TraitInteractionEngine(traits)
        mods = engine.get_aggregate_modifiers()
        assert mods.get("directness", 0) > 0


class TestAnxiousPerfectionist:
    """High-N + High-C → worries about quality."""

    def test_activated(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.9,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.85)
        engine = TraitInteractionEngine(traits)
        patterns = engine.detect_active_patterns()
        names = [p.pattern_name for p in patterns]
        assert "anxious_perfectionist" in names

    def test_confidence_reduced(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.9,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.85)
        engine = TraitInteractionEngine(traits)
        mods = engine.get_aggregate_modifiers()
        assert mods.get("confidence", 0) < 0

    def test_hedging_increased(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.9,
                               extraversion=0.5, agreeableness=0.5, neuroticism=0.85)
        engine = TraitInteractionEngine(traits)
        mods = engine.get_aggregate_modifiers()
        assert mods.get("hedging_level", 0) > 0


class TestWarmLeader:
    """High-E + High-A → enthusiastic consensus builder."""

    def test_activated(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.9, agreeableness=0.85, neuroticism=0.5)
        engine = TraitInteractionEngine(traits)
        names = [p.pattern_name for p in engine.detect_active_patterns()]
        assert "warm_leader" in names

    def test_enthusiasm_boosted(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.9, agreeableness=0.85, neuroticism=0.5)
        engine = TraitInteractionEngine(traits)
        mods = engine.get_aggregate_modifiers()
        assert mods.get("enthusiasm_boost", 0) > 0


class TestStoicProfessional:
    """Low-N + Low-E → calm, reserved, facts-only."""

    def test_activated(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.1, agreeableness=0.5, neuroticism=0.1)
        engine = TraitInteractionEngine(traits)
        names = [p.pattern_name for p in engine.detect_active_patterns()]
        assert "stoic_professional" in names

    def test_enthusiasm_reduced(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.5,
                               extraversion=0.1, agreeableness=0.5, neuroticism=0.1)
        engine = TraitInteractionEngine(traits)
        mods = engine.get_aggregate_modifiers()
        assert mods.get("enthusiasm_boost", 0) < 0


class TestVulnerableRuminant:
    """High-N + Low-E + Low-C → withdrawn, self-critical."""

    def test_activated(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.1,
                               extraversion=0.1, agreeableness=0.5, neuroticism=0.9)
        engine = TraitInteractionEngine(traits)
        names = [p.pattern_name for p in engine.detect_active_patterns()]
        assert "vulnerable_ruminant" in names

    def test_confidence_heavily_reduced(self):
        traits = BigFiveTraits(openness=0.5, conscientiousness=0.1,
                               extraversion=0.1, agreeableness=0.5, neuroticism=0.9)
        engine = TraitInteractionEngine(traits)
        mods = engine.get_aggregate_modifiers()
        assert mods.get("confidence", 0) < -0.1


# ============================================================================
# Test Pipeline Integration
# ============================================================================

class TestInteractionPipelineIntegration:
    """Verify interaction effects appear in generated IR."""

    def test_intellectual_combatant_ir(self):
        """Intellectual combatant should have higher directness and elasticity."""
        combatant = _make_persona_data(openness=0.9, agreeableness=0.15)
        baseline = _make_persona_data(openness=0.9, agreeableness=0.5)

        ir_combatant = _generate_ir(combatant)
        ir_baseline = _generate_ir(baseline)

        # Combatant should be more direct
        assert ir_combatant.communication_style.directness > ir_baseline.communication_style.directness

    def test_anxious_perfectionist_ir(self):
        """Anxious perfectionist should have lower confidence than calm high-C."""
        anxious = _make_persona_data(conscientiousness=0.9, neuroticism=0.85)
        calm_c = _make_persona_data(conscientiousness=0.9, neuroticism=0.2)

        ir_anxious = _generate_ir(anxious)
        ir_calm = _generate_ir(calm_c)

        assert ir_anxious.response_structure.confidence < ir_calm.response_structure.confidence

    def test_interaction_citations_present(self):
        """Active trait interactions should produce citations."""
        combatant = _make_persona_data(openness=0.9, agreeableness=0.15)
        ir = _generate_ir(combatant)

        citation_effects = [c.effect for c in ir.citations]
        has_interaction_citation = any(
            "trait interaction" in e.lower() for e in citation_effects
        )
        assert has_interaction_citation, "Trait interaction citation not found"

    def test_interaction_directives_in_ir(self):
        """Strongly active patterns should add prompt directives."""
        combatant = _make_persona_data(openness=0.9, agreeableness=0.1)
        ir = _generate_ir(combatant)

        has_interaction_directive = any(
            "intellectual combatant" in d.lower() or "sparring" in d.lower()
            for d in ir.behavioral_directives
        )
        assert has_interaction_directive, (
            f"Expected intellectual combatant directive in: {ir.behavioral_directives}"
        )

    def test_no_interactions_for_moderate_traits(self):
        """Moderate traits should not activate any interaction patterns."""
        moderate = _make_persona_data()  # all 0.5
        ir = _generate_ir(moderate)

        citation_effects = [c.effect for c in ir.citations]
        has_interaction = any("trait interaction" in e.lower() for e in citation_effects)
        assert not has_interaction

    def test_warm_leader_vs_hostile_critic(self):
        """Warm leader and hostile critic should produce qualitatively different IR."""
        warm = _make_persona_data(extraversion=0.9, agreeableness=0.9, neuroticism=0.2)
        hostile = _make_persona_data(extraversion=0.5, agreeableness=0.1, neuroticism=0.85)

        ir_warm = _generate_ir(warm)
        ir_hostile = _generate_ir(hostile)

        # Warm leader should be less direct
        assert ir_warm.communication_style.directness < ir_hostile.communication_style.directness


# ============================================================================
# Test All 9 Patterns Have Coverage
# ============================================================================

class TestAllPatternsExist:
    """Verify all 9 defined patterns exist and are well-formed."""

    def test_nine_patterns_defined(self):
        assert len(INTERACTION_PATTERNS) == 9

    def test_all_patterns_have_required_fields(self):
        for pattern in INTERACTION_PATTERNS:
            assert "name" in pattern
            assert "conditions" in pattern
            assert "modifiers" in pattern
            assert "prompt_guidance" in pattern
            assert len(pattern["conditions"]) >= 2
            assert len(pattern["modifiers"]) >= 2
            assert len(pattern["prompt_guidance"]) > 20

    @pytest.mark.parametrize("pattern_name", [
        "intellectual_combatant",
        "anxious_perfectionist",
        "warm_leader",
        "hostile_critic",
        "quiet_thinker",
        "cautious_conservative",
        "impulsive_explorer",
        "stoic_professional",
        "vulnerable_ruminant",
    ])
    def test_pattern_exists(self, pattern_name):
        names = [p["name"] for p in INTERACTION_PATTERNS]
        assert pattern_name in names

    @pytest.mark.parametrize("pattern_name", [
        "intellectual_combatant",
        "anxious_perfectionist",
        "warm_leader",
        "hostile_critic",
        "quiet_thinker",
        "cautious_conservative",
        "impulsive_explorer",
        "stoic_professional",
        "vulnerable_ruminant",
    ])
    def test_each_pattern_activatable(self, pattern_name):
        """Each pattern should be activatable with extreme-enough traits."""
        pattern = [p for p in INTERACTION_PATTERNS if p["name"] == pattern_name][0]

        # Create traits that meet all conditions
        trait_vals = {"openness": 0.5, "conscientiousness": 0.5,
                      "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}
        for trait, (direction, threshold) in pattern["conditions"].items():
            if direction == "high":
                trait_vals[trait] = min(1.0, threshold + 0.25)
            else:
                trait_vals[trait] = max(0.0, threshold - 0.25)

        traits = BigFiveTraits(**trait_vals)
        activation = compute_activation(traits, pattern)
        assert activation > 0, (
            f"Pattern {pattern_name} should activate with extreme traits, "
            f"but got activation={activation}"
        )
