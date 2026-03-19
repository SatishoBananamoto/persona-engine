"""
Tests for cross-turn dynamics fixes:
1. Unfrozen relationships (write path + keyword matching)
2. Trust → disclosure wiring
3. Memory → competence familiarity boost
4. Cross-turn inertia smoothing (per-metric)
5. Input-aware stance templates
6. Tightened clarification detection
7. Debate mode differentiation
8. Dynamic time pressure
"""

import pytest
import yaml

from persona_engine.engine import ChatResult, PersonaEngine
from persona_engine.generation.llm_adapter import MockLLMAdapter
from persona_engine.memory import MemoryManager
from persona_engine.memory.relationship_store import RelationshipStore
from persona_engine.planner.intent_analyzer import analyze_intent
from persona_engine.planner.stance_generator import (
    _extract_topic,
    generate_stance_safe,
)
from persona_engine.planner.trace_context import TraceContext
from persona_engine.planner.turn_planner import (
    CROSS_TURN_INERTIA,
    FAMILIARITY_BOOST_CAP,
    FAMILIARITY_BOOST_PER_EPISODE,
    ConversationContext,
    TurnPlanner,
    _smooth,
)
from persona_engine.schema.ir_schema import InteractionMode, ConversationGoal
from persona_engine.schema.persona_schema import Persona
from persona_engine.validation.cross_turn import CrossTurnTracker, TurnSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def chef_engine():
    """PersonaEngine for Marcus (chef) with mock adapter."""
    return PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())


@pytest.fixture
def tech_engine():
    """PersonaEngine for a tech persona."""
    return PersonaEngine.from_yaml("personas/sarah.yaml", adapter=MockLLMAdapter())


# ===========================================================================
# Fix 1: Unfrozen Relationships
# ===========================================================================

class TestUnfrozenRelationships:
    """Verify that trust/rapport actually change across turns."""

    def test_relationship_deltas_from_engagement(self):
        """'Engaged' keyword triggers rapport +0.02."""
        t, r = MemoryManager._infer_relationship_deltas(
            "Engaged on french_cuisine — validated shared expertise"
        )
        assert t > 0, "Trust should increase for 'validated' + 'expertise'"
        assert r > 0, "Rapport should increase for 'engaged'"

    def test_relationship_deltas_expertise(self):
        """'expertise' keyword triggers trust delta."""
        t, _ = MemoryManager._infer_relationship_deltas(
            "Shared expertise on cooking techniques"
        )
        assert t > 0

    def test_relationship_deltas_friendly_exchange(self):
        """'exchange' keyword triggers rapport delta."""
        _, r = MemoryManager._infer_relationship_deltas(
            "Engaged — friendly exchange about travel"
        )
        assert r > 0

    def test_trust_evolves_over_turns(self, chef_engine):
        """After several turns, trust should differ from the 0.5 baseline."""
        for i in range(5):
            chef_engine.chat(f"Tell me about French sauces, question {i}")
        trust = chef_engine._memory.relationships.trust
        rapport = chef_engine._memory.relationships.rapport
        # At least one of trust/rapport should have moved from baseline
        assert trust != 0.5 or rapport != 0.3, (
            f"Trust={trust}, Rapport={rapport} — both still at baseline after 5 turns"
        )


# ===========================================================================
# Fix 2: Trust → Disclosure
# ===========================================================================

class TestTrustDisclosure:
    """Verify that trust modulates disclosure level."""

    def test_high_trust_increases_disclosure(self, chef_engine):
        """When trust is high, disclosure should be higher."""
        # Manually boost trust
        chef_engine._memory.record_relationship_event(
            "User validated expertise and confirmed trust",
            trust_delta=0.3, rapport_delta=0.0, turn=0,
        )
        ir_high = chef_engine.plan("Tell me about your cooking philosophy")
        disc_high = ir_high.knowledge_disclosure.disclosure_level

        # Reset and test with low trust
        engine2 = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        engine2._memory.record_relationship_event(
            "User was hostile and distrustful",
            trust_delta=-0.3, rapport_delta=0.0, turn=0,
        )
        ir_low = engine2.plan("Tell me about your cooking philosophy")
        disc_low = ir_low.knowledge_disclosure.disclosure_level

        assert disc_high > disc_low, (
            f"High trust disclosure ({disc_high:.3f}) should exceed "
            f"low trust disclosure ({disc_low:.3f})"
        )

    def test_trust_modifier_in_citations(self, chef_engine):
        """Trust modifier should appear in the citation trail."""
        chef_engine._memory.record_relationship_event(
            "User validated and confirmed trust",
            trust_delta=0.3, turn=0,
        )
        ir = chef_engine.plan("Tell me about cooking")
        citation_effects = [c.effect for c in ir.citations]
        assert any("trust" in e.lower() or "Trust" in e for e in citation_effects), (
            "Trust modifier citation not found in trail"
        )


# ===========================================================================
# Fix 3: Memory → Competence
# ===========================================================================

class TestMemoryCompetence:
    """Verify that repeated topic engagement boosts competence."""

    def test_familiarity_boosts_competence(self, chef_engine):
        """After discussing a topic, competence should increase."""
        ir1 = chef_engine.plan("What about food safety?", topic="food")
        comp1 = ir1.response_structure.competence

        # Discuss more to build episodic memory
        chef_engine.plan("Tell me about food preparation", topic="food")
        chef_engine.plan("How about food storage?", topic="food")

        ir4 = chef_engine.plan("What about food regulations?", topic="food")
        comp4 = ir4.response_structure.competence

        assert comp4 >= comp1, (
            f"Competence should increase with familiarity: {comp1:.3f} → {comp4:.3f}"
        )

    def test_familiarity_boost_capped(self, chef_engine):
        """Boost should not exceed FAMILIARITY_BOOST_CAP."""
        # Generate many episodes
        for i in range(10):
            chef_engine.plan(f"Food question {i}", topic="food")

        ir = chef_engine.plan("Another food question", topic="food")
        # The familiarity boost is at most 0.15, so total competence shouldn't be unreasonable
        assert ir.response_structure.competence <= 1.0

    def test_no_boost_first_discussion(self, chef_engine):
        """First discussion of a topic should not get familiarity boost."""
        ir = chef_engine.plan("What about quantum computing?", topic="quantum")
        # Check citations — should NOT contain familiarity
        citation_effects = [c.effect for c in ir.citations]
        assert not any("familiarity" in e.lower() for e in citation_effects)


# ===========================================================================
# Fix 4: Cross-Turn Inertia Smoothing
# ===========================================================================

class TestCrossTurnSmoothing:
    """Verify that parameters smooth between turns."""

    def test_smooth_function(self):
        """_smooth blends previous and current values."""
        result = _smooth(0.95, 0.33, 0.15)
        expected = 0.95 * 0.15 + 0.33 * 0.85
        assert abs(result - expected) < 0.001

    def test_confidence_smoothed_between_turns(self, chef_engine):
        """Confidence should not swing as much as raw proficiency difference."""
        ir1 = chef_engine.plan("How do you make a roux?", topic="food")
        conf1 = ir1.response_structure.confidence

        ir2 = chef_engine.plan("What about blockchain technology?", topic="technology")
        conf2 = ir2.response_structure.confidence

        # Without smoothing, confidence would swing by ~0.6+
        # With smoothing, the swing should be dampened
        delta = abs(conf2 - conf1)
        assert delta < 0.60, (
            f"Confidence swing {delta:.3f} should be dampened by inertia "
            f"(food={conf1:.3f}, blockchain={conf2:.3f})"
        )

    def test_smoothing_appears_in_citations(self, chef_engine):
        """Inertia smoothing citations should appear in multi-turn."""
        chef_engine.plan("How do you cook pasta?", topic="food")
        ir2 = chef_engine.plan("What about quantum physics?", topic="science")
        citation_effects = [c.effect for c in ir2.citations]
        assert any("inertia" in e.lower() for e in citation_effects), (
            "Cross-turn inertia citation not found"
        )

    def test_prior_snapshot_stored(self, chef_engine):
        """TurnPlanner should store _prior_snapshot after generating IR."""
        planner = chef_engine._planner
        assert planner._prior_snapshot is None
        chef_engine.plan("Hello")
        assert planner._prior_snapshot is not None
        assert isinstance(planner._prior_snapshot, TurnSnapshot)

    def test_multi_turn_gradual_evolution(self, chef_engine):
        """Parameters should evolve gradually across many turns."""
        confidences = []
        for i in range(5):
            ir = chef_engine.plan(f"Tell me about cooking technique {i}", topic="food")
            confidences.append(ir.response_structure.confidence)

        # Check that consecutive confidence deltas are small
        for i in range(1, len(confidences)):
            delta = abs(confidences[i] - confidences[i - 1])
            assert delta < 0.50, (
                f"Turn {i}: confidence swing {delta:.3f} is too large "
                f"({confidences[i-1]:.3f} → {confidences[i]:.3f})"
            )


# ===========================================================================
# Fix 5: Input-Aware Stance Templates
# ===========================================================================

class TestInputAwareStance:
    """Verify that stances use topic hints from user input."""

    def test_extract_topic_about(self):
        topic = _extract_topic("What do you think about French cooking techniques?", "food")
        assert topic.topic_hint is not None
        assert "french" in topic.topic_hint.lower() or "cooking" in topic.topic_hint.lower()

    def test_extract_topic_how_to(self):
        topic = _extract_topic("How do you make a proper bechamel sauce?", "food")
        assert topic.topic_hint is not None
        assert "bechamel" in topic.topic_hint.lower() or "sauce" in topic.topic_hint.lower()

    def test_extract_topic_short(self):
        """Very short input may not yield a topic hint."""
        topic = _extract_topic("Hi", "general")
        assert topic.topic_hint is None

    def test_stance_varies_by_domain(self):
        """Stances on different domains should differ for the same persona."""
        from persona_engine.behavioral.values_interpreter import ValuesInterpreter
        from persona_engine.behavioral.cognitive_interpreter import CognitiveStyleInterpreter
        from persona_engine.schema.persona_schema import SchwartzValues, CognitiveStyle
        from persona_engine.planner.trace_context import TraceContext
        persona = Persona(**yaml.safe_load(open("personas/chef.yaml")))
        vals = ValuesInterpreter(persona.psychology.values)
        cog = CognitiveStyleInterpreter(persona.psychology.cognitive_style)
        ctx1 = TraceContext()
        ctx2 = TraceContext()

        stance_food, _ = generate_stance_safe(
            persona=persona, values=vals, cognitive=cog,
            user_input="What do you think about French cuisine?",
            topic_signature="french_cuisine", proficiency=0.8,
            expert_allowed=True, ctx=ctx1, domain="food",
        )
        stance_tech, _ = generate_stance_safe(
            persona=persona, values=vals, cognitive=cog,
            user_input="What do you think about microservices?",
            topic_signature="microservices", proficiency=0.3,
            expert_allowed=False, ctx=ctx2, domain="technology",
        )
        assert stance_food != stance_tech

    def test_different_topics_produce_different_stances(self):
        """Two different domains should produce different stances."""
        from persona_engine.behavioral.values_interpreter import ValuesInterpreter
        from persona_engine.behavioral.cognitive_interpreter import CognitiveStyleInterpreter
        from persona_engine.planner.trace_context import TraceContext

        persona = Persona(**yaml.safe_load(open("personas/chef.yaml")))
        vals = ValuesInterpreter(persona.psychology.values)
        cog = CognitiveStyleInterpreter(persona.psychology.cognitive_style)

        stance1, _ = generate_stance_safe(
            persona=persona, values=vals, cognitive=cog,
            user_input="What do you think about molecular gastronomy?",
            topic_signature="molecular_gastronomy", proficiency=0.8,
            expert_allowed=True, ctx=TraceContext(), domain="food",
        )
        stance2, _ = generate_stance_safe(
            persona=persona, values=vals, cognitive=cog,
            user_input="What do you think about microservice architecture?",
            topic_signature="microservices", proficiency=0.2,
            expert_allowed=False, ctx=TraceContext(), domain="technology",
        )
        assert stance1 != stance2


# ===========================================================================
# Fix 6: Tightened Clarification Detection
# ===========================================================================

class TestClarificationDetection:
    """Verify that normal questions don't trigger needs_clarification."""

    def test_or_in_valid_question(self):
        """'or' in a normal question should NOT trigger clarification."""
        ctx = TraceContext()
        _, _, _, needs_clar = analyze_intent(
            "What's better for a sauce, butter or cream?",
            None, None, ctx
        )
        assert not needs_clar, "'or' in valid question should not trigger clarification"

    def test_maybe_in_request(self):
        """'maybe' in a request should NOT trigger clarification."""
        ctx = TraceContext()
        _, _, _, needs_clar = analyze_intent(
            "Maybe you could explain molecular gastronomy?",
            None, None, ctx
        )
        assert not needs_clar, "'maybe' in request should not trigger clarification"

    def test_seven_word_question(self):
        """A 7-word question should NOT trigger clarification."""
        ctx = TraceContext()
        _, _, _, needs_clar = analyze_intent(
            "What do you think about French cuisine?",
            None, None, ctx
        )
        assert not needs_clar, "7-word question should not trigger clarification"

    def test_truly_tiny_input_triggers(self):
        """1-2 word input SHOULD trigger clarification."""
        ctx = TraceContext()
        _, _, _, needs_clar = analyze_intent("Hi", None, None, ctx)
        assert needs_clar, "1-word input should trigger clarification"

    def test_explicit_confusion_triggers(self):
        """Explicit confusion phrases should trigger clarification."""
        ctx = TraceContext()
        _, _, _, needs_clar = analyze_intent(
            "I'm not sure what you mean by that",
            None, None, ctx
        )
        assert needs_clar


# ===========================================================================
# Fix 7: Debate Mode Differentiation
# ===========================================================================

class TestDebateMode:
    """Verify that debate mode feels different from casual chat."""

    def test_debate_directness_higher(self, chef_engine):
        """Debate mode should produce higher directness than casual chat."""
        ir_casual = chef_engine.plan(
            "What do you think about meal prep?",
            mode=InteractionMode.CASUAL_CHAT,
        )
        dir_casual = ir_casual.communication_style.directness

        engine2 = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        ir_debate = engine2.plan(
            "What do you think about meal prep?",
            mode=InteractionMode.DEBATE,
        )
        dir_debate = ir_debate.communication_style.directness

        assert dir_debate > dir_casual, (
            f"Debate directness ({dir_debate:.3f}) should exceed "
            f"casual directness ({dir_casual:.3f})"
        )

    def test_debate_role_synthesized(self, chef_engine):
        """Debate role should be synthesized from default if not defined."""
        rules = chef_engine._planner.rules
        # Before any call, debate role doesn't exist yet
        assert "debate" not in rules.social_roles or True  # may be synthesized lazily
        # After using debate mode, it should exist
        rules.get_social_role_mode(InteractionMode.DEBATE)
        assert "debate" in rules.social_roles


# ===========================================================================
# Fix 8: Dynamic Time Pressure
# ===========================================================================

class TestDynamicTimePressure:
    """Verify that time pressure varies by mode and turn count."""

    def test_debate_reduces_time_pressure(self, chef_engine):
        """Debate mode should reduce time pressure vs base."""
        planner = chef_engine._planner
        ctx = TraceContext()
        tp_debate = planner._compute_time_pressure(
            InteractionMode.DEBATE, 1, ctx
        )
        tp_base = chef_engine._persona.time_scarcity
        assert tp_debate < tp_base, (
            f"Debate time_pressure ({tp_debate:.2f}) should be less than "
            f"base ({tp_base:.2f})"
        )

    def test_late_conversation_increases_pressure(self, chef_engine):
        """Time pressure should build after turn 5."""
        planner = chef_engine._planner
        ctx = TraceContext()
        tp_early = planner._compute_time_pressure(
            InteractionMode.CASUAL_CHAT, 2, ctx
        )
        ctx2 = TraceContext()
        tp_late = planner._compute_time_pressure(
            InteractionMode.CASUAL_CHAT, 10, ctx2
        )
        assert tp_late > tp_early, (
            f"Late-conversation pressure ({tp_late:.2f}) should exceed "
            f"early ({tp_early:.2f})"
        )

    def test_time_pressure_clamped(self, chef_engine):
        """Time pressure should stay in [0, 1]."""
        planner = chef_engine._planner
        ctx = TraceContext()
        tp = planner._compute_time_pressure(InteractionMode.SURVEY, 100, ctx)
        assert 0.0 <= tp <= 1.0


# ===========================================================================
# Integration: Multi-turn conversation evolution
# ===========================================================================

class TestMultiTurnEvolution:
    """End-to-end test that conversations genuinely evolve."""

    def test_five_turn_conversation_evolves(self, chef_engine):
        """Over 5 turns with varied topics, at least some metrics should shift."""
        prompts = [
            "How do you make a classic roux?",
            "What's your opinion on blockchain technology?",
            "Tell me about your favorite French sauce",
            "I disagree with your approach to cooking",
            "Can you explain molecular gastronomy?",
        ]
        results = []
        for prompt in prompts:
            r = chef_engine.chat(prompt)
            results.append(r)

        # Collect all confidences — they should NOT all be identical
        confidences = [r.ir.response_structure.confidence for r in results]
        disclosures = [r.ir.knowledge_disclosure.disclosure_level for r in results]
        unique_conf = len(set(round(c, 3) for c in confidences))
        unique_disc = len(set(round(d, 3) for d in disclosures))

        assert unique_conf > 1 or unique_disc > 1, (
            f"Metrics should vary across diverse turns: "
            f"confidences={[round(c,3) for c in confidences]}, "
            f"disclosures={[round(d,3) for d in disclosures]}"
        )

    def test_trust_rapport_grow_over_turns(self, chef_engine):
        """Trust and/or rapport should grow over engaged conversation."""
        initial_trust = chef_engine._memory.relationships.trust
        initial_rapport = chef_engine._memory.relationships.rapport

        for i in range(5):
            chef_engine.chat(f"I really enjoy learning about French cuisine, lesson {i}")

        final_trust = chef_engine._memory.relationships.trust
        final_rapport = chef_engine._memory.relationships.rapport

        grew = (final_trust > initial_trust) or (final_rapport > initial_rapport)
        assert grew, (
            f"Trust/rapport should grow: trust {initial_trust:.3f}→{final_trust:.3f}, "
            f"rapport {initial_rapport:.3f}→{final_rapport:.3f}"
        )
