"""
Tests for PersonaEngine SDK — the unified entry point.

Covers:
- Construction (from Persona, from YAML)
- chat() with mock adapter (full pipeline)
- plan() (IR-only, no LLM call)
- Multi-turn state (turn counting, memory, history)
- reset() lifecycle
- Validation integration
- Interaction mode / goal / topic overrides
- ChatResult properties
- Edge cases
"""

import pytest
import yaml

from persona_engine.engine import ChatResult, PersonaEngine
from persona_engine.generation.llm_adapter import MockLLMAdapter
from persona_engine.memory import MemoryManager
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    IRValidationResult,
)
from persona_engine.schema.persona_schema import Persona


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_PERSONA_DATA = {
    "persona_id": "TEST_SDK",
    "version": "1.0",
    "label": "Test SDK Persona",
    "identity": {
        "age": 30,
        "gender": "female",
        "location": "London",
        "education": "Master's in CS",
        "occupation": "Engineer",
        "background": "Software engineer.",
    },
    "psychology": {
        "big_five": {
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.5,
            "agreeableness": 0.6,
            "neuroticism": 0.3,
        },
        "values": {
            "self_direction": 0.7,
            "stimulation": 0.5,
            "hedonism": 0.4,
            "achievement": 0.6,
            "power": 0.3,
            "security": 0.5,
            "conformity": 0.3,
            "tradition": 0.3,
            "benevolence": 0.6,
            "universalism": 0.5,
        },
        "cognitive_style": {
            "analytical_intuitive": 0.7,
            "systematic_heuristic": 0.6,
            "risk_tolerance": 0.5,
            "need_for_closure": 0.4,
            "cognitive_complexity": 0.7,
        },
        "communication": {
            "verbosity": 0.5,
            "formality": 0.5,
            "directness": 0.6,
            "emotional_expressiveness": 0.4,
        },
    },
    "knowledge_domains": [
        {"domain": "Technology", "proficiency": 0.85, "subdomains": ["Software", "AI"]},
        {"domain": "Business", "proficiency": 0.50, "subdomains": ["Project management"]},
    ],
    "languages": [{"language": "English", "proficiency": 1.0}],
    "cultural_knowledge": {
        "primary_culture": "British",
        "exposure_level": {"european": 0.8},
    },
    "primary_goals": [{"goal": "Build great software", "weight": 0.8}],
    "social_roles": {
        "default": {"formality": 0.5, "directness": 0.6, "emotional_expressiveness": 0.4},
    },
    "uncertainty": {
        "admission_threshold": 0.4,
        "hedging_frequency": 0.4,
        "clarification_tendency": 0.5,
        "knowledge_boundary_strictness": 0.7,
    },
    "claim_policy": {
        "allowed_claim_types": ["personal_experience", "domain_expert"],
        "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
        "lookup_behavior": "hedge",
    },
    "invariants": {
        "identity_facts": ["Lives in London", "Age 30, female"],
        "cannot_claim": ["doctor"],
        "must_avoid": ["revealing employer"],
    },
    "initial_state": {
        "mood_valence": 0.1,
        "mood_arousal": 0.5,
        "fatigue": 0.3,
        "stress": 0.3,
        "engagement": 0.6,
    },
    "biases": [{"type": "confirmation_bias", "strength": 0.3}],
    "time_scarcity": 0.5,
    "privacy_sensitivity": 0.5,
    "disclosure_policy": {
        "base_openness": 0.5,
        "factors": {
            "topic_sensitivity": -0.2,
            "trust_level": 0.3,
            "formal_context": -0.1,
            "positive_mood": 0.15,
        },
        "bounds": [0.1, 0.9],
    },
}


@pytest.fixture
def persona() -> Persona:
    return Persona(**MINIMAL_PERSONA_DATA)


@pytest.fixture
def engine(persona: Persona) -> PersonaEngine:
    """Engine with mock adapter — no API calls."""
    return PersonaEngine(
        persona,
        adapter=MockLLMAdapter(),
        seed=42,
        validate=True,
    )


@pytest.fixture
def yaml_path(tmp_path) -> str:
    """Write persona data to a temp YAML file."""
    path = tmp_path / "test_persona.yaml"
    path.write_text(yaml.dump(MINIMAL_PERSONA_DATA))
    return str(path)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_init(self, persona):
        engine = PersonaEngine(persona, adapter=MockLLMAdapter())
        assert engine.persona is persona
        assert engine.turn_count == 0
        assert engine.history == []
        assert engine.conversation_id  # auto-generated

    def test_custom_conversation_id(self, persona):
        engine = PersonaEngine(
            persona, adapter=MockLLMAdapter(), conversation_id="my_conv"
        )
        assert engine.conversation_id == "my_conv"

    def test_from_yaml(self, yaml_path):
        engine = PersonaEngine.from_yaml(yaml_path, adapter=MockLLMAdapter())
        assert engine.persona.label == "Test SDK Persona"
        assert engine.turn_count == 0

    def test_from_yaml_with_legacy_domains_key(self, tmp_path):
        data = dict(MINIMAL_PERSONA_DATA)
        data["domains"] = data.pop("knowledge_domains")
        path = tmp_path / "legacy.yaml"
        path.write_text(yaml.dump(data))
        engine = PersonaEngine.from_yaml(str(path), adapter=MockLLMAdapter())
        assert len(engine.persona.knowledge_domains) == 2

    def test_validation_disabled(self, persona):
        engine = PersonaEngine(persona, adapter=MockLLMAdapter(), validate=False)
        assert engine.validator is None

    def test_validation_enabled(self, persona):
        engine = PersonaEngine(persona, adapter=MockLLMAdapter(), validate=True)
        assert engine.validator is not None


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

class TestChat:
    def test_basic_chat(self, engine):
        result = engine.chat("Hello, how are you?")
        assert isinstance(result, ChatResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.turn_number == 1

    def test_chat_returns_ir(self, engine):
        result = engine.chat("Tell me about software engineering.")
        assert isinstance(result.ir, IntermediateRepresentation)
        assert result.ir.response_structure.confidence > 0

    def test_chat_returns_validation(self, engine):
        result = engine.chat("What do you think about Python?")
        assert isinstance(result.validation, IRValidationResult)

    def test_chat_increments_turn(self, engine):
        r1 = engine.chat("First message.")
        r2 = engine.chat("Second message.")
        assert r1.turn_number == 1
        assert r2.turn_number == 2
        assert engine.turn_count == 2

    def test_chat_with_mode_override(self, engine):
        result = engine.chat("Debate me.", mode=InteractionMode.DEBATE)
        assert result.ir.conversation_frame.interaction_mode == InteractionMode.DEBATE

    def test_chat_with_goal_override(self, engine):
        result = engine.chat("Convince me.", goal=ConversationGoal.PERSUADE)
        assert result.ir.conversation_frame.goal == ConversationGoal.PERSUADE

    def test_chat_with_topic_override(self, engine):
        result = engine.chat(
            "Tell me about software.",
            topic="software_engineering",
        )
        assert result.turn_number == 1

    def test_chat_appends_to_history(self, engine):
        engine.chat("First.")
        engine.chat("Second.")
        engine.chat("Third.")
        assert len(engine.history) == 3
        assert engine.history[0].turn_number == 1
        assert engine.history[2].turn_number == 3

    def test_chat_validation_disabled(self, persona):
        engine = PersonaEngine(persona, adapter=MockLLMAdapter(), validate=False)
        result = engine.chat("Hello")
        assert result.passed is True  # synthetic pass
        assert result.validation.violations == []


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------

class TestPlan:
    def test_plan_returns_ir(self, engine):
        ir = engine.plan("What is machine learning?")
        assert isinstance(ir, IntermediateRepresentation)

    def test_plan_increments_turn(self, engine):
        engine.plan("First.")
        engine.plan("Second.")
        assert engine.turn_count == 2

    def test_plan_does_not_add_to_history(self, engine):
        engine.plan("Just planning.")
        assert len(engine.history) == 0

    def test_plan_with_mode_override(self, engine):
        ir = engine.plan("Let's debate.", mode=InteractionMode.DEBATE)
        assert ir.conversation_frame.interaction_mode == InteractionMode.DEBATE

    def test_plan_generates_citations(self, engine):
        ir = engine.plan("Tell me about AI.")
        assert len(ir.citations) > 0

    def test_plan_has_competence(self, engine):
        ir = engine.plan("Tell me about software architecture.")
        assert 0 <= ir.response_structure.competence <= 1


# ---------------------------------------------------------------------------
# ChatResult properties
# ---------------------------------------------------------------------------

class TestChatResult:
    def test_citations_shortcut(self, engine):
        result = engine.chat("Hello")
        assert result.citations == result.ir.citations

    def test_competence_shortcut(self, engine):
        result = engine.chat("Hello")
        assert result.competence == result.ir.response_structure.competence

    def test_confidence_shortcut(self, engine):
        result = engine.chat("Hello")
        assert result.confidence == result.ir.response_structure.confidence

    def test_passed_shortcut(self, engine):
        result = engine.chat("Hello")
        assert result.passed == result.validation.passed

    def test_repr(self, engine):
        result = engine.chat("Hello")
        r = repr(result)
        assert "ChatResult" in r
        assert "turn=1" in r


# ---------------------------------------------------------------------------
# Multi-turn & memory
# ---------------------------------------------------------------------------

class TestMultiTurn:
    def test_memory_is_wired(self, engine):
        engine.chat("My name is Alice.")
        stats = engine.memory_stats()
        # Memory should have at least attempted to store something
        assert isinstance(stats, dict)
        assert "facts" in stats

    def test_stance_cache_persists_across_turns(self, engine):
        engine.chat("I think Python is great.")
        engine.chat("Going back to Python, right?")
        # Should not crash — stance cache reused

    def test_history_is_copy(self, engine):
        engine.chat("Hello")
        h1 = engine.history
        engine.chat("World")
        h2 = engine.history
        assert len(h1) == 1
        assert len(h2) == 2  # Original list not mutated


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_turn_count(self, engine):
        engine.chat("First.")
        engine.chat("Second.")
        assert engine.turn_count == 2
        engine.reset()
        assert engine.turn_count == 0

    def test_reset_clears_history(self, engine):
        engine.chat("First.")
        engine.reset()
        assert engine.history == []

    def test_reset_generates_new_conversation_id(self, engine):
        old_id = engine.conversation_id
        engine.reset()
        assert engine.conversation_id != old_id

    def test_reset_allows_fresh_conversation(self, engine):
        engine.chat("Turn one of first conversation.")
        engine.reset()
        r = engine.chat("Turn one of second conversation.")
        assert r.turn_number == 1

    def test_reset_clears_validator_state(self, engine):
        engine.chat("Turn one.")
        engine.chat("Turn two.")
        assert engine.validator is not None
        assert engine.validator.turn_count > 0
        engine.reset()
        assert engine.validator.turn_count == 0


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_system_prompt(self, engine):
        prompt = engine.system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain persona identity information
        assert "Engineer" in prompt or "London" in prompt or "Test SDK" in prompt

    def test_memory_stats(self, engine):
        stats = engine.memory_stats()
        assert "facts" in stats
        assert "trust" in stats
        assert "episodes" in stats

    def test_memory_accessor(self, engine):
        assert isinstance(engine.memory, MemoryManager)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input(self, engine):
        from persona_engine.exceptions import InputValidationError
        with pytest.raises(InputValidationError):
            engine.chat("")

    def test_long_input(self, engine):
        result = engine.chat("Tell me everything about " + "software " * 200)
        assert isinstance(result, ChatResult)

    def test_plan_then_chat(self, engine):
        ir = engine.plan("Planning first.")
        result = engine.chat("Now chatting.")
        assert ir is not None
        assert result.turn_number == 2  # plan was turn 1

    def test_interleave_plan_and_chat(self, engine):
        engine.plan("Plan 1")
        engine.chat("Chat 1")
        engine.plan("Plan 2")
        engine.chat("Chat 2")
        assert engine.turn_count == 4
        assert len(engine.history) == 2  # Only chat() adds to history


# ---------------------------------------------------------------------------
# Integration: from_yaml with real persona files
# ---------------------------------------------------------------------------

class TestRealPersonas:
    @pytest.mark.parametrize("path", [
        "personas/ux_researcher.yaml",
        "personas/chef.yaml",
    ])
    def test_from_yaml_real_personas(self, path):
        engine = PersonaEngine.from_yaml(path, adapter=MockLLMAdapter())
        result = engine.chat("Hello, tell me about yourself.")
        assert result.passed
        assert result.turn_number == 1
        assert len(result.text) > 0

    def test_chef_domain_competence(self):
        engine = PersonaEngine.from_yaml(
            "personas/chef.yaml", adapter=MockLLMAdapter(),
        )
        ir_food = engine.plan(
            "How do you cook a steak?", topic="french_cuisine",
        )
        ir_tech = engine.plan("Explain quantum computing.", topic="quantum")
        # Chef should be more competent on food than quantum computing
        assert ir_food.response_structure.competence > ir_tech.response_structure.competence

    def test_multi_turn_chef(self):
        engine = PersonaEngine.from_yaml(
            "personas/chef.yaml", adapter=MockLLMAdapter(),
        )
        r1 = engine.chat("What makes a good sauce?")
        r2 = engine.chat("And how about knife skills?")
        r3 = engine.chat("What do you think about blockchain?")
        assert r1.passed
        assert r2.passed
        assert r3.competence < r1.competence  # blockchain < food
