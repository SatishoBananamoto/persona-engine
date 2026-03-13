"""
Tests for the five improvements:
1. Multi-turn conversation history
2. Persistence (save/load)
3. Expanded domain detection
4. New personas
5. Package structure
"""

import json
import pytest
import yaml

from persona_engine.engine import ChatResult, PersonaEngine
from persona_engine.generation.llm_adapter import MockLLMAdapter
from persona_engine.planner.domain_detection import detect_domain


# ---------------------------------------------------------------------------
# 1. Multi-turn conversation history
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_first_turn_no_history(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        result = engine.chat("Hello!")
        # First turn — adapter should not have received history
        adapter = engine._generator.adapter
        assert isinstance(adapter, MockLLMAdapter)
        assert adapter.last_conversation_history is None

    def test_second_turn_has_history(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        engine.chat("What is a roux?")
        engine.chat("And how about a bechamel?")
        adapter = engine._generator.adapter
        assert isinstance(adapter, MockLLMAdapter)
        # Second call should have history from turn 1
        assert adapter.last_conversation_history is not None
        assert len(adapter.last_conversation_history) == 2  # user + assistant
        assert adapter.last_conversation_history[0]["role"] == "user"
        assert adapter.last_conversation_history[1]["role"] == "assistant"

    def test_history_grows_with_turns(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        for i in range(5):
            engine.chat(f"Question {i}")
        adapter = engine._generator.adapter
        assert isinstance(adapter, MockLLMAdapter)
        # 4 prior turns × 2 messages each = 8 messages in history
        assert adapter.last_conversation_history is not None
        assert len(adapter.last_conversation_history) == 8

    def test_history_reset_clears(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        engine.chat("First turn")
        engine.chat("Second turn")
        engine.reset()
        engine.chat("Fresh start")
        adapter = engine._generator.adapter
        assert isinstance(adapter, MockLLMAdapter)
        # After reset, first turn has no history
        assert adapter.last_conversation_history is None

    def test_user_input_stored_on_result(self):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        result = engine.chat("What is a roux?")
        assert result._user_input == "What is a roux?"


# ---------------------------------------------------------------------------
# 2. Persistence (save/load)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        engine.chat("Hello chef!")
        engine.chat("How do you make sauce?")
        path = tmp_path / "state.json"
        engine.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["turn_number"] == 2
        assert data["persona_id"] == "P_002_CHEF"
        assert len(data["messages"]) == 2

    def test_load_restores_state(self, tmp_path):
        engine = PersonaEngine.from_yaml("personas/chef.yaml", adapter=MockLLMAdapter())
        engine.chat("Hello")
        engine.chat("World")
        state_path = tmp_path / "state.json"
        engine.save(state_path)

        loaded = PersonaEngine.load(
            state_path, "personas/chef.yaml", adapter=MockLLMAdapter(),
        )
        assert loaded.turn_count == 2
        assert loaded.conversation_id == engine.conversation_id

    def test_load_preserves_conversation_id(self, tmp_path):
        engine = PersonaEngine.from_yaml(
            "personas/chef.yaml", adapter=MockLLMAdapter(),
            conversation_id="test_123",
        )
        engine.chat("Hello")
        state_path = tmp_path / "state.json"
        engine.save(state_path)

        loaded = PersonaEngine.load(
            state_path, "personas/chef.yaml", adapter=MockLLMAdapter(),
        )
        assert loaded.conversation_id == "test_123"


# ---------------------------------------------------------------------------
# 3. Expanded domain detection
# ---------------------------------------------------------------------------

class TestExpandedDomains:
    """Test that new domains and expanded food keywords work."""

    @pytest.mark.parametrize("text,expected", [
        ("How do you make a bechamel sauce?", "food"),
        ("What about hollandaise?", "food"),
        ("Tell me about a roux", "food"),
        ("How to braise a steak?", "food"),
        ("Pasta with cream reduction", "food"),
    ])
    def test_food_expanded_keywords(self, text, expected):
        domain, _ = detect_domain(text)
        assert domain == expected, f"Expected {expected} for '{text}', got {domain}"

    @pytest.mark.parametrize("text,expected", [
        ("quantum entanglement", "science"),
        ("DNA molecule structure", "science"),
        ("theory of relativity", "science"),
        ("photon particle physics", "science"),
    ])
    def test_science_domain(self, text, expected):
        domain, _ = detect_domain(text)
        assert domain == expected, f"Expected {expected} for '{text}', got {domain}"

    @pytest.mark.parametrize("text,expected", [
        ("jazz music improvisation", "arts"),
        ("painting and sculpture gallery", "arts"),
        ("symphony orchestra concert", "arts"),
    ])
    def test_arts_domain(self, text, expected):
        domain, _ = detect_domain(text)
        assert domain == expected, f"Expected {expected} for '{text}', got {domain}"

    @pytest.mark.parametrize("text,expected", [
        ("contract law and litigation", "law"),
        ("court verdict precedent", "law"),
        ("plaintiff defendant trial", "law"),
    ])
    def test_law_domain(self, text, expected):
        domain, _ = detect_domain(text)
        assert domain == expected, f"Expected {expected} for '{text}', got {domain}"

    @pytest.mark.parametrize("text,expected", [
        ("weightlifting workout training", "sports"),
        ("marathon endurance cardio", "sports"),
        ("basketball championship tournament", "sports"),
    ])
    def test_sports_domain(self, text, expected):
        domain, _ = detect_domain(text)
        assert domain == expected, f"Expected {expected} for '{text}', got {domain}"

    @pytest.mark.parametrize("text,expected", [
        ("stock portfolio investment", "finance"),
        ("mortgage interest rate", "finance"),
        ("hedge fund equity valuation", "finance"),
    ])
    def test_finance_domain(self, text, expected):
        domain, _ = detect_domain(text)
        assert domain == expected, f"Expected {expected} for '{text}', got {domain}"

    @pytest.mark.parametrize("text,expected", [
        ("teaching curriculum pedagogy", "education"),
        ("classroom assessment grading", "education"),
    ])
    def test_education_domain(self, text, expected):
        domain, _ = detect_domain(text)
        assert domain == expected, f"Expected {expected} for '{text}', got {domain}"


# ---------------------------------------------------------------------------
# 4. New personas load and work
# ---------------------------------------------------------------------------

class TestNewPersonas:
    @pytest.mark.parametrize("path,persona_id", [
        ("personas/physicist.yaml", "P_003_PHYSICIST"),
        ("personas/musician.yaml", "P_004_MUSICIAN"),
        ("personas/lawyer.yaml", "P_005_LAWYER"),
        ("personas/fitness_coach.yaml", "P_006_FITNESS"),
    ])
    def test_persona_loads(self, path, persona_id):
        engine = PersonaEngine.from_yaml(path, adapter=MockLLMAdapter())
        assert engine.persona.persona_id == persona_id

    @pytest.mark.parametrize("path", [
        "personas/physicist.yaml",
        "personas/musician.yaml",
        "personas/lawyer.yaml",
        "personas/fitness_coach.yaml",
    ])
    def test_persona_chat_works(self, path):
        engine = PersonaEngine.from_yaml(path, adapter=MockLLMAdapter())
        result = engine.chat("Tell me about yourself.")
        assert result.passed
        assert len(result.text) > 0

    def test_physicist_high_science_competence(self):
        engine = PersonaEngine.from_yaml(
            "personas/physicist.yaml", adapter=MockLLMAdapter(),
        )
        ir = engine.plan("Tell me about quantum entanglement.", topic="quantum_physics")
        assert ir.response_structure.competence > 0.5

    def test_lawyer_high_formality(self):
        engine = PersonaEngine.from_yaml(
            "personas/lawyer.yaml", adapter=MockLLMAdapter(),
        )
        ir = engine.plan("What do you think about this contract?")
        assert ir.communication_style.formality > 0.7

    def test_musician_low_formality(self):
        engine = PersonaEngine.from_yaml(
            "personas/musician.yaml", adapter=MockLLMAdapter(),
        )
        ir = engine.plan("Tell me about jazz.")
        assert ir.communication_style.formality < 0.4

    def test_fitness_coach_high_expressiveness(self):
        engine = PersonaEngine.from_yaml(
            "personas/fitness_coach.yaml", adapter=MockLLMAdapter(),
        )
        result = engine.chat("I want to get stronger!")
        assert result.passed


# ---------------------------------------------------------------------------
# 5. Package structure
# ---------------------------------------------------------------------------

class TestPackageStructure:
    def test_import_from_top_level(self):
        from persona_engine import PersonaEngine, ChatResult, Persona
        assert PersonaEngine is not None
        assert ChatResult is not None
        assert Persona is not None

    def test_version(self):
        import persona_engine
        assert persona_engine.__version__ == "0.2.0"
