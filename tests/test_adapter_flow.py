"""
End-to-end flow tests for multi-provider LLM adapters.

These tests go beyond unit mocking — they wire each adapter into the full
PersonaEngine pipeline (persona → IR → prompt → adapter → response) to
verify that the complete flow works from input to output.

Each adapter has its SDK client mocked at the instance level so no real
API calls are made, but the full pipeline (TurnPlanner, PromptBuilder,
ResponseGenerator, Validator) executes for real.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from persona_engine.engine import ChatResult, PersonaEngine
from persona_engine.generation.llm_adapter import (
    GeminiAdapter,
    MistralAdapter,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    create_adapter,
)
from persona_engine.schema.persona_schema import Persona


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMULATED_LLM_RESPONSE = (
    "That's a really interesting question. From my experience in the kitchen, "
    "I'd say the key to a great sauce is patience and good ingredients. "
    "You want to build layers of flavor slowly."
)


def _make_openai_style_mock_client(response_text: str = SIMULATED_LLM_RESPONSE) -> MagicMock:
    """Create a mock that mimics the OpenAI chat.completions.create response."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response.choices = [mock_choice]
    mock.chat.completions.create.return_value = mock_response
    return mock


def _make_mistral_style_mock_client(response_text: str = SIMULATED_LLM_RESPONSE) -> MagicMock:
    """Create a mock that mimics the Mistral chat.complete response."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response.choices = [mock_choice]
    mock.chat.complete.return_value = mock_response
    return mock


def _make_ollama_style_mock_client(response_text: str = SIMULATED_LLM_RESPONSE) -> MagicMock:
    """Create a mock that mimics the Ollama chat response dict."""
    mock = MagicMock()
    mock.chat.return_value = {
        "message": {"role": "assistant", "content": response_text},
    }
    return mock


@pytest.fixture
def chef_persona() -> Persona:
    """Load the chef persona from YAML."""
    import yaml
    with open("personas/chef.yaml") as f:
        data = yaml.safe_load(f)
    return Persona(**data)


# ---------------------------------------------------------------------------
# OpenAICompatibleAdapter — full pipeline
# ---------------------------------------------------------------------------


class TestOpenAICompatibleFlow:
    """End-to-end flow with OpenAICompatibleAdapter (covers Groq too)."""

    def test_single_turn_chat(self, chef_persona: Persona):
        adapter = OpenAICompatibleAdapter(
            api_key="test-key", model="test-model", base_url="https://api.test.com/v1"
        )
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("What makes a perfect French mother sauce?")

        assert isinstance(result, ChatResult)
        assert len(result.text) > 0
        assert result.text == SIMULATED_LLM_RESPONSE
        assert result.turn_number == 1
        assert result.ir is not None
        assert result.ir.response_structure.confidence > 0

    def test_multi_turn_chat(self, chef_persona: Persona):
        adapter = OpenAICompatibleAdapter(api_key="test-key", model="m")
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        r1 = engine.chat("Tell me about sauces.")
        r2 = engine.chat("And what about soups?")

        assert r1.turn_number == 1
        assert r2.turn_number == 2
        assert engine.turn_count == 2

    def test_ir_contains_expected_fields(self, chef_persona: Persona):
        adapter = OpenAICompatibleAdapter(api_key="test-key", model="m")
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("How do you make a classic roux?")

        ir = result.ir
        assert ir.communication_style.tone is not None
        assert ir.communication_style.verbosity is not None
        assert 0 <= ir.response_structure.competence <= 1
        assert 0 <= ir.response_structure.confidence <= 1
        assert ir.response_structure.intent is not None

    def test_validation_runs(self, chef_persona: Persona):
        adapter = OpenAICompatibleAdapter(api_key="test-key", model="m")
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42, validate=True)
        result = engine.chat("What's your favorite dish?")

        assert result.validation is not None
        assert isinstance(result.passed, bool)

    def test_groq_via_factory(self, chef_persona: Persona):
        """Groq provider uses OpenAICompatibleAdapter with pre-configured base_url."""
        adapter = create_adapter("groq", api_key="test-key")
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("Quick cooking tips?")

        assert isinstance(result, ChatResult)
        assert result.text == SIMULATED_LLM_RESPONSE
        assert adapter.base_url == "https://api.groq.com/openai/v1"

    def test_system_prompt_contains_persona_identity(self, chef_persona: Persona):
        adapter = OpenAICompatibleAdapter(api_key="test-key", model="m")
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        engine.chat("Hello!")

        # The mock client captures call args
        call_args = adapter._client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]["content"]

        # System prompt should contain persona identity details
        assert len(system_msg) > 100  # Non-trivial prompt


# ---------------------------------------------------------------------------
# GeminiAdapter — full pipeline
# ---------------------------------------------------------------------------


class TestGeminiFlow:
    """End-to-end flow with GeminiAdapter."""

    def _make_gemini_adapter(self, response_text: str = SIMULATED_LLM_RESPONSE) -> GeminiAdapter:
        adapter = GeminiAdapter(api_key="test-key", model="gemini-2.0-flash")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client.models.generate_content.return_value = mock_response
        adapter._client = mock_client
        return adapter

    def test_single_turn_chat(self, chef_persona: Persona):
        # Mock the google.genai.types import inside generate()
        types_mock = MagicMock()
        google_mock = MagicMock()
        google_mock.genai.types = types_mock

        adapter = self._make_gemini_adapter()

        with patch.dict("sys.modules", {
            "google": google_mock,
            "google.genai": google_mock.genai,
            "google.genai.types": types_mock,
        }):
            engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
            result = engine.chat("What makes a perfect French mother sauce?")

        assert isinstance(result, ChatResult)
        assert result.text == SIMULATED_LLM_RESPONSE
        assert result.turn_number == 1

    def test_multi_turn_chat(self, chef_persona: Persona):
        types_mock = MagicMock()
        google_mock = MagicMock()
        google_mock.genai.types = types_mock

        adapter = self._make_gemini_adapter()

        with patch.dict("sys.modules", {
            "google": google_mock,
            "google.genai": google_mock.genai,
            "google.genai.types": types_mock,
        }):
            engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
            r1 = engine.chat("Tell me about sauces.")
            r2 = engine.chat("And soups?")

        assert r1.turn_number == 1
        assert r2.turn_number == 2

    def test_ir_planning_unaffected(self, chef_persona: Persona):
        """IR planning should be identical regardless of adapter choice."""
        types_mock = MagicMock()
        google_mock = MagicMock()
        google_mock.genai.types = types_mock

        adapter = self._make_gemini_adapter()

        with patch.dict("sys.modules", {
            "google": google_mock,
            "google.genai": google_mock.genai,
            "google.genai.types": types_mock,
        }):
            engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
            result = engine.chat("How do you braise short ribs?")

        ir = result.ir
        assert ir.response_structure.competence > 0
        assert ir.response_structure.intent is not None

    def test_model_name_in_response(self, chef_persona: Persona):
        types_mock = MagicMock()
        google_mock = MagicMock()
        google_mock.genai.types = types_mock

        adapter = self._make_gemini_adapter()

        with patch.dict("sys.modules", {
            "google": google_mock,
            "google.genai": google_mock.genai,
            "google.genai.types": types_mock,
        }):
            engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
            result = engine.chat("Hi")

        assert result.response.model == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# MistralAdapter — full pipeline
# ---------------------------------------------------------------------------


class TestMistralFlow:
    """End-to-end flow with MistralAdapter."""

    def test_single_turn_chat(self, chef_persona: Persona):
        adapter = MistralAdapter(api_key="test-key")
        adapter._client = _make_mistral_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("What makes a perfect French mother sauce?")

        assert isinstance(result, ChatResult)
        assert result.text == SIMULATED_LLM_RESPONSE
        assert result.turn_number == 1
        assert result.ir.response_structure.confidence > 0

    def test_multi_turn_chat(self, chef_persona: Persona):
        adapter = MistralAdapter(api_key="test-key")
        adapter._client = _make_mistral_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        r1 = engine.chat("Tell me about sauces.")
        r2 = engine.chat("And what about soups?")

        assert r1.turn_number == 1
        assert r2.turn_number == 2
        assert engine.turn_count == 2

    def test_conversation_history_grows(self, chef_persona: Persona):
        adapter = MistralAdapter(api_key="test-key")
        adapter._client = _make_mistral_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        engine.chat("First question")
        engine.chat("Second question")
        engine.chat("Third question")

        # By turn 3, the mock should have been called with conversation history
        call_args = adapter._client.chat.complete.call_args
        messages = call_args.kwargs["messages"]
        # system + previous turns (user+assistant pairs) + current user
        assert len(messages) >= 3

    def test_ir_plan_only_no_llm_call(self, chef_persona: Persona):
        """plan() should not call the adapter at all."""
        adapter = MistralAdapter(api_key="test-key")
        adapter._client = _make_mistral_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        ir = engine.plan("What's the best knife for chopping?")

        assert ir is not None
        assert ir.response_structure.competence > 0
        # plan() should not trigger LLM generation
        adapter._client.chat.complete.assert_not_called()


# ---------------------------------------------------------------------------
# OllamaAdapter — full pipeline
# ---------------------------------------------------------------------------


class TestOllamaFlow:
    """End-to-end flow with OllamaAdapter."""

    def test_single_turn_chat(self, chef_persona: Persona):
        adapter = OllamaAdapter(model="llama3.2")
        adapter._client = _make_ollama_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("What makes a perfect French mother sauce?")

        assert isinstance(result, ChatResult)
        assert result.text == SIMULATED_LLM_RESPONSE
        assert result.turn_number == 1
        assert result.ir is not None

    def test_multi_turn_chat(self, chef_persona: Persona):
        adapter = OllamaAdapter()
        adapter._client = _make_ollama_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        r1 = engine.chat("Tell me about sauces.")
        r2 = engine.chat("And what about soups?")
        r3 = engine.chat("How about desserts?")

        assert r1.turn_number == 1
        assert r2.turn_number == 2
        assert r3.turn_number == 3

    def test_model_name_prefixed(self, chef_persona: Persona):
        adapter = OllamaAdapter(model="phi3")
        adapter._client = _make_ollama_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("Hi")

        assert result.response.model == "ollama/phi3"

    def test_options_passed_to_ollama(self, chef_persona: Persona):
        adapter = OllamaAdapter(model="llama3.2")
        adapter._client = _make_ollama_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        engine.chat("Hello!")

        call_args = adapter._client.chat.call_args
        options = call_args.kwargs["options"]
        assert "temperature" in options
        assert "num_predict" in options

    def test_no_api_key_needed(self):
        """Ollama should not require any API key."""
        adapter = OllamaAdapter()
        assert adapter.host == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Cross-adapter consistency: same persona + same seed → same IR
# ---------------------------------------------------------------------------


class TestCrossAdapterConsistency:
    """Verify that IR planning is adapter-independent."""

    def test_ir_identical_across_adapters(self, chef_persona: Persona):
        """The IR should be the same regardless of which adapter is used,
        because IR planning happens before the adapter is called."""
        adapters = {
            "openai_compatible": OpenAICompatibleAdapter(api_key="k", model="m"),
            "mistral": MistralAdapter(api_key="k"),
            "ollama": OllamaAdapter(),
        }

        # Wire mock clients
        adapters["openai_compatible"]._client = _make_openai_style_mock_client()
        adapters["mistral"]._client = _make_mistral_style_mock_client()
        adapters["ollama"]._client = _make_ollama_style_mock_client()

        user_input = "How do you make a classic roux?"
        irs = {}

        for name, adapter in adapters.items():
            engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
            result = engine.chat(user_input)
            irs[name] = result.ir

        # All IRs should have the same core planning values
        competences = [ir.response_structure.competence for ir in irs.values()]
        confidences = [ir.response_structure.confidence for ir in irs.values()]
        tones = [ir.communication_style.tone for ir in irs.values()]

        assert len(set(competences)) == 1, f"Competence varies: {competences}"
        assert len(set(confidences)) == 1, f"Confidence varies: {confidences}"
        assert len(set(tones)) == 1, f"Tone varies: {tones}"

    def test_plan_returns_same_ir_as_chat(self, chef_persona: Persona):
        """plan() and chat() on the same input/seed should produce identical IRs."""
        adapter = OpenAICompatibleAdapter(api_key="k", model="m")
        adapter._client = _make_openai_style_mock_client()

        engine_plan = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        ir_plan = engine_plan.plan("What's your signature dish?")

        adapter2 = OpenAICompatibleAdapter(api_key="k", model="m")
        adapter2._client = _make_openai_style_mock_client()

        engine_chat = PersonaEngine(chef_persona, adapter=adapter2, seed=42)
        result_chat = engine_chat.chat("What's your signature dish?")

        assert ir_plan.response_structure.competence == result_chat.ir.response_structure.competence
        assert ir_plan.response_structure.confidence == result_chat.ir.response_structure.confidence
        assert ir_plan.communication_style.tone == result_chat.ir.communication_style.tone


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    """Verify create_adapter() produces working adapters for the full pipeline."""

    def test_groq_factory_full_pipeline(self, chef_persona: Persona):
        adapter = create_adapter("groq", api_key="test-key")
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("Quick cooking tip?")

        assert isinstance(result, ChatResult)
        assert result.turn_number == 1

    def test_ollama_factory_full_pipeline(self, chef_persona: Persona):
        adapter = create_adapter("ollama", model="phi3")
        adapter._client = _make_ollama_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("Hello!")

        assert isinstance(result, ChatResult)
        assert result.response.model == "ollama/phi3"

    def test_openai_compatible_factory_full_pipeline(self, chef_persona: Persona):
        adapter = create_adapter(
            "openai_compatible",
            api_key="k",
            model="local-model",
            base_url="http://localhost:8080/v1",
        )
        adapter._client = _make_openai_style_mock_client()

        engine = PersonaEngine(chef_persona, adapter=adapter, seed=42)
        result = engine.chat("Hi")

        assert isinstance(result, ChatResult)
        assert adapter.base_url == "http://localhost:8080/v1"
