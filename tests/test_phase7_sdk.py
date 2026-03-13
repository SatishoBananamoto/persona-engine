"""
Tests for Phase 7: Python SDK (Local Mode)

Covers:
- Conversation class (multi-turn, iteration, export)
- CLI tool (validate, info, plan, list)
- SDK public exports
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from persona_engine import (
    ChatResult,
    Conversation,
    PersonaEngine,
    PersonaBuilder,
    # Schema types
    Persona,
    PersonalityProfile,
    BigFiveTraits,
    SchwartzValues,
    CognitiveStyle,
    CommunicationPreferences,
    DomainKnowledge,
    Goal,
    # IR types
    IntermediateRepresentation,
    ConversationFrame,
    ResponseStructure,
    CommunicationStyle,
    KnowledgeAndDisclosure,
    # IR enums
    InteractionMode,
    ConversationGoal,
    Tone,
    Verbosity,
    UncertaintyAction,
)
from persona_engine.__main__ import main


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def engine():
    """Create a mock-backed PersonaEngine from the chef persona."""
    return PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")


@pytest.fixture
def conversation(engine):
    """Create a Conversation wrapping a mock engine."""
    return Conversation(engine, metadata={"test": True})


# =============================================================================
# Conversation Class
# =============================================================================

class TestConversation:
    """Test the Conversation multi-turn wrapper."""

    def test_say_returns_chat_result(self, conversation):
        result = conversation.say("What makes a good sauce?")
        assert isinstance(result, ChatResult)
        assert result.turn_number == 1
        assert len(result.text) > 0

    def test_multi_turn(self, conversation):
        conversation.say("Hello")
        conversation.say("Tell me about sauces")
        conversation.say("And soups?")
        assert conversation.turn_count == 3
        assert len(conversation.turns) == 3

    def test_say_all(self, conversation):
        results = conversation.say_all(["Hello", "How are you?", "Goodbye"])
        assert len(results) == 3
        assert all(isinstance(r, ChatResult) for r in results)

    def test_iteration(self, conversation):
        conversation.say("Hello")
        conversation.say("Hi again")
        turns = list(conversation)
        assert len(turns) == 2
        assert turns[0].turn_number == 1
        assert turns[1].turn_number == 2

    def test_len(self, conversation):
        assert len(conversation) == 0
        conversation.say("Hello")
        assert len(conversation) == 1

    def test_getitem(self, conversation):
        conversation.say("First")
        conversation.say("Second")
        assert conversation[0].turn_number == 1
        assert conversation[1].turn_number == 2

    def test_last(self, conversation):
        assert conversation.last() is None
        conversation.say("Hello")
        assert conversation.last().turn_number == 1

    def test_persona_name(self, conversation):
        assert len(conversation.persona_name) > 0

    def test_engine_property(self, conversation, engine):
        assert conversation.engine is engine

    def test_repr(self, conversation):
        r = repr(conversation)
        assert "Conversation(" in r
        assert "turns=" in r


class TestConversationSummary:
    """Test conversation summary generation."""

    def test_empty_summary(self, conversation):
        summary = conversation.summary()
        assert summary["turn_count"] == 0

    def test_summary_with_turns(self, conversation):
        conversation.say("Hello")
        conversation.say("Tell me more")
        summary = conversation.summary()
        assert summary["turn_count"] == 2
        assert "avg_confidence" in summary
        assert "avg_competence" in summary
        assert "all_passed_validation" in summary
        assert "memory_stats" in summary
        assert summary["metadata"] == {"test": True}


class TestConversationExport:
    """Test conversation export formats."""

    def test_to_dict(self, conversation):
        conversation.say("Hello")
        d = conversation.to_dict()
        assert "persona" in d
        assert "turns" in d
        assert len(d["turns"]) == 1
        turn = d["turns"][0]
        assert "user" in turn
        assert "response" in turn
        assert "confidence" in turn
        assert "tone" in turn

    def test_export_json(self, conversation, tmp_path):
        conversation.say("Hello")
        path = tmp_path / "convo.json"
        conversation.export_json(path)
        data = json.loads(path.read_text())
        assert len(data["turns"]) == 1

    def test_export_yaml(self, conversation, tmp_path):
        conversation.say("Hello")
        path = tmp_path / "convo.yaml"
        conversation.export_yaml(path)
        data = yaml.safe_load(path.read_text())
        assert len(data["turns"]) == 1

    def test_export_transcript(self, conversation):
        conversation.say("Hello there")
        transcript = conversation.export_transcript()
        assert "Turn 1" in transcript
        assert conversation.persona_name in transcript

    def test_export_transcript_to_file(self, conversation, tmp_path):
        conversation.say("Hello")
        path = tmp_path / "transcript.md"
        transcript = conversation.export_transcript(path)
        assert path.read_text() == transcript


# =============================================================================
# CLI Tool
# =============================================================================

class TestCLI:
    """Test the CLI tool (python -m persona_engine)."""

    def test_no_command_shows_help(self):
        with patch("sys.argv", ["persona_engine"]):
            result = main()
        assert result == 0

    def test_validate_valid_persona(self):
        with patch("sys.argv", ["persona_engine", "validate", "personas/chef.yaml"]):
            result = main()
        assert result == 0

    def test_validate_deep(self):
        with patch("sys.argv", ["persona_engine", "validate", "--deep", "personas/chef.yaml"]):
            result = main()
        assert result == 0

    def test_validate_invalid_file(self):
        with patch("sys.argv", ["persona_engine", "validate", "nonexistent.yaml"]):
            result = main()
        assert result == 1

    def test_info(self):
        with patch("sys.argv", ["persona_engine", "info", "personas/chef.yaml"]):
            result = main()
        assert result == 0

    def test_plan(self):
        with patch("sys.argv", ["persona_engine", "plan", "personas/chef.yaml", "What is cooking?"]):
            result = main()
        assert result == 0

    def test_plan_json(self):
        with patch("sys.argv", ["persona_engine", "plan", "--json", "personas/chef.yaml", "What is cooking?"]):
            result = main()
        assert result == 0

    def test_chat(self):
        with patch("sys.argv", ["persona_engine", "chat", "personas/chef.yaml", "Hello"]):
            result = main()
        assert result == 0

    def test_list(self):
        with patch("sys.argv", ["persona_engine", "list", "personas"]):
            result = main()
        assert result == 0

    def test_list_default_directory(self):
        with patch("sys.argv", ["persona_engine", "list"]):
            result = main()
        assert result == 0

    def test_module_invocation(self):
        """Test python -m persona_engine works."""
        result = subprocess.run(
            [sys.executable, "-m", "persona_engine", "validate", "personas/chef.yaml"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout


# =============================================================================
# SDK Public Exports
# =============================================================================

class TestSDKExports:
    """Test that all expected types are importable from persona_engine."""

    def test_core_sdk_exports(self):
        """Core SDK classes are importable."""
        assert PersonaEngine is not None
        assert Conversation is not None
        assert PersonaBuilder is not None
        assert ChatResult is not None

    def test_persona_schema_exports(self):
        """Persona schema types are importable."""
        assert Persona is not None
        assert PersonalityProfile is not None
        assert BigFiveTraits is not None
        assert SchwartzValues is not None
        assert CognitiveStyle is not None
        assert CommunicationPreferences is not None
        assert DomainKnowledge is not None
        assert Goal is not None

    def test_ir_schema_exports(self):
        """IR schema types are importable."""
        assert IntermediateRepresentation is not None
        assert ConversationFrame is not None
        assert ResponseStructure is not None
        assert CommunicationStyle is not None
        assert KnowledgeAndDisclosure is not None

    def test_enum_exports(self):
        """IR enums are importable."""
        assert InteractionMode.CASUAL_CHAT is not None
        assert ConversationGoal.EXPLORE_IDEAS is not None
        assert Tone.THOUGHTFUL_ENGAGED is not None
        assert Verbosity.MEDIUM is not None
        assert UncertaintyAction.HEDGE is not None

    def test_version(self):
        import persona_engine
        assert persona_engine.__version__ == "0.2.0"


# =============================================================================
# Conversation with PersonaEngine integration
# =============================================================================

class TestConversationIntegration:
    """Test Conversation integrates properly with PersonaEngine."""

    def test_conversation_shares_engine_state(self, engine):
        """Conversation and engine share the same state."""
        convo = Conversation(engine)
        convo.say("Hello")
        assert engine.turn_count == 1
        assert len(engine.history) == 1

    def test_conversation_with_custom_persona(self):
        """Conversation works with builder-created personas."""
        persona = (
            PersonaBuilder("Test Person", "Engineer")
            .trait("curious", "analytical")
            .build()
        )
        engine = PersonaEngine(persona, llm_provider="mock")
        convo = Conversation(engine)
        result = convo.say("How do you approach problems?")
        assert result.turn_number == 1

    def test_conversation_mode_and_goal(self, conversation):
        """Conversation forwards mode and goal to engine."""
        result = conversation.say(
            "Help me fix this bug",
            mode=InteractionMode.CUSTOMER_SUPPORT,
            goal=ConversationGoal.RESOLVE_ISSUE,
        )
        assert result.turn_number == 1

    def test_multiple_conversations_same_engine(self, engine):
        """Multiple Conversation objects can wrap the same engine."""
        c1 = Conversation(engine)
        c1.say("Hello")
        # Reset for new conversation
        engine.reset()
        c2 = Conversation(engine)
        c2.say("Hi again")
        assert c2.turn_count == 1
