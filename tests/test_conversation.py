"""
Tests for the Conversation multi-turn dialogue wrapper.

Covers: say, say_all, properties, iteration, analysis, export, and repr.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from persona_engine import Conversation, PersonaEngine


@pytest.fixture
def engine():
    """Create a mock-backed engine from the chef persona."""
    return PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock")


@pytest.fixture
def convo(engine):
    """Create a Conversation with optional metadata."""
    return Conversation(engine, metadata={"test": True, "scenario": "unit"})


@pytest.fixture
def convo_with_turns(convo):
    """A Conversation with two completed turns."""
    convo.say("What makes a perfect sauce?")
    convo.say("And what about soups?")
    return convo


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_init(self, engine):
        convo = Conversation(engine)
        assert convo.engine is engine
        assert convo.turn_count == 0

    def test_init_with_metadata(self, engine):
        convo = Conversation(engine, metadata={"key": "value"})
        assert convo._metadata == {"key": "value"}

    def test_init_metadata_defaults_to_empty(self, engine):
        convo = Conversation(engine)
        assert convo._metadata == {}


# ---------------------------------------------------------------------------
# Core API: say / say_all
# ---------------------------------------------------------------------------


class TestSay:
    def test_say_returns_chat_result(self, convo):
        result = convo.say("Hello!")
        assert result.text
        assert result.turn_number == 1

    def test_say_increments_turns(self, convo):
        convo.say("First message")
        convo.say("Second message")
        assert convo.turn_count == 2

    def test_say_all_returns_list(self, convo):
        results = convo.say_all(["Hello", "How are you?", "Goodbye"])
        assert len(results) == 3
        assert results[0].turn_number == 1
        assert results[2].turn_number == 3

    def test_say_all_empty_list(self, convo):
        results = convo.say_all([])
        assert results == []
        assert convo.turn_count == 0


# ---------------------------------------------------------------------------
# Introspection properties
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_turns_property(self, convo_with_turns):
        turns = convo_with_turns.turns
        assert len(turns) == 2
        assert turns[0].turn_number == 1
        assert turns[1].turn_number == 2

    def test_turn_count(self, convo_with_turns):
        assert convo_with_turns.turn_count == 2

    def test_engine_property(self, convo, engine):
        assert convo.engine is engine

    def test_persona_name(self, convo):
        name = convo.persona_name
        assert isinstance(name, str)
        assert len(name) > 0

    def test_last_with_turns(self, convo_with_turns):
        last = convo_with_turns.last()
        assert last is not None
        assert last.turn_number == 2

    def test_last_no_turns(self, convo):
        assert convo.last() is None


# ---------------------------------------------------------------------------
# Iteration / collection protocol
# ---------------------------------------------------------------------------


class TestIteration:
    def test_iter(self, convo_with_turns):
        turns = list(convo_with_turns)
        assert len(turns) == 2

    def test_len(self, convo_with_turns):
        assert len(convo_with_turns) == 2

    def test_len_empty(self, convo):
        assert len(convo) == 0

    def test_getitem(self, convo_with_turns):
        first = convo_with_turns[0]
        assert first.turn_number == 1
        last = convo_with_turns[-1]
        assert last.turn_number == 2

    def test_getitem_out_of_range(self, convo_with_turns):
        with pytest.raises(IndexError):
            convo_with_turns[99]


# ---------------------------------------------------------------------------
# Analysis: summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_empty(self, convo):
        s = convo.summary()
        assert s["turn_count"] == 0
        assert s["persona"] == convo.persona_name
        assert "conversation_id" in s

    def test_summary_with_turns(self, convo_with_turns):
        s = convo_with_turns.summary()
        assert s["turn_count"] == 2
        assert 0.0 <= s["avg_confidence"] <= 1.0
        assert 0.0 <= s["avg_competence"] <= 1.0
        assert isinstance(s["all_passed_validation"], bool)
        assert "memory_stats" in s
        assert s["metadata"] == {"test": True, "scenario": "unit"}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_to_dict(self, convo_with_turns):
        d = convo_with_turns.to_dict()
        assert d["persona"] == convo_with_turns.persona_name
        assert d["persona_id"] == convo_with_turns.engine.persona.persona_id
        assert len(d["turns"]) == 2
        turn = d["turns"][0]
        assert "turn" in turn
        assert "user" in turn
        assert "response" in turn
        assert "confidence" in turn
        assert "competence" in turn
        assert "tone" in turn
        assert "passed_validation" in turn
        assert "summary" in d

    def test_export_json(self, convo_with_turns):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            convo_with_turns.export_json(path)
            data = json.loads(Path(path).read_text())
            assert data["persona"] == convo_with_turns.persona_name
            assert len(data["turns"]) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_yaml(self, convo_with_turns):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            convo_with_turns.export_yaml(path)
            data = yaml.safe_load(Path(path).read_text())
            assert data["persona"] == convo_with_turns.persona_name
            assert len(data["turns"]) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_transcript_returns_string(self, convo_with_turns):
        transcript = convo_with_turns.export_transcript()
        assert isinstance(transcript, str)
        assert convo_with_turns.persona_name in transcript
        assert "Turn 1" in transcript
        assert "Turn 2" in transcript

    def test_export_transcript_to_file(self, convo_with_turns):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            result = convo_with_turns.export_transcript(path)
            assert Path(path).read_text() == result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_markdown_returns_string(self, convo_with_turns):
        md = convo_with_turns.export_markdown()
        assert isinstance(md, str)
        assert "# Conversation Report" in md
        assert "## Conversation" in md
        assert "## Summary" in md
        assert "| Confidence |" in md

    def test_export_markdown_to_file(self, convo_with_turns):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            result = convo_with_turns.export_markdown(path)
            assert Path(path).read_text() == result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_markdown_includes_metadata(self, convo_with_turns):
        md = convo_with_turns.export_markdown()
        assert "## Metadata" in md
        assert "test" in md

    def test_export_transcript_empty(self, convo):
        transcript = convo.export_transcript()
        assert convo.persona_name in transcript
        assert "Turn" not in transcript


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_empty(self, convo):
        r = repr(convo)
        assert "Conversation(" in r
        assert "turns=0" in r

    def test_repr_with_turns(self, convo_with_turns):
        r = repr(convo_with_turns)
        assert "turns=2" in r
