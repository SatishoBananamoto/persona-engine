"""
Tests for Phase 4 fixes — Developer Experience & SDK Polish.

Covers:
- Fix 4.1: README code examples parse as valid Python
- Fix 4.2: __repr__ methods on PersonaEngine and IntermediateRepresentation
- Fix 4.3: Context manager support (__enter__/__exit__/close)
- Fix 4.4: Docstring examples on public methods
- Fix 4.5: Example scripts parse as valid Python
"""

import ast
import re
from pathlib import Path

import pytest

from persona_engine.engine import PersonaEngine
from persona_engine.generation.llm_adapter import MockLLMAdapter
from persona_engine.persona_builder import PersonaBuilder
from persona_engine.schema.persona_schema import Persona


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PERSONA_DATA = {
    "persona_id": "PHASE4_TEST",
    "version": "1.0",
    "label": "Phase 4 Test Persona",
    "identity": {
        "age": 30, "gender": "female", "location": "Boston",
        "education": "MS Biology", "occupation": "Researcher",
        "background": "Marine biologist studying coral reefs.",
    },
    "psychology": {
        "big_five": {
            "openness": 0.8, "conscientiousness": 0.7,
            "extraversion": 0.4, "agreeableness": 0.7, "neuroticism": 0.3,
        },
        "values": {
            "self_direction": 0.7, "stimulation": 0.5, "hedonism": 0.4,
            "achievement": 0.6, "power": 0.2, "security": 0.5,
            "conformity": 0.4, "tradition": 0.3, "benevolence": 0.7,
            "universalism": 0.8,
        },
        "cognitive_style": {
            "analytical_intuitive": 0.7, "systematic_heuristic": 0.7,
            "risk_tolerance": 0.4, "need_for_closure": 0.5,
            "cognitive_complexity": 0.7,
        },
        "communication": {
            "verbosity": 0.5, "formality": 0.5,
            "directness": 0.5, "emotional_expressiveness": 0.5,
        },
    },
    "knowledge_domains": [
        {"domain": "Biology", "proficiency": 0.9, "subdomains": ["Marine"]},
    ],
    "languages": [{"language": "English", "proficiency": 1.0}],
    "cultural_knowledge": {
        "primary_culture": "American",
        "exposure_level": {"european": 0.4},
    },
    "primary_goals": [{"goal": "Protect coral reefs", "weight": 0.9}],
    "social_roles": {
        "default": {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5},
    },
    "uncertainty": {
        "admission_threshold": 0.4, "hedging_frequency": 0.4,
        "clarification_tendency": 0.5, "knowledge_boundary_strictness": 0.7,
    },
    "claim_policy": {
        "allowed_claim_types": ["personal_experience", "domain_expert"],
        "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
        "lookup_behavior": "hedge",
    },
    "invariants": {
        "identity_facts": ["Marine biologist"],
        "cannot_claim": [],
        "must_avoid": [],
    },
    "initial_state": {
        "mood_valence": 0.1, "mood_arousal": 0.5,
        "fatigue": 0.2, "stress": 0.2, "engagement": 0.7,
    },
    "biases": [{"type": "confirmation_bias", "strength": 0.2}],
    "time_scarcity": 0.4,
    "privacy_sensitivity": 0.4,
    "disclosure_policy": {
        "base_openness": 0.6,
        "factors": {
            "topic_sensitivity": -0.2, "trust_level": 0.3,
            "formal_context": -0.1, "positive_mood": 0.15,
        },
        "bounds": [0.2, 0.8],
    },
}


@pytest.fixture
def persona():
    return Persona(**PERSONA_DATA)


@pytest.fixture
def engine(persona):
    return PersonaEngine(persona, adapter=MockLLMAdapter())


# ===========================================================================
# Fix 4.1: README examples parse as valid Python
# ===========================================================================


class TestReadmeExamples:
    """Extract code blocks from README and verify they parse."""

    def test_readme_code_blocks_parse(self):
        readme = Path("README.md").read_text()
        # Extract all ```python ... ``` blocks
        blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
        assert len(blocks) >= 3, f"Expected at least 3 code blocks, found {len(blocks)}"
        for i, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as e:
                pytest.fail(f"README code block {i+1} has syntax error: {e}\n{block}")

    def test_readme_no_old_api(self):
        """Ensure removed APIs are not referenced in README."""
        readme = Path("README.md").read_text()
        assert "start_conversation" not in readme
        assert ".send(" not in readme
        assert "load_persona" not in readme


# ===========================================================================
# Fix 4.2: __repr__ methods
# ===========================================================================


class TestReprMethods:

    def test_persona_engine_repr(self, engine):
        r = repr(engine)
        assert "PersonaEngine" in r
        assert "Phase 4 Test Persona" in r
        assert "turns=0" in r

    def test_persona_engine_repr_after_turns(self, engine):
        engine.chat("Hello")
        r = repr(engine)
        assert "turns=1" in r

    def test_ir_repr(self, engine):
        ir = engine.plan("Tell me about marine biology")
        r = repr(ir)
        assert "IR(" in r
        assert "comp=" in r
        assert "conf=" in r

    def test_chat_result_repr(self, engine):
        result = engine.chat("Hello")
        r = repr(result)
        assert "ChatResult" in r
        assert "turn=1" in r


# ===========================================================================
# Fix 4.3: Context manager
# ===========================================================================


class TestContextManager:

    def test_enter_returns_engine(self, persona):
        engine = PersonaEngine(persona, adapter=MockLLMAdapter())
        with engine as e:
            assert e is engine

    def test_with_block_works(self, persona):
        with PersonaEngine(persona, adapter=MockLLMAdapter()) as engine:
            result = engine.chat("Hello")
            assert result.text

    def test_close_is_callable(self, engine):
        engine.close()  # Should not raise


# ===========================================================================
# Fix 4.4: Docstring examples
# ===========================================================================


class TestDocstrings:
    """Verify that key public methods have docstring examples."""

    @pytest.mark.parametrize("method_name", [
        "chat", "plan", "from_yaml", "from_description",
        "reset", "save", "load", "memory_stats", "system_prompt",
    ])
    def test_public_methods_have_examples(self, method_name):
        method = getattr(PersonaEngine, method_name)
        doc = method.__doc__ or ""
        assert "Example" in doc or "example" in doc, (
            f"PersonaEngine.{method_name} missing docstring example"
        )


# ===========================================================================
# Fix 4.5: Example scripts parse
# ===========================================================================


class TestExampleScripts:
    """Verify all example scripts in examples/ are valid Python."""

    def test_example_scripts_parse(self):
        examples_dir = Path("examples")
        scripts = list(examples_dir.glob("*.py"))
        assert len(scripts) >= 5, f"Expected at least 5 example scripts, found {len(scripts)}"
        for script in scripts:
            try:
                ast.parse(script.read_text())
            except SyntaxError as e:
                pytest.fail(f"{script.name} has syntax error: {e}")
