"""
Shared pytest fixtures for the persona-engine test suite.

These fixtures provide reusable building blocks so that individual test
modules can focus on assertions rather than setup.  When a test module
defines its own fixture with the same name, pytest gives the local
fixture priority -- so nothing here will break existing inline fixtures.
"""

from __future__ import annotations

import yaml
import pytest

from persona_engine.engine import PersonaEngine
from persona_engine.generation.llm_adapter import MockLLMAdapter
from persona_engine.memory import MemoryManager
from persona_engine.memory.memory_manager import _make_id
from persona_engine.memory.models import (
    Fact,
    MemorySource,
    MemoryType,
)
from persona_engine.schema.ir_schema import (
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    KnowledgeClaimType,
    MemoryOps,
    ResponseStructure,
    SafetyPlan,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Persona data dictionary (moderate/mid-range traits)
# ============================================================================

_SAMPLE_PERSONA_DATA: dict = {
    "persona_id": "CONFTEST_SAMPLE",
    "version": "1.0",
    "label": "Conftest Sample Persona",
    "identity": {
        "age": 35,
        "gender": "non-binary",
        "location": "Portland, Oregon",
        "education": "Bachelor's in Psychology",
        "occupation": "Product Manager",
        "background": "Former UX designer turned product manager with 8 years of tech experience.",
    },
    "psychology": {
        "big_five": {
            "openness": 0.65,
            "conscientiousness": 0.60,
            "extraversion": 0.50,
            "agreeableness": 0.55,
            "neuroticism": 0.40,
        },
        "values": {
            "self_direction": 0.65,
            "stimulation": 0.50,
            "hedonism": 0.45,
            "achievement": 0.60,
            "power": 0.35,
            "security": 0.50,
            "conformity": 0.40,
            "tradition": 0.35,
            "benevolence": 0.60,
            "universalism": 0.55,
        },
        "cognitive_style": {
            "analytical_intuitive": 0.60,
            "systematic_heuristic": 0.55,
            "risk_tolerance": 0.50,
            "need_for_closure": 0.45,
            "cognitive_complexity": 0.65,
        },
        "communication": {
            "verbosity": 0.50,
            "formality": 0.45,
            "directness": 0.55,
            "emotional_expressiveness": 0.50,
        },
    },
    "knowledge_domains": [
        {"domain": "Product Management", "proficiency": 0.80, "subdomains": ["Agile", "User Research"]},
        {"domain": "Psychology", "proficiency": 0.60, "subdomains": ["Cognitive Psychology"]},
        {"domain": "Technology", "proficiency": 0.55, "subdomains": ["Web Development"]},
    ],
    "languages": [{"language": "English", "proficiency": 1.0}],
    "cultural_knowledge": {
        "primary_culture": "American",
        "exposure_level": {"european": 0.5, "asian": 0.3},
    },
    "primary_goals": [
        {"goal": "Ship impactful products", "weight": 0.8},
        {"goal": "Grow as a leader", "weight": 0.6},
    ],
    "social_roles": {
        "default": {"formality": 0.45, "directness": 0.55, "emotional_expressiveness": 0.50},
    },
    "uncertainty": {
        "admission_threshold": 0.45,
        "hedging_frequency": 0.40,
        "clarification_tendency": 0.50,
        "knowledge_boundary_strictness": 0.60,
    },
    "claim_policy": {
        "allowed_claim_types": ["personal_experience", "domain_expert", "general_common_knowledge"],
        "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
        "lookup_behavior": "hedge",
    },
    "invariants": {
        "identity_facts": ["Lives in Portland, Oregon", "Age 35", "Product Manager"],
        "cannot_claim": ["medical doctor", "licensed therapist"],
        "must_avoid": ["sharing confidential product roadmaps"],
    },
    "initial_state": {
        "mood_valence": 0.2,
        "mood_arousal": 0.5,
        "fatigue": 0.25,
        "stress": 0.30,
        "engagement": 0.65,
    },
    "biases": [{"type": "confirmation_bias", "strength": 0.25}],
    "time_scarcity": 0.45,
    "privacy_sensitivity": 0.50,
    "disclosure_policy": {
        "base_openness": 0.55,
        "factors": {
            "topic_sensitivity": -0.25,
            "trust_level": 0.30,
            "formal_context": -0.15,
            "positive_mood": 0.10,
        },
        "bounds": [0.1, 0.9],
    },
}


# ============================================================================
# Shared test helper — canonical persona data factory
# ============================================================================

# Baseline persona data with all traits at 0.5.  Designed for R-phase tests
# where a single trait is tweaked via overrides.
_BASELINE_PERSONA_DATA: dict = {
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


def make_persona_data(**overrides) -> dict:
    """Create a baseline persona data dict with trait overrides.

    Supports three override styles:
    - Big Five traits directly: ``openness=0.9``
    - Cognitive style with ``cog_`` prefix: ``cog_analytical_intuitive=0.9``
    - Values with ``val_`` prefix: ``val_self_direction=0.85``
    - Values as dict: ``values={"self_direction": 0.85}``

    Returns a fresh dict (safe to mutate).
    """
    import copy
    base = copy.deepcopy(_BASELINE_PERSONA_DATA)
    for key, val in overrides.items():
        if key in base["psychology"]["big_five"]:
            base["psychology"]["big_five"][key] = val
        elif key.startswith("cog_"):
            base["psychology"]["cognitive_style"][key[4:]] = val
        elif key.startswith("val_"):
            base["psychology"]["values"][key[4:]] = val
        elif key == "values" and isinstance(val, dict):
            base["psychology"]["values"].update(val)
    return base


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_persona() -> Persona:
    """A basic Persona object with moderate/mid-range traits.

    All Big Five traits are near 0.5-0.65, making this persona
    psychologically "average" and useful as a neutral baseline.
    """
    return Persona(**_SAMPLE_PERSONA_DATA)


@pytest.fixture
def chef_engine() -> PersonaEngine:
    """PersonaEngine loaded from personas/chef.yaml with a mock LLM provider.

    Uses MockLLMAdapter so tests never make real API calls.
    """
    return PersonaEngine.from_yaml(
        "personas/chef.yaml",
        adapter=MockLLMAdapter(),
        seed=42,
    )


@pytest.fixture
def template_engine() -> PersonaEngine:
    """PersonaEngine loaded from personas/chef.yaml with the template provider.

    The template provider returns structured placeholder text instead of
    calling an LLM, useful for testing prompt construction and format.
    """
    return PersonaEngine.from_yaml(
        "personas/chef.yaml",
        llm_provider="template",
        seed=42,
    )


@pytest.fixture
def mock_adapter() -> MockLLMAdapter:
    """A fresh MockLLMAdapter instance for unit-level generation tests."""
    return MockLLMAdapter()


@pytest.fixture
def sample_ir() -> IntermediateRepresentation:
    """A pre-built IntermediateRepresentation with sensible defaults.

    Represents a moderate, mid-confidence response in casual-chat mode.
    Useful for testing validators, generators, and serialization without
    going through the full planning pipeline.
    """
    return IntermediateRepresentation(
        conversation_frame=ConversationFrame(
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
        ),
        response_structure=ResponseStructure(
            intent="Share perspective on the topic based on personal experience",
            stance="Generally supportive with minor reservations",
            rationale="Based on professional experience in product management",
            elasticity=0.6,
            confidence=0.7,
            competence=0.65,
        ),
        communication_style=CommunicationStyle(
            tone=Tone.THOUGHTFUL_ENGAGED,
            verbosity=Verbosity.MEDIUM,
            formality=0.45,
            directness=0.55,
        ),
        knowledge_disclosure=KnowledgeAndDisclosure(
            disclosure_level=0.5,
            uncertainty_action=UncertaintyAction.HEDGE,
            knowledge_claim_type=KnowledgeClaimType.PERSONAL_EXPERIENCE,
        ),
        citations=[],
        safety_plan=SafetyPlan(),
        memory_ops=MemoryOps(),
        seed=42,
    )


@pytest.fixture
def twin_pair() -> tuple[PersonaEngine, PersonaEngine]:
    """A (high_openness_engine, low_openness_engine) pair for counterfactual testing.

    Both engines use MockLLMAdapter and the same seed so that any
    behavioral difference can be attributed to the openness trait alone.
    """
    high = PersonaEngine.from_yaml(
        "personas/twins/high_openness.yaml",
        adapter=MockLLMAdapter(),
        seed=42,
    )
    low = PersonaEngine.from_yaml(
        "personas/twins/low_openness.yaml",
        adapter=MockLLMAdapter(),
        seed=42,
    )
    return (high, low)


@pytest.fixture
def memory_manager() -> MemoryManager:
    """A MemoryManager pre-loaded with several facts for multi-turn tests.

    Contains three facts covering occupation, preference, and hobby so
    that tests exercising memory retrieval have data to work with
    immediately.
    """
    manager = MemoryManager()

    facts = [
        Fact(
            memory_id=_make_id("User works as a data scientist", 1),
            memory_type=MemoryType.FACT,
            content="User works as a data scientist",
            confidence=0.9,
            privacy_level=0.3,
            source=MemorySource.USER_STATED,
            turn_created=1,
            conversation_id="conftest-conv-001",
            category="occupation",
        ),
        Fact(
            memory_id=_make_id("User prefers concise answers", 2),
            memory_type=MemoryType.FACT,
            content="User prefers concise answers",
            confidence=0.8,
            privacy_level=0.1,
            source=MemorySource.INFERRED,
            turn_created=2,
            conversation_id="conftest-conv-001",
            category="preference",
        ),
        Fact(
            memory_id=_make_id("User enjoys hiking on weekends", 3),
            memory_type=MemoryType.FACT,
            content="User enjoys hiking on weekends",
            confidence=0.85,
            privacy_level=0.2,
            source=MemorySource.USER_STATED,
            turn_created=3,
            conversation_id="conftest-conv-001",
            category="hobby",
        ),
    ]

    for fact in facts:
        manager.facts.store(fact)

    return manager
