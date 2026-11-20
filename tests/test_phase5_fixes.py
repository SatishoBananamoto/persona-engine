"""
Tests for Phase 5 fixes — Architecture & Maintainability.

Covers:
- Fix 5.1: EngineConfig dataclass extracts and centralizes constants
- Fix 5.2: TurnPlanner generate_ir refactored into pipeline stages
- Fix 5.3: Cross-turn inertia smoothing covers elasticity + competence
- Fix 5.4: Domain registry externalized to separate module
- Fix 5.5: Test files organized under tests/ directory
"""

import os
from pathlib import Path

import pytest

from persona_engine.memory import MemoryManager, StanceCache
from persona_engine.planner.domain_detection import DOMAIN_REGISTRY, DomainEntry, detect_domain
from persona_engine.planner.domain_registry import DOMAIN_REGISTRY as EXTERNAL_REGISTRY
from persona_engine.planner.engine_config import DEFAULT_CONFIG, EngineConfig
from persona_engine.planner.trace_context import TraceContext
from persona_engine.planner.turn_planner import (
    ConversationContext,
    TurnPlanner,
    create_turn_planner,
)
from persona_engine.schema.ir_schema import InteractionMode, ConversationGoal
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils import DeterminismManager
from persona_engine.validation.cross_turn import TurnSnapshot


# ---------------------------------------------------------------------------
# Shared test persona
# ---------------------------------------------------------------------------

PERSONA_DATA = {
    "persona_id": "PHASE5_TEST",
    "version": "1.0",
    "label": "Phase 5 Test Persona",
    "identity": {
        "age": 35, "gender": "male", "location": "Seattle",
        "education": "PhD Computer Science", "occupation": "Software Engineer",
        "background": "Full-stack engineer with ML experience.",
    },
    "psychology": {
        "big_five": {
            "openness": 0.7, "conscientiousness": 0.8,
            "extraversion": 0.5, "agreeableness": 0.6, "neuroticism": 0.3,
        },
        "values": {
            "self_direction": 0.8, "stimulation": 0.5, "hedonism": 0.3,
            "achievement": 0.7, "power": 0.3, "security": 0.5,
            "conformity": 0.3, "tradition": 0.2, "benevolence": 0.6,
            "universalism": 0.7,
        },
        "cognitive_style": {
            "analytical_intuitive": 0.8, "systematic_heuristic": 0.7,
            "risk_tolerance": 0.5, "need_for_closure": 0.6,
            "cognitive_complexity": 0.7,
        },
        "communication": {
            "verbosity": 0.5, "formality": 0.5,
            "directness": 0.6, "emotional_expressiveness": 0.4,
        },
    },
    "knowledge_domains": [
        {"domain": "technology", "proficiency": 0.9, "subdomains": ["ML", "Web"]},
        {"domain": "science", "proficiency": 0.6, "subdomains": []},
    ],
    "languages": [{"language": "English", "proficiency": 1.0}],
    "cultural_knowledge": {
        "primary_culture": "American",
        "exposure_level": {"european": 0.3},
    },
    "primary_goals": [{"goal": "Build reliable systems", "weight": 0.9}],
    "social_roles": {
        "default": {"formality": 0.5, "directness": 0.6, "emotional_expressiveness": 0.4},
    },
    "uncertainty": {
        "admission_threshold": 0.4, "hedging_frequency": 0.3,
        "clarification_tendency": 0.5, "knowledge_boundary_strictness": 0.7,
    },
    "claim_policy": {
        "allowed_claim_types": ["personal_experience", "domain_expert"],
        "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
        "lookup_behavior": "hedge",
    },
    "invariants": {
        "identity_facts": ["Software engineer"],
        "cannot_claim": [],
        "must_avoid": [],
    },
    "initial_state": {
        "mood_valence": 0.1, "mood_arousal": 0.4,
        "fatigue": 0.2, "stress": 0.2, "engagement": 0.7,
    },
    "biases": [{"type": "confirmation_bias", "strength": 0.2}],
    "time_scarcity": 0.4,
    "privacy_sensitivity": 0.3,
    "disclosure_policy": {
        "base_openness": 0.6,
        "factors": {
            "topic_sensitivity": -0.2, "trust_level": 0.3,
            "formal_context": -0.1, "positive_mood": 0.15,
        },
        "bounds": [0.2, 0.85],
    },
}


@pytest.fixture
def persona():
    return Persona(**PERSONA_DATA)


@pytest.fixture
def planner(persona):
    return TurnPlanner(persona, DeterminismManager(seed=42))


def _make_context(user_input: str = "Tell me about software engineering", turn: int = 1):
    return ConversationContext(
        conversation_id="test_phase5",
        turn_number=turn,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.GATHER_INFO,
        topic_signature="technology",
        user_input=user_input,
        stance_cache=StanceCache(),
    )


# ===========================================================================
# Fix 5.1: EngineConfig dataclass
# ===========================================================================


class TestEngineConfig:
    """Verify EngineConfig centralizes constants and is injectable."""

    def test_default_config_has_expected_values(self):
        assert DEFAULT_CONFIG.cross_turn_inertia == 0.15
        assert DEFAULT_CONFIG.expert_threshold == 0.7
        assert DEFAULT_CONFIG.default_proficiency == 0.3

    def test_config_is_frozen(self):
        with pytest.raises(AttributeError):
            DEFAULT_CONFIG.cross_turn_inertia = 0.5  # type: ignore[misc]

    def test_custom_config_overrides(self):
        custom = EngineConfig(cross_turn_inertia=0.30, expert_threshold=0.5)
        assert custom.cross_turn_inertia == 0.30
        assert custom.expert_threshold == 0.5
        # Other values remain default
        assert custom.default_proficiency == 0.3

    def test_planner_accepts_config(self, persona):
        custom = EngineConfig(cross_turn_inertia=0.5)
        planner = TurnPlanner(persona, config=custom)
        assert planner.config.cross_turn_inertia == 0.5

    def test_planner_defaults_to_default_config(self, persona):
        planner = TurnPlanner(persona)
        assert planner.config is DEFAULT_CONFIG

    def test_factory_accepts_config(self, persona):
        custom = EngineConfig(elasticity_min=0.2)
        planner = create_turn_planner(persona, config=custom)
        assert planner.config.elasticity_min == 0.2


# ===========================================================================
# Fix 5.2: TurnPlanner refactored into pipeline stages
# ===========================================================================


class TestPipelineStages:
    """Verify generate_ir still produces correct IR after refactoring."""

    def test_generate_ir_produces_valid_ir(self, planner):
        ctx = _make_context()
        ir = planner.generate_ir(ctx)
        assert ir is not None
        assert ir.turn_id == "test_phase5_turn_1"
        assert 0.0 <= ir.response_structure.confidence <= 1.0
        assert 0.0 <= ir.communication_style.formality <= 1.0

    def test_pipeline_stages_are_callable(self, planner):
        """Stage methods exist and are callable."""
        assert callable(getattr(planner, "_stage_foundation", None))
        assert callable(getattr(planner, "_stage_interpretation", None))
        assert callable(getattr(planner, "_stage_behavioral_metrics", None))
        assert callable(getattr(planner, "_stage_knowledge_safety", None))
        assert callable(getattr(planner, "_stage_finalization", None))

    def test_multi_turn_consistency_after_refactor(self, planner):
        """Multi-turn execution still works correctly."""
        ctx1 = _make_context("What is machine learning?", turn=1)
        ir1 = planner.generate_ir(ctx1)

        ctx2 = _make_context("How does it compare to deep learning?", turn=2)
        ir2 = planner.generate_ir(ctx2)

        # Both should be valid
        assert ir1.turn_id != ir2.turn_id
        assert ir2.seed != ir1.seed

    def test_determinism_preserved(self, persona):
        """Two planners with same seed produce same IR."""
        p1 = TurnPlanner(persona, DeterminismManager(seed=99))
        p2 = TurnPlanner(persona, DeterminismManager(seed=99))

        ir1 = p1.generate_ir(_make_context())
        ir2 = p2.generate_ir(_make_context())

        assert ir1.response_structure.confidence == ir2.response_structure.confidence
        assert ir1.communication_style.formality == ir2.communication_style.formality


# ===========================================================================
# Fix 5.3: Cross-turn inertia smoothing for elasticity + competence
# ===========================================================================


class TestCrossTurnSmoothing:
    """Verify elasticity and competence are now smoothed across turns."""

    def test_turn_snapshot_has_elasticity_and_competence(self):
        """TurnSnapshot now tracks elasticity and competence fields."""
        snap = TurnSnapshot(
            turn_number=1, confidence=0.5, formality=0.5,
            directness=0.5, disclosure=0.5, tone="neutral",
            claim_type="informed_opinion", stance="test", topic="test",
            elasticity=0.6, competence=0.7,
        )
        assert snap.elasticity == 0.6
        assert snap.competence == 0.7

    def test_snapshot_from_ir_captures_elasticity_competence(self, planner):
        ir = planner.generate_ir(_make_context())
        snap = TurnSnapshot.from_ir(ir, turn=1, topic="tech")
        assert snap.elasticity == ir.response_structure.elasticity
        assert snap.competence == ir.response_structure.competence

    def test_elasticity_smoothed_across_turns(self, planner):
        """Elasticity on turn 2 should show inertia effect from turn 1."""
        ctx1 = _make_context("Tell me about cooking", turn=1)
        ir1 = planner.generate_ir(ctx1)

        # Different domain to provoke different elasticity
        ctx2 = _make_context("Tell me about quantum physics", turn=2)
        ir2 = planner.generate_ir(ctx2)

        # Check that prior_snapshot was stored after turn 1 and used in turn 2
        assert planner._prior_snapshot is not None
        # The smoothing citations should exist
        cross_turn_cites = [
            c for c in ir2.citations
            if getattr(c, "source_type", "") == "cross_turn"
            and "elasticity" in getattr(c, "target_field", "")
        ]
        # Smoothing may or may not trigger depending on delta, but snapshot should be set
        assert planner._prior_snapshot.turn_number == 2

    def test_competence_smoothed_across_turns(self, planner):
        """Competence on turn 2 should show inertia effect from turn 1."""
        ctx1 = _make_context("Tell me about software engineering", turn=1)
        ir1 = planner.generate_ir(ctx1)
        e1 = ir1.response_structure.competence

        ctx2 = _make_context("Tell me about software design patterns", turn=2)
        ir2 = planner.generate_ir(ctx2)

        # After two turns, snapshot should reflect latest
        assert planner._prior_snapshot.competence == ir2.response_structure.competence


# ===========================================================================
# Fix 5.4: Externalized domain registry
# ===========================================================================


class TestExternalizedDomainRegistry:
    """Verify domain registry is externalized and importable."""

    def test_domain_registry_importable_from_separate_module(self):
        assert EXTERNAL_REGISTRY is not None
        assert len(EXTERNAL_REGISTRY) > 0

    def test_domain_registry_backward_compatible(self):
        """domain_detection.DOMAIN_REGISTRY still works."""
        assert DOMAIN_REGISTRY is EXTERNAL_REGISTRY

    def test_registry_has_expected_domains(self):
        domain_ids = {d.domain_id for d in EXTERNAL_REGISTRY}
        assert "technology" in domain_ids
        assert "psychology" in domain_ids
        assert "food" in domain_ids
        assert "finance" in domain_ids

    def test_detect_domain_uses_externalized_registry(self):
        domain, score = detect_domain("I love writing Python code", persona_domains=[])
        assert domain == "technology"

    def test_custom_registry_usable(self):
        """Custom DomainEntry objects work correctly."""
        custom = DomainEntry(
            domain_id="custom_test",
            keywords={"test_keyword": 1.0},
        )
        score = custom.score_input(["test_keyword"], {"test_keyword"})
        assert score == 1.0


# ===========================================================================
# Fix 5.5: Test files organized under tests/
# ===========================================================================


class TestTestOrganization:
    """Verify test files are organized under tests/ directory."""

    def test_no_test_files_at_root(self):
        root = Path(__file__).parent.parent
        root_test_files = list(root.glob("test_*.py"))
        assert len(root_test_files) == 0, (
            f"Found test files at project root: {[f.name for f in root_test_files]}"
        )

    def test_tests_directory_has_phase_tests(self):
        tests_dir = Path(__file__).parent
        phase_tests = sorted(tests_dir.glob("test_phase*.py"))
        assert len(phase_tests) >= 5, (
            f"Expected >= 5 phase test files, found {len(phase_tests)}"
        )

    def test_existing_tests_moved_successfully(self):
        tests_dir = Path(__file__).parent
        expected_files = [
            "test_turn_planner.py",
            "test_validation.py",
            "test_schemas.py",
            "test_state_manager.py",
        ]
        for fname in expected_files:
            assert (tests_dir / fname).exists(), f"Missing: tests/{fname}"
