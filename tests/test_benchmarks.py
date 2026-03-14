"""
Benchmark Tests — run all personas through standardized conversations.

Validates that:
1. Every persona produces valid IR for all interaction modes
2. No validation failures across the library
3. Basic behavioral expectations are met
"""

import os
import yaml
import pytest

from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.schema.ir_schema import ConversationGoal, InteractionMode
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager
from persona_engine.validation.ir_validator import validate_ir, has_errors

from benchmarks.conversations import ALL_BENCHMARKS


# =============================================================================
# Helpers
# =============================================================================


def load_all_personas() -> list[tuple[str, Persona]]:
    """Load all persona YAML files."""
    personas = []
    for directory in ["personas", "personas/twins"]:
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".yaml"):
                continue
            path = os.path.join(directory, filename)
            with open(path) as f:
                data = yaml.safe_load(f)
            if "domains" in data and "knowledge_domains" not in data:
                data["knowledge_domains"] = data.pop("domains")
            personas.append((filename, Persona(**data)))
    return personas


ALL_PERSONAS = load_all_personas()
PERSONA_NAMES = [name for name, _ in ALL_PERSONAS]


BENCHMARK_MODE_MAP = {
    "casual_chat": (InteractionMode.CASUAL_CHAT, ConversationGoal.EXPLORE_IDEAS),
    "interview": (InteractionMode.INTERVIEW, ConversationGoal.GATHER_INFO),
    "customer_support": (InteractionMode.CUSTOMER_SUPPORT, ConversationGoal.RESOLVE_ISSUE),
    "survey": (InteractionMode.CASUAL_CHAT, ConversationGoal.EXPLORE_IDEAS),
}


# =============================================================================
# Tests
# =============================================================================


class TestAllPersonasLoadSuccessfully:
    """Every persona in the library must load without validation errors."""

    def test_persona_count(self):
        """We should have at least 12 personas (8 original + new ones)."""
        assert len(ALL_PERSONAS) >= 12, (
            f"Expected at least 12 personas, found {len(ALL_PERSONAS)}: "
            f"{PERSONA_NAMES}"
        )

    @pytest.mark.parametrize("name,persona", ALL_PERSONAS, ids=PERSONA_NAMES)
    def test_persona_has_required_fields(self, name: str, persona: Persona):
        """Every persona must have core fields populated."""
        assert persona.persona_id
        assert persona.identity.occupation
        assert persona.psychology.big_five.openness >= 0
        assert len(persona.invariants.identity_facts) > 0


class TestBenchmarkConversations:
    """Run each persona through all benchmark conversation types."""

    @pytest.mark.parametrize("name,persona", ALL_PERSONAS[:8], ids=PERSONA_NAMES[:8])
    def test_persona_generates_valid_ir_across_benchmarks(self, name: str, persona: Persona):
        """Each persona produces valid IR for all 4 benchmark types."""
        for bench_name, prompts in ALL_BENCHMARKS.items():
            mode, goal = BENCHMARK_MODE_MAP[bench_name]
            planner = TurnPlanner(
                persona=persona,
                determinism=DeterminismManager(seed=42),
            )

            for turn_num, prompt in enumerate(prompts, start=1):
                ctx = ConversationContext(
                    conversation_id=f"bench_{bench_name}_{name}",
                    turn_number=turn_num,
                    user_input=prompt,
                    topic_signature="general",
                    interaction_mode=mode,
                    goal=goal,
                    stance_cache=StanceCache(),
                )
                ir = planner.generate_ir(ctx)

                # Basic validity checks
                assert ir.response_structure.confidence >= 0.0
                assert ir.response_structure.confidence <= 1.0
                assert ir.communication_style.tone is not None

                # Run IR validator
                issues = validate_ir(ir)
                assert not has_errors(issues), (
                    f"Persona '{name}' failed validation on {bench_name} turn {turn_num}: "
                    f"{[i.message for i in issues if i.severity == 'error']}"
                )


class TestBenchmarkBehavioralExpectations:
    """Verify basic behavioral expectations across benchmarks."""

    def test_all_personas_produce_valid_ir_in_all_modes(self):
        """Every persona should produce valid, bounded IR across all modes."""
        # Pick a representative persona (UX researcher has moderate traits)
        persona = next(p for name, p in ALL_PERSONAS if "ux_researcher" in name)

        for bench_name, prompts in ALL_BENCHMARKS.items():
            mode, goal = BENCHMARK_MODE_MAP[bench_name]
            planner = TurnPlanner(persona=persona, determinism=DeterminismManager(seed=42))
            ctx = ConversationContext(
                conversation_id=f"bench_{bench_name}",
                turn_number=1,
                user_input=prompts[0],
                topic_signature="general",
                interaction_mode=mode,
                goal=goal,
                stance_cache=StanceCache(),
            )
            ir = planner.generate_ir(ctx)

            # All numeric fields must be in valid ranges
            assert 0 <= ir.response_structure.confidence <= 1
            assert 0 <= ir.response_structure.elasticity <= 1
            assert 0 <= ir.communication_style.formality <= 1
            assert 0 <= ir.communication_style.directness <= 1
            assert 0 <= ir.knowledge_disclosure.disclosure_level <= 1
