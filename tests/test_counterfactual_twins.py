"""
Counterfactual Twin Tests — proving traits drive behavior.

For each Big Five dimension, we load a high/low twin pair (identical except
for one trait) and verify that changing that trait produces measurably
different IR output on the same prompt.

If twins produce identical IR, something is wrong with trait propagation.
"""

import yaml
import pytest

from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner
from persona_engine.memory.stance_cache import StanceCache
from persona_engine.schema.ir_schema import ConversationGoal, InteractionMode
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


# =============================================================================
# Helpers
# =============================================================================


def load_twin(name: str) -> Persona:
    path = f"personas/twins/{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def make_context(
    user_input: str,
    topic: str = "general",
    domain: str = "general",
    turn: int = 1,
) -> ConversationContext:
    return ConversationContext(
        conversation_id="twin_test",
        turn_number=turn,
        user_input=user_input,
        topic_signature=topic,
        interaction_mode=InteractionMode.CASUAL_CHAT,
        goal=ConversationGoal.EXPLORE_IDEAS,
        stance_cache=StanceCache(),
        domain=domain,
    )


def generate_ir_for_twin(twin_name: str, user_input: str, seed: int = 42):
    persona = load_twin(twin_name)
    planner = TurnPlanner(
        persona=persona,
        determinism=DeterminismManager(seed=seed),
    )
    ctx = make_context(user_input)
    return planner.generate_ir(ctx)


# =============================================================================
# Prompts
# =============================================================================

PROMPTS = [
    "What do you think about trying new approaches at work?",
    "How would you handle a disagreement with a colleague?",
    "Tell me about something you're passionate about.",
    "What's your take on taking risks in your career?",
    "How do you feel about strict deadlines?",
]


# =============================================================================
# Twin Pair Tests
# =============================================================================


class TestOpennessTwins:
    """High vs low openness should differ in elasticity."""

    def test_elasticity_differs(self):
        ir_high = generate_ir_for_twin("high_openness", PROMPTS[0])
        ir_low = generate_ir_for_twin("low_openness", PROMPTS[0])

        assert ir_high.response_structure.elasticity != ir_low.response_structure.elasticity, (
            f"Openness twins should differ in elasticity: "
            f"high={ir_high.response_structure.elasticity:.3f}, "
            f"low={ir_low.response_structure.elasticity:.3f}"
        )
        # High openness → higher elasticity (more open to changing views)
        assert ir_high.response_structure.elasticity > ir_low.response_structure.elasticity

    def test_ir_not_identical(self):
        ir_high = generate_ir_for_twin("high_openness", PROMPTS[1])
        ir_low = generate_ir_for_twin("low_openness", PROMPTS[1])
        # At least one metric must differ
        assert (
            ir_high.response_structure.elasticity != ir_low.response_structure.elasticity
            or ir_high.response_structure.confidence != ir_low.response_structure.confidence
            or ir_high.communication_style.tone != ir_low.communication_style.tone
        )


class TestExtraversionTwins:
    """High vs low extraversion should differ in disclosure."""

    def test_disclosure_differs(self):
        ir_high = generate_ir_for_twin("high_extraversion", PROMPTS[2])
        ir_low = generate_ir_for_twin("low_extraversion", PROMPTS[2])

        assert ir_high.knowledge_disclosure.disclosure_level != ir_low.knowledge_disclosure.disclosure_level, (
            f"Extraversion twins should differ in disclosure: "
            f"high={ir_high.knowledge_disclosure.disclosure_level:.3f}, "
            f"low={ir_low.knowledge_disclosure.disclosure_level:.3f}"
        )
        # High extraversion → higher self-disclosure
        assert ir_high.knowledge_disclosure.disclosure_level > ir_low.knowledge_disclosure.disclosure_level


class TestNeuroticismTwins:
    """High vs low neuroticism should differ in confidence."""

    def test_confidence_differs(self):
        ir_high = generate_ir_for_twin("high_neuroticism", PROMPTS[3])
        ir_low = generate_ir_for_twin("low_neuroticism", PROMPTS[3])

        assert ir_high.response_structure.confidence != ir_low.response_structure.confidence, (
            f"Neuroticism twins should differ in confidence: "
            f"high={ir_high.response_structure.confidence:.3f}, "
            f"low={ir_low.response_structure.confidence:.3f}"
        )
        # High neuroticism → lower confidence
        assert ir_high.response_structure.confidence < ir_low.response_structure.confidence


class TestConscientiousnessTwins:
    """High vs low conscientiousness should differ in verbosity or confidence."""

    def test_behavioral_differs(self):
        ir_high = generate_ir_for_twin("high_conscientiousness", PROMPTS[4])
        ir_low = generate_ir_for_twin("low_conscientiousness", PROMPTS[4])

        # Conscientiousness affects confidence and verbosity
        differs = (
            ir_high.response_structure.confidence != ir_low.response_structure.confidence
            or ir_high.communication_style.verbosity != ir_low.communication_style.verbosity
        )
        assert differs, (
            f"Conscientiousness twins should differ: "
            f"conf_high={ir_high.response_structure.confidence:.3f}, "
            f"conf_low={ir_low.response_structure.confidence:.3f}, "
            f"verb_high={ir_high.communication_style.verbosity}, "
            f"verb_low={ir_low.communication_style.verbosity}"
        )


class TestAgreeablenessTwins:
    """High vs low agreeableness should differ in directness or tone."""

    def test_style_differs(self):
        ir_high = generate_ir_for_twin("high_agreeableness", PROMPTS[1])
        ir_low = generate_ir_for_twin("low_agreeableness", PROMPTS[1])

        # Agreeableness affects directness, tone, and disclosure
        differs = (
            ir_high.communication_style.directness != ir_low.communication_style.directness
            or ir_high.communication_style.tone != ir_low.communication_style.tone
            or ir_high.knowledge_disclosure.disclosure_level != ir_low.knowledge_disclosure.disclosure_level
        )
        assert differs, (
            f"Agreeableness twins should differ in style: "
            f"dir_high={ir_high.communication_style.directness:.3f}, "
            f"dir_low={ir_low.communication_style.directness:.3f}"
        )


class TestAllTwinsProduceDifferentIR:
    """Meta-test: every twin pair must produce different IR on the same prompt."""

    @pytest.mark.parametrize("trait", [
        "openness", "extraversion", "neuroticism",
        "conscientiousness", "agreeableness",
    ])
    def test_twin_pair_differs(self, trait: str):
        ir_high = generate_ir_for_twin(f"high_{trait}", PROMPTS[0])
        ir_low = generate_ir_for_twin(f"low_{trait}", PROMPTS[0])

        # Collect all numeric fields
        diffs = []
        for field_name in ["confidence", "elasticity", "competence"]:
            h = getattr(ir_high.response_structure, field_name)
            l = getattr(ir_low.response_structure, field_name)
            if h != l:
                diffs.append(f"{field_name}: high={h:.3f}, low={l:.3f}")

        for field_name in ["formality", "directness"]:
            h = getattr(ir_high.communication_style, field_name)
            l = getattr(ir_low.communication_style, field_name)
            if h != l:
                diffs.append(f"{field_name}: high={h:.3f}, low={l:.3f}")

        if ir_high.communication_style.tone != ir_low.communication_style.tone:
            diffs.append(f"tone: high={ir_high.communication_style.tone.value}, low={ir_low.communication_style.tone.value}")

        if ir_high.knowledge_disclosure.disclosure_level != ir_low.knowledge_disclosure.disclosure_level:
            diffs.append(f"disclosure: high={ir_high.knowledge_disclosure.disclosure_level:.3f}, low={ir_low.knowledge_disclosure.disclosure_level:.3f}")

        assert len(diffs) > 0, (
            f"Trait '{trait}' twins produced identical IR! "
            f"This means changing {trait} has no behavioral effect."
        )
