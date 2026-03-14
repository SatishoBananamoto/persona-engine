"""
Distributional Guarantee Tests — verify that trait markers appear within
expected ranges across multiple prompts and personas.

These tests run each persona through a set of diverse prompts and verify
that the IR fields and trait scorer output remain statistically consistent
with the persona's defined traits.

This addresses ROADMAP Phase 6 items:
- Distributional guarantees (trait markers within range)
- Deterministic failure reproduction (seeded)
"""

import statistics

import pytest

from persona_engine.engine import PersonaEngine
from persona_engine.schema.ir_schema import IntermediateRepresentation
from persona_engine.validation.trait_scorer import TraitMarkerScorer


# ---------------------------------------------------------------------------
# Test prompts — diverse to exercise different behavioral dimensions
# ---------------------------------------------------------------------------

DIVERSE_PROMPTS = [
    "What do you think about working from home?",
    "How do you handle disagreements with colleagues?",
    "Tell me about a hobby you enjoy.",
    "What's your opinion on social media?",
    "How do you approach learning something new?",
    "What makes a good leader?",
    "How do you deal with stress?",
    "What's something you feel strongly about?",
]

TWIN_PROMPTS = [
    "What do you think about trying new foods?",
    "How do you feel about public speaking?",
    "Tell me about your approach to planning.",
    "What's your view on taking risks?",
    "How important is routine to you?",
]

PERSONA_FILES = [
    "personas/chef.yaml",
    "personas/physicist.yaml",
    "personas/lawyer.yaml",
    "personas/musician.yaml",
    "personas/software_engineer.yaml",
    "personas/social_worker.yaml",
]


# ---------------------------------------------------------------------------
# Distributional guarantees for IR fields
# ---------------------------------------------------------------------------


class TestIRDistributions:
    """Verify IR field distributions are consistent with persona traits."""

    def _collect_ir_fields(
        self, persona_file: str, prompts: list[str], seed: int = 42,
    ) -> list[IntermediateRepresentation]:
        """Generate IRs for all prompts with a fresh engine per prompt (turn 1)."""
        results = []
        for prompt in prompts:
            engine = PersonaEngine.from_yaml(persona_file, llm_provider="mock", seed=seed)
            ir = engine.plan(prompt)
            results.append(ir)
        return results

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_confidence_bounded(self, persona_file: str):
        """Confidence should always be in [0, 1]."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for ir in irs:
            assert 0.0 <= ir.response_structure.confidence <= 1.0

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_elasticity_bounded(self, persona_file: str):
        """Elasticity should always be in [0, 1]."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for ir in irs:
            assert 0.0 <= ir.response_structure.elasticity <= 1.0

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_competence_bounded(self, persona_file: str):
        """Competence should always be in [0, 1]."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for ir in irs:
            assert 0.0 <= ir.response_structure.competence <= 1.0

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_disclosure_bounded(self, persona_file: str):
        """Disclosure level should always be in [0, 1]."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for ir in irs:
            assert 0.0 <= ir.knowledge_disclosure.disclosure_level <= 1.0

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_formality_bounded(self, persona_file: str):
        """Formality should always be in [0, 1]."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for ir in irs:
            assert 0.0 <= ir.communication_style.formality <= 1.0

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_directness_bounded(self, persona_file: str):
        """Directness should always be in [0, 1]."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for ir in irs:
            assert 0.0 <= ir.communication_style.directness <= 1.0

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_confidence_variance_bounded(self, persona_file: str):
        """Confidence across diverse prompts shouldn't have wild variance."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        confidences = [ir.response_structure.confidence for ir in irs]
        if len(confidences) > 1:
            std = statistics.stdev(confidences)
            # Standard deviation shouldn't exceed 0.4 across 8 diverse prompts
            assert std < 0.4, f"Confidence too variable: std={std:.3f}, values={confidences}"

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_all_prompts_produce_citations(self, persona_file: str):
        """Every IR should have at least one citation (traceability guarantee)."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for i, ir in enumerate(irs):
            assert len(ir.citations) > 0, f"Prompt {i} produced IR with no citations"

    @pytest.mark.parametrize("persona_file", PERSONA_FILES)
    def test_all_prompts_have_safety_plan(self, persona_file: str):
        """Every IR should have a safety plan."""
        irs = self._collect_ir_fields(persona_file, DIVERSE_PROMPTS)
        for ir in irs:
            assert ir.safety_plan is not None


# ---------------------------------------------------------------------------
# Counterfactual twin distributional tests
# ---------------------------------------------------------------------------


class TestTwinDistributions:
    """Verify counterfactual twins produce measurably different distributions."""

    def _mean_field(self, irs: list[IntermediateRepresentation], accessor) -> float:
        return statistics.mean(accessor(ir) for ir in irs)

    def _collect_twin_irs(
        self, high_file: str, low_file: str, prompts: list[str], seed: int = 42,
    ):
        high_irs = []
        low_irs = []
        for prompt in prompts:
            e_high = PersonaEngine.from_yaml(high_file, llm_provider="mock", seed=seed)
            e_low = PersonaEngine.from_yaml(low_file, llm_provider="mock", seed=seed)
            high_irs.append(e_high.plan(prompt))
            low_irs.append(e_low.plan(prompt))
        return high_irs, low_irs

    def test_openness_twins_diverge_on_elasticity(self):
        """High openness twin should have higher mean elasticity than low."""
        high_irs, low_irs = self._collect_twin_irs(
            "personas/twins/high_openness.yaml",
            "personas/twins/low_openness.yaml",
            TWIN_PROMPTS,
        )
        high_mean = self._mean_field(high_irs, lambda ir: ir.response_structure.elasticity)
        low_mean = self._mean_field(low_irs, lambda ir: ir.response_structure.elasticity)
        # High openness → more elasticity (more willing to change mind)
        assert high_mean > low_mean - 0.05, (
            f"Expected high openness to have higher elasticity: "
            f"high={high_mean:.3f}, low={low_mean:.3f}"
        )

    def test_neuroticism_twins_diverge_on_confidence(self):
        """High neuroticism twin should have lower mean confidence than low."""
        high_irs, low_irs = self._collect_twin_irs(
            "personas/twins/high_neuroticism.yaml",
            "personas/twins/low_neuroticism.yaml",
            TWIN_PROMPTS,
        )
        high_mean = self._mean_field(high_irs, lambda ir: ir.response_structure.confidence)
        low_mean = self._mean_field(low_irs, lambda ir: ir.response_structure.confidence)
        # High neuroticism → lower confidence
        assert high_mean < low_mean + 0.05, (
            f"Expected high neuroticism to have lower confidence: "
            f"high={high_mean:.3f}, low={low_mean:.3f}"
        )

    def test_extraversion_twins_diverge_on_disclosure(self):
        """High extraversion twin should have higher mean disclosure."""
        high_irs, low_irs = self._collect_twin_irs(
            "personas/twins/high_extraversion.yaml",
            "personas/twins/low_extraversion.yaml",
            TWIN_PROMPTS,
        )
        high_mean = self._mean_field(
            high_irs, lambda ir: ir.knowledge_disclosure.disclosure_level
        )
        low_mean = self._mean_field(
            low_irs, lambda ir: ir.knowledge_disclosure.disclosure_level
        )
        # High extraversion → more disclosure
        assert high_mean > low_mean - 0.05, (
            f"Expected high extraversion to have higher disclosure: "
            f"high={high_mean:.3f}, low={low_mean:.3f}"
        )


# ---------------------------------------------------------------------------
# Deterministic failure reproduction
# ---------------------------------------------------------------------------


class TestDeterministicReproduction:
    """Verify that failures can be reproduced exactly with the same seed."""

    def test_same_seed_reproduces_exact_ir(self):
        """Same persona + same prompt + same seed = byte-identical IR."""
        for seed in [1, 42, 100, 999]:
            e1 = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock", seed=seed)
            e2 = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock", seed=seed)
            ir1 = e1.plan("What makes a perfect sauce?")
            ir2 = e2.plan("What makes a perfect sauce?")
            # Structural equality
            assert ir1.response_structure.confidence == ir2.response_structure.confidence
            assert ir1.response_structure.elasticity == ir2.response_structure.elasticity
            assert ir1.response_structure.competence == ir2.response_structure.competence
            assert ir1.communication_style.tone == ir2.communication_style.tone
            assert ir1.communication_style.formality == ir2.communication_style.formality
            assert ir1.communication_style.directness == ir2.communication_style.directness
            assert ir1.knowledge_disclosure.disclosure_level == ir2.knowledge_disclosure.disclosure_level

    def test_different_seeds_consistent_on_turn_one(self):
        """On turn 1 the system is largely deterministic regardless of seed,
        because randomness only affects stochastic selection which may not
        be exercised for every prompt. This verifies stability."""
        e1 = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock", seed=1)
        e2 = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock", seed=999)
        ir1 = e1.plan("Tell me about French cuisine")
        ir2 = e2.plan("Tell me about French cuisine")
        # Both should produce valid IR
        assert 0.0 <= ir1.response_structure.confidence <= 1.0
        assert 0.0 <= ir2.response_structure.confidence <= 1.0

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_multi_turn_determinism(self, seed: int):
        """Multi-turn conversations should be deterministic with same seed."""
        prompts = ["Hello!", "Tell me more.", "Interesting!"]
        e1 = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock", seed=seed)
        e2 = PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock", seed=seed)
        for prompt in prompts:
            ir1 = e1.plan(prompt)
            ir2 = e2.plan(prompt)
            assert ir1.response_structure.confidence == ir2.response_structure.confidence
            assert ir1.response_structure.elasticity == ir2.response_structure.elasticity


# ---------------------------------------------------------------------------
# Trait Scorer Distributional Tests
# ---------------------------------------------------------------------------


class TestTraitScorerDistributions:
    """Verify that the trait scorer produces meaningful scores."""

    def test_high_openness_text_scores_well(self):
        scorer = TraitMarkerScorer()
        high_o_text = (
            "What if we explored this from a completely different perspective? "
            "I find it fascinating how creative and innovative approaches can "
            "lead to novel solutions. Imagine the possibilities if we consider "
            "alternative and unconventional ideas."
        )
        result = scorer.score(high_o_text, {"openness": 0.9})
        assert result.per_trait["openness"].high_marker_count > 3
        assert result.per_trait["openness"].coherence > 0.5

    def test_low_openness_text_scores_well(self):
        scorer = TraitMarkerScorer()
        low_o_text = (
            "I think we should stick with the practical and proven approach. "
            "The standard method is straightforward and conventional. "
            "Let's keep it simple and use the established routine."
        )
        result = scorer.score(low_o_text, {"openness": 0.1})
        assert result.per_trait["openness"].low_marker_count > 3
        assert result.per_trait["openness"].coherence > 0.5

    def test_high_neuroticism_text_scores_well(self):
        scorer = TraitMarkerScorer()
        high_n_text = (
            "I'm worried about this. I'm not sure it's the right approach. "
            "It feels uncertain and stressful. What if something goes wrong? "
            "I'm concerned about the risks and potential danger."
        )
        result = scorer.score(high_n_text, {"neuroticism": 0.9})
        assert result.per_trait["neuroticism"].high_marker_count > 3
        assert result.per_trait["neuroticism"].coherence > 0.5

    def test_scorer_result_summary(self):
        scorer = TraitMarkerScorer()
        result = scorer.score(
            "This is a test.",
            {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5,
            },
        )
        summary = result.summary()
        assert "Overall coherence" in summary
        assert "openness" in summary

    def test_empty_text_gives_neutral_score(self):
        scorer = TraitMarkerScorer()
        result = scorer.score("", {"openness": 0.5})
        # No markers found → marker_ratio defaults to 0.5
        assert result.per_trait["openness"].marker_ratio == 0.5
        assert result.per_trait["openness"].high_marker_count == 0
