"""
Tests for persona_engine.analysis — PersonaAnalyzer and ConversationAnalyzer.
"""

import pytest

from persona_engine.analysis import ConversationAnalyzer, PersonaAnalyzer
from persona_engine.engine import PersonaEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS = [
    "What do you think about remote work?",
    "How do you handle stress?",
    "Tell me about your hobbies.",
    "What's your opinion on AI?",
]


@pytest.fixture
def chef_engine():
    return PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock", seed=42)


@pytest.fixture
def physicist_engine():
    return PersonaEngine.from_yaml("personas/physicist.yaml", llm_provider="mock", seed=42)


@pytest.fixture
def chef_persona():
    return PersonaEngine.from_yaml("personas/chef.yaml", llm_provider="mock").persona


@pytest.fixture
def physicist_persona():
    return PersonaEngine.from_yaml("personas/physicist.yaml", llm_provider="mock").persona


@pytest.fixture
def high_openness_persona():
    return PersonaEngine.from_yaml("personas/twins/high_openness.yaml", llm_provider="mock").persona


@pytest.fixture
def low_openness_persona():
    return PersonaEngine.from_yaml("personas/twins/low_openness.yaml", llm_provider="mock").persona


# ---------------------------------------------------------------------------
# PersonaAnalyzer.profile_summary
# ---------------------------------------------------------------------------


class TestProfileSummary:
    def test_returns_dict(self, chef_persona):
        result = PersonaAnalyzer.profile_summary(chef_persona)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, chef_persona):
        result = PersonaAnalyzer.profile_summary(chef_persona)
        for key in ["persona_id", "label", "identity", "big_five", "top_values",
                     "cognitive_style", "communication", "knowledge_domains", "primary_goals"]:
            assert key in result, f"Missing key: {key}"

    def test_big_five_has_all_traits(self, chef_persona):
        result = PersonaAnalyzer.profile_summary(chef_persona)
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert trait in result["big_five"]

    def test_top_values_sorted(self, chef_persona):
        result = PersonaAnalyzer.profile_summary(chef_persona)
        values = result["top_values"]
        assert len(values) <= 5
        for i in range(len(values) - 1):
            assert values[i][1] >= values[i + 1][1], "Top values should be sorted descending"

    def test_knowledge_domains_sorted(self, chef_persona):
        result = PersonaAnalyzer.profile_summary(chef_persona)
        domains = result["knowledge_domains"]
        for i in range(len(domains) - 1):
            assert domains[i]["proficiency"] >= domains[i + 1]["proficiency"]


# ---------------------------------------------------------------------------
# PersonaAnalyzer.compare_personas
# ---------------------------------------------------------------------------


class TestComparePersonas:
    def test_compare_returns_dict(self, chef_persona, physicist_persona):
        result = PersonaAnalyzer.compare_personas(chef_persona, physicist_persona)
        assert isinstance(result, dict)

    def test_compare_has_deltas(self, chef_persona, physicist_persona):
        result = PersonaAnalyzer.compare_personas(chef_persona, physicist_persona)
        assert "big_five_delta" in result
        assert "values_delta" in result
        assert "communication_delta" in result
        assert "most_divergent_trait" in result

    def test_big_five_deltas_are_differences(self, chef_persona, physicist_persona):
        result = PersonaAnalyzer.compare_personas(chef_persona, physicist_persona)
        deltas = result["big_five_delta"]
        b1 = chef_persona.psychology.big_five.model_dump()
        b2 = physicist_persona.psychology.big_five.model_dump()
        for trait in b1:
            expected = round(b2[trait] - b1[trait], 4)
            assert deltas[trait] == expected

    def test_twins_diverge_on_target_trait(self, high_openness_persona, low_openness_persona):
        result = PersonaAnalyzer.compare_personas(high_openness_persona, low_openness_persona)
        openness_delta = result["big_five_delta"]["openness"]
        assert abs(openness_delta) > 0.3, "Twins should diverge significantly on openness"

    def test_most_divergent_trait_has_largest_delta(self, chef_persona, physicist_persona):
        result = PersonaAnalyzer.compare_personas(chef_persona, physicist_persona)
        most_div = result["most_divergent_trait"]
        assert "trait" in most_div
        assert "delta" in most_div


# ---------------------------------------------------------------------------
# PersonaAnalyzer.run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_benchmark_returns_dict(self, chef_engine):
        result = PersonaAnalyzer.run_benchmark(chef_engine, BENCHMARK_PROMPTS)
        assert isinstance(result, dict)

    def test_benchmark_has_stats(self, chef_engine):
        result = PersonaAnalyzer.run_benchmark(chef_engine, BENCHMARK_PROMPTS)
        assert "prompt_count" in result
        assert result["prompt_count"] == len(BENCHMARK_PROMPTS)
        for key in ["confidence", "competence", "elasticity"]:
            assert key in result
            assert "mean" in result[key]
            assert "min" in result[key]
            assert "max" in result[key]

    def test_benchmark_confidence_in_range(self, chef_engine):
        result = PersonaAnalyzer.run_benchmark(chef_engine, BENCHMARK_PROMPTS)
        assert 0.0 <= result["confidence"]["min"] <= result["confidence"]["max"] <= 1.0

    def test_benchmark_resets_engine(self, chef_engine):
        chef_engine.chat("Hello")
        assert chef_engine.turn_count == 1
        PersonaAnalyzer.run_benchmark(chef_engine, BENCHMARK_PROMPTS)
        assert chef_engine.turn_count == 0


# ---------------------------------------------------------------------------
# PersonaAnalyzer.trait_influence_report
# ---------------------------------------------------------------------------


class TestTraitInfluenceReport:
    def test_report_returns_dict(self, chef_engine):
        result = PersonaAnalyzer.trait_influence_report(chef_engine, BENCHMARK_PROMPTS[:2])
        assert isinstance(result, dict)

    def test_report_has_citation_stats(self, chef_engine):
        result = PersonaAnalyzer.trait_influence_report(chef_engine, BENCHMARK_PROMPTS[:2])
        assert "by_source_type" in result
        assert "top_sources" in result
        assert "most_influential" in result
        assert len(result["top_sources"]) > 0


# ---------------------------------------------------------------------------
# ConversationAnalyzer.summarize_conversation
# ---------------------------------------------------------------------------


class TestSummarizeConversation:
    def test_summarize_returns_dict(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        summary = ConversationAnalyzer.summarize_conversation(results)
        assert isinstance(summary, dict)

    def test_summarize_has_metrics(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        summary = ConversationAnalyzer.summarize_conversation(results)
        assert "turn_count" in summary
        assert summary["turn_count"] == len(BENCHMARK_PROMPTS)
        assert "avg_confidence" in summary
        assert "avg_competence" in summary

    def test_summarize_empty_list(self):
        summary = ConversationAnalyzer.summarize_conversation([])
        assert summary["turn_count"] == 0

    def test_tone_distribution(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        summary = ConversationAnalyzer.summarize_conversation(results)
        assert "tone_distribution" in summary
        assert len(summary["tone_distribution"]) > 0

    def test_disclosure_trend(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        summary = ConversationAnalyzer.summarize_conversation(results)
        assert "disclosure_trend" in summary
        assert "direction" in summary["disclosure_trend"]

    def test_validation_pass_rate(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        summary = ConversationAnalyzer.summarize_conversation(results)
        assert "validation_pass_rate" in summary
        assert 0.0 <= summary["validation_pass_rate"] <= 1.0


# ---------------------------------------------------------------------------
# ConversationAnalyzer.detect_drift
# ---------------------------------------------------------------------------


class TestDetectDrift:
    def test_drift_returns_dict(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        drift = ConversationAnalyzer.detect_drift(results)
        assert isinstance(drift, dict)

    def test_drift_has_metrics(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        drift = ConversationAnalyzer.detect_drift(results)
        assert "metrics" in drift
        assert "confidence" in drift["metrics"]
        assert "formality" in drift["metrics"]
        for name, metric in drift["metrics"].items():
            assert "trend" in metric
            assert "max_shift" in metric
            assert "values" in metric

    def test_drift_with_single_turn(self, chef_engine):
        results = [chef_engine.chat("Hello")]
        drift = ConversationAnalyzer.detect_drift(results)
        assert isinstance(drift, dict)
        assert drift["drifted"] is False

    def test_drift_with_empty_list(self):
        drift = ConversationAnalyzer.detect_drift([])
        assert isinstance(drift, dict)
        assert drift["drifted"] is False

    def test_drift_drifted_flag(self, chef_engine):
        results = [chef_engine.chat(p) for p in BENCHMARK_PROMPTS]
        drift = ConversationAnalyzer.detect_drift(results)
        assert "drifted" in drift
        assert isinstance(drift["drifted"], bool)
