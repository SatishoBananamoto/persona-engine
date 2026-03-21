"""
Tests for the provider benchmark module.

Validates benchmark execution, result aggregation, cost estimation,
IR consistency checks, and report formatting.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from persona_engine.benchmark import (
    DEFAULT_BENCHMARK_TURNS,
    PROVIDER_PRICING,
    BenchmarkReport,
    ProviderResult,
    TurnResult,
    _estimate_cost,
    run_benchmark,
    run_provider_benchmark,
)
from persona_engine.generation.llm_adapter import MockLLMAdapter
from persona_engine.schema.persona_schema import Persona


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chef_persona() -> Persona:
    import yaml
    with open("personas/chef.yaml") as f:
        data = yaml.safe_load(f)
    return Persona(**data)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimation:

    def test_anthropic_cost(self):
        # 1M input + 1M output at Anthropic pricing
        cost = _estimate_cost("anthropic", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_free_providers(self):
        assert _estimate_cost("ollama", 100_000, 50_000) == 0.0
        assert _estimate_cost("mock", 100_000, 50_000) == 0.0

    def test_unknown_provider_zero_cost(self):
        assert _estimate_cost("unknown_provider", 100_000, 50_000) == 0.0

    def test_gemini_cheaper_than_anthropic(self):
        tokens = 100_000
        gemini = _estimate_cost("gemini", tokens, tokens)
        anthropic = _estimate_cost("anthropic", tokens, tokens)
        assert gemini < anthropic

    def test_all_providers_have_pricing(self):
        expected = {"anthropic", "openai", "gemini", "mistral", "groq", "ollama", "mock", "template", "openai_compatible"}
        assert set(PROVIDER_PRICING.keys()) == expected


# ---------------------------------------------------------------------------
# Single provider benchmark
# ---------------------------------------------------------------------------


class TestRunProviderBenchmark:

    def test_basic_run(self, chef_persona: Persona):
        adapter = MockLLMAdapter()
        turns = [{"input": "Hello", "label": "greeting"}]
        result = run_provider_benchmark(chef_persona, adapter, "mock", turns)

        assert isinstance(result, ProviderResult)
        assert result.provider == "mock"
        assert result.model == "mock-llm"
        assert result.error is None
        assert len(result.turns) == 1
        assert result.turns[0].label == "greeting"
        assert result.validation_pass_rate > 0

    def test_multi_turn(self, chef_persona: Persona):
        adapter = MockLLMAdapter()
        turns = [
            {"input": "Hello", "label": "t1"},
            {"input": "How are you?", "label": "t2"},
            {"input": "Goodbye", "label": "t3"},
        ]
        result = run_provider_benchmark(chef_persona, adapter, "mock", turns)

        assert len(result.turns) == 3
        assert result.turns[0].turn_number == 1
        assert result.turns[2].turn_number == 3
        assert result.total_tokens > 0

    def test_default_benchmark_turns(self, chef_persona: Persona):
        adapter = MockLLMAdapter()
        result = run_provider_benchmark(
            chef_persona, adapter, "mock", DEFAULT_BENCHMARK_TURNS
        )

        assert len(result.turns) == 5
        assert result.validation_pass_rate > 0
        assert all(t.response_text for t in result.turns)

    def test_error_handling(self, chef_persona: Persona):
        adapter = MagicMock()
        adapter.get_model_name.return_value = "broken"
        adapter.generate.side_effect = RuntimeError("API down")

        turns = [{"input": "Hi", "label": "t1"}]
        result = run_provider_benchmark(chef_persona, adapter, "broken", turns)

        assert result.error is not None
        assert "RuntimeError" in result.error

    def test_latency_tracked(self, chef_persona: Persona):
        adapter = MockLLMAdapter()
        turns = [{"input": "Hello", "label": "t1"}]
        result = run_provider_benchmark(chef_persona, adapter, "mock", turns)

        # Mock is near-instant, but latency should be measured
        assert result.turns[0].latency_ms is not None
        assert result.turns[0].latency_ms >= 0

    def test_confidence_and_competence_populated(self, chef_persona: Persona):
        adapter = MockLLMAdapter()
        turns = [{"input": "What's the best knife for chopping?", "label": "domain"}]
        result = run_provider_benchmark(chef_persona, adapter, "mock", turns)

        assert 0 <= result.turns[0].confidence <= 1
        assert 0 <= result.turns[0].competence <= 1


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------


class TestRunBenchmark:

    def test_dry_run(self):
        report = run_benchmark(dry_run=True)

        assert isinstance(report, BenchmarkReport)
        assert report.turn_count == 5
        assert len(report.providers) == 5
        assert report.ir_consistent is True
        assert all(p.error is None for p in report.providers)

    def test_custom_persona(self):
        report = run_benchmark(
            persona_path="personas/physicist.yaml",
            dry_run=True,
        )
        assert len(report.providers) > 0
        assert report.turn_count == 5

    def test_custom_turns(self):
        custom_turns = [
            {"input": "Test question 1", "label": "q1"},
            {"input": "Test question 2", "label": "q2"},
        ]
        report = run_benchmark(dry_run=True, turns=custom_turns)
        assert report.turn_count == 2

    def test_specific_providers(self):
        report = run_benchmark(
            providers=["mock_a", "mock_b"],
            dry_run=True,
        )
        assert len(report.providers) == 2
        assert report.providers[0].provider == "mock_a"
        assert report.providers[1].provider == "mock_b"

    def test_ir_consistency_across_mock_providers(self):
        report = run_benchmark(dry_run=True)

        # All mock providers should produce identical IR
        assert report.ir_consistent is True

        # Verify numerically
        providers = [p for p in report.providers if not p.error]
        ref = providers[0]
        for other in providers[1:]:
            for i in range(len(ref.turns)):
                assert abs(ref.turns[i].confidence - other.turns[i].confidence) < 0.001
                assert abs(ref.turns[i].competence - other.turns[i].competence) < 0.001

    def test_unconfigured_provider_reports_error(self):
        report = run_benchmark(
            providers=["nonexistent_provider"],
            dry_run=False,
        )
        assert len(report.providers) == 1
        assert report.providers[0].error is not None
        assert "Setup failed" in report.providers[0].error


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestBenchmarkReport:

    def test_summary_table_format(self):
        report = run_benchmark(dry_run=True)
        table = report.summary_table()

        assert "PROVIDER BENCHMARK" in table
        assert "Latency" in table
        assert "Tokens" in table
        assert "Cost" in table
        assert "Pass%" in table
        assert "IR consistency" in table
        assert "PER-TURN LATENCY BREAKDOWN" in table

    def test_summary_table_shows_all_providers(self):
        report = run_benchmark(
            providers=["mock_a", "mock_b"],
            dry_run=True,
        )
        table = report.summary_table()
        assert "mock_a" in table
        assert "mock_b" in table

    def test_error_provider_in_table(self):
        report = run_benchmark(
            providers=["nonexistent_provider"],
            dry_run=False,
        )
        table = report.summary_table()
        assert "ERROR" in table

    def test_summary_with_custom_seed(self):
        report = run_benchmark(dry_run=True, seed=123)
        table = report.summary_table()
        assert "seed=123" in table


# ---------------------------------------------------------------------------
# Latency in GeneratedResponse
# ---------------------------------------------------------------------------


class TestLatencyTracking:

    def test_mock_adapter_has_latency(self, chef_persona: Persona):
        from persona_engine.engine import PersonaEngine
        from persona_engine.generation.llm_adapter import MockLLMAdapter

        engine = PersonaEngine(chef_persona, adapter=MockLLMAdapter(), seed=42)
        result = engine.chat("Hello!")

        # Mock adapter should still have latency measured
        assert result.response.latency_ms is not None
        assert result.response.latency_ms >= 0

    def test_template_adapter_no_latency(self, chef_persona: Persona):
        from persona_engine.engine import PersonaEngine

        engine = PersonaEngine(chef_persona, llm_provider="template", seed=42)
        result = engine.chat("Hello!")

        # Template adapter bypasses LLM call, so no latency
        assert result.response.latency_ms is None
