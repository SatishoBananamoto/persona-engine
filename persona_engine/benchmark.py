"""
Provider Benchmark — compare LLM providers on the same persona conversations.

Runs identical multi-turn conversations across providers and collects:
- Latency (per-turn and aggregate)
- Estimated token usage
- Cost estimates (based on published pricing)
- IR consistency (same persona+seed should produce identical IR regardless of provider)
- Response quality signals (validation pass rate, constraint violations)

Usage:
    # Compare all configured providers on the chef persona
    python -m persona_engine.benchmark

    # Compare specific providers
    python -m persona_engine.benchmark --providers anthropic openai gemini

    # Use a different persona
    python -m persona_engine.benchmark --persona personas/physicist.yaml

    # Dry run with mock adapters (no API keys needed)
    python -m persona_engine.benchmark --dry-run
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from persona_engine.engine import ChatResult, PersonaEngine
from persona_engine.generation.llm_adapter import (
    BaseLLMAdapter,
    MockLLMAdapter,
    create_adapter,
)
from persona_engine.schema.persona_schema import Persona

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost data (USD per 1M tokens, as of early 2025)
# ---------------------------------------------------------------------------

PROVIDER_PRICING: dict[str, dict[str, float]] = {
    # provider -> {input_per_1m, output_per_1m}
    "anthropic": {"input_per_1m": 3.00, "output_per_1m": 15.00},       # Claude Sonnet 4
    "openai": {"input_per_1m": 0.15, "output_per_1m": 0.60},           # GPT-4o-mini
    "gemini": {"input_per_1m": 0.10, "output_per_1m": 0.40},           # Gemini 2.0 Flash
    "mistral": {"input_per_1m": 0.10, "output_per_1m": 0.30},          # Mistral Small
    "groq": {"input_per_1m": 0.59, "output_per_1m": 0.79},             # Llama 3.3 70B
    "ollama": {"input_per_1m": 0.00, "output_per_1m": 0.00},           # Local — free
    "openai_compatible": {"input_per_1m": 0.00, "output_per_1m": 0.00},  # Depends on provider
    "mock": {"input_per_1m": 0.00, "output_per_1m": 0.00},
    "template": {"input_per_1m": 0.00, "output_per_1m": 0.00},
}


# ---------------------------------------------------------------------------
# Benchmark prompts — diverse conversation types
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK_TURNS: list[dict[str, str]] = [
    {
        "input": "What's the most important skill in your field?",
        "label": "expertise_question",
    },
    {
        "input": "I completely disagree with that. Can you defend your position?",
        "label": "challenge",
    },
    {
        "input": "Tell me something personal about yourself related to this.",
        "label": "personal_disclosure",
    },
    {
        "input": "Can you explain quantum entanglement to me?",
        "label": "out_of_domain",
    },
    {
        "input": "Thanks, that was really helpful!",
        "label": "rapport_building",
    },
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Results from a single conversation turn."""

    turn_number: int
    label: str
    user_input: str
    response_text: str
    latency_ms: float | None
    estimated_tokens: int
    validation_passed: bool
    violation_count: int
    confidence: float
    competence: float


@dataclass
class ProviderResult:
    """Aggregated results for one provider across all turns."""

    provider: str
    model: str
    turns: list[TurnResult]
    total_latency_ms: float
    avg_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    validation_pass_rate: float
    total_violations: int
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Complete benchmark comparison across providers."""

    persona_name: str
    seed: int
    turn_count: int
    providers: list[ProviderResult]
    ir_consistent: bool  # True if IR is identical across all providers

    def summary_table(self) -> str:
        """Format results as a readable comparison table."""
        lines = []
        lines.append(f"\n{'=' * 90}")
        lines.append(f"  PROVIDER BENCHMARK — {self.persona_name} ({self.turn_count} turns, seed={self.seed})")
        lines.append(f"{'=' * 90}")

        # Header
        lines.append(
            f"  {'Provider':<22} {'Model':<28} {'Latency':>10} {'Tokens':>8} "
            f"{'Cost':>10} {'Pass%':>7} {'Violations':>10}"
        )
        lines.append(f"  {'-' * 22} {'-' * 28} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 7} {'-' * 10}")

        for p in self.providers:
            if p.error:
                lines.append(f"  {p.provider:<22} {'ERROR: ' + p.error:<28}")
                continue

            latency_str = f"{p.avg_latency_ms:.0f}ms" if p.avg_latency_ms > 0 else "N/A"
            cost_str = f"${p.estimated_cost_usd:.6f}" if p.estimated_cost_usd > 0 else "free"

            lines.append(
                f"  {p.provider:<22} {p.model:<28} {latency_str:>10} {p.total_tokens:>8} "
                f"{cost_str:>10} {p.validation_pass_rate:>6.0%} {p.total_violations:>10}"
            )

        lines.append(f"\n  IR consistency: {'PASS — identical IR across all providers' if self.ir_consistent else 'DIFFER — IR varies (unexpected)'}")
        lines.append(f"{'=' * 90}\n")

        # Per-turn breakdown
        lines.append(f"  PER-TURN LATENCY BREAKDOWN")
        lines.append(f"  {'-' * 88}")
        header_parts = [f"  {'Turn':<25}"]
        for p in self.providers:
            if not p.error:
                header_parts.append(f"{p.provider:>14}")
        lines.append("".join(header_parts))
        lines.append(f"  {'-' * 88}")

        for i in range(self.turn_count):
            label = self.providers[0].turns[i].label if not self.providers[0].error else f"turn_{i+1}"
            row_parts = [f"  {label:<25}"]
            for p in self.providers:
                if p.error:
                    continue
                t = p.turns[i]
                val = f"{t.latency_ms:.0f}ms" if t.latency_ms and t.latency_ms > 0 else "N/A"
                row_parts.append(f"{val:>14}")
            lines.append("".join(row_parts))

        lines.append(f"{'=' * 90}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _estimate_cost(
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost in USD based on provider pricing."""
    pricing = PROVIDER_PRICING.get(provider, {"input_per_1m": 0, "output_per_1m": 0})
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    return input_cost + output_cost


def run_provider_benchmark(
    persona: Persona,
    adapter: BaseLLMAdapter,
    provider_name: str,
    turns: list[dict[str, str]],
    seed: int = 42,
) -> ProviderResult:
    """
    Run a multi-turn conversation with a single provider and collect metrics.

    Args:
        persona: The persona to use.
        adapter: Pre-configured LLM adapter.
        provider_name: Name label for this provider.
        turns: List of dicts with "input" and "label" keys.
        seed: Determinism seed.

    Returns:
        ProviderResult with per-turn and aggregate metrics.
    """
    engine = PersonaEngine(persona, adapter=adapter, seed=seed, validate=True)
    turn_results: list[TurnResult] = []

    try:
        for i, turn in enumerate(turns, 1):
            result: ChatResult = engine.chat(turn["input"])

            turn_results.append(TurnResult(
                turn_number=i,
                label=turn.get("label", f"turn_{i}"),
                user_input=turn["input"],
                response_text=result.text,
                latency_ms=result.response.latency_ms,
                estimated_tokens=result.response.estimated_tokens,
                validation_passed=result.passed,
                violation_count=len(result.response.violations),
                confidence=result.confidence,
                competence=result.competence,
            ))
    except Exception as e:
        return ProviderResult(
            provider=provider_name,
            model=adapter.get_model_name(),
            turns=turn_results,
            total_latency_ms=0,
            avg_latency_ms=0,
            total_tokens=0,
            estimated_cost_usd=0,
            validation_pass_rate=0,
            total_violations=0,
            error=f"{type(e).__name__}: {e}",
        )

    # Aggregate
    latencies = [t.latency_ms for t in turn_results if t.latency_ms is not None]
    total_latency = sum(latencies) if latencies else 0
    avg_latency = total_latency / len(latencies) if latencies else 0
    total_tokens = sum(t.estimated_tokens for t in turn_results)
    passed = sum(1 for t in turn_results if t.validation_passed)
    pass_rate = passed / len(turn_results) if turn_results else 0
    total_violations = sum(t.violation_count for t in turn_results)

    # Rough split: 70% input, 30% output (typical for persona engine prompts)
    input_tokens = int(total_tokens * 0.7)
    output_tokens = total_tokens - input_tokens
    cost = _estimate_cost(provider_name, input_tokens, output_tokens)

    return ProviderResult(
        provider=provider_name,
        model=adapter.get_model_name(),
        turns=turn_results,
        total_latency_ms=total_latency,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
        validation_pass_rate=pass_rate,
        total_violations=total_violations,
    )


def run_benchmark(
    persona_path: str = "personas/chef.yaml",
    providers: list[str] | None = None,
    turns: list[dict[str, str]] | None = None,
    seed: int = 42,
    adapters: dict[str, BaseLLMAdapter] | None = None,
    dry_run: bool = False,
) -> BenchmarkReport:
    """
    Run a full benchmark comparing multiple providers.

    Args:
        persona_path: Path to persona YAML file.
        providers: List of provider names to compare (default: all configured).
        turns: Custom conversation turns (default: DEFAULT_BENCHMARK_TURNS).
        seed: Determinism seed for IR.
        adapters: Pre-configured adapters keyed by name (overrides provider creation).
        dry_run: If True, use MockLLMAdapter for all providers (no API calls).

    Returns:
        BenchmarkReport with full comparison data.
    """
    with open(persona_path) as f:
        data = yaml.safe_load(f)
    persona = Persona(**data)
    persona_name = getattr(getattr(persona, "identity", None), "name", persona_path)

    if turns is None:
        turns = DEFAULT_BENCHMARK_TURNS

    if providers is None:
        if dry_run:
            providers = ["mock_anthropic", "mock_openai", "mock_gemini", "mock_mistral", "mock_ollama"]
        else:
            providers = ["anthropic", "openai"]  # Safe default — only commonly configured

    results: list[ProviderResult] = []

    for provider in providers:
        logger.info(f"Benchmarking provider: {provider}")

        if adapters and provider in adapters:
            adapter = adapters[provider]
        elif dry_run or provider.startswith("mock_"):
            # Dry run: use mock adapters that report the "real" provider name
            label = provider.removeprefix("mock_") if provider.startswith("mock_") else provider
            adapter = MockLLMAdapter()
        else:
            try:
                adapter = create_adapter(provider)
            except Exception as e:
                results.append(ProviderResult(
                    provider=provider,
                    model="N/A",
                    turns=[],
                    total_latency_ms=0,
                    avg_latency_ms=0,
                    total_tokens=0,
                    estimated_cost_usd=0,
                    validation_pass_rate=0,
                    total_violations=0,
                    error=f"Setup failed: {e}",
                ))
                continue

        result = run_provider_benchmark(persona, adapter, provider, turns, seed)
        results.append(result)

    # Check IR consistency across providers (same seed → same IR)
    ir_consistent = True
    successful = [r for r in results if not r.error and r.turns]
    if len(successful) >= 2:
        ref = successful[0]
        for other in successful[1:]:
            for i in range(min(len(ref.turns), len(other.turns))):
                if (
                    abs(ref.turns[i].confidence - other.turns[i].confidence) > 0.001
                    or abs(ref.turns[i].competence - other.turns[i].competence) > 0.001
                ):
                    ir_consistent = False
                    break

    return BenchmarkReport(
        persona_name=str(persona_name),
        seed=seed,
        turn_count=len(turns),
        providers=results,
        ir_consistent=ir_consistent,
    )
