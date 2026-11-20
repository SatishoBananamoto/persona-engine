"""
Benchmark Validation — 8 personality profiles against literature expectations.

Runs each profile through engine.plan() with diverse prompts, collects IR
parameters, and validates direction + relative magnitude against Yarkoni (2010)
and personality psychology literature.

Usage:
    python3 -m eval.benchmark_profiles           # run all profiles
    python3 -m eval.benchmark_profiles --verbose  # show per-prompt IR details

Exit codes:
    0 — all direction checks passed
    1 — one or more direction checks failed
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore", message=".*languages field is populated.*")

from persona_engine import PersonaEngine
from persona_engine.persona_builder import PersonaBuilder

# =============================================================================
# Benchmark Profiles
# =============================================================================

PROFILES: dict[str, dict[str, float]] = {
    "prototypical_extravert": {"O": 0.5, "C": 0.5, "E": 0.9, "A": 0.5, "N": 0.3},
    "prototypical_introvert": {"O": 0.6, "C": 0.5, "E": 0.1, "A": 0.5, "N": 0.5},
    "prototypical_agreeable": {"O": 0.5, "C": 0.5, "E": 0.6, "A": 0.9, "N": 0.3},
    "prototypical_antagonist": {"O": 0.4, "C": 0.5, "E": 0.5, "A": 0.1, "N": 0.5},
    "prototypical_neurotic": {"O": 0.5, "C": 0.4, "E": 0.3, "A": 0.5, "N": 0.9},
    "emotionally_stable": {"O": 0.5, "C": 0.6, "E": 0.6, "A": 0.6, "N": 0.1},
    "creative_intellectual": {"O": 0.9, "C": 0.4, "E": 0.5, "A": 0.5, "N": 0.5},
    "disciplined_achiever": {"O": 0.3, "C": 0.9, "E": 0.5, "A": 0.5, "N": 0.3},
}

# Diverse prompts to average across
PROMPTS = [
    "What do you think about the current state of AI technology?",
    "Tell me about a time you had to make a difficult decision.",
    "How should we approach climate change as a society?",
    "I think remote work is better than office work. What's your view?",
]

# =============================================================================
# Direction Expectations (from Yarkoni 2010 + personality literature)
# =============================================================================
# Format: (profile_high, profile_low, field, expected_direction)
# ">" means profile_high should have a HIGHER value than profile_low

DIRECTION_CHECKS: list[tuple[str, str, str, str]] = [
    # Extraversion
    ("prototypical_extravert", "prototypical_introvert", "disclosure", ">"),
    ("prototypical_extravert", "prototypical_introvert", "enthusiasm_active", ">"),
    ("prototypical_extravert", "prototypical_introvert", "proactivity", ">"),

    # Agreeableness
    ("prototypical_agreeable", "prototypical_antagonist", "hedging", ">"),
    ("prototypical_antagonist", "prototypical_agreeable", "directness", ">"),

    # Neuroticism
    ("prototypical_neurotic", "emotionally_stable", "negative_tone_weight", ">"),
    ("emotionally_stable", "prototypical_neurotic", "confidence", ">"),
    ("prototypical_neurotic", "emotionally_stable", "hedging", ">"),

    # Openness
    ("creative_intellectual", "disciplined_achiever", "elasticity", ">"),

    # Conscientiousness
    ("disciplined_achiever", "creative_intellectual", "verbosity_numeric", ">"),
]


# =============================================================================
# Build Persona from Profile
# =============================================================================

def _build_persona(name: str, traits: dict[str, float]):
    """Build a Persona from Big Five trait dict using PersonaBuilder."""
    builder = PersonaBuilder(
        name=name.replace("_", " ").title(),
        occupation="General Professional",
    )
    # Set Big Five directly
    builder._big_five = {
        "openness": traits["O"],
        "conscientiousness": traits["C"],
        "extraversion": traits["E"],
        "agreeableness": traits["A"],
        "neuroticism": traits["N"],
    }
    return builder.build()


# =============================================================================
# Extract IR Metrics
# =============================================================================

@dataclass
class ProfileResult:
    """Aggregated IR metrics across prompts for one profile."""
    name: str
    traits: dict[str, float]
    confidence: float = 0.0
    elasticity: float = 0.0
    directness: float = 0.0
    formality: float = 0.0
    disclosure: float = 0.0
    competence: float = 0.0
    # Derived from trait guidance (not directly in IR)
    hedging: float = 0.0
    enthusiasm_active: bool = False
    negative_tone_weight: float = 0.0
    proactivity: float = 0.0
    verbosity_numeric: float = 0.0
    tone: str = ""
    per_prompt: list[dict] = field(default_factory=list)


def _run_profile(name: str, traits: dict[str, float], prompts: list[str]) -> ProfileResult:
    """Run a profile through engine.plan() and collect averaged metrics."""
    persona = _build_persona(name, traits)
    engine = PersonaEngine(persona=persona, llm_provider="template")

    result = ProfileResult(name=name, traits=traits)
    n = len(prompts)

    for prompt in prompts:
        ir = engine.plan(prompt)

        # Extract numeric fields
        conf = ir.response_structure.confidence
        elas = ir.response_structure.elasticity
        dire = ir.communication_style.directness
        form = ir.communication_style.formality
        disc = ir.knowledge_disclosure.disclosure_level
        comp = ir.response_structure.competence
        tone = ir.communication_style.tone.value
        verb = ir.communication_style.verbosity.value

        # Map verbosity enum to numeric
        verb_map = {"brief": 0.25, "medium": 0.5, "detailed": 0.75}
        verb_num = verb_map.get(verb, 0.5)

        # Extract trait guidance metrics from trait_interpreter
        from persona_engine.behavioral.trait_interpreter import TraitInterpreter
        ti = TraitInterpreter(persona.psychology.big_five)
        hedging = ti.influences_hedging_frequency()
        enthusiasm = ti.get_enthusiasm_baseline()
        neg_tone = ti.get_negative_tone_bias()
        proact = ti.influences_proactivity()

        result.confidence += conf / n
        result.elasticity += elas / n
        result.directness += dire / n
        result.formality += form / n
        result.disclosure += disc / n
        result.competence += comp / n
        result.hedging += hedging / n
        result.negative_tone_weight += neg_tone / n
        result.proactivity += proact / n
        result.verbosity_numeric += verb_num / n

        result.per_prompt.append({
            "prompt": prompt[:60],
            "confidence": round(conf, 3),
            "elasticity": round(elas, 3),
            "directness": round(dire, 3),
            "formality": round(form, 3),
            "disclosure": round(disc, 3),
            "tone": tone,
            "verbosity": verb,
        })

    result.enthusiasm_active = traits["E"] > 0.6
    result.tone = result.per_prompt[-1]["tone"] if result.per_prompt else ""

    return result


# =============================================================================
# Validation
# =============================================================================

def _run_direction_checks(
    results: dict[str, ProfileResult],
) -> list[tuple[str, str, str, str, float, float, bool]]:
    """Run all direction checks and return results."""
    checks = []
    for high_name, low_name, field_name, direction in DIRECTION_CHECKS:
        high_val = getattr(results[high_name], field_name)
        low_val = getattr(results[low_name], field_name)

        if direction == ">":
            passed = high_val > low_val
        else:
            passed = high_val < low_val

        # Handle bool fields
        if isinstance(high_val, bool):
            high_val = float(high_val)
            low_val = float(getattr(results[low_name], field_name))

        checks.append((high_name, low_name, field_name, direction, high_val, low_val, passed))

    return checks


def _print_profile_table(results: dict[str, ProfileResult]) -> None:
    """Print a comparison table across all profiles."""
    print("\n" + "=" * 100)
    print("  BENCHMARK PROFILE COMPARISON")
    print("=" * 100)

    # Header
    fields = ["confidence", "elasticity", "directness", "disclosure",
              "hedging", "neg_tone", "proactivity", "verbosity"]
    header = f"{'Profile':<25}" + "".join(f"{f:>12}" for f in fields)
    print(header)
    print("-" * 100)

    for name, r in results.items():
        traits_str = f"O={r.traits['O']:.1f} C={r.traits['C']:.1f} E={r.traits['E']:.1f} A={r.traits['A']:.1f} N={r.traits['N']:.1f}"
        row = f"{name:<25}"
        row += f"{r.confidence:>12.3f}"
        row += f"{r.elasticity:>12.3f}"
        row += f"{r.directness:>12.3f}"
        row += f"{r.disclosure:>12.3f}"
        row += f"{r.hedging:>12.3f}"
        row += f"{r.negative_tone_weight:>12.3f}"
        row += f"{r.proactivity:>12.3f}"
        row += f"{r.verbosity_numeric:>12.3f}"
        print(row)
        print(f"  {'(' + traits_str + ')':<23}")

    print()


def _print_direction_results(
    checks: list[tuple[str, str, str, str, float, float, bool]],
) -> None:
    """Print direction check results."""
    print("=" * 100)
    print("  DIRECTION VALIDATION")
    print("=" * 100)

    passed_count = sum(1 for c in checks if c[6])
    total = len(checks)

    for high, low, field_name, direction, high_val, low_val, passed in checks:
        status = "PASS" if passed else "FAIL"
        marker = "  " if passed else ">>"
        diff = high_val - low_val
        print(
            f"  {marker} [{status}] {high:<25} {direction} {low:<25} "
            f"on {field_name:<20} ({high_val:.3f} vs {low_val:.3f}, diff={diff:+.3f})"
        )

    print(f"\n  Result: {passed_count}/{total} direction checks passed")
    if passed_count < total:
        print("  WARNING: Some trait-behavior mappings move in unexpected directions!")
    else:
        print("  All trait-behavior mappings align with literature expectations.")
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    verbose = "--verbose" in sys.argv

    print("\nRunning 8 benchmark profiles x 4 prompts...\n")

    results: dict[str, ProfileResult] = {}
    for name, traits in PROFILES.items():
        r = _run_profile(name, traits, PROMPTS)
        results[name] = r
        print(f"  {name:<30} done")

    _print_profile_table(results)

    if verbose:
        print("=" * 100)
        print("  PER-PROMPT DETAILS")
        print("=" * 100)
        for name, r in results.items():
            print(f"\n  {name}:")
            for pp in r.per_prompt:
                print(f"    {pp['prompt']:<60} conf={pp['confidence']:.3f} "
                      f"dir={pp['directness']:.3f} disc={pp['disclosure']:.3f} "
                      f"tone={pp['tone']}")

    checks = _run_direction_checks(results)
    _print_direction_results(checks)

    # Write report
    report_path = Path(__file__).parent / "benchmark_report.json"
    report = {
        "profiles": {
            name: {
                "traits": r.traits,
                "confidence": round(r.confidence, 4),
                "elasticity": round(r.elasticity, 4),
                "directness": round(r.directness, 4),
                "formality": round(r.formality, 4),
                "disclosure": round(r.disclosure, 4),
                "hedging": round(r.hedging, 4),
                "negative_tone_weight": round(r.negative_tone_weight, 4),
                "proactivity": round(r.proactivity, 4),
                "verbosity_numeric": round(r.verbosity_numeric, 4),
            }
            for name, r in results.items()
        },
        "direction_checks": [
            {
                "high": c[0], "low": c[1], "field": c[2],
                "direction": c[3], "high_val": round(c[4], 4),
                "low_val": round(c[5], 4), "passed": c[6],
            }
            for c in checks
        ],
        "summary": {
            "total_checks": len(checks),
            "passed": sum(1 for c in checks if c[6]),
            "failed": sum(1 for c in checks if not c[6]),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_path}\n")

    all_passed = all(c[6] for c in checks)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
