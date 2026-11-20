"""
Correlation Analysis — Psychometric validation via trait × parameter correlations.

Generates 200+ personas from 3 sources (Layer Zero, random uniform, shipped YAMLs),
runs each through engine.plan(), computes correlation matrix of Big Five traits
vs IR parameters, and compares against published literature values.

Usage:
    python3 -m eval.correlation_analysis              # run analysis
    python3 -m eval.correlation_analysis --verbose     # show per-persona details

No API key needed — uses template adapter.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*languages field is populated.*")

from persona_engine import PersonaEngine
from persona_engine.persona_builder import PersonaBuilder

VERBOSE = "--verbose" in sys.argv


# =============================================================================
# Data Collection
# =============================================================================

@dataclass
class PersonaData:
    """Collected data for one persona."""
    source: str  # "layer_zero", "random", "shipped"
    name: str
    O: float
    C: float
    E: float
    A: float
    N: float
    # IR parameters (averaged across prompts)
    confidence: float = 0.0
    elasticity: float = 0.0
    directness: float = 0.0
    disclosure: float = 0.0
    formality: float = 0.0
    competence: float = 0.0
    # Derived from trait guidance
    hedging: float = 0.0
    proactivity: float = 0.0
    neg_tone_weight: float = 0.0
    enthusiasm: float = 0.0


PROMPTS = [
    "What do you think about this approach?",
    "How should we handle this situation?",
    "Tell me about your perspective on this.",
]


def _collect_ir_data(engine: PersonaEngine, traits: dict) -> PersonaData:
    """Run prompts and collect averaged IR parameters."""
    from persona_engine.behavioral.trait_interpreter import TraitInterpreter
    from persona_engine.schema.persona_schema import BigFiveTraits

    ti = TraitInterpreter(BigFiveTraits(**{
        "openness": traits["O"], "conscientiousness": traits["C"],
        "extraversion": traits["E"], "agreeableness": traits["A"],
        "neuroticism": traits["N"],
    }))

    data = PersonaData(
        source="", name="",
        O=traits["O"], C=traits["C"], E=traits["E"],
        A=traits["A"], N=traits["N"],
    )

    n = len(PROMPTS)
    for prompt in PROMPTS:
        ir = engine.plan(prompt)
        data.confidence += ir.response_structure.confidence / n
        data.elasticity += ir.response_structure.elasticity / n
        data.directness += ir.communication_style.directness / n
        data.disclosure += ir.knowledge_disclosure.disclosure_level / n
        data.formality += ir.communication_style.formality / n
        data.competence += ir.response_structure.competence / n

    # Trait-derived (constant across prompts)
    data.hedging = ti.influences_hedging_frequency()
    data.proactivity = ti.influences_proactivity()
    data.neg_tone_weight = ti.get_negative_tone_bias()
    data.enthusiasm = ti.get_enthusiasm_baseline()

    return data


# =============================================================================
# Persona Sources
# =============================================================================

def _collect_layer_zero(count: int = 100) -> list[PersonaData]:
    """Mint personas via Layer Zero."""
    try:
        import layer_zero
    except ImportError:
        print("  Layer Zero not available, skipping")
        return []

    occupations = [
        "nurse", "software engineer", "chef", "lawyer", "teacher",
        "musician", "scientist", "social worker", "firefighter", "journalist",
        "accountant", "artist", "police officer", "therapist", "mechanic",
        "architect", "farmer", "pilot", "librarian", "surgeon",
    ]

    results = []
    per_occ = max(1, count // len(occupations))

    for occ in occupations:
        try:
            personas = layer_zero.mint(occupation=occ, count=per_occ, seed=42, validate="silent")
            for mp in personas:
                bf = mp.persona.psychology.big_five
                engine = PersonaEngine(persona=mp.persona, llm_provider="template")
                traits = {"O": bf.openness, "C": bf.conscientiousness,
                          "E": bf.extraversion, "A": bf.agreeableness, "N": bf.neuroticism}
                data = _collect_ir_data(engine, traits)
                data.source = "layer_zero"
                data.name = mp.label
                results.append(data)
        except Exception:
            pass

    return results


def _collect_random_uniform(count: int = 100, seed: int = 123) -> list[PersonaData]:
    """Generate personas with uniformly sampled Big Five traits."""
    rng = np.random.default_rng(seed)
    results = []

    for i in range(count):
        traits = {
            "O": float(rng.uniform(0.05, 0.95)),
            "C": float(rng.uniform(0.05, 0.95)),
            "E": float(rng.uniform(0.05, 0.95)),
            "A": float(rng.uniform(0.05, 0.95)),
            "N": float(rng.uniform(0.05, 0.95)),
        }
        builder = PersonaBuilder(f"Random_{i}", "General Professional")
        builder._big_five = {
            "openness": traits["O"], "conscientiousness": traits["C"],
            "extraversion": traits["E"], "agreeableness": traits["A"],
            "neuroticism": traits["N"],
        }
        try:
            persona = builder.build()
            engine = PersonaEngine(persona=persona, llm_provider="template")
            data = _collect_ir_data(engine, traits)
            data.source = "random"
            data.name = f"Random_{i}"
            results.append(data)
        except Exception:
            pass

    return results


def _collect_shipped() -> list[PersonaData]:
    """Load shipped persona YAML files."""
    persona_dir = Path(__file__).parent.parent / "personas"
    results = []

    for yaml_file in sorted(persona_dir.glob("*.yaml")):
        try:
            engine = PersonaEngine.from_yaml(str(yaml_file), llm_provider="template")
            bf = engine._persona.psychology.big_five
            traits = {"O": bf.openness, "C": bf.conscientiousness,
                      "E": bf.extraversion, "A": bf.agreeableness, "N": bf.neuroticism}
            data = _collect_ir_data(engine, traits)
            data.source = "shipped"
            data.name = yaml_file.stem
            results.append(data)
        except Exception:
            pass

    return results


# =============================================================================
# Correlation Computation
# =============================================================================

def pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0
    return cov / (std_x * std_y)


TRAITS = ["O", "C", "E", "A", "N"]
IR_FIELDS = ["confidence", "elasticity", "directness", "disclosure",
             "hedging", "proactivity", "neg_tone_weight", "enthusiasm"]


def compute_correlation_matrix(data: list[PersonaData]) -> dict[str, dict[str, float]]:
    """Compute trait × IR parameter correlation matrix."""
    matrix: dict[str, dict[str, float]] = {}
    for trait in TRAITS:
        matrix[trait] = {}
        trait_vals = [getattr(d, trait) for d in data]
        for field in IR_FIELDS:
            field_vals = [getattr(d, field) for d in data]
            r = pearson_r(trait_vals, field_vals)
            matrix[trait][field] = round(r, 3)
    return matrix


# =============================================================================
# Literature Expectations
# =============================================================================

# Expected correlation DIRECTIONS from Yarkoni (2010) + personality literature
# Format: (trait, field, expected_sign, citation)
EXPECTED_CORRELATIONS = [
    ("O", "elasticity", "+", "High-O → more willing to change mind"),
    ("C", "confidence", "+", "High-C → more methodical certainty"),
    ("E", "disclosure", "+", "High-E → more self-disclosure"),
    ("E", "proactivity", "+", "High-E → more proactive engagement"),
    ("E", "enthusiasm", "+", "High-E → higher enthusiasm baseline"),
    ("A", "directness", "-", "High-A → less direct, more diplomatic"),
    ("A", "hedging", "+", "High-A → more hedging language"),
    ("N", "confidence", "-", "High-N → lower confidence"),
    ("N", "neg_tone_weight", "+", "High-N → more negative tone bias"),
    ("N", "hedging", "+", "High-N → more hedging (uncertainty)"),
]


# =============================================================================
# Output
# =============================================================================

def _print_matrix(matrix: dict[str, dict[str, float]], title: str) -> None:
    """Print correlation matrix as table."""
    print(f"\n{title}")
    print("=" * 100)
    header = f"{'Trait':<5}" + "".join(f"{f:>14}" for f in IR_FIELDS)
    print(header)
    print("-" * 100)
    for trait in TRAITS:
        row = f"{trait:<5}"
        for field in IR_FIELDS:
            r = matrix[trait][field]
            # Mark significant correlations
            marker = "*" if abs(r) > 0.15 else " "
            row += f"{r:>+8.3f}{marker}     "
        print(row)
    print("\n  * = |r| > 0.15 (visible effect)")


def _check_expected(matrix: dict[str, dict[str, float]]) -> list[dict]:
    """Check correlation directions against literature expectations."""
    results = []
    for trait, field, sign, citation in EXPECTED_CORRELATIONS:
        r = matrix[trait][field]
        if sign == "+":
            passed = r > 0
        else:
            passed = r < 0
        results.append({
            "trait": trait, "field": field, "expected_sign": sign,
            "actual_r": r, "passed": passed, "citation": citation,
        })
    return results


def main() -> int:
    print("\n" + "=" * 100)
    print("  CORRELATION ANALYSIS — Psychometric Validation")
    print("  (Big Five traits × IR parameters, compared against literature)")
    print("=" * 100)

    # Collect data
    print("\n  Collecting personas...")

    lz_data = _collect_layer_zero(100)
    print(f"    Layer Zero: {len(lz_data)} personas")

    rand_data = _collect_random_uniform(100)
    print(f"    Random uniform: {len(rand_data)} personas")

    shipped_data = _collect_shipped()
    print(f"    Shipped YAMLs: {len(shipped_data)} personas")

    all_data = lz_data + rand_data + shipped_data
    print(f"    Total: {len(all_data)} personas")

    # Compute matrices
    matrix_all = compute_correlation_matrix(all_data)
    _print_matrix(matrix_all, "CORRELATION MATRIX — ALL PERSONAS")

    if lz_data:
        matrix_lz = compute_correlation_matrix(lz_data)
        _print_matrix(matrix_lz, "CORRELATION MATRIX — LAYER ZERO ONLY (realistic range)")

    matrix_rand = compute_correlation_matrix(rand_data)
    _print_matrix(matrix_rand, "CORRELATION MATRIX — RANDOM UNIFORM (full range)")

    # Check expected directions
    checks = _check_expected(matrix_all)
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    print(f"\n{'=' * 100}")
    print(f"  DIRECTION VALIDATION ({passed}/{total})")
    print(f"{'=' * 100}")
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        marker = "  " if c["passed"] else ">>"
        print(f"  {marker} [{status}] {c['trait']} → {c['field']}: "
              f"expected {c['expected_sign']}, actual r={c['actual_r']:+.3f}  "
              f"({c['citation']})")

    print(f"\n  Result: {passed}/{total} correlation directions match literature")

    # Trait distribution stats
    print(f"\n{'=' * 100}")
    print(f"  TRAIT DISTRIBUTIONS")
    print(f"{'=' * 100}")
    for trait in TRAITS:
        vals = [getattr(d, trait) for d in all_data]
        print(f"  {trait}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
              f"min={min(vals):.3f}  max={max(vals):.3f}")

    # Save report
    report = {
        "n_personas": len(all_data),
        "sources": {
            "layer_zero": len(lz_data),
            "random": len(rand_data),
            "shipped": len(shipped_data),
        },
        "correlation_matrix": matrix_all,
        "direction_checks": checks,
        "summary": {"passed": passed, "total": total},
    }
    report_path = Path(__file__).parent / "correlation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report: {report_path}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
