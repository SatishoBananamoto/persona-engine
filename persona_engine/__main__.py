"""
CLI entry point for persona_engine.

Usage::

    # Validate a persona YAML file
    python -m persona_engine validate personas/chef.yaml

    # Show persona info
    python -m persona_engine info personas/chef.yaml

    # Quick chat (IR-only, no LLM)
    python -m persona_engine plan personas/chef.yaml "What makes a good sauce?"

    # Chat with mock LLM
    python -m persona_engine chat personas/chef.yaml "What makes a good sauce?"

    # List available personas
    python -m persona_engine list personas/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def _load_persona(path: str) -> Any:
    """Load and validate a persona YAML file."""
    from persona_engine.schema.persona_schema import Persona

    with open(path) as f:
        data = yaml.safe_load(f)
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a persona YAML file."""
    from pydantic import ValidationError

    path = args.file
    try:
        persona = _load_persona(path)
        print(f"OK  {path}")
        print(f"    Persona: {persona.label} ({persona.persona_id})")
        print(f"    Domains: {len(persona.knowledge_domains)}")
        print(f"    Goals:   {len(persona.primary_goals)} primary, {len(persona.secondary_goals)} secondary")

        # Run a quick IR generation to validate planner compatibility
        if args.deep:
            from persona_engine import PersonaEngine
            engine = PersonaEngine(persona, llm_provider="mock")
            ir = engine.plan("Hello, tell me about yourself.")
            print(f"    IR:      confidence={ir.response_structure.confidence:.2f}, "
                  f"competence={ir.response_structure.competence:.2f}")
            print(f"    Deep validation: PASS")

        return 0

    except ValidationError as e:
        print(f"FAIL {path}", file=sys.stderr)
        for err in e.errors():
            loc = " → ".join(str(x) for x in err["loc"])
            print(f"     {loc}: {err['msg']}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Display persona information."""
    try:
        persona = _load_persona(args.file)
    except Exception as e:
        print(f"Error loading {args.file}: {e}", file=sys.stderr)
        return 1

    big5 = persona.psychology.big_five
    values = persona.psychology.values
    cog = persona.psychology.cognitive_style
    comm = persona.psychology.communication

    print(f"Persona: {persona.label}")
    print(f"ID:      {persona.persona_id}")
    print(f"Version: {persona.version}")
    print()

    print("Identity:")
    print(f"  Age:        {persona.identity.age}")
    print(f"  Location:   {persona.identity.location}")
    print(f"  Occupation: {persona.identity.occupation}")
    print(f"  Education:  {persona.identity.education}")
    print()

    print("Big Five Traits:")
    print(f"  Openness:          {big5.openness:.2f}")
    print(f"  Conscientiousness: {big5.conscientiousness:.2f}")
    print(f"  Extraversion:      {big5.extraversion:.2f}")
    print(f"  Agreeableness:     {big5.agreeableness:.2f}")
    print(f"  Neuroticism:       {big5.neuroticism:.2f}")
    print()

    print("Top Values:")
    value_dict = {
        "self_direction": values.self_direction,
        "stimulation": values.stimulation,
        "hedonism": values.hedonism,
        "achievement": values.achievement,
        "power": values.power,
        "security": values.security,
        "conformity": values.conformity,
        "tradition": values.tradition,
        "benevolence": values.benevolence,
        "universalism": values.universalism,
    }
    sorted_vals = sorted(value_dict.items(), key=lambda x: x[1], reverse=True)
    for name, weight in sorted_vals[:5]:
        print(f"  {name:20s} {weight:.2f}")
    print()

    print("Communication:")
    print(f"  Verbosity:     {comm.verbosity:.2f}")
    print(f"  Formality:     {comm.formality:.2f}")
    print(f"  Directness:    {comm.directness:.2f}")
    print()

    print(f"Knowledge Domains: {len(persona.knowledge_domains)}")
    for d in persona.knowledge_domains:
        print(f"  {d.domain}: {d.proficiency:.2f} ({', '.join(d.subdomains[:3])})")

    if persona.invariants.must_avoid:
        print(f"\nMust Avoid: {', '.join(persona.invariants.must_avoid)}")
    if persona.invariants.cannot_claim:
        print(f"Cannot Claim: {', '.join(persona.invariants.cannot_claim)}")

    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Generate IR without LLM call."""
    from persona_engine import PersonaEngine

    try:
        engine = PersonaEngine.from_yaml(args.file, llm_provider="mock")
        ir = engine.plan(args.message)

        if args.json:
            print(ir.to_json_deterministic())
        else:
            print(f"Stance:      {ir.response_structure.stance}")
            print(f"Confidence:  {ir.response_structure.confidence:.3f}")
            print(f"Competence:  {ir.response_structure.competence:.3f}")
            print(f"Tone:        {ir.communication_style.tone.value}")
            print(f"Verbosity:   {ir.communication_style.verbosity.value}")
            print(f"Formality:   {ir.communication_style.formality:.3f}")
            print(f"Disclosure:  {ir.knowledge_disclosure.disclosure_level:.3f}")
            print(f"Uncertainty: {ir.knowledge_disclosure.uncertainty_action.value}")
            print(f"Claim Type:  {ir.knowledge_disclosure.knowledge_claim_type.value}")
            print(f"Citations:   {len(ir.citations)}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_chat(args: argparse.Namespace) -> int:
    """Chat with a persona using mock LLM."""
    from persona_engine import PersonaEngine

    try:
        engine = PersonaEngine.from_yaml(
            args.file, llm_provider=args.backend,
        )
        result = engine.chat(args.message)
        print(f"{engine.persona.label}: {result.text}")
        print(f"\n  [confidence={result.confidence:.2f}, "
              f"competence={result.competence:.2f}, "
              f"tone={result.ir.communication_style.tone.value}]")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run provider benchmark comparison."""
    from persona_engine.benchmark import run_benchmark

    providers = args.providers if args.providers else None

    report = run_benchmark(
        persona_path=args.persona,
        providers=providers,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print(report.summary_table())

    if args.json:
        import json

        data = {
            "persona": report.persona_name,
            "seed": report.seed,
            "turn_count": report.turn_count,
            "ir_consistent": report.ir_consistent,
            "providers": [
                {
                    "provider": p.provider,
                    "model": p.model,
                    "avg_latency_ms": round(p.avg_latency_ms, 2),
                    "total_tokens": p.total_tokens,
                    "estimated_cost_usd": round(p.estimated_cost_usd, 6),
                    "validation_pass_rate": round(p.validation_pass_rate, 3),
                    "total_violations": p.total_violations,
                    "error": p.error,
                    "turns": [
                        {
                            "label": t.label,
                            "latency_ms": round(t.latency_ms, 2) if t.latency_ms else None,
                            "tokens": t.estimated_tokens,
                            "passed": t.validation_passed,
                            "confidence": round(t.confidence, 3),
                            "competence": round(t.competence, 3),
                        }
                        for t in p.turns
                    ],
                }
                for p in report.providers
            ],
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  JSON saved to {args.output}")
        else:
            print(json.dumps(data, indent=2))

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List persona YAML files in a directory."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 1

    files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not files:
        print(f"No persona files found in {directory}")
        return 0

    for f in files:
        try:
            persona = _load_persona(str(f))
            domains = len(persona.knowledge_domains)
            print(f"  {f.name:30s} {persona.label} ({domains} domains)")
        except Exception as e:
            print(f"  {f.name:30s} [ERROR: {e}]")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="persona_engine",
        description="Persona Engine CLI — validate, inspect, and test personas.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # validate
    p_val = sub.add_parser("validate", help="Validate a persona YAML file")
    p_val.add_argument("file", help="Path to persona YAML file")
    p_val.add_argument("--deep", action="store_true",
                       help="Also run IR generation to validate planner compatibility")

    # info
    p_info = sub.add_parser("info", help="Display persona information")
    p_info.add_argument("file", help="Path to persona YAML file")

    # plan
    p_plan = sub.add_parser("plan", help="Generate IR (no LLM call)")
    p_plan.add_argument("file", help="Path to persona YAML file")
    p_plan.add_argument("message", help="User message to plan for")
    p_plan.add_argument("--json", action="store_true", help="Output as JSON")

    # chat
    p_chat = sub.add_parser("chat", help="Chat with a persona")
    p_chat.add_argument("file", help="Path to persona YAML file")
    p_chat.add_argument("message", help="User message")
    p_chat.add_argument("--backend", default="mock",
                        choices=["mock", "template", "anthropic", "openai"],
                        help="LLM backend (default: mock)")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Compare LLM providers on the same conversation")
    p_bench.add_argument("--persona", default="personas/chef.yaml",
                         help="Path to persona YAML (default: personas/chef.yaml)")
    p_bench.add_argument("--providers", nargs="+",
                         help="Provider names to compare (default: auto-detect)")
    p_bench.add_argument("--seed", type=int, default=42, help="Determinism seed")
    p_bench.add_argument("--dry-run", action="store_true",
                         help="Use mock adapters (no API keys needed)")
    p_bench.add_argument("--json", action="store_true", help="Output as JSON")
    p_bench.add_argument("--output", help="Save JSON to file")

    # list
    p_list = sub.add_parser("list", help="List personas in a directory")
    p_list.add_argument("directory", nargs="?", default="personas",
                        help="Directory to scan (default: personas/)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "validate": cmd_validate,
        "info": cmd_info,
        "plan": cmd_plan,
        "chat": cmd_chat,
        "benchmark": cmd_benchmark,
        "list": cmd_list,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
