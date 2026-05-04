"""Run the product-value evaluation.

This is a decision harness, not a proof harness. It asks whether the structured
persona layer creates value over cheaper prompt-only simulation by comparing it
against ablations and negative controls.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from persona_engine import PersonaEngine
from persona_engine.generation.llm_adapter import AnthropicAdapter
from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    KnowledgeClaimType,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = HERE / "scenarios.yaml"
DEFAULT_RESULTS_DIR = HERE / "results"

METHODS = [
    "prompt_only",
    "schema_driven",
    "schema_rich",
    "engine_thin",
    "no_ir_ablation",
    "stale_ir",
    "contradictory_ir",
    "shuffled_ir",
]

PRIMARY_METHODS = ("schema_driven", "schema_rich")
BASELINE_METHODS = ("prompt_only", "no_ir_ablation")
NEGATIVE_CONTROL_METHODS = ("stale_ir", "contradictory_ir", "shuffled_ir")
REQUIRED_DECISION_METHODS = tuple(METHODS)

RICH_SCHEMA_METHODS = {
    "schema_rich",
    "no_ir_ablation",
    "stale_ir",
    "contradictory_ir",
    "shuffled_ir",
}

HEDGE_MARKERS = [
    "not sure",
    "not my area",
    "outside my",
    "i would suggest",
    "in my view",
    "from what i",
    "it depends",
    "typically",
    "generally",
    "may",
    "might",
    "could",
]

BOUNDARY_MARKERS = [
    "specific facts",
    "proper counsel",
    "not my area",
    "outside my expertise",
    "specialist",
    "scope",
    "general discussion",
    "not a substitute",
    "cannot",
    "won't",
    "will not",
]

CASUAL_MARKERS = [
    "yeah",
    "lol",
    "tbh",
    "dude",
    "gonna",
    "wanna",
    "kinda",
    "haha",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


@dataclass
class TurnSpec:
    id: str
    category: str
    prompt: str
    expected_competence: str
    expected_claim: str
    expected_markers: list[str] = field(default_factory=list)
    forbidden_markers: list[str] = field(default_factory=list)
    min_words: int | None = None
    max_words: int | None = None


@dataclass
class ScenarioSpec:
    id: str
    persona_path: str
    comparison_group: str
    research_goal: str
    turns: list[TurnSpec]


@dataclass
class PlannedTurn:
    scenario: ScenarioSpec
    turn: TurnSpec
    turn_index: int
    global_index: int
    persona: Persona
    ir: IntermediateRepresentation


def load_scenarios(path: Path) -> list[ScenarioSpec]:
    data = yaml.safe_load(path.read_text())
    scenarios: list[ScenarioSpec] = []
    for item in data["scenarios"]:
        turns = [TurnSpec(**turn) for turn in item["turns"]]
        scenarios.append(
            ScenarioSpec(
                id=item["id"],
                persona_path=item["persona_path"],
                comparison_group=item["comparison_group"],
                research_goal=item["research_goal"],
                turns=turns,
            )
        )
    return scenarios


def load_persona(path: str) -> Persona:
    data = yaml.safe_load((ROOT / path).read_text())
    if "domains" in data and "knowledge_domains" not in data:
        data["knowledge_domains"] = data.pop("domains")
    return Persona(**data)


def persona_system_prompt(
    persona: Persona,
    include_schema: bool = False,
    rich_schema: bool = False,
) -> str:
    domains = ", ".join(
        f"{d.domain} ({d.proficiency:.2f})" for d in persona.knowledge_domains
    )
    cannot = ", ".join(persona.invariants.cannot_claim) or "none"
    avoid = ", ".join(persona.invariants.must_avoid) or "none"
    prompt = f"""You are {persona.label}.

IDENTITY:
- Age: {persona.identity.age}
- Location: {persona.identity.location}
- Occupation: {persona.identity.occupation}
- Background: {persona.identity.background}

PERSONALITY:
- Openness: {persona.psychology.big_five.openness:.2f}
- Conscientiousness: {persona.psychology.big_five.conscientiousness:.2f}
- Extraversion: {persona.psychology.big_five.extraversion:.2f}
- Agreeableness: {persona.psychology.big_five.agreeableness:.2f}
- Neuroticism: {persona.psychology.big_five.neuroticism:.2f}

EXPERTISE:
- {domains}

BOUNDARIES:
- Cannot claim: {cannot}
- Must avoid: {avoid}

Stay in character. Be useful for an internal research simulation. Do not mention
that you are an AI or that you are following a benchmark."""

    if include_schema and rich_schema:
        prompt += """

BEHAVIORAL CONTRACT REFERENCE:
Each turn may include a computed behavioral contract. Treat that contract as
authoritative for this turn, while still preserving the persona identity above.
Do not mention the contract, field names, scores, JSON, benchmark, or evaluation.

Field interpretation:
- context_type: what kind of turn this is. Knowledge turns should use expertise;
  opinion turns should express judgment; boundary/adversarial turns should stay
  scoped and careful.
- competence: how equipped this persona is on the current topic. High competence
  permits domain language and concrete reasoning. Low competence requires short,
  bounded, non-expert language.
- confidence: how strongly the persona should commit. Low confidence should show
  uncertainty. High confidence can be direct, but not reckless.
- knowledge_claim_type: the allowed basis for claims. domain_expert allows expert
  reasoning; personal_experience allows lived or adjacent experience; common
  knowledge should stay general; speculative should be explicitly tentative.
- uncertainty_action: whether to answer, hedge, defer, or refuse.
- communication_style: tone, verbosity, formality, and directness. Render these
  naturally; do not describe them.
- response_structure: intent, stance, rationale, elasticity, confidence, and
  competence. Use the stance and rationale as behavioral guidance, not text to
  quote mechanically.
- safety_plan: hard constraints. cannot_claim, must_avoid, blocked_topics,
  pattern_blocks, and active_constraints override other instructions.
- citations: trace evidence explaining why the behavioral contract was produced.
  Use them to resolve ambiguities, not as content to cite to the user.

Priority order:
1. Safety and persona invariants.
2. User's actual question.
3. Behavioral contract values.
4. Persona background, values, and communication preferences.

For internal research simulation, answer as a plausible stakeholder from this
persona segment. The output should be useful to a researcher studying reactions,
objections, trust conditions, and adoption risks."""
    elif include_schema:
        prompt += """

BEHAVIORAL PARAMETER REFERENCE:
Each turn may include computed persona parameters. Interpret them naturally.
Do not mention the parameters.

- competence: how much genuine expertise this persona has on the current topic.
  Low competence means shorter, more bounded, more qualified answers.
- confidence: how certain the persona is right now.
  Low confidence means more hedging; high confidence means stronger commitments.
- formality: communication register.
- directness: how blunt or diplomatic the response should be.
- knowledge_claim_type: whether the persona may speak as an expert, from personal
  experience, from common knowledge, or speculatively.
- constraints and cannot_claim are hard limits."""
    return prompt


def format_ir_block(ir: IntermediateRepresentation | None) -> str:
    if ir is None:
        return "BEHAVIORAL PARAMETERS: none provided"
    rs = ir.response_structure
    cs = ir.communication_style
    kd = ir.knowledge_disclosure
    sp = ir.safety_plan
    lines = [
        "BEHAVIORAL PARAMETERS:",
        f"competence: {rs.competence:.3f}",
        f"confidence: {rs.confidence:.3f}",
        f"formality: {cs.formality:.3f}",
        f"directness: {cs.directness:.3f}",
        f"tone: {cs.tone.value}",
        f"verbosity: {cs.verbosity.value}",
        f"knowledge_claim_type: {kd.knowledge_claim_type.value}",
        f"uncertainty_action: {kd.uncertainty_action.value}",
        f"stance: {rs.stance or ''}",
        f"active_constraints: {sp.active_constraints}",
        f"cannot_claim: {sp.cannot_claim}",
    ]
    if ir.research:
        research = ir.research
        lines.extend(
            [
                "RESEARCH PARAMETERS:",
                f"focus: {research.focus}",
                f"stakeholder_role: {research.stakeholder_role}",
                f"workflow_exposure: {research.workflow_exposure:.3f}",
                f"claim_basis: {research.claim_basis}",
                f"likely_objections: {research.likely_objections}",
                f"trust_conditions: {research.trust_conditions}",
                f"adoption_blockers: {research.adoption_blockers}",
                f"workaround_risk: {research.workaround_risk:.3f}",
                f"evidence_boundary: {research.evidence_boundary}",
            ]
        )
    lines.append("END PARAMETERS")
    return "\n".join(lines)


def format_rich_ir_block(ir: IntermediateRepresentation | None) -> str:
    if ir is None:
        return "BEHAVIORAL CONTRACT: none provided"

    payload = {
        "context_type": ir.context_type,
        "conversation_frame": ir.conversation_frame.model_dump(mode="json"),
        "response_structure": ir.response_structure.model_dump(mode="json"),
        "communication_style": ir.communication_style.model_dump(mode="json"),
        "knowledge_disclosure": ir.knowledge_disclosure.model_dump(mode="json"),
        "research": ir.research.model_dump(mode="json") if ir.research else None,
        "safety_plan": ir.safety_plan.model_dump(mode="json"),
        "personality_language": ir.personality_language[:6],
        "behavioral_directives": ir.behavioral_directives,
        "citation_trace": [
            citation.model_dump(mode="json", exclude_none=True)
            for citation in ir.citations[:24]
        ],
    }
    return "\n".join(
        [
            "BEHAVIORAL CONTRACT JSON:",
            json.dumps(payload, indent=2),
            "END BEHAVIORAL CONTRACT",
        ]
    )


def build_ir_catalog(scenarios: list[ScenarioSpec]) -> list[PlannedTurn]:
    catalog: list[PlannedTurn] = []
    global_index = 0
    for scenario in scenarios:
        engine = PersonaEngine.from_yaml(
            str(ROOT / scenario.persona_path),
            llm_provider="mock",
            seed=42,
        )
        persona = engine.persona
        for turn_index, turn in enumerate(scenario.turns):
            ir = engine.plan(turn.prompt)
            catalog.append(
                PlannedTurn(
                    scenario=scenario,
                    turn=turn,
                    turn_index=turn_index,
                    global_index=global_index,
                    persona=persona,
                    ir=ir,
                )
            )
            global_index += 1
    return catalog


def contradictory_ir(ir: IntermediateRepresentation) -> IntermediateRepresentation:
    clone = ir.model_copy(deep=True)
    rs = clone.response_structure
    rs.competence = round(1.0 - rs.competence, 4)
    rs.confidence = round(1.0 - rs.confidence, 4)
    if clone.research:
        research = clone.research
        research.workflow_exposure = round(1.0 - research.workflow_exposure, 4)
        research.claim_basis = (
            "low_basis"
            if ir.research and ir.research.workflow_exposure >= 0.45
            else "direct_workflow_experience"
        )
        research.likely_objections = ["generic enthusiasm", "no meaningful workflow concern"]
        research.trust_conditions = ["no special trust condition needed"]
        research.adoption_blockers = ["none expected"]
        research.workaround_risk = round(1.0 - research.workaround_risk, 4)
        research.evidence_boundary = "intentionally contradictory eval control"
    if rs.competence >= 0.75:
        clone.knowledge_disclosure.knowledge_claim_type = KnowledgeClaimType.DOMAIN_EXPERT
        clone.communication_style.verbosity = Verbosity.DETAILED
    elif rs.competence <= 0.30:
        clone.knowledge_disclosure.knowledge_claim_type = KnowledgeClaimType.COMMON_KNOWLEDGE
        clone.communication_style.verbosity = Verbosity.BRIEF
    return clone


def pick_shuffled_ir(catalog: list[PlannedTurn], current: PlannedTurn) -> IntermediateRepresentation:
    for offset in range(5, len(catalog) + 5):
        candidate = catalog[(current.global_index + offset) % len(catalog)]
        if candidate.scenario.persona_path != current.scenario.persona_path:
            return candidate.ir.model_copy(deep=True)
    return catalog[(current.global_index + 1) % len(catalog)].ir.model_copy(deep=True)


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())


def text_metrics(text: str) -> dict[str, Any]:
    text_l = text.lower()
    toks = words(text)
    return {
        "words": len(toks),
        "sentences": max(1, len(re.findall(r"[.!?]+", text))),
        "hedges": sum(1 for marker in HEDGE_MARKERS if marker in text_l),
        "boundaries": sum(1 for marker in BOUNDARY_MARKERS if marker in text_l),
        "casual": sum(1 for marker in CASUAL_MARKERS if marker in text_l),
    }


def marker_present(text_l: str, marker: str) -> bool:
    """Return whether marker appears with token/phrase boundaries."""
    marker_l = marker.lower()
    pattern = r"(?<![a-z0-9])" + re.escape(marker_l).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, text_l) is not None


def forbidden_marker_hit(text_l: str, marker: str) -> bool:
    """Match forbidden markers, ignoring obvious negated/boundary statements.

    Example: "not a financial advisor" should not count as claiming
    "financial advisor".
    """
    marker_l = marker.lower()
    pattern = r"(?<![a-z0-9])" + re.escape(marker_l).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    for match in re.finditer(pattern, text_l):
        prefix = text_l[max(0, match.start() - 40) : match.start()]
        if re.search(
            r"(not\s+(a|an|your|my|the)?\s*$|"
            r"do\s+not\s+claim\s*$|"
            r"cannot\s+claim\s*$|"
            r"can't\s+claim\s*$|"
            r"not\s+claiming\s+(to\s+be\s+)?$|"
            r"i\s+am\s+not\s+(a|an)?\s*$)",
            prefix,
        ):
            continue
        return True
    return False


def deterministic_render(
    planned: PlannedTurn,
    method: str,
    ir: IntermediateRepresentation | None,
) -> str:
    """Zero-cost renderer for harness smoke tests.

    This is not evidence of model quality. It exists so the eval pipeline can be
    tested offline. Decision runs should use `--backend anthropic`.
    """
    persona = planned.persona
    turn = planned.turn
    occupation = persona.identity.occupation.lower()
    marker_phrase = ", ".join(turn.expected_markers[:4])
    prompt_l = turn.prompt.lower()

    if method in {"stale_ir", "shuffled_ir", "contradictory_ir"} and ir is not None:
        # Let bad IR visibly damage the response so the scoring pipeline can
        # verify that negative controls are detected.
        comp = ir.response_structure.competence
    elif ir is not None:
        comp = ir.response_structure.competence
    else:
        comp = 0.5

    if method == "prompt_only":
        prefix = f"As {persona.label}, I would look at this through my usual lens."
    elif method == "no_ir_ablation":
        prefix = "Without specific turn parameters, I can only give a broad persona-consistent reaction."
    else:
        prefix = "My response depends on the context and the limits of my actual experience."

    if "lawyer" in occupation or "law" in occupation:
        if "legal advice" in prompt_l:
            body = (
                "This should remain general discussion, not advice on specific facts. "
                "A proper answer needs counsel, scope, conflicts checks, and the full record."
            )
        elif comp < 0.35:
            body = (
                "That is outside my core practice, so I would be careful not to overstate it. "
                f"The useful adjacent issues are {marker_phrase}, but a specialist should handle the details."
            )
        else:
            body = (
                f"The material issues are {marker_phrase}. I would structure the analysis, identify red flags, "
                "and separate legal risk from financial or operational judgment."
            )
    elif "engineer" in occupation:
        body = (
            f"I would focus on {marker_phrase}. Engineers will adopt this only if it reduces review noise, "
            "fits CI and pull-request workflow, and earns trust through accurate suggestions."
        )
    elif "social" in occupation:
        body = (
            f"I would raise {marker_phrase}. Frontline staff need privacy, trust, training, and room for human context."
        )
    elif "chef" in occupation:
        body = (
            f"I would care about {marker_phrase}. If it slows service during a rush or ignores supplier realities, "
            "managers will work around it."
        )
    else:
        body = f"The main concerns are {marker_phrase}."

    if method in {"shuffled_ir", "contradictory_ir"}:
        body += " Some of this may not map cleanly to the role or situation."

    if turn.expected_competence == "low":
        return f"{prefix} {body}"
    return f"{prefix} {body} The practical research signal is what this person would object to first, what would build trust, and what would make adoption fail."


def call_anthropic(
    client: Any,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
) -> tuple[str, dict[str, int], float]:
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
    elapsed = time.time() - t0
    return (
        resp.content[0].text,
        {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
        elapsed,
    )


def build_user_message(
    turn: TurnSpec,
    ir: IntermediateRepresentation | None,
    rich_ir: bool = False,
) -> str:
    if ir is None:
        return turn.prompt
    block = format_rich_ir_block(ir) if rich_ir else format_ir_block(ir)
    return f"{block}\n\nUSER QUESTION:\n{turn.prompt}"


def generate_method_results(
    method: str,
    scenarios: list[ScenarioSpec],
    catalog: list[PlannedTurn],
    backend: str,
    model: str,
    repeat: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    catalog_by_key = {
        (p.scenario.id, p.turn.id, p.turn_index): p
        for p in catalog
    }
    anthropic_client = None
    if backend == "anthropic":
        import anthropic

        anthropic_client = anthropic.Anthropic()

    for scenario in scenarios:
        engine_provider = "template" if backend == "template" else "anthropic"
        engine_adapter = None
        if backend == "anthropic":
            engine_adapter = AnthropicAdapter(model=model)
        engine = PersonaEngine.from_yaml(
            str(ROOT / scenario.persona_path),
            llm_provider=engine_provider,
            adapter=engine_adapter,
            seed=42 + repeat,
        )
        persona = engine.persona
        rich_schema = method in RICH_SCHEMA_METHODS
        system_prompt = persona_system_prompt(
            persona,
            include_schema=method != "prompt_only",
            rich_schema=rich_schema,
        )
        messages: list[dict[str, str]] = []
        prior_ir: IntermediateRepresentation | None = None

        for turn_index, turn in enumerate(scenario.turns):
            planned = catalog_by_key[(scenario.id, turn.id, turn_index)]
            correct_ir = engine.plan(turn.prompt) if method != "engine_thin" else None

            effective_ir: IntermediateRepresentation | None
            if method == "prompt_only":
                effective_ir = None
            elif method == "no_ir_ablation":
                effective_ir = None
            elif method == "stale_ir":
                effective_ir = prior_ir.model_copy(deep=True) if prior_ir else correct_ir
            elif method == "contradictory_ir":
                effective_ir = contradictory_ir(correct_ir) if correct_ir else None
            elif method == "shuffled_ir":
                effective_ir = pick_shuffled_ir(catalog, planned)
            elif method in {"schema_driven", "schema_rich"}:
                effective_ir = correct_ir
            elif method == "engine_thin":
                effective_ir = None
            else:
                raise ValueError(f"Unknown method: {method}")

            started = time.time()
            usage = {"input_tokens": 0, "output_tokens": 0}
            if method == "engine_thin":
                chat = engine.chat(turn.prompt)
                text = chat.text
                result_ir = chat.ir
                elapsed = time.time() - started
                usage["output_tokens"] = len(text.split())
            elif backend == "anthropic":
                user_message = build_user_message(turn, effective_ir, rich_ir=rich_schema)
                messages.append({"role": "user", "content": user_message})
                text, usage, elapsed = call_anthropic(
                    anthropic_client,
                    system_prompt,
                    messages,
                    model,
                    max_tokens,
                )
                messages.append({"role": "assistant", "content": text})
                result_ir = effective_ir
            else:
                text = deterministic_render(planned, method, effective_ir)
                elapsed = time.time() - started
                usage["output_tokens"] = len(text.split())
                if method != "prompt_only":
                    messages.append(
                        {
                            "role": "user",
                            "content": build_user_message(turn, effective_ir, rich_ir=rich_schema),
                        }
                    )
                    messages.append({"role": "assistant", "content": text})
                else:
                    messages.append({"role": "user", "content": turn.prompt})
                    messages.append({"role": "assistant", "content": text})
                result_ir = effective_ir

            if correct_ir is not None:
                prior_ir = correct_ir.model_copy(deep=True)

            item = {
                "method": method,
                "backend": backend,
                "model": model if backend == "anthropic" else "template",
                "repeat": repeat,
                "scenario_id": scenario.id,
                "comparison_group": scenario.comparison_group,
                "research_goal": scenario.research_goal,
                "persona_path": scenario.persona_path,
                "persona_id": persona.persona_id,
                "persona_label": persona.label,
                "turn_id": turn.id,
                "turn_index": turn_index,
                "category": turn.category,
                "prompt": turn.prompt,
                "response": text,
                "latency_s": round(elapsed, 4),
                "usage": usage,
                "expected": {
                    "competence": turn.expected_competence,
                    "claim": turn.expected_claim,
                    "markers": turn.expected_markers,
                    "forbidden_markers": turn.forbidden_markers,
                    "min_words": turn.min_words,
                    "max_words": turn.max_words,
                },
                "ir": summarize_ir(result_ir),
                "correct_ir": summarize_ir(correct_ir),
            }
            item["auto_scores"] = score_item(item)
            results.append(item)
    return results


def summarize_ir(ir: IntermediateRepresentation | None) -> dict[str, Any] | None:
    if ir is None:
        return None
    summary = {
        "context_type": ir.context_type,
        "competence": round(ir.response_structure.competence, 4),
        "confidence": round(ir.response_structure.confidence, 4),
        "formality": round(ir.communication_style.formality, 4),
        "directness": round(ir.communication_style.directness, 4),
        "tone": ir.communication_style.tone.value,
        "verbosity": ir.communication_style.verbosity.value,
        "claim_type": ir.knowledge_disclosure.knowledge_claim_type.value,
        "uncertainty_action": ir.knowledge_disclosure.uncertainty_action.value,
        "citations": len(ir.citations),
        "constraints": len(ir.safety_plan.active_constraints),
        "cannot_claim": list(ir.safety_plan.cannot_claim),
    }
    if ir.research:
        summary["research"] = ir.research.model_dump(mode="json")
    return summary


def band_match(value: float, expected: str) -> bool:
    if expected == "high":
        return value >= 0.65
    if expected == "medium":
        return 0.35 <= value < 0.65
    if expected == "low":
        return value < 0.35
    return True


def score_item(item: dict[str, Any]) -> dict[str, Any]:
    response = item["response"]
    response_l = response.lower()
    metrics = text_metrics(response)
    expected = item["expected"]

    expected_markers = [m.lower() for m in expected["markers"]]
    forbidden_markers = [m.lower() for m in expected["forbidden_markers"]]
    marker_hits = [m for m in expected_markers if marker_present(response_l, m)]
    forbidden_hits = [m for m in forbidden_markers if forbidden_marker_hit(response_l, m)]

    marker_score = len(marker_hits) / max(1, len(expected_markers))
    forbidden_score = 1.0 if not forbidden_hits else 0.0

    length_score = 1.0
    if expected["min_words"] is not None and metrics["words"] < expected["min_words"]:
        length_score = min(length_score, metrics["words"] / max(1, expected["min_words"]))
    if expected["max_words"] is not None and metrics["words"] > expected["max_words"]:
        length_score = min(length_score, expected["max_words"] / max(1, metrics["words"]))

    boundary_expected = item["category"] in {"boundary", "adversarial"} or expected["competence"] == "low"
    boundary_score = 1.0
    if boundary_expected:
        boundary_score = 1.0 if metrics["boundaries"] > 0 or metrics["hedges"] > 0 else 0.3
    if metrics["casual"] > 0 and "lawyer" in item["persona_label"].lower():
        boundary_score *= 0.6

    output_score = (
        marker_score * 0.35
        + forbidden_score * 0.30
        + length_score * 0.20
        + boundary_score * 0.15
    )

    ir_summary = item["ir"]
    if ir_summary is None:
        trace_presence_score = 0.0
        trace_correctness_score = 0.0
        ir_match = False
        claim_match = False
    else:
        ir_match = band_match(ir_summary["competence"], expected["competence"])
        claim_match = ir_summary["claim_type"] == expected["claim"]
        claim_compatible = claim_match or (
            expected["claim"] == "personal_experience"
            and ir_summary["claim_type"] in {"personal_experience", "general_common_knowledge"}
        )
        trace_presence_score = (
            min(0.70, ir_summary["citations"] / 40)
            + (0.30 if ir_summary["cannot_claim"] else 0.0)
        )
        trace_correctness_score = (
            (0.60 if ir_match else 0.0)
            + (0.40 if claim_compatible else 0.0)
        )

    safety_score = forbidden_score * 0.70 + boundary_score * 0.30
    causal_score = output_score * 0.70 + trace_correctness_score * 0.30
    product_score = (
        output_score * 0.60
        + trace_correctness_score * 0.30
        + safety_score * 0.10
    )

    return {
        **metrics,
        "marker_hits": marker_hits,
        "forbidden_hits": forbidden_hits,
        "marker_score": round(marker_score, 4),
        "forbidden_score": round(forbidden_score, 4),
        "length_score": round(length_score, 4),
        "boundary_score": round(boundary_score, 4),
        "output_score": round(output_score, 4),
        "trace_presence_score": round(trace_presence_score, 4),
        "trace_correctness_score": round(trace_correctness_score, 4),
        "traceability_score": round(trace_correctness_score, 4),
        "safety_score": round(safety_score, 4),
        "causal_score": round(causal_score, 4),
        "product_score": round(product_score, 4),
        "ir_competence_matches_expected": ir_match,
        "ir_claim_matches_expected": claim_match,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    sx = statistics.stdev(xs)
    sy = statistics.stdev(ys)
    if sx == 0 or sy == 0:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
    return cov / (sx * sy)


def token_set(text: str) -> set[str]:
    return {w for w in words(text) if len(w) > 3 and w not in STOPWORDS}


def jaccard_distance(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / len(sa | sb))


def segment_differentiation(results: list[dict[str, Any]], method: str) -> float | None:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for item in results:
        if item["method"] != method:
            continue
        key = (item["comparison_group"], item["turn_id"], item["repeat"])
        grouped.setdefault(key, []).append(item)

    distances: list[float] = []
    for group_items in grouped.values():
        if len(group_items) < 2:
            continue
        for i, left in enumerate(group_items):
            for right in group_items[i + 1 :]:
                if left["persona_id"] != right["persona_id"]:
                    distances.append(jaccard_distance(left["response"], right["response"]))
    return round(statistics.mean(distances), 4) if distances else None


def matrix_status(
    results: list[dict[str, Any]],
    required_methods: tuple[str, ...] = REQUIRED_DECISION_METHODS,
) -> dict[str, Any]:
    keys_by_method: dict[str, set[tuple[Any, ...]]] = {
        method: set() for method in required_methods
    }
    counts: dict[tuple[str, tuple[Any, ...]], int] = {}
    extra_methods: set[str] = set()

    for item in results:
        method = item["method"]
        key = (
            item["repeat"],
            item["scenario_id"],
            item["turn_id"],
            item["turn_index"],
            item["persona_id"],
        )
        if method in keys_by_method:
            keys_by_method[method].add(key)
            counts[(method, key)] = counts.get((method, key), 0) + 1
        else:
            extra_methods.add(method)

    missing_methods = [
        method for method in required_methods if not keys_by_method[method]
    ]
    expected_keys: set[tuple[Any, ...]] = set()
    for keys in keys_by_method.values():
        expected_keys.update(keys)

    missing_cells: list[str] = []
    for method in required_methods:
        for key in expected_keys - keys_by_method[method]:
            missing_cells.append(f"{method}:{key}")

    duplicate_cells = [
        f"{method}:{key}"
        for (method, key), count in counts.items()
        if count > 1
    ]

    complete = (
        bool(expected_keys)
        and not missing_methods
        and not missing_cells
        and not duplicate_cells
    )
    return {
        "complete": complete,
        "required_methods": list(required_methods),
        "missing_methods": missing_methods,
        "missing_cell_count": len(missing_cells),
        "missing_cells_sample": missing_cells[:10],
        "duplicate_cells": duplicate_cells[:10],
        "extra_methods": sorted(extra_methods),
        "expected_cell_count_per_method": len(expected_keys),
    }


def summarize(
    results: list[dict[str, Any]],
    *,
    run_type: str = "smoke",
    required_methods: tuple[str, ...] = REQUIRED_DECISION_METHODS,
) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_method.setdefault(item["method"], []).append(item)

    method_summary: dict[str, Any] = {}
    for method, items in by_method.items():
        scores = [i["auto_scores"] for i in items]
        ir_items = [i for i in items if i["ir"]]
        comp = [i["ir"]["competence"] for i in ir_items]
        word_counts = [i["auto_scores"]["words"] for i in ir_items]
        method_summary[method] = {
            "n": len(items),
            "avg_output_score": round(statistics.mean(s["output_score"] for s in scores), 4),
            "avg_trace_presence_score": round(statistics.mean(s["trace_presence_score"] for s in scores), 4),
            "avg_trace_correctness_score": round(statistics.mean(s["trace_correctness_score"] for s in scores), 4),
            "avg_traceability_score": round(statistics.mean(s["traceability_score"] for s in scores), 4),
            "avg_safety_score": round(statistics.mean(s["safety_score"] for s in scores), 4),
            "avg_causal_score": round(statistics.mean(s["causal_score"] for s in scores), 4),
            "avg_product_score": round(statistics.mean(s["product_score"] for s in scores), 4),
            "forbidden_hits": sum(len(s["forbidden_hits"]) for s in scores),
            "avg_words": round(statistics.mean(s["words"] for s in scores), 1),
            "boundary_markers": sum(s["boundaries"] for s in scores),
            "casual_markers": sum(s["casual"] for s in scores),
            "ir_competence_match_rate": round(
                sum(1 for s in scores if s["ir_competence_matches_expected"]) / len(scores),
                4,
            ),
            "ir_claim_match_rate": round(
                sum(1 for s in scores if s["ir_claim_matches_expected"]) / len(scores),
                4,
            ),
            "competence_word_corr": (
                round(pearson(comp, word_counts), 4)
                if len(comp) >= 2 and pearson(comp, word_counts) is not None
                else None
            ),
            "segment_differentiation": segment_differentiation(results, method),
        }

    matrix = matrix_status(results, required_methods=required_methods)
    gates = decision_gates(method_summary, matrix)
    return {
        "run_type": run_type,
        "matrix_status": matrix,
        "method_summary": method_summary,
        "decision_gates": gates,
    }


def decision_gates(
    method_summary: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    def score(method: str) -> float | None:
        return method_summary.get(method, {}).get("avg_product_score")

    prompt = score("prompt_only")
    no_ir = score("no_ir_ablation")
    negative_scores = [score(method) for method in NEGATIVE_CONTROL_METHODS]
    negative = max(s for s in negative_scores if s is not None) if all(
        s is not None for s in negative_scores
    ) else None

    def primary_gates(primary: str) -> dict[str, bool]:
        schema = score(primary)
        has_required = (
            matrix["complete"]
            and schema is not None
            and prompt is not None
            and no_ir is not None
            and negative is not None
        )
        gates = {
            f"{primary}_matrix_complete": matrix["complete"],
            f"{primary}_beats_prompt_only": bool(
                has_required and schema > prompt + 0.03
            ),
            f"{primary}_beats_no_ir": bool(
                has_required and schema > no_ir + 0.03
            ),
            f"{primary}_beats_negative_controls": bool(
                has_required and schema > negative + 0.03
            ),
            f"{primary}_boundary_violations_no_worse_than_prompt": (
                has_required
                and
                method_summary.get(primary, {}).get("forbidden_hits", 999)
                <= method_summary.get("prompt_only", {}).get("forbidden_hits", 999)
            ),
            f"{primary}_segment_diff_no_worse_than_prompt": (
                has_required
                and
                (method_summary.get(primary, {}).get("segment_differentiation") or 0)
                >= (method_summary.get("prompt_only", {}).get("segment_differentiation") or 0)
            ),
        }
        gates[f"{primary}_proceed_signal"] = all(gates.values())
        return gates

    gates = {
        **primary_gates("schema_driven"),
        **primary_gates("schema_rich"),
    }
    gates["proceed_signal"] = (
        gates["schema_driven_proceed_signal"] or gates["schema_rich_proceed_signal"]
    )
    gates["matrix_complete"] = matrix["complete"]
    return gates


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    rng = random.Random(42)
    review_items = copy.deepcopy(results)
    rng.shuffle(review_items)
    key: dict[str, Any] = {}
    with (out_dir / "blind_review.jsonl").open("w") as fh:
        for item in review_items:
            raw_id = (
                f"{item['method']}|{item['repeat']}|{item['scenario_id']}|"
                f"{item['turn_id']}|{item['persona_id']}"
            )
            sample_id = hashlib.sha256(raw_id.encode()).hexdigest()[:12]
            key[sample_id] = {
                "method": item["method"],
                "scenario_id": item["scenario_id"],
                "turn_id": item["turn_id"],
                "persona_id": item["persona_id"],
                "repeat": item["repeat"],
            }
            review_record = {
                "sample_id": sample_id,
                "persona_label": item["persona_label"],
                "research_goal": item["research_goal"],
                "category": item["category"],
                "prompt": item["prompt"],
                "response": item["response"],
                "rubric": {
                    "expected_markers": item["expected"]["markers"],
                    "forbidden_markers": item["expected"]["forbidden_markers"],
                    "review_dimensions_1_to_5": [
                        "behavioral_fidelity",
                        "research_usefulness",
                        "boundary_adherence",
                        "segment_specificity",
                        "consistency",
                    ],
                },
            }
            fh.write(json.dumps(review_record) + "\n")
    (out_dir / "blind_review_key.json").write_text(json.dumps(key, indent=2))
    (out_dir / "report.md").write_text(render_report(summary))


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Product Value Eval Report",
        "",
        f"Run type: `{summary.get('run_type', 'smoke')}`",
        "",
        "## Method Summary",
        "",
        "| Method | Product | Output | Trace | Forbidden | IR Match | Segment Diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, data in sorted(summary["method_summary"].items()):
        lines.append(
            "| {method} | {product:.3f} | {output:.3f} | {trace:.3f} | {forbidden} | {ir:.2%} | {seg} |".format(
                method=method,
                product=data["avg_product_score"],
                output=data["avg_output_score"],
                trace=data["avg_traceability_score"],
                forbidden=data["forbidden_hits"],
                ir=data["ir_competence_match_rate"],
                seg=data["segment_differentiation"],
            )
        )

    matrix = summary.get("matrix_status", {})
    lines.extend(
        [
            "",
            "## Matrix Status",
            "",
            f"- complete: {'PASS' if matrix.get('complete') else 'FAIL'}",
            f"- required_methods: {', '.join(matrix.get('required_methods', []))}",
            f"- missing_methods: {', '.join(matrix.get('missing_methods', [])) or 'none'}",
            f"- missing_cell_count: {matrix.get('missing_cell_count', 0)}",
            f"- duplicate_cells: {len(matrix.get('duplicate_cells', []))}",
        ]
    )

    lines.extend(["", "## Decision Gates", ""])
    for gate, passed in summary["decision_gates"].items():
        lines.append(f"- {gate}: {'PASS' if passed else 'FAIL'}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this report as a triage signal, not as the final customer proof. "
            "The blind review export should be scored by humans before making a product decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--backend", choices=["template", "anthropic"], default="template")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--run-type", choices=["smoke", "decision"], default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for --backend anthropic")

    scenarios = load_scenarios(args.scenarios)
    catalog = build_ir_catalog(scenarios)
    all_results: list[dict[str, Any]] = []

    for repeat in range(args.repeats):
        for method in args.methods:
            if method not in METHODS:
                raise SystemExit(f"Unknown method '{method}'. Valid: {', '.join(METHODS)}")
            print(f"Running {method} repeat={repeat + 1}/{args.repeats} backend={args.backend}")
            all_results.extend(
                generate_method_results(
                    method=method,
                    scenarios=scenarios,
                    catalog=catalog,
                    backend=args.backend,
                    model=args.model,
                    repeat=repeat,
                    max_tokens=args.max_tokens,
                )
            )

    summary = summarize(all_results, run_type=args.run_type)
    write_outputs(all_results, summary, args.out)
    print(json.dumps(summary["decision_gates"], indent=2))
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
