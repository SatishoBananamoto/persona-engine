"""
Enterprise research IR builder.

This module keeps the research signal deterministic and compact. It does not
try to predict a market; it records how directly the simulated stakeholder's
work touches the rollout and what objections or adoption risks follow.
"""

from __future__ import annotations

from persona_engine.planner.trace_context import TraceContext
from persona_engine.schema.ir_schema import ResearchIR
from persona_engine.schema.persona_schema import Persona


SOFTWARE_TERMS = [
    "software", "engineer", "developer", "pull request", "code review",
    "review", "ci", "repository", "backend", "devops", "architecture",
]
CARE_TERMS = [
    "social worker", "case note", "client", "frontline", "care",
    "welfare", "family services", "safety", "privacy", "community",
]
KITCHEN_TERMS = [
    "chef", "kitchen", "restaurant", "service", "shift", "supplier",
    "staff", "rush", "menu", "manager",
]
LEGAL_TERMS = [
    "lawyer", "legal", "counsel", "contract", "client", "privilege",
    "m&a", "due diligence", "compliance",
]


def detect_research_focus(user_input: str) -> str:
    """Return the narrow research focus requested by the turn."""
    text = user_input.lower()
    if any(term in text for term in ("quietly ignore", "adoption", "blocker", "blockers")):
        return "adoption_blockers"
    if any(term in text for term in ("trust", "would make", "credible")):
        return "trust_conditions"
    if any(term in text for term in ("workaround", "route around", "bypass")):
        return "workaround_risk"
    if any(term in text for term in ("concern", "concerns", "objection", "objections", "raise")):
        return "policy_reaction"
    return "general_research"


def build_research_ir(
    persona: Persona,
    user_input: str,
    ctx: TraceContext | None = None,
) -> ResearchIR:
    """Build a compact enterprise-research contract for the current turn."""
    text = user_input.lower()
    occupation = getattr(persona.identity, "occupation", "") or "stakeholder"
    role_text = _persona_role_text(persona)
    role_family = _role_family(role_text, text)
    workflow_exposure = _workflow_exposure(role_family, text)
    claim_basis = _claim_basis(workflow_exposure)
    focus = detect_research_focus(user_input)

    likely_objections = _likely_objections(role_family, focus)
    trust_conditions = _trust_conditions(role_family)
    adoption_blockers = _adoption_blockers(role_family)
    workaround_risk = _workaround_risk(workflow_exposure, focus)

    research = ResearchIR(
        focus=focus,  # type: ignore[arg-type]
        stakeholder_role=occupation,
        workflow_exposure=workflow_exposure,
        claim_basis=claim_basis,  # type: ignore[arg-type]
        likely_objections=likely_objections,
        trust_conditions=trust_conditions,
        adoption_blockers=adoption_blockers,
        workaround_risk=workaround_risk,
        evidence_boundary=_evidence_boundary(claim_basis),
    )

    if ctx:
        ctx.add_basic_citation(
            source_type="rule",
            source_id="research_ir_focus",
            effect=f"Enterprise research focus: {focus}",
            weight=0.9,
        )
        ctx.add_basic_citation(
            source_type="state",
            source_id="research_workflow_exposure",
            effect=(
                f"Workflow exposure {workflow_exposure:.2f} for "
                f"{occupation} in {role_family} rollout context"
            ),
            weight=0.9,
        )
        ctx.add_basic_citation(
            source_type="rule",
            source_id="research_claim_basis",
            effect=f"Research claim basis: {claim_basis}",
            weight=0.9,
        )

    return research


def _persona_role_text(persona: Persona) -> str:
    domains = " ".join(
        " ".join([kd.domain, *getattr(kd, "subdomains", [])])
        for kd in persona.knowledge_domains
    )
    identity = persona.identity
    return " ".join(
        [
            identity.occupation,
            identity.background,
            domains,
        ]
    ).lower()


def _role_family(role_text: str, user_text: str) -> str:
    combined = f"{role_text} {user_text}"
    if _contains_any(combined, SOFTWARE_TERMS):
        return "software"
    if _contains_any(combined, CARE_TERMS):
        return "care"
    if _contains_any(combined, KITCHEN_TERMS):
        return "kitchen"
    if _contains_any(combined, LEGAL_TERMS):
        return "legal"
    return "general_workplace"


def _workflow_exposure(role_family: str, user_text: str) -> float:
    if role_family == "software":
        return 0.88 if _contains_any(user_text, SOFTWARE_TERMS) else 0.72
    if role_family == "care":
        return 0.74 if _contains_any(user_text, CARE_TERMS) else 0.58
    if role_family == "kitchen":
        return 0.84 if _contains_any(user_text, KITCHEN_TERMS) else 0.62
    if role_family == "legal":
        return 0.76 if _contains_any(user_text, LEGAL_TERMS) else 0.56
    if any(term in user_text for term in ("employees", "staff", "team", "managers")):
        return 0.48
    return 0.28


def _claim_basis(workflow_exposure: float) -> str:
    if workflow_exposure >= 0.70:
        return "direct_workflow_experience"
    if workflow_exposure >= 0.45:
        return "adjacent_professional_experience"
    if workflow_exposure >= 0.25:
        return "general_workplace_experience"
    return "low_basis"


def _likely_objections(role_family: str, focus: str) -> list[str]:
    common = ["unclear accountability", "extra workflow friction"]
    role_specific = {
        "software": ["false positives in review", "security of source code", "loss of reviewer ownership"],
        "care": ["client privacy", "loss of case context", "safety risk from shallow notes"],
        "kitchen": ["service-rush timing", "supplier-note accuracy", "staff scheduling reality"],
        "legal": ["client confidentiality", "privilege boundaries", "liability for generated language"],
        "general_workplace": ["monitoring concerns", "unclear benefit", "training burden"],
    }
    focus_specific = {
        "adoption_blockers": ["people may comply superficially"],
        "trust_conditions": ["trust requires visible error handling"],
        "workaround_risk": ["shadow process risk"],
        "policy_reaction": ["mandatory use will invite pushback"],
        "general_research": [],
    }
    return _unique([
        *role_specific.get(role_family, role_specific["general_workplace"]),
        *common,
        *focus_specific.get(focus, []),
    ])[:5]


def _trust_conditions(role_family: str) -> list[str]:
    role_specific = {
        "software": ["low false-positive rate", "clear security boundaries", "human reviewer keeps final call"],
        "care": ["privacy safeguards", "client-context controls", "human override without penalty"],
        "kitchen": ["fast enough during service", "works with supplier reality", "manager override remains normal"],
        "legal": ["confidentiality controls", "clear scope limits", "review trail for accountability"],
        "general_workplace": ["transparent purpose", "training time", "visible correction path"],
    }
    return role_specific.get(role_family, role_specific["general_workplace"])


def _adoption_blockers(role_family: str) -> list[str]:
    role_specific = {
        "software": ["CI noise", "review latency", "security review friction", "poor fit with existing PR flow"],
        "care": ["documentation burden", "privacy anxiety", "training gaps", "safety escalation ambiguity"],
        "kitchen": ["time pressure", "rush periods", "staff buy-in", "supplier exceptions"],
        "legal": ["client-specific nuance", "privilege concerns", "approval bottlenecks", "liability ambiguity"],
        "general_workplace": ["workflow disruption", "low trust", "unclear ownership", "managerial pressure"],
    }
    return role_specific.get(role_family, role_specific["general_workplace"])


def _workaround_risk(workflow_exposure: float, focus: str) -> float:
    risk = 0.35 + workflow_exposure * 0.45
    if focus in {"adoption_blockers", "workaround_risk"}:
        risk += 0.10
    if focus == "trust_conditions":
        risk -= 0.05
    return max(0.0, min(1.0, round(risk, 3)))


def _evidence_boundary(claim_basis: str) -> str:
    if claim_basis == "direct_workflow_experience":
        return "simulated direct stakeholder experience, not measured market evidence"
    if claim_basis == "adjacent_professional_experience":
        return "simulated adjacent professional experience, not measured market evidence"
    if claim_basis == "general_workplace_experience":
        return "simulated general workplace hypothesis, not measured market evidence"
    return "low-basis simulated hypothesis; use only for follow-up research design"


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
