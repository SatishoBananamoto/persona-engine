"""
Stage 2: Interpretation — topic relevance, bias, state evolution, intent, domain, expert eligibility.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from persona_engine.planner.domain_detection import (
    compute_topic_relevance,
    detect_domain,
)
from persona_engine.planner.engine_config import DEFAULT_CONFIG
from persona_engine.planner.intent_analyzer import analyze_intent
from persona_engine.planner.stages.stage_results import InterpretationResult
from persona_engine.planner.trace_context import TraceContext

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner

logger = logging.getLogger(__name__)

DEFAULT_TOPIC_RELEVANCE = DEFAULT_CONFIG.default_topic_relevance
EXPERT_THRESHOLD = DEFAULT_CONFIG.expert_threshold
EVIDENCE_STRESS_THRESHOLD = DEFAULT_CONFIG.evidence_stress_threshold


class InterpretationStage:
    """Analyses topic, evolves state, detects intent/domain, and determines expert eligibility."""

    def __init__(self, planner: TurnPlanner) -> None:
        self.planner = planner

    def execute(
        self,
        context: ConversationContext,
        ctx: TraceContext,
    ) -> InterpretationResult:
        """Run the interpretation stage."""
        p = self.planner

        # Topic relevance
        persona_domains: list[dict] = []
        if p.persona.knowledge_domains:
            for kd in p.persona.knowledge_domains:
                persona_domains.append({
                    "domain": kd.domain,
                    "proficiency": kd.proficiency,
                    "subdomains": getattr(kd, "subdomains", None) or [],
                })

        persona_goals: list[dict] = []
        if hasattr(p.persona, "primary_goals"):
            for g in p.persona.primary_goals:
                persona_goals.append({"goal": getattr(g, "goal", ""), "weight": getattr(g, "weight", 1.0)})
        if hasattr(p.persona, "secondary_goals"):
            for g in p.persona.secondary_goals:
                persona_goals.append({"goal": getattr(g, "goal", ""), "weight": getattr(g, "weight", 0.5)})

        topic_relevance = compute_topic_relevance(
            user_input=context.user_input,
            persona_domains=persona_domains,
            persona_goals=persona_goals,
            ctx=ctx,
            default_relevance=DEFAULT_TOPIC_RELEVANCE,
        )

        # Intent analysis
        inferred_mode, inferred_goal, user_intent, needs_clarification = analyze_intent(
            user_input=context.user_input,
            current_mode=context.interaction_mode,
            current_goal=context.goal,
            ctx=ctx,
        )
        context.interaction_mode = inferred_mode
        context.goal = inferred_goal

        # Domain + proficiency
        domain = context.domain or self.detect_domain(context.user_input, ctx=ctx)
        proficiency = self.get_domain_proficiency(domain)
        ctx.add_basic_citation(
            source_type="state",
            source_id="domain_proficiency",
            effect=f"Domain '{domain}' proficiency: {proficiency:.2f}",
            weight=1.0,
        )

        # Expert eligibility
        is_domain_specific = any(
            kd.domain.lower() == domain.lower() for kd in p.persona.knowledge_domains
        )
        expert_threshold = getattr(p.persona.claim_policy, "expert_threshold", EXPERT_THRESHOLD)
        expert_allowed = is_domain_specific and (proficiency >= expert_threshold)
        ctx.add_basic_citation(
            source_type="rule",
            source_id="expert_eligibility",
            effect=f"Expert allowed: {expert_allowed} (is_domain={is_domain_specific}, prof={proficiency:.2f}, thresh={expert_threshold:.2f})",
            weight=1.0,
        )

        # Phase R1: Check decision policies
        matched_policy = p.rules.check_decision_policy(context.user_input)
        policy_modifications: dict[str, Any] = {}
        if matched_policy:
            policy_modifications = p.rules.apply_decision_policy(matched_policy)
            ctx.add_basic_citation(
                source_type="rule",
                source_id="decision_policy",
                effect=(
                    f"Decision policy matched: '{matched_policy.condition}' "
                    f"→ approach='{matched_policy.approach}'"
                ),
                weight=0.9,
            )

        # Bias modifiers (now with proficiency for DK bias)
        p._current_bias_modifiers = p.bias_simulator.compute_modifiers(
            user_input=context.user_input,
            value_alignment=topic_relevance,
            ctx=ctx,
            proficiency=proficiency,
        )

        return InterpretationResult(
            topic_relevance=topic_relevance,
            persona_domains=persona_domains,
            domain=domain,
            proficiency=proficiency,
            expert_allowed=expert_allowed,
            user_intent=user_intent,
            needs_clarification=needs_clarification,
            policy_modifications=policy_modifications,
        )

    def detect_domain(self, user_input: str, ctx: TraceContext | None = None) -> str:
        """Detect domain from user input using keyword scoring."""
        p = self.planner
        persona_domains = [
            {"domain": kd.domain, "proficiency": kd.proficiency, "subdomains": kd.subdomains}
            for kd in p.persona.knowledge_domains
        ] if p.persona.knowledge_domains else []

        domain, _score = detect_domain(
            user_input=user_input,
            persona_domains=persona_domains,
            ctx=ctx,
        )
        return domain

    def get_domain_proficiency(self, domain: str) -> float:
        """Get persona's proficiency in domain."""
        p = self.planner
        for knowledge in p.persona.knowledge_domains:
            if knowledge.domain.lower() == domain.lower():
                return knowledge.proficiency
        return DEFAULT_CONFIG.default_proficiency
