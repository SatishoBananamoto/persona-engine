"""
Stage 4: Knowledge & Safety — disclosure, uncertainty action, claim type,
response patterns, and constraint validation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from persona_engine.behavioral import (
    MAX_BIAS_IMPACT,
    apply_response_pattern_safely,
    infer_knowledge_claim_type,
    resolve_uncertainty_action,
    validate_stance_against_invariants,
)
from persona_engine.behavioral.social_cognition import (
    AdaptationDirectives,
    SchemaEffect,
)
from persona_engine.planner.engine_config import DEFAULT_CONFIG
from persona_engine.planner.stages.stage_results import (
    BehavioralMetricsResult,
    InterpretationResult,
    KnowledgeSafetyResult,
)
from persona_engine.planner.trace_context import TraceContext, clamp01
from persona_engine.schema.ir_schema import (
    Citation,
    InteractionMode,
    KnowledgeClaimType,
    UncertaintyAction,
)

if TYPE_CHECKING:
    from persona_engine.planner.turn_planner import ConversationContext, TurnPlanner

logger = logging.getLogger(__name__)

# Config aliases
PERSONALITY_FIELD_INERTIA = DEFAULT_CONFIG.personality_field_inertia
EVIDENCE_STRESS_THRESHOLD = DEFAULT_CONFIG.evidence_stress_threshold
TIME_PRESSURE_TURN_THRESHOLD = DEFAULT_CONFIG.time_pressure_turn_threshold
TIME_PRESSURE_PER_TURN = DEFAULT_CONFIG.time_pressure_per_turn
TIME_PRESSURE_MAX_BUILDUP = DEFAULT_CONFIG.time_pressure_max_buildup


def _smooth(prev: float, new: float, alpha: float) -> float:
    """Cross-turn inertia smoothing."""
    return prev * alpha + new * (1 - alpha)


class KnowledgeSafetyStage:
    """Computes disclosure, uncertainty action, claim type, applies patterns,
    and validates constraints."""

    def __init__(self, planner: TurnPlanner) -> None:
        self.planner = planner

    def execute(
        self,
        context: ConversationContext,
        ctx: TraceContext,
        foundation: InterpretationResult,
        metrics: BehavioralMetricsResult,
    ) -> KnowledgeSafetyResult:
        """Run the knowledge & safety stage."""
        p = self.planner
        proficiency = foundation.proficiency
        domain = foundation.domain
        confidence = metrics.confidence
        stance = metrics.stance
        rationale = metrics.rationale

        # Disclosure (+ cross-turn smoothing)
        disclosure_level = self.compute_disclosure(context.topic_signature, ctx)
        if p._prior_snapshot:
            before_disc = disclosure_level
            disclosure_level = _smooth(p._prior_snapshot.disclosure, disclosure_level, PERSONALITY_FIELD_INERTIA)
            if abs(disclosure_level - before_disc) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="knowledge_disclosure.disclosure_level",
                    operation="blend",
                    before=before_disc,
                    after=disclosure_level,
                    effect=f"Cross-turn inertia: {before_disc:.3f} → {disclosure_level:.3f}",
                    weight=0.7,
                    reason=f"inertia={PERSONALITY_FIELD_INERTIA}",
                )

        # Apply empathy gap bias to disclosure
        disclosure_bias = p.bias_simulator.get_total_modifier_for_field(
            p._current_bias_modifiers,
            "knowledge_disclosure.disclosure_level"
        )
        if abs(disclosure_bias) > 0.001:
            before_bias = disclosure_level
            disclosure_level = max(0.0, min(1.0, disclosure_level + disclosure_bias))
            ctx.num(
                source_type="trait",
                source_id="empathy_gap_bias",
                target_field="knowledge_disclosure.disclosure_level",
                operation="add",
                before=before_bias,
                after=disclosure_level,
                effect=f"Empathy gap bias: disclosure {disclosure_bias:+.3f}",
                weight=min(abs(disclosure_bias) / MAX_BIAS_IMPACT, 1.0),
                reason=f"empathy_gap={disclosure_bias:+.3f}"
            )

        # Schema validation disclosure boost
        schema_effect: SchemaEffect | None = metrics.schema_effect
        if schema_effect and abs(schema_effect.disclosure_modifier) > 0.001:
            before_sd = disclosure_level
            disclosure_level = max(0.0, min(1.0, disclosure_level + schema_effect.disclosure_modifier))
            ctx.num(
                source_type="rule",
                source_id="self_schema_disclosure",
                target_field="knowledge_disclosure.disclosure_level",
                operation="add",
                before=before_sd,
                after=disclosure_level,
                effect=f"Schema validation: disclosure {schema_effect.disclosure_modifier:+.3f}",
                weight=0.6,
            )

        # Disclosure reciprocity
        adaptation: AdaptationDirectives | None = metrics.adaptation
        if adaptation and adaptation.disclosure_reciprocity > 0.01:
            before_dr = disclosure_level
            disclosure_level = max(0.0, min(1.0, disclosure_level + adaptation.disclosure_reciprocity))
            ctx.num(
                source_type="rule",
                source_id="disclosure_reciprocity",
                target_field="knowledge_disclosure.disclosure_level",
                operation="add",
                before=before_dr,
                after=disclosure_level,
                effect=f"Disclosure reciprocity: {adaptation.disclosure_reciprocity:+.3f}",
                weight=0.6,
            )

        # Uncertainty action
        time_pressure = self.compute_time_pressure(
            context.interaction_mode, context.turn_number, ctx
        )
        resolver_citations: list[Citation] = []
        uncertainty_action = resolve_uncertainty_action(
            proficiency=proficiency,
            confidence=confidence,
            risk_tolerance=p.cognitive.style.risk_tolerance,
            need_for_closure=p.cognitive.style.need_for_closure,
            time_pressure=time_pressure,
            claim_policy_lookup_behavior=p.persona.claim_policy.lookup_behavior,
            citations=resolver_citations,
            stress=p.state.get_stress(),
            fatigue=p.state.get_fatigue(),
        )
        for cite in resolver_citations:
            ctx.citations.append(cite)

        ctx.enum(
            source_type="rule",
            source_id="uncertainty_resolver",
            target_field="knowledge_disclosure.uncertainty_action",
            operation="set",
            before="none",
            after=uncertainty_action.value,
            effect=f"Uncertainty action resolved: {uncertainty_action.value}",
            weight=1.0,
        )

        # Claim type
        knowledge_claim_type = self.infer_claim_type(
            proficiency, uncertainty_action, domain, user_input=context.user_input
        )
        claim_enum = KnowledgeClaimType(knowledge_claim_type)
        ctx.enum(
            source_type="rule",
            source_id="claim_type_inference",
            target_field="knowledge_disclosure.knowledge_claim_type",
            operation="set",
            before="none",
            after=claim_enum.value,
            effect=f"Knowledge claim type inferred: {claim_enum.value}",
            weight=1.0,
        )

        # Response patterns
        self.apply_patterns_safely(context, disclosure_level, ctx)

        # Constraint validation
        validation = validate_stance_against_invariants(
            stance,
            rationale,
            p.persona.invariants.identity_facts,
            p.persona.invariants.cannot_claim,
            must_avoid=p.persona.invariants.must_avoid,
        )
        if not validation["valid"]:
            ctx.activate_constraint("invariants")
            for violation in validation["violations"]:
                ctx.add_basic_citation(
                    source_type="rule",
                    source_id="invariant_violation",
                    effect=f"VIOLATION: {violation['message']}",
                    weight=1.0,
                )

        return KnowledgeSafetyResult(
            disclosure_level=disclosure_level,
            uncertainty_action=uncertainty_action,
            claim_enum=claim_enum,
        )

    # ========================================================================
    # Disclosure
    # ========================================================================

    def compute_disclosure(self, topic_signature: str, ctx: TraceContext) -> float:
        """Compute disclosure level with canonical composition.

        Sequence: base → trait → state → trust → privacy clamp → topic clamp → bounds
        """
        p = self.planner

        disclosure = ctx.base(
            field_name="communication.disclosure",
            target_field="knowledge_disclosure.disclosure_level",
            value=p.persona.disclosure_policy.base_openness,
            effect=f"Base disclosure from persona policy ({p.persona.disclosure_policy.base_openness:.2f})"
        )

        # Trait modifier (extraversion)
        extraversion_mod = p.traits.get_self_disclosure_modifier()
        disclosure = ctx.num(
            source_type="trait",
            source_id="extraversion",
            target_field="knowledge_disclosure.disclosure_level",
            operation="add",
            before=disclosure,
            after=disclosure + extraversion_mod,
            effect=f"Extraversion modifier: {extraversion_mod:+.2f}",
            weight=0.7,
            reason=f"E={p.persona.psychology.big_five.extraversion:.2f}"
        )

        # State modifier
        state_mod = p.state.get_disclosure_modifier()
        if abs(state_mod) > 0.01:
            disclosure = ctx.num(
                source_type="state",
                source_id="mood_stress_fatigue",
                target_field="knowledge_disclosure.disclosure_level",
                operation="add",
                before=disclosure,
                after=disclosure + state_mod,
                effect=f"State modifier: {state_mod:+.2f}",
                weight=0.6,
                reason="Mood/stress/fatigue influence"
            )

        # Trust modifier
        trust = p.memory.relationships.trust if p.memory else 0.5
        trust_factor = p.persona.disclosure_policy.factors.get("trust_level", 0.0)
        if abs(trust_factor) > 0.001:
            trust_mod = (trust - 0.5) * trust_factor * 2
            if abs(trust_mod) > 0.001:
                before_trust = disclosure
                disclosure = disclosure + trust_mod
                ctx.num(
                    source_type="memory",
                    source_id="trust_modifier",
                    target_field="knowledge_disclosure.disclosure_level",
                    operation="add",
                    before=before_trust,
                    after=disclosure,
                    effect=f"Trust modifier: {trust_mod:+.3f} (trust={trust:.2f}, factor={trust_factor:.2f})",
                    weight=0.7,
                    reason=f"trust={trust:.2f}, base=0.5, factor={trust_factor}"
                )

        # Privacy filter clamp
        privacy_filter = p.rules.get_privacy_filter_level(topic_signature)
        max_disclosure_privacy = 1.0 - max(p.persona.privacy_sensitivity, privacy_filter)

        if disclosure > max_disclosure_privacy:
            disclosure = ctx.clamp(
                field_name="disclosure_level",
                target_field="knowledge_disclosure.disclosure_level",
                proposed=disclosure,
                minimum=None,
                maximum=max_disclosure_privacy,
                constraint_name="privacy_filter",
                reason=f"Privacy filter limits disclosure (max={max_disclosure_privacy:.2f})"
            )

        # Topic sensitivity clamp
        for ts in p.persona.topic_sensitivities:
            if ts.topic.lower() in topic_signature.lower():
                max_topic_disclosure = 1.0 - ts.sensitivity
                if disclosure > max_topic_disclosure:
                    disclosure = ctx.clamp(
                        field_name="disclosure_level",
                        target_field="knowledge_disclosure.disclosure_level",
                        proposed=disclosure,
                        minimum=None,
                        maximum=max_topic_disclosure,
                        constraint_name="topic_sensitivity",
                        reason=f"Topic '{ts.topic}' is sensitive (max={max_topic_disclosure:.2f})"
                    )
                break

        # Disclosure policy bounds clamp
        policy_bounds = p.persona.disclosure_policy.bounds
        policy_min, policy_max = policy_bounds[0], policy_bounds[1]

        disclosure = ctx.clamp(
            field_name="disclosure_level",
            target_field="knowledge_disclosure.disclosure_level",
            proposed=disclosure,
            minimum=policy_min,
            maximum=policy_max,
            constraint_name="disclosure_policy_bounds",
            reason=f"Disclosure policy bounds [{policy_min:.2f}, {policy_max:.2f}]",
        )

        return disclosure

    # ========================================================================
    # Time Pressure
    # ========================================================================

    def compute_time_pressure(
        self,
        interaction_mode: InteractionMode,
        turn_number: int,
        ctx: TraceContext,
    ) -> float:
        """Compute context-sensitive time pressure."""
        p = self.planner
        base = p.persona.time_scarcity

        mode_modifiers = {
            InteractionMode.DEBATE: -0.25,
            InteractionMode.COACHING: -0.20,
            InteractionMode.INTERVIEW: -0.15,
            InteractionMode.BRAINSTORM: -0.20,
            InteractionMode.CASUAL_CHAT: -0.10,
            InteractionMode.SMALL_TALK: -0.05,
            InteractionMode.CUSTOMER_SUPPORT: 0.0,
            InteractionMode.SURVEY: 0.10,
        }

        mode_mod = mode_modifiers.get(interaction_mode, 0.0)
        adjusted = base + mode_mod

        if turn_number > TIME_PRESSURE_TURN_THRESHOLD:
            length_pressure = min(
                TIME_PRESSURE_MAX_BUILDUP,
                (turn_number - TIME_PRESSURE_TURN_THRESHOLD) * TIME_PRESSURE_PER_TURN,
            )
            adjusted += length_pressure

        adjusted = max(0.0, min(1.0, adjusted))

        if abs(adjusted - base) > 0.01:
            ctx.num(
                source_type="state",
                source_id="dynamic_time_pressure",
                target_field="time_pressure",
                operation="add",
                before=base,
                after=adjusted,
                effect=f"Mode={interaction_mode.value} → time_pressure {base:.2f} → {adjusted:.2f}",
                weight=0.8,
                reason=f"mode_mod={mode_mod:+.2f}"
            )

        return adjusted

    # ========================================================================
    # Claim Type
    # ========================================================================

    def infer_claim_type(
        self,
        proficiency: float,
        uncertainty_action: UncertaintyAction,
        domain: str,
        user_input: str = "",
    ) -> str:
        """Infer knowledge claim type."""
        p = self.planner
        is_domain_specific = any(
            k.domain.lower() == domain.lower()
            for k in p.persona.knowledge_domains
        )

        is_personal_experience = self.detect_personal_experience(
            user_input, domain, is_domain_specific
        )

        return infer_knowledge_claim_type(
            proficiency,
            uncertainty_action,
            is_personal_experience,
            is_domain_specific
        )

    def detect_personal_experience(
        self,
        user_input: str,
        domain: str,
        is_domain_specific: bool,
    ) -> bool:
        """Detect if user is asking about personal experience."""
        if not user_input:
            return False

        input_lower = user_input.lower()

        experience_patterns = [
            "have you ever",
            "have you tried",
            "what's your experience",
            "what is your experience",
            "do you like",
            "do you prefer",
            "what do you think about",
            "how do you feel about",
            "in your experience",
            "your opinion on",
            "your thoughts on",
            "what's your take",
            "what is your take",
            "from your perspective",
            "personally",
            "your favorite",
            "your favourite",
        ]

        asks_personal = any(pat in input_lower for pat in experience_patterns)
        if not asks_personal:
            return False

        return is_domain_specific

    # ========================================================================
    # Response Patterns
    # ========================================================================

    def apply_patterns_safely(
        self,
        context: ConversationContext,
        disclosure_level: float,
        ctx: TraceContext,
    ) -> None:
        """Apply response patterns with safety checks."""
        p = self.planner
        pattern = p.rules.check_response_pattern(context.user_input)

        if not pattern:
            return

        privacy_filter = p.rules.get_privacy_filter_level(context.topic_signature)
        modifications = apply_response_pattern_safely(
            pattern,
            disclosure_level,
            privacy_filter,
            p.persona.topic_sensitivities,
            p.persona.invariants.must_avoid,
            context.topic_signature
        )

        if modifications.get("pattern_blocked"):
            ctx.activate_constraint("pattern_safety")
            ctx.block_pattern(
                pattern_trigger=pattern.trigger if hasattr(pattern, 'trigger') else 'unknown_pattern',
                reason=modifications["block_reason"]
            )
            ctx.add_basic_citation(
                source_type="rule",
                source_id="pattern_safety",
                effect=f"BLOCKED: {modifications['block_reason']}",
                weight=1.0
            )

        elif modifications.get("pattern_constrained"):
            ctx.add_basic_citation(
                source_type="rule",
                source_id="pattern_safety",
                effect=f"CONSTRAINED: {modifications.get('constraint_reason', 'privacy')}",
                weight=0.8
            )
