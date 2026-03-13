"""
Turn Planner - Core Orchestration

The Turn Planner is the heart of the persona engine. It orchestrates all
behavioral interpreters to generate a complete IR following the canonical
modifier composition sequence.

This ensures:
- No double-counting of modifiers
- Clear citation trail
- Deterministic behavior
- Constraint enforcement
"""

from dataclasses import dataclass
from typing import Any, Optional

from persona_engine.behavioral import (
    BehavioralRulesEngine,
    BiasModifier,
    BiasSimulator,
    CognitiveStyleInterpreter,
    StateManager,
    TraitInterpreter,
    ValuesInterpreter,
    apply_response_pattern_safely,
    infer_knowledge_claim_type,
    resolve_uncertainty_action,
    validate_stance_against_invariants,
)
from persona_engine.memory import MemoryManager, StanceCache
from persona_engine.planner.domain_detection import (
    compute_domain_adjacency,
    compute_topic_relevance,
    detect_domain,
    detect_evidence_strength,
    generate_intent_string,
)
from persona_engine.planner.intent_analyzer import analyze_intent
from persona_engine.planner.stance_generator import generate_stance_safe
from persona_engine.planner.trace_context import TraceContext, clamp01, create_turn_seed
from persona_engine.schema.ir_schema import (
    Citation,
    CommunicationStyle,
    ConversationFrame,
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    KnowledgeAndDisclosure,
    KnowledgeClaimType,
    MemoryOps,
    MemoryWriteIntent,
    ResponseStructure,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils import DeterminismManager
from persona_engine.validation.cross_turn import TurnSnapshot

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# Extracted magic numbers for maintainability and tuning

# Domain & Expertise
DEFAULT_PROFICIENCY = 0.3            # Proficiency for unknown domains
EXPERT_THRESHOLD = 0.7               # Min proficiency for expert assertions

# Topic Relevance
# (Now computed dynamically via domain_detection.compute_topic_relevance)
DEFAULT_TOPIC_RELEVANCE = 0.5  # Fallback for sparse personas

# Communication Style
FORMALITY_ROLE_WEIGHT = 0.7          # Role's weight in formality blend
FORMALITY_BASE_WEIGHT = 0.3          # Base's weight in formality blend
DIRECTNESS_IMPATIENCE_BUMP = 0.1     # Directness increase when patience < threshold
PATIENCE_THRESHOLD = 0.3             # Below this, persona gets more direct

# Elasticity
ELASTICITY_MIN = 0.1
ELASTICITY_MAX = 0.9

# Disclosure
DISCLOSURE_MIN = 0.0
DISCLOSURE_MAX = 1.0

# Evidence/Stress
EVIDENCE_STRESS_THRESHOLD = 0.4  # Evidence strength above this triggers conflict stress

# Competence
UNKNOWN_DOMAIN_BASE = 0.10       # Floor competence for completely unknown topics
OPENNESS_COMPETENCE_WEIGHT = 0.1 # How much openness boosts comfort with unknown topics

# Cross-Turn Dynamics
CROSS_TURN_INERTIA = 0.15        # 15% anchor to previous turn's values
FAMILIARITY_BOOST_PER_EPISODE = 0.05  # Competence boost per prior topic episode
FAMILIARITY_BOOST_CAP = 0.15     # Maximum familiarity boost

# Dynamic Time Pressure
TIME_PRESSURE_TURN_THRESHOLD = 5  # Pressure starts building after this turn
TIME_PRESSURE_PER_TURN = 0.03    # Pressure increase per turn past threshold
TIME_PRESSURE_MAX_BUILDUP = 0.15 # Max additional pressure from conversation length


def _smooth(prev: float, new: float, alpha: float) -> float:
    """Cross-turn inertia smoothing: blend previous and current values."""
    return prev * alpha + new * (1 - alpha)


@dataclass
class ConversationContext:
    """Context for current conversation"""
    conversation_id: str
    turn_number: int
    interaction_mode: InteractionMode | None  # Can be None, inferred by analyze_intent
    goal: ConversationGoal | None  # Can be None, inferred by analyze_intent
    topic_signature: str  # Hash/key for topic
    user_input: str
    stance_cache: StanceCache
    domain: str | None = None


class TurnPlanner:
    """
    Orchestrates all behavioral components to generate IR.

    Follows canonical modifier composition sequence:
    base → role → trait → state → constraints
    """

    def __init__(
        self,
        persona: Persona,
        determinism: DeterminismManager | None = None,
        memory_manager: MemoryManager | None = None,
    ):
        self.persona = persona
        self.determinism = determinism or DeterminismManager()
        self.memory = memory_manager

        # Initialize all interpreters
        self.traits = TraitInterpreter(persona.psychology.big_five)
        self.values = ValuesInterpreter(persona.psychology.values)
        self.cognitive = CognitiveStyleInterpreter(persona.psychology.cognitive_style)
        self.state = StateManager(
            persona.initial_state,
            persona.psychology.big_five,
            self.determinism
        )
        self.rules = BehavioralRulesEngine(persona)

        # Initialize bias simulator
        self.bias_simulator = BiasSimulator(
            traits={
                "openness": persona.psychology.big_five.openness,
                "conscientiousness": persona.psychology.big_five.conscientiousness,
                "extraversion": persona.psychology.big_five.extraversion,
                "agreeableness": persona.psychology.big_five.agreeableness,
                "neuroticism": persona.psychology.big_five.neuroticism,
            },
            value_priorities=self.values.get_value_priorities(),
        )

        # Track bias modifiers for current turn (reset per turn)
        self._current_bias_modifiers: list[BiasModifier] = []

        # Cross-turn dynamics: previous turn's IR snapshot for inertia smoothing
        self._prior_snapshot: TurnSnapshot | None = None

    def generate_ir(self, context: ConversationContext) -> IntermediateRepresentation:
        """
        Generate Intermediate Representation for a single turn.

        This is the central orchestration method that coordinates all behavioral
        interpreters using canonical modifier composition with full citation tracking.

        P0/P1 PRODUCTION FEATURES:
        - TraceContext for all citations + SafetyPlan
        - Per-turn deterministic seeding
        - Early intent analysis (mode/goal inference)
        - Expert eligibility computed early (for stance guardrails)
        - Multi-clamp chains preserved in SafetyPlan
        - Pattern blocks recorded
        """
        # Imports already at top level

        # ====================================================================
        # SETUP: TraceContext + Per-Turn Seed
        # ====================================================================
        ctx = TraceContext()
        memory_ops = MemoryOps()

        # Per-turn deterministic seed
        turn_seed = create_turn_seed(
            base_seed=self.determinism.seed if self.determinism.seed is not None else 0,
            conversation_id=context.conversation_id,
            turn_number=context.turn_number
        )
        self.determinism.set_seed(turn_seed)

        # ====================================================================
        # 0. MEMORY CONTEXT (read before planning)
        # ====================================================================
        memory_context: dict[str, Any] = {}
        if self.memory:
            memory_context = self.memory.get_context_for_turn(
                topic=context.topic_signature,
                current_turn=context.turn_number,
            )
            if memory_context.get("known_facts"):
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="fact_store",
                    effect=f"Loaded {len(memory_context['known_facts'])} known facts from memory",
                    weight=0.8,
                )
            if memory_context.get("active_preferences"):
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="preference_store",
                    effect=f"Loaded {len(memory_context['active_preferences'])} active preferences",
                    weight=0.6,
                )
            if memory_context.get("previously_discussed"):
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="episodic_store",
                    effect=f"Topic '{context.topic_signature}' previously discussed",
                    weight=0.7,
                )

        # ====================================================================
        # 1. TOPIC RELEVANCE (computed before state evolution)
        # ====================================================================
        # Compute topic relevance using real scoring (not hardcoded)
        persona_domains = []
        if self.persona.knowledge_domains:
            for kd in self.persona.knowledge_domains:
                persona_domains.append({
                    "domain": kd.domain,
                    "proficiency": kd.proficiency,
                    "subdomains": getattr(kd, "subdomains", None) or []
                })

        persona_goals = []
        if hasattr(self.persona, 'primary_goals'):
            for g in self.persona.primary_goals:
                persona_goals.append({
                    "goal": getattr(g, "goal", ""),
                    "weight": getattr(g, "weight", 1.0)
                })
        if hasattr(self.persona, 'secondary_goals'):
            for g in self.persona.secondary_goals:
                persona_goals.append({
                    "goal": getattr(g, "goal", ""),
                    "weight": getattr(g, "weight", 0.5)
                })

        topic_relevance = compute_topic_relevance(
            user_input=context.user_input,
            persona_domains=persona_domains,
            persona_goals=persona_goals,
            ctx=ctx,
            default_relevance=DEFAULT_TOPIC_RELEVANCE
        )

        # ====================================================================
        # 1.5 COMPUTE BIAS MODIFIERS (Phase 2 - Bounded Bias Simulation)
        # ====================================================================
        # Use topic_relevance as proxy for value alignment
        self._current_bias_modifiers = self.bias_simulator.compute_modifiers(
            user_input=context.user_input,
            value_alignment=topic_relevance,
            ctx=ctx,
        )

        # ====================================================================
        # 2. STATE EVOLUTION (before generating response)
        # ====================================================================
        # NOTE: Method name is "post_turn" because it updates state based on
        # previous turn's metrics. This runs BEFORE generating the current response.
        self.state.evolve_state_post_turn(
            conversation_length=context.turn_number,
            topic_relevance=topic_relevance  # Now using real computed value
        )

        # ====================================================================
        # 1.5. INTENT ANALYSIS (EARLY - infers mode/goal if missing)
        # ====================================================================
        inferred_mode, inferred_goal, user_intent, needs_clarification = analyze_intent(
            user_input=context.user_input,
            current_mode=context.interaction_mode,
            current_goal=context.goal,
            ctx=ctx
        )

        # DESIGN NOTE: We intentionally mutate the caller's context object here.
        # This is a side effect, but it's required because:
        # 1. IR assembly needs the inferred mode/goal (not the original None values)
        # 2. Downstream code (stance cache key, etc.) depends on the updated values
        # 3. Callers expect context to reflect the "resolved" state after generate_ir()
        # Alternative: return a new context, but that would break expected semantics.
        context.interaction_mode = inferred_mode
        context.goal = inferred_goal

        # ====================================================================
        # 3. DOMAIN + PROFICIENCY (with keyword scoring + citations)
        # ====================================================================
        domain = context.domain or self._detect_domain(context.user_input, ctx=ctx)
        proficiency = self._get_domain_proficiency(domain)

        ctx.add_basic_citation(
            source_type="state",
            source_id="domain_proficiency",
            effect=f"Domain '{domain}' proficiency: {proficiency:.2f}",
            weight=1.0
        )

        # ====================================================================
        # 2.5. EXPERT ELIGIBILITY (P1: compute EARLY for stance guardrails)
        # ====================================================================
        # Compute once; reused in _infer_claim_type via domain_context
        is_domain_specific = any(
            kd.domain.lower() == domain.lower()
            for kd in self.persona.knowledge_domains
        )
        expert_threshold = getattr(self.persona.claim_policy, "expert_threshold", EXPERT_THRESHOLD)
        expert_allowed = is_domain_specific and (proficiency >= expert_threshold)

        ctx.add_basic_citation(
            source_type="rule",
            source_id="expert_eligibility",
            effect=f"Expert allowed: {expert_allowed} (is_domain={is_domain_specific}, prof={proficiency:.2f}, thresh={expert_threshold:.2f})",
            weight=1.0
        )
        # NOTE: domain_context was previously stored here for reuse in _infer_claim_type,
        # but _infer_claim_type currently recomputes is_domain_specific. If performance
        # becomes a concern, pass domain_context as a parameter instead.

        # ====================================================================
        # 4. CORE METRICS (ELASTICITY) - Computed EARLY for cache logic
        # ====================================================================
        elasticity = self._compute_elasticity(proficiency, ctx)

        # ====================================================================
        # 3. STANCE (with cache + expert guardrails + reconsideration)
        # ====================================================================
        # Detect evidence strength (challenge detection)
        evidence_strength = detect_evidence_strength(context.user_input, ctx=ctx)

        # Apply stress if challenged
        if evidence_strength > EVIDENCE_STRESS_THRESHOLD:
            self.state.apply_stress_trigger("conflict", intensity=evidence_strength)
            ctx.add_basic_citation(
                source_type="state",
                source_id="stress_trigger",
                effect=f"Evidence {evidence_strength:.2f} > threshold {EVIDENCE_STRESS_THRESHOLD} → stress trigger",
                weight=1.0
            )

        stance, rationale = self._generate_stance(
            context=context,
            proficiency=proficiency,
            expert_allowed=expert_allowed,
            evidence_strength=evidence_strength,
            current_elasticity=elasticity,
            ctx=ctx
        )

        # ====================================================================
        # 5-7. REMAINING METRICS (confidence, competence, tone, verbosity)
        # ====================================================================
        confidence = self._compute_confidence(proficiency, ctx)

        # Fix 4: Cross-turn inertia smoothing — confidence
        # Smooth BEFORE uncertainty resolver / claim type use the value
        if self._prior_snapshot:
            before_smooth = confidence
            confidence = _smooth(self._prior_snapshot.confidence, confidence, CROSS_TURN_INERTIA)
            if abs(confidence - before_smooth) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="response_structure.confidence",
                    operation="blend",
                    before=before_smooth,
                    after=confidence,
                    effect=f"Cross-turn inertia: {before_smooth:.3f} → {confidence:.3f} (prev={self._prior_snapshot.confidence:.3f})",
                    weight=0.7,
                    reason=f"inertia={CROSS_TURN_INERTIA}"
                )

        competence = self._compute_competence(domain, proficiency, persona_domains, ctx)
        tone = self._select_tone(ctx)
        verbosity = self._compute_verbosity(ctx)

        # ====================================================================
        # 8. COMMUNICATION STYLE (formality + directness)
        # ====================================================================
        formality, directness = self._compute_communication_style(
            context.interaction_mode,
            ctx
        )

        # Fix 4: Cross-turn inertia smoothing — formality + directness
        if self._prior_snapshot:
            before_f = formality
            formality = _smooth(self._prior_snapshot.formality, formality, CROSS_TURN_INERTIA)
            if abs(formality - before_f) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="communication_style.formality",
                    operation="blend",
                    before=before_f,
                    after=formality,
                    effect=f"Cross-turn inertia: {before_f:.3f} → {formality:.3f}",
                    weight=0.7,
                    reason=f"inertia={CROSS_TURN_INERTIA}"
                )

            before_d = directness
            directness = _smooth(self._prior_snapshot.directness, directness, CROSS_TURN_INERTIA)
            if abs(directness - before_d) > 0.001:
                ctx.num(
                    source_type="cross_turn",
                    source_id="inertia_smoothing",
                    target_field="communication_style.directness",
                    operation="blend",
                    before=before_d,
                    after=directness,
                    effect=f"Cross-turn inertia: {before_d:.3f} → {directness:.3f}",
                    weight=0.7,
                    reason=f"inertia={CROSS_TURN_INERTIA}"
                )

        # ====================================================================
        # 9. DISCLOSURE (P1: multi-clamp with SafetyPlan)
        # ====================================================================
        disclosure_level = self._compute_disclosure(
            context.topic_signature,
            ctx
        )

        # Fix 4: Cross-turn inertia smoothing — disclosure
        if self._prior_snapshot:
            before_disc = disclosure_level
            disclosure_level = _smooth(self._prior_snapshot.disclosure, disclosure_level, CROSS_TURN_INERTIA)
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
                    reason=f"inertia={CROSS_TURN_INERTIA}"
                )

        # ====================================================================
        # 10. UNCERTAINTY ACTION (single resolver)
        # ====================================================================
        # Fix 8: Dynamic time pressure (mode + conversation length)
        time_pressure = self._compute_time_pressure(
            context.interaction_mode, context.turn_number, ctx
        )

        # Use separate list to avoid bypassing TraceContext's structured API
        resolver_citations: list[Citation] = []
        uncertainty_action = resolve_uncertainty_action(
            proficiency=proficiency,
            confidence=confidence,
            risk_tolerance=self.cognitive.style.risk_tolerance,
            need_for_closure=self.cognitive.style.need_for_closure,
            time_pressure=time_pressure,
            claim_policy_lookup_behavior=self.persona.claim_policy.lookup_behavior,
            citations=resolver_citations  # Isolated list
        )

        # Merge resolver citations into TraceContext (preserves audit trail)
        for cite in resolver_citations:
            ctx.citations.append(cite)


        # Add enum citation for uncertainty_action (derived, not base)
        ctx.enum(
            source_type="rule",
            source_id="uncertainty_resolver",
            target_field="knowledge_disclosure.uncertainty_action",
            operation="set",
            before="none",
            after=uncertainty_action.value,
            effect=f"Uncertainty action resolved: {uncertainty_action.value}",
            weight=1.0
        )

        # ====================================================================
        # 11. KNOWLEDGE CLAIM TYPE
        # ====================================================================
        knowledge_claim_type = self._infer_claim_type(
            proficiency,
            uncertainty_action,
            domain,
            user_input=context.user_input
        )

        # Enum citation for claim type (derived from inference, not base)
        claim_enum = KnowledgeClaimType(knowledge_claim_type)
        ctx.enum(
            source_type="rule",
            source_id="claim_type_inference",
            target_field="knowledge_disclosure.knowledge_claim_type",
            operation="set",
            before="none",
            after=claim_enum.value,
            effect=f"Knowledge claim type inferred: {claim_enum.value}",
            weight=1.0
        )

        # ====================================================================
        # 12. APPLY RESPONSE PATTERNS (P1: populates SafetyPlan.pattern_blocks)
        # ====================================================================
        self._apply_patterns_safely(
            context,
            disclosure_level,
            ctx
        )

        # ====================================================================
        # 13. VALIDATE CONSTRAINTS (invariants)
        # ====================================================================
        validation = validate_stance_against_invariants(
            stance,
            rationale,
            self.persona.invariants.identity_facts,
            self.persona.invariants.cannot_claim,
            must_avoid=self.persona.invariants.must_avoid,
        )

        if not validation["valid"]:
            ctx.activate_constraint("invariants")
            for violation in validation["violations"]:
                ctx.add_basic_citation(
                    source_type="rule",
                    source_id="invariant_violation",
                    effect=f"VIOLATION: {violation['message']}",
                    weight=1.0
                )

        # ====================================================================
        # 14. POPULATE MEMORY WRITE INTENTS
        # ====================================================================
        write_intents: list[MemoryWriteIntent] = []

        # Auto-generate memory write intents from this turn
        if self.memory:
            # Always record an episode summary
            write_intents.append(MemoryWriteIntent(
                content_type="episode",
                content=f"Discussed {context.topic_signature}: persona {uncertainty_action.value}ed with {claim_enum.value} claim",
                confidence=0.9,
                privacy_level=0.2,
                source="observed_behavior",
            ))

            # Fix 1A: Richer relationship content that triggers delta detection
            if stance and confidence > 0.5:
                rel_content = f"Engaged on {context.topic_signature}"
                if confidence > 0.7:
                    rel_content += " — validated shared expertise"
                if claim_enum.value == "personal_experience":
                    rel_content += " — shared personal perspective"
                if user_intent == "challenge":
                    rel_content += " — challenged viewpoint"
                elif user_intent == "share":
                    rel_content += " — friendly exchange"
                write_intents.append(MemoryWriteIntent(
                    content_type="relationship",
                    content=rel_content,
                    confidence=0.7,
                    privacy_level=0.1,
                    source="observed_behavior",
                ))

            memory_ops = MemoryOps(write_intents=write_intents)

        # Propagate invariants into safety plan for post-generation checks
        ctx.safety_plan.cannot_claim = list(self.persona.invariants.cannot_claim)
        ctx.safety_plan.must_avoid = list(self.persona.invariants.must_avoid)

        # ====================================================================
        # 15. ASSEMBLE IR (P0/P1: includes SafetyPlan + MemoryOps)
        # ====================================================================
        ir = IntermediateRepresentation(
            conversation_frame=ConversationFrame(
                interaction_mode=context.interaction_mode,
                goal=context.goal,
                success_criteria=None  # TODO: Implement goal-based criteria
            ),
            response_structure=ResponseStructure(
                intent=self._generate_intent(
                    user_intent=user_intent,
                    conversation_goal=str(context.goal.value) if context.goal else "inform",
                    uncertainty_action=str(uncertainty_action.value),
                    needs_clarification=needs_clarification,
                    ctx=ctx
                ),
                stance=stance,

                rationale=rationale,
                elasticity=elasticity,
                confidence=confidence,
                competence=competence,
            ),
            communication_style=CommunicationStyle(
                tone=tone,
                verbosity=verbosity,
                formality=formality,
                directness=directness
            ),
            knowledge_disclosure=KnowledgeAndDisclosure(
                disclosure_level=disclosure_level,
                uncertainty_action=uncertainty_action,
                knowledge_claim_type=claim_enum  # Already validated above
            ),
            citations=ctx.citations,          # P0: delta-based citations
            safety_plan=ctx.safety_plan,      # P1: clamps + blocks + constraints
            memory_ops=memory_ops,             # P0: Phase 4 prep
            turn_id=f"{context.conversation_id}_turn_{context.turn_number}",
            seed=turn_seed                     # P0: per-turn seed
        )

        # ====================================================================
        # 16. CACHE STANCE (for future consistency)
        # ====================================================================
        if stance:
            context.stance_cache.store_stance(
                topic_signature=context.topic_signature,
                interaction_mode=str(context.interaction_mode.value),
                stance=stance,
                rationale=rationale,
                elasticity=elasticity,
                confidence=confidence,
                turn_number=context.turn_number
            )

        # ====================================================================
        # 17. PROCESS MEMORY WRITES (after IR is final)
        # ====================================================================
        if self.memory and ir.memory_ops.write_intents:
            self.memory.process_write_intents(
                intents=ir.memory_ops.write_intents,
                turn=context.turn_number,
                conversation_id=context.conversation_id,
            )

        # ====================================================================
        # 18. STORE SNAPSHOT FOR CROSS-TURN DYNAMICS
        # ====================================================================
        self._prior_snapshot = TurnSnapshot.from_ir(
            ir, context.turn_number, context.topic_signature
        )

        return ir

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _detect_domain(self, user_input: str, ctx: TraceContext | None = None) -> str:
        """
        Detect domain from user input using keyword scoring.

        Uses domain_detection module for deterministic keyword matching.
        Falls back to 'general' with citation when no match exceeds threshold.
        """
        # Build persona domains for detection boost
        persona_domains = [
            {"domain": kd.domain, "proficiency": kd.proficiency, "subdomains": kd.subdomains}
            for kd in self.persona.knowledge_domains
        ] if self.persona.knowledge_domains else []

        domain, _score = detect_domain(
            user_input=user_input,
            persona_domains=persona_domains,
            ctx=ctx
        )

        return domain

    def _get_domain_proficiency(self, domain: str) -> float:
        """Get persona's proficiency in domain"""
        for knowledge in self.persona.knowledge_domains:
            if knowledge.domain.lower() == domain.lower():
                return knowledge.proficiency
        return DEFAULT_PROFICIENCY  # Low proficiency in unknown domains

    def _generate_stance(
        self,
        context: ConversationContext,
        proficiency: float,
        expert_allowed: bool,
        evidence_strength: float,
        current_elasticity: float,
        ctx: TraceContext
    ) -> tuple[str, str]:
        """
        Generate or retrieve cached stance on topic (P1 guardrails via expert_allowed).

        Handles reconsideration if new evidence (challenge) > threshold.

        Returns: (stance, rationale)
        """
        # Check cache first
        cached = context.stance_cache.get_stance(
            context.topic_signature,
            str(context.interaction_mode.value) if context.interaction_mode else "casual_chat",
            context.turn_number
        )

        # Decide whether to use cached stance
        if cached:
            should_reconsider = context.stance_cache.should_reconsider(
                cached=cached,
                new_evidence_strength=evidence_strength,
                current_elasticity=current_elasticity,
                current_turn=context.turn_number
            )

            if not should_reconsider:
                ctx.add_basic_citation(
                    source_type="memory",
                    source_id="stance_cache",
                    effect=f"Using cached stance (age: {context.turn_number - cached.created_turn} turns)",
                    weight=1.0
                )
                return cached.stance, cached.rationale_seeds

            else:
                ctx.add_basic_citation(
                    source_type="rule",
                    source_id="stance_reconsideration",
                    effect=f"Reconsidering cached stance (evidence={evidence_strength:.2f}, elasticity={current_elasticity:.2f})",
                    weight=1.0
                )

        # Generate new stance with guardrails
        stance, rationale = generate_stance_safe(
            persona=self.persona,
            values=self.values,
            cognitive=self.cognitive,
            user_input=context.user_input,
            topic_signature=context.topic_signature,
            proficiency=proficiency,
            expert_allowed=expert_allowed,
            ctx=ctx
        )

        return stance, rationale


    def _compute_elasticity(
        self,
        proficiency: float,
        ctx: "TraceContext"
    ) -> float:
        """
        Compute elasticity (openness to persuasion).

        Sequence: base (trait) → cognitive blend → confirmation bias → bounds [0.1, 0.9]
        """
        # Trait-based elasticity
        trait_elasticity = self.traits.get_elasticity(proficiency)
        elasticity = ctx.base(
            field_name="elasticity.trait_base",
            target_field="response_structure.elasticity",
            value=trait_elasticity,
            effect=f"Trait-based elasticity ({trait_elasticity:.2f})"
        )

        # Cognitive complexity modifier
        cognitive_elasticity = self.cognitive.get_elasticity_from_cognitive_style()
        blended = (elasticity + cognitive_elasticity) / 2.0

        elasticity = ctx.num(
            source_type="trait",
            source_id="cognitive_complexity",
            target_field="response_structure.elasticity",
            operation="blend",
            before=elasticity,
            after=blended,
            effect="Blend trait elasticity with cognitive elasticity",
            weight=0.7,
            reason=f"trait={trait_elasticity:.2f}, cognitive={cognitive_elasticity:.2f}"
        )

        # Apply Confirmation Bias modifier (Phase 2)
        conf_mod = self.bias_simulator.get_modifier_for_field(
            self._current_bias_modifiers,
            "response_structure.elasticity"
        )
        if conf_mod:
            biased = elasticity + conf_mod.modifier
            elasticity = ctx.num(
                source_type="rule",
                source_id="confirmation_bias",
                target_field="response_structure.elasticity",
                operation="add",
                before=elasticity,
                after=biased,
                effect=f"Confirmation bias: {conf_mod.modifier:+.3f}",
                weight=conf_mod.strength,
                reason=conf_mod.trigger
            )

        # Clamp to [0.1, 0.9] bounds
        # NOTE: Always attempt clamp; ctx.clamp only records if clamping actually occurs
        elasticity = ctx.clamp(
            field_name="elasticity",
            target_field="response_structure.elasticity",
            proposed=elasticity,
            minimum=ELASTICITY_MIN,
            maximum=ELASTICITY_MAX,
            constraint_name="bounds_check",
            reason=f"Elasticity bounds [{ELASTICITY_MIN}, {ELASTICITY_MAX}]"
        )

        return elasticity

    def _compute_confidence(
        self,
        proficiency: float,
        ctx: "TraceContext"
    ) -> float:
        """
        Compute response confidence.

        Sequence: base (proficiency) → trait → cognitive → authority bias → bounds
        """

        # Base from proficiency
        confidence = ctx.base(
            field_name="confidence.proficiency_base",
            target_field="response_structure.confidence",
            value=proficiency,
            effect=f"Base confidence from domain proficiency ({proficiency:.2f})"
        )

        # Trait adjustments (conscientiousness, neuroticism)
        adjusted = self.traits.get_confidence_modifier(confidence)
        confidence = ctx.num(
            source_type="trait",
            source_id="confidence_traits",
            target_field="response_structure.confidence",
            operation="add",
            before=confidence,
            after=adjusted,
            effect="Traits adjust confidence (conscientiousness/neuroticism)",
            weight=0.8,
            reason=f"C={self.persona.psychology.big_five.conscientiousness:.2f}, N={self.persona.psychology.big_five.neuroticism:.2f}"
        )

        # Cognitive style adjustments
        adjusted2 = self.cognitive.get_confidence_adjustment(confidence)
        confidence = ctx.num(
            source_type="trait",
            source_id="cognitive_style",
            target_field="response_structure.confidence",
            operation="add",
            before=confidence,
            after=adjusted2,
            effect="Cognitive style adjusts confidence",
            weight=0.6,
            reason="Risk tolerance/need for closure influence"
        )

        # Apply Authority Bias modifier (Phase 2)
        auth_mod = self.bias_simulator.get_modifier_for_field(
            self._current_bias_modifiers,
            "response_structure.confidence"
        )
        if auth_mod:
            biased = confidence + auth_mod.modifier
            confidence = ctx.num(
                source_type="value",
                source_id="authority_bias",
                target_field="response_structure.confidence",
                operation="add",
                before=confidence,
                after=biased,
                effect=f"Authority bias: {auth_mod.modifier:+.3f}",
                weight=auth_mod.strength,
                reason=auth_mod.trigger
            )

        # Bounds clamp
        confidence = clamp01(
            ctx,
            field_name="confidence",
            target_field="response_structure.confidence",
            value=confidence
        )

        return confidence

    def _compute_competence(
        self,
        domain: str,
        proficiency: float,
        persona_domains: list[dict],
        ctx: "TraceContext",
    ) -> float:
        """
        Compute how equipped the persona is to engage with this topic.

        Distinct from confidence: competence measures knowledge depth,
        confidence measures certainty. A knowledgeable persona can feel
        uncertain (low confidence, high competence), and vice versa.

        Sequence: direct match | adjacency fallback → openness modifier → clamp
        """
        # Step 1: Direct domain match
        is_direct_match = any(
            kd.domain.lower() == domain.lower()
            for kd in self.persona.knowledge_domains
        )

        if is_direct_match:
            competence = ctx.base(
                field_name="competence.domain_match",
                target_field="response_structure.competence",
                value=proficiency,
                effect=f"Direct domain match '{domain}' — competence = proficiency ({proficiency:.2f})",
            )
        else:
            # Step 2: Adjacency fallback
            adjacency_score, nearest = compute_domain_adjacency(
                detected_domain=domain,
                persona_domains=persona_domains,
                ctx=ctx,
            )

            if adjacency_score > 0 and nearest:
                competence = ctx.base(
                    field_name="competence.adjacency",
                    target_field="response_structure.competence",
                    value=adjacency_score,
                    effect=(
                        f"No direct match for '{domain}' — adjacent to "
                        f"'{nearest}' (adjacency={adjacency_score:.3f})"
                    ),
                )
            else:
                # Step 3: Unknown domain floor
                competence = ctx.base(
                    field_name="competence.unknown",
                    target_field="response_structure.competence",
                    value=UNKNOWN_DOMAIN_BASE,
                    effect=f"No domain match or adjacency for '{domain}' — base floor ({UNKNOWN_DOMAIN_BASE})",
                )

        # Step 4: Openness modifier (comfort with novelty)
        openness = self.persona.psychology.big_five.openness
        openness_boost = openness * OPENNESS_COMPETENCE_WEIGHT
        before = competence
        competence = competence + openness_boost

        competence = ctx.num(
            source_type="trait",
            source_id="openness",
            target_field="response_structure.competence",
            operation="add",
            before=before,
            after=competence,
            effect=f"Openness ({openness:.2f}) adds novelty comfort ({openness_boost:.3f})",
            weight=0.5,
            reason=f"O={openness:.2f} * weight={OPENNESS_COMPETENCE_WEIGHT}",
        )

        # Step 4.5: Familiarity boost from repeated topic engagement
        if self.memory:
            previously_discussed = self.memory.episodes.has_discussed(domain)
            topic_episodes = self.memory.episodes.get_by_topic(domain)
            if previously_discussed and topic_episodes:
                boost = min(
                    len(topic_episodes) * FAMILIARITY_BOOST_PER_EPISODE,
                    FAMILIARITY_BOOST_CAP,
                )
                before_fam = competence
                competence = competence + boost
                ctx.num(
                    source_type="memory",
                    source_id="topic_familiarity",
                    target_field="response_structure.competence",
                    operation="add",
                    before=before_fam,
                    after=competence,
                    effect=f"Familiarity boost: +{boost:.3f} ({len(topic_episodes)} prior episodes on '{domain}')",
                    weight=0.6,
                    reason=f"episodes={len(topic_episodes)}, boost_per={FAMILIARITY_BOOST_PER_EPISODE}, cap={FAMILIARITY_BOOST_CAP}",
                )

        # Step 4.6: Known fact boost — relevant stored facts increase competence
        if self.memory:
            relevant_facts = self.memory.facts.search(
                domain, current_turn=0, min_confidence=0.5,
            )
            if relevant_facts:
                fact_boost = min(len(relevant_facts) * 0.03, 0.10)
                before_fact = competence
                competence = competence + fact_boost
                ctx.num(
                    source_type="memory",
                    source_id="known_facts",
                    target_field="response_structure.competence",
                    operation="add",
                    before=before_fact,
                    after=competence,
                    effect=f"Known facts boost: +{fact_boost:.3f} ({len(relevant_facts)} facts about '{domain}')",
                    weight=0.5,
                    reason=f"facts={len(relevant_facts)}, boost_per=0.03, cap=0.10",
                )

        # Step 5: Clamp [0, 1]
        competence = clamp01(
            ctx,
            field_name="competence",
            target_field="response_structure.competence",
            value=competence,
        )

        return competence

    def _select_tone(self, ctx: "TraceContext") -> Tone:
        """
        Select tone from mood + stress + traits + negativity bias.

        Sequence: base (mood/stress/traits) → negativity bias adjustment
        """
        valence, arousal = self.state.get_mood()
        stress = self.state.get_stress()

        # Check for negativity bias (Phase 2)
        # Negativity bias increases arousal, which may shift tone selection
        neg_mod = self.bias_simulator.get_modifier_for_field(
            self._current_bias_modifiers,
            "communication_style.arousal"
        )
        if neg_mod:
            arousal_before = arousal
            arousal = min(1.0, arousal + neg_mod.modifier)
            ctx.num(
                source_type="trait",
                source_id="negativity_bias",
                target_field="communication_style.arousal",
                operation="add",
                before=arousal_before,
                after=arousal,
                effect=f"Negativity bias: arousal {neg_mod.modifier:+.3f}",
                weight=neg_mod.strength,
                reason=neg_mod.trigger
            )

        tone = self.traits.get_tone_from_mood(valence, arousal, stress)

        ctx.base_enum(
            field_name="tone.mood_stress_traits",
            target_field="communication_style.tone",
            value=tone.value,
            effect=f"Mood (v={valence:.2f}, a={arousal:.2f}) + stress ({stress:.2f}) → {tone.value}"
        )

        return tone

    def _compute_verbosity(
        self,
        ctx: "TraceContext"
    ) -> Verbosity:
        """
        Compute verbosity level.

        Base from traits, override by state (fatigue/engagement).
        """
        base_verbosity = self.persona.psychology.communication.verbosity
        verbosity_enum = self.traits.influences_verbosity(base_verbosity)

        current = ctx.base_enum(
            field_name="verbosity.trait_derived",
            target_field="communication_style.verbosity",
            value=verbosity_enum.value,
            effect=f"Trait-derived verbosity ({verbosity_enum.value})"
        )

        # State overrides
        modifier = self.state.get_verbosity_modifier()

        if modifier == -1 and verbosity_enum != Verbosity.BRIEF:
            ctx.enum(
                source_type="state",
                source_id="fatigue",
                target_field="communication_style.verbosity",
                operation="override",
                before=current,
                after=Verbosity.BRIEF.value,
                effect="High fatigue reduces verbosity",
                weight=0.6
            )
            return Verbosity.BRIEF

        if modifier == 1 and verbosity_enum != Verbosity.DETAILED:
            ctx.enum(
                source_type="state",
                source_id="engagement",
                target_field="communication_style.verbosity",
                operation="override",
                before=current,
                after=Verbosity.DETAILED.value,
                effect="High engagement increases verbosity",
                weight=0.6
            )
            return Verbosity.DETAILED

        return verbosity_enum

    def _compute_communication_style(
        self,
        interaction_mode: InteractionMode,
        ctx: "TraceContext"  # Changed from citations
    ) -> tuple[float, float]:
        """
        Compute formality and directness using canonical sequence.

        Sequence: base → role → trait → state → constraints

        Returns: (formality, directness)
        """

        # ==================================================================
        # 1. BASE CITATIONS (clean trail start)
        # ==================================================================
        formality = ctx.base(
            field_name="communication.formality",
            target_field="communication_style.formality",
            value=self.persona.psychology.communication.formality,
            effect=f"Base formality from persona ({self.persona.psychology.communication.formality:.2f})"
        )

        directness = ctx.base(
            field_name="communication.directness",
            target_field="communication_style.directness",
            value=self.persona.psychology.communication.directness,
            effect=f"Base directness from persona ({self.persona.psychology.communication.directness:.2f})"
        )

        # ==================================================================
        # 2. ROLE BLEND (70% role, 30% base)
        # ==================================================================
        role_mode = self.rules.get_social_role_mode(interaction_mode)
        role_adjustments = self.rules.apply_social_role_adjustments(
            interaction_mode,
            formality,
            directness,
            self.persona.psychology.communication.emotional_expressiveness
        )

        formality = ctx.num(
            source_type="rule",
            source_id=f"social_role_{role_mode}",
            target_field="communication_style.formality",
            operation="blend",
            before=formality,
            after=role_adjustments["formality"],
            effect=f"Role blend applied (70/30) for {role_mode}",
            weight=1.0,
            reason=f"interaction_mode={interaction_mode.value}"
        )

        directness = ctx.num(
            source_type="rule",
            source_id=f"social_role_{role_mode}",
            target_field="communication_style.directness",
            operation="blend",
            before=directness,
            after=role_adjustments["directness"],
            effect=f"Role blend applied (70/30) for {role_mode}",
            weight=1.0,
            reason=f"interaction_mode={interaction_mode.value}"
        )

        # ==================================================================
        # 2.5. MODE-SPECIFIC OVERLAY (Fix 7B: debate differentiation)
        # ==================================================================
        if interaction_mode == InteractionMode.DEBATE:
            before_d_debate = directness
            directness = min(1.0, directness + 0.15)
            if abs(directness - before_d_debate) > 0.001:
                ctx.num(
                    source_type="rule",
                    source_id="debate_mode_overlay",
                    target_field="communication_style.directness",
                    operation="add",
                    before=before_d_debate,
                    after=directness,
                    effect="Debate mode: +0.15 directness for argumentative stance",
                    weight=0.9,
                    reason="interaction_mode=debate"
                )

        # ==================================================================
        # 3. TRAIT MODIFIER (agreeableness affects directness)
        # ==================================================================
        before_directness = directness
        directness_after_trait = self.traits.influences_directness(directness)

        directness = ctx.num(
            source_type="trait",
            source_id="agreeableness",
            target_field="communication_style.directness",
            operation="add",
            before=before_directness,
            after=directness_after_trait,
            effect=f"Agreeableness ({self.persona.psychology.big_five.agreeableness:.2f}) reduces directness",
            weight=0.8,
            reason=f"A={self.persona.psychology.big_five.agreeableness:.2f} → modifier={directness_after_trait - before_directness:+.3f}"
        )

        # ==================================================================
        # 4. STATE MODIFIER (patience affects directness)
        # ==================================================================
        patience = self.state.get_patience_level()
        if patience < PATIENCE_THRESHOLD:
            before_patience = directness
            directness = min(1.0, directness + DIRECTNESS_IMPATIENCE_BUMP)

            directness = ctx.num(
                source_type="state",
                source_id="patience",
                target_field="communication_style.directness",
                operation="add",
                before=before_patience,
                after=directness,
                effect=f"Low patience ({patience:.2f}) increases directness",
                weight=0.5,
                reason=f"patience={patience:.2f} < {PATIENCE_THRESHOLD} → +{DIRECTNESS_IMPATIENCE_BUMP} directness"
            )

        # ==================================================================
        # 5. CONSTRAINTS CLAMP (bounds check with safety plan recording)
        # ==================================================================
        formality = clamp01(
            ctx,
            field_name="formality",
            target_field="communication_style.formality",
            value=formality
        )

        directness = clamp01(
            ctx,
            field_name="directness",
            target_field="communication_style.directness",
            value=directness
        )

        return formality, directness

    def _compute_disclosure(
        self,
        topic_signature: str,
        ctx: "TraceContext"  # Changed from citations
    ) -> float:
        """
        Compute disclosure level with canonical composition.

        Sequence: base → trait modifier → state modifier → privacy clamp → topic clamp → bounds

        This is P1 critical: all clamps must be recorded in SafetyPlan.
        """

        # ==================================================================
        # 1. BASE (from disclosure policy)
        # ==================================================================
        disclosure = ctx.base(
            field_name="communication.disclosure",
            target_field="knowledge_disclosure.disclosure_level",
            value=self.persona.disclosure_policy.base_openness,
            effect=f"Base disclosure from persona policy ({self.persona.disclosure_policy.base_openness:.2f})"
        )

        # ==================================================================
        # 2. TRAIT MODIFIER (extraversion)
        # ==================================================================
        extraversion_mod = self.traits.get_self_disclosure_modifier()
        disclosure = ctx.num(
            source_type="trait",
            source_id="extraversion",
            target_field="knowledge_disclosure.disclosure_level",
            operation="add",
            before=disclosure,
            after=disclosure + extraversion_mod,
            effect=f"Extraversion modifier: {extraversion_mod:+.2f}",
            weight=0.7,
            reason=f"E={self.persona.psychology.big_five.extraversion:.2f}"
        )

        # ==================================================================
        # 3. STATE MODIFIER (mood/stress/fatigue)
        # ==================================================================
        state_mod = self.state.get_disclosure_modifier()
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

        # ==================================================================
        # 3.5. TRUST MODIFIER (relationship store → disclosure)
        # ==================================================================
        # Fix 2: Wire trust into disclosure using the already-defined
        # disclosure_policy.factors["trust_level"] coefficient
        trust = self.memory.relationships.trust if self.memory else 0.5
        trust_factor = self.persona.disclosure_policy.factors.get("trust_level", 0.0)
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

        # ==================================================================
        # 4. PRIVACY FILTER CLAMP (P1: records in SafetyPlan)
        # ==================================================================
        privacy_filter = self.rules.get_privacy_filter_level(topic_signature)
        max_disclosure_privacy = 1.0 - max(self.persona.privacy_sensitivity, privacy_filter)

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

        # ==================================================================
        # 5. TOPIC SENSITIVITY CLAMP (P1: can chain with privacy clamp)
        # ==================================================================
        # Deterministic: iterate topic_sensitivities in list order
        for ts in self.persona.topic_sensitivities:
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
                # DESIGN: First-match-wins semantics. Topics are checked in
                # topic_sensitivities list order; only the first matching topic
                # applies its constraint. This is intentional for predictability.
                break

        # ==================================================================
        # 6. DISCLOSURE POLICY BOUNDS CLAMP
        # ==================================================================
        # Use the persona's declared bounds instead of hardcoded [0, 1].
        policy_bounds = self.persona.disclosure_policy.bounds
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

    def _compute_time_pressure(
        self,
        interaction_mode: InteractionMode,
        turn_number: int,
        ctx: "TraceContext",
    ) -> float:
        """
        Compute context-sensitive time pressure.

        Base from persona.time_scarcity, modified by interaction mode and
        conversation length. Debate/coaching reduce pressure; surveys increase it.
        """
        base = self.persona.time_scarcity

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

        # Conversation length: pressure builds after threshold
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

    def _infer_claim_type(
        self,
        proficiency: float,
        uncertainty_action: UncertaintyAction,
        domain: str,
        user_input: str = "",
    ) -> str:
        """Infer knowledge claim type"""
        # Check if domain matches persona's knowledge
        is_domain_specific = any(
            k.domain.lower() == domain.lower()
            for k in self.persona.knowledge_domains
        )

        # Detect if user is asking about personal experience
        is_personal_experience = self._detect_personal_experience(
            user_input, domain, is_domain_specific
        )

        return infer_knowledge_claim_type(
            proficiency,
            uncertainty_action,
            is_personal_experience,
            is_domain_specific
        )

    def _detect_personal_experience(
        self,
        user_input: str,
        domain: str,
        is_domain_specific: bool,
    ) -> bool:
        """Detect if user is asking about personal experience.

        Checks for personal-experience question patterns AND whether
        the persona has relevant domain knowledge or preferences.
        """
        if not user_input:
            return False

        input_lower = user_input.lower()

        # Personal experience question patterns
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

        # Only mark as personal experience if persona has relevant knowledge
        return is_domain_specific

    def _generate_intent(
        self,
        user_intent: str,
        conversation_goal: str,
        uncertainty_action: str,
        needs_clarification: bool = False,
        ctx: Optional["TraceContext"] = None
    ) -> str:
        """
        Generate meaningful response intent using template logic.

        Intent describes what the persona plans to do, not just cosmetic label.
        Uses user_intent + uncertainty_action to determine response approach.
        """
        return generate_intent_string(
            user_intent=user_intent,
            conversation_goal=conversation_goal,
            uncertainty_action=uncertainty_action,
            needs_clarification=needs_clarification,
            ctx=ctx
        )

    def _apply_patterns_safely(
        self,
        context: ConversationContext,
        disclosure_level: float,
        ctx: "TraceContext"  # Changed from citations
    ) -> None:
        """Apply response patterns with safety checks (P1: populates SafetyPlan)"""
        pattern = self.rules.check_response_pattern(context.user_input)

        if not pattern:
            return

        privacy_filter = self.rules.get_privacy_filter_level(context.topic_signature)
        modifications = apply_response_pattern_safely(
            pattern,
            disclosure_level,
            privacy_filter,
            self.persona.topic_sensitivities,
            self.persona.invariants.must_avoid,
            context.topic_signature
        )

        if modifications.get("pattern_blocked"):
            # P1: populate SafetyPlan
            ctx.activate_constraint("pattern_safety")
            ctx.block_pattern(
                pattern_trigger=pattern.trigger if hasattr(pattern, 'trigger') else 'unknown_pattern',
                reason=modifications["block_reason"]
            )

            # Keep citation for audit trail
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


def create_turn_planner(
    persona: Persona,
    determinism: DeterminismManager | None = None,
    memory_manager: MemoryManager | None = None,
) -> TurnPlanner:
    """Factory function to create turn planner"""
    return TurnPlanner(persona, determinism, memory_manager=memory_manager)
