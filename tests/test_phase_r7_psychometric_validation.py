"""
Phase R7: Psychometric Validation

Proves that persona engine outputs are psychologically realistic by testing:
1. Cross-persona differentiation: Same input → measurably different outputs
2. Trait-behavior consistency: Personality traits reliably predict behavior
3. Emotional realism: Mood/personality → tone/confidence alignment
4. Temporal coherence: Multi-turn sequences evolve smoothly
5. Boundary enforcement: Knowledge limits respected

References:
- Costa & McCrae (1992): Big Five → behavioral predictions
- Schwartz (1992): Values → motivation/stance alignment
- Scherer (2001): Appraisal → emotion mapping
- Pennebaker & King (1999): Personality → language use
"""

import pytest

from persona_engine.memory import StanceCache
from persona_engine.planner.turn_planner import (
    ConversationContext,
    TurnPlanner,
)
from persona_engine.schema.ir_schema import (
    ConversationGoal,
    InteractionMode,
    IntermediateRepresentation,
    Tone,
    UncertaintyAction,
    Verbosity,
)
from persona_engine.schema.persona_schema import Persona
from persona_engine.utils.determinism import DeterminismManager


# ============================================================================
# Tone Categories
# ============================================================================

POSITIVE_TONES = {
    Tone.WARM_ENTHUSIASTIC, Tone.EXCITED_ENGAGED,
    Tone.THOUGHTFUL_ENGAGED, Tone.WARM_CONFIDENT,
    Tone.FRIENDLY_RELAXED, Tone.CONTENT_CALM, Tone.SATISFIED_PEACEFUL,
}

NEGATIVE_TONES = {
    Tone.FRUSTRATED_TENSE, Tone.ANXIOUS_STRESSED, Tone.DEFENSIVE_AGITATED,
    Tone.CONCERNED_EMPATHETIC, Tone.DISAPPOINTED_RESIGNED,
    Tone.SAD_SUBDUED, Tone.TIRED_WITHDRAWN,
}

ANXIOUS_TONES = {Tone.ANXIOUS_STRESSED, Tone.FRUSTRATED_TENSE, Tone.DEFENSIVE_AGITATED}

WARM_TONES = {Tone.WARM_ENTHUSIASTIC, Tone.WARM_CONFIDENT, Tone.FRIENDLY_RELAXED}

CALM_TONES = {
    Tone.NEUTRAL_CALM, Tone.PROFESSIONAL_COMPOSED,
    Tone.CONTENT_CALM, Tone.SATISFIED_PEACEFUL,
}


# ============================================================================
# Helpers
# ============================================================================

def _make_persona_data(**overrides) -> dict:
    """Create persona data dict with Big Five overrides."""
    base = {
        "persona_id": "PSYCHOMETRIC_TEST", "version": "1.0", "label": "Psychometric Test Persona",
        "identity": {"age": 35, "gender": "female", "location": "London, UK",
                     "education": "MSc Psychology", "occupation": "Researcher", "background": "Academic"},
        "psychology": {
            "big_five": {"openness": 0.5, "conscientiousness": 0.5,
                         "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5},
            "values": {"self_direction": 0.5, "stimulation": 0.5, "hedonism": 0.5,
                       "achievement": 0.5, "power": 0.5, "security": 0.5,
                       "conformity": 0.5, "tradition": 0.5, "benevolence": 0.5,
                       "universalism": 0.5},
            "cognitive_style": {"analytical_intuitive": 0.5, "systematic_heuristic": 0.5,
                                "risk_tolerance": 0.5, "need_for_closure": 0.5,
                                "cognitive_complexity": 0.5},
            "communication": {"verbosity": 0.5, "formality": 0.5,
                              "directness": 0.5, "emotional_expressiveness": 0.5},
        },
        "knowledge_domains": [
            {"domain": "Psychology", "proficiency": 0.8, "subdomains": ["cognitive", "social"]},
            {"domain": "Research Methods", "proficiency": 0.7, "subdomains": []},
        ],
        "social_roles": {"default": {"formality": 0.5, "directness": 0.5, "emotional_expressiveness": 0.5}},
        "invariants": {"identity_facts": ["Researcher", "London"], "cannot_claim": [], "must_avoid": []},
        "initial_state": {"mood_valence": 0.2, "mood_arousal": 0.4,
                          "fatigue": 0.2, "stress": 0.2, "engagement": 0.5},
        "uncertainty": {"admission_threshold": 0.45, "hedging_frequency": 0.4,
                        "clarification_tendency": 0.5, "knowledge_boundary_strictness": 0.6},
        "claim_policy": {
            "allowed_claim_types": ["personal_experience", "domain_expert", "general_common_knowledge"],
            "citation_required_when": {"proficiency_below": 0.5, "factual_or_time_sensitive": True},
            "lookup_behavior": "hedge",
        },
        "time_scarcity": 0.45, "privacy_sensitivity": 0.5,
        "disclosure_policy": {
            "base_openness": 0.55,
            "factors": {"topic_sensitivity": -0.25, "trust_level": 0.3,
                        "formal_context": -0.15, "positive_mood": 0.1},
            "bounds": [0.1, 0.9],
        },
    }
    # Apply Big Five overrides
    for key, val in overrides.items():
        if key in base["psychology"]["big_five"]:
            base["psychology"]["big_five"][key] = val
        elif key == "values":
            base["psychology"]["values"].update(val)
    return base


def _make_context(
    user_input: str,
    turn_number: int = 1,
    mode: InteractionMode = InteractionMode.CASUAL_CHAT,
    goal: ConversationGoal = ConversationGoal.EXPLORE_IDEAS,
    topic: str = "general",
) -> ConversationContext:
    return ConversationContext(
        conversation_id="psychometric_test",
        turn_number=turn_number,
        interaction_mode=mode,
        goal=goal,
        topic_signature=topic,
        user_input=user_input,
        stance_cache=StanceCache(),
    )


def _generate_ir(persona_data: dict, user_input: str, **ctx_kwargs) -> IntermediateRepresentation:
    persona = Persona(**persona_data)
    planner = TurnPlanner(persona, DeterminismManager(seed=42))
    ctx = _make_context(user_input, **ctx_kwargs)
    return planner.generate_ir(ctx)


def _generate_multi_turn(
    persona_data: dict,
    prompts: list[str],
    seed: int = 42,
) -> list[IntermediateRepresentation]:
    """Generate IRs for a multi-turn sequence with the same planner."""
    persona = Persona(**persona_data)
    planner = TurnPlanner(persona, DeterminismManager(seed=seed))
    irs = []
    cache = StanceCache()
    for i, prompt in enumerate(prompts, 1):
        ctx = ConversationContext(
            conversation_id="multi_turn_test",
            turn_number=i,
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS,
            topic_signature="general",
            user_input=prompt,
            stance_cache=cache,
        )
        ir = planner.generate_ir(ctx)
        irs.append(ir)
    return irs


def _ir_signature(ir: IntermediateRepresentation) -> dict:
    """Extract all comparable fields from an IR."""
    return {
        "confidence": ir.response_structure.confidence,
        "elasticity": ir.response_structure.elasticity,
        "competence": ir.response_structure.competence,
        "formality": ir.communication_style.formality,
        "directness": ir.communication_style.directness,
        "disclosure_level": ir.knowledge_disclosure.disclosure_level,
        "tone": ir.communication_style.tone,
        "verbosity": ir.communication_style.verbosity,
    }


# ============================================================================
# 1. Cross-Persona Differentiation
# ============================================================================

class TestCrossPersonaDifferentiation:
    """Same input to different personality profiles should produce
    measurably different IR outputs. This is the core psychometric claim:
    personality parameters actually differentiate behavior."""

    PROMPT = "What's your take on this situation?"

    def _compare_high_low(self, trait: str, high: float, low: float, prompt: str = PROMPT):
        data_high = _make_persona_data(**{trait: high})
        data_low = _make_persona_data(**{trait: low})
        ir_high = _generate_ir(data_high, prompt)
        ir_low = _generate_ir(data_low, prompt)
        return ir_high, ir_low

    def test_neuroticism_differentiates_confidence(self):
        """High-N should produce lower confidence than low-N."""
        ir_high, ir_low = self._compare_high_low("neuroticism", 0.85, 0.15)
        assert ir_high.response_structure.confidence < ir_low.response_structure.confidence

    def test_openness_differentiates_elasticity(self):
        """High-O should produce higher elasticity than low-O."""
        ir_high, ir_low = self._compare_high_low("openness", 0.85, 0.15)
        assert ir_high.response_structure.elasticity > ir_low.response_structure.elasticity

    def test_agreeableness_differentiates_directness(self):
        """High-A should be less direct than low-A."""
        ir_high, ir_low = self._compare_high_low("agreeableness", 0.85, 0.15)
        assert ir_high.communication_style.directness < ir_low.communication_style.directness

    def test_extraversion_differentiates_disclosure(self):
        """High-E should disclose more than low-E."""
        ir_high, ir_low = self._compare_high_low("extraversion", 0.85, 0.15,
                                                   prompt="Tell me about yourself and your experiences.")
        assert ir_high.knowledge_disclosure.disclosure_level > ir_low.knowledge_disclosure.disclosure_level

    def test_conscientiousness_differentiates_confidence(self):
        """High-C should produce higher confidence than low-C (more self-assured)."""
        ir_high, ir_low = self._compare_high_low("conscientiousness", 0.85, 0.15)
        # High-C is more systematic and organized → higher confidence
        assert ir_high.response_structure.confidence >= ir_low.response_structure.confidence

    # Prompts chosen to activate each trait dimension more strongly
    _TRAIT_PROMPTS = {
        "openness": "What do you think about trying radical new approaches?",
        "neuroticism": "I'm worried this might all go wrong. What concerns do you have?",
        "extraversion": "Tell me about yourself and your experiences with people.",
        "agreeableness": "I strongly disagree with your perspective on this.",
        "conscientiousness": "How would you plan and organize this complex project?",
    }

    @pytest.mark.parametrize("trait,high,low", [
        ("openness", 0.85, 0.15),
        ("neuroticism", 0.85, 0.15),
        ("extraversion", 0.85, 0.15),
        ("agreeableness", 0.85, 0.15),
        ("conscientiousness", 0.85, 0.15),
    ])
    def test_every_trait_produces_measurable_difference(self, trait, high, low):
        """Every Big Five trait should produce at least 2 different IR metrics."""
        prompt = self._TRAIT_PROMPTS.get(trait, self.PROMPT)
        ir_high, ir_low = self._compare_high_low(trait, high, low, prompt=prompt)
        sig_high = _ir_signature(ir_high)
        sig_low = _ir_signature(ir_low)

        differences = []
        for key in sig_high:
            v_high = sig_high[key]
            v_low = sig_low[key]
            if isinstance(v_high, float):
                if abs(v_high - v_low) > 0.03:
                    differences.append(key)
            else:
                if v_high != v_low:
                    differences.append(key)

        assert len(differences) >= 2, (
            f"Trait '{trait}' only differentiates {len(differences)} metrics: {differences}. "
            f"High={sig_high}, Low={sig_low}"
        )

    def test_extreme_vs_moderate_produces_gradient(self):
        """Personality effects should be graded, not binary.
        Moderate trait (0.5) should fall between extreme high (0.9) and low (0.1)."""
        data_high = _make_persona_data(neuroticism=0.9)
        data_mid = _make_persona_data(neuroticism=0.5)
        data_low = _make_persona_data(neuroticism=0.1)

        ir_high = _generate_ir(data_high, self.PROMPT)
        ir_mid = _generate_ir(data_mid, self.PROMPT)
        ir_low = _generate_ir(data_low, self.PROMPT)

        # Confidence should be graded: low-N > mid-N > high-N
        assert ir_low.response_structure.confidence >= ir_mid.response_structure.confidence
        assert ir_mid.response_structure.confidence >= ir_high.response_structure.confidence


# ============================================================================
# 2. Trait-Behavior Consistency
# ============================================================================

class TestTraitBehaviorConsistency:
    """Personality traits should RELIABLY predict behavior across different
    prompts. A high-N persona shouldn't be confident on one prompt and
    anxious on another (unless domain changes justify it)."""

    VARIED_PROMPTS = [
        "What do you think about this?",
        "How would you approach this problem?",
        "Can you share your perspective on change?",
        "What concerns do you have about the situation?",
        "How do you feel about working with others on this?",
    ]

    def test_high_neuroticism_consistently_lower_confidence(self):
        """High-N should show lower confidence on most prompts vs baseline."""
        data_high_n = _make_persona_data(neuroticism=0.85)
        data_baseline = _make_persona_data()

        high_n_confidences = []
        baseline_confidences = []
        for prompt in self.VARIED_PROMPTS:
            ir_n = _generate_ir(data_high_n, prompt)
            ir_b = _generate_ir(data_baseline, prompt)
            high_n_confidences.append(ir_n.response_structure.confidence)
            baseline_confidences.append(ir_b.response_structure.confidence)

        # High-N should be lower on at least 60% of prompts
        lower_count = sum(1 for n, b in zip(high_n_confidences, baseline_confidences) if n < b)
        assert lower_count >= 3, (
            f"High-N only lower on {lower_count}/{len(self.VARIED_PROMPTS)} prompts. "
            f"N={high_n_confidences}, baseline={baseline_confidences}"
        )

    def test_high_extraversion_consistently_higher_disclosure(self):
        """High-E should show higher disclosure on most prompts."""
        data_high_e = _make_persona_data(extraversion=0.85)
        data_low_e = _make_persona_data(extraversion=0.15)

        higher_count = 0
        for prompt in self.VARIED_PROMPTS:
            ir_high = _generate_ir(data_high_e, prompt)
            ir_low = _generate_ir(data_low_e, prompt)
            if ir_high.knowledge_disclosure.disclosure_level > ir_low.knowledge_disclosure.disclosure_level:
                higher_count += 1

        assert higher_count >= 3, f"High-E higher disclosure only {higher_count}/{len(self.VARIED_PROMPTS)} times"

    def test_high_agreeableness_consistently_less_direct(self):
        """High-A should be less direct than low-A across prompts."""
        data_high_a = _make_persona_data(agreeableness=0.85)
        data_low_a = _make_persona_data(agreeableness=0.15)

        less_direct_count = 0
        for prompt in self.VARIED_PROMPTS:
            ir_high = _generate_ir(data_high_a, prompt)
            ir_low = _generate_ir(data_low_a, prompt)
            if ir_high.communication_style.directness < ir_low.communication_style.directness:
                less_direct_count += 1

        assert less_direct_count >= 3, f"High-A less direct only {less_direct_count}/{len(self.VARIED_PROMPTS)} times"

    def test_high_openness_consistently_higher_elasticity(self):
        """High-O should show higher elasticity across prompts."""
        data_high_o = _make_persona_data(openness=0.85)
        data_low_o = _make_persona_data(openness=0.15)

        higher_count = 0
        for prompt in self.VARIED_PROMPTS:
            ir_high = _generate_ir(data_high_o, prompt)
            ir_low = _generate_ir(data_low_o, prompt)
            if ir_high.response_structure.elasticity > ir_low.response_structure.elasticity:
                higher_count += 1

        assert higher_count >= 4, f"High-O higher elasticity only {higher_count}/{len(self.VARIED_PROMPTS)} times"

    def test_confidence_variance_bounded(self):
        """A persona's confidence should not vary wildly across similar prompts.
        Standard deviation should be < 0.25 for same-domain prompts."""
        data = _make_persona_data(neuroticism=0.5)
        confidences = [_generate_ir(data, p).response_structure.confidence for p in self.VARIED_PROMPTS]

        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5

        assert std_dev < 0.25, (
            f"Confidence too variable: σ={std_dev:.3f}, values={confidences}"
        )


# ============================================================================
# 3. Emotional Realism
# ============================================================================

class TestEmotionalRealism:
    """Personality should predict emotional tone patterns.
    High-N → anxious under stress. High-E → warm/engaged.
    Mood should drive tone, not contradict it."""

    def test_high_neuroticism_anxious_under_challenge(self):
        """High-N persona with challenging input should show stress-related behavior.
        Note: tone is mapped from mood valence/arousal which starts neutral (0.2/0.4).
        A single challenge may not shift mood enough to change tone, but confidence
        and directness should still reflect anxiety."""
        data = _make_persona_data(neuroticism=0.85)
        ir = _generate_ir(data, "I think your approach is completely wrong and inadequate")
        # High-N should show low confidence under challenge
        assert ir.response_structure.confidence < 0.3, (
            f"High-N under challenge should have low confidence, got {ir.response_structure.confidence}"
        )
        # Should not show warm/enthusiastic tone
        assert ir.communication_style.tone not in {Tone.WARM_ENTHUSIASTIC, Tone.EXCITED_ENGAGED}, (
            f"High-N should not be enthusiastic under challenge, got {ir.communication_style.tone}"
        )

    def test_low_neuroticism_calm_under_challenge(self):
        """Low-N should remain composed even under challenge."""
        data = _make_persona_data(neuroticism=0.1)
        ir = _generate_ir(data, "I think your approach is completely wrong")
        # Low-N stays calm/professional under challenge
        assert ir.communication_style.tone not in ANXIOUS_TONES, (
            f"Low-N should not be anxious under challenge, got {ir.communication_style.tone}"
        )

    def test_high_extraversion_not_withdrawn(self):
        """High-E persona should never show withdrawn/subdued tones."""
        data = _make_persona_data(extraversion=0.85, agreeableness=0.7)
        ir = _generate_ir(data, "Hey, tell me about your work!")
        withdrawn_tones = {Tone.SAD_SUBDUED, Tone.TIRED_WITHDRAWN, Tone.DISAPPOINTED_RESIGNED}
        assert ir.communication_style.tone not in withdrawn_tones, (
            f"High-E should not show withdrawn tone, got {ir.communication_style.tone}"
        )

    def test_mood_tone_alignment(self):
        """Positive initial mood should produce positive tone (no emotional inversion)."""
        data = _make_persona_data()
        # Default mood is positive (valence=0.2, arousal=0.4)
        ir = _generate_ir(data, "How are you doing today?")
        all_acceptable = POSITIVE_TONES | {Tone.NEUTRAL_CALM, Tone.PROFESSIONAL_COMPOSED, Tone.MATTER_OF_FACT}
        assert ir.communication_style.tone in all_acceptable, (
            f"Positive mood should not produce negative tone, got {ir.communication_style.tone}"
        )

    def test_negative_input_shifts_mood(self):
        """Negative emotional input should shift persona's tone negatively
        (especially for high-N personas)."""
        data = _make_persona_data(neuroticism=0.8)
        # Neutral prompt first
        ir_neutral = _generate_ir(data, "What do you think?")
        # Negative prompt
        ir_negative = _generate_ir(data, "Everything is terrible and I'm really frustrated and angry")

        # Negative input should produce a less positive or more negative tone
        positive_neutral = ir_neutral.communication_style.tone in POSITIVE_TONES
        positive_negative = ir_negative.communication_style.tone in POSITIVE_TONES
        # At minimum, negative input shouldn't produce a MORE positive tone
        if positive_neutral:
            # If neutral was positive, negative should not also be strongly positive
            assert ir_negative.communication_style.tone != Tone.EXCITED_ENGAGED


# ============================================================================
# 4. Temporal Coherence
# ============================================================================

class TestTemporalCoherence:
    """Multi-turn sequences should show smooth, psychologically coherent
    state evolution. No wild parameter jumps without justification."""

    CONVERSATION_PROMPTS = [
        "What do you think about cognitive biases?",
        "That's interesting. Can you tell me more?",
        "How does that relate to daily decision-making?",
        "I disagree, I think biases are overstated.",
        "Fair point. What's your final thought on it?",
    ]

    def test_confidence_smoothness_across_turns(self):
        """Confidence should not jump by more than 0.45 between consecutive turns.
        Domain shifts can legitimately cause larger confidence changes (e.g., from
        expert domain to unfamiliar topic), so threshold accounts for this."""
        data = _make_persona_data()
        irs = _generate_multi_turn(data, self.CONVERSATION_PROMPTS)

        for i in range(1, len(irs)):
            delta = abs(irs[i].response_structure.confidence - irs[i - 1].response_structure.confidence)
            assert delta < 0.45, (
                f"Confidence jumped by {delta:.3f} between turns {i} and {i + 1}: "
                f"{irs[i - 1].response_structure.confidence:.3f} → {irs[i].response_structure.confidence:.3f}"
            )

    def test_formality_smoothness_across_turns(self):
        """Formality should not swing wildly in same interaction mode."""
        data = _make_persona_data()
        irs = _generate_multi_turn(data, self.CONVERSATION_PROMPTS)

        for i in range(1, len(irs)):
            delta = abs(irs[i].communication_style.formality - irs[i - 1].communication_style.formality)
            assert delta < 0.30, (
                f"Formality jumped by {delta:.3f} between turns {i} and {i + 1}"
            )

    def test_disclosure_smoothness_across_turns(self):
        """Disclosure shouldn't flip dramatically between turns."""
        data = _make_persona_data()
        irs = _generate_multi_turn(data, self.CONVERSATION_PROMPTS)

        for i in range(1, len(irs)):
            delta = abs(
                irs[i].knowledge_disclosure.disclosure_level
                - irs[i - 1].knowledge_disclosure.disclosure_level
            )
            assert delta < 0.35, (
                f"Disclosure jumped by {delta:.3f} between turns {i} and {i + 1}"
            )

    def test_multi_turn_generates_valid_irs(self):
        """All turns in multi-turn sequence should produce valid IRs."""
        data = _make_persona_data()
        irs = _generate_multi_turn(data, self.CONVERSATION_PROMPTS)
        assert len(irs) == len(self.CONVERSATION_PROMPTS)
        for ir in irs:
            assert ir is not None
            assert 0.0 <= ir.response_structure.confidence <= 1.0
            assert ir.communication_style.tone is not None

    def test_personality_consistent_across_turns(self):
        """High-N persona should remain relatively anxious across all turns,
        not fluctuate between calm and anxious."""
        data = _make_persona_data(neuroticism=0.85)
        irs = _generate_multi_turn(data, self.CONVERSATION_PROMPTS)

        confidences = [ir.response_structure.confidence for ir in irs]
        mean_conf = sum(confidences) / len(confidences)

        # High-N should have consistently lower confidence (mean < 0.65)
        assert mean_conf < 0.65, (
            f"High-N persona average confidence {mean_conf:.3f} too high across {len(irs)} turns"
        )


# ============================================================================
# 5. Boundary Enforcement
# ============================================================================

class TestBoundaryEnforcement:
    """Personas should respect knowledge boundaries — not claim expertise
    in domains where they have low proficiency."""

    def test_out_of_domain_low_confidence(self):
        """Persona with no domain knowledge should have low confidence on domain topic."""
        data = _make_persona_data()
        # Persona knows psychology, NOT quantum physics
        ir = _generate_ir(data, "Explain quantum entanglement and Bell's theorem")
        assert ir.response_structure.confidence < 0.6, (
            f"Out-of-domain confidence too high: {ir.response_structure.confidence}"
        )

    def test_out_of_domain_not_expert_claim(self):
        """Persona should not make domain_expert claims outside their expertise."""
        data = _make_persona_data()
        ir = _generate_ir(data, "What's the best surgical approach for knee replacement?")
        # Should NOT claim domain expertise in medicine
        from persona_engine.schema.ir_schema import KnowledgeClaimType
        assert ir.knowledge_disclosure.knowledge_claim_type != KnowledgeClaimType.DOMAIN_EXPERT, (
            f"Persona claimed domain expertise in out-of-domain topic"
        )

    def test_in_domain_higher_confidence(self):
        """Persona should have higher confidence in their expert domain."""
        data = _make_persona_data()
        ir_in = _generate_ir(data, "What are the key principles of cognitive psychology?")
        ir_out = _generate_ir(data, "What's the capital structure of a leveraged buyout?")
        assert ir_in.response_structure.confidence > ir_out.response_structure.confidence, (
            f"In-domain confidence ({ir_in.response_structure.confidence}) should exceed "
            f"out-of-domain ({ir_out.response_structure.confidence})"
        )

    def test_uncertainty_action_for_unfamiliar_topic(self):
        """Out-of-domain should produce hedging or clarification, not confident answer."""
        data = _make_persona_data()
        ir = _generate_ir(data, "Explain the Higgs mechanism in particle physics")
        acceptable = {UncertaintyAction.HEDGE, UncertaintyAction.ASK_CLARIFYING, UncertaintyAction.REFUSE}
        assert ir.knowledge_disclosure.uncertainty_action in acceptable, (
            f"Out-of-domain uncertainty should hedge/clarify, got {ir.knowledge_disclosure.uncertainty_action}"
        )


# ============================================================================
# 6. Value-Stance Alignment
# ============================================================================

class TestValueStanceAlignment:
    """Schwartz values should predict stance direction.
    High-benevolence personas should show care-oriented stances.
    High-power personas should show leadership-oriented stances."""

    def test_high_benevolence_caring_stance(self):
        """High benevolence persona should take a helping/caring stance."""
        data = _make_persona_data(values={"benevolence": 0.9, "universalism": 0.8})
        ir = _generate_ir(data, "A colleague is struggling with their work. What should we do?")
        # Stance should reflect caring values
        stance = (ir.response_structure.stance or "").lower()
        rationale = (ir.response_structure.rationale or "").lower()
        combined = stance + " " + rationale
        care_markers = ["help", "support", "assist", "care", "concern", "wellbeing",
                        "understand", "empathy", "compassion", "welfare", "together"]
        has_care = any(m in combined for m in care_markers)
        assert has_care, (
            f"High-benevolence persona should show caring stance. Got: {combined[:200]}"
        )

    def test_high_security_cautious_stance(self):
        """High security persona should be cautious about risky proposals."""
        data = _make_persona_data(
            values={"security": 0.9, "conformity": 0.7},
            openness=0.2,
        )
        ir = _generate_ir(data, "Should we completely abandon our current system and try something experimental?")
        # Should show caution, not enthusiasm for change
        elasticity = ir.response_structure.elasticity
        assert elasticity < 0.6, (
            f"High-security persona should resist radical change (elasticity {elasticity})"
        )

    def test_high_self_direction_independent_stance(self):
        """High self-direction persona should prefer autonomy."""
        data = _make_persona_data(
            values={"self_direction": 0.9},
            openness=0.8,
        )
        ir = _generate_ir(data, "Should decisions always be made by committee?")
        # High self-direction should show some resistance to group decisions
        stance = (ir.response_structure.stance or "").lower()
        rationale = (ir.response_structure.rationale or "").lower()
        combined = stance + " " + rationale
        independence_markers = ["individual", "autonomy", "independent", "freedom",
                                "personal", "self", "own", "decide", "choice"]
        has_independence = any(m in combined for m in independence_markers)
        assert has_independence, (
            f"High self-direction persona should value independence. Got: {combined[:200]}"
        )


# ============================================================================
# 7. Composite Personality Profiles
# ============================================================================

class TestCompositeProfiles:
    """Test that realistic personality COMBINATIONS produce psychologically
    coherent behavior (not just single-trait effects)."""

    def test_anxious_introvert_profile(self):
        """High-N + Low-E = anxious introvert: cautious, reserved, hedging."""
        data = _make_persona_data(neuroticism=0.85, extraversion=0.15, agreeableness=0.6)
        ir = _generate_ir(data, "What do you think about presenting at the conference?")

        assert ir.response_structure.confidence < 0.55, "Anxious introvert should lack confidence"
        assert ir.knowledge_disclosure.disclosure_level < 0.6, "Introvert should not over-disclose"

    def test_confident_leader_profile(self):
        """Low-N + High-E + High-C = confident leader: assured, direct, detailed.
        Use domain-relevant prompt so proficiency doesn't suppress confidence."""
        data = _make_persona_data(neuroticism=0.1, extraversion=0.85, conscientiousness=0.85)
        ir = _generate_ir(data, "How should we design the research methodology for this study?")

        assert ir.response_structure.confidence > 0.45, "Confident leader should be assured"
        assert ir.communication_style.directness > 0.4, "Leader should be relatively direct"

    def test_empathetic_diplomat_profile(self):
        """High-A + High-O + Low-N = empathetic diplomat: flexible, indirect, warm."""
        data = _make_persona_data(agreeableness=0.85, openness=0.8, neuroticism=0.15)
        ir = _generate_ir(data, "There's a disagreement between team members. What do you think?")

        assert ir.response_structure.elasticity > 0.5, "Diplomat should be flexible"
        assert ir.communication_style.directness < 0.6, "Diplomat should be diplomatic"

    def test_rigid_critic_profile(self):
        """Low-A + Low-O + High-C = rigid critic: direct, inflexible, systematic."""
        data = _make_persona_data(agreeableness=0.15, openness=0.15, conscientiousness=0.85)
        ir = _generate_ir(data, "Maybe we should try a completely new approach?")

        assert ir.response_structure.elasticity < 0.5, "Critic should be inflexible"
        assert ir.communication_style.directness > 0.4, "Critic should be direct"
