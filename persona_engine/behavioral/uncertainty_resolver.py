"""
Central Uncertainty Resolver

Single authoritative decision point for uncertainty_action to prevent
conflicting logic across interpreters.
"""


from persona_engine.schema.ir_schema import Citation, UncertaintyAction


def resolve_uncertainty_action(
    proficiency: float,
    confidence: float,
    risk_tolerance: float,
    need_for_closure: float,
    time_pressure: float,
    claim_policy_lookup_behavior: str,
    citations: list[Citation],
    stress: float = 0.0,
    fatigue: float = 0.0,
    cognitive_complexity: float = 0.5,
    competence: float = 0.5,
    has_adjacent_knowledge: bool = False,
) -> UncertaintyAction:
    """
    Single authoritative decision for uncertainty handling.

    Precedence (highest to lowest):
    1. Hard constraints (claim policy + proficiency)
    2. Time pressure (overrides preferences)
    3. Cognitive style (default behavior, modulated by stress/fatigue)

    Args:
        proficiency: Domain proficiency (0-1)
        confidence: Current confidence level (0-1)
        risk_tolerance: Willingness to take risks (0-1)
        need_for_closure: Desire for definite answers (0-1)
        time_pressure: Time scarcity (0-1)
        claim_policy_lookup_behavior: "ask" | "hedge" | "refuse" | "speculate"
        citations: List to append citations to
        stress: Current stress level (0-1). High stress lowers effective confidence.
        fatigue: Current fatigue level (0-1). High fatigue biases toward HEDGE/REFUSE.

    Returns:
        Resolved UncertaintyAction
    """
    # Apply dynamic state to effective confidence
    # High stress → second-guessing (lowers confidence)
    # High fatigue → avoids effortful reasoning (lowers confidence)
    stress_penalty = stress * 0.15
    fatigue_penalty = fatigue * 0.10
    effective_confidence = max(0.0, min(1.0, confidence - stress_penalty - fatigue_penalty))

    if stress > 0.3 or fatigue > 0.3:
        state_effects = []
        if stress > 0.3:
            state_effects.append(f"stress={stress:.2f}")
        if fatigue > 0.3:
            state_effects.append(f"fatigue={fatigue:.2f}")
        citations.append(Citation(
            source_type="state",
            source_id="dynamic_state",
            effect=f"Dynamic state ({', '.join(state_effects)}) reduces effective confidence: {confidence:.2f} → {effective_confidence:.2f}",
            weight=stress_penalty + fatigue_penalty,
        ))

    # 1. Hard constraint: Very low proficiency → enforce claim policy
    if proficiency < 0.3:
        if claim_policy_lookup_behavior == "refuse":
            citations.append(Citation(
                source_type="rule",
                source_id="claim_policy",
                effect=f"Proficiency {proficiency:.2f} < 0.3 → refuse per claim policy",
                weight=1.0
            ))
            return UncertaintyAction.REFUSE

        elif claim_policy_lookup_behavior == "hedge":
            citations.append(Citation(
                source_type="rule",
                source_id="claim_policy",
                effect=f"Low proficiency ({proficiency:.2f}) → hedge per claim policy",
                weight=1.0
            ))
            return UncertaintyAction.HEDGE

        elif claim_policy_lookup_behavior == "ask":
            citations.append(Citation(
                source_type="rule",
                source_id="claim_policy",
                effect="Low proficiency → ask per claim policy",
                weight=1.0
            ))
            return UncertaintyAction.ASK_CLARIFYING

        # Speculate if policy allows
        elif claim_policy_lookup_behavior == "speculate":
            citations.append(Citation(
                source_type="rule",
                source_id="claim_policy",
                effect=f"Low proficiency ({proficiency:.2f}) → speculate per claim policy",
                weight=0.9
            ))
            return UncertaintyAction.SPECULATE_WITH_DISCLAIMER

    # 2. Time pressure override (only if effective confidence is moderate)
    if time_pressure > 0.7 and effective_confidence > 0.4:
        citations.append(Citation(
            source_type="state",
            source_id="time_scarcity",
            effect=f"High time pressure ({time_pressure:.2f}) + moderate confidence → answer quickly",
            weight=0.8
        ))
        return UncertaintyAction.ANSWER

    # Reframe: very low confidence + high cognitive complexity
    if effective_confidence < 0.3 and cognitive_complexity > 0.7 and risk_tolerance > 0.4:
        citations.append(Citation(
            source_type="trait",
            source_id="cognitive_complexity",
            effect=f"High cognitive complexity ({cognitive_complexity:.2f}) + low confidence → reframe",
            weight=0.8
        ))
        return UncertaintyAction.REFRAME_QUESTION

    # 3. Cognitive style default behavior (uses effective_confidence)

    # High fatigue biases toward HEDGE/REFUSE regardless of confidence
    if fatigue > 0.7 and effective_confidence < 0.7:
        citations.append(Citation(
            source_type="state",
            source_id="fatigue",
            effect=f"High fatigue ({fatigue:.2f}) biases toward hedging",
            weight=0.7,
        ))
        return UncertaintyAction.HEDGE

    # High confidence → answer
    if effective_confidence > 0.7:
        return UncertaintyAction.ANSWER

    # Moderate confidence
    elif effective_confidence > 0.4:
        if risk_tolerance > 0.6:
            if competence < 0.7 and competence > 0.4:
                citations.append(Citation(
                    source_type="trait",
                    source_id="risk_tolerance",
                    effect=f"High risk tolerance ({risk_tolerance:.2f}) + moderate competence → speculate with disclaimer",
                    weight=0.7
                ))
                return UncertaintyAction.SPECULATE_WITH_DISCLAIMER
            citations.append(Citation(
                source_type="trait",
                source_id="risk_tolerance",
                effect=f"High risk tolerance ({risk_tolerance:.2f}) → answer despite moderate confidence",
                weight=0.7
            ))
            return UncertaintyAction.ANSWER
        elif competence > 0.4 and abs(competence - effective_confidence) < 0.25:
            citations.append(Citation(
                source_type="rule",
                source_id="partial_knowledge",
                effect=f"Moderate competence ({competence:.2f}) ≈ confidence ({effective_confidence:.2f}) → offer partial",
                weight=0.7
            ))
            return UncertaintyAction.OFFER_PARTIAL
        else:
            return UncertaintyAction.HEDGE

    # Low confidence
    else:
        # High need for closure → ask for clarity
        if need_for_closure > 0.6:
            citations.append(Citation(
                source_type="trait",
                source_id="need_for_closure",
                effect=f"High need for closure ({need_for_closure:.2f}) + low confidence → ask for clarity",
                weight=0.8
            ))
            return UncertaintyAction.ASK_CLARIFYING

        # Low risk tolerance → refuse
        elif risk_tolerance < 0.3:
            citations.append(Citation(
                source_type="trait",
                source_id="risk_tolerance",
                effect=f"Low risk tolerance ({risk_tolerance:.2f}) + low confidence → refuse",
                weight=0.9
            ))
            return UncertaintyAction.REFUSE

        elif has_adjacent_knowledge and cognitive_complexity > 0.4:
            citations.append(Citation(
                source_type="rule",
                source_id="adjacent_knowledge",
                effect="Low confidence but adjacent knowledge → acknowledge and redirect",
                weight=0.7
            ))
            return UncertaintyAction.ACKNOWLEDGE_AND_REDIRECT

        # Default: hedge
        else:
            return UncertaintyAction.HEDGE


def infer_knowledge_claim_type(
    proficiency: float,
    uncertainty_action: UncertaintyAction,
    is_personal_experience: bool = False,
    is_domain_specific: bool = False,
    cognitive_complexity: float = 0.5,
    is_second_hand: bool = False,
    is_hypothetical_context: bool = False,
    has_research_basis: bool = False,
) -> str:
    """
    Infer knowledge claim type from context.

    Args:
        proficiency: Domain proficiency
        uncertainty_action: How uncertainty is being handled
        is_personal_experience: Whether claim is from personal experience
        is_domain_specific: Whether claim requires domain expertise
        cognitive_complexity: Cognitive complexity level (0-1)
        is_second_hand: Whether knowledge is second-hand
        is_hypothetical_context: Whether context is hypothetical
        has_research_basis: Whether claim has research backing

    Returns:
        Knowledge claim type
    """
    if uncertainty_action == UncertaintyAction.REFUSE:
        return "none"
    if uncertainty_action == UncertaintyAction.DEFER_TO_AUTHORITY:
        return "none"
    if is_personal_experience:
        return "personal_experience"
    if is_domain_specific and proficiency > 0.7:
        if has_research_basis and cognitive_complexity >= 0.5:
            return "academic_cited"
        return "domain_expert"
    if is_hypothetical_context and cognitive_complexity >= 0.4:
        return "hypothetical"
    if is_second_hand:
        return "anecdotal"
    if proficiency > 0.4 and cognitive_complexity >= 0.5 and uncertainty_action == UncertaintyAction.HEDGE:
        return "inferential"
    if proficiency < 0.4 and uncertainty_action in (UncertaintyAction.ANSWER, UncertaintyAction.SPECULATE_WITH_DISCLAIMER):
        return "received_wisdom"
    if uncertainty_action in (UncertaintyAction.HEDGE, UncertaintyAction.ASK_CLARIFYING,
                              UncertaintyAction.OFFER_PARTIAL, UncertaintyAction.REFRAME_QUESTION):
        return "speculative"
    return "general_common_knowledge"
