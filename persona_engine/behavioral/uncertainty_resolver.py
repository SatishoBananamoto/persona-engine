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
    citations: list[Citation]
) -> UncertaintyAction:
    """
    Single authoritative decision for uncertainty handling.

    Precedence (highest to lowest):
    1. Hard constraints (claim policy + proficiency)
    2. Time pressure (overrides preferences)
    3. Cognitive style (default behavior)

    Args:
        proficiency: Domain proficiency (0-1)
        confidence: Current confidence level (0-1)
        risk_tolerance: Willingness to take risks (0-1)
        need_for_closure: Desire for definite answers (0-1)
        time_pressure: Time scarcity (0-1)
        claim_policy_lookup_behavior: "ask" | "hedge" | "refuse" | "speculate"
        citations: List to append citations to

    Returns:
        Resolved UncertaintyAction
    """

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

    # 2. Time pressure override (only if confidence is moderate)
    if time_pressure > 0.7 and confidence > 0.4:
        citations.append(Citation(
            source_type="state",
            source_id="time_scarcity",
            effect=f"High time pressure ({time_pressure:.2f}) + moderate confidence → answer quickly",
            weight=0.8
        ))
        return UncertaintyAction.ANSWER

    # 3. Cognitive style default behavior

    # High confidence → answer
    if confidence > 0.7:
        return UncertaintyAction.ANSWER

    # Moderate confidence
    elif confidence > 0.4:
        if risk_tolerance > 0.6:
            citations.append(Citation(
                source_type="trait",
                source_id="risk_tolerance",
                effect=f"High risk tolerance ({risk_tolerance:.2f}) → answer despite moderate confidence",
                weight=0.7
            ))
            return UncertaintyAction.ANSWER
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

        # Default: hedge
        else:
            return UncertaintyAction.HEDGE


def infer_knowledge_claim_type(
    proficiency: float,
    uncertainty_action: UncertaintyAction,
    is_personal_experience: bool = False,
    is_domain_specific: bool = False
) -> str:
    """
    Infer knowledge claim type from context.

    Args:
        proficiency: Domain proficiency
        uncertainty_action: How uncertainty is being handled
        is_personal_experience: Whether claim is from personal experience
        is_domain_specific: Whether claim requires domain expertise

    Returns:
        Knowledge claim type
    """

    if uncertainty_action == UncertaintyAction.REFUSE:
        return "none"

    if is_personal_experience:
        return "personal_experience"

    if is_domain_specific and proficiency > 0.7:
        return "domain_expert"

    if uncertainty_action in [UncertaintyAction.HEDGE, UncertaintyAction.ASK_CLARIFYING]:
        return "speculative"

    # Default: common knowledge
    return "general_common_knowledge"
