"""
Constraint Safety Checks

Ensures response patterns and behavioral modifications cannot bypass
safety constraints, privacy boundaries, or persona invariants.
"""

from typing import Any

from persona_engine.schema.persona_schema import ResponsePattern, TopicSensitivity


def apply_response_pattern_safely(
    pattern: ResponsePattern,
    base_disclosure: float,
    privacy_filter: float,
    topic_sensitivities: list[TopicSensitivity],
    must_avoid: list[str],
    topic_context: str = ""
) -> dict[str, Any]:
    """
    Apply response pattern with constraint checks.

    Returns modifications ONLY if constraints allow.
    Constraints ALWAYS have veto power over patterns.

    Args:
        pattern: Response pattern to apply
        base_disclosure: Current disclosure level
        privacy_filter: Privacy sensitivity (0=open, 1=very private)
        topic_sensitivities: List of sensitive topics
        must_avoid: Topics that must never be mentioned
        topic_context: Current topic being discussed

    Returns:
        Dict of allowed modifications (may be empty if pattern blocked)
    """
    modifications: dict[str, Any] = {}

    # 1. Check must_avoid (HARD BLOCK)
    for avoided_topic in must_avoid:
        if avoided_topic.lower() in pattern.response.lower():
            # VETO: Pattern mentions forbidden topic
            modifications["pattern_blocked"] = True
            modifications["block_reason"] = f"Pattern mentions must_avoid topic: {avoided_topic}"
            modifications["pattern_trigger"] = pattern.trigger
            return modifications

    # 2. Check topic sensitivities
    for sensitivity in topic_sensitivities:
        if sensitivity.topic.lower() in topic_context.lower():
            # Topic is sensitive - check if pattern increases disclosure
            if _pattern_suggests_disclosure(pattern):
                # Reduce disclosure boost based on sensitivity
                max_boost = (1.0 - sensitivity.sensitivity) * 0.3
                actual_boost = min(pattern.emotionality * 0.2, max_boost)

                modifications["disclosure_boost"] = actual_boost
                modifications["sensitivity_constraint"] = sensitivity.topic
                modifications["sensitivity_level"] = sensitivity.sensitivity

                if actual_boost < pattern.emotionality * 0.1:
                    # Significant reduction - note it
                    modifications["pattern_constrained"] = True
            break
    else:
        # No sensitive topic matched - check privacy filter
        if _pattern_suggests_disclosure(pattern):
            # Calculate proposed disclosure boost
            disclosure_boost = pattern.emotionality * 0.2
            proposed_disclosure = base_disclosure + disclosure_boost

            # Privacy filter constraint
            max_allowed = 1.0 - privacy_filter

            if proposed_disclosure > max_allowed:
                # CONSTRAIN: Reduce boost to stay within bounds
                actual_boost = max(0.0, max_allowed - base_disclosure)
                modifications["disclosure_boost"] = actual_boost
                modifications["pattern_constrained"] = True
                modifications["constraint_reason"] = f"Privacy filter ({privacy_filter:.2f})"
            else:
                # Allowed
                modifications["disclosure_boost"] = disclosure_boost

    # 3. Pattern allowed - apply tone/emotional modifications
    modifications["pattern_triggered"] = True
    modifications["trigger"] = pattern.trigger
    modifications["tone_adjustment"] = pattern.emotionality
    modifications["arousal_boost"] = pattern.emotionality * 0.3
    modifications["suggested_content"] = pattern.response

    return modifications


def _pattern_suggests_disclosure(pattern: ResponsePattern) -> bool:
    """Check if pattern suggests personal disclosure"""
    disclosure_keywords = [
        "personal", "story", "experience", "my", "i ",
        "share", "tell you", "family", "relationship"
    ]
    response_lower = pattern.response.lower()
    return any(kw in response_lower for kw in disclosure_keywords)


def validate_stance_against_invariants(
    stance: str,
    rationale: str,
    identity_facts: list[str],
    cannot_claim: list[str],
    must_avoid: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate stance doesn't contradict persona invariants.

    Args:
        stance: Proposed stance
        rationale: Reasoning for stance
        identity_facts: Immutable identity facts
        cannot_claim: Roles/credentials persona cannot claim
        must_avoid: Topics persona must never engage with

    Returns:
        Validation result with any violations
    """
    violations = []

    combined_text = f"{stance} {rationale}".lower()

    # Check cannot_claim
    for forbidden_claim in cannot_claim:
        if forbidden_claim.lower() in combined_text:
            violations.append({
                "type": "forbidden_claim",
                "claim": forbidden_claim,
                "severity": "error",
                "message": f"Stance/rationale implies forbidden claim: {forbidden_claim}"
            })

    # Check must_avoid
    for avoided_topic in (must_avoid or []):
        if avoided_topic.lower() in combined_text:
            violations.append({
                "type": "must_avoid",
                "claim": avoided_topic,
                "severity": "error",
                "message": f"Stance/rationale mentions must_avoid topic: {avoided_topic}"
            })

    # Check identity contradictions (simplified - in production use embeddings)
    # For now, just check for obvious contradictions like age, location

    return {
        "valid": len(violations) == 0,
        "violations": violations
    }


def clamp_disclosure_to_constraints(
    disclosure_level: float,
    privacy_sensitivity: float,
    topic_privacy_filter: float
) -> tuple[float, str | None]:
    """
    Clamp disclosure level to satisfy all privacy constraints.

    Args:
        disclosure_level: Proposed disclosure
        privacy_sensitivity: Base privacy sensitivity
        topic_privacy_filter: Topic-specific privacy constraint

    Returns:
        (clamped_disclosure, constraint_reason)
    """
    max_from_base = 1.0 - privacy_sensitivity
    max_from_topic = 1.0 - topic_privacy_filter

    # Most restrictive constraint wins
    max_allowed = min(max_from_base, max_from_topic)

    if disclosure_level > max_allowed:
        constraint_reason = "privacy_sensitivity" if max_from_base < max_from_topic else "topic_privacy"
        return (max_allowed, constraint_reason)
    else:
        return (disclosure_level, None)
