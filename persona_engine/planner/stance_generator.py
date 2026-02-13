"""
Stance Generator with Expertise Guardrails

Generates persona stance on topics with built-in safety:
- Non-experts produce opinion/preference (NOT factual claims)
- Experts can make informed assertions
- All stance generation is value-driven

This is a P1 critical component that prevents non-experts from
making authoritative claims about topics outside their proficiency.
"""

from persona_engine.behavioral.cognitive_interpreter import CognitiveStyleInterpreter
from persona_engine.behavioral.values_interpreter import ValuesInterpreter
from persona_engine.planner.trace_context import TraceContext
from persona_engine.schema.persona_schema import Persona


def generate_stance_safe(
    persona: Persona,
    values: ValuesInterpreter,
    cognitive: CognitiveStyleInterpreter,
    user_input: str,
    topic_signature: str,
    proficiency: float,
    expert_allowed: bool,  # NOT knowledge_claim_type (that comes later)
    ctx: TraceContext,
) -> tuple[str, str]:
    """
    Generate stance with expertise guardrails.

    CRITICAL RULE: Non-experts produce opinions/preferences, NOT factual claims.

    Args:
        persona: Full persona object
        values: Values interpreter for value priorities
        cognitive: Cognitive interpreter for nuance
        user_input: User's question/statement
        topic_signature: Topic identifier
        proficiency: Domain proficiency (0-1)
        expert_allowed: Whether persona is allowed to make expert claims
                       (computed early: is_domain_specific AND proficiency >= threshold)
        ctx: TraceContext for recording decisions

    Returns:
        (stance_text, rationale)
    """

    # Get top values for value-driven reasoning
    top_values = values.get_top_values(n=3)
    primary_value = top_values[0][0]
    primary_strength = top_values[0][1]

    # Get cognitive nuance capacity
    nuance_capacity = cognitive.style.cognitive_complexity

    # ========================================================================
    # GUARDRAIL: Expert vs Opinion Mode
    # ========================================================================

    if expert_allowed:
        # Expert can make informed assertions
        stance = _generate_expert_stance(
            primary_value=primary_value,
            proficiency=proficiency,
            nuance_capacity=nuance_capacity,
            user_input=user_input
        )

        rationale = f"Informed view based on {proficiency:.2f} proficiency + {primary_value} value priority"

        # Citation: proficiency enables expert stance
        ctx.num(
            source_type="rule",
            source_id="domain_expertise",
            target_field="response_structure.confidence",
            operation="set",
            before=0.0,
            after=proficiency,
            effect=f"Domain expert ({proficiency:.2f} proficiency) can make informed assertions",
            weight=1.0,
            reason="expert_allowed=True"
        )

    else:
        # Non-expert: opinion-first approach
        stance = _generate_opinion_stance(
            primary_value=primary_value,
            primary_strength=primary_strength,
            nuance_capacity=nuance_capacity,
            proficiency=proficiency,
            user_input=user_input
        )

        rationale = f"Personal perspective based on {primary_value} value (strength {primary_strength:.2f})"

        # Add uncertainty markers for very low proficiency
        if proficiency < 0.5:
            rationale += f" | Low proficiency ({proficiency:.2f}), avoiding factual claims"

        # Citation: values drove stance (not expertise)
        ctx.enum(
            source_type="value",
            source_id=primary_value,
            target_field="response_structure.stance",
            operation="set",
            before="none",
            after=stance[:60],  # Abbreviated for citation
            effect=f"Stance driven by {primary_value} value",
            weight=0.9,
            reason="Non-expert: opinion-first approach"
        )

    return stance, rationale


def _generate_opinion_stance(
    primary_value: str,
    primary_strength: float,
    nuance_capacity: float,
    proficiency: float,
    user_input: str
) -> str:
    """
    Generate value-driven OPINION (not factual claim).

    Templates are explicitly subjective to prevent non-experts
    from sounding authoritative.
    """

    # Value-based opinion templates (subjective phrasing)
    value_templates = {
        "achievement": "I tend to favor approaches that demonstrate competence and measurable results",
        "benevolence": "I prefer solutions that prioritize people's wellbeing and collective benefit",
        "security": "I gravitate toward stable, proven methods that minimize risk",
        "self_direction": "I value autonomy and innovative approaches that allow creative freedom",
        "universalism": "I'm inclined toward solutions that consider broader societal impact",
        "hedonism": "I appreciate approaches that balance effectiveness with enjoyment",
        "stimulation": "I favor dynamic, novel solutions over conventional ones",
        "conformity": "I tend to prefer established norms and proven processes",
        "tradition": "I value methods that respect established practices",
        "power": "I favor approaches that demonstrate influence and capability",
    }

    base_stance = value_templates.get(
        primary_value,
        "I have mixed feelings about this"
    )

    # Adjust based on nuance capacity
    if nuance_capacity > 0.7:
        base_stance += ", though I can see merit in alternative approaches"

    # If user asked a question, acknowledge uncertainty for non-experts
    if "?" in user_input:
        if proficiency < 0.4:
            return f"{base_stance}, though I'm curious to understand the specific constraints here"
        else:
            return f"From my experience, {base_stance.lower()}"

    # If user is sharing, be more exploratory
    if any(word in user_input.lower() for word in ["i think", "i feel", "my view"]):
        return f"I see where you're coming from. {base_stance}"

    return base_stance


def _extract_topic_hint(user_input: str) -> str | None:
    """Extract a brief topic descriptor from user input for stance specificity."""
    import re

    user_lower = user_input.lower()

    # "What do you think about X?"
    match = re.search(r"(?:about|regarding|on|of)\s+(.{5,40}?)(?:\?|$)", user_lower)
    if match:
        return match.group(1).strip().rstrip("?.,")

    # "How should we approach X?"
    match = re.search(r"(?:approach|handle|deal with|manage|make|cook|prepare)\s+(.{5,30})", user_lower)
    if match:
        return match.group(1).strip().rstrip("?.,")

    # "What's the best way to X?"
    match = re.search(r"(?:best way to|how to|how do you)\s+(.{5,30})", user_lower)
    if match:
        return match.group(1).strip().rstrip("?.,")

    return None


def _generate_expert_stance(
    primary_value: str,
    proficiency: float,
    nuance_capacity: float,
    user_input: str
) -> str:
    """
    Generate expert stance (can include informed assertions).

    Still values-informed but can make factual claims based on expertise.
    Uses user_input to make stance topic-specific rather than generic.
    """
    topic_hint = _extract_topic_hint(user_input)

    # Expert templates — now input-aware
    if proficiency > 0.8:
        if topic_hint:
            stance = (
                f"When it comes to {topic_hint}, I have a clear perspective "
                f"shaped by hands-on experience and a focus on {primary_value}"
            )
        else:
            stance = (
                f"Based on my experience, I have a strong view on this, "
                f"particularly when {primary_value} is at stake"
            )
    elif proficiency > 0.6:
        if topic_hint:
            stance = f"In most cases I've encountered around {topic_hint}, {primary_value}-focused approaches tend to work"
        else:
            stance = f"In most cases I've seen, {primary_value}-driven solutions tend to succeed here"
    else:
        stance = f"I'd approach {topic_hint or 'this'} with {primary_value} in mind, though context matters significantly"

    # Add nuance for high cognitive complexity
    if nuance_capacity > 0.7:
        stance += ". That said, there are important tradeoffs to consider"

    return stance
