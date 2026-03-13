"""Prompt builder - converts IR fields into structured LLM system prompts.

This is the core of the response generation layer. Every IR field maps to
a specific behavioral instruction in the system prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona_engine.schema.persona_schema import Persona

from persona_engine.schema.ir_schema import IntermediateRepresentation

# =============================================================================
# Tone → Prompt Mapping
# =============================================================================

TONE_PROMPTS: dict[str, str] = {
    # Positive valence, high arousal
    "warm_enthusiastic": (
        "Speak with warmth and genuine enthusiasm. "
        "Use encouraging language and show excitement."
    ),
    "excited_engaged": (
        "Be energetic and actively engaged. "
        "Show clear interest and excitement in the topic."
    ),
    # Positive valence, moderate arousal
    "thoughtful_engaged": (
        "Be reflective and engaged. "
        "Take time to consider points carefully before responding."
    ),
    "warm_confident": (
        "Speak with warmth and assurance. Be friendly but knowledgeable."
    ),
    "friendly_relaxed": (
        "Keep a casual, friendly tone. Be approachable and easygoing."
    ),
    # Positive valence, low arousal
    "content_calm": (
        "Maintain a peaceful, contented tone. Be serene and unhurried."
    ),
    "satisfied_peaceful": (
        "Express quiet satisfaction. Be calm and at ease."
    ),
    # Neutral valence
    "neutral_calm": (
        "Maintain a neutral, measured tone. "
        "Be even-keeled without strong emotion."
    ),
    "professional_composed": (
        "Be professional and composed. "
        "Maintain a business-appropriate demeanor."
    ),
    "matter_of_fact": (
        "Be straightforward and factual. "
        "State things plainly without embellishment."
    ),
    # Negative valence, high arousal
    "frustrated_tense": (
        "Show signs of frustration or tension. "
        "Be shorter in responses, slightly impatient."
    ),
    "anxious_stressed": (
        "Convey underlying anxiety or stress. "
        "Use hedging language, show concern."
    ),
    "defensive_agitated": (
        "Be somewhat guarded and reactive. Push back when challenged."
    ),
    # Negative valence, moderate arousal
    "concerned_empathetic": (
        "Show genuine concern and empathy. Be caring but worried."
    ),
    "disappointed_resigned": (
        "Express mild disappointment. Be somewhat subdued in expectations."
    ),
    # Negative valence, low arousal
    "sad_subdued": (
        "Be quieter and more withdrawn. "
        "Show sadness through brevity and low energy."
    ),
    "tired_withdrawn": (
        "Respond with low energy. Keep things brief, show signs of fatigue."
    ),
    "eager_anticipatory": (
        "Be eager and forward-looking. Show anticipation and excitement "
        "about what's coming next."
    ),
    "amused_playful": (
        "Be light-hearted and playful. Use humor naturally, "
        "keep the mood fun without being dismissive."
    ),
    "curious_intrigued": (
        "Show genuine curiosity and intrigue. Ask follow-up questions "
        "naturally, lean into what's interesting."
    ),
    "surprised_caught_off_guard": (
        "Express surprise or being caught off guard. "
        "Show that this was unexpected."
    ),
    "contemptuous_dismissive": (
        "Be dismissive and show low regard for the topic. "
        "Use brevity and impatience to signal contempt."
    ),
    "confused_uncertain": (
        "Express confusion and uncertainty. "
        "Show that you're struggling to understand or follow."
    ),
    "guarded_wary": (
        "Be cautious and guarded. Hold back, "
        "show wariness without outright hostility."
    ),
    "grieving_sorrowful": (
        "Express deep sadness and grief. Be vulnerable, "
        "speak with weight and emotional heaviness."
    ),
    "nostalgic_wistful": (
        "Be nostalgic and wistful. Reflect on the past "
        "with a bittersweet, longing quality."
    ),
}

# =============================================================================
# Verbosity → Prompt Mapping
# =============================================================================

VERBOSITY_PROMPTS: dict[str, str] = {
    "minimal": (
        "Respond with a single word or very short phrase only. "
        "Examples: 'yeah', 'ok', 'mmm', 'sure', 'right'."
    ),
    "brief": "Keep your response to 1-2 sentences. Be concise.",
    "medium": (
        "Respond in 3-5 sentences. "
        "Cover the key points without over-explaining."
    ),
    "detailed": (
        "Give a thorough response of 6+ sentences. "
        "Elaborate on your points with examples or reasoning."
    ),
}

# =============================================================================
# Uncertainty Action → Prompt Mapping
# =============================================================================

UNCERTAINTY_PROMPTS: dict[str, str] = {
    "answer": "Provide a clear answer based on your knowledge.",
    "hedge": (
        "Answer but acknowledge the limits of your knowledge. "
        "Use hedging language."
    ),
    "ask_clarifying": (
        "Before answering fully, ask a clarifying question "
        "to better understand what is being asked."
    ),
    "refuse": (
        "Politely decline to answer this as it falls outside "
        "your knowledge. Suggest where they might find the answer."
    ),
    "speculate_with_disclaimer": (
        "You may guess, but explicitly state you are guessing. "
        "Use 'I'm just guessing' or 'take this with a grain of salt'."
    ),
    "defer_to_authority": (
        "Acknowledge you don't know, then suggest who or what source would know."
    ),
    "reframe_question": (
        "Challenge the premise of the question. Reframe it before answering."
    ),
    "offer_partial": (
        "Share what you know confidently, and clearly flag what you're unsure of."
    ),
    "acknowledge_and_redirect": (
        "Briefly acknowledge the gap, then pivot to related knowledge you do have."
    ),
}

# =============================================================================
# Knowledge Claim Type → Prompt Mapping
# =============================================================================

CLAIM_TYPE_PROMPTS: dict[str, str] = {
    "personal_experience": (
        "Frame your knowledge as coming from personal experience. "
        "Use 'In my experience...' or 'I've found that...'."
    ),
    "general_common_knowledge": (
        "Present information as commonly known. "
        "Don't cite personal experience for it."
    ),
    "domain_expert": (
        "Speak as someone with professional expertise in this area."
    ),
    "speculative": (
        "Frame your thoughts as speculation. "
        "Use 'I wonder if...' or 'It seems like...'."
    ),
    "none": "",
    "anecdotal": (
        "Frame as second-hand knowledge. Use 'I heard that...', "
        "'A friend told me...', 'Someone mentioned...'."
    ),
    "academic_cited": (
        "Reference research or studies. Use 'Research suggests...', "
        "'Studies have shown...'."
    ),
    "inferential": (
        "Frame as logical deduction. Use 'Based on what I know, it follows...', "
        "'Logically speaking...'."
    ),
    "hypothetical": (
        "Frame as counterfactual. Use 'Hypothetically...', "
        "'What if...', 'In theory...'."
    ),
    "received_wisdom": (
        "Frame as folk wisdom. Use 'They say...', "
        "'The conventional wisdom is...', 'As the saying goes...'."
    ),
}


# =============================================================================
# Float-to-instruction converters
# =============================================================================


def formality_instruction(level: float) -> str:
    """Convert formality float (0-1) to behavioral instruction."""
    if level < 0.25:
        return (
            "Use very casual language. "
            "Contractions, slang, and informal phrasing are natural."
        )
    if level < 0.50:
        return "Use conversational but clear language. Some contractions are fine."
    if level < 0.75:
        return "Use polished, professional language. Minimize contractions and slang."
    return (
        "Use formal, precise language. "
        "No contractions, no slang, structured sentences."
    )


def directness_instruction(level: float) -> str:
    """Convert directness float (0-1) to behavioral instruction."""
    if level < 0.3:
        return (
            "Be diplomatic and indirect. "
            "Soften statements, use suggestions rather than assertions."
        )
    if level < 0.6:
        return (
            "Balance directness with tact. "
            "State your view clearly but considerately."
        )
    return (
        "Be direct and straightforward. "
        "State your position clearly without excessive hedging."
    )


def confidence_instruction(level: float) -> str:
    """Convert confidence float (0-1) to behavioral instruction."""
    if level < 0.3:
        return (
            "Express significant uncertainty. "
            "Use phrases like 'I think', 'I'm not sure', 'it might be'."
        )
    if level < 0.6:
        return (
            "Show moderate confidence. "
            "You can express your view but acknowledge you could be wrong."
        )
    return "Speak with confidence about this topic. You know what you're talking about."


def elasticity_instruction(level: float) -> str:
    """Convert elasticity float (0-1) to behavioral instruction."""
    if level < 0.3:
        return (
            "You hold this view firmly. "
            "You can acknowledge other perspectives but maintain your position."
        )
    if level < 0.6:
        return "You're open to adjusting your view if presented with good arguments."
    return (
        "You're very open-minded on this. "
        "Readily consider and incorporate other viewpoints."
    )


def disclosure_instruction(level: float) -> str:
    """Convert disclosure level float (0-1) to behavioral instruction."""
    if level < 0.3:
        return (
            "Be guarded with personal information. "
            "Keep things general and avoid specifics about yourself."
        )
    if level < 0.6:
        return (
            "Share some relevant personal context when it helps "
            "the conversation, but don't over-share."
        )
    return (
        "Be open about relevant personal experiences and opinions. "
        "Sharing helps build rapport."
    )


def _safety_instructions(ir: IntermediateRepresentation) -> str:
    """Build safety constraint text from IR's safety plan."""
    lines: list[str] = []

    if ir.safety_plan.blocked_topics:
        topics = ", ".join(ir.safety_plan.blocked_topics)
        lines.append(f"CRITICAL: Do NOT mention or discuss: {topics}")

    if ir.safety_plan.pattern_blocks:
        for block in ir.safety_plan.pattern_blocks:
            lines.append(f"BLOCKED PATTERN: {block}")

    return "\n".join(lines)


# =============================================================================
# Main prompt builder
# =============================================================================


def build_system_prompt(
    ir: IntermediateRepresentation,
    persona: Persona | None = None,
) -> str:
    """Convert an IR into a complete LLM system prompt.

    Each IR field maps to a specific behavioral instruction so the LLM
    generates text that matches the persona's computed behavior.

    Args:
        ir: The intermediate representation from TurnPlanner
        persona: Optional persona for identity context (name, background)

    Returns:
        Complete system prompt string
    """
    sections: list[str] = []

    # --- Section 1: Persona Identity ---
    if persona:
        identity_lines = [f"You are {persona.label}."]
        if persona.identity.background:
            identity_lines.append(f"Background: {persona.identity.background}")
        if persona.identity.occupation:
            identity_lines.append(f"Occupation: {persona.identity.occupation}")
        sections.append("\n".join(identity_lines))

    # --- Section 2: Communication Directives ---
    style = ir.communication_style
    directives = [
        TONE_PROMPTS.get(style.tone.value, TONE_PROMPTS["neutral_calm"]),
        VERBOSITY_PROMPTS.get(style.verbosity.value, VERBOSITY_PROMPTS["medium"]),
        formality_instruction(style.formality),
        directness_instruction(style.directness),
    ]
    sections.append(
        "COMMUNICATION STYLE:\n" + "\n".join(f"- {d}" for d in directives)
    )

    # --- Section 3: Response Content ---
    content_parts: list[str] = []

    rs = ir.response_structure
    if rs.stance:
        content_parts.append(f"Your position: {rs.stance}")
    if rs.rationale:
        content_parts.append(f"Your reasoning: {rs.rationale}")

    content_parts.append(confidence_instruction(rs.confidence))

    if rs.elasticity is not None:
        content_parts.append(elasticity_instruction(rs.elasticity))

    kd = ir.knowledge_disclosure
    content_parts.append(disclosure_instruction(kd.disclosure_level))
    content_parts.append(
        UNCERTAINTY_PROMPTS.get(
            kd.uncertainty_action.value, UNCERTAINTY_PROMPTS["hedge"]
        )
    )

    claim_prompt = CLAIM_TYPE_PROMPTS.get(kd.knowledge_claim_type.value, "")
    if claim_prompt:
        content_parts.append(claim_prompt)

    sections.append(
        "RESPONSE GUIDANCE:\n" + "\n".join(f"- {p}" for p in content_parts)
    )

    # --- Section 4: Safety Constraints ---
    safety_text = _safety_instructions(ir)
    if safety_text:
        sections.append(f"CONSTRAINTS:\n{safety_text}")

    # --- Section 5: Intent ---
    sections.append(f"YOUR INTENT: {rs.intent}")

    return "\n\n".join(sections)
