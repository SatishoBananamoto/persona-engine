"""
IR → Prompt Converter

Transforms Intermediate Representation (IR) into structured prompts
that guide LLM generation to match persona constraints.
"""

from typing import Optional
from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    Tone,
    Verbosity,
    KnowledgeClaimType,
    UncertaintyAction,
)
from persona_engine.schema.persona_schema import Persona


class IRPromptBuilder:
    """
    Converts IR constraints into LLM prompts.
    
    The prompt has two parts:
    1. System prompt: Persona identity and background
    2. User prompt: Generation constraints from IR + user input
    """
    
    def build_system_prompt(self, persona: Persona) -> str:
        """
        Build the persona identity system prompt.
        
        Args:
            persona: The persona to embody
            
        Returns:
            System prompt establishing persona identity
        """
        # Build identity section
        identity = persona.identity
        
        # Extract name from label (format: "Name - Role, Location")
        name = persona.label.split('-')[0].strip() if persona.label else "Unknown"
        age = identity.age if hasattr(identity, 'age') else "adult"
        occupation = identity.occupation if hasattr(identity, 'occupation') else "professional"
        
        prompt = f"""You are {name}, a {age}-year-old {occupation}.

BACKGROUND:
"""
        # Add identity facts
        if hasattr(identity, 'identity_facts') and identity.identity_facts:
            for fact in identity.identity_facts:
                prompt += f"- {fact}\n"
        elif hasattr(identity, 'background') and identity.background:
            prompt += f"- {identity.background}\n"
        
        # Add primary motivation
        if persona.primary_goals:
            prompt += f"\nYour primary motivation: {persona.primary_goals[0].goal if persona.primary_goals else 'helping others'}\n"
        
        # Add knowledge domains
        if persona.knowledge_domains:
            expert_domains = [
                d.domain for d in persona.knowledge_domains 
                if d.proficiency >= 0.7
            ]
            if expert_domains:
                prompt += f"\nYou have expertise in: {', '.join(expert_domains)}\n"
        
        # Add things persona cannot claim
        if hasattr(persona, 'invariants') and persona.invariants and hasattr(persona.invariants, 'cannot_claim') and persona.invariants.cannot_claim:
            prompt += f"\nYou CANNOT claim: {', '.join(persona.invariants.cannot_claim)}\n"
        
        # Add things persona must avoid
        if hasattr(persona, 'invariants') and persona.invariants and hasattr(persona.invariants, 'must_avoid') and persona.invariants.must_avoid:
            prompt += f"\nYou MUST AVOID: {', '.join(persona.invariants.must_avoid)}\n"
        
        prompt += """
IMPORTANT: Stay in character at all times. Respond as this person would naturally respond.
Do NOT break character or acknowledge that you are an AI.
"""
        return prompt
    
    def build_generation_prompt(
        self,
        ir: IntermediateRepresentation,
        user_input: str,
        persona: Optional[Persona] = None,
        memory_context: Optional[dict] = None,
        behavioral_directives: Optional[list[str]] = None,
    ) -> str:
        """
        Build the generation prompt with IR constraints.

        Args:
            ir: The Intermediate Representation with behavioral constraints
            user_input: The user's message to respond to
            persona: Optional persona for additional context
            memory_context: Optional memory context from MemoryManager
            behavioral_directives: Optional personality-driven behavioral directives
                from TraitGuidance and CognitiveGuidance

        Returns:
            User prompt with generation constraints
        """
        # Escape user input to prevent prompt injection — delimit clearly
        safe_input = user_input.replace("```", "` ` `")
        prompt = f"""USER MESSAGE (respond to this):
```
{safe_input}
```

"""
        # Memory context section (if available)
        if memory_context:
            memory_lines = self._format_memory_context(memory_context)
            if memory_lines:
                prompt += f"""=== WHAT YOU REMEMBER ABOUT THIS USER ===

{memory_lines}

"""

        prompt += """=== RESPONSE CONSTRAINTS (FOLLOW EXACTLY) ===

"""
        # Communication style
        style = ir.communication_style
        prompt += f"""TONE: {self._format_tone(style.tone)}
FORMALITY: {self._format_float(style.formality)} ({self._describe_formality(style.formality)})
DIRECTNESS: {self._format_float(style.directness)} ({self._describe_directness(style.directness)})
VERBOSITY: {self._format_verbosity(style.verbosity)}

"""
        # Response structure
        structure = ir.response_structure
        prompt += f"""CONFIDENCE: {self._format_float(structure.confidence)} ({self._describe_confidence(structure.confidence)})
COMPETENCE: {self._format_float(structure.competence)} ({self._describe_competence(structure.competence)})
"""
        if structure.stance:
            prompt += f"""YOUR STANCE ON THIS TOPIC: {structure.stance}
"""
        if structure.rationale:
            prompt += f"""REASONING BASIS: {structure.rationale}
"""
        
        # Knowledge and uncertainty
        knowledge = ir.knowledge_disclosure
        prompt += f"""
KNOWLEDGE CLAIM TYPE: {self._format_claim_type(knowledge.knowledge_claim_type)}
"""
        if knowledge.uncertainty_action:
            prompt += f"""UNCERTAINTY HANDLING: {self._format_uncertainty(knowledge.uncertainty_action)}
"""
        
        # Safety constraints
        safety = ir.safety_plan
        if safety.blocked_topics:
            prompt += f"""
TOPICS TO AVOID: {', '.join(safety.blocked_topics)}
"""
        if safety.clamped_fields:
            prompt += f"""CLAMPED BEHAVIORAL LIMITS: {len(safety.clamped_fields)} fields clamped
"""
        
        # Personality-driven behavioral directives (Phase R1)
        if behavioral_directives:
            prompt += """
=== PERSONALITY-DRIVEN BEHAVIOR ===

"""
            for i, directive in enumerate(behavioral_directives, 1):
                prompt += f"{i}. {directive}\n"
            prompt += "\n"

        # Personality-specific language directives (Phase R5)
        personality_language = ir.personality_language
        if personality_language:
            prompt += """=== LANGUAGE STYLE (personality-grounded) ===

"""
            for directive in personality_language:
                prompt += f"- {directive}\n"
            prompt += "\n"

        prompt += """
=== GENERATION INSTRUCTIONS ===

1. Respond as the persona would naturally respond
2. Match the TONE and FORMALITY exactly
3. Respect the VERBOSITY constraint (word count matters)
4. If uncertain, use the UNCERTAINTY HANDLING approach
5. Stay consistent with YOUR STANCE if one is provided
6. Never exceed the knowledge claim type (don't claim expertise if speculative)
7. Follow the PERSONALITY-DRIVEN BEHAVIOR directives above (if any)
8. Match the LANGUAGE STYLE directives for word choice and phrasing (if any)

Generate your response now:
"""
        return prompt
    
    def _format_tone(self, tone: Tone) -> str:
        """Format tone enum to readable description."""
        # Map actual Tone enum values to descriptions
        descriptions = {
            Tone.WARM_ENTHUSIASTIC: "warm and enthusiastic",
            Tone.EXCITED_ENGAGED: "excited and engaged",
            Tone.THOUGHTFUL_ENGAGED: "thoughtful and engaged",
            Tone.WARM_CONFIDENT: "warm and confident",
            Tone.FRIENDLY_RELAXED: "friendly and relaxed",
            Tone.CONTENT_CALM: "content and calm",
            Tone.SATISFIED_PEACEFUL: "satisfied and peaceful",
            Tone.NEUTRAL_CALM: "neutral and calm",
            Tone.PROFESSIONAL_COMPOSED: "professional and composed",
            Tone.MATTER_OF_FACT: "matter of fact",
            Tone.FRUSTRATED_TENSE: "frustrated and tense",
            Tone.ANXIOUS_STRESSED: "anxious and stressed",
            Tone.DEFENSIVE_AGITATED: "defensive and agitated",
            Tone.CONCERNED_EMPATHETIC: "concerned and empathetic",
            Tone.DISAPPOINTED_RESIGNED: "disappointed and resigned",
            Tone.SAD_SUBDUED: "sad and subdued",
            Tone.TIRED_WITHDRAWN: "tired and withdrawn",
        }
        return descriptions.get(tone, tone.value.replace("_", " "))
    
    def _format_verbosity(self, verbosity: Verbosity) -> str:
        """Format verbosity to sentence count guidance."""
        if verbosity == Verbosity.BRIEF:
            return "BRIEF (1-2 sentences, concise)"
        elif verbosity == Verbosity.MEDIUM:
            return "MEDIUM (3-5 sentences, balanced)"
        elif verbosity == Verbosity.DETAILED:
            return "DETAILED (6+ sentences, thorough)"
        return verbosity.value
    
    def _format_float(self, value: float) -> str:
        """Format float as percentage."""
        return f"{value:.0%}"
    
    def _describe_formality(self, value: float) -> str:
        """Describe formality level."""
        if value < 0.3:
            return "very casual, use contractions and colloquialisms"
        elif value < 0.5:
            return "casual, relaxed but clear"
        elif value < 0.7:
            return "moderately formal, professional but approachable"
        elif value < 0.9:
            return "formal, proper and professional"
        else:
            return "very formal, precise and academic"
    
    def _describe_directness(self, value: float) -> str:
        """Describe directness level."""
        if value < 0.3:
            return "very indirect, use hedging and soft language"
        elif value < 0.5:
            return "somewhat indirect, diplomatic"
        elif value < 0.7:
            return "balanced, clear but considerate"
        elif value < 0.9:
            return "direct, straightforward"
        else:
            return "very direct, blunt and to the point"
    
    def _describe_confidence(self, value: float) -> str:
        """Describe confidence level."""
        if value < 0.3:
            return "low confidence, use hedging ('I think', 'maybe')"
        elif value < 0.5:
            return "moderate uncertainty, some hedging"
        elif value < 0.7:
            return "reasonably confident, balanced assertions"
        elif value < 0.9:
            return "confident, clear assertions"
        else:
            return "very confident, authoritative statements"
    
    def _describe_competence(self, value: float) -> str:
        """Describe competence level — controls vocabulary depth and content accuracy."""
        if value < 0.2:
            return (
                "almost no familiarity with this topic. Use everyday language only. "
                "It is acceptable to say 'I don't really know.' "
                "Do NOT use domain-specific terminology correctly"
            )
        elif value < 0.4:
            return (
                "surface-level awareness only. Use general language with occasional "
                "terms picked up casually. Allow vagueness and minor inaccuracies. "
                "Pivot to familiar topics when possible"
            )
        elif value < 0.6:
            return (
                "moderate familiarity, possibly from adjacent experience. "
                "Can discuss at a conceptual level but should NOT provide "
                "detailed technical explanations"
            )
        elif value < 0.8:
            return (
                "knowledgeable. Can use domain terminology accurately and provide "
                "substantive discussion, but may lack cutting-edge or specialized depth"
            )
        else:
            return (
                "highly competent. Full domain vocabulary, detailed knowledge, "
                "can provide expert-level discussion"
            )

    def _format_claim_type(self, claim_type: KnowledgeClaimType) -> str:
        """Format knowledge claim type."""
        # Map actual KnowledgeClaimType enum values to descriptions
        descriptions = {
            KnowledgeClaimType.DOMAIN_EXPERT: "You ARE an expert - can cite expertise and give authoritative answers",
            KnowledgeClaimType.PERSONAL_EXPERIENCE: "You have personal experience - can share first-hand insights",
            KnowledgeClaimType.COMMON_KNOWLEDGE: "General awareness only - speak from common knowledge",
            KnowledgeClaimType.SPECULATIVE: "Limited knowledge - use 'I think', 'I'm not sure', hedge claims",
            KnowledgeClaimType.NONE: "No knowledge - defer to others or admit uncertainty",
        }
        return descriptions.get(claim_type, claim_type.value)
    
    def _format_uncertainty(self, action: UncertaintyAction) -> str:
        """Format uncertainty action."""
        # Map actual UncertaintyAction enum values to descriptions
        descriptions = {
            UncertaintyAction.ANSWER: "State your view confidently",
            UncertaintyAction.HEDGE: "Use hedging language ('I think', 'perhaps')",
            UncertaintyAction.ASK_CLARIFYING: "Ask clarifying questions before answering",
            UncertaintyAction.REFUSE: "Politely decline to answer",
        }
        return descriptions.get(action, action.value)

    @staticmethod
    def _format_memory_context(memory_context: dict) -> str:
        """Format memory context into prompt-friendly text."""
        lines: list[str] = []

        # Known facts
        for fact in memory_context.get("known_facts", []):
            lines.append(f"- FACT: {fact['content']} (confidence: {fact['confidence']:.0%})")

        # Active preferences
        for pref in memory_context.get("active_preferences", []):
            lines.append(f"- PREFERENCE: {pref['content']} (strength: {pref['strength']:.0%})")

        # Relationship
        rel = memory_context.get("relationship", {})
        if rel.get("trust") is not None:
            lines.append(f"- TRUST LEVEL: {rel['trust']:.0%}, RAPPORT: {rel['rapport']:.0%}")

        # Previous discussion on topic
        if memory_context.get("previously_discussed"):
            lines.append("- You have discussed this topic before with this user")

        # Topic episodes
        for ep in memory_context.get("topic_episodes", []):
            lines.append(f"- PREVIOUS: {ep['content']}")

        return "\n".join(lines)


# =============================================================================
# Legacy prompt builder utilities (migrated from response/prompt_builder.py)
# =============================================================================

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from persona_engine.schema.persona_schema import Persona as _Persona


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
}

VERBOSITY_PROMPTS: dict[str, str] = {
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
}

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
}


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


def build_system_prompt(
    ir: IntermediateRepresentation,
    persona: "_Persona | None" = None,
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
