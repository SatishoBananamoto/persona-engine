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
        persona: Optional[Persona] = None
    ) -> str:
        """
        Build the generation prompt with IR constraints.
        
        Args:
            ir: The Intermediate Representation with behavioral constraints
            user_input: The user's message to respond to
            persona: Optional persona for additional context
            
        Returns:
            User prompt with generation constraints
        """
        prompt = f"""USER MESSAGE: {user_input}

=== RESPONSE CONSTRAINTS (FOLLOW EXACTLY) ===

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
        
        prompt += """
=== GENERATION INSTRUCTIONS ===

1. Respond as the persona would naturally respond
2. Match the TONE and FORMALITY exactly
3. Respect the VERBOSITY constraint (word count matters)
4. If uncertain, use the UNCERTAINTY HANDLING approach
5. Stay consistent with YOUR STANCE if one is provided
6. Never exceed the knowledge claim type (don't claim expertise if speculative)

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
