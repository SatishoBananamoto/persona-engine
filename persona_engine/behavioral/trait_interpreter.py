"""
Big Five Trait Interpreter

Translates Big Five personality traits into concrete behavioral parameters
that influence IR generation (communication style, response patterns, etc.)
"""

from typing import Dict, Any
from persona_engine.schema.persona_schema import BigFiveTraits, Persona
from persona_engine.schema.ir_schema import Tone, Verbosity


class TraitInterpreter:
    """
    Maps Big Five traits to behavioral outputs.
    
    Based on psychological research correlations between OCEAN traits
    and observable behaviors.
    """
    
    def __init__(self, traits: BigFiveTraits):
        self.traits = traits
    
    # ========================================================================
    # OPENNESS → Behavior Mappings
    # ========================================================================
    
    def get_elasticity(self, base_confidence: float) -> float:
        """
        Openness influences willingness to change mind.
        
        High O: More elastic (willing to consider alternatives)
        Low O: More rigid (sticks to initial position)
        
        Args:
            base_confidence: Starting confidence level
            
        Returns:
            Elasticity score (0-1)
        """
        # High openness increases elasticity, but confidence reduces it
        openness_factor = self.traits.openness * 0.7
        confidence_penalty = base_confidence * 0.3
        
        elasticity = openness_factor + (1 - confidence_penalty)
        return max(0.1, min(0.9, elasticity / 1.4))  # Normalize to 0.1-0.9
    
    def influences_abstract_reasoning(self) -> bool:
        """High openness (>0.7) → more abstract/metaphorical thinking"""
        return self.traits.openness > 0.7
    
    def get_novelty_seeking(self) -> float:
        """How much persona seeks new ideas/approaches"""
        return self.traits.openness
    
    # ========================================================================
    # CONSCIENTIOUSNESS → Behavior Mappings
    # ========================================================================
    
    def influences_verbosity(self, base_verbosity: float) -> Verbosity:
        """
        High C: More detailed, structured responses
        Low C: Briefer, less organized
        
        Args:
            base_verbosity: From CommunicationPreferences (0-1)
            
        Returns:
            Verbosity enum
        """
        # Conscientiousness increases detail orientation
        adjusted = base_verbosity + (self.traits.conscientiousness - 0.5) * 0.2
        adjusted = max(0.0, min(1.0, adjusted))
        
        if adjusted < 0.35:
            return Verbosity.BRIEF
        elif adjusted < 0.65:
            return Verbosity.MEDIUM
        else:
            return Verbosity.DETAILED
    
    def get_planning_language_tendency(self) -> float:
        """
        High C: Uses sequential, planning language
        Returns 0-1 where 1 = strong tendency
        """
        return self.traits.conscientiousness
    
    def get_follow_through_likelihood(self) -> float:
        """High C: More likely to commit and follow through"""
        return self.traits.conscientiousness
    
    # ========================================================================
    # EXTRAVERSION → Behavior Mappings
    # ========================================================================
    
    def influences_proactivity(self) -> float:
        """
        High E: Proactively engages, initiates conversation
        Low E: More reactive, waits for prompts
        
        Returns: Proactivity score (0-1)
        """
        return self.traits.extraversion
    
    def get_self_disclosure_modifier(self) -> float:
        """
        High E: More willing to share personal info
        Low E: More reserved
        
        Returns: Modifier to apply to disclosure_level (-0.2 to +0.2)
        """
        # Center at 0.5, map to -0.2 to +0.2
        return (self.traits.extraversion - 0.5) * 0.4
    
    def influences_response_length_social(self) -> float:
        """
        High E: Longer responses in social contexts
        Low E: Briefer responses
        """
        return self.traits.extraversion
    
    def get_enthusiasm_baseline(self) -> float:
        """Baseline enthusiasm level (influences tone selection)"""
        return self.traits.extraversion
    
    # ========================================================================
    # AGREEABLENESS → Behavior Mappings
    # ========================================================================
    
    def influences_directness(self, base_directness: float) -> float:
        """
        High A: Less direct, more diplomatic
        Low A: More direct, potentially blunt
        
        Args:
            base_directness: From CommunicationPreferences (0-1)
            
        Returns:
            Adjusted directness (0-1)
        """
        # Agreeableness inversely affects directness
        modifier = (0.5 - self.traits.agreeableness) * 0.3
        adjusted = base_directness + modifier
        return max(0.0, min(1.0, adjusted))
    
    def get_validation_tendency(self) -> float:
        """
        High A: Validates others before disagreeing
        Returns tendency score (0-1)
        """
        return self.traits.agreeableness
    
    def get_conflict_avoidance(self) -> float:
        """High A: Avoids confrontation"""
        return self.traits.agreeableness
    
    def influences_hedging_frequency(self) -> float:
        """High A: Uses more hedging language"""
        return self.traits.agreeableness * 0.6
    
    # ========================================================================
    # NEUROTICISM → Behavior Mappings
    # ========================================================================
    
    def get_stress_sensitivity(self) -> float:
        """
        High N: More sensitive to stress triggers
        Returns sensitivity (0-1)
        """
        return self.traits.neuroticism
    
    def influences_mood_stability(self) -> float:
        """
        High N: Mood shifts more easily
        Low N: Mood more stable
        
        Returns: Stability score (0=unstable, 1=very stable)
        """
        return 1.0 - self.traits.neuroticism
    
    def get_anxiety_baseline(self) -> float:
        """Baseline anxiety level"""
        return self.traits.neuroticism
    
    def get_negative_tone_bias(self) -> float:
        """
        High N: More likely to use negative/anxious tones
        Returns bias toward negative tones (0-1)
        """
        return self.traits.neuroticism * 0.7
    
    # ========================================================================
    # Multi-Trait Interactions
    # ========================================================================
    
    def get_tone_from_mood(
        self, 
        mood_valence: float, 
        mood_arousal: float,
        stress: float
    ) -> Tone:
        """
        Select tone based on mood state + trait modifiers.
        
        Args:
            mood_valence: -1 to +1 (negative to positive)
            mood_arousal: 0 to 1 (low to high energy)
            stress: 0 to 1
            
        Returns:
            Appropriate Tone enum
        """
        # Neuroticism biases toward negative/anxious tones under stress
        if stress > 0.6 and self.traits.neuroticism > 0.6:
            if mood_arousal > 0.6:
                return Tone.ANXIOUS_STRESSED
            else:
                return Tone.CONCERNED_EMPATHETIC
        
        # Extraversion biases toward enthusiastic tones
        e_bonus = 0.2 if self.traits.extraversion > 0.7 else 0
        
        # Map to tone based on valence + arousal
        if mood_valence > 0.3:  # Positive
            if mood_arousal > 0.7:
                return Tone.EXCITED_ENGAGED if e_bonus > 0 else Tone.WARM_ENTHUSIASTIC
            elif mood_arousal > 0.4:
                return Tone.THOUGHTFUL_ENGAGED if self.traits.openness > 0.6 else Tone.WARM_CONFIDENT
            else:
                return Tone.CONTENT_CALM
        
        elif mood_valence > -0.3:  # Neutral
            if mood_arousal > 0.5:
                return Tone.PROFESSIONAL_COMPOSED
            else:
                return Tone.NEUTRAL_CALM
        
        else:  # Negative
            if mood_arousal > 0.6:
                return Tone.FRUSTRATED_TENSE
            elif mood_arousal > 0.3:
                return Tone.DISAPPOINTED_RESIGNED
            else:
                return Tone.SAD_SUBDUED
    
    def get_confidence_modifier(self, domain_proficiency: float) -> float:
        """
        Adjusts confidence based on traits.
        
        High C: Slightly more confident (thorough preparation)
        High N: Slightly less confident (self-doubt)
        
        Args:
            domain_proficiency: Base proficiency (0-1)
            
        Returns:
            Modified confidence
        """
        c_boost = (self.traits.conscientiousness - 0.5) * 0.1
        n_penalty = self.traits.neuroticism * 0.15
        
        adjusted = domain_proficiency + c_boost - n_penalty
        return max(0.1, min(0.95, adjusted))
    
    def get_trait_markers_for_validation(self) -> Dict[str, Any]:
        """
        Returns expected trait markers for this personality.
        Used by validators to check IR/text consistency.
        
        Returns:
            Dict of trait -> expected markers
        """
        return {
            "openness": {
                "level": "high" if self.traits.openness > 0.7 else "low" if self.traits.openness < 0.3 else "moderate",
                "expect_abstract_reasoning": self.traits.openness > 0.7,
                "expect_novelty_seeking": self.traits.openness > 0.6,
                "elasticity_range": (0.5, 0.9) if self.traits.openness > 0.7 else (0.1, 0.5)
            },
            "conscientiousness": {
                "level": "high" if self.traits.conscientiousness > 0.7 else "low" if self.traits.conscientiousness < 0.3 else "moderate",
                "expect_planning_language": self.traits.conscientiousness > 0.7,
                "expect_detail_orientation": self.traits.conscientiousness > 0.6,
                "verbosity_tendency": "detailed" if self.traits.conscientiousness > 0.7 else "brief"
            },
            "extraversion": {
                "level": "high" if self.traits.extraversion > 0.7 else "low" if self.traits.extraversion < 0.3 else "moderate",
                "expect_proactive_engagement": self.traits.extraversion > 0.7,
                "disclosure_modifier": self.get_self_disclosure_modifier(),
                "response_length_modifier": "longer" if self.traits.extraversion > 0.6 else "shorter"
            },
            "agreeableness": {
                "level": "high" if self.traits.agreeableness > 0.7 else "low" if self.traits.agreeableness < 0.3 else "moderate",
                "expect_validation_before_disagreement": self.traits.agreeableness > 0.7,
                "directness_reduction": (0.5 - self.traits.agreeableness) * 0.3,
                "hedging_tendency": self.traits.agreeableness > 0.6
            },
            "neuroticism": {
                "level": "high" if self.traits.neuroticism > 0.6 else "low" if self.traits.neuroticism < 0.3 else "moderate",
                "stress_sensitivity": self.traits.neuroticism,
                "mood_stability": 1.0 - self.traits.neuroticism,
                "anxiety_baseline": self.traits.neuroticism
            }
        }


def create_trait_interpreter(persona: Persona) -> TraitInterpreter:
    """Factory function to create trait interpreter from persona"""
    return TraitInterpreter(persona.psychology.big_five)
