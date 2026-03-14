"""
Big Five Trait Interpreter

Translates Big Five personality traits into concrete behavioral parameters
that influence IR generation (communication style, response patterns, etc.)

Phase R2: Effect sizes recalibrated to match real psychology research.
Sigmoid activation for extreme traits, Dunning-Kruger confidence curve.
"""

import math
from typing import Any

from persona_engine.schema.ir_schema import Tone, Verbosity
from persona_engine.schema.persona_schema import BigFiveTraits, Persona


def trait_effect(trait_value: float, center: float = 0.5, steepness: float = 8.0) -> float:
    """Sigmoid activation: extreme traits have disproportionately stronger effects.

    - trait 0.5 → effect 0.5 (moderate)
    - trait 0.8 → effect ~0.88 (amplified)
    - trait 0.2 → effect ~0.12 (amplified opposite)
    - trait 0.95 → effect ~0.98 (very strong)
    """
    return 1.0 / (1.0 + math.exp(-steepness * (trait_value - center)))


def dunning_kruger_confidence(proficiency: float, neuroticism: float) -> float:
    """Non-linear proficiency → confidence mapping (Dunning-Kruger).

    Continuous piecewise curve:
    - Novice (< 0.2): Overconfident — lacks metacognition
    - Transition (0.2-0.4): Gradual reality check
    - Intermediate (0.4-0.6): "Valley of despair" — most uncertain
    - Expert (> 0.6): Calibrated confidence, approaching proficiency
    """
    if proficiency < 0.2:
        # DK inflation: overconfident novice (neuroticism partially counteracts)
        inflation = (1 - neuroticism) * 0.25
        return proficiency + inflation
    elif proficiency < 0.4:
        # Transition: inflation fades linearly from full to zero
        inflation = (1 - neuroticism) * 0.25
        fade = (proficiency - 0.2) / 0.2  # 0→1 as proficiency 0.2→0.4
        valley_penalty = 0.08
        return proficiency + inflation * (1 - fade) - valley_penalty * fade
    elif proficiency < 0.55:
        # Valley of despair: knows enough to doubt
        return proficiency - 0.08
    elif proficiency < 0.7:
        # Transition out of valley: penalty fades from 0.08 to 0.03
        fade = (proficiency - 0.55) / 0.15  # 0→1 as proficiency 0.55→0.7
        penalty = 0.08 * (1 - fade) + 0.03 * fade
        return proficiency - penalty
    else:
        # Expert calibration: confident but humble
        return proficiency - 0.03


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
        # Phase R2: Sigmoid activation for openness → elasticity
        # openness=0.5 → factor~0.5; openness=0.8 → factor~0.88; openness=0.2 → factor~0.12
        openness_activated = trait_effect(self.traits.openness)
        openness_factor = openness_activated * 0.7
        confidence_penalty = base_confidence * 0.3

        elasticity = openness_factor - confidence_penalty
        return max(0.1, min(0.9, elasticity + 0.2))  # Shift up, clamp to 0.1-0.9

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
        # Phase R2: C verbosity ±0.1 → ±0.25 (high-C are observably more detailed)
        adjusted = base_verbosity + (self.traits.conscientiousness - 0.5) * 0.5
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

        Returns: Modifier to apply to disclosure_level (~±0.21)
        """
        # Phase R2: Sigmoid-amplified E disclosure. Multiplier 0.45 chosen to
        # avoid swamping privacy clamp while producing ≥0.4 spread at extremes.
        e_effect = trait_effect(self.traits.extraversion)
        return (e_effect - 0.5) * 0.45

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
        # Phase R2: Sigmoid-amplified A directness. Multiplier 0.5 chosen to
        # avoid saturation at extremes while maintaining meaningful spread.
        a_effect = trait_effect(self.traits.agreeableness)
        modifier = (0.5 - a_effect) * 0.5
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
        Adjusts confidence based on traits + Dunning-Kruger curve.

        Phase R2: C confidence ±0.05 → ±0.15, N penalty max -0.15 → -0.25.
        Also applies Dunning-Kruger non-linear proficiency → confidence mapping.

        Args:
            domain_proficiency: Base proficiency (0-1)

        Returns:
            Modified confidence
        """
        # Phase R2: Apply Dunning-Kruger curve first
        dk_confidence = dunning_kruger_confidence(domain_proficiency, self.traits.neuroticism)

        # Phase R2: C confidence ±0.05 → ±0.15
        c_boost = (self.traits.conscientiousness - 0.5) * 0.3
        # Phase R2: N penalty -0.15 → -0.25, with sigmoid for extreme amplification
        n_effect = trait_effect(self.traits.neuroticism)
        n_penalty = n_effect * 0.25

        adjusted = dk_confidence + c_boost - n_penalty
        return max(0.1, min(0.95, adjusted))

    def get_trait_markers_for_validation(self) -> dict[str, Any]:
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
