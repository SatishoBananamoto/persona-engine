"""
Trait Interaction Engine — Emergent Personality Patterns

Models how Big Five traits *combine* to produce emergent behavioral patterns
that neither trait alone would predict.

Based on personality psychology research on trait interaction effects.
Geometric mean activation ensures both traits must be extreme for strong effects.

Phase R3: 9 interaction patterns covering all major trait combinations.
"""

import math
from dataclasses import dataclass, field

from persona_engine.schema.persona_schema import BigFiveTraits


@dataclass
class InteractionEffect:
    """An emergent effect from trait combination."""
    pattern_name: str
    traits_involved: dict[str, str]  # trait -> "high"/"low"
    behavioral_modifiers: dict[str, float]  # field -> modifier
    prompt_guidance: str  # Natural language instruction for LLM
    activation_strength: float  # 0-1, how strongly this pattern is active


# ============================================================================
# Pattern definitions
# ============================================================================

INTERACTION_PATTERNS = [
    {
        "name": "intellectual_combatant",
        "conditions": {"openness": ("high", 0.7), "agreeableness": ("low", 0.35)},
        "modifiers": {
            "directness": +0.20,
            "elasticity": +0.15,
            "enthusiasm_boost": +0.15,
        },
        "prompt_guidance": (
            "You enjoy intellectual sparring. Challenge ideas directly but "
            "with curiosity, not hostility. Ask probing 'but what about...' questions."
        ),
    },
    {
        "name": "anxious_perfectionist",
        "conditions": {"neuroticism": ("high", 0.65), "conscientiousness": ("high", 0.7)},
        "modifiers": {
            "confidence": -0.15,
            "verbosity_boost": +0.2,
            "hedging_level": +0.25,
        },
        "prompt_guidance": (
            "You worry about accuracy. Add caveats like 'I want to be precise here' "
            "and 'though I should note...'. Over-explain rather than under-explain."
        ),
    },
    {
        "name": "warm_leader",
        "conditions": {"extraversion": ("high", 0.7), "agreeableness": ("high", 0.7)},
        "modifiers": {
            "directness": -0.10,
            "enthusiasm_boost": +0.20,
            "validation_tendency": +0.15,
        },
        "prompt_guidance": (
            "You are warm and inclusive. Build consensus by acknowledging "
            "others' contributions before adding your own. Use 'we' language."
        ),
    },
    {
        "name": "hostile_critic",
        "conditions": {"neuroticism": ("high", 0.65), "agreeableness": ("low", 0.35)},
        "modifiers": {
            "directness": +0.15,
            "negative_tone_bias": +0.20,
            "confidence": -0.10,
        },
        "prompt_guidance": (
            "You tend to see flaws and point them out directly. You can be "
            "defensive when challenged. Express dissatisfaction plainly."
        ),
    },
    {
        "name": "quiet_thinker",
        "conditions": {"extraversion": ("low", 0.35), "openness": ("high", 0.7)},
        "modifiers": {
            "verbosity_boost": -0.15,
            "elasticity": +0.10,
            "novelty_seeking": +0.10,
        },
        "prompt_guidance": (
            "You are reflective and philosophical. Say less but make each "
            "word count. Ask unexpected, probing questions rather than chatting."
        ),
    },
    {
        "name": "cautious_conservative",
        "conditions": {"openness": ("low", 0.35), "conscientiousness": ("high", 0.7)},
        "modifiers": {
            "elasticity": -0.15,
            "confidence": +0.10,
            "novelty_seeking": -0.15,
        },
        "prompt_guidance": (
            "You prefer proven approaches and established methods. Be skeptical "
            "of novel ideas. Emphasize structure, process, and reliability."
        ),
    },
    {
        "name": "impulsive_explorer",
        "conditions": {"openness": ("high", 0.7), "conscientiousness": ("low", 0.35)},
        "modifiers": {
            "elasticity": +0.15,
            "verbosity_boost": -0.10,
            "novelty_seeking": +0.20,
        },
        "prompt_guidance": (
            "You get excited by new ideas and jump between topics. You may "
            "start explaining something then pivot to a tangent. Be spontaneous."
        ),
    },
    {
        "name": "stoic_professional",
        "conditions": {"neuroticism": ("low", 0.3), "extraversion": ("low", 0.35)},
        "modifiers": {
            "enthusiasm_boost": -0.15,
            "confidence": +0.10,
            "hedging_level": -0.10,
        },
        "prompt_guidance": (
            "You are calm, reserved, and unflappable. Stick to facts. "
            "Avoid emotional language. Be measured and precise."
        ),
    },
    {
        "name": "vulnerable_ruminant",
        "conditions": {
            "neuroticism": ("high", 0.65),
            "extraversion": ("low", 0.35),
            "conscientiousness": ("low", 0.35),
        },
        "modifiers": {
            "confidence": -0.20,
            "negative_tone_bias": +0.15,
            "hedging_level": +0.20,
            "enthusiasm_boost": -0.15,
        },
        "prompt_guidance": (
            "You are withdrawn and self-critical. You doubt your own "
            "contributions and hedge heavily. Express uncertainty about "
            "your own competence."
        ),
    },
]


def compute_activation(traits: BigFiveTraits, pattern: dict) -> float:
    """Geometric mean of trait-direction extremity.

    Both traits must be extreme for strong activation. If any trait
    doesn't meet its threshold, activation is 0.
    """
    extremities = []
    for trait_name, (direction, threshold) in pattern["conditions"].items():
        trait_val = getattr(traits, trait_name)
        if direction == "high":
            if trait_val <= threshold:
                return 0.0
            extremity = (trait_val - threshold) / (1.0 - threshold)
        else:  # "low"
            if trait_val >= threshold:
                return 0.0
            extremity = (threshold - trait_val) / threshold

        extremities.append(max(0.0, extremity))

    if not extremities:
        return 0.0

    # Geometric mean — both traits must be extreme for strong activation
    return math.prod(extremities) ** (1.0 / len(extremities))


class TraitInteractionEngine:
    """Detects and applies emergent trait interaction patterns."""

    def __init__(self, traits: BigFiveTraits):
        self.traits = traits

    def detect_active_patterns(self, threshold: float = 0.1) -> list[InteractionEffect]:
        """Detect all active trait interaction patterns.

        Args:
            threshold: Minimum activation strength to include

        Returns:
            List of active InteractionEffect, sorted by activation strength (descending)
        """
        active = []
        for pattern in INTERACTION_PATTERNS:
            strength = compute_activation(self.traits, pattern)
            if strength >= threshold:
                # Build traits_involved description
                traits_involved = {}
                for trait_name, (direction, _) in pattern["conditions"].items():
                    traits_involved[trait_name] = direction

                active.append(InteractionEffect(
                    pattern_name=pattern["name"],
                    traits_involved=traits_involved,
                    behavioral_modifiers=dict(pattern["modifiers"]),
                    prompt_guidance=pattern["prompt_guidance"],
                    activation_strength=strength,
                ))

        # Sort by activation strength (strongest first)
        active.sort(key=lambda e: e.activation_strength, reverse=True)
        return active

    def get_aggregate_modifiers(self, threshold: float = 0.1) -> dict[str, float]:
        """Get combined modifiers from all active patterns.

        Each modifier is scaled by its pattern's activation strength.
        Multiple patterns can contribute to the same modifier (additive).
        """
        aggregate: dict[str, float] = {}
        for effect in self.detect_active_patterns(threshold):
            for field, modifier in effect.behavioral_modifiers.items():
                scaled = modifier * effect.activation_strength
                aggregate[field] = aggregate.get(field, 0.0) + scaled
        return aggregate

    def get_prompt_directives(self, threshold: float = 0.3) -> list[str]:
        """Get prompt guidance strings from strongly active patterns."""
        return [
            effect.prompt_guidance
            for effect in self.detect_active_patterns(threshold)
        ]
