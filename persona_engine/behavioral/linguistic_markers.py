"""
Linguistic Marker Engine — Research-Grounded Personality-to-Language Mapping

Based on empirical findings from:
- Pennebaker & King (1999): LIWC and Big Five correlations
- Yarkoni (2010): Personality in 100,000 Words (N=69,792)
- Koutsoumpis et al. meta-analysis: Big Five and linguistic markers
- Mairesse & Walker: PERSONAGE personality language generation
- Tausczik & Pennebaker (2010): Psychological Meaning of Words

Key insight: Individual traits explain 5-15% of linguistic variance.
Correlations are |r| = .08-.14 (self-report), |r| = .18-.39 (observer-report).
We model density distributions, not deterministic switches.

Phase R5: Personality-driven language generation for psychological realism.
"""

from dataclasses import dataclass, field

from persona_engine.schema.persona_schema import BigFiveTraits
from persona_engine.utils.determinism import DeterminismManager


@dataclass
class LinguisticProfile:
    """Personality-specific language directives for prompt generation."""
    personality_directives: list[str] = field(default_factory=list)
    marker_directives: list[str] = field(default_factory=list)
    emotional_coloring: str = ""


# ============================================================================
# Stochastic Trait Expression (Whole Trait Theory)
# ============================================================================

def should_express_trait(
    trait_value: float,
    determinism: DeterminismManager,
    base_probability: float = 0.3,
    trait_weight: float = 0.5,
) -> bool:
    """Probabilistically determine if a trait marker is expressed THIS turn.

    Implements Whole Trait Theory: traits are density distributions of states,
    not deterministic switches. A person at the 80th percentile on neuroticism
    shows anxious patterns ~30-40% of the time, not 100%.

    At trait=0.5: ~55% chance of expression
    At trait=0.9: ~75% chance of expression
    At trait=0.1: ~35% chance of expression

    Apply to secondary linguistic markers (hedging, enthusiasm, validation),
    NOT to primary structural effects (directness, confidence).
    """
    probability = base_probability + trait_value * trait_weight
    return determinism.should_trigger(probability)


# ============================================================================
# Personality-Specific Language Directives (R5.1 + R5.3)
# ============================================================================

def build_personality_language_directives(
    traits: BigFiveTraits,
    determinism: DeterminismManager,
    mood_valence: float = 0.0,
    mood_arousal: float = 0.0,
    interaction_formality: float = 0.5,
) -> LinguisticProfile:
    """Generate personality-specific language directives grounded in LIWC research.

    Args:
        traits: Big Five personality traits
        determinism: For stochastic trait expression
        mood_valence: Current mood valence (-1 to 1)
        mood_arousal: Current mood arousal (0 to 1)
        interaction_formality: Situational formality (0-1), high values compress
            trait expression (situational strength effect)

    Returns:
        LinguisticProfile with personality and marker directives
    """
    profile = LinguisticProfile()

    # Situational strength: formal contexts compress trait expression
    # (Tett & Guterman, 2000: strong situations reduce personality effects)
    situation_multiplier = 1.0 - interaction_formality * 0.3

    # ---- OPENNESS (r=.08-.12 with linguistic markers) ----
    _apply_openness_markers(traits, determinism, situation_multiplier, profile)

    # ---- CONSCIENTIOUSNESS (r=.10-.19) ----
    _apply_conscientiousness_markers(traits, determinism, situation_multiplier, profile)

    # ---- EXTRAVERSION (r=.10-.18) — "public trait" ----
    _apply_extraversion_markers(traits, determinism, situation_multiplier, profile)

    # ---- AGREEABLENESS (r=.12-.20) ----
    _apply_agreeableness_markers(traits, determinism, situation_multiplier, profile)

    # ---- NEUROTICISM (r=.08-.14) — "private trait" ----
    _apply_neuroticism_markers(traits, determinism, situation_multiplier,
                                interaction_formality, profile)

    # ---- Emotional state coloring ----
    if abs(mood_valence) > 0.15 or mood_arousal > 0.5:
        _apply_emotional_coloring(mood_valence, mood_arousal, profile)

    return profile


def _apply_openness_markers(
    traits: BigFiveTraits,
    determinism: DeterminismManager,
    sit_mult: float,
    profile: LinguisticProfile,
) -> None:
    """High-O: longer words, articles/prepositions, metaphorical language, tangents."""
    o = traits.openness

    if o > 0.7:
        profile.personality_directives.append(
            "You're naturally curious and drawn to connections between ideas. "
            "You tend to think in metaphors and see parallels across different domains. "
            "Your vocabulary is varied and you enjoy exploring ideas."
        )
        if should_express_trait(o, determinism, 0.25, 0.4) and sit_mult > 0.7:
            profile.marker_directives.append(
                "Let your mind wander a bit — an exploratory tangent or speculative "
                "question feels natural to you right now."
            )
    elif o < 0.3:
        profile.personality_directives.append(
            "You're practical and grounded. You prefer concrete, specific language "
            "and real examples over abstract theorizing. You stay on topic."
        )
        if should_express_trait(1 - o, determinism, 0.25, 0.4):
            profile.marker_directives.append(
                "Keep it simple and direct. Get to the practical point."
            )


def _apply_conscientiousness_markers(
    traits: BigFiveTraits,
    determinism: DeterminismManager,
    sit_mult: float,
    profile: LinguisticProfile,
) -> None:
    """High-C: certainty words, structure, fewer negations, commitment language."""
    c = traits.conscientiousness

    if c > 0.7:
        profile.personality_directives.append(
            "You're organized and thorough. You naturally structure your thoughts "
            "and follow through on points. You value precision and completeness."
        )
        if should_express_trait(c, determinism, 0.2, 0.5) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Your natural certainty and commitment comes through here. "
                "You speak with conviction and a sense of personal responsibility."
            )
    elif c < 0.3:
        profile.personality_directives.append(
            "You're casual and go-with-the-flow. You don't overthink structure "
            "and your thoughts come out naturally, sometimes jumping between points."
        )
        if should_express_trait(1 - c, determinism, 0.2, 0.5) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Your relaxed nature shows — you're not trying to be perfectly organized."
            )


def _apply_extraversion_markers(
    traits: BigFiveTraits,
    determinism: DeterminismManager,
    sit_mult: float,
    profile: LinguisticProfile,
) -> None:
    """High-E: positive emotion words, social references, higher word count, informal.
    NOTE: Extraversion is a 'public trait' — markers strongest in social contexts."""
    e = traits.extraversion

    if e > 0.7:
        profile.personality_directives.append(
            "You're energetic and people-oriented. You naturally include others "
            "in your thinking, show genuine enthusiasm, and enjoy the exchange. "
            "Conversation energizes you."
        )
        if should_express_trait(e, determinism, 0.3, 0.4) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Your social warmth and positive energy are especially present "
                "right now. You think in terms of 'we' and shared experience."
            )
    elif e < 0.3:
        profile.personality_directives.append(
            "You're reserved and self-contained. You speak from your own perspective "
            "and don't elaborate more than needed. You're thoughtful, not withdrawn "
            "— just private."
        )
        if should_express_trait(1 - e, determinism, 0.3, 0.4):
            profile.marker_directives.append(
                "You're in your own head right now — speaking from personal "
                "perspective, not reaching outward."
            )


def _apply_agreeableness_markers(
    traits: BigFiveTraits,
    determinism: DeterminismManager,
    sit_mult: float,
    profile: LinguisticProfile,
) -> None:
    """High-A: positive emotion words, fewer negations, first-person plural,
    validation language, softer criticism framing, accommodation.
    A accounts for ~20% of variance in conflict avoidance style."""
    a = traits.agreeableness

    if a > 0.7:
        profile.personality_directives.append(
            "You genuinely care about the other person's perspective. When you "
            "disagree, you do it gently — you'd rather find common ground than "
            "win an argument. Cooperation comes naturally to you."
        )
        if should_express_trait(a, determinism, 0.25, 0.45) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Your warmth and desire to connect are especially present. "
                "You naturally validate before offering a different view."
            )
    elif a < 0.3:
        profile.personality_directives.append(
            "You value honesty over politeness. If you disagree, you say so. "
            "You don't waste words on pleasantries when there's substance to address."
        )
        if should_express_trait(1 - a, determinism, 0.25, 0.45) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Your directness is especially strong right now. "
                "You're cutting through to what matters."
            )


def _apply_neuroticism_markers(
    traits: BigFiveTraits,
    determinism: DeterminismManager,
    sit_mult: float,
    interaction_formality: float,
    profile: LinguisticProfile,
) -> None:
    """High-N: first-person singular, negative emotion words, anxiety words,
    hedging, over-apologizing, cognitive distortion patterns.
    NOTE: Neuroticism is a 'private trait' — markers emerge more in intimate contexts."""
    n = traits.neuroticism

    # Private trait: expression suppressed in formal contexts
    privacy_factor = 1.0 - interaction_formality * 0.5

    if n > 0.65:
        profile.personality_directives.append(
            "You tend to second-guess yourself, even when you know things. "
            "Uncertainty is a constant companion — you naturally see the risks "
            "and potential problems others might miss. This isn't weakness, "
            "it's how you process."
        )
        if should_express_trait(n * privacy_factor, determinism, 0.2, 0.5):
            profile.marker_directives.append(
                "Your inner anxiety is closer to the surface right now. "
                "You're more self-aware than usual, noticing your own uncertainty "
                "and wanting reassurance that you're making sense."
            )
    elif n < 0.25:
        profile.personality_directives.append(
            "You're emotionally steady. You don't worry much and it shows — "
            "you express views calmly without unnecessary caveats or self-doubt."
        )
        if should_express_trait(1 - n, determinism, 0.2, 0.5):
            profile.marker_directives.append(
                "Your calm steadiness is especially evident right now. "
                "You see no reason to hedge or apologize."
            )


def _apply_emotional_coloring(
    mood_valence: float,
    mood_arousal: float,
    profile: LinguisticProfile,
) -> None:
    """Add subtle emotional coloring based on current mood state."""
    if mood_valence > 0.3:
        if mood_arousal > 0.5:
            profile.emotional_coloring = (
                "Your current mood is upbeat and energized. "
                "Let this subtly color your response with warmth and enthusiasm."
            )
        else:
            profile.emotional_coloring = (
                "Your current mood is content and at ease. "
                "Let this produce a calm, warm undertone."
            )
    elif mood_valence < -0.2:
        if mood_arousal > 0.5:
            profile.emotional_coloring = (
                "Your current mood is tense or stressed. "
                "Let this subtly show as slightly shorter patience or more cautious phrasing."
            )
        else:
            profile.emotional_coloring = (
                "Your current mood is subdued or melancholy. "
                "Let this show as a quieter, more reflective tone."
            )
