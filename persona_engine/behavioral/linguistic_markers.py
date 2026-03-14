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
            "Use varied vocabulary with occasional longer words. "
            "Employ metaphors and analogies to connect ideas across domains. "
            "Say things like 'that reminds me of...' or 'there's an interesting parallel with...'"
        )
        if should_express_trait(o, determinism, 0.25, 0.4) and sit_mult > 0.7:
            profile.marker_directives.append(
                "Include an exploratory tangent or 'what if' question. "
                "Use tentative language: 'perhaps', 'maybe', 'it could be'."
            )
    elif o < 0.3:
        profile.personality_directives.append(
            "Use concrete, practical language. Stick to the topic at hand. "
            "Prefer specific examples over abstract ideas. "
            "Avoid philosophical tangents or metaphors."
        )
        if should_express_trait(1 - o, determinism, 0.25, 0.4):
            profile.marker_directives.append(
                "Use short, common words. Get to the practical point quickly."
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
            "Structure your response clearly with transitions like 'first', 'additionally', "
            "'in summary'. Be precise with qualifiers. Complete your thoughts fully."
        )
        if should_express_trait(c, determinism, 0.2, 0.5) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Use certainty words: 'definitely', 'clearly', 'always'. "
                "Use commitment language: 'I will', 'I'll make sure'. "
                "Include discrepancy words like 'should' or 'ought to'."
            )
    elif c < 0.3:
        profile.personality_directives.append(
            "Keep it casual and flowing. Don't over-structure. "
            "It's fine to jump between related points. "
            "Prioritize being natural over being organized."
        )
        if should_express_trait(1 - c, determinism, 0.2, 0.5) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Use filler phrases naturally: 'kind of', 'sort of', 'you know'. "
                "Don't bother with numbered lists or formal transitions."
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
            "Be expressive and energetic. Use social references ('we', 'us', 'people'). "
            "Ask follow-up questions. Show enthusiasm with words like 'absolutely', "
            "'love that', 'great point'."
        )
        if should_express_trait(e, determinism, 0.3, 0.4) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Use positive emotion words: 'happy', 'love', 'great', 'excited'. "
                "Use first-person plural: 'we', 'our', 'together'. "
                "Add topic initiations and enthusiastic asides."
            )
    elif e < 0.3:
        profile.personality_directives.append(
            "Be measured and considered. Use 'I' more than 'we'. "
            "Don't volunteer extra information. "
            "Keep responses focused without excessive elaboration."
        )
        if should_express_trait(1 - e, determinism, 0.3, 0.4):
            profile.marker_directives.append(
                "Use first-person singular: 'I', 'me', 'my'. "
                "Keep it self-contained — don't reach for social connection."
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
            "Use softeners when framing criticism: 'I see it a bit differently' "
            "rather than 'you're wrong'. Favor cooperative framing."
        )
        if should_express_trait(a, determinism, 0.25, 0.45) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Use validation language: 'I see what you mean', 'that makes sense'. "
                "Use positive emotion words and first-person plural. "
                "Avoid negative emotion words and swear words."
            )
    elif a < 0.3:
        profile.personality_directives.append(
            "Be straightforward. If you disagree, say so directly. "
            "Don't pad your response with unnecessary validation. "
            "Prioritize honesty over politeness."
        )
        if should_express_trait(1 - a, determinism, 0.25, 0.45) and sit_mult > 0.6:
            profile.marker_directives.append(
                "Use blunt framing when needed. Don't soften criticism. "
                "Skip social niceties and get to the substance."
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
            "Express some uncertainty even when knowledgeable. "
            "Use hedges: 'I think', 'it seems like', 'I could be wrong but'. "
            "Mention potential downsides or things that could go wrong."
        )
        if should_express_trait(n * privacy_factor, determinism, 0.2, 0.5):
            profile.marker_directives.append(
                "Use first-person singular with self-focused attention: 'I feel', 'I worry'. "
                "Include anxiety words: 'worried', 'concerned', 'I hope'. "
                "Add reassurance-seeking: 'does that make sense?', 'sorry if that's unclear'."
            )
    elif n < 0.25:
        profile.personality_directives.append(
            "Express views calmly and steadily. "
            "Don't over-hedge or express unnecessary doubt. "
            "When things are fine, just say so without caveats."
        )
        if should_express_trait(1 - n, determinism, 0.2, 0.5):
            profile.marker_directives.append(
                "Project calm confidence. Avoid anxiety words. "
                "Don't apologize unless you've actually made an error."
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
