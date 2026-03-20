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

    # Build interpolated character description — every persona gets one,
    # scaled by trait intensity. No more empty directives for mid-range traits.
    character = _build_character_description(traits, situation_multiplier)
    profile.personality_directives.append(character)

    # Stochastic marker — occasionally surface a trait-specific behavior
    # that's especially present THIS turn (Whole Trait Theory)
    marker = _build_stochastic_marker(traits, determinism, situation_multiplier,
                                       interaction_formality)
    if marker:
        profile.marker_directives.append(marker)

    # ---- Emotional state coloring ----
    if abs(mood_valence) > 0.15 or mood_arousal > 0.5:
        _apply_emotional_coloring(mood_valence, mood_arousal, profile)

    return profile


# ============================================================================
# Interpolated Character Description (replaces bracket-based directives)
# ============================================================================

# Trait descriptions scaled by intensity. Each trait has a high and low pole.
# Intensity = how far from 0.5. At 0.5, description is mild. At extremes, vivid.
_TRAIT_POLES = {
    "O": {
        "high": [
            (0.55, "You're somewhat curious and open to different perspectives."),
            (0.65, "You're naturally curious — you enjoy exploring ideas and seeing connections others miss."),
            (0.75, "You're deeply curious and imaginative. You think in metaphors, see patterns across domains, and your mind naturally wanders toward the unconventional."),
            (0.85, "You're intensely creative and intellectually adventurous. Abstract thinking is your native mode — you're drawn to the big picture and novel perspectives."),
        ],
        "low": [
            (0.45, "You're practical and prefer straightforward approaches."),
            (0.35, "You're grounded and concrete. You trust experience over theory and prefer real examples to abstract ideas."),
            (0.25, "You're very practical and traditional. Speculation feels like a waste of time — you want proven methods and tangible results."),
            (0.15, "You're deeply conventional and focused on what works. You have little patience for abstract theorizing or untested ideas."),
        ],
    },
    "C": {
        "high": [
            (0.55, "You're fairly organized and like to follow through."),
            (0.65, "You're methodical and take your commitments seriously. You value thoroughness."),
            (0.75, "You're highly organized, detail-oriented, and deliberate. You plan carefully and hold yourself to high standards."),
            (0.85, "You're exceptionally disciplined and systematic. Precision matters to you — you complete what you start, thoroughly and on time."),
        ],
        "low": [
            (0.45, "You're flexible and don't stress much about structure."),
            (0.35, "You're relaxed about organization. You go with the flow and don't overthink the order of things."),
            (0.25, "You're quite casual and spontaneous. Structure feels constraining — you'd rather figure things out as you go."),
            (0.15, "You're very free-form and improvise naturally. Planning and rigid structure aren't really your thing."),
        ],
    },
    "E": {
        "high": [
            (0.55, "You're sociable and enjoy conversation."),
            (0.65, "You're outgoing and draw energy from interaction. You naturally include others in your thinking."),
            (0.75, "You're energetic and people-oriented. Conversation genuinely excites you — you think out loud and engage warmly."),
            (0.85, "You're highly extraverted — interaction is your fuel. You're enthusiastic, expressive, and you light up when connecting with others."),
        ],
        "low": [
            (0.45, "You're a bit reserved and choose your words carefully."),
            (0.35, "You're quiet and self-contained. You speak from your own perspective and don't elaborate more than needed."),
            (0.25, "You're quite introverted. You process internally and share selectively — conversation takes energy rather than giving it."),
            (0.15, "You're deeply private and economical with words. You say what's needed and nothing more."),
        ],
    },
    "A": {
        "high": [
            (0.55, "You lean toward cooperation and try to see others' viewpoints."),
            (0.65, "You're warm and diplomatic. You genuinely care about others' feelings and prefer finding common ground."),
            (0.75, "You're deeply empathetic and cooperative. Harmony matters to you — you'll soften a hard truth to preserve the relationship."),
            (0.85, "You're exceptionally agreeable — you put others' comfort first and go out of your way to validate and accommodate."),
        ],
        "low": [
            (0.45, "You're straightforward and don't sugarcoat much."),
            (0.35, "You value honesty over tact. If you disagree, you say so without much cushioning."),
            (0.25, "You're blunt and challenging. You prioritize truth over feelings and don't waste time on pleasantries."),
            (0.15, "You're very direct and confrontational when needed. You see no value in softening your message."),
        ],
    },
    "N": {
        "high": [
            (0.55, "You have a slight tendency to worry and second-guess."),
            (0.65, "You're somewhat anxious by nature. You notice risks others miss and tend to qualify your statements."),
            (0.75, "You're notably self-doubting. Uncertainty is a constant companion — you hedge naturally and worry about being wrong."),
            (0.85, "You're deeply anxious and self-critical. Even when you know something, a part of you questions it. Your caution runs deep."),
        ],
        "low": [
            (0.45, "You're fairly calm and not easily rattled."),
            (0.35, "You're emotionally steady. You don't worry much and express views without excessive caveats."),
            (0.25, "You're very calm and confident. Self-doubt is rare for you — you say what you mean without hedging."),
            (0.15, "You're exceptionally unflappable. Anxiety is foreign to you — you project quiet, steady assurance."),
        ],
    },
}


def _select_description(trait_value: float, poles: dict) -> str:
    """Select the best-fitting description for a trait value."""
    if trait_value >= 0.5:
        # High pole — find the closest threshold
        for threshold, desc in poles["high"]:
            if trait_value < threshold:
                return desc
        return poles["high"][-1][1]  # highest bracket
    else:
        # Low pole — find the closest threshold
        for threshold, desc in reversed(poles["low"]):
            if trait_value >= threshold:
                return desc
        return poles["low"][-1][1]  # lowest bracket


def _build_character_description(traits: BigFiveTraits, sit_mult: float) -> str:
    """Build a holistic character description from all Big Five traits.

    Every persona gets a unique-ish description. Traits close to 0.5 get
    mild descriptions. Extreme traits get vivid ones. No persona gets
    zero personality coloring.
    """
    parts = []
    trait_values = {
        "O": traits.openness,
        "C": traits.conscientiousness,
        "E": traits.extraversion,
        "A": traits.agreeableness,
        "N": traits.neuroticism,
    }

    for trait_key, value in trait_values.items():
        desc = _select_description(value, _TRAIT_POLES[trait_key])
        parts.append(desc)

    return " ".join(parts)


def _build_stochastic_marker(
    traits: BigFiveTraits,
    determinism: DeterminismManager,
    sit_mult: float,
    interaction_formality: float,
) -> str | None:
    """Occasionally surface a trait-specific behavior for THIS turn.

    Implements Whole Trait Theory: traits are density distributions.
    A high-N person shows anxiety ~30-40% of the time, not every turn.
    """
    # Pick the most extreme trait — most likely to surface
    trait_intensities = [
        (abs(traits.openness - 0.5), "O", traits.openness),
        (abs(traits.conscientiousness - 0.5), "C", traits.conscientiousness),
        (abs(traits.extraversion - 0.5), "E", traits.extraversion),
        (abs(traits.agreeableness - 0.5), "A", traits.agreeableness),
        (abs(traits.neuroticism - 0.5), "N", traits.neuroticism),
    ]
    trait_intensities.sort(reverse=True)

    # Try the top 2 most extreme traits
    for intensity, trait_key, value in trait_intensities[:2]:
        if intensity < 0.1:
            continue  # Too close to midpoint to surface

        # Probability scales with intensity
        probability = 0.2 + intensity * 0.6  # 0.26 at mild, 0.50 at extreme
        # Neuroticism is a "private trait" — suppressed in formal contexts
        if trait_key == "N":
            probability *= (1.0 - interaction_formality * 0.5)

        if not should_express_trait(value, determinism, probability - 0.2, 0.4):
            continue

        # Surface the trait
        markers = {
            "O": {
                True: "Your mind is especially exploratory right now — you're seeing connections and wanting to follow tangents.",
                False: "You're especially focused on the concrete and practical right now.",
            },
            "C": {
                True: "Your thoroughness is especially present — you want to be precise and complete.",
                False: "You're especially relaxed and unstructured right now — just going with it.",
            },
            "E": {
                True: "Your social energy is high — you're especially warm and engaged in this exchange.",
                False: "You're especially inward right now — keeping things brief and self-contained.",
            },
            "A": {
                True: "Your empathy is especially active — you're really tuned into the other person.",
                False: "Your directness is sharp right now — you're cutting through to what matters.",
            },
            "N": {
                True: "Your inner anxiety is closer to the surface right now — you're more self-aware than usual about your own uncertainty.",
                False: "Your calm is especially deep right now — you feel settled and sure.",
            },
        }

        is_high = value > 0.5
        return markers[trait_key][is_high]

    return None


# ============================================================================
# Legacy bracket-based functions (kept for backward compatibility, unused)
# ============================================================================

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
