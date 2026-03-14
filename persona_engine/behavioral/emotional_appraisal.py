"""
Emotional Appraisal Engine — Personality-Dependent Emotion Generation

Based on Scherer's Component Process Model, emotions don't arise from
mechanical triggers — they emerge from how the persona *interprets*
events, which depends on personality.

Appraisal dimensions:
1. Novelty: Is this unexpected? (modulated by Openness)
2. Pleasantness: Is this good/bad? (modulated by Neuroticism, Agreeableness)
3. Goal relevance: Does this matter? (modulated by Conscientiousness)
4. Coping potential: Can I handle this? (modulated by Neuroticism)
5. Norm compatibility: Does this fit values? (modulated by Agreeableness)

Phase R4: Emotional cognition for psychological realism.
"""

import re
from dataclasses import dataclass

from persona_engine.schema.persona_schema import BigFiveTraits


@dataclass
class EmotionalAppraisal:
    """Result of personality-dependent emotional appraisal."""
    valence_delta: float  # Change in mood valence (-1 to +1)
    arousal_delta: float  # Change in arousal (-1 to +1)
    dominant_emotion: str  # e.g. "interest", "anger", "joy", "fear"
    appraisal_notes: list[str]  # Explanation of what drove the appraisal


# ============================================================================
# User Emotion Detection (keyword-based)
# ============================================================================

_ENTHUSIASM_MARKERS = {
    "excited", "amazing", "wonderful", "fantastic", "love", "great",
    "awesome", "incredible", "brilliant", "thrilled", "excellent",
}
_FRUSTRATION_MARKERS = {
    "frustrated", "annoying", "terrible", "awful", "hate", "stupid",
    "ridiculous", "useless", "worst", "broken", "wrong", "fail",
}
_WORRY_MARKERS = {
    "worried", "anxious", "nervous", "scared", "afraid", "concerned",
    "dangerous", "risk", "uncertain", "doubt",
}
_CURIOSITY_MARKERS = {
    "curious", "wonder", "interesting", "fascinated", "how", "why",
    "what if", "explore", "think about",
}
_CHALLENGE_MARKERS = {
    "wrong", "disagree", "incorrect", "mistake", "actually", "but",
    "however", "no way", "impossible", "doubt",
}
_PRAISE_MARKERS = {
    "thank", "helpful", "appreciate", "good job", "well done",
    "impressive", "smart", "insightful",
}


def detect_user_emotion(user_input: str) -> dict[str, float]:
    """Detect emotional signals in user input via keyword matching.

    Returns dict of emotion category → intensity (0-1).
    """
    lower = user_input.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    signals: dict[str, float] = {}

    # Count marker hits
    enthusiasm_hits = len(words & _ENTHUSIASM_MARKERS)
    frustration_hits = len(words & _FRUSTRATION_MARKERS)
    worry_hits = len(words & _WORRY_MARKERS)
    curiosity_hits = len(words & _CURIOSITY_MARKERS)
    challenge_hits = len(words & _CHALLENGE_MARKERS)
    praise_hits = len(words & _PRAISE_MARKERS)

    # Also check multi-word patterns
    for phrase in ["what if", "good job", "well done", "no way"]:
        if phrase in lower:
            if phrase in ("what if",):
                curiosity_hits += 1
            elif phrase in ("good job", "well done"):
                praise_hits += 1
            elif phrase == "no way":
                challenge_hits += 1

    # Exclamation marks amplify
    exclamation_boost = min(lower.count("!") * 0.1, 0.3)

    if enthusiasm_hits:
        signals["joy"] = min(1.0, enthusiasm_hits * 0.25 + exclamation_boost)
    if frustration_hits:
        signals["anger"] = min(1.0, frustration_hits * 0.25 + exclamation_boost)
    if worry_hits:
        signals["fear"] = min(1.0, worry_hits * 0.2)
    if curiosity_hits:
        signals["interest"] = min(1.0, curiosity_hits * 0.2)
    if challenge_hits:
        signals["challenge"] = min(1.0, challenge_hits * 0.25)
    if praise_hits:
        signals["trust"] = min(1.0, praise_hits * 0.3)

    # Question marks suggest curiosity
    if "?" in user_input:
        signals["interest"] = min(1.0, signals.get("interest", 0.0) + 0.15)

    return signals


# ============================================================================
# Personality-Dependent Appraisal
# ============================================================================

def appraise_event(
    user_emotion: dict[str, float],
    traits: BigFiveTraits,
    current_stress: float = 0.0,
) -> EmotionalAppraisal:
    """Appraise user input through personality lens.

    Same event produces different emotional responses based on Big Five:
    - High-N: Amplifies threats, reduces coping → more fear/sadness
    - High-A: Reduces anger from challenges, amplifies empathy
    - High-O: Novelty is pleasant → more interest, less fear
    - High-C: Disorder triggers concern → more goal-relevance
    - High-E: Social engagement is pleasant → more joy
    """
    valence_delta = 0.0
    arousal_delta = 0.0
    dominant_emotion = "neutral"
    notes: list[str] = []

    # ---- Joy / positive contagion ----
    user_joy = user_emotion.get("joy", 0.0)
    if user_joy > 0.1:
        # Extraversion amplifies positive contagion
        contagion = user_joy * (0.15 + traits.extraversion * 0.2)
        # Agreeableness also increases empathetic mirroring
        contagion += user_joy * traits.agreeableness * 0.1
        valence_delta += contagion
        arousal_delta += contagion * 0.5
        notes.append(f"Positive contagion: +{contagion:.3f} valence (E={traits.extraversion:.2f})")

    # ---- Anger / frustration ----
    user_anger = user_emotion.get("anger", 0.0)
    if user_anger > 0.1:
        # High-A buffers against anger contagion
        anger_resistance = traits.agreeableness * 0.15
        # Low-N also resists
        anger_resistance += (1 - traits.neuroticism) * 0.1
        anger_impact = max(0.0, user_anger * 0.2 - anger_resistance)
        valence_delta -= anger_impact
        arousal_delta += anger_impact * 0.3
        notes.append(f"Frustration absorbed: -{anger_impact:.3f} valence (A-buffer={anger_resistance:.3f})")

    # ---- Challenge / disagreement ----
    user_challenge = user_emotion.get("challenge", 0.0)
    if user_challenge > 0.1:
        # High-N appraises challenge as threat
        if traits.neuroticism > 0.6:
            threat = user_challenge * traits.neuroticism * 0.2
            valence_delta -= threat
            arousal_delta += threat * 0.5
            notes.append(f"Challenge → threat (N={traits.neuroticism:.2f}): -{threat:.3f}")
            dominant_emotion = "fear"
        # High-O appraises challenge as interesting
        elif traits.openness > 0.6:
            curiosity = user_challenge * traits.openness * 0.15
            valence_delta += curiosity * 0.3
            arousal_delta += curiosity * 0.3
            notes.append(f"Challenge → curiosity (O={traits.openness:.2f}): +{curiosity:.3f}")
            dominant_emotion = "interest"
        # Low-A appraises challenge as confrontation
        elif traits.agreeableness < 0.4:
            fight = user_challenge * (1 - traits.agreeableness) * 0.15
            arousal_delta += fight * 0.4
            notes.append(f"Challenge → confrontation (low-A={traits.agreeableness:.2f}): arousal +{fight*0.4:.3f}")
            dominant_emotion = "anger"
        else:
            # Default: mild concern
            valence_delta -= user_challenge * 0.05
            notes.append("Challenge → mild concern")

    # ---- Fear / worry contagion ----
    user_fear = user_emotion.get("fear", 0.0)
    if user_fear > 0.1:
        # High-N amplifies worry contagion
        worry_contagion = user_fear * (0.1 + traits.neuroticism * 0.25)
        # Low-N is resistant
        valence_delta -= worry_contagion
        arousal_delta += worry_contagion * 0.3
        notes.append(f"Worry contagion: -{worry_contagion:.3f} (N={traits.neuroticism:.2f})")
        if dominant_emotion == "neutral":
            dominant_emotion = "fear"

    # ---- Interest / curiosity ----
    user_interest = user_emotion.get("interest", 0.0)
    if user_interest > 0.1:
        # High-O amplifies interest
        interest_boost = user_interest * (0.1 + traits.openness * 0.2)
        valence_delta += interest_boost * 0.5
        arousal_delta += interest_boost * 0.3
        notes.append(f"Interest boost: +{interest_boost:.3f} (O={traits.openness:.2f})")
        if dominant_emotion == "neutral":
            dominant_emotion = "interest"

    # ---- Trust / praise ----
    user_trust = user_emotion.get("trust", 0.0)
    if user_trust > 0.1:
        # Everyone responds to praise, but high-N benefits more (reduces anxiety)
        praise_effect = user_trust * 0.2
        if traits.neuroticism > 0.5:
            praise_effect += user_trust * 0.1  # Extra relief for anxious
        valence_delta += praise_effect
        notes.append(f"Praise effect: +{praise_effect:.3f}")
        if dominant_emotion == "neutral":
            dominant_emotion = "joy"

    # ---- Stress amplification ----
    if current_stress > 0.5:
        # Under stress, negative appraisals are amplified
        if valence_delta < 0:
            stress_amplifier = 1.0 + current_stress * 0.3
            valence_delta *= stress_amplifier
            notes.append(f"Stress amplifies negative: ×{stress_amplifier:.2f}")

    # Clamp
    valence_delta = max(-0.4, min(0.4, valence_delta))
    arousal_delta = max(-0.3, min(0.3, arousal_delta))

    if dominant_emotion == "neutral":
        if valence_delta > 0.05:
            dominant_emotion = "joy"
        elif valence_delta < -0.05:
            dominant_emotion = "sadness"

    return EmotionalAppraisal(
        valence_delta=valence_delta,
        arousal_delta=arousal_delta,
        dominant_emotion=dominant_emotion,
        appraisal_notes=notes,
    )
