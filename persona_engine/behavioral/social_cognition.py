"""
Social Cognition — Theory of Mind & User Modeling

Models how the persona perceives and adapts to the conversational partner.
Based on social cognition research:
- Theory of Mind: Inferring others' mental states
- Perspective-taking: Adjusting to perceived user expertise/emotion
- Self-disclosure reciprocity: Matching disclosure depth
- Self-schema protection: Defending identity-relevant dimensions

Phase R6: Social cognition for psychological realism.
"""

import re
from dataclasses import dataclass, field

from persona_engine.schema.persona_schema import BigFiveTraits


# ============================================================================
# User Model
# ============================================================================

@dataclass
class UserModel:
    """Inferred model of the conversational partner."""
    inferred_expertise: float = 0.5  # 0=novice, 1=expert
    inferred_formality: float = 0.5  # 0=casual, 1=formal
    inferred_emotion_valence: float = 0.0  # -1 to 1
    disclosed: bool = False  # Did user self-disclose?
    disclosure_depth: float = 0.0  # 0-1 depth of disclosure


@dataclass
class AdaptationDirectives:
    """How the persona should adapt to the inferred user model."""
    formality_shift: float = 0.0  # Shift toward user's formality
    depth_shift: float = 0.0  # Adjust content depth to user expertise
    disclosure_reciprocity: float = 0.0  # Increase disclosure to match user
    prompt_directives: list[str] = field(default_factory=list)


# ============================================================================
# User Inference
# ============================================================================

# Technical vocabulary markers for expertise detection
_TECHNICAL_MARKERS = {
    "algorithm", "implementation", "architecture", "framework", "paradigm",
    "methodology", "empirical", "hypothesis", "correlation", "regression",
    "optimization", "throughput", "latency", "infrastructure", "deployment",
    "specification", "abstraction", "polymorphism", "asynchronous", "concurrent",
}

# Formal language markers
_FORMAL_MARKERS = {
    "would", "shall", "furthermore", "moreover", "regarding", "concerning",
    "consequently", "therefore", "nonetheless", "notwithstanding",
    "respectfully", "accordingly", "pursuant",
}

# Disclosure markers (user sharing personal info)
_DISCLOSURE_MARKERS = [
    "i feel", "i think", "personally", "in my experience",
    "i've been", "i've had", "i struggle", "i worry",
    "my family", "my partner", "my boss", "my friend",
    "to be honest", "honestly", "between us",
]


def infer_user_model(user_input: str) -> UserModel:
    """Infer a model of the user from their input.

    Lightweight keyword-based inference — not deep NLP, just enough
    for personality-modulated adaptation.
    """
    lower = user_input.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    # Expertise inference
    tech_hits = len(words & _TECHNICAL_MARKERS)
    # Longer sentences with more complex vocabulary suggest expertise
    word_count = len(lower.split())
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    expertise = min(1.0, tech_hits * 0.15 + (avg_word_len - 4) * 0.1)
    expertise = max(0.0, expertise)

    # Formality inference
    formal_hits = len(words & _FORMAL_MARKERS)
    has_contractions = any(c in lower for c in ["'t", "'m", "'re", "'ve", "'ll", "'d"])
    formality = min(1.0, formal_hits * 0.15 + (0.0 if has_contractions else 0.15))
    formality = max(0.0, formality)

    # Disclosure inference
    disclosed = any(marker in lower for marker in _DISCLOSURE_MARKERS)
    disclosure_depth = 0.0
    if disclosed:
        disclosure_hits = sum(1 for m in _DISCLOSURE_MARKERS if m in lower)
        disclosure_depth = min(1.0, disclosure_hits * 0.25)
        # Longer personal messages suggest deeper disclosure
        if word_count > 30:
            disclosure_depth = min(1.0, disclosure_depth + 0.15)

    # Emotion inference (reuse detect_user_emotion would be circular, keep simple)
    emotion_valence = 0.0
    positive_words = {"great", "love", "amazing", "wonderful", "excited", "happy"}
    negative_words = {"frustrated", "angry", "worried", "sad", "terrible", "awful"}
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    if pos > neg:
        emotion_valence = min(1.0, (pos - neg) * 0.2)
    elif neg > pos:
        emotion_valence = max(-1.0, -(neg - pos) * 0.2)

    return UserModel(
        inferred_expertise=expertise,
        inferred_formality=formality,
        inferred_emotion_valence=emotion_valence,
        disclosed=disclosed,
        disclosure_depth=disclosure_depth,
    )


# ============================================================================
# Personality-Modulated Adaptation
# ============================================================================

def compute_adaptation(
    user_model: UserModel,
    traits: BigFiveTraits,
    base_disclosure: float = 0.5,
) -> AdaptationDirectives:
    """Compute how persona adapts to perceived user, modulated by personality.

    - High-A: mirrors user's formality (accommodation)
    - High-E: matches user's energy/enthusiasm
    - High-O: adjusts depth to user's expertise
    - Low-A/Low-E: maintains own style regardless of user

    Self-disclosure reciprocity: People disclose more when the other person does.
    Modulated by E + A (social warmth increases reciprocity).
    """
    directives = AdaptationDirectives()

    # ---- Formality mirroring (High-A) ----
    if traits.agreeableness > 0.6:
        # High-A mirrors user formality
        mirror_strength = (traits.agreeableness - 0.5) * 0.4
        directives.formality_shift = (user_model.inferred_formality - 0.5) * mirror_strength
        if abs(directives.formality_shift) > 0.05:
            direction = "more formal" if directives.formality_shift > 0 else "more casual"
            directives.prompt_directives.append(
                f"The user's tone is {direction} — match their register slightly."
            )

    # ---- Depth calibration (High-O) ----
    if traits.openness > 0.6:
        if user_model.inferred_expertise > 0.6:
            directives.depth_shift = 0.15
            directives.prompt_directives.append(
                "The user seems knowledgeable — you can go deeper and use more "
                "technical language."
            )
        elif user_model.inferred_expertise < 0.2:
            directives.depth_shift = -0.15
            directives.prompt_directives.append(
                "The user seems unfamiliar with this topic — simplify and "
                "use more accessible language."
            )

    # ---- Energy matching (High-E) ----
    if traits.extraversion > 0.7:
        if user_model.inferred_emotion_valence > 0.3:
            directives.prompt_directives.append(
                "The user is enthusiastic — match their energy with equal warmth."
            )
        elif user_model.inferred_emotion_valence < -0.3:
            directives.prompt_directives.append(
                "The user seems down — offer genuine warmth and support."
            )

    # ---- Self-disclosure reciprocity ----
    if user_model.disclosed:
        # Reciprocity strength modulated by E + A
        reciprocity_strength = (traits.extraversion + traits.agreeableness) / 2.0
        boost = user_model.disclosure_depth * reciprocity_strength * 0.3
        directives.disclosure_reciprocity = boost
        if boost > 0.05:
            directives.prompt_directives.append(
                "The user shared something personal — reciprocate with appropriate "
                "openness. Share a relevant personal perspective."
            )

    return directives


# ============================================================================
# Self-Schema Protection (Markus, 1977)
# ============================================================================

# Common self-schema patterns and their challenge keywords
_SCHEMA_CHALLENGE_KEYWORDS: dict[str, list[str]] = {
    "competent_researcher": ["research", "methodology", "findings", "study", "analysis"],
    "empathetic_listener": ["listen", "understand", "empathy", "care", "feeling"],
    "independent_thinker": ["think", "opinion", "independent", "original", "creative"],
    "skilled_professional": ["skill", "professional", "competent", "experience", "expert"],
    "caring_person": ["care", "kind", "help", "support", "generous"],
}

_CHALLENGE_WORDS = {"wrong", "bad", "poor", "weak", "lacking", "inadequate",
                    "not", "don't", "can't", "fail", "unable", "incompetent"}


def detect_schema_relevance(
    user_input: str,
    self_schemas: list[str],
) -> tuple[str | None, bool]:
    """Detect if user input is relevant to a persona's self-schema.

    Returns:
        (matched_schema, is_challenge): schema name and whether it's a challenge
    """
    if not self_schemas:
        return None, False

    lower = user_input.lower()
    words = set(re.findall(r'\b\w+\b', lower))

    for schema in self_schemas:
        keywords = _SCHEMA_CHALLENGE_KEYWORDS.get(schema, [])
        if not keywords:
            # Unknown schema: use the schema name words as keywords
            keywords = schema.replace("_", " ").split()

        hits = sum(1 for kw in keywords if kw in lower)
        if hits >= 1:
            # Check if it's a challenge (negative framing of schema-relevant content)
            challenge_hits = len(words & _CHALLENGE_WORDS)
            is_challenge = challenge_hits >= 1
            return schema, is_challenge

    return None, False


@dataclass
class SchemaEffect:
    """Effect of self-schema activation on IR fields."""
    elasticity_modifier: float = 0.0
    confidence_modifier: float = 0.0
    disclosure_modifier: float = 0.0
    prompt_directive: str = ""


def compute_schema_effect(
    matched_schema: str | None,
    is_challenge: bool,
) -> SchemaEffect:
    """Compute IR modifications from self-schema activation.

    When challenged on schema-relevant dimension:
    - elasticity decreases (resist identity-threatening change)
    - confidence increases (assert competence)

    When validated on schema-relevant dimension:
    - disclosure increases (willingness to elaborate on who they are)
    """
    if matched_schema is None:
        return SchemaEffect()

    if is_challenge:
        return SchemaEffect(
            elasticity_modifier=-0.10,
            confidence_modifier=0.05,
            prompt_directive=(
                f"Your identity as a '{matched_schema.replace('_', ' ')}' is being challenged. "
                f"Defend this aspect of yourself with calm conviction."
            ),
        )
    else:
        # Validation — share more about this aspect
        return SchemaEffect(
            disclosure_modifier=0.05,
            prompt_directive=(
                f"This topic relates to your identity as a '{matched_schema.replace('_', ' ')}'. "
                f"You can elaborate confidently on this dimension of yourself."
            ),
        )
