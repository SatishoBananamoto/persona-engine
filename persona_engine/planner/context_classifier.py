"""
Context Classifier — determines what kind of input drives the response.

Separate from InteractionMode (conversation shape) and ConversationGoal
(what we're trying to achieve). Context type determines WHICH pipeline
computations are relevant for this specific input.

Categories:
    knowledge   — domain expertise question ("explain quantum entanglement")
    opinion     — asking for persona's view ("what do you think about X?")
    social      — social situation/scenario ("you're at a party...")
    emotional   — emotional check-in ("how are you feeling?")
    personal    — about the persona's life ("tell me about your routine")
    adversarial — challenging the persona ("you're wrong", "defend your view")

Default: knowledge (preserves current pipeline behavior for unmatched inputs).

Upgrade path: embedding similarity or LLM classifier. This module is
intentionally isolated so the interface stays the same.
"""

from __future__ import annotations

# Context type constants
KNOWLEDGE = "knowledge"
OPINION = "opinion"
SOCIAL = "social"
EMOTIONAL = "emotional"
PERSONAL = "personal"
ADVERSARIAL = "adversarial"

ALL_CONTEXT_TYPES = (KNOWLEDGE, OPINION, SOCIAL, EMOTIONAL, PERSONAL, ADVERSARIAL)

# Keyword lists — ordered from most specific to least.
# First match wins. Knowledge is the fallback, not matched by keywords.

_ADVERSARIAL_KEYWORDS = [
    "you're wrong", "you are wrong", "that's wrong", "that is wrong",
    "defend your", "prove it", "prove that", "i disagree",
    "that's not true", "that is not true", "challenge",
    "how can you say", "justify", "you can't seriously",
]

_EMOTIONAL_KEYWORDS = [
    "how are you feeling", "how do you feel", "are you okay",
    "are you worried", "are you happy", "are you stressed",
    "are you anxious", "are you excited", "are you sad",
    "are you angry", "are you scared", "are you nervous",
    "how's your mood", "how is your mood",
    "what's bothering you", "what is bothering you",
    "cheer up", "feeling down", "feeling good",
]

_PERSONAL_KEYWORDS = [
    "tell me about yourself", "tell me about your",
    "about yourself", "your routine", "your morning",
    "your life", "your day", "your childhood", "your family",
    "your background", "your history", "your story",
    "what do you enjoy", "what do you do for fun",
    "your hobbies", "your interests", "your favorite",
    "your favourite", "describe yourself",
    "who are you", "where are you from",
    "where did you grow up",
]

_SOCIAL_KEYWORDS = [
    "at a party", "at a dinner", "at an event", "at a gathering",
    "meeting someone", "meet someone new",
    "introduce yourself", "group of people",
    "social situation", "social setting", "social event",
    "networking event", "a stranger", "new people",
    "making friends", "small talk with",
    "if you were at", "imagine you're at",
    "you're at a", "you are at a",
    "a friend asks", "your friend asks",
    "a colleague asks", "your colleague",
    "someone approaches you",
]

_OPINION_KEYWORDS = [
    "what do you think", "what's your opinion",
    "what is your opinion", "your view on",
    "your views on", "how do you feel about",
    "what's your take", "what is your take",
    "your thoughts on", "your perspective on",
    "do you agree", "do you believe",
    "do you prefer", "would you rather",
    "what would you choose", "which do you prefer",
    "do you support", "are you for or against",
    "what's your stance", "what is your stance",
    "what side are you on",
]


def classify_context(user_input: str) -> str:
    """Classify user input into a context type.

    Returns one of: knowledge, opinion, social, emotional, personal, adversarial.

    Priority order: adversarial > emotional > personal > social > opinion > knowledge.
    Adversarial first because "you're wrong about parties" is adversarial, not social.
    Emotional before personal because "how are you feeling about your routine" is emotional.
    """
    lower = user_input.lower()

    for keyword in _ADVERSARIAL_KEYWORDS:
        if keyword in lower:
            return ADVERSARIAL

    for keyword in _EMOTIONAL_KEYWORDS:
        if keyword in lower:
            return EMOTIONAL

    for keyword in _PERSONAL_KEYWORDS:
        if keyword in lower:
            return PERSONAL

    for keyword in _SOCIAL_KEYWORDS:
        if keyword in lower:
            return SOCIAL

    for keyword in _OPINION_KEYWORDS:
        if keyword in lower:
            return OPINION

    return KNOWLEDGE
