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
    enterprise_research — simulated stakeholder reaction to workplace rollout

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
ENTERPRISE_RESEARCH = "enterprise_research"

ALL_CONTEXT_TYPES = (
    KNOWLEDGE,
    OPINION,
    SOCIAL,
    EMOTIONAL,
    PERSONAL,
    ADVERSARIAL,
    ENTERPRISE_RESEARCH,
)

# Keyword lists — ordered from most specific to least.
# First match wins. Knowledge is the fallback, not matched by keywords.

_ADVERSARIAL_KEYWORDS = [
    "you're wrong", "you are wrong", "that's wrong", "that is wrong",
    "defend your", "prove it", "prove that", "i disagree",
    "that's not true", "that is not true", "challenge",
    "how can you say", "justify", "you can't seriously",
]

_ENTERPRISE_ACTOR_KEYWORDS = [
    "company", "organization", "organisation", "enterprise",
    "our company", "our organization", "our organisation",
    "restaurant group", "frontline staff", "employees", "managers",
    "engineers", "kitchen managers", "staff", "team", "teams",
]

_ENTERPRISE_ROLLOUT_KEYWORDS = [
    "rollout", "roll out", "require", "requires", "requiring",
    "mandatory", "policy", "tool", "assistant", "use an ai",
    "ai assistant", "ai code review", "every pull request",
    "case note", "shift plan", "supplier note",
]

_ENTERPRISE_RESEARCH_INTENT_KEYWORDS = [
    "concerns", "concern", "objection", "objections", "raise",
    "adoption", "blocker", "blockers", "trust", "quietly ignore",
    "ignore this rollout", "workaround", "resistance", "what would make",
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


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_enterprise_research(text: str) -> bool:
    """Detect workplace rollout research without matching generic business talk."""
    has_actor = _contains_any(text, _ENTERPRISE_ACTOR_KEYWORDS)
    has_rollout = _contains_any(text, _ENTERPRISE_ROLLOUT_KEYWORDS)
    has_research_intent = _contains_any(text, _ENTERPRISE_RESEARCH_INTENT_KEYWORDS)
    return has_actor and has_rollout and has_research_intent


def classify_context(user_input: str) -> str:
    """Classify user input into a context type.

    Returns one of the context type constants in ALL_CONTEXT_TYPES.

    Priority order: adversarial > enterprise_research > emotional > personal > social > opinion > knowledge.
    Adversarial first because "you're wrong about parties" is adversarial, not social.
    Enterprise research next because rollout prompts may use opinion/social wording.
    Emotional before personal because "how are you feeling about your routine" is emotional.
    """
    lower = user_input.lower()

    for keyword in _ADVERSARIAL_KEYWORDS:
        if keyword in lower:
            return ADVERSARIAL

    if _is_enterprise_research(lower):
        return ENTERPRISE_RESEARCH

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
