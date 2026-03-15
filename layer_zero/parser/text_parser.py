"""
Text description parser — Tier 1 input.

Extracts structured fields from natural language descriptions like:
    "A 35-year-old product manager in fintech who values innovation"

Uses regex + heuristics only. No NLP dependencies.

Known limitations (documented, not fixed in v1):
- No negation handling ("not very analytical" → incorrectly extracts "analytical")
- No compound roles ("product manager turned data scientist")
- No sarcasm detection
- Regex covers ~80% of natural description patterns
"""

from __future__ import annotations

import re

from layer_zero.models import MintRequest

# =============================================================================
# Known vocabularies
# =============================================================================

OCCUPATIONS = {
    "nurse", "doctor", "physician", "surgeon", "therapist", "psychologist",
    "teacher", "professor", "lecturer", "tutor",
    "engineer", "developer", "programmer", "architect", "designer",
    "software engineer", "data scientist", "data analyst", "researcher",
    "product manager", "project manager", "program manager",
    "chef", "cook", "baker",
    "lawyer", "attorney", "paralegal", "judge",
    "accountant", "auditor", "financial analyst", "banker",
    "journalist", "writer", "editor", "reporter",
    "musician", "artist", "painter", "sculptor", "photographer",
    "entrepreneur", "founder", "ceo", "cto", "cfo",
    "salesperson", "marketer", "marketing manager",
    "consultant", "advisor", "analyst",
    "social worker", "counselor",
    "police officer", "firefighter", "paramedic",
    "pilot", "mechanic", "electrician", "plumber",
    "fitness coach", "personal trainer", "coach",
    "pharmacist", "dentist", "veterinarian",
    "librarian", "archivist",
    "scientist", "biologist", "chemist", "physicist",
}

# Two-word occupations to check first (before single-word)
MULTI_WORD_OCCUPATIONS = sorted(
    [o for o in OCCUPATIONS if " " in o],
    key=len,
    reverse=True,  # longest first
)

INDUSTRIES = {
    "fintech", "finance", "banking", "healthcare", "health", "tech",
    "technology", "education", "legal", "law", "media", "journalism",
    "entertainment", "music", "art", "food", "culinary", "hospitality",
    "retail", "ecommerce", "manufacturing", "automotive", "aerospace",
    "biotech", "pharma", "pharmaceutical", "energy", "oil", "gas",
    "real estate", "construction", "agriculture", "nonprofit",
    "government", "military", "defense", "consulting", "marketing",
    "advertising", "insurance", "telecommunications", "logistics",
    "transportation", "sports", "fitness", "fashion", "beauty",
    "gaming", "cybersecurity", "ai", "machine learning",
}

TRAIT_ADJECTIVES = {
    # Maps to Big Five directions
    "analytical": ("openness", 0.1),
    "creative": ("openness", 0.1),
    "curious": ("openness", 0.1),
    "innovative": ("openness", 0.1),
    "open-minded": ("openness", 0.1),
    "traditional": ("openness", -0.1),
    "conventional": ("openness", -0.1),
    "organized": ("conscientiousness", 0.1),
    "disciplined": ("conscientiousness", 0.1),
    "methodical": ("conscientiousness", 0.1),
    "meticulous": ("conscientiousness", 0.1),
    "reliable": ("conscientiousness", 0.1),
    "spontaneous": ("conscientiousness", -0.1),
    "carefree": ("conscientiousness", -0.1),
    "outgoing": ("extraversion", 0.1),
    "social": ("extraversion", 0.1),
    "energetic": ("extraversion", 0.1),
    "enthusiastic": ("extraversion", 0.1),
    "assertive": ("extraversion", 0.1),
    "introverted": ("extraversion", -0.1),
    "reserved": ("extraversion", -0.1),
    "quiet": ("extraversion", -0.1),
    "shy": ("extraversion", -0.1),
    "warm": ("agreeableness", 0.1),
    "empathetic": ("agreeableness", 0.1),
    "compassionate": ("agreeableness", 0.1),
    "friendly": ("agreeableness", 0.1),
    "cooperative": ("agreeableness", 0.1),
    "nurturing": ("agreeableness", 0.1),
    "direct": ("agreeableness", -0.1),
    "blunt": ("agreeableness", -0.1),
    "competitive": ("agreeableness", -0.1),
    "tough": ("agreeableness", -0.1),
    "calm": ("neuroticism", -0.1),
    "relaxed": ("neuroticism", -0.1),
    "stable": ("neuroticism", -0.1),
    "confident": ("neuroticism", -0.1),
    "anxious": ("neuroticism", 0.1),
    "nervous": ("neuroticism", 0.1),
    "cautious": ("neuroticism", 0.1),
    "sensitive": ("neuroticism", 0.1),
    "risk-tolerant": ("neuroticism", -0.05),
    "risk-averse": ("neuroticism", 0.05),
}

GENDER_KEYWORDS = {
    "male": ["male", "man", "he", "his", "boy", "gentleman"],
    "female": ["female", "woman", "she", "her", "girl", "lady"],
    "non-binary": ["non-binary", "nonbinary", "nb", "enby", "they/them"],
}

# =============================================================================
# Parser
# =============================================================================

_AGE_PATTERN = re.compile(
    r"(?:^|\s)(\d{1,2})[\s-]?year[\s-]?old(?:s)?(?:\s|$|,|\.)",
    re.IGNORECASE,
)
_AGE_PATTERN_2 = re.compile(
    r"(?:age[d]?\s*[:=]?\s*(\d{1,2}))",
    re.IGNORECASE,
)
_AGE_PATTERN_3 = re.compile(
    r"(?:^|\s|,)(\d{2})(?:\s*,|\s*$|\s+(?:from|in|who|based|living))",
)


def parse_description(text: str) -> MintRequest:
    """Parse a natural language description into a MintRequest.

    Examples:
        "A 35-year-old product manager in fintech"
        → MintRequest(age=35, occupation="product manager", industry="fintech")

        "Cautious nurse from Tokyo, 28"
        → MintRequest(age=28, occupation="nurse", location="Tokyo", trait_hints=["cautious"])
    """
    if not text or not text.strip():
        raise ValueError("Description must not be empty")

    lower = text.lower().strip()
    req = MintRequest()

    # --- Age ---
    for pattern in (_AGE_PATTERN, _AGE_PATTERN_2, _AGE_PATTERN_3):
        match = pattern.search(lower)
        if match:
            age = int(match.group(1))
            if 18 <= age <= 100:
                req.age = age
                break

    # --- Occupation (multi-word first, then single-word) ---
    for occ in MULTI_WORD_OCCUPATIONS:
        if occ in lower:
            req.occupation = occ
            break
    if req.occupation is None:
        for occ in OCCUPATIONS:
            if " " not in occ and re.search(r"\b" + re.escape(occ) + r"\b", lower):
                req.occupation = occ
                break

    # --- Industry ---
    for ind in INDUSTRIES:
        if re.search(r"\b" + re.escape(ind) + r"\b", lower):
            req.industry = ind
            break

    # --- Location (after "from", "in", "based in") ---
    loc_match = re.search(
        r"(?:from|in|based in|living in)\s+([A-Z][a-zA-Z\s,]+?)(?:\s+who|\s+that|\s*,|\s*\.|$)",
        text,  # original case for proper nouns
    )
    if loc_match:
        location = loc_match.group(1).strip().rstrip(",.")
        if len(location) > 2 and location.lower() not in INDUSTRIES:
            req.location = location

    # --- Gender ---
    for gender, keywords in GENDER_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                req.gender = gender
                break
        if req.gender:
            break

    # --- Trait adjectives ---
    for adj in TRAIT_ADJECTIVES:
        if re.search(r"\b" + re.escape(adj) + r"\b", lower):
            req.trait_hints.append(adj)

    # --- Goals (after "who values", "values", "prioritizes") ---
    goal_match = re.search(
        r"(?:who\s+)?(?:values?|prioritizes?|cares?\s+about)\s+(.+?)(?:\s+and\s+is|\s*,|\s*\.|$)",
        lower,
    )
    if goal_match:
        goal_text = goal_match.group(1).strip()
        req.goals = [g.strip() for g in re.split(r"\s+and\s+|\s*,\s*", goal_text) if g.strip()]

    return req
