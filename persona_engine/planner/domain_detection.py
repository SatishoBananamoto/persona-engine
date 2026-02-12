"""
Domain Detection - Deterministic Keyword-Based Domain Scoring

Production-grade domain detection that:
- Uses weighted keyword matching with n-gram support for phrases
- Provides deterministic tie-breaking (persona wins ties)
- Includes input snippet in fallback citations
- Falls back gracefully when registry/persona is sparse
- Uses strict whitelisting for short tokens to prevent false positives
- prevents double-counting of phrases and unigrams
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from persona_engine.planner.trace_context import TraceContext


# =============================================================================
# CONFIGURATION
# =============================================================================

DOMAIN_MIN_SCORE = 0.30          # Minimum score to consider a domain match
INPUT_SNIPPET_LENGTH = 40        # Characters to include in fallback citations
DEFAULT_DOMAIN = "general"
DEFAULT_TOPIC_RELEVANCE = 0.5
DEFAULT_EVIDENCE_STRENGTH = 0.2
STRONG_CHALLENGE_STRENGTH = 0.8
MODERATE_CHALLENGE_STRENGTH = 0.6
MAX_NGRAM_SIZE = 3               # Up to trigrams for phrase matching
NEGATIVE_KEYWORD_PENALTY = 0.3

# Short tokens whitelist - authorized 2-letter technical terms
# All other tokens < 3 chars are ignored to prevent stopword noise ("to", "in", etc)
TECHNICAL_TERMS_WHITELIST = {"ux", "ai", "ml", "qa", "ui", "db", "api", "io", "vr", "ar"}


# =============================================================================
# UTILITIES
# =============================================================================

def _get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Safe field accessor that works for both dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def tokenize_with_ngrams(
    text: str,
    max_n: int = MAX_NGRAM_SIZE,
) -> tuple[list[str], set[str]]:
    """
    Tokenize input into unigrams + phrase n-grams.

    Returns:
        (tokens, ngrams) where tokens is the unigram list and ngrams
        is a set of bigram/trigram phrases joined by single spaces.

    Deterministic: lowercase, strip punctuation, split on whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]

    ngrams: set[str] = set()
    for n in range(2, max_n + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            ngrams.add(phrase)

    return tokens, ngrams


def _extract_keywords(text: str, filter_short: bool = False) -> list[str]:
    """
    Extract keywords from a string using the same tokenization rules.

    Args:
        text: Input string to tokenize
        filter_short: If True, filter out tokens < 3 chars unless whitelisted
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]

    if filter_short:
        return [t for t in tokens if len(t) >= 3 or t in TECHNICAL_TERMS_WHITELIST]

    return tokens


def get_input_snippet(text: str, max_length: int = INPUT_SNIPPET_LENGTH) -> str:
    """Get truncated input for citation messages."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


# =============================================================================
# DOMAIN REGISTRY
# =============================================================================

@dataclass
class DomainEntry:
    """Single domain in the registry."""

    domain_id: str
    keywords: dict[str, float]  # keyword -> weight (0.0-1.0)
    negative_keywords: list[str] = field(default_factory=list)

    def score_input(self, tokens: list[str], ngrams: set[str]) -> float:
        """
        Score input against this domain's keywords.

        Uses exact matching for unigrams and phrase matching for
        multi-word keywords via the ngram set.

        Note: Cumulative scoring - scores both phrases and constituent unigrams
        if they appear in the keyword list. This provides a natural boost
        for phrase matches.
        """
        score = 0.0

        # 1. Penalize negative keywords first
        for neg in self.negative_keywords:
            hit = (neg in ngrams) if (" " in neg) else (neg in tokens)
            if hit:
                score -= NEGATIVE_KEYWORD_PENALTY

        # 2. Score positive keywords
        matched_keywords: set[str] = set()

        # Check all keywords against ngrams/tokens
        for kw, weight in self.keywords.items():
            is_phrase = " " in kw
            hit = False

            if is_phrase:
                if kw in ngrams:
                    hit = True
            else:
                if kw in tokens:
                    hit = True

            if hit and kw not in matched_keywords:
                score += weight
                matched_keywords.add(kw)

        return max(0.0, score)


# Built-in domain registry — can be extended per deployment
# Note: Whitelist filtering is NOT applied to these hardcoded keywords.
# Short tokens like "ux", "ai" in the registry are preserved.
DOMAIN_REGISTRY: list[DomainEntry] = [
    DomainEntry(
        domain_id="psychology",
        keywords={
            "psychology": 1.0, "behavior": 0.8, "cognitive": 0.9, "research": 0.6,
            "user": 0.5, "ux": 0.9, "experience": 0.5, "mental": 0.7,
            "emotions": 0.6, "perception": 0.6, "learning": 0.5, "memory": 0.5,
            "motivation": 0.6, "personality": 0.6, "bias": 0.7, "heuristic": 0.7,
            "usability": 0.8, "interview": 0.5, "participant": 0.7, "study": 0.5,
        },
        negative_keywords=["physics", "chemistry", "engineering"],
    ),
    DomainEntry(
        domain_id="technology",
        keywords={
            "technology": 1.0, "software": 0.9, "code": 0.8, "programming": 0.9,
            "computer": 0.7, "data": 0.6, "algorithm": 0.8, "system": 0.5,
            "app": 0.6, "web": 0.7, "design": 0.5, "tool": 0.5, "platform": 0.5,
            "api": 0.8, "database": 0.7, "cloud": 0.6, "ai": 0.7, "ml": 0.7,
            "prototype": 0.6, "figma": 0.7, "sketch": 0.6,
        },
        negative_keywords=["biology", "medicine"],
    ),
    DomainEntry(
        domain_id="business",
        keywords={
            "business": 1.0, "company": 0.7, "market": 0.7, "strategy": 0.8,
            "management": 0.7, "project": 0.6, "stakeholder": 0.8, "agile": 0.7,
            "deadline": 0.5, "budget": 0.6, "roi": 0.8, "kpi": 0.7, "revenue": 0.7,
            "customer": 0.6, "client": 0.6, "meeting": 0.4, "team": 0.5,
            "sprint": 0.6, "roadmap": 0.7,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="health",
        keywords={
            "health": 1.0, "medical": 0.9, "doctor": 0.8, "disease": 0.7,
            "symptom": 0.8, "treatment": 0.8, "diagnosis": 0.9, "medicine": 0.8,
            "therapy": 0.7, "wellness": 0.6, "exercise": 0.5, "nutrition": 0.6,
            "diet": 0.5, "mental health": 0.8, "anxiety": 0.7, "depression": 0.7,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="personal",
        keywords={
            "personal": 0.8, "family": 0.7, "relationship": 0.8, "friend": 0.6,
            "hobby": 0.6, "travel": 0.5, "home": 0.5, "life": 0.4, "feel": 0.5,
            "opinion": 0.4, "think": 0.3, "believe": 0.4, "prefer": 0.4,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="food",
        keywords={
            "food": 1.0, "cook": 0.9, "cooking": 0.9, "chef": 0.9, "kitchen": 0.8,
            "recipe": 0.9, "cuisine": 0.9, "restaurant": 0.8, "meal": 0.7,
            "ingredient": 0.8, "sauce": 0.8, "bake": 0.7, "baking": 0.7,
            "grill": 0.7, "roast": 0.7, "ferment": 0.7, "fermentation": 0.7,
            "pastry": 0.8, "butcher": 0.7, "butchery": 0.7, "flavor": 0.7,
            "flavour": 0.7, "culinary": 0.9, "gastronomy": 0.9, "dish": 0.6,
            "menu": 0.7, "plating": 0.7, "seasoning": 0.7, "spice": 0.6,
            "broth": 0.7, "stock": 0.5, "knife": 0.6, "oven": 0.6,
            "saut\u00e9": 0.7, "simmer": 0.7, "blanch": 0.7, "frozen": 0.4,
            "fresh": 0.4, "organic": 0.5, "farm": 0.5, "table": 0.3,
            # Classical sauces & techniques
            "bechamel": 0.9, "b\u00e9chamel": 0.9, "hollandaise": 0.9,
            "veloute": 0.9, "velout\u00e9": 0.9, "roux": 0.8, "reduction": 0.7,
            "emulsion": 0.7, "deglaze": 0.8, "whisk": 0.6, "fold": 0.5,
            "temper": 0.6, "braise": 0.8, "poach": 0.7, "stew": 0.7,
            # Ingredients
            "butter": 0.5, "cream": 0.5, "egg": 0.4, "flour": 0.5,
            "cheese": 0.5, "meat": 0.6, "steak": 0.7, "chicken": 0.6,
            "fish": 0.5, "vegetable": 0.5, "pasta": 0.7, "bread": 0.6,
            "dough": 0.7, "soup": 0.7, "salad": 0.6,
            # Taste & texture
            "umami": 0.8, "savory": 0.6, "caramelize": 0.8, "sear": 0.8,
        },
        negative_keywords=["computer", "software", "code"],
    ),
    DomainEntry(
        domain_id="science",
        keywords={
            "science": 1.0, "physics": 0.9, "chemistry": 0.9, "biology": 0.9,
            "experiment": 0.8, "hypothesis": 0.9, "theory": 0.7, "quantum": 0.9,
            "molecule": 0.8, "atom": 0.8, "cell": 0.6, "evolution": 0.8,
            "gravity": 0.8, "particle": 0.8, "electron": 0.8, "photon": 0.8,
            "equation": 0.7, "formula": 0.6, "laboratory": 0.7, "research": 0.5,
            "genome": 0.8, "dna": 0.8, "protein": 0.7, "entropy": 0.8,
            "thermodynamics": 0.9, "relativity": 0.9, "astronomy": 0.9,
            "telescope": 0.7, "planet": 0.7, "star": 0.5, "galaxy": 0.8,
            "neuroscience": 0.9, "ecology": 0.8, "species": 0.7,
        },
        negative_keywords=["business", "stakeholder"],
    ),
    DomainEntry(
        domain_id="arts",
        keywords={
            "art": 1.0, "music": 0.9, "painting": 0.9, "sculpture": 0.8,
            "theater": 0.8, "theatre": 0.8, "poetry": 0.8, "novel": 0.7,
            "literature": 0.9, "compose": 0.7, "composition": 0.7, "melody": 0.8,
            "harmony": 0.7, "rhythm": 0.7, "guitar": 0.7, "piano": 0.7,
            "orchestra": 0.8, "symphony": 0.8, "gallery": 0.7, "exhibition": 0.7,
            "canvas": 0.8, "sketch": 0.6, "creative": 0.5, "aesthetic": 0.7,
            "film": 0.7, "cinema": 0.8, "photography": 0.7, "dance": 0.7,
            "opera": 0.8, "ceramic": 0.7, "printmaking": 0.8,
        },
        negative_keywords=["database", "software"],
    ),
    DomainEntry(
        domain_id="education",
        keywords={
            "education": 1.0, "teaching": 0.9, "teacher": 0.8, "student": 0.7,
            "curriculum": 0.9, "pedagogy": 0.9, "classroom": 0.8, "school": 0.7,
            "university": 0.7, "lecture": 0.7, "homework": 0.7, "exam": 0.7,
            "assessment": 0.7, "grading": 0.7, "syllabus": 0.8, "tutoring": 0.8,
            "literacy": 0.8, "lesson": 0.7, "academic": 0.6, "scholarship": 0.7,
            "diploma": 0.7, "degree": 0.5, "mentor": 0.6,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="law",
        keywords={
            "law": 1.0, "legal": 0.9, "court": 0.8, "judge": 0.8, "lawyer": 0.9,
            "attorney": 0.9, "statute": 0.9, "regulation": 0.7, "contract": 0.8,
            "plaintiff": 0.9, "defendant": 0.9, "verdict": 0.9, "trial": 0.8,
            "litigation": 0.9, "jurisdiction": 0.8, "constitution": 0.8,
            "precedent": 0.8, "testimony": 0.8, "prosecution": 0.8,
            "compliance": 0.7, "liability": 0.8, "tort": 0.9, "patent": 0.8,
            "copyright": 0.7, "intellectual property": 0.8,
        },
        negative_keywords=[],
    ),
    DomainEntry(
        domain_id="sports",
        keywords={
            "sport": 1.0, "sports": 1.0, "athlete": 0.9, "training": 0.7,
            "coach": 0.8, "fitness": 0.7, "workout": 0.7, "competition": 0.7,
            "championship": 0.8, "tournament": 0.8, "match": 0.5, "game": 0.5,
            "team": 0.4, "score": 0.5, "goal": 0.4, "basketball": 0.8,
            "football": 0.8, "soccer": 0.8, "tennis": 0.8, "swimming": 0.7,
            "marathon": 0.8, "sprint": 0.6, "weightlifting": 0.8, "gym": 0.7,
            "endurance": 0.7, "cardio": 0.7, "muscle": 0.6, "protein": 0.4,
        },
        negative_keywords=["software", "code", "programming"],
    ),
    DomainEntry(
        domain_id="finance",
        keywords={
            "finance": 1.0, "investment": 0.9, "stock": 0.7, "bond": 0.8,
            "portfolio": 0.9, "dividend": 0.9, "interest rate": 0.9,
            "inflation": 0.8, "mortgage": 0.8, "banking": 0.8, "credit": 0.7,
            "debt": 0.7, "loan": 0.7, "tax": 0.7, "accounting": 0.8,
            "audit": 0.7, "hedge fund": 0.9, "mutual fund": 0.8,
            "cryptocurrency": 0.8, "blockchain": 0.7, "equity": 0.8,
            "valuation": 0.8, "asset": 0.7, "liability": 0.6, "capital": 0.6,
        },
        negative_keywords=[],
    ),
]


# =============================================================================
# DOMAIN DETECTION
# =============================================================================

# Priority constants for deterministic tie-breaking
_PRIORITY_REGISTRY = 0
_PRIORITY_PERSONA = 1  # persona wins ties


def detect_domain(
    user_input: str,
    persona_domains: list[dict] | None = None,
    ctx: Optional["TraceContext"] = None,
) -> tuple[str, float]:
    """
    Detect domain from user input using keyword scoring.

    Fallback behavior (all emitted as citations):
    - Empty input → "general"
    - Empty registry + no persona domains → "general"
    - No domain exceeds min_score → "general" (citation includes best score + input snippet)

    Tie-breaking: highest score wins; on tie, persona domains beat registry;
    on further tie, alphabetical domain_id.

    Args:
        user_input: The user's message
        persona_domains: Optional list of persona knowledge_domain objects/dicts
        ctx: TraceContext for citations

    Returns:
        (domain_id, score)
    """
    tokens, ngrams = tokenize_with_ngrams(user_input)
    input_snippet = get_input_snippet(user_input)

    if not tokens:
        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="domain_detection",
                effect=f"Empty input → fallback to '{DEFAULT_DOMAIN}'",
                weight=1.0,
            )
        return DEFAULT_DOMAIN, 0.0

    # ------------------------------------------------------------------
    # Score all candidates: (score, priority, domain_id)
    # ------------------------------------------------------------------
    scores: list[tuple[float, int, str]] = []

    for entry in DOMAIN_REGISTRY:
        score = entry.score_input(tokens, ngrams)
        scores.append((score, _PRIORITY_REGISTRY, entry.domain_id))

    if persona_domains:
        for pd in persona_domains:
            # Safe access using _get_field
            domain_name = (_get_field(pd, "domain") or "").strip().lower()
            if not domain_name:
                continue

            subdomains = _get_field(pd, "subdomains", [])

            # Build keywords from domain name + subdomains
            domain_keywords: dict[str, float] = {domain_name: 1.0}
            for sd in subdomains:
                # Use unified tokenizer to split "UI/UX" -> "ui", "ux"
                sd_keywords = _extract_keywords(sd, filter_short=False)

                # Check for phrasal subdomains vs single words
                if len(sd_keywords) > 1:
                    # Add exact phrase
                    phrase = " ".join(sd_keywords)
                    domain_keywords[phrase] = 0.8

                # Add individual tokens
                for word in sd_keywords:
                    # Apply whitelist filter here for dynamic keywords
                    if len(word) >= 3 or word in TECHNICAL_TERMS_WHITELIST:
                        domain_keywords[word] = 0.5

            persona_entry = DomainEntry(
                domain_id=domain_name,
                keywords=domain_keywords,
            )
            score = persona_entry.score_input(tokens, ngrams)
            scores.append((score, _PRIORITY_PERSONA, domain_name))

    if not scores:
        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="domain_detection",
                effect=f"No candidates from any source → fallback to '{DEFAULT_DOMAIN}'",
                weight=1.0,
            )
        return DEFAULT_DOMAIN, 0.0

    # Sort: Score desc, Priority desc (persona > registry), Domain asc (ABC)
    scores.sort(key=lambda x: (-x[0], -x[1], x[2]))
    best_score, _, best_domain = scores[0]

    # Check minimum score threshold
    if best_score < DOMAIN_MIN_SCORE:
        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="domain_detection",
                effect=(
                    f"No domain exceeded min_score={DOMAIN_MIN_SCORE:.2f} "
                    f"for input='{input_snippet}' "
                    f"(best={best_score:.2f}: '{best_domain}') "
                    f"→ fallback to '{DEFAULT_DOMAIN}'"
                ),
                weight=1.0,
            )
        return DEFAULT_DOMAIN, best_score

    # Success
    if ctx:
        ctx.add_basic_citation(
            source_type="rule",
            source_id="domain_detection",
            effect=(
                f"Matched domain='{best_domain}' "
                f"(score={best_score:.2f}) "
                f"for input='{input_snippet}'"
            ),
            weight=1.0,
        )

    return best_domain, best_score


# =============================================================================
# DOMAIN ADJACENCY (for competence computation)
# =============================================================================

# Decay factor for adjacency: adjacent domain proficiency is reduced by this
ADJACENCY_DECAY = 0.4


def compute_domain_adjacency(
    detected_domain: str,
    persona_domains: list[dict] | None = None,
    ctx: Optional["TraceContext"] = None,
) -> tuple[float, str | None]:
    """
    Compute how close the detected domain is to the persona's known domains.

    Uses keyword overlap between the detected domain's keyword set and
    each persona domain's keyword set. Returns the best adjacent domain's
    proficiency * decay factor.

    Args:
        detected_domain: The domain detected from user input
        persona_domains: List of persona knowledge_domain dicts with
                         'domain', 'proficiency', 'subdomains' keys
        ctx: TraceContext for citations

    Returns:
        (adjacency_score, nearest_domain_name) — score in [0, 1], or
        (0.0, None) if no adjacency found
    """
    if not persona_domains:
        return 0.0, None

    detected_lower = detected_domain.lower()

    # Direct match → not adjacency, handled elsewhere
    for pd in persona_domains:
        if (_get_field(pd, "domain") or "").strip().lower() == detected_lower:
            return 0.0, None

    # Build keyword set for detected domain from registry
    detected_keywords: set[str] = set()
    for entry in DOMAIN_REGISTRY:
        if entry.domain_id.lower() == detected_lower:
            detected_keywords = set(entry.keywords.keys())
            break

    # If detected domain not in registry, tokenize the domain name itself
    if not detected_keywords:
        detected_keywords = set(_extract_keywords(detected_domain, filter_short=True))

    if not detected_keywords:
        return 0.0, None

    # Score each persona domain by keyword overlap with detected domain
    best_score = 0.0
    best_domain: str | None = None

    for pd in persona_domains:
        domain_name = (_get_field(pd, "domain") or "").strip().lower()
        proficiency = float(_get_field(pd, "proficiency", 0.0))
        subdomains = _get_field(pd, "subdomains", [])

        # Build keyword set for this persona domain
        persona_keywords: set[str] = set()

        # From registry
        for entry in DOMAIN_REGISTRY:
            if entry.domain_id.lower() == domain_name:
                persona_keywords.update(entry.keywords.keys())
                break

        # From domain name + subdomains
        persona_keywords.update(_extract_keywords(domain_name, filter_short=True))
        for sd in subdomains:
            persona_keywords.update(_extract_keywords(sd, filter_short=True))

        if not persona_keywords:
            continue

        # Jaccard-like overlap
        overlap = len(detected_keywords & persona_keywords)
        union = len(detected_keywords | persona_keywords)
        if union == 0:
            continue

        similarity = overlap / union
        adjacency = proficiency * similarity * ADJACENCY_DECAY

        if adjacency > best_score:
            best_score = adjacency
            best_domain = domain_name

    if ctx and best_domain:
        ctx.add_basic_citation(
            source_type="rule",
            source_id="domain_adjacency",
            effect=(
                f"Nearest domain to '{detected_domain}' is '{best_domain}' "
                f"(adjacency={best_score:.3f})"
            ),
            weight=0.7,
        )

    return best_score, best_domain


# =============================================================================
# TOPIC RELEVANCE
# =============================================================================

def compute_topic_relevance(
    user_input: str,
    persona_domains: list[Any] | None = None,
    persona_goals: list[Any] | None = None,
    ctx: Optional["TraceContext"] = None,
    default_relevance: float = DEFAULT_TOPIC_RELEVANCE,
) -> float:
    """
    Compute topic relevance as overlap between input and persona interests.

    Relevance = (# overlapping keywords) / (# input tokens).

    Prevents double-counting by consuming tokens matched by phrases.

    Args:
        user_input: The user's message
        persona_domains: List of knowledge_domain objects/dicts
        persona_goals: List of goal objects/dicts
        ctx: TraceContext for citations
        default_relevance: Fallback when no interests defined

    Returns:
        Topic relevance score [0.0, 1.0]
    """
    tokens_list, ngrams = tokenize_with_ngrams(user_input)
    if not tokens_list:
        if ctx:
            ctx.add_basic_citation(
                source_type="state",
                source_id="topic_relevance",
                effect=f"Empty input → topic_relevance default {default_relevance:.2f}",
                weight=1.0,
            )
        return default_relevance

    # 1. Build Interest Sets
    interest_keywords: set[str] = set()
    interest_phrases: set[str] = set()

    if persona_domains:
        for pd in persona_domains:
            domain = (_get_field(pd, "domain") or "").strip().lower()
            if domain:
                domain_tokens = _extract_keywords(domain, filter_short=True)
                if len(domain_tokens) > 1:
                     interest_phrases.add(" ".join(domain_tokens))
                for t in domain_tokens:
                     interest_keywords.add(t)

            for sd in _get_field(pd, "subdomains", []):
                sd_tokens = _extract_keywords(sd, filter_short=False)
                if not sd_tokens:
                    continue

                if len(sd_tokens) > 1:
                    interest_phrases.add(" ".join(sd_tokens))

                # Add individual words with filter
                for word in sd_tokens:
                    if len(word) >= 3 or word in TECHNICAL_TERMS_WHITELIST:
                        interest_keywords.add(word)

    if persona_goals:
        for g in persona_goals:
            goal_text = (_get_field(g, "goal") or "")
            if not goal_text.strip():
                continue

            # Goals usually sentences -> extract just keywords with filter
            goal_tokens = _extract_keywords(goal_text, filter_short=True)
            for word in goal_tokens:
                interest_keywords.add(word)

    if not interest_keywords and not interest_phrases:
        if ctx:
            ctx.add_basic_citation(
                source_type="state",
                source_id="topic_relevance",
                effect=(
                    f"No persona interests/goals defined "
                    f"→ topic_relevance default {default_relevance:.2f}"
                ),
                weight=1.0,
            )
        return default_relevance

    # 2. Compute Coverage (Prevent double-counting)
    # We want to find how many *tokens* are covered by interests.
    # Phrases cover multiple tokens. Unigrams cover 1.

    covered_indices: set[int] = set()

    # A. Check phrases first (longest first ideally, but ngrams set is flat)
    # Since we don't have indices in `ngrams`, we must reconstruct detection
    # to associate matches with token indices.

    # Iterate through generated ngrams again to map to indices
    # MAX_NGRAM_SIZE is small (3), so this is cheap.
    for n in range(MAX_NGRAM_SIZE, 1, -1):  # 3, then 2
        for i in range(len(tokens_list) - n + 1):
            # If any part of this span is already covered, skip?
            # Or allow overlapping phrases? simpler: strict consumption.
            is_overlap = any(j in covered_indices for j in range(i, i + n))
            if is_overlap:
                continue

            phrase = " ".join(tokens_list[i : i + n])
            if phrase in interest_phrases:
                # MARK CONSUMED
                for j in range(i, i + n):
                    covered_indices.add(j)

    # B. Check unigrams for remaining (unconsumed) tokens
    for i, token in enumerate(tokens_list):
        if i in covered_indices:
            continue
        if len(token) < 3 and token not in TECHNICAL_TERMS_WHITELIST:
            continue
        if token in interest_keywords:
            covered_indices.add(i)

    # 3. Compute relevance = covered / total tokens
    total_tokens = len(tokens_list)
    relevance = len(covered_indices) / max(1, total_tokens)
    relevance = max(0.0, min(1.0, relevance))

    if ctx:
        ctx.add_basic_citation(
            source_type="state",
            source_id="topic_relevance",
            effect=f"Topic relevance={relevance:.2f} (covered={len(covered_indices)}/{total_tokens} tokens)",
            weight=1.0,
        )

    return relevance


# =============================================================================
# EVIDENCE STRENGTH
# =============================================================================

# Strong challenges — always trigger high evidence regardless of position
STRONG_CHALLENGE_PHRASES = [
    "that's wrong",
    "you're wrong",
    "incorrect",
    "actually no",
    "i disagree",
    "that's not right",
    "you're mistaken",
]

# Moderate challenges — only match at sentence start
MODERATE_CHALLENGE_PHRASES = [
    "actually",
    # "but",  <-- REMOVED per review (too noisy)
    "however",
    "not really",
    "i don't think",
    "i'm not sure that",
    "are you sure",
]

# Sentence-start pattern: beginning of string or after sentence-ending punctuation
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?]\s+)")


def _is_sentence_start(text: str, phrase: str) -> bool:
    """Check if phrase appears at a sentence boundary, not mid-sentence."""
    for match in _SENTENCE_START_RE.finditer(text):
        pos = match.end()
        if text[pos:].startswith(phrase):
            return True
    return text.startswith(phrase)


def detect_evidence_strength(
    user_input: str,
    ctx: Optional["TraceContext"] = None,
) -> float:
    """
    Detect if user input contains evidence of disagreement/challenge.

    Triggers stance reconsideration when cached stance exists.

    Strong challenges (position-independent): 0.8
    Moderate challenges (sentence-start only): 0.6
    Neutral: 0.2

    Args:
        user_input: The user's message
        ctx: TraceContext for citations (traces challenge detection)

    Returns:
        Evidence strength [0.0, 1.0]
    """
    text = user_input.lower().strip()

    if not text:
        if ctx:
            ctx.add_basic_citation(
                source_type="state",
                source_id="evidence_strength",
                effect=f"Empty input → evidence_strength={DEFAULT_EVIDENCE_STRENGTH:.2f}",
                weight=1.0,
            )
        return DEFAULT_EVIDENCE_STRENGTH

    # Check strong challenges (match anywhere)
    for phrase in STRONG_CHALLENGE_PHRASES:
        if phrase in text:
            if ctx:
                ctx.add_basic_citation(
                    source_type="state",
                    source_id="evidence_strength",
                    effect=(
                        f"Strong challenge detected: '{phrase}' "
                        f"→ evidence_strength={STRONG_CHALLENGE_STRENGTH:.2f}"
                    ),
                    weight=1.0,
                )
            return STRONG_CHALLENGE_STRENGTH

    # moderate challenges only at sentence start
    for phrase in MODERATE_CHALLENGE_PHRASES:
        if phrase in text and _is_sentence_start(text, phrase):
            if ctx:
                ctx.add_basic_citation(
                    source_type="state",
                    source_id="evidence_strength",
                    effect=(
                        f"Moderate challenge detected: '{phrase}' (sentence-start) "
                        f"→ evidence_strength={MODERATE_CHALLENGE_STRENGTH:.2f}"
                    ),
                    weight=1.0,
                )
            return MODERATE_CHALLENGE_STRENGTH

    # No challenge detected
    if ctx:
        ctx.add_basic_citation(
            source_type="state",
            source_id="evidence_strength",
            effect=f"No challenge detected → evidence_strength={DEFAULT_EVIDENCE_STRENGTH:.2f}",
            weight=1.0,
        )
    return DEFAULT_EVIDENCE_STRENGTH


# =============================================================================
# INTENT GENERATION
# =============================================================================

def _norm_action(action: str) -> str:
    """
    Normalize uncertainty action string to canonical form.

    Handles common enum value variations so the intent map doesn't
    silently fall through to the fallback.
    """
    a = (action or "").strip().upper()
    return {
        "ASK_CLARIFYING": "ASK_CLARIFICATION",
        "ASK_CLARIFY": "ASK_CLARIFICATION",
    }.get(a, a)


# Intent map: (user_intent, normalized_uncertainty_action) → intent string
_INTENT_MAP: dict[tuple[str, str], str] = {
    # User is asking
    ("ask", "ANSWER"): "Provide direct answer with domain knowledge",
    ("ask", "HEDGE"): "Provide tentative answer with uncertainty markers",
    ("ask", "ASK_CLARIFICATION"): "Request more context before answering",
    ("ask", "REFUSE"): "Decline to answer due to constraint",
    ("ask", "LOOKUP"): "Indicate need to verify before answering",
    # User is requesting action
    ("request", "ANSWER"): "Acknowledge request and provide guidance",
    ("request", "HEDGE"): "Acknowledge request with conditional commitment",
    ("request", "ASK_CLARIFICATION"): "Clarify requirements before committing",
    ("request", "REFUSE"): "Politely decline request with explanation",
    ("request", "LOOKUP"): "Acknowledge request pending verification",
    # User is challenging
    ("challenge", "ANSWER"): "Address challenge with evidence and reasoning",
    ("challenge", "HEDGE"): "Acknowledge validity of challenge, reconsider position",
    ("challenge", "ASK_CLARIFICATION"): "Seek to understand specific objection",
    ("challenge", "REFUSE"): "Maintain position or redirect topic",
    ("challenge", "LOOKUP"): "Acknowledge need to verify claim",
    # User is sharing
    ("share", "ANSWER"): "Acknowledge and relate to shared experience",
    ("share", "HEDGE"): "Express tentative connection to shared content",
    ("share", "ASK_CLARIFICATION"): "Ask follow-up about shared experience",
    ("share", "REFUSE"): "Acknowledge but redirect to safer topic",
    ("share", "LOOKUP"): "Acknowledge shared content",
    # User is clarifying
    ("clarify", "ANSWER"): "Incorporate clarification and proceed with answer",
    ("clarify", "HEDGE"): "Acknowledge clarification, proceed cautiously",
    ("clarify", "ASK_CLARIFICATION"): "Request additional clarification",
    ("clarify", "REFUSE"): "Clarification does not change constraint",
    ("clarify", "LOOKUP"): "Incorporate clarification, verify updated context",
}


def generate_intent_string(
    user_intent: str,
    conversation_goal: str,
    uncertainty_action: str,
    needs_clarification: bool = False,
    ctx: Optional["TraceContext"] = None,
) -> str:
    """
    Generate meaningful intent string for IR.

    Intent describes what the persona plans to do, informed by the
    user's intent type and the resolved uncertainty action.

    Args:
        user_intent: From intent analyzer (ask/request/challenge/share/clarify)
        conversation_goal: Current conversation goal value
        uncertainty_action: The resolved uncertainty action value
        needs_clarification: Whether clarification is needed first
        ctx: TraceContext for citations (traces intent selection)

    Returns:
        Canonical intent description string
    """
    # clarification override includes goal context
    if needs_clarification:
        intent = f"Ask clarifying question before {conversation_goal.lower().replace('_', ' ')}"
        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="intent_generation",
                effect=f"Intent: needs_clarification=True, goal='{conversation_goal}' → '{intent}'",
                weight=1.0,
            )
        return intent

    norm = _norm_action(uncertainty_action)
    key = (user_intent.lower(), norm)
    mapped_intent = _INTENT_MAP.get(key)

    if mapped_intent:
        if ctx:
            ctx.add_basic_citation(
                source_type="rule",
                source_id="intent_generation",
                effect=(
                    f"Intent: user_intent='{user_intent}' + "
                    f"uncertainty='{norm}' → '{mapped_intent}'"
                ),
                weight=1.0,
            )
        return mapped_intent

    # Fallback for unknown combinations
    fallback = f"Respond to {user_intent} with {uncertainty_action.lower()} approach"
    if ctx:
        ctx.add_basic_citation(
            source_type="rule",
            source_id="intent_generation",
            effect=(
                f"Intent fallback: user_intent='{user_intent}' + "
                f"uncertainty='{norm}' → '{fallback}'"
            ),
            weight=1.0,
        )
    return fallback
