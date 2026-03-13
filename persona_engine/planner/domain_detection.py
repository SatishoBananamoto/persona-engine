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


# Domain registry imported from external module for easy customization.
# Backward-compatible: DOMAIN_REGISTRY is still available at this module level.
from persona_engine.planner.domain_registry import DOMAIN_REGISTRY  # noqa: E402


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
