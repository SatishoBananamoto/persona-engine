"""
Negation-Aware Marker Detection

Shared utility for detecting keyword markers while respecting negation.
Used by both emotional_appraisal.py and bias_simulator.py.

Negation window: if a negation word ("not", "don't", "never", etc.) appears
within 3 tokens before a marker, that marker is considered negated and not
counted.

Examples:
    count_unnegated("I'm not excited", ENTHUSIASM) → 0
    count_unnegated("I'm excited", ENTHUSIASM)     → 1
    count_unnegated("no problem at all", NEGATIVE)  → 0
    count_unnegated("serious problem", NEGATIVE)    → 1
"""

import re
from collections.abc import Set

NEGATION_WORDS: frozenset[str] = frozenset([
    "not", "no", "never", "nothing", "none",
    "hardly", "barely", "neither",
    "don't", "doesn't", "isn't", "wasn't", "weren't",
    "won't", "can't", "couldn't", "shouldn't", "wouldn't",
])

_TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b")


def count_unnegated_markers(text: str, markers: Set[str]) -> int:
    """Count markers in *text* that are NOT preceded by a negation word.

    Handles both single-word and multi-word markers.  Negation is detected
    within a 3-token look-back window.

    Args:
        text: Raw user input.
        markers: Set of marker words/phrases to detect.

    Returns:
        Number of unnegated marker hits.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    count = 0

    for i, token in enumerate(tokens):
        matched = False
        for marker in markers:
            marker_tokens = marker.split()
            if len(marker_tokens) == 1:
                if token == marker:
                    matched = True
                    break
            else:
                end = i + len(marker_tokens)
                if end <= len(tokens) and tokens[i:end] == marker_tokens:
                    matched = True
                    break

        if not matched:
            continue

        # Check preceding 1-3 tokens for negation
        window_start = max(0, i - 3)
        negated = any(
            tokens[j] in NEGATION_WORDS for j in range(window_start, i)
        )

        if not negated:
            count += 1

    return count
