"""
Cognitive style prior engine — derives cognitive style from Big Five + occupation.

This module provides the mapping logic referenced by the architecture
as priors/cognitive.py. The actual derivation with residuals happens
in gap_filler.py; this module provides the base computation and
occupation-specific cognitive profiles.

Sources:
- DeYoung (2015): Openness/Intellect as cognitive exploration
- Research on Need for Closure and Big Five correlations
"""

from __future__ import annotations

from typing import Any

# Occupation-specific cognitive style adjustments
# These shift the base derivation from Big Five
OCCUPATION_COGNITIVE_PROFILES: dict[str, dict[str, float]] = {
    "software engineer": {"analytical_intuitive": 0.08, "systematic_heuristic": 0.08, "cognitive_complexity": 0.05},
    "engineer": {"analytical_intuitive": 0.05, "systematic_heuristic": 0.08},
    "data scientist": {"analytical_intuitive": 0.10, "systematic_heuristic": 0.05, "cognitive_complexity": 0.08},
    "researcher": {"analytical_intuitive": 0.08, "cognitive_complexity": 0.10},
    "scientist": {"analytical_intuitive": 0.10, "systematic_heuristic": 0.05, "cognitive_complexity": 0.08},
    "artist": {"analytical_intuitive": -0.08, "systematic_heuristic": -0.08, "cognitive_complexity": 0.05},
    "musician": {"analytical_intuitive": -0.05, "systematic_heuristic": -0.05},
    "entrepreneur": {"risk_tolerance": 0.10, "need_for_closure": -0.05},
    "lawyer": {"analytical_intuitive": 0.08, "systematic_heuristic": 0.05, "need_for_closure": 0.05},
    "accountant": {"systematic_heuristic": 0.10, "need_for_closure": 0.05, "risk_tolerance": -0.08},
    "nurse": {"systematic_heuristic": 0.03},
    "teacher": {"cognitive_complexity": 0.03},
    "chef": {"risk_tolerance": 0.03, "systematic_heuristic": -0.03},
    "journalist": {"analytical_intuitive": 0.03, "risk_tolerance": 0.03},
    "consultant": {"analytical_intuitive": 0.05, "cognitive_complexity": 0.05},
    "pilot": {"systematic_heuristic": 0.08, "need_for_closure": 0.05, "risk_tolerance": 0.03},
    "police officer": {"systematic_heuristic": 0.05, "need_for_closure": 0.05},
    "social worker": {"cognitive_complexity": 0.05, "analytical_intuitive": -0.03},
}


def get_occupation_cognitive_shifts(occupation: str | None) -> dict[str, float]:
    """Get occupation-specific cognitive style shifts.

    Returns shifts to be added to the Big Five-derived base values.
    """
    if not occupation:
        return {}

    occ_lower = occupation.lower().strip()

    # Exact match first
    if occ_lower in OCCUPATION_COGNITIVE_PROFILES:
        return dict(OCCUPATION_COGNITIVE_PROFILES[occ_lower])

    # Partial match
    for key, profile in OCCUPATION_COGNITIVE_PROFILES.items():
        if key in occ_lower:
            return dict(profile)

    return {}
