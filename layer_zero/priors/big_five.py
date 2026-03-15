"""
Big Five prior engine — maps demographics to trait distributions.

Sources: occupation (RIASEC), age trajectories, gender differences, culture baselines.
Each source shifts the distribution mean. Standard deviation remains at baseline (0.15)
unless the user provides an explicit override (tiny SD = nearly fixed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from layer_zero.models import BIG_FIVE_TRAITS, MintRequest, TraitPrior

# =============================================================================
# Load mapping data
# =============================================================================

_DATA_DIR = Path(__file__).parent / "data"


def _load_json(filename: str) -> dict:
    with open(_DATA_DIR / filename) as f:
        return json.load(f)


_OCCUPATION_DATA: dict[str, Any] = _load_json("occupation_traits.json")
_AGE_DATA: dict[str, Any] = _load_json("age_trajectories.json")
_CULTURE_DATA: dict[str, Any] = _load_json("culture_baselines.json")

# Strip metadata keys
_OCCUPATIONS = {k: v for k, v in _OCCUPATION_DATA.items() if not k.startswith("_")}
_AGE_BRACKETS = _AGE_DATA["brackets"]
_CULTURES = {k: v for k, v in _CULTURE_DATA.items() if not k.startswith("_")}

# Population baseline
BASELINE_MEAN = 0.5
BASELINE_SD = 0.15
OVERRIDE_SD = 0.02  # tiny SD for user-specified values


# =============================================================================
# Normalization
# =============================================================================

def normalize_occupation(occupation: str) -> str | None:
    """Normalize occupation string to match mapping table keys."""
    lower = occupation.lower().strip()
    if lower in _OCCUPATIONS:
        return lower
    # Try partial match (e.g., "senior software engineer" → "software engineer")
    for key in sorted(_OCCUPATIONS.keys(), key=len, reverse=True):
        if key in lower:
            return key
    return None


def infer_culture_region(location: str) -> str | None:
    """Infer culture region from location string. Uses word-boundary matching."""
    import re
    lower = location.lower()

    # Mapping: (region_name, markers). Word-boundary matching prevents "us" in "Houston".
    _REGION_MARKERS: list[tuple[str, list[str]]] = [
        ("western", [
            "us", "usa", "united states", "america", "canada", "uk", "united kingdom",
            "england", "france", "germany", "australia", "new zealand", "netherlands",
            "sweden", "norway", "denmark", "finland", "ireland", "spain", "italy",
            "portugal", "belgium", "austria", "switzerland",
        ]),
        ("east_asian", [
            "china", "japan", "korea", "taiwan", "hong kong", "singapore",
            "tokyo", "beijing", "shanghai", "seoul",
        ]),
        ("south_asian", [
            "india", "pakistan", "bangladesh", "sri lanka", "nepal",
            "mumbai", "delhi", "bangalore", "kolkata",
        ]),
        ("latin_american", [
            "mexico", "brazil", "argentina", "colombia", "chile", "peru",
            "venezuela", "ecuador", "sao paulo", "buenos aires",
        ]),
        ("middle_eastern", [
            "saudi", "emirates", "uae", "dubai", "qatar", "iran", "iraq",
            "egypt", "turkey", "israel", "jordan", "lebanon",
        ]),
        ("sub_saharan_african", [
            "nigeria", "kenya", "south africa", "ethiopia", "ghana", "tanzania",
            "uganda", "rwanda", "nairobi", "lagos", "johannesburg",
        ]),
    ]

    # Short markers that need exact or delimiter-bounded matching
    _SHORT_EXACT = {"us", "uk"}

    for region, markers in _REGION_MARKERS:
        for marker in markers:
            if marker in _SHORT_EXACT:
                # Require delimiter: "US" at end, ", US", "US," etc.
                if re.search(r'(?:^|[,\s])' + re.escape(marker) + r'(?:$|[,\s.])', lower):
                    return region
            elif re.search(r'\b' + re.escape(marker) + r'\b', lower):
                return region

    return None


# =============================================================================
# Prior computation
# =============================================================================

def compute_big_five_prior(request: MintRequest) -> dict[str, TraitPrior]:
    """Compute Big Five trait priors from demographics.

    Returns a dict of {trait_name: TraitPrior} with mean, std_dev, and provenance.

    Priority: explicit overrides > occupation > age > gender > culture > baseline.
    Each source shifts the mean; they stack additively.
    """
    priors: dict[str, TraitPrior] = {
        trait: TraitPrior(mean=BASELINE_MEAN, std_dev=BASELINE_SD, source="default", mapping_strength=0.3)
        for trait in BIG_FIVE_TRAITS
    }

    # --- Occupation (strongest signal, r = 0.19-0.48) ---
    if request.occupation:
        occ_key = normalize_occupation(request.occupation)
        if occ_key and occ_key in _OCCUPATIONS:
            shifts = _OCCUPATIONS[occ_key]["shifts"]
            for trait in BIG_FIVE_TRAITS:
                delta = shifts.get(trait, 0.0)
                if abs(delta) > 0.001:
                    priors[trait] = TraitPrior(
                        mean=priors[trait].mean + delta,
                        std_dev=BASELINE_SD,
                        source="occupation",
                        mapping_strength=0.6,
                    )

    # --- Age (moderate signal) ---
    if request.age:
        age_shifts = _get_age_shifts(request.age)
        for trait in BIG_FIVE_TRAITS:
            delta = age_shifts.get(trait, 0.0)
            if abs(delta) > 0.001:
                priors[trait] = TraitPrior(
                    mean=priors[trait].mean + delta,
                    std_dev=priors[trait].std_dev,
                    source=priors[trait].source if priors[trait].source != "default" else "age",
                    mapping_strength=max(priors[trait].mapping_strength, 0.4),
                )

    # --- Culture region (moderate-strong for values, moderate for traits) ---
    culture = request.culture_region
    if not culture and request.location:
        culture = infer_culture_region(request.location)

    if culture and culture in _CULTURES:
        culture_data = _CULTURES[culture]
        shifts = culture_data.get("big_five_shifts", {})
        culture_confidence = culture_data.get("confidence", 0.5)
        for trait in BIG_FIVE_TRAITS:
            delta = shifts.get(trait, 0.0)
            if abs(delta) > 0.001:
                priors[trait] = TraitPrior(
                    mean=priors[trait].mean + delta,
                    std_dev=priors[trait].std_dev,
                    source=priors[trait].source if priors[trait].source != "default" else "culture",
                    mapping_strength=max(priors[trait].mapping_strength, culture_confidence * 0.6),
                )

    # --- Trait hint adjectives (deduplicate: only strongest hint per trait) ---
    from layer_zero.parser.text_parser import TRAIT_ADJECTIVES
    best_hint_per_trait: dict[str, float] = {}
    for hint in request.trait_hints:
        if hint in TRAIT_ADJECTIVES:
            target_trait, delta = TRAIT_ADJECTIVES[hint]
            if target_trait not in best_hint_per_trait or abs(delta) > abs(best_hint_per_trait[target_trait]):
                best_hint_per_trait[target_trait] = delta

    for target_trait, delta in best_hint_per_trait.items():
        priors[target_trait] = TraitPrior(
            mean=priors[target_trait].mean + delta,
            std_dev=priors[target_trait].std_dev,
            source="trait_hint",
            mapping_strength=max(priors[target_trait].mapping_strength, 0.5),
        )

    # --- Cap total shift from baseline (prevent extreme means) ---
    MAX_TOTAL_SHIFT = 0.25
    for trait in BIG_FIVE_TRAITS:
        if priors[trait].source != "override":
            capped_mean = max(BASELINE_MEAN - MAX_TOTAL_SHIFT, min(BASELINE_MEAN + MAX_TOTAL_SHIFT, priors[trait].mean))
            priors[trait] = TraitPrior(
                mean=capped_mean,
                std_dev=priors[trait].std_dev,
                source=priors[trait].source,
                mapping_strength=priors[trait].mapping_strength,
            )

    # --- Explicit overrides (REPLACE, not shift — these are user-specified) ---
    for trait, value in request.big_five_overrides.items():
        if trait in BIG_FIVE_TRAITS:
            priors[trait] = TraitPrior(
                mean=value,
                std_dev=OVERRIDE_SD,
                source="override",
                mapping_strength=1.0,
            )

    return priors


def _get_age_shifts(age: int) -> dict[str, float]:
    """Get trait shifts for a given age from age bracket data."""
    for bracket in _AGE_BRACKETS:
        if bracket["age_min"] <= age <= bracket["age_max"]:
            return bracket["shifts"]
    # Fallback: use last bracket for very old ages
    return _AGE_BRACKETS[-1]["shifts"]


def get_culture_value_shifts(culture_region: str) -> dict[str, float]:
    """Get Schwartz value shifts for a culture region (used by values prior engine)."""
    if culture_region in _CULTURES:
        return _CULTURES[culture_region].get("value_shifts", {})
    return {}


def get_culture_confidence(culture_region: str) -> float:
    """Get confidence level for a culture region's data quality."""
    if culture_region in _CULTURES:
        return _CULTURES[culture_region].get("confidence", 0.3)
    return 0.3
