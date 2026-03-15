"""
Gap Filler — derives remaining persona-layer fields from parent traits.

Each derived field: derived_value = f(parent_traits) + calibrated_residual

Residuals prevent cascade collapse by ensuring downstream fields retain
variance even when parent traits are similar. Related fields share latent
residuals so they vary together but still differ from parents.

This module fills: cognitive style, communication prefs, goals, social roles,
initial state, decision tendencies, knowledge domains, name, education.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from layer_zero.models import (
    BIG_FIVE_TRAITS,
    SCHWARTZ_VALUES,
    FieldProvenance,
    MintRequest,
    TraitPrior,
)

# =============================================================================
# Name pools (diverse, multi-cultural)
# =============================================================================

FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Jamie", "Quinn",
    "Avery", "Cameron", "Drew", "Emery", "Finley", "Harper", "Kai", "Logan",
    "Noor", "Priya", "Ravi", "Sana", "Yuki", "Wei", "Chen", "Aisha",
    "Omar", "Fatima", "Kenji", "Mei", "Lucia", "Carlos", "Elena", "Marco",
    "Ingrid", "Lars", "Olga", "Dmitri", "Amara", "Kofi", "Nia", "Tariq",
    "Liam", "Sophia", "Ethan", "Olivia", "Noah", "Isabella", "Mateo", "Zara",
    "Hana", "Raj", "Min", "Suki", "Adaeze", "Kwame", "Esperanza", "Viktor",
]


# =============================================================================
# Derivation functions with residuals
# =============================================================================

def fill_gaps(
    big_five: dict[str, float],
    values: dict[str, float],
    request: MintRequest,
    big_five_priors: dict[str, TraitPrior],
    seed: int = 42,
    persona_index: int = 0,
) -> dict[str, Any]:
    """Fill all unspecified persona-layer fields from parent traits + residuals.

    Args:
        big_five: Sampled Big Five traits {trait: float}.
        values: Sampled Schwartz values {value: float}.
        request: Original MintRequest (for overrides and explicit inputs).
        big_five_priors: Original priors (for provenance tracking).
        seed: Base seed for residual generation.
        persona_index: Index within batch (for name/ID variation).

    Returns:
        Dict with all persona-layer fields populated. Provenance metadata
        attached under '_provenance' key.
    """
    rng = np.random.default_rng(seed + persona_index * 1000)
    provenance: dict[str, FieldProvenance] = {}
    O = big_five["openness"]
    C = big_five["conscientiousness"]
    E = big_five["extraversion"]
    A = big_five["agreeableness"]
    N = big_five["neuroticism"]

    result: dict[str, Any] = {}

    # --- Name ---
    if request.name:
        result["name"] = request.name
        provenance["identity.name"] = FieldProvenance(
            value=request.name, source="explicit", confidence=0.95,
        )
    else:
        name = FIRST_NAMES[(seed + persona_index) % len(FIRST_NAMES)]
        result["name"] = name
        provenance["identity.name"] = FieldProvenance(
            value=name, source="template", confidence=0.3,
        )

    # --- Education ---
    if request.education:
        result["education"] = request.education
    else:
        result["education"] = _infer_education(request.occupation)

    # --- Cognitive Style (derived from Big Five + residual SD=0.08) ---
    # Shared latent for analytical ↔ systematic
    shared_cognitive = rng.normal(0, 0.04)

    cognitive = {}
    cognitive["analytical_intuitive"] = _derive_with_residual(
        base=0.3 + O * 0.4,  # O drives analytical thinking
        override=request.cognitive_overrides.get("analytical_intuitive"),
        rng=rng, sd=0.08, shared=shared_cognitive,
    )
    cognitive["systematic_heuristic"] = _derive_with_residual(
        base=0.3 + C * 0.4,  # C drives systematic approach
        override=request.cognitive_overrides.get("systematic_heuristic"),
        rng=rng, sd=0.08, shared=shared_cognitive,
    )
    cognitive["risk_tolerance"] = _derive_with_residual(
        base=0.3 + O * 0.2 + E * 0.15 - N * 0.15,
        override=request.cognitive_overrides.get("risk_tolerance"),
        rng=rng, sd=0.08,
    )
    cognitive["need_for_closure"] = _derive_with_residual(
        base=0.3 + C * 0.3 - O * 0.15 + N * 0.1,
        override=request.cognitive_overrides.get("need_for_closure"),
        rng=rng, sd=0.08,
    )
    cognitive["cognitive_complexity"] = _derive_with_residual(
        base=0.3 + O * 0.35 + C * 0.1,
        override=request.cognitive_overrides.get("cognitive_complexity"),
        rng=rng, sd=0.08,
    )
    result["cognitive_style"] = cognitive

    # Provenance with accurate parent fields per derivation
    _cog_parents = {
        "analytical_intuitive": ("psychology.big_five.openness",),
        "systematic_heuristic": ("psychology.big_five.conscientiousness",),
        "risk_tolerance": ("psychology.big_five.openness", "psychology.big_five.extraversion", "psychology.big_five.neuroticism"),
        "need_for_closure": ("psychology.big_five.conscientiousness", "psychology.big_five.openness", "psychology.big_five.neuroticism"),
        "cognitive_complexity": ("psychology.big_five.openness", "psychology.big_five.conscientiousness"),
    }
    for field_name, val in cognitive.items():
        provenance[f"psychology.cognitive_style.{field_name}"] = FieldProvenance(
            value=val, source="derived", mapping_strength=0.5,
            inferential_depth=1,
            confidence=FieldProvenance.compute_confidence("derived", 0.5, 1),
            parent_fields=_cog_parents.get(field_name, ()),
        )

    # --- Communication Preferences (derived from Big Five + residual SD=0.08) ---
    # Shared latent for formality ↔ directness
    shared_comm = rng.normal(0, 0.04)

    communication = {}
    communication["verbosity"] = _derive_with_residual(
        base=0.3 + C * 0.2 + E * 0.15,
        override=request.communication_overrides.get("verbosity"),
        rng=rng, sd=0.08,
    )
    communication["formality"] = _derive_with_residual(
        base=0.3 + C * 0.3 - E * 0.1,
        override=request.communication_overrides.get("formality"),
        rng=rng, sd=0.08, shared=shared_comm,
    )
    communication["directness"] = _derive_with_residual(
        base=0.5 + E * 0.15 - A * 0.2,
        override=request.communication_overrides.get("directness"),
        rng=rng, sd=0.08, shared=-shared_comm * 0.5,  # inversely related to formality
    )
    communication["emotional_expressiveness"] = _derive_with_residual(
        base=0.3 + E * 0.3 + N * 0.1 - C * 0.05,
        override=request.communication_overrides.get("emotional_expressiveness"),
        rng=rng, sd=0.08,
    )
    result["communication"] = communication

    _comm_parents = {
        "verbosity": ("psychology.big_five.conscientiousness", "psychology.big_five.extraversion"),
        "formality": ("psychology.big_five.conscientiousness", "psychology.big_five.extraversion"),
        "directness": ("psychology.big_five.extraversion", "psychology.big_five.agreeableness"),
        "emotional_expressiveness": ("psychology.big_five.extraversion", "psychology.big_five.neuroticism"),
    }
    for field_name, val in communication.items():
        provenance[f"psychology.communication.{field_name}"] = FieldProvenance(
            value=val, source="derived", mapping_strength=0.5,
            inferential_depth=1,
            confidence=FieldProvenance.compute_confidence("derived", 0.5, 1),
            parent_fields=_comm_parents.get(field_name, ()),
        )

    # --- Knowledge Domains (from occupation, proficiency 0.4-0.6 = familiarity) ---
    if request.domains:
        result["knowledge_domains"] = request.domains
    else:
        result["knowledge_domains"] = _infer_domains(request.occupation, request.industry)

    # --- Goals (from occupation + top values + residual via shuffled value ranking) ---
    if request.goals:
        result["goals"] = request.goals
    else:
        result["goals"] = _infer_goals(request.occupation, values, rng)

    # --- Social Roles (template + communication prefs + residual) ---
    result["social_roles"] = _build_social_roles(communication, rng)

    # --- Initial State (from Big Five + residual SD=0.05) ---
    shared_mood = rng.normal(0, 0.03)
    initial_state = {
        "mood_valence": _derive_with_residual(
            base=0.1 + E * 0.1 - N * 0.1, rng=rng, sd=0.05, shared=shared_mood,
            clamp_lo=-1.0, clamp_hi=1.0,
        ),
        "mood_arousal": _derive_with_residual(
            base=0.3 + E * 0.2, rng=rng, sd=0.05, shared=shared_mood,
        ),
        "fatigue": 0.3,
        "stress": _derive_with_residual(
            base=0.2 + N * 0.2, rng=rng, sd=0.05,
        ),
        "engagement": 0.6,
    }
    result["initial_state"] = initial_state

    # --- Decision Tendencies (renamed from "biases", low confidence) ---
    result["decision_tendencies"] = _build_decision_tendencies(big_five, values, rng)

    # --- Privacy Sensitivity ---
    result["privacy_sensitivity"] = _derive_with_residual(
        base=0.3 + N * 0.2 + (1 - A) * 0.15 + (1 - E) * 0.1,
        rng=rng, sd=0.05,
    )

    # --- Provenance for remaining fields ---
    provenance["identity.education"] = FieldProvenance(
        value=result["education"],
        source="explicit" if request.education else "template",
        confidence=0.95 if request.education else 0.4,
    )
    provenance["goals"] = FieldProvenance(
        value=result["goals"],
        source="explicit" if request.goals else "derived",
        confidence=0.95 if request.goals else FieldProvenance.compute_confidence("derived", 0.4, 1),
        parent_fields=("psychology.values",) if not request.goals else (),
    )
    provenance["knowledge_domains"] = FieldProvenance(
        value=result["knowledge_domains"],
        source="explicit" if request.domains else "template",
        confidence=0.95 if request.domains else 0.4,
        parent_fields=("identity.occupation",) if not request.domains else (),
    )
    provenance["privacy_sensitivity"] = FieldProvenance(
        value=result["privacy_sensitivity"],
        source="derived", mapping_strength=0.4, inferential_depth=1,
        confidence=FieldProvenance.compute_confidence("derived", 0.4, 1),
        parent_fields=("psychology.big_five.neuroticism", "psychology.big_five.agreeableness", "psychology.big_five.extraversion"),
    )
    for role_name in result["social_roles"]:
        provenance[f"social_roles.{role_name}"] = FieldProvenance(
            value=result["social_roles"][role_name],
            source="derived", mapping_strength=0.4, inferential_depth=2,
            confidence=FieldProvenance.compute_confidence("derived", 0.4, 2),
            parent_fields=("psychology.communication",),
        )
    for state_field in ("mood_valence", "mood_arousal", "stress"):
        provenance[f"initial_state.{state_field}"] = FieldProvenance(
            value=result["initial_state"][state_field],
            source="derived", mapping_strength=0.4, inferential_depth=1,
            confidence=FieldProvenance.compute_confidence("derived", 0.4, 1),
            parent_fields=("psychology.big_five",),
        )
    for dt in result.get("decision_tendencies", []):
        provenance[f"decision_tendencies.{dt['type']}"] = FieldProvenance(
            value=dt["strength"],
            source="derived", mapping_strength=0.3, inferential_depth=2,
            confidence=FieldProvenance.compute_confidence("derived", 0.3, 2),
            parent_fields=("psychology.values", "psychology.big_five"),
            notes="Low confidence — decision tendency, not empirical bias",
        )

    result["_provenance"] = provenance
    return result


# =============================================================================
# Derivation helpers
# =============================================================================

def _derive_with_residual(
    base: float,
    override: float | None = None,
    rng: np.random.Generator | None = None,
    sd: float = 0.08,
    shared: float = 0.0,
    clamp_lo: float = 0.0,
    clamp_hi: float = 1.0,
) -> float:
    """Compute derived value with calibrated residual variance."""
    if override is not None:
        return float(np.clip(override, clamp_lo, clamp_hi))

    residual = 0.0
    if rng is not None:
        # Decompose variance budget: total_var = independent_var + shared_var
        # so independent_sd = sqrt(sd^2 - shared^2) to maintain documented total SD
        shared_abs = abs(shared)
        independent_sd = math.sqrt(max(0.0, sd ** 2 - shared_abs ** 2))
        residual = rng.normal(0, independent_sd) + shared

    return float(np.clip(base + residual, clamp_lo, clamp_hi))


def _infer_education(occupation: str | None) -> str:
    """Infer education level from occupation."""
    if not occupation:
        return "Bachelor's degree"

    occ = occupation.lower()
    advanced = ["doctor", "physician", "surgeon", "professor", "scientist", "lawyer",
                "psychologist", "pharmacist", "researcher", "data scientist"]
    trade = ["chef", "cook", "mechanic", "electrician", "plumber", "firefighter",
             "police officer", "paramedic", "baker"]

    for a in advanced:
        if a in occ:
            return "Doctoral or professional degree"
    for t in trade:
        if t in occ:
            return "Trade certification or associate degree"

    return "Bachelor's degree"


DOMAIN_MAP: dict[str, list[dict[str, Any]]] = {
    "nurse": [{"domain": "Healthcare", "proficiency": 0.55, "subdomains": ["Patient Care", "Clinical Practice"]}],
    "doctor": [{"domain": "Medicine", "proficiency": 0.55, "subdomains": ["Diagnostics", "Treatment"]}],
    "teacher": [{"domain": "Education", "proficiency": 0.50, "subdomains": ["Pedagogy", "Curriculum"]}],
    "software engineer": [{"domain": "Technology", "proficiency": 0.55, "subdomains": ["Software Development", "Systems Design"]}],
    "data scientist": [{"domain": "Technology", "proficiency": 0.50, "subdomains": ["Data Analysis", "Machine Learning"]}],
    "researcher": [{"domain": "Research", "proficiency": 0.55, "subdomains": ["Methodology", "Analysis"]}],
    "lawyer": [{"domain": "Law", "proficiency": 0.55, "subdomains": ["Legal Analysis", "Regulation"]}],
    "artist": [{"domain": "Arts", "proficiency": 0.50, "subdomains": ["Visual Arts", "Creative Expression"]}],
    "musician": [{"domain": "Music", "proficiency": 0.50, "subdomains": ["Performance", "Composition"]}],
    "entrepreneur": [{"domain": "Business", "proficiency": 0.50, "subdomains": ["Strategy", "Operations"]}],
    "chef": [{"domain": "Culinary Arts", "proficiency": 0.55, "subdomains": ["Cooking", "Food Science"]}],
    "journalist": [{"domain": "Media", "proficiency": 0.50, "subdomains": ["Writing", "Research"]}],
    "social worker": [{"domain": "Social Services", "proficiency": 0.50, "subdomains": ["Counseling", "Community"]}],
    "consultant": [{"domain": "Business", "proficiency": 0.50, "subdomains": ["Strategy", "Analysis"]}],
    "scientist": [{"domain": "Science", "proficiency": 0.55, "subdomains": ["Research", "Methodology"]}],
    "product manager": [{"domain": "Technology", "proficiency": 0.45, "subdomains": ["Product Strategy"]}],
}


def _infer_domains(occupation: str | None, industry: str | None) -> list[dict]:
    """Infer knowledge domains from occupation. Proficiency 0.4-0.6 = familiarity only."""
    if not occupation:
        return [{"domain": "General", "proficiency": 0.3, "subdomains": []}]

    occ = occupation.lower()
    for key, domains in DOMAIN_MAP.items():
        if key in occ:
            return domains

    # Fallback: use industry if available
    if industry:
        return [{"domain": industry.title(), "proficiency": 0.40, "subdomains": []}]

    return [{"domain": "General", "proficiency": 0.3, "subdomains": []}]


def _infer_goals(
    occupation: str | None,
    values: dict[str, float],
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Infer goals from occupation and top values with residual variance."""
    goals = []

    if occupation:
        goals.append(f"Excel in {occupation} role")

    # Add small noise to value ranking to vary which goals get picked
    noisy_values = {
        k: v + (rng.normal(0, 0.05) if rng else 0.0)
        for k, v in values.items()
    }
    sorted_values = sorted(noisy_values.items(), key=lambda x: x[1], reverse=True)
    value_to_goal = {
        "self_direction": "Maintain autonomy and independence",
        "stimulation": "Seek new experiences and challenges",
        "hedonism": "Enjoy life and find pleasure in work",
        "achievement": "Achieve professional success",
        "power": "Build influence and authority",
        "security": "Ensure stability and safety",
        "conformity": "Maintain social harmony",
        "tradition": "Respect customs and heritage",
        "benevolence": "Help and care for those close",
        "universalism": "Promote fairness and equality",
    }
    for val_name, _ in sorted_values[:2]:
        if val_name in value_to_goal:
            goals.append(value_to_goal[val_name])

    return goals or ["Personal growth and development"]


def _build_social_roles(communication: dict[str, float], rng: np.random.Generator) -> dict[str, dict]:
    """Build social roles (default, at_work, friend) from communication prefs + residual."""
    base_f = communication["formality"]
    base_d = communication["directness"]
    base_e = communication["emotional_expressiveness"]

    return {
        "default": {
            "formality": base_f,
            "directness": base_d,
            "emotional_expressiveness": base_e,
        },
        "at_work": {
            "formality": float(np.clip(base_f + 0.15 + rng.normal(0, 0.03), 0, 1)),
            "directness": float(np.clip(base_d + rng.normal(0, 0.03), 0, 1)),
            "emotional_expressiveness": float(np.clip(base_e - 0.1 + rng.normal(0, 0.03), 0, 1)),
        },
        "friend": {
            "formality": float(np.clip(base_f - 0.15 + rng.normal(0, 0.03), 0, 1)),
            "directness": float(np.clip(base_d + 0.1 + rng.normal(0, 0.03), 0, 1)),
            "emotional_expressiveness": float(np.clip(base_e + 0.1 + rng.normal(0, 0.03), 0, 1)),
        },
    }


def _build_decision_tendencies(
    big_five: dict[str, float],
    values: dict[str, float],
    rng: np.random.Generator,
) -> list[dict]:
    """Build decision tendencies (renamed from 'biases'). Low confidence, internal-only."""
    tendencies = []

    # Confirmation tendency: from value alignment strength
    top_val = max(values.values()) if values else 0.5
    if top_val > 0.6:
        strength = float(np.clip(0.3 + (top_val - 0.6) * 0.5 + rng.normal(0, 0.04), 0.1, 0.7))
        tendencies.append({"type": "confirmation_bias", "strength": strength})

    # Authority tendency: from conformity + tradition
    conformity = values.get("conformity", 0.5)
    tradition = values.get("tradition", 0.5)
    if (conformity + tradition) / 2 > 0.5:
        strength = float(np.clip(0.2 + (conformity + tradition - 1.0) * 0.3 + rng.normal(0, 0.04), 0.1, 0.6))
        tendencies.append({"type": "authority_bias", "strength": strength})

    # Negativity tendency: from neuroticism
    N = big_five.get("neuroticism", 0.5)
    if N > 0.5:
        strength = float(np.clip(0.2 + (N - 0.5) * 0.4 + rng.normal(0, 0.04), 0.1, 0.6))
        tendencies.append({"type": "negativity_bias", "strength": strength})

    return tendencies
