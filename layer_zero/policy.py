"""
Policy Applier — system-governed defaults that do NOT vary by personality.

Policy fields are system-governed defaults with persona-bounded presentation,
not persona-derived permissions. A high-openness persona gets more exploratory
communication style, but the same safety floor as everyone else.
"""

from __future__ import annotations

from typing import Any

from layer_zero.models import FieldProvenance


# =============================================================================
# System defaults — these are invariant across all personas
# =============================================================================

SYSTEM_POLICY_DEFAULTS: dict[str, Any] = {
    "uncertainty": {
        "admission_threshold": 0.4,
        "hedging_frequency": 0.5,
        "clarification_tendency": 0.5,
        "knowledge_boundary_strictness": 0.7,
    },
    "claim_policy": {
        "allowed_claim_types": ["personal_experience", "general_common_knowledge"],
        "lookup_behavior": "hedge",
        "expert_threshold": 0.7,
        "citation_required_when": {"proficiency_below": 0.6, "factual_or_time_sensitive": True},
    },
    "disclosure_policy": {
        "bounds": (0.1, 0.9),
        "factors": {
            "topic_sensitivity": -0.25,
            "trust_level": 0.35,
            "emotional_state": 0.1,
        },
    },
    "invariants": {
        "must_avoid": ["revealing private personal information"],
    },
}

# Cannot-claim mappings by occupation category
_CANNOT_CLAIM_BY_OCCUPATION: dict[str, list[str]] = {
    "nurse": ["licensed physician", "surgeon", "psychiatrist"],
    "teacher": ["licensed therapist", "medical doctor"],
    "software engineer": ["licensed engineer (PE)", "lawyer"],
    "chef": ["nutritionist", "dietitian"],
    "journalist": ["lawyer", "medical doctor"],
    "fitness coach": ["physical therapist", "medical doctor"],
    "social worker": ["psychiatrist", "licensed physician"],
    "consultant": ["licensed professional in client's domain"],
    "musician": ["music therapist"],
    "artist": ["art therapist"],
}


# =============================================================================
# Policy application
# =============================================================================

def apply_policy_defaults(
    persona_fields: dict[str, Any],
    occupation: str | None = None,
) -> dict[str, Any]:
    """Apply system-governed policy defaults to persona fields.

    These fields do NOT vary by personality. They are safety floors.

    Args:
        persona_fields: Dict with persona-layer fields (from gap_filler).
        occupation: Persona's occupation for cannot_claim inference.

    Returns:
        Tuple of (policy_fields_dict, policy_provenance_dict).
    """
    policy: dict[str, Any] = {}
    provenance: dict[str, FieldProvenance] = {}

    # --- Uncertainty policy (system default) ---
    policy["uncertainty"] = dict(SYSTEM_POLICY_DEFAULTS["uncertainty"])
    for field_name in policy["uncertainty"]:
        provenance[f"uncertainty.{field_name}"] = FieldProvenance(
            value=policy["uncertainty"][field_name],
            source="default",
            confidence=0.9,  # system defaults are high-confidence by design
            notes="System-governed safety floor — does not vary by personality",
        )

    # --- Claim policy (system default) ---
    policy["claim_policy"] = dict(SYSTEM_POLICY_DEFAULTS["claim_policy"])
    for field_name in ["allowed_claim_types", "lookup_behavior", "expert_threshold"]:
        provenance[f"claim_policy.{field_name}"] = FieldProvenance(
            value=policy["claim_policy"][field_name],
            source="default",
            confidence=0.9,
            notes="System-governed — not persona-derived",
        )

    # --- Disclosure policy ---
    # base_openness is persona-influenced (from agreeableness + openness in gap_filler)
    # but bounds are system-governed
    privacy_sensitivity = persona_fields.get("privacy_sensitivity", 0.5)
    base_openness = max(0.1, min(0.9, 1.0 - privacy_sensitivity))

    policy["disclosure_policy"] = {
        "base_openness": base_openness,
        "bounds": SYSTEM_POLICY_DEFAULTS["disclosure_policy"]["bounds"],
        "factors": dict(SYSTEM_POLICY_DEFAULTS["disclosure_policy"]["factors"]),
    }
    provenance["disclosure_policy.bounds"] = FieldProvenance(
        value=SYSTEM_POLICY_DEFAULTS["disclosure_policy"]["bounds"],
        source="default",
        confidence=0.9,
        notes="System-governed disclosure bounds",
    )
    provenance["disclosure_policy.base_openness"] = FieldProvenance(
        value=base_openness,
        source="derived",
        confidence=FieldProvenance.compute_confidence("derived", 0.5, 2),
        inferential_depth=2,
        parent_fields=("privacy_sensitivity",),
        notes="Derived from privacy_sensitivity (persona layer), clamped by system bounds",
    )

    # --- Invariants ---
    age = persona_fields.get("age")
    occupation_str = occupation or persona_fields.get("occupation", "")
    location = persona_fields.get("location", "")

    identity_facts = []
    if age:
        identity_facts.append(f"Age {age}")
    if occupation_str:
        identity_facts.append(occupation_str.title())
    if location:
        identity_facts.append(f"Lives in {location}")

    cannot_claim: list[str] = []
    if occupation_str:
        occ_lower = occupation_str.lower()
        for key, claims in _CANNOT_CLAIM_BY_OCCUPATION.items():
            if key in occ_lower:
                cannot_claim.extend(claims)
                break

    policy["invariants"] = {
        "identity_facts": identity_facts,
        "cannot_claim": cannot_claim,
        "must_avoid": list(SYSTEM_POLICY_DEFAULTS["invariants"]["must_avoid"]),
    }

    # --- Time scarcity and topic sensitivities (defaults) ---
    policy["time_scarcity"] = 0.3
    policy["topic_sensitivities"] = []

    return policy, provenance
