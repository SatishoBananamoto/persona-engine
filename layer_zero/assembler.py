"""
Persona Assembler — constructs engine-compatible Persona objects.

This is where Layer Zero output meets the persona-engine schema.
Maps internal field dicts → Persona(**fields) with Pydantic validation.
Attaches provenance metadata to MintedPersona wrapper.
"""

from __future__ import annotations

import hashlib
from typing import Any

from layer_zero.models import (
    BIG_FIVE_TRAITS,
    SCHWARTZ_VALUES,
    FieldProvenance,
    MintedPersona,
)

# Import engine schema
from persona_engine.schema.persona_schema import (
    BigFiveTraits,
    CognitiveStyle,
    CommunicationPreferences,
    DomainKnowledge,
    PersonalityProfile,
    SchwartzValues,
    Identity,
    Goal,
    SocialRole,
    UncertaintyPolicy,
    ClaimPolicy,
    PersonaInvariants,
    DisclosurePolicy,
    DynamicState,
    ResponsePattern,
    DecisionPolicy,
    Bias,
    Persona,
)


def assemble_persona(
    big_five: dict[str, float],
    schwartz: dict[str, float],
    filled: dict[str, Any],
    policy: dict[str, Any],
    request_occupation: str | None = None,
    request_age: int | None = None,
    request_location: str | None = None,
    request_gender: str | None = None,
    seed: int = 42,
    persona_index: int = 0,
) -> MintedPersona:
    """Assemble a complete engine-compatible Persona from Layer Zero fields.

    Args:
        big_five: Sampled Big Five traits.
        schwartz: Sampled Schwartz values.
        filled: Gap-filled fields (from gap_filler).
        policy: Policy fields (from policy applier).
        request_*: Original request fields for identity.
        seed: Base seed for ID generation.
        persona_index: Batch index.

    Returns:
        MintedPersona with engine Persona + provenance.
    """
    name = filled.get("name", f"Persona_{persona_index}")
    occupation = request_occupation or "Professional"
    age = request_age or 35
    location = request_location or "United States"
    gender = request_gender
    education = filled.get("education", "Bachelor's degree")

    # Generate persona_id: content hash + batch index
    id_content = f"{name}:{occupation}:{seed}:{persona_index}"
    content_hash = hashlib.sha256(id_content.encode()).hexdigest()[:12]
    persona_id = f"P_GEN_{content_hash}_{persona_index}"

    label = f"{name} - {occupation}, {location}"

    # Build identity
    background = f"{name} is a {age}-year-old {occupation} based in {location}."
    identity = Identity(
        age=age,
        gender=gender,
        location=location,
        education=education,
        occupation=occupation,
        background=background,
    )

    # Build psychology
    big_five_model = BigFiveTraits(**{t: big_five[t] for t in BIG_FIVE_TRAITS})
    schwartz_model = SchwartzValues(**{v: schwartz[v] for v in SCHWARTZ_VALUES})

    cog = filled["cognitive_style"]
    cognitive_model = CognitiveStyle(
        analytical_intuitive=cog["analytical_intuitive"],
        systematic_heuristic=cog["systematic_heuristic"],
        risk_tolerance=cog["risk_tolerance"],
        need_for_closure=cog["need_for_closure"],
        cognitive_complexity=cog["cognitive_complexity"],
    )

    comm = filled["communication"]
    comm_model = CommunicationPreferences(
        verbosity=comm["verbosity"],
        formality=comm["formality"],
        directness=comm["directness"],
        emotional_expressiveness=comm["emotional_expressiveness"],
    )

    psychology = PersonalityProfile(
        big_five=big_five_model,
        values=schwartz_model,
        cognitive_style=cognitive_model,
        communication=comm_model,
    )

    # Build knowledge domains
    domains = []
    for d in filled.get("knowledge_domains", []):
        domains.append(DomainKnowledge(
            domain=d["domain"],
            proficiency=d.get("proficiency", 0.4),
            subdomains=d.get("subdomains", []),
        ))

    # Build goals
    goals_list = filled.get("goals", [])
    primary_goals = []
    secondary_goals = []
    for i, g in enumerate(goals_list):
        goal_obj = Goal(goal=g, weight=0.8 if i == 0 else 0.5)
        if i < 2:
            primary_goals.append(goal_obj)
        else:
            secondary_goals.append(goal_obj)

    # Build social roles
    social_roles_dict = {}
    for role_name, role_data in filled.get("social_roles", {}).items():
        social_roles_dict[role_name] = SocialRole(
            formality=role_data["formality"],
            directness=role_data["directness"],
            emotional_expressiveness=role_data["emotional_expressiveness"],
        )
    if "default" not in social_roles_dict:
        social_roles_dict["default"] = SocialRole(
            formality=comm["formality"],
            directness=comm["directness"],
            emotional_expressiveness=comm["emotional_expressiveness"],
        )

    # Build policies
    unc = policy["uncertainty"]
    uncertainty = UncertaintyPolicy(
        admission_threshold=unc["admission_threshold"],
        hedging_frequency=unc["hedging_frequency"],
        clarification_tendency=unc["clarification_tendency"],
        knowledge_boundary_strictness=unc["knowledge_boundary_strictness"],
    )

    cp = policy["claim_policy"]
    claim_policy = ClaimPolicy(
        allowed_claim_types=cp["allowed_claim_types"],
        lookup_behavior=cp["lookup_behavior"],
        expert_threshold=cp.get("expert_threshold", 0.7),
        citation_required_when=cp.get("citation_required_when", {}),
    )

    inv = policy["invariants"]
    invariants = PersonaInvariants(
        identity_facts=inv["identity_facts"],
        cannot_claim=inv["cannot_claim"],
        must_avoid=inv["must_avoid"],
    )

    dp = policy["disclosure_policy"]
    disclosure_policy = DisclosurePolicy(
        base_openness=dp["base_openness"],
        factors=dp.get("factors", {}),
        bounds=tuple(dp["bounds"]),
    )

    # Build initial state
    state = filled["initial_state"]
    initial_state = DynamicState(
        mood_valence=state["mood_valence"],
        mood_arousal=state["mood_arousal"],
        fatigue=state["fatigue"],
        stress=state["stress"],
        engagement=state["engagement"],
    )

    # Build decision tendencies as biases
    biases = []
    for dt in filled.get("decision_tendencies", []):
        biases.append(Bias(type=dt["type"], strength=dt["strength"]))

    # Build decision policies and response patterns (templates)
    decision_policies: list[DecisionPolicy] = []
    response_patterns: list[ResponsePattern] = []

    # Assemble Persona
    persona = Persona(
        persona_id=persona_id,
        version="1.0",
        label=label,
        identity=identity,
        psychology=psychology,
        knowledge_domains=domains,
        primary_goals=primary_goals,
        secondary_goals=secondary_goals,
        social_roles=social_roles_dict,
        uncertainty=uncertainty,
        claim_policy=claim_policy,
        invariants=invariants,
        disclosure_policy=disclosure_policy,
        initial_state=initial_state,
        biases=biases,
        decision_policies=decision_policies,
        response_patterns=response_patterns,
        privacy_sensitivity=filled.get("privacy_sensitivity", 0.5),
        time_scarcity=policy.get("time_scarcity", 0.3),
        topic_sensitivities=policy.get("topic_sensitivities", []),
    )

    # Collect provenance
    provenance = filled.get("_provenance", {})

    # Add provenance for Big Five (sampled)
    for trait in BIG_FIVE_TRAITS:
        key = f"psychology.big_five.{trait}"
        if key not in provenance:
            provenance[key] = FieldProvenance(
                value=big_five[trait], source="sampled",
                confidence=FieldProvenance.compute_confidence("sampled", 0.6),
            )

    # Add provenance for Schwartz values (sampled via circumplex)
    for val in SCHWARTZ_VALUES:
        key = f"psychology.values.{val}"
        if key not in provenance:
            provenance[key] = FieldProvenance(
                value=schwartz[val], source="sampled",
                confidence=FieldProvenance.compute_confidence("sampled", 0.5),
                notes="Generated from circumplex structure",
            )

    return MintedPersona(
        persona=persona,
        provenance=provenance,
    )
