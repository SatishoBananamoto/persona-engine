"""
Layer Zero — Persona Minting Machine

Mints psychologically coherent Persona objects from user inputs
(text descriptions, structured fields, CSV segments, or direct specification)
for use with the persona-engine.

Usage:
    import layer_zero

    # From text
    personas = layer_zero.from_description("35-year-old nurse from Chicago", count=5)

    # From structured fields
    personas = layer_zero.mint(occupation="nurse", age=35, count=10)

    # From CSV segments
    personas = layer_zero.from_csv("segments.csv", count_per_segment=20)

    # Direct specification
    personas = layer_zero.mint(
        big_five={"openness": 0.8}, occupation="researcher", count=1
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np

from layer_zero.models import (
    BIG_FIVE_TRAITS,
    SCHWARTZ_VALUES,
    FieldProvenance,
    MintRequest,
    MintedPersona,
    TraitPrior,
)
from layer_zero.parser.text_parser import parse_description
from layer_zero.parser.csv_parser import parse_csv, segment_to_mint_requests
from layer_zero.priors.big_five import (
    compute_big_five_prior,
    get_culture_value_shifts,
    infer_culture_region,
)
from layer_zero.priors.values import generate_schwartz_values
from layer_zero.sampler import sample_big_five
from layer_zero.gap_filler import fill_gaps
from layer_zero.policy import apply_policy_defaults
from layer_zero.validator import validate_persona, validate_batch_diversity, validate_cascade_collapse
from layer_zero.assembler import assemble_persona

__version__ = "0.1.0"


# =============================================================================
# Public API
# =============================================================================

def mint(
    *,
    name: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    occupation: str | None = None,
    industry: str | None = None,
    location: str | None = None,
    education: str | None = None,
    culture_region: str | None = None,
    traits: list[str] | None = None,
    big_five: dict[str, float] | None = None,
    values: dict[str, float] | None = None,
    cognitive: dict[str, float] | None = None,
    communication: dict[str, float] | None = None,
    goals: list[str] | None = None,
    domains: list[dict] | None = None,
    count: int = 1,
    seed: int = 42,
    validate: str = "warn",
    validator_config: dict[str, float] | None = None,
) -> list[MintedPersona]:
    """Mint psychologically coherent personas from structured fields.

    This is the primary entry point for Tier 2 (structured) and Tier 4 (direct) inputs.

    Args:
        name: Persona name (generated if None).
        age: Age 18-100 (sampled from defaults if None).
        gender: Gender string (optional).
        occupation: Occupation (primary signal for personality inference).
        industry: Industry sector.
        location: Location (used for culture region inference).
        education: Education level (inferred from occupation if None).
        culture_region: Explicit culture region override.
        traits: Adjectives like ["analytical", "warm"] that shift Big Five priors.
        big_five: Direct Big Five overrides {trait: float 0-1}.
        values: Direct Schwartz value overrides {value: float 0-1}.
        cognitive: Direct cognitive style overrides.
        communication: Direct communication preference overrides.
        goals: Explicit goals list.
        domains: Explicit knowledge domains.
        count: Number of personas to generate.
        seed: Random seed for reproducibility.
        validate: "strict", "warn", or "silent".
        validator_config: Custom threshold overrides for validator.

    Returns:
        List of MintedPersona objects (engine-compatible + provenance).
    """
    request = MintRequest(
        name=name,
        age=age,
        gender=gender,
        occupation=occupation,
        industry=industry,
        location=location,
        education=education,
        culture_region=culture_region,
        trait_hints=traits or [],
        big_five_overrides=big_five or {},
        values_overrides=values or {},
        cognitive_overrides=cognitive or {},
        communication_overrides=communication or {},
        goals=goals or [],
        domains=domains or [],
        count=count,
        seed=seed,
    )

    return _run_pipeline(request, validate=validate, validator_config=validator_config)


def from_description(
    description: str,
    *,
    count: int = 1,
    seed: int = 42,
    validate: str = "warn",
    validator_config: dict[str, float] | None = None,
) -> list[MintedPersona]:
    """Mint personas from a natural language description (Tier 1).

    Examples:
        from_description("A 35-year-old product manager in fintech", count=5)
        from_description("Cautious nurse from Tokyo who values security")

    Args:
        description: Natural language persona description.
        count: Number of personas to generate.
        seed: Random seed.
        validate: Validator mode.
        validator_config: Custom thresholds.

    Returns:
        List of MintedPersona objects.
    """
    request = parse_description(description)
    request.count = count
    request.seed = seed

    return _run_pipeline(request, validate=validate, validator_config=validator_config)


def from_csv(
    path: str,
    *,
    count_per_segment: int = 10,
    seed: int = 42,
    validate: str = "warn",
    validator_config: dict[str, float] | None = None,
) -> list[MintedPersona]:
    """Mint personas from a CSV segment file (Tier 3).

    Each CSV row defines a segment with age ranges, gender distributions,
    occupations, and locations. Generates count_per_segment personas per row.

    CSV columns: segment_name, age_min, age_max, occupation, location,
                 gender_dist (format: "female:0.6,male:0.4"), count

    Args:
        path: Path to CSV file.
        count_per_segment: Default personas per segment (overridden by row 'count' column).
        seed: Random seed.
        validate: Validator mode.
        validator_config: Custom thresholds.

    Returns:
        List of MintedPersona objects (all segments combined).
    """
    segments = parse_csv(path)
    all_personas: list[MintedPersona] = []

    for seg_idx, segment in enumerate(segments):
        if segment.count <= 0:
            segment.count = count_per_segment

        seg_seed = seed + seg_idx * 10000
        mint_requests = segment_to_mint_requests(segment, seed=seg_seed)

        for req in mint_requests:
            result = _run_pipeline(req, validate=validate, validator_config=validator_config)
            all_personas.extend(result)

    return all_personas


# =============================================================================
# Internal pipeline
# =============================================================================

def _run_pipeline(
    request: MintRequest,
    validate: str = "warn",
    validator_config: dict[str, float] | None = None,
) -> list[MintedPersona]:
    """Run the full Layer Zero pipeline: parse → priors → sample → fill → policy → validate → assemble."""

    seed = request.seed or 42
    count = request.count

    # Stage 1: Compute Big Five priors
    bf_priors = compute_big_five_prior(request)

    # Stage 2: Sample Big Five trait vectors
    bf_samples = sample_big_five(bf_priors, count=count, seed=seed)

    # Stage 3: Get culture region for value priors
    culture = request.culture_region
    if not culture and request.location:
        culture = infer_culture_region(request.location)
    culture_shifts = get_culture_value_shifts(culture) if culture else None

    # Stage 4: Generate Schwartz value profiles
    sv_samples = generate_schwartz_values(
        request, count=count, seed=seed + 1000, culture_value_shifts=culture_shifts,
    )

    # Stage 5-9: For each persona in batch
    personas: list[MintedPersona] = []
    all_bf_for_diversity = bf_samples  # for batch diversity check

    for i in range(count):
        # Convert numpy row to dict, pin overrides exactly
        bf_dict = {t: float(bf_samples[i, j]) for j, t in enumerate(BIG_FIVE_TRAITS)}
        for trait, value in request.big_five_overrides.items():
            if trait in bf_dict:
                bf_dict[trait] = value  # exact pin, not approximate
        sv_dict = {v: float(sv_samples[i, j]) for j, v in enumerate(SCHWARTZ_VALUES)}

        # Fill gaps with residuals
        filled = fill_gaps(
            bf_dict, sv_dict, request, bf_priors,
            seed=seed, persona_index=i,
        )

        # Apply policy defaults
        filled["age"] = request.age
        filled["occupation"] = request.occupation
        filled["location"] = request.location
        policy, policy_provenance = apply_policy_defaults(filled, occupation=request.occupation)
        # Merge policy provenance into filled provenance
        if "_provenance" not in filled:
            filled["_provenance"] = {}
        filled["_provenance"].update(policy_provenance)

        # Validate
        validation = validate_persona(
            big_five=bf_dict,
            values=sv_dict,
            cognitive=filled["cognitive_style"],
            communication=filled["communication"],
            disclosure_base_openness=policy["disclosure_policy"]["base_openness"],
            privacy_sensitivity=filled.get("privacy_sensitivity", 0.5),
            knowledge_domains=filled.get("knowledge_domains"),
            occupation=request.occupation,
            culture_region=culture,
            thresholds=validator_config,
            mode=validate,
        )

        # Assemble
        mp = assemble_persona(
            big_five=bf_dict,
            schwartz=sv_dict,
            filled=filled,
            policy=policy,
            request_occupation=request.occupation,
            request_age=request.age,
            request_location=request.location,
            request_gender=request.gender,
            seed=seed,
            persona_index=i,
        )
        mp.warnings = [w.message for w in validation.warnings]
        personas.append(mp)

    # Batch checks (only meaningful for count > 1)
    if count > 1 and validate != "silent":
        # Rule 9: Batch diversity
        diversity_warnings = validate_batch_diversity(all_bf_for_diversity, validator_config)
        if diversity_warnings:
            for mp in personas:
                mp.warnings.extend([w.message for w in diversity_warnings])

        # Rule 11: Cascade collapse — check downstream fields retain entropy
        cognitive_analytical = [
            p.persona.psychology.cognitive_style.analytical_intuitive for p in personas
        ]
        comm_formality = [
            p.persona.psychology.communication.formality for p in personas
        ]
        for field_name, values_list in [
            ("cognitive_style.analytical_intuitive", cognitive_analytical),
            ("communication.formality", comm_formality),
        ]:
            collapse_warnings = validate_cascade_collapse(values_list, field_name, validator_config)
            if collapse_warnings:
                for mp in personas:
                    mp.warnings.extend([w.message for w in collapse_warnings])

    return personas
