"""
Consistency Validator — checks psychological coherence of assembled personas.

11 configurable rules. All thresholds are heuristics, not empirical laws.
Modes: strict (raise), warn (log), silent (skip).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal

import numpy as np

from layer_zero.models import (
    BIG_FIVE_TRAITS,
    SCHWARTZ_VALUES,
    SCHWARTZ_ADJACENT_PAIRS,
    SCHWARTZ_OPPOSING_PAIRS,
    SCHWARTZ_POSITIONS,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_THRESHOLDS = {
    "metatrait_all_high": 0.8,           # Rule 1: flag if A, C, N all above this
    "plasticity_gap": 0.6,               # Rule 1: flag if |E - O| > this
    "cognitive_risk_closure_max": 0.7,    # Rule 2: flag if both above this
    "adjacent_value_max_delta": 0.5,      # Rule 3: max diff between adjacent values
    "opposing_value_max": 0.7,            # Rule 4: flag if both opposing above this
    "sinusoid_min_r_squared": 0.3,        # Rule 5: min R² for circumplex fit
    "big_five_value_gap": 0.5,            # Rule 6: max gap in cross-check
    "domain_proficiency_warn": 0.7,       # Rule 7: flag if proficiency above this without occupation link
    "disclosure_privacy_both_high": 0.8,  # Rule 8: flag if both above this
    "batch_min_distance": 0.05,           # Rule 9: min pairwise distance in trait space
    "cascade_min_entropy_sd": 0.03,       # Rule 11: min SD in downstream fields
}


@dataclass
class ValidationWarning:
    rule: str
    message: str
    severity: Literal["info", "warning", "error"]
    field: str = ""
    values: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class ValidationResult:
    passed: bool
    warnings: list[ValidationWarning] = dataclass_field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "warning")


# =============================================================================
# Validator
# =============================================================================

def validate_persona(
    big_five: dict[str, float],
    values: dict[str, float],
    cognitive: dict[str, float],
    communication: dict[str, float],
    disclosure_base_openness: float = 0.5,
    privacy_sensitivity: float = 0.5,
    knowledge_domains: list[dict] | None = None,
    occupation: str | None = None,
    culture_region: str | None = None,
    thresholds: dict[str, float] | None = None,
    mode: Literal["strict", "warn", "silent"] = "warn",
) -> ValidationResult:
    """Validate a persona profile for psychological coherence.

    All thresholds are configurable heuristics, not empirical laws.
    """
    if mode == "silent":
        return ValidationResult(passed=True)

    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warnings: list[ValidationWarning] = []

    # Rule 1: Big Five metatrait coherence
    warnings.extend(_check_metatrait_coherence(big_five, t))

    # Rule 2: Cognitive style compatibility
    warnings.extend(_check_cognitive_compatibility(cognitive, t))

    # Rule 3: Schwartz adjacent value coherence
    warnings.extend(_check_adjacent_values(values, t))

    # Rule 4: Schwartz opposing value conflict
    warnings.extend(_check_opposing_values(values, t))

    # Rule 5: Schwartz sinusoidal fit
    warnings.extend(_check_sinusoidal_fit(values, t))

    # Rule 6: Big Five × Schwartz cross-check
    warnings.extend(_check_big_five_value_cross(big_five, values, t))

    # Rule 7: Domain-expertise consistency
    if knowledge_domains:
        warnings.extend(_check_domain_expertise(knowledge_domains, occupation, t))

    # Rule 8: Disclosure-privacy coherence
    warnings.extend(_check_disclosure_privacy(disclosure_base_openness, privacy_sensitivity, t))

    # Rule 10: Cultural confidence scoring
    if culture_region:
        warnings.extend(_check_cultural_confidence(culture_region))

    # In strict mode, warnings are treated as errors
    if mode == "strict":
        for w in warnings:
            if w.severity == "warning":
                w.severity = "error"

    passed = not any(w.severity == "error" for w in warnings) if mode == "strict" else True

    if mode == "strict" and not passed:
        errors = [w for w in warnings if w.severity == "error"]
        raise ValueError(
            f"Persona validation failed (strict mode): {len(errors)} errors. "
            + "; ".join(w.message for w in errors[:3])
        )

    return ValidationResult(passed=passed, warnings=warnings)


def validate_batch_diversity(
    big_five_batch: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> list[ValidationWarning]:
    """Rule 9: Batch diversity check — are generated personas sufficiently different?"""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warnings = []

    if len(big_five_batch) < 2:
        return warnings

    # Compute pairwise Euclidean distances
    from scipy.spatial.distance import pdist
    distances = pdist(big_five_batch)
    min_dist = distances.min() if len(distances) > 0 else 0.0

    if min_dist < t["batch_min_distance"]:
        warnings.append(ValidationWarning(
            rule="batch_diversity",
            message=f"Minimum pairwise distance ({min_dist:.4f}) below threshold ({t['batch_min_distance']}). Personas may be too similar.",
            severity="warning",
            values={"min_distance": float(min_dist)},
        ))

    return warnings


def validate_cascade_collapse(
    downstream_values: list[float],
    field_name: str = "",
    thresholds: dict[str, float] | None = None,
) -> list[ValidationWarning]:
    """Rule 11: Cascade collapse check — downstream fields retain entropy."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warnings = []

    if len(downstream_values) < 5:
        return warnings

    sd = float(np.std(downstream_values))
    if sd < t["cascade_min_entropy_sd"]:
        warnings.append(ValidationWarning(
            rule="cascade_collapse",
            message=f"Downstream field '{field_name}' has low variance (SD={sd:.4f} < {t['cascade_min_entropy_sd']}). Possible cascade collapse.",
            severity="warning",
            field=field_name,
            values={"std_dev": sd},
        ))

    return warnings


# =============================================================================
# Individual rule implementations
# =============================================================================

def _check_metatrait_coherence(big_five: dict[str, float], t: dict) -> list[ValidationWarning]:
    warnings = []
    A, C, N = big_five.get("agreeableness", 0.5), big_five.get("conscientiousness", 0.5), big_five.get("neuroticism", 0.5)
    E, O = big_five.get("extraversion", 0.5), big_five.get("openness", 0.5)

    thresh = t["metatrait_all_high"]
    if A > thresh and C > thresh and N > thresh:
        warnings.append(ValidationWarning(
            rule="metatrait_stability",
            message=f"High A ({A:.2f}), C ({C:.2f}), AND N ({N:.2f}) is statistically rare — Stability metatrait contradiction.",
            severity="warning",
            values={"agreeableness": A, "conscientiousness": C, "neuroticism": N},
        ))

    gap = t["plasticity_gap"]
    if abs(E - O) > gap:
        warnings.append(ValidationWarning(
            rule="metatrait_plasticity",
            message=f"Large gap between E ({E:.2f}) and O ({O:.2f}) — Plasticity metatrait usually shows co-movement.",
            severity="info",
            values={"extraversion": E, "openness": O},
        ))

    return warnings


def _check_cognitive_compatibility(cognitive: dict[str, float], t: dict) -> list[ValidationWarning]:
    warnings = []
    rt = cognitive.get("risk_tolerance", 0.5)
    nfc = cognitive.get("need_for_closure", 0.5)
    thresh = t["cognitive_risk_closure_max"]

    if rt > thresh and nfc > thresh:
        warnings.append(ValidationWarning(
            rule="cognitive_compatibility",
            message=f"High risk_tolerance ({rt:.2f}) AND need_for_closure ({nfc:.2f}) — empirically incompatible.",
            severity="warning",
            values={"risk_tolerance": rt, "need_for_closure": nfc},
        ))

    return warnings


def _check_adjacent_values(values: dict[str, float], t: dict) -> list[ValidationWarning]:
    warnings = []
    max_delta = t["adjacent_value_max_delta"]

    for v1, v2 in SCHWARTZ_ADJACENT_PAIRS:
        val1 = values.get(v1, 0.5)
        val2 = values.get(v2, 0.5)
        delta = abs(val1 - val2)
        if delta > max_delta:
            warnings.append(ValidationWarning(
                rule="adjacent_value_coherence",
                message=f"Adjacent values {v1} ({val1:.2f}) and {v2} ({val2:.2f}) differ by {delta:.2f} > {max_delta}.",
                severity="info",
                values={v1: val1, v2: val2},
            ))

    return warnings


def _check_opposing_values(values: dict[str, float], t: dict) -> list[ValidationWarning]:
    warnings = []
    thresh = t["opposing_value_max"]

    for v1, v2 in SCHWARTZ_OPPOSING_PAIRS:
        val1 = values.get(v1, 0.5)
        val2 = values.get(v2, 0.5)
        if val1 > thresh and val2 > thresh:
            warnings.append(ValidationWarning(
                rule="opposing_value_conflict",
                message=f"Opposing values {v1} ({val1:.2f}) and {v2} ({val2:.2f}) both > {thresh} — motivationally incoherent.",
                severity="warning",
                values={v1: val1, v2: val2},
            ))

    return warnings


def _check_sinusoidal_fit(values: dict[str, float], t: dict) -> list[ValidationWarning]:
    warnings = []

    # Fit a sinusoid to the value profile in circle order
    positions = np.array([SCHWARTZ_POSITIONS[v] for v in SCHWARTZ_VALUES])
    vals = np.array([values.get(v, 0.5) for v in SCHWARTZ_VALUES])

    # Fit: value = a + b*cos(pos) + c*sin(pos) via least squares
    X = np.column_stack([np.ones(10), np.cos(positions), np.sin(positions)])
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X, vals, rcond=None)
        ss_res = np.sum((vals - X @ coeffs) ** 2)
        ss_tot = np.sum((vals - vals.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    except np.linalg.LinAlgError:
        r_squared = 0.0

    if r_squared < t["sinusoid_min_r_squared"]:
        warnings.append(ValidationWarning(
            rule="sinusoidal_fit",
            message=f"Value profile R²={r_squared:.3f} < {t['sinusoid_min_r_squared']} — may lack circumplex structure.",
            severity="info",
            values={"r_squared": float(r_squared)},
        ))

    return warnings


def _check_big_five_value_cross(big_five: dict[str, float], values: dict[str, float], t: dict) -> list[ValidationWarning]:
    warnings = []
    gap = t["big_five_value_gap"]

    # Known correlations: O should correlate with self_direction, stimulation, universalism
    O = big_five.get("openness", 0.5)
    if O > 0.8:
        sd = values.get("self_direction", 0.5)
        stim = values.get("stimulation", 0.5)
        univ = values.get("universalism", 0.5)
        if sd < (O - gap) and stim < (O - gap) and univ < (O - gap):
            warnings.append(ValidationWarning(
                rule="big_five_value_cross",
                message=f"High O ({O:.2f}) but low self_direction ({sd:.2f}), stimulation ({stim:.2f}), universalism ({univ:.2f}).",
                severity="info",
                values={"openness": O, "self_direction": sd, "stimulation": stim, "universalism": univ},
            ))

    # A should correlate with benevolence
    A = big_five.get("agreeableness", 0.5)
    if A > 0.8 and values.get("benevolence", 0.5) < (A - gap):
        warnings.append(ValidationWarning(
            rule="big_five_value_cross",
            message=f"High A ({A:.2f}) but low benevolence ({values.get('benevolence', 0.5):.2f}).",
            severity="info",
            values={"agreeableness": A, "benevolence": values.get("benevolence", 0.5)},
        ))

    return warnings


def _check_domain_expertise(domains: list[dict], occupation: str | None, t: dict) -> list[ValidationWarning]:
    warnings = []
    thresh = t["domain_proficiency_warn"]

    for domain in domains:
        prof = domain.get("proficiency", 0.0)
        if prof > thresh and occupation:
            # Very basic check — domain name should relate to occupation
            domain_name = domain.get("domain", "").lower()
            occ = occupation.lower()
            if domain_name not in occ and occ not in domain_name:
                warnings.append(ValidationWarning(
                    rule="domain_expertise",
                    message=f"Domain '{domain.get('domain')}' has proficiency {prof:.2f} but occupation '{occupation}' may not justify expertise.",
                    severity="warning",
                    values={"domain": domain.get("domain"), "proficiency": prof, "occupation": occupation},
                ))

    return warnings


def _check_disclosure_privacy(base_openness: float, privacy_sensitivity: float, t: dict) -> list[ValidationWarning]:
    warnings = []
    thresh = t["disclosure_privacy_both_high"]

    if base_openness > thresh and privacy_sensitivity > thresh:
        warnings.append(ValidationWarning(
            rule="disclosure_privacy",
            message=f"High disclosure openness ({base_openness:.2f}) AND high privacy sensitivity ({privacy_sensitivity:.2f}) — contradictory.",
            severity="warning",
            values={"base_openness": base_openness, "privacy_sensitivity": privacy_sensitivity},
        ))

    return warnings


def _check_cultural_confidence(culture_region: str) -> list[ValidationWarning]:
    from layer_zero.priors.big_five import get_culture_confidence
    confidence = get_culture_confidence(culture_region)

    warnings = []
    if confidence < 0.5:
        warnings.append(ValidationWarning(
            rule="cultural_confidence",
            message=f"Culture region '{culture_region}' has low research confidence ({confidence:.2f}). Big Five validity may be reduced.",
            severity="info",
            values={"culture_region": culture_region, "confidence": confidence},
        ))

    return warnings
