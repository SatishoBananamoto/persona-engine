"""
Diversity and Population Alignment — world-class persona set quality.

Provides:
1. Population alignment via importance sampling against empirical norms
2. Diversity reporting (Simpson's index, coverage, distribution alignment)
3. Representativeness scoring

Sources:
- Population-Aligned Persona Generation (Microsoft Research, 2025)
- SCOPE: Socially-Grounded Persona Framework (2026)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from layer_zero.models import BIG_FIVE_TRAITS, SCHWARTZ_VALUES, MintedPersona


# =============================================================================
# Population norms (empirical reference distributions)
# =============================================================================

# Population Big Five norms (meta-analytic means and SDs on 0-1 scale)
# Source: McCrae & Costa (2004), Schmitt et al. (2007)
POPULATION_BIG_FIVE_NORMS = {
    "global": {
        "openness":          {"mean": 0.55, "sd": 0.12},
        "conscientiousness": {"mean": 0.52, "sd": 0.13},
        "extraversion":      {"mean": 0.50, "sd": 0.13},
        "agreeableness":     {"mean": 0.54, "sd": 0.12},
        "neuroticism":       {"mean": 0.45, "sd": 0.13},
    },
}


# =============================================================================
# Diversity Report
# =============================================================================

@dataclass
class DiversityReport:
    """Comprehensive diversity analysis of a generated persona set."""
    count: int
    simpsons_index: float  # 0 = no diversity, 1 = maximum diversity
    trait_coverage: dict[str, float]  # per-trait coverage of [0,1] range
    mean_pairwise_distance: float
    min_pairwise_distance: float
    distribution_alignment: float  # KL divergence from population norms (lower = better)
    quadrant_coverage: float  # what fraction of trait-space quadrants are populated
    warnings: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """0-1 composite diversity score."""
        # Weighted combination of metrics
        return (
            self.simpsons_index * 0.25 +
            self.quadrant_coverage * 0.25 +
            min(1.0, self.mean_pairwise_distance * 5) * 0.25 +  # scale up
            max(0.0, 1.0 - self.distribution_alignment) * 0.25  # invert KL
        )


def analyze_diversity(personas: list[MintedPersona]) -> DiversityReport:
    """Generate a comprehensive diversity report for a persona set.

    Args:
        personas: List of MintedPersona objects.

    Returns:
        DiversityReport with metrics and warnings.
    """
    n = len(personas)
    if n < 2:
        return DiversityReport(
            count=n, simpsons_index=0.0, trait_coverage={},
            mean_pairwise_distance=0.0, min_pairwise_distance=0.0,
            distribution_alignment=0.0, quadrant_coverage=0.0,
            warnings=["Need at least 2 personas for diversity analysis"],
        )

    # Extract Big Five matrix
    bf_matrix = np.array([
        [getattr(p.persona.psychology.big_five, t) for t in BIG_FIVE_TRAITS]
        for p in personas
    ])

    # --- Pairwise distances ---
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(np.linalg.norm(bf_matrix[i] - bf_matrix[j]))
    distances = np.array(distances)
    mean_dist = float(distances.mean())
    min_dist = float(distances.min())

    # --- Simpson's Diversity Index ---
    # Bin each persona into a trait-space cell (high/low per trait = 2^5 = 32 cells)
    cells: dict[tuple, int] = {}
    for row in bf_matrix:
        cell = tuple(1 if v > 0.5 else 0 for v in row)
        cells[cell] = cells.get(cell, 0) + 1

    simpsons = 1.0 - sum(count * (count - 1) for count in cells.values()) / (n * (n - 1)) if n > 1 else 0.0

    # --- Trait coverage (what fraction of [0,1] range is populated per trait) ---
    n_bins = 5  # divide [0,1] into 5 bins
    trait_coverage = {}
    for j, trait in enumerate(BIG_FIVE_TRAITS):
        bins_hit = set()
        for val in bf_matrix[:, j]:
            bins_hit.add(min(n_bins - 1, int(val * n_bins)))
        trait_coverage[trait] = len(bins_hit) / n_bins

    # --- Distribution alignment (KL divergence from population norms) ---
    kl_total = 0.0
    norms = POPULATION_BIG_FIVE_NORMS["global"]
    for j, trait in enumerate(BIG_FIVE_TRAITS):
        sample_mean = float(bf_matrix[:, j].mean())
        sample_sd = float(bf_matrix[:, j].std())
        pop_mean = norms[trait]["mean"]
        pop_sd = norms[trait]["sd"]
        # KL divergence between two Gaussians
        if sample_sd > 0.01 and pop_sd > 0.01:
            kl = (
                math.log(pop_sd / sample_sd)
                + (sample_sd ** 2 + (sample_mean - pop_mean) ** 2) / (2 * pop_sd ** 2)
                - 0.5
            )
            kl_total += max(0.0, kl)
    distribution_alignment = kl_total / len(BIG_FIVE_TRAITS)  # average per trait

    # --- Quadrant coverage (how many of 32 high/low cells are populated) ---
    quadrant_coverage = len(cells) / 32.0

    # --- Warnings ---
    warnings = []
    if simpsons < 0.7:
        warnings.append(f"Low Simpson's diversity ({simpsons:.2f}) — personas may cluster")
    if min_dist < 0.05:
        warnings.append(f"Very similar personas detected (min distance {min_dist:.3f})")
    if distribution_alignment > 0.5:
        warnings.append(f"Distribution diverges from population norms (KL={distribution_alignment:.3f})")
    for trait, cov in trait_coverage.items():
        if cov < 0.4:
            warnings.append(f"Low coverage for {trait} ({cov:.0%} of range)")

    return DiversityReport(
        count=n,
        simpsons_index=simpsons,
        trait_coverage=trait_coverage,
        mean_pairwise_distance=mean_dist,
        min_pairwise_distance=min_dist,
        distribution_alignment=distribution_alignment,
        quadrant_coverage=quadrant_coverage,
        warnings=warnings,
    )


# =============================================================================
# Population Alignment (Importance Sampling)
# =============================================================================

def align_to_population(
    personas: list[MintedPersona],
    target_norms: dict[str, dict[str, float]] | None = None,
    n_select: int | None = None,
) -> list[MintedPersona]:
    """Select a subset of personas that best matches a target population distribution.

    Uses importance sampling weights to select personas whose trait
    distributions align with empirical norms.

    Args:
        personas: Pool of generated personas (generate more than you need).
        target_norms: Target Big Five norms {trait: {mean, sd}}. Defaults to global norms.
        n_select: How many to select. Defaults to len(personas) // 2.

    Returns:
        Subset of personas aligned to target distribution.
    """
    if len(personas) < 3:
        return personas

    norms = target_norms or POPULATION_BIG_FIVE_NORMS["global"]
    n_select = n_select or max(2, len(personas) // 2)
    n_select = min(n_select, len(personas))

    # Compute importance weights for each persona
    weights = np.zeros(len(personas))
    for i, mp in enumerate(personas):
        bf = mp.persona.psychology.big_five
        log_weight = 0.0
        for trait in BIG_FIVE_TRAITS:
            val = getattr(bf, trait)
            target_mean = norms[trait]["mean"]
            target_sd = norms[trait]["sd"]
            # Gaussian likelihood under target distribution
            log_weight += -0.5 * ((val - target_mean) / target_sd) ** 2
        weights[i] = math.exp(log_weight)

    # Normalize weights to probabilities
    total = weights.sum()
    if total > 0:
        probs = weights / total
    else:
        probs = np.ones(len(personas)) / len(personas)

    # Select without replacement, weighted by alignment to target
    rng = np.random.default_rng(42)
    selected_indices = []
    remaining_probs = probs.copy()

    for _ in range(n_select):
        remaining_sum = remaining_probs.sum()
        if remaining_sum <= 0:
            break
        normalized = remaining_probs / remaining_sum
        idx = rng.choice(len(personas), p=normalized)
        selected_indices.append(idx)
        remaining_probs[idx] = 0  # don't select again

    return [personas[i] for i in sorted(selected_indices)]
