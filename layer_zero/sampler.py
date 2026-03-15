"""
Logit-normal sampler for correlated bounded traits.

Samples Big Five trait vectors in unbounded logit space using multivariate
normal distribution, then applies sigmoid to map back to [0, 1]. This
preserves the correlation structure at boundaries (unlike clamp-based approaches).

The correlation matrix is from van der Linden et al. (2010), K=212, N=144,117.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from layer_zero.models import BIG_FIVE_TRAITS, TraitPrior

# =============================================================================
# Load correlation matrix
# =============================================================================

_DATA_DIR = Path(__file__).parent / "priors" / "data"


def _load_correlation_matrix() -> np.ndarray:
    with open(_DATA_DIR / "correlation_matrix.json") as f:
        data = json.load(f)
    return np.array(data["matrix"], dtype=np.float64)


DEFAULT_CORRELATION = _load_correlation_matrix()


# =============================================================================
# Logit-normal transforms
# =============================================================================

def _logit(p: float | np.ndarray) -> float | np.ndarray:
    """Logit transform: p ∈ (0,1) → x ∈ (-∞, +∞)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid (inverse logit): x ∈ (-∞, +∞) → p ∈ (0,1)."""
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# Sampler
# =============================================================================

def sample_big_five(
    priors: dict[str, TraitPrior],
    count: int = 1,
    seed: int | None = None,
    correlation: np.ndarray | None = None,
) -> np.ndarray:
    """Sample Big Five trait vectors using logit-normal multivariate distribution.

    Args:
        priors: {trait_name: TraitPrior} with mean and std_dev in [0,1] space.
        count: Number of trait vectors to generate.
        seed: Random seed for reproducibility.
        correlation: 5×5 correlation matrix. Defaults to van der Linden (2010).

    Returns:
        np.ndarray of shape (count, 5) with values in [0, 1].
        Column order matches BIG_FIVE_TRAITS.
    """
    if correlation is None:
        correlation = DEFAULT_CORRELATION

    rng = np.random.default_rng(seed)

    # Build mean and SD vectors in [0,1] space (trait order)
    means_01 = np.array([priors[t].mean for t in BIG_FIVE_TRAITS], dtype=np.float64)
    sds_01 = np.array([priors[t].std_dev for t in BIG_FIVE_TRAITS], dtype=np.float64)

    # Transform to logit space
    logit_means = _logit(means_01)

    # Transform SDs to logit space (delta method: Jacobian of logit at the mean)
    # d(logit)/dp = 1 / (p * (1-p)), so logit_sd ≈ sd_01 / (mean * (1-mean))
    # NOTE: This approximation blows up near 0 and 1. We cap logit_sd to prevent
    # pathological sampling at boundary means. Cap of 3.0 keeps 99.7% of samples
    # within ~6 logit units, which maps to roughly [0.002, 0.998] in output space.
    MAX_LOGIT_SD = 3.0
    jacobian = 1.0 / (means_01 * (1.0 - means_01))
    logit_sds = np.minimum(sds_01 * jacobian, MAX_LOGIT_SD)

    # Build covariance matrix in logit space: Cov = diag(SD) @ Corr @ diag(SD)
    # NOTE: Using raw-space correlation as logit-space correlation is an approximation.
    # The nonlinear sigmoid transform changes correlation structure, but for moderate
    # correlations (max |r|=0.43) the distortion is small. Documented trade-off for v1.
    logit_cov = np.diag(logit_sds) @ correlation @ np.diag(logit_sds)

    # Ensure positive semi-definite (numerical safety)
    logit_cov = _nearest_positive_semidefinite(logit_cov)

    # Sample in logit space
    logit_samples = rng.multivariate_normal(logit_means, logit_cov, size=count)

    # Map back to [0, 1] via sigmoid
    samples_01 = _sigmoid(logit_samples)

    return samples_01


def _nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Project matrix to nearest positive semi-definite matrix.

    Uses eigenvalue thresholding: negative eigenvalues are set to a small positive.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
