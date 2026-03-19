"""Tests for logit-normal Big Five sampler."""

import numpy as np
import pytest

from layer_zero.models import BIG_FIVE_TRAITS, TraitPrior
from layer_zero.sampler import sample_big_five, DEFAULT_CORRELATION, _logit, _sigmoid


def _default_priors(mean: float = 0.5, sd: float = 0.15) -> dict[str, TraitPrior]:
    return {t: TraitPrior(mean=mean, std_dev=sd) for t in BIG_FIVE_TRAITS}


class TestTransforms:
    def test_logit_sigmoid_inverse(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert _sigmoid(_logit(p)) == pytest.approx(p, abs=1e-6)

    def test_logit_center(self):
        assert _logit(0.5) == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_center(self):
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)

    def test_logit_extreme_clipped(self):
        # Should not crash on 0 or 1
        result = _logit(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(result))


class TestCorrelationMatrix:
    def test_shape(self):
        assert DEFAULT_CORRELATION.shape == (5, 5)

    def test_symmetric(self):
        assert np.allclose(DEFAULT_CORRELATION, DEFAULT_CORRELATION.T)

    def test_diagonal_ones(self):
        assert np.allclose(np.diag(DEFAULT_CORRELATION), 1.0)

    def test_positive_semidefinite(self):
        eigenvalues = np.linalg.eigvalsh(DEFAULT_CORRELATION)
        assert np.all(eigenvalues >= -1e-10)


class TestSampling:
    def test_output_shape(self):
        priors = _default_priors()
        samples = sample_big_five(priors, count=10, seed=42)
        assert samples.shape == (10, 5)

    def test_bounded_zero_one(self):
        priors = _default_priors()
        samples = sample_big_five(priors, count=1000, seed=42)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_deterministic_with_seed(self):
        priors = _default_priors()
        s1 = sample_big_five(priors, count=5, seed=42)
        s2 = sample_big_five(priors, count=5, seed=42)
        assert np.allclose(s1, s2)

    def test_different_seeds_differ(self):
        priors = _default_priors()
        s1 = sample_big_five(priors, count=5, seed=42)
        s2 = sample_big_five(priors, count=5, seed=99)
        assert not np.allclose(s1, s2)

    def test_mean_approximately_correct(self):
        priors = _default_priors(mean=0.5)
        samples = sample_big_five(priors, count=2000, seed=42)
        sample_means = samples.mean(axis=0)
        for i, trait in enumerate(BIG_FIVE_TRAITS):
            assert sample_means[i] == pytest.approx(0.5, abs=0.05), f"{trait} mean off"

    def test_high_mean_stays_high(self):
        priors = _default_priors(mean=0.8, sd=0.10)
        samples = sample_big_five(priors, count=500, seed=42)
        sample_means = samples.mean(axis=0)
        for i, trait in enumerate(BIG_FIVE_TRAITS):
            assert sample_means[i] > 0.7, f"{trait} mean too low for high prior"

    def test_low_mean_stays_low(self):
        priors = _default_priors(mean=0.2, sd=0.10)
        samples = sample_big_five(priors, count=500, seed=42)
        sample_means = samples.mean(axis=0)
        for i, trait in enumerate(BIG_FIVE_TRAITS):
            assert sample_means[i] < 0.3, f"{trait} mean too high for low prior"

    def test_override_tiny_sd(self):
        """Overridden traits (tiny SD) should have near-zero variance."""
        priors = _default_priors()
        priors["openness"] = TraitPrior(mean=0.9, std_dev=0.02, source="override")
        samples = sample_big_five(priors, count=100, seed=42)
        openness_std = samples[:, 0].std()
        assert openness_std < 0.05, f"Override trait should have low variance, got {openness_std}"

    def test_correlation_preserved(self):
        """Sample correlations should approximately match input correlations."""
        priors = _default_priors()
        samples = sample_big_five(priors, count=5000, seed=42)
        sample_corr = np.corrcoef(samples.T)

        # Check strongest correlations: E-O (0.43) and C-A (0.43)
        e_idx, o_idx = 2, 0  # extraversion, openness
        c_idx, a_idx = 1, 3  # conscientiousness, agreeableness
        assert sample_corr[e_idx, o_idx] > 0.2, "E-O correlation too low"
        assert sample_corr[c_idx, a_idx] > 0.2, "C-A correlation too low"

        # Check N is negatively correlated with C and E
        n_idx = 4
        assert sample_corr[n_idx, c_idx] < 0, "N-C should be negative"
        assert sample_corr[n_idx, e_idx] < 0, "N-E should be negative"

    def test_single_sample(self):
        priors = _default_priors()
        samples = sample_big_five(priors, count=1, seed=42)
        assert samples.shape == (1, 5)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)
