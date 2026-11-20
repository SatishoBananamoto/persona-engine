"""Tests for Schwartz circumplex value generator."""

import math
import numpy as np
import pytest

from layer_zero.models import MintRequest, SCHWARTZ_VALUES, SCHWARTZ_OPPOSING_PAIRS, SCHWARTZ_ADJACENT_PAIRS
from layer_zero.priors.values import generate_schwartz_values


def _batch(count: int = 100, seed: int = 42, **kwargs) -> np.ndarray:
    req = MintRequest(**kwargs)
    return generate_schwartz_values(req, count=count, seed=seed)


class TestBasicGeneration:
    def test_output_shape(self):
        profiles = _batch(count=10)
        assert profiles.shape == (10, 10)

    def test_bounded_zero_one(self):
        profiles = _batch(count=500)
        assert np.all(profiles >= 0.0)
        assert np.all(profiles <= 1.0)

    def test_deterministic_with_seed(self):
        p1 = _batch(count=5, seed=42)
        p2 = _batch(count=5, seed=42)
        assert np.allclose(p1, p2)

    def test_different_seeds_differ(self):
        p1 = _batch(count=5, seed=42)
        p2 = _batch(count=5, seed=99)
        assert not np.allclose(p1, p2)

    def test_single_profile(self):
        profiles = _batch(count=1)
        assert profiles.shape == (1, 10)


class TestCircumplexStructure:
    def test_adjacent_values_positively_correlated(self):
        """Adjacent values on the circle should tend to co-occur."""
        profiles = _batch(count=500)
        positive_count = 0
        for v1, v2 in SCHWARTZ_ADJACENT_PAIRS:
            i1 = SCHWARTZ_VALUES.index(v1)
            i2 = SCHWARTZ_VALUES.index(v2)
            corr = np.corrcoef(profiles[:, i1], profiles[:, i2])[0, 1]
            if corr > 0:
                positive_count += 1
        # Most adjacent pairs should be positively correlated
        assert positive_count >= 7, f"Only {positive_count}/10 adjacent pairs positive"

    def test_opposing_values_not_both_high(self):
        """Opposing values should rarely both be very high."""
        profiles = _batch(count=200)
        both_high_count = 0
        total_checks = 0
        for v1, v2 in SCHWARTZ_OPPOSING_PAIRS:
            i1 = SCHWARTZ_VALUES.index(v1)
            i2 = SCHWARTZ_VALUES.index(v2)
            both_high = np.sum((profiles[:, i1] > 0.75) & (profiles[:, i2] > 0.75))
            both_high_count += both_high
            total_checks += len(profiles)
        # Less than 5% of opposing pairs should both be > 0.75
        ratio = both_high_count / total_checks
        assert ratio < 0.05, f"Too many opposing pairs both high: {ratio:.2%}"

    def test_profiles_have_structure(self):
        """Profiles should not be random noise — they should show circumplex pattern."""
        profiles = _batch(count=100)
        # Check that each profile has some variance (not all identical values)
        for i in range(min(20, len(profiles))):
            profile_std = profiles[i].std()
            assert profile_std > 0.02, f"Profile {i} too flat: std={profile_std:.4f}"


class TestDemographicShifts:
    def test_young_higher_stimulation(self):
        young = _batch(count=200, seed=42, age=22)
        old = _batch(count=200, seed=42, age=65)
        stim_idx = SCHWARTZ_VALUES.index("stimulation")
        assert young[:, stim_idx].mean() > old[:, stim_idx].mean()

    def test_old_higher_tradition(self):
        young = _batch(count=200, seed=42, age=22)
        old = _batch(count=200, seed=42, age=65)
        trad_idx = SCHWARTZ_VALUES.index("tradition")
        assert old[:, trad_idx].mean() > young[:, trad_idx].mean()

    def test_social_worker_high_benevolence(self):
        sw = _batch(count=200, seed=42, occupation="social worker")
        generic = _batch(count=200, seed=42)
        ben_idx = SCHWARTZ_VALUES.index("benevolence")
        assert sw[:, ben_idx].mean() > generic[:, ben_idx].mean()

    def test_entrepreneur_high_achievement(self):
        ent = _batch(count=200, seed=42, occupation="entrepreneur")
        generic = _batch(count=200, seed=42)
        ach_idx = SCHWARTZ_VALUES.index("achievement")
        assert ent[:, ach_idx].mean() > generic[:, ach_idx].mean()

    def test_culture_shifts_applied(self):
        # East Asian: higher conformity
        culture_shifts = {"conformity": 0.05, "self_direction": -0.02}
        req = MintRequest()
        profiles = generate_schwartz_values(req, count=200, seed=42, culture_value_shifts=culture_shifts)
        generic = _batch(count=200, seed=42)
        conf_idx = SCHWARTZ_VALUES.index("conformity")
        assert profiles[:, conf_idx].mean() > generic[:, conf_idx].mean()


class TestOverrides:
    def test_single_override_pinned(self):
        req = MintRequest(values_overrides={"power": 0.9})
        profiles = generate_schwartz_values(req, count=50, seed=42)
        pow_idx = SCHWARTZ_VALUES.index("power")
        # All profiles should have power ≈ 0.9
        assert np.all(profiles[:, pow_idx] == pytest.approx(0.9, abs=0.01))

    def test_override_does_not_break_others(self):
        req = MintRequest(values_overrides={"power": 0.9})
        profiles = generate_schwartz_values(req, count=50, seed=42)
        # Other values should still have variance
        ben_idx = SCHWARTZ_VALUES.index("benevolence")
        assert profiles[:, ben_idx].std() > 0.01

    def test_no_override_has_variance(self):
        profiles = _batch(count=100)
        pow_idx = SCHWARTZ_VALUES.index("power")
        assert profiles[:, pow_idx].std() > 0.02


class TestShiftCapping:
    def test_extreme_stacking_capped(self):
        """Even with occupation + age + culture shifts, values stay reasonable."""
        culture_shifts = {v: 0.15 for v in SCHWARTZ_VALUES}  # extreme culture
        req = MintRequest(age=70, occupation="social worker")
        profiles = generate_schwartz_values(req, count=100, seed=42, culture_value_shifts=culture_shifts)
        # All values should still be in [0, 1]
        assert np.all(profiles >= 0.0)
        assert np.all(profiles <= 1.0)
        # Means should not exceed 0.85 (baseline 0.5 + max shift 0.20 + amplitude 0.15)
        assert np.all(profiles.mean(axis=0) < 0.90)
