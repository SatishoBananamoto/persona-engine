"""Tests for Big Five prior engine."""

import pytest
from layer_zero.models import MintRequest, BIG_FIVE_TRAITS
from layer_zero.priors.big_five import (
    compute_big_five_prior,
    normalize_occupation,
    infer_culture_region,
    BASELINE_MEAN,
    BASELINE_SD,
    OVERRIDE_SD,
)


class TestNormalizeOccupation:
    def test_exact_match(self):
        assert normalize_occupation("nurse") == "nurse"

    def test_case_insensitive(self):
        assert normalize_occupation("Nurse") == "nurse"

    def test_partial_match(self):
        assert normalize_occupation("senior software engineer") == "software engineer"

    def test_unknown(self):
        assert normalize_occupation("astronaut") is None

    def test_whitespace(self):
        assert normalize_occupation("  nurse  ") == "nurse"


class TestInferCultureRegion:
    def test_us_locations(self):
        assert infer_culture_region("San Francisco, US") == "western"
        assert infer_culture_region("New York, USA") == "western"
        assert infer_culture_region("Chicago, United States") == "western"

    def test_european(self):
        assert infer_culture_region("London, UK") == "western"
        assert infer_culture_region("Berlin, Germany") == "western"

    def test_east_asian(self):
        assert infer_culture_region("Tokyo, Japan") == "east_asian"
        assert infer_culture_region("Seoul, Korea") == "east_asian"

    def test_south_asian(self):
        assert infer_culture_region("Mumbai, India") == "south_asian"

    def test_latin_american(self):
        assert infer_culture_region("Sao Paulo, Brazil") == "latin_american"

    def test_unknown(self):
        assert infer_culture_region("Mars") is None


class TestComputeBigFivePrior:
    def test_baseline_defaults(self):
        req = MintRequest()
        priors = compute_big_five_prior(req)
        for trait in BIG_FIVE_TRAITS:
            assert priors[trait].mean == pytest.approx(BASELINE_MEAN, abs=0.01)
            assert priors[trait].std_dev == pytest.approx(BASELINE_SD, abs=0.01)

    def test_occupation_shifts_nurse(self):
        req = MintRequest(occupation="nurse")
        priors = compute_big_five_prior(req)
        # Nurse: Social RIASEC → A should be above baseline
        assert priors["agreeableness"].mean > BASELINE_MEAN
        assert priors["agreeableness"].source == "occupation"

    def test_occupation_shifts_entrepreneur(self):
        req = MintRequest(occupation="entrepreneur")
        priors = compute_big_five_prior(req)
        # Entrepreneur: N should be below baseline, E above
        assert priors["neuroticism"].mean < BASELINE_MEAN
        assert priors["extraversion"].mean > BASELINE_MEAN

    def test_occupation_shifts_artist(self):
        req = MintRequest(occupation="artist")
        priors = compute_big_five_prior(req)
        # Artist: O should be well above baseline
        assert priors["openness"].mean > BASELINE_MEAN + 0.1

    def test_age_shifts_young(self):
        req = MintRequest(age=22)
        priors = compute_big_five_prior(req)
        # Young: higher N, lower C, higher O
        assert priors["neuroticism"].mean > BASELINE_MEAN
        assert priors["conscientiousness"].mean < BASELINE_MEAN

    def test_age_shifts_old(self):
        req = MintRequest(age=65)
        priors = compute_big_five_prior(req)
        # Old: higher A, higher C, lower N
        assert priors["agreeableness"].mean > BASELINE_MEAN
        assert priors["conscientiousness"].mean > BASELINE_MEAN
        assert priors["neuroticism"].mean < BASELINE_MEAN

    def test_culture_shifts(self):
        req = MintRequest(location="Tokyo, Japan")
        priors = compute_big_five_prior(req)
        # East Asian: lower E
        assert priors["extraversion"].mean < BASELINE_MEAN

    def test_explicit_culture_region(self):
        req = MintRequest(culture_region="east_asian")
        priors = compute_big_five_prior(req)
        assert priors["extraversion"].mean < BASELINE_MEAN

    def test_override_replaces_prior(self):
        req = MintRequest(occupation="nurse", big_five_overrides={"openness": 0.9})
        priors = compute_big_five_prior(req)
        assert priors["openness"].mean == pytest.approx(0.9, abs=0.02)
        assert priors["openness"].std_dev == pytest.approx(OVERRIDE_SD, abs=0.01)
        assert priors["openness"].source == "override"

    def test_override_does_not_affect_other_traits(self):
        req = MintRequest(occupation="nurse", big_five_overrides={"openness": 0.9})
        priors = compute_big_five_prior(req)
        # Agreeableness should still be shifted by occupation
        assert priors["agreeableness"].mean > BASELINE_MEAN

    def test_trait_hints_shift(self):
        req = MintRequest(trait_hints=["analytical", "warm"])
        priors = compute_big_five_prior(req)
        assert priors["openness"].mean > BASELINE_MEAN  # analytical → O+
        assert priors["agreeableness"].mean > BASELINE_MEAN  # warm → A+

    def test_unknown_occupation_uses_baseline(self):
        req = MintRequest(occupation="astronaut")
        priors = compute_big_five_prior(req)
        # No shifts applied — should be near baseline
        for trait in BIG_FIVE_TRAITS:
            assert abs(priors[trait].mean - BASELINE_MEAN) < 0.02

    def test_all_signals_combine(self):
        req = MintRequest(
            occupation="nurse",
            age=60,
            location="London, UK",
            trait_hints=["warm"],
        )
        priors = compute_big_five_prior(req)
        # Agreeableness should be boosted by: occupation + age + trait hint
        assert priors["agreeableness"].mean > BASELINE_MEAN + 0.15

    def test_all_traits_present(self):
        req = MintRequest(occupation="nurse")
        priors = compute_big_five_prior(req)
        for trait in BIG_FIVE_TRAITS:
            assert trait in priors
