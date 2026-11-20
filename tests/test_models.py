"""Tests for Layer Zero core models."""

import pytest
from layer_zero.models import (
    MintRequest,
    SegmentRequest,
    FieldProvenance,
    MintedPersona,
    TraitPrior,
    BIG_FIVE_TRAITS,
    SCHWARTZ_VALUES,
    SCHWARTZ_POSITIONS,
    SCHWARTZ_OPPOSING_PAIRS,
    SCHWARTZ_ADJACENT_PAIRS,
    DEPTH_DECAY,
)


# =============================================================================
# MintRequest
# =============================================================================

class TestMintRequest:
    def test_default_construction(self):
        req = MintRequest()
        assert req.age is None
        assert req.occupation is None
        assert req.count == 1
        assert req.big_five_overrides == {}

    def test_full_construction(self):
        req = MintRequest(
            name="Alex",
            age=35,
            occupation="nurse",
            location="Chicago",
            big_five_overrides={"openness": 0.8},
            count=5,
            seed=42,
        )
        assert req.name == "Alex"
        assert req.age == 35
        assert req.count == 5

    def test_age_validation_low(self):
        with pytest.raises(ValueError, match="age must be 18-100"):
            MintRequest(age=10)

    def test_age_validation_high(self):
        with pytest.raises(ValueError, match="age must be 18-100"):
            MintRequest(age=150)

    def test_count_validation(self):
        with pytest.raises(ValueError, match="count must be >= 1"):
            MintRequest(count=0)

    def test_big_five_override_range(self):
        with pytest.raises(ValueError, match="big_five override"):
            MintRequest(big_five_overrides={"openness": 1.5})

    def test_big_five_override_negative(self):
        with pytest.raises(ValueError, match="big_five override"):
            MintRequest(big_five_overrides={"openness": -0.1})

    def test_values_override_range(self):
        with pytest.raises(ValueError, match="values override"):
            MintRequest(values_overrides={"power": 2.0})

    def test_trait_hints(self):
        req = MintRequest(trait_hints=["analytical", "warm", "cautious"])
        assert len(req.trait_hints) == 3

    def test_domains(self):
        req = MintRequest(domains=[{"domain": "Healthcare", "proficiency": 0.5}])
        assert req.domains[0]["domain"] == "Healthcare"


# =============================================================================
# SegmentRequest
# =============================================================================

class TestSegmentRequest:
    def test_default_construction(self):
        req = SegmentRequest()
        assert req.age_range == (25, 55)
        assert req.count == 10

    def test_custom_segment(self):
        req = SegmentRequest(
            segment_name="junior_nurses",
            age_range=(22, 30),
            gender_distribution={"female": 0.8, "male": 0.2},
            occupations=["nurse"],
            count=20,
        )
        assert req.segment_name == "junior_nurses"
        assert req.age_range == (22, 30)

    def test_invalid_age_range(self):
        with pytest.raises(ValueError, match="age_range"):
            SegmentRequest(age_range=(50, 30))

    def test_age_range_too_low(self):
        with pytest.raises(ValueError, match="age_range"):
            SegmentRequest(age_range=(10, 30))

    def test_gender_dist_must_sum_to_one(self):
        with pytest.raises(ValueError, match="gender_distribution"):
            SegmentRequest(gender_distribution={"female": 0.3, "male": 0.3})


# =============================================================================
# FieldProvenance
# =============================================================================

class TestFieldProvenance:
    def test_explicit_confidence(self):
        conf = FieldProvenance.compute_confidence("explicit")
        assert conf == pytest.approx(0.95, abs=0.01)

    def test_sampled_confidence(self):
        conf = FieldProvenance.compute_confidence("sampled")
        assert conf == pytest.approx(0.70, abs=0.01)

    def test_derived_confidence_with_depth(self):
        conf = FieldProvenance.compute_confidence("derived", mapping_strength=0.8, inferential_depth=2)
        expected = 0.60 * 0.8 * (DEPTH_DECAY ** 2)
        assert conf == pytest.approx(expected, abs=0.01)

    def test_default_confidence(self):
        conf = FieldProvenance.compute_confidence("default")
        assert conf == pytest.approx(0.30, abs=0.01)

    def test_depth_decay_reduces_confidence(self):
        c0 = FieldProvenance.compute_confidence("derived", inferential_depth=0)
        c1 = FieldProvenance.compute_confidence("derived", inferential_depth=1)
        c3 = FieldProvenance.compute_confidence("derived", inferential_depth=3)
        assert c0 > c1 > c3

    def test_frozen(self):
        prov = FieldProvenance(value=0.5, source="sampled", confidence=0.7)
        with pytest.raises(AttributeError):
            prov.value = 0.6  # type: ignore

    def test_provenance_with_parents(self):
        prov = FieldProvenance(
            value=0.55,
            source="derived",
            confidence=0.42,
            parent_fields=("psychology.big_five.openness",),
            notes="from O→analytical mapping, r=0.35",
        )
        assert prov.parent_fields == ("psychology.big_five.openness",)
        assert "r=0.35" in prov.notes


# =============================================================================
# TraitPrior
# =============================================================================

class TestTraitPrior:
    def test_basic(self):
        p = TraitPrior(mean=0.5, std_dev=0.15)
        assert p.mean == 0.5
        assert p.std_dev == 0.15

    def test_mean_clamped_near_zero(self):
        p = TraitPrior(mean=0.0, std_dev=0.1)
        assert p.mean >= 0.01  # clamped for logit safety

    def test_mean_clamped_near_one(self):
        p = TraitPrior(mean=1.0, std_dev=0.1)
        assert p.mean <= 0.99  # clamped for logit safety

    def test_std_dev_floor(self):
        p = TraitPrior(mean=0.5, std_dev=0.0)
        assert p.std_dev >= 0.01


# =============================================================================
# Constants
# =============================================================================

class TestConstants:
    def test_big_five_count(self):
        assert len(BIG_FIVE_TRAITS) == 5

    def test_schwartz_count(self):
        assert len(SCHWARTZ_VALUES) == 10

    def test_schwartz_positions_count(self):
        assert len(SCHWARTZ_POSITIONS) == 10

    def test_schwartz_opposing_pairs_exist(self):
        assert len(SCHWARTZ_OPPOSING_PAIRS) >= 8

    def test_schwartz_adjacent_pairs_count(self):
        assert len(SCHWARTZ_ADJACENT_PAIRS) == 10

    def test_schwartz_adjacent_wraps(self):
        # Last value (universalism) is adjacent to first (self_direction)
        assert ("universalism", "self_direction") in SCHWARTZ_ADJACENT_PAIRS
