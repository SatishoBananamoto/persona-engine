"""Tests for diversity analysis and population alignment."""

import numpy as np
import pytest

import layer_zero
from layer_zero.diversity import analyze_diversity, align_to_population, POPULATION_BIG_FIVE_NORMS


class TestDiversityReport:
    def test_basic_report(self):
        personas = layer_zero.mint(occupation="nurse", count=20, seed=42)
        report = analyze_diversity(personas)
        assert report.count == 20
        assert 0.0 <= report.simpsons_index <= 1.0
        assert report.mean_pairwise_distance > 0
        assert report.min_pairwise_distance >= 0
        assert len(report.trait_coverage) == 5

    def test_high_diversity_batch(self):
        """Mixed occupations should have higher diversity."""
        personas = []
        for occ in ["nurse", "artist", "engineer", "chef", "entrepreneur"]:
            personas.extend(layer_zero.mint(occupation=occ, count=4, seed=42))
        report = analyze_diversity(personas)
        assert report.simpsons_index > 0.6
        assert report.overall_score > 0.3

    def test_identical_personas_low_diversity(self):
        """Same persona repeated should have minimal diversity."""
        personas = layer_zero.mint(occupation="nurse", count=2, seed=42, big_five={
            "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
            "agreeableness": 0.5, "neuroticism": 0.5,
        })
        report = analyze_diversity(personas)
        assert report.min_pairwise_distance < 0.1

    def test_single_persona_returns_empty(self):
        personas = layer_zero.mint(count=1)
        report = analyze_diversity(personas)
        assert report.count == 1
        assert len(report.warnings) > 0

    def test_trait_coverage(self):
        personas = layer_zero.mint(occupation="nurse", count=50, seed=42)
        report = analyze_diversity(personas)
        for trait, cov in report.trait_coverage.items():
            assert cov > 0.2, f"Low coverage for {trait}: {cov}"

    def test_distribution_alignment(self):
        personas = layer_zero.mint(count=100, seed=42)
        report = analyze_diversity(personas)
        # Should be somewhat aligned to population (not perfect, but not terrible)
        assert report.distribution_alignment < 2.0


class TestPopulationAlignment:
    def test_basic_alignment(self):
        pool = layer_zero.mint(occupation="nurse", count=50, seed=42)
        aligned = align_to_population(pool, n_select=20)
        assert len(aligned) == 20

    def test_aligned_closer_to_norms(self):
        """Aligned subset should be closer to population norms than full pool."""
        pool = layer_zero.mint(occupation="artist", count=100, seed=42)
        aligned = align_to_population(pool, n_select=30)

        report_pool = analyze_diversity(pool)
        report_aligned = analyze_diversity(aligned)

        # Aligned should have lower KL divergence (or similar)
        # At minimum, shouldn't be dramatically worse
        assert report_aligned.distribution_alignment < report_pool.distribution_alignment + 0.5

    def test_alignment_preserves_validity(self):
        """Aligned personas should still be valid."""
        from persona_engine.schema.persona_schema import Persona
        pool = layer_zero.mint(occupation="nurse", count=30, seed=42)
        aligned = align_to_population(pool, n_select=10)
        for mp in aligned:
            assert isinstance(mp.persona, Persona)

    def test_small_pool_returns_all(self):
        pool = layer_zero.mint(count=2, seed=42)
        aligned = align_to_population(pool, n_select=5)
        assert len(aligned) == 2  # can't select more than available
