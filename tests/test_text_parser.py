"""Tests for text description parser."""

import pytest
from layer_zero.parser.text_parser import parse_description


class TestAgeExtraction:
    def test_year_old_pattern(self):
        req = parse_description("A 35-year-old engineer")
        assert req.age == 35

    def test_year_old_with_space(self):
        req = parse_description("A 28 year old nurse")
        assert req.age == 28

    def test_aged_pattern(self):
        req = parse_description("Engineer, aged 42")
        assert req.age == 42

    def test_age_colon_pattern(self):
        req = parse_description("Nurse, age: 30")
        assert req.age == 30

    def test_trailing_age(self):
        req = parse_description("Software engineer, 38, from Seattle")
        assert req.age == 38

    def test_no_age(self):
        req = parse_description("A nurse from Chicago")
        assert req.age is None

    def test_invalid_age_ignored(self):
        req = parse_description("A 12-year-old student")
        assert req.age is None  # below 18


class TestOccupationExtraction:
    def test_single_word(self):
        req = parse_description("A 35-year-old nurse")
        assert req.occupation == "nurse"

    def test_multi_word(self):
        req = parse_description("A product manager in fintech")
        assert req.occupation == "product manager"

    def test_software_engineer(self):
        req = parse_description("Senior software engineer, 42")
        assert req.occupation == "software engineer"

    def test_data_scientist(self):
        req = parse_description("A data scientist who values precision")
        assert req.occupation == "data scientist"

    def test_no_occupation(self):
        req = parse_description("A 35-year-old person from Tokyo")
        assert req.occupation is None


class TestIndustryExtraction:
    def test_fintech(self):
        req = parse_description("Product manager in fintech")
        assert req.industry == "fintech"

    def test_healthcare(self):
        req = parse_description("Nurse working in healthcare")
        assert req.industry == "healthcare"

    def test_no_industry(self):
        req = parse_description("A 35-year-old nurse")
        assert req.industry is None


class TestLocationExtraction:
    def test_from_city(self):
        req = parse_description("A nurse from Chicago")
        assert req.location == "Chicago"

    def test_based_in(self):
        req = parse_description("Engineer based in San Francisco")
        assert req.location == "San Francisco"

    def test_in_city(self):
        req = parse_description("A 30-year-old teacher in London")
        assert req.location == "London"

    def test_no_location(self):
        req = parse_description("A 35-year-old nurse")
        assert req.location is None


class TestGenderExtraction:
    def test_female(self):
        req = parse_description("A female nurse from Chicago")
        assert req.gender == "female"

    def test_male(self):
        req = parse_description("A male engineer, 35")
        assert req.gender == "male"

    def test_no_gender(self):
        req = parse_description("A 35-year-old nurse")
        assert req.gender is None


class TestTraitExtraction:
    def test_single_trait(self):
        req = parse_description("A cautious nurse")
        assert "cautious" in req.trait_hints

    def test_multiple_traits(self):
        req = parse_description("An analytical and creative engineer")
        assert "analytical" in req.trait_hints
        assert "creative" in req.trait_hints

    def test_no_traits(self):
        req = parse_description("A 35-year-old nurse from Chicago")
        assert req.trait_hints == []


class TestGoalExtraction:
    def test_values_keyword(self):
        req = parse_description("A nurse who values patient care")
        assert "patient care" in req.goals

    def test_no_goals(self):
        req = parse_description("A 35-year-old nurse")
        assert req.goals == []


class TestFullDescriptions:
    def test_complex_description(self):
        req = parse_description(
            "A 35-year-old product manager in fintech who values innovation"
        )
        assert req.age == 35
        assert req.occupation == "product manager"
        assert req.industry == "fintech"

    def test_traits_and_location(self):
        req = parse_description(
            "A cautious nurse from Tokyo who values security"
        )
        assert req.occupation == "nurse"
        assert req.location == "Tokyo"
        assert "cautious" in req.trait_hints

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_description("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_description("   ")
