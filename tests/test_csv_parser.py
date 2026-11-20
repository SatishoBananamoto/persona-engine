"""Tests for CSV segment parser and from_csv pipeline."""

import os
import tempfile
import pytest
import numpy as np

from layer_zero.parser.csv_parser import parse_csv, segment_to_mint_requests
from layer_zero.models import SegmentRequest
import layer_zero


TEST_CSV = os.path.join(os.path.dirname(__file__), "test_segments.csv")


class TestParseCSV:
    def test_parses_test_file(self):
        segments = parse_csv(TEST_CSV)
        assert len(segments) == 3

    def test_segment_names(self):
        segments = parse_csv(TEST_CSV)
        names = [s.segment_name for s in segments]
        assert "junior_nurses" in names
        assert "senior_engineers" in names

    def test_age_ranges(self):
        segments = parse_csv(TEST_CSV)
        nurses = [s for s in segments if s.segment_name == "junior_nurses"][0]
        assert nurses.age_range == (22, 30)

    def test_occupations(self):
        segments = parse_csv(TEST_CSV)
        engineers = [s for s in segments if s.segment_name == "senior_engineers"][0]
        assert "software engineer" in engineers.occupations

    def test_gender_distribution(self):
        segments = parse_csv(TEST_CSV)
        nurses = [s for s in segments if s.segment_name == "junior_nurses"][0]
        assert nurses.gender_distribution.get("female", 0) > 0.7

    def test_count(self):
        segments = parse_csv(TEST_CSV)
        assert all(s.count == 5 for s in segments)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_csv("nonexistent.csv")

    def test_empty_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("segment_name,age_min,age_max\n")  # headers only
            f.flush()
            try:
                with pytest.raises(ValueError, match="no data"):
                    parse_csv(f.name)
            finally:
                os.unlink(f.name)


class TestSegmentToMintRequests:
    def test_generates_correct_count(self):
        seg = SegmentRequest(occupations=["nurse"], count=10)
        reqs = segment_to_mint_requests(seg, seed=42)
        assert len(reqs) == 10

    def test_ages_within_range(self):
        seg = SegmentRequest(age_range=(25, 35), occupations=["nurse"], count=20)
        reqs = segment_to_mint_requests(seg, seed=42)
        for req in reqs:
            assert 25 <= req.age <= 35

    def test_gender_sampled(self):
        seg = SegmentRequest(
            gender_distribution={"female": 0.8, "male": 0.2},
            occupations=["nurse"], count=50,
        )
        reqs = segment_to_mint_requests(seg, seed=42)
        genders = [r.gender for r in reqs]
        female_ratio = genders.count("female") / len(genders)
        assert 0.5 < female_ratio < 1.0  # should be ~80%

    def test_occupation_round_robin(self):
        seg = SegmentRequest(occupations=["nurse", "doctor"], count=4)
        reqs = segment_to_mint_requests(seg, seed=42)
        occs = [r.occupation for r in reqs]
        assert occs == ["nurse", "doctor", "nurse", "doctor"]


class TestFromCSV:
    def test_from_csv_produces_personas(self):
        personas = layer_zero.from_csv(TEST_CSV, seed=42)
        assert len(personas) == 15  # 3 segments × 5 each

    def test_from_csv_all_valid(self):
        from persona_engine.schema.persona_schema import Persona
        personas = layer_zero.from_csv(TEST_CSV, seed=42)
        for mp in personas:
            assert isinstance(mp.persona, Persona)

    def test_from_csv_ages_in_range(self):
        personas = layer_zero.from_csv(TEST_CSV, seed=42)
        # First 5 are junior nurses (22-30)
        for mp in personas[:5]:
            assert 22 <= mp.persona.identity.age <= 30

    def test_from_csv_deterministic(self):
        p1 = layer_zero.from_csv(TEST_CSV, seed=42)
        p2 = layer_zero.from_csv(TEST_CSV, seed=42)
        for a, b in zip(p1, p2):
            assert a.persona_id == b.persona_id
