"""Tests for YAML and JSON export."""

import json
import os
import tempfile
import pytest
import yaml

from persona_engine import PersonaEngine
import layer_zero
from layer_zero.export import to_yaml, to_json


@pytest.fixture
def sample_personas():
    return layer_zero.mint(occupation="nurse", age=30, count=3, seed=42)


class TestYAMLExport:
    def test_creates_files(self, sample_personas, tmp_path):
        paths = to_yaml(sample_personas, tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".yaml"

    def test_yaml_is_valid(self, sample_personas, tmp_path):
        paths = to_yaml(sample_personas, tmp_path)
        for p in paths:
            with open(p) as f:
                data = yaml.safe_load(f)
            assert "persona_id" in data
            assert "identity" in data
            assert "psychology" in data

    def test_yaml_loads_in_engine(self, sample_personas, tmp_path):
        paths = to_yaml(sample_personas, tmp_path)
        for p in paths:
            engine = PersonaEngine.from_yaml(str(p), llm_provider="template")
            ir = engine.plan("Hello")
            assert ir.response_structure.confidence > 0

    def test_yaml_with_provenance(self, sample_personas, tmp_path):
        paths = to_yaml(sample_personas, tmp_path, include_provenance=True)
        with open(paths[0]) as f:
            data = yaml.safe_load(f)
        assert "_provenance" in data


class TestJSONExport:
    def test_creates_file(self, sample_personas, tmp_path):
        path = to_json(sample_personas, tmp_path / "personas.json")
        assert path.exists()

    def test_json_is_valid(self, sample_personas, tmp_path):
        path = to_json(sample_personas, tmp_path / "personas.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 3
        assert "persona_id" in data[0]

    def test_json_round_trip(self, sample_personas, tmp_path):
        path = to_json(sample_personas, tmp_path / "personas.json")
        with open(path) as f:
            data = json.load(f)
        # Check key fields survived serialization
        for entry in data:
            assert "identity" in entry
            assert "psychology" in entry
            assert entry["psychology"]["big_five"]["openness"] > 0

    def test_json_with_provenance(self, sample_personas, tmp_path):
        path = to_json(sample_personas, tmp_path / "personas.json", include_provenance=True)
        with open(path) as f:
            data = json.load(f)
        assert "_provenance" in data[0]
