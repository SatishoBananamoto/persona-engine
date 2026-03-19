"""
Export — save generated personas to YAML and JSON formats.

YAML output is compatible with PersonaEngine.from_yaml().
JSON output is a single array of persona dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from layer_zero.models import MintedPersona


def to_yaml(
    personas: list[MintedPersona],
    output_dir: str | Path,
    include_provenance: bool = False,
) -> list[Path]:
    """Export personas as individual YAML files.

    Each file is compatible with PersonaEngine.from_yaml().

    Args:
        personas: List of MintedPersona objects.
        output_dir: Directory to write YAML files into.
        include_provenance: If True, add _provenance section to YAML.

    Returns:
        List of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i, mp in enumerate(personas):
        data = _persona_to_dict(mp.persona)
        if include_provenance:
            data["_provenance"] = _provenance_to_dict(mp.provenance)

        filename = f"persona_{i:03d}_{mp.persona.persona_id[:12]}.yaml"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        paths.append(filepath)

    return paths


def to_json(
    personas: list[MintedPersona],
    output_path: str | Path,
    include_provenance: bool = False,
) -> Path:
    """Export personas as a single JSON file.

    Args:
        personas: List of MintedPersona objects.
        output_path: Path for the JSON file.
        include_provenance: If True, include provenance metadata.

    Returns:
        Written file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for mp in personas:
        entry = _persona_to_dict(mp.persona)
        if include_provenance:
            entry["_provenance"] = _provenance_to_dict(mp.provenance)
        if mp.warnings:
            entry["_warnings"] = mp.warnings
        data.append(entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path


def _persona_to_dict(persona: Any) -> dict:
    """Convert a Persona pydantic model to a plain dict for serialization."""
    if hasattr(persona, "model_dump"):
        data = persona.model_dump()
    elif hasattr(persona, "dict"):
        data = persona.dict()
    else:
        data = dict(persona)

    # Convert tuples to lists for YAML/JSON compatibility
    _convert_tuples(data)
    return data


def _convert_tuples(obj: Any) -> None:
    """Recursively convert tuples to lists in a dict (for YAML/JSON compat)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, tuple):
                obj[key] = list(value)
            elif isinstance(value, (dict, list)):
                _convert_tuples(value)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, tuple):
                obj[i] = list(item)
            elif isinstance(item, (dict, list)):
                _convert_tuples(item)


def _provenance_to_dict(provenance: dict) -> dict:
    """Convert provenance FieldProvenance objects to plain dicts."""
    result = {}
    for key, prov in provenance.items():
        if hasattr(prov, "__dict__"):
            result[key] = {
                "value": str(prov.value) if not isinstance(prov.value, (int, float, str, bool)) else prov.value,
                "source": prov.source,
                "confidence": round(prov.confidence, 4),
                "inferential_depth": prov.inferential_depth,
                "notes": prov.notes,
            }
        else:
            result[key] = str(prov)
    return result
