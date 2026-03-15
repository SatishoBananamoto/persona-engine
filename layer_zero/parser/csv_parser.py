"""
CSV segment parser — Tier 3 input.

Parses CSV files where each row defines a segment with ranges and distributions.
Generates MintRequest objects by sampling from segment specifications.

Expected CSV columns (all optional except segment_name):
    segment_name, age_min, age_max, occupation, location, gender_dist, count
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from layer_zero.models import MintRequest, SegmentRequest


def parse_csv(path: str | Path) -> list[SegmentRequest]:
    """Parse a CSV file into SegmentRequest objects.

    Each row becomes one SegmentRequest. Missing columns get defaults.

    Args:
        path: Path to CSV file.

    Returns:
        List of SegmentRequest objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    segments: list[SegmentRequest] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # 2 = first data row
            try:
                segments.append(_row_to_segment(row, row_num))
            except ValueError as e:
                raise ValueError(f"CSV row {row_num}: {e}") from e

    if not segments:
        raise ValueError("CSV file contains no data rows")

    return segments


def segment_to_mint_requests(
    segment: SegmentRequest,
    seed: int = 42,
) -> list[MintRequest]:
    """Convert a SegmentRequest into individual MintRequests by sampling.

    Samples age from range, gender from distribution, picks occupations.

    Args:
        segment: SegmentRequest with ranges and distributions.
        seed: Random seed for reproducibility.

    Returns:
        List of MintRequest objects (one per persona in segment).
    """
    rng = np.random.default_rng(seed)
    requests: list[MintRequest] = []

    for i in range(segment.count):
        # Sample age uniformly from range
        age = int(rng.integers(segment.age_range[0], segment.age_range[1] + 1))

        # Sample gender from distribution
        gender = None
        if segment.gender_distribution:
            genders = list(segment.gender_distribution.keys())
            probs = list(segment.gender_distribution.values())
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]  # normalize
                gender = rng.choice(genders, p=probs)

        # Pick occupation (round-robin if multiple)
        occupation = None
        if segment.occupations:
            occupation = segment.occupations[i % len(segment.occupations)]

        requests.append(MintRequest(
            age=age,
            gender=gender,
            occupation=occupation,
            location=segment.location,
            culture_region=segment.culture_region,
            count=1,
            seed=seed + i,
        ))

    return requests


def _row_to_segment(row: dict[str, str], row_num: int) -> SegmentRequest:
    """Convert a CSV row dict into a SegmentRequest."""
    segment_name = row.get("segment_name", "").strip() or f"segment_{row_num}"

    # Age range
    age_min = int(row.get("age_min", "25").strip() or "25")
    age_max = int(row.get("age_max", "55").strip() or "55")

    # Occupation(s) — comma-separated
    occ_str = row.get("occupation", "").strip()
    occupations = [o.strip() for o in occ_str.split(",") if o.strip()] if occ_str else []

    # Location
    location = row.get("location", "").strip() or None

    # Culture region
    culture_region = row.get("culture_region", "").strip() or None

    # Gender distribution — format: "female:0.6,male:0.4" or just ignored
    gender_dist: dict[str, float] = {}
    gender_str = row.get("gender_dist", "").strip()
    if gender_str:
        # Support both space-separated and semicolon-separated pairs
        # (comma is the CSV delimiter, so we use space or semicolon)
        for pair in gender_str.replace(";", " ").split():
            parts = pair.strip().split(":")
            if len(parts) == 2:
                gender_dist[parts[0].strip()] = float(parts[1].strip())
    if not gender_dist:
        gender_dist = {"female": 0.5, "male": 0.5}

    # Count
    count = int(row.get("count", "10").strip() or "10")

    return SegmentRequest(
        segment_name=segment_name,
        age_range=(age_min, age_max),
        gender_distribution=gender_dist,
        occupations=occupations,
        location=location,
        culture_region=culture_region,
        count=count,
    )
