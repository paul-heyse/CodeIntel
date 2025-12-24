"""Golden file helpers for structured outputs."""

from __future__ import annotations

from tests._helpers.goldens.artifact_goldens import (
    assert_json_artifact_matches_golden,
    load_json_artifact,
)
from tests._helpers.goldens.table_goldens import assert_table_matches_golden, dump_table

__all__ = [
    "assert_json_artifact_matches_golden",
    "assert_table_matches_golden",
    "dump_table",
    "load_json_artifact",
]
