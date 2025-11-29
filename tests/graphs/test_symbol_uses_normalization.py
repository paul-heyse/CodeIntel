"""Normalization behavior for SCIP document loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.graphs.symbol_uses import load_scip_documents


def test_load_scip_documents_skips_invalid_entries(tmp_path: Path) -> None:
    """Invalid SCIP docs should be skipped while valid ones load."""
    data = [
        {"relative_path": "pkg/a.py", "occurrences": [{"symbol": "x#def", "symbol_roles": "1"}]},
        {"occurrences": [{"symbol": "missing_path"}]},
        {"relative_path": "pkg/b.py", "occurrences": [{"symbol_roles": 2}]},
    ]
    scip_path = tmp_path / "index.scip.json"
    scip_path.write_text(json.dumps(data), encoding="utf-8")

    docs = load_scip_documents(scip_path)
    if docs is None:
        pytest.fail("SCIP documents should be parsed")
    if len(docs) != 1:
        pytest.fail("Invalid SCIP docs were not filtered out")
    first = docs[0]
    if "relative_path" not in first:
        pytest.fail("Normalized SCIP doc missing relative_path")
    if first["relative_path"] != "pkg/a.py":
        pytest.fail("Valid SCIP doc path not preserved")
    occurrences = first.get("occurrences", [])
    if not occurrences:
        pytest.fail("Occurrences missing after normalization")
    occurrence = occurrences[0]
    if "symbol" not in occurrence:
        pytest.fail("Occurrence symbol missing after normalization")
    if occurrence["symbol"] != "x#def":
        pytest.fail("Valid occurrence was altered during normalization")
