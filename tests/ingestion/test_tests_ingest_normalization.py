"""Normalization of pytest JSON ingest inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.ingestion.tests_ingest import load_tests_from_report


def test_load_tests_from_report_normalizes_entries(tmp_path: Path) -> None:
    """Entries without nodeid are skipped; valid entries are normalized."""
    duration = 0.5
    call_duration = 0.1
    payload = {
        "tests": [
            {"keywords": {"slow": True}},  # missing nodeid -> skipped
            {
                "nodeid": "tests/test_app.py::test_ok",
                "keywords": {"slow": True, "skip": False},
                "duration": str(duration),
                "call": {"duration": str(call_duration)},
            },
        ]
    }
    report_path = tmp_path / "pytest-report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    entries = load_tests_from_report(report_path)
    if len(entries) != 1:
        pytest.fail("Invalid pytest entries should be filtered out")

    entry = entries[0]
    expected = {
        "nodeid": "tests/test_app.py::test_ok",
        "keywords": ["slow"],
        "duration": duration,
    }
    for key, expected_value in expected.items():
        if entry.get(key) != expected_value:
            pytest.fail(f"{key} not normalized as expected")

    call = entry.get("call")
    if not isinstance(call, dict):
        pytest.fail("Call object should be present")
    if call.get("duration") != call_duration:
        pytest.fail("Call duration was not coerced to float")
