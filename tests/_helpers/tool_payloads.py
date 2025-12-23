"""Payload builders for deterministic tool outputs in tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def pytest_report_payload(
    *,
    tests: Iterable[Mapping[str, object]] | None = None,
    summary: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Build a minimal pytest JSON report payload.

    Parameters
    ----------
    tests
        Test entry payloads to include in the report.
    summary
        Optional summary counts.

    Returns
    -------
    dict[str, object]
        Pytest JSON report payload.
    """
    return {
        "created": "1970-01-01T00:00:00Z",
        "duration": 0.0,
        "exitcode": 0,
        "summary": dict(summary or {"passed": 1, "failed": 0, "skipped": 0}),
        "tests": list(
            tests
            or [
                {
                    "nodeid": "tests/test_example.py::test_example",
                    "outcome": "passed",
                    "duration": 0.01,
                }
            ]
        ),
    }


def coverage_json_payload(
    *,
    files: Mapping[str, Mapping[str, list[int]]] | None = None,
) -> dict[str, object]:
    """Build a minimal coverage JSON payload.

    Parameters
    ----------
    files
        Mapping of filename to executed/missing line lists.

    Returns
    -------
    dict[str, object]
        Coverage JSON payload.
    """
    return {
        "files": files
        or {
            "src/example.py": {
                "executed_lines": [1, 2, 3],
                "missing_lines": [4],
            }
        }
    }


def scip_json_payload(*, documents: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build a minimal SCIP JSON payload.

    Parameters
    ----------
    documents
        SCIP document payloads.

    Returns
    -------
    dict[str, Any]
        SCIP JSON payload with documents.
    """
    return {
        "documents": list(
            documents
            or [
                {
                    "relativePath": "src/example.py",
                    "symbols": [],
                    "occurrences": [],
                }
            ]
        )
    }


__all__ = [
    "coverage_json_payload",
    "pytest_report_payload",
    "scip_json_payload",
]
