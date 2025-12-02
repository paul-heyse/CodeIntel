"""Test results ingestion facade with convenient function-based API.

This module provides a function-based API for test results ingestion
that wraps the class-based TestsIngestStep with sensible adapter defaults.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion.adapters import DuckDBStorageAdapter
from codeintel.ingestion.steps.tests_ingest import TestsIngestStep

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway


def ingest_tests(  # noqa: PLR0913
    gateway: StorageGateway,
    modules: Sequence[ModuleRecord] | None = None,
    *,
    repo: str,
    commit: str,
    repo_root: Path | None = None,
    report_path: Path | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """Ingest test results and persist to storage.

    This function provides a convenient entry point for test ingestion
    that creates the necessary adapters and executes the step.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    modules
        Modules to process; if not provided, uses tracker modules.
    repo
        Repository identifier.
    commit
        Commit identifier.
    repo_root
        Repository root path (reserved for future use).
    report_path
        Path to pytest JSON report.
    tool_service
        Tool service for running external tools (reserved for future use).
    tracker
        Optional change tracker for incremental processing.
    """
    # Reserved parameters for API compatibility
    del repo_root, tool_service
    if report_path is None:
        return  # Skip if no report path provided

    # Get modules from tracker if not provided
    actual_modules: Sequence[ModuleRecord]
    if modules is not None:
        actual_modules = modules
    elif tracker is not None:
        actual_modules = tracker.modules
    else:
        actual_modules = []

    # Create adapter
    storage = DuckDBStorageAdapter(gateway)

    # Create and execute step
    step = TestsIngestStep(storage=storage)
    step.execute(
        _modules=actual_modules,
        repo=repo,
        commit=commit,
        json_report_path=report_path,
    )


def _normalize_keywords(keywords: object) -> list[str]:
    """Normalize keywords from dict or list to list of strings.

    Parameters
    ----------
    keywords
        Keywords value from test entry (dict, list, or other).

    Returns
    -------
    list[str]
        Normalized list of keyword strings.
    """
    if isinstance(keywords, dict):
        return [k for k, v in keywords.items() if v]
    if isinstance(keywords, list):
        return keywords
    return []


def _coerce_duration(value: object) -> float:
    """Coerce a value to float, returning 0.0 on failure.

    Parameters
    ----------
    value
        Value to coerce to float.

    Returns
    -------
    float
        The coerced float value, or 0.0 if coercion fails.
    """
    if value is None:
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _normalize_call(call: dict[str, object]) -> dict[str, object]:
    """Normalize a call object, coercing duration to float.

    Parameters
    ----------
    call
        The call dict from a test entry.

    Returns
    -------
    dict[str, object]
        Normalized call dict with duration as float.
    """
    normalized: dict[str, object] = {k: v for k, v in call.items() if k != "duration"}
    if "duration" in call:
        normalized["duration"] = _coerce_duration(call["duration"])
    return normalized


def _normalize_test_entry(test: dict[str, object]) -> dict[str, object] | None:
    """Normalize a single test entry.

    Parameters
    ----------
    test
        Raw test entry from pytest report.

    Returns
    -------
    dict[str, object] | None
        Normalized entry, or None if entry should be filtered out.
    """
    nodeid = test.get("nodeid")
    if not nodeid:
        return None

    entry: dict[str, object] = {"nodeid": nodeid}
    entry["keywords"] = _normalize_keywords(test.get("keywords"))

    duration = test.get("duration")
    if duration is not None:
        entry["duration"] = _coerce_duration(duration)

    call = test.get("call")
    if isinstance(call, dict):
        entry["call"] = _normalize_call(call)

    # Copy remaining fields not already in entry
    entry.update({k: v for k, v in test.items() if k not in entry})

    return entry


def load_tests_from_report(report_path: Path | None = None) -> list[dict[str, object]]:
    """Load tests from a pytest report file.

    Load and normalize pytest JSON report entries. Entries without a `nodeid`
    are filtered out. Keywords are normalized from dict to list of truthy keys.
    Duration values are coerced to float.

    Parameters
    ----------
    report_path
        Path to the pytest JSON report file.

    Returns
    -------
    list[dict[str, object]]
        List of normalized test entries.
    """
    if report_path is None or not report_path.exists():
        return []

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    tests = data.get("tests", [])
    normalized: list[dict[str, object]] = []

    for test in tests:
        if not isinstance(test, dict):
            continue
        entry = _normalize_test_entry(test)
        if entry is not None:
            normalized.append(entry)

    return normalized


# Re-export step class for direct usage
__all__ = ["TestsIngestStep", "ingest_tests", "load_tests_from_report"]
