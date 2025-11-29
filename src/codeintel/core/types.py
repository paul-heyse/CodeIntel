"""Core domain TypedDicts and light validators shared across layers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypedDict

from codeintel.config.primitives import (
    GraphBackendConfig,
    GraphFeatureFlags,
    SnapshotRef,
)

# ---------------------------------------------------------------------------
# Pytest JSON report types
# ---------------------------------------------------------------------------


class PytestCallEntry(TypedDict, total=False):
    """Subset of pytest-json-report call data."""

    duration: float


class PytestTestEntry(TypedDict, total=False):
    """Shape of a pytest-json-report test object."""

    nodeid: str
    outcome: str
    status: str
    keywords: dict[str, bool] | list[str]
    duration: float | None
    call: PytestCallEntry | None


# ---------------------------------------------------------------------------
# SCIP JSON types
# ---------------------------------------------------------------------------


class ScipRange(TypedDict, total=False):
    """SCIP range structure (0-based line/column indices)."""

    start_line: int
    start_character: int
    end_line: int
    end_character: int


class ScipOccurrence(TypedDict, total=False):
    """Occurrence entry within a SCIP document."""

    range: ScipRange
    symbol: str
    symbol_roles: int | None


class ScipDocument(TypedDict, total=False):
    """SCIP JSON document emitted by scip-python."""

    relative_path: str
    occurrences: list[ScipOccurrence]


def _normalize_path(path: object) -> str | None:
    """
    Normalize a path-like value to a forward-slash string.

    Returns
    -------
    str | None
        Normalized path or None when the input is falsy or not a string.
    """
    if not isinstance(path, str):
        return None
    return path.replace("\\", "/")


def _normalize_occurrence(raw: object) -> ScipOccurrence | None:
    """
    Normalize a raw occurrence mapping into a ScipOccurrence.

    Filters out entries missing a symbol.

    Returns
    -------
    ScipOccurrence | None
        Normalized occurrence or None when invalid.
    """
    if not isinstance(raw, Mapping):
        return None
    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        return None
    symbol_roles_raw = raw.get("symbol_roles")
    symbol_roles = None
    if symbol_roles_raw is not None:
        try:
            symbol_roles = int(symbol_roles_raw)
        except (TypeError, ValueError):
            symbol_roles = None
    range_raw = raw.get("range")
    range_val: ScipRange | None = None
    if isinstance(range_raw, Mapping):
        range_val = ScipRange(
            start_line=int(range_raw.get("start_line", 0))
            if range_raw.get("start_line") is not None
            else 0,
            start_character=int(range_raw.get("start_character", 0))
            if range_raw.get("start_character") is not None
            else 0,
            end_line=int(range_raw.get("end_line", 0))
            if range_raw.get("end_line") is not None
            else 0,
            end_character=int(range_raw.get("end_character", 0))
            if range_raw.get("end_character") is not None
            else 0,
        )
    occurrence: ScipOccurrence = ScipOccurrence(
        symbol=symbol,
        symbol_roles=symbol_roles,
    )
    if range_val is not None:
        occurrence["range"] = range_val
    return occurrence


def normalize_scip_document(raw: Mapping[str, object]) -> ScipDocument | None:
    """
    Normalize a raw mapping into a ScipDocument.

    Returns
    -------
    ScipDocument | None
        Normalized document or None when required fields are missing.
    """
    path = _normalize_path(raw.get("relative_path"))
    if path is None:
        return None
    occurrences_raw = raw.get("occurrences")
    occurrences: list[ScipOccurrence] = []
    if isinstance(occurrences_raw, Iterable):
        occurrences = [occ for occ in (_normalize_occurrence(o) for o in occurrences_raw) if occ]
    if not occurrences:
        return None
    return ScipDocument(relative_path=path, occurrences=occurrences)


def validate_scip_document(doc: ScipDocument) -> None:
    """
    Validate required SCIP document fields.

    Raises
    ------
    ValueError
        When relative_path is missing or occurrences contain invalid symbols.
    """
    if not doc.get("relative_path"):
        message = "SCIP document missing relative_path"
        raise ValueError(message)
    for occ in doc.get("occurrences", []):
        if not occ.get("symbol"):
            message = "SCIP occurrence missing symbol"
            raise ValueError(message)


def normalize_pytest_entry(raw: Mapping[str, object]) -> PytestTestEntry | None:
    """
    Normalize a raw pytest-json-report entry into a PytestTestEntry.

    Returns
    -------
    PytestTestEntry | None
        Normalized entry or None when nodeid is missing.
    """
    nodeid_raw = raw.get("nodeid")
    if not isinstance(nodeid_raw, str) or not nodeid_raw:
        return None
    outcome_raw = raw.get("outcome")
    outcome = outcome_raw if isinstance(outcome_raw, str) else None
    status_raw = raw.get("status")
    status = status_raw if isinstance(status_raw, str) else None
    keywords_raw = raw.get("keywords")
    keywords: list[str]
    if isinstance(keywords_raw, Mapping):
        keywords = sorted(str(k) for k, v in keywords_raw.items() if v)
    elif isinstance(keywords_raw, list):
        keywords = sorted(str(k) for k in keywords_raw)
    else:
        keywords = []
    duration_val = raw.get("duration")
    duration: float | None
    if isinstance(duration_val, (int, float, str)):
        try:
            duration = float(duration_val)
        except (TypeError, ValueError):
            duration = None
    else:
        duration = None
    call_raw = raw.get("call")
    call: PytestCallEntry | None = None
    if isinstance(call_raw, Mapping):
        call_duration_raw = call_raw.get("duration")
        call_duration: float | None
        if isinstance(call_duration_raw, (int, float, str)):
            try:
                call_duration = float(call_duration_raw)
            except (TypeError, ValueError):
                call_duration = None
        else:
            call_duration = None
        call = (
            PytestCallEntry(duration=call_duration)
            if call_duration is not None
            else PytestCallEntry()
        )
    return PytestTestEntry(
        nodeid=nodeid_raw,
        outcome=outcome or status or "unknown",
        status=status or outcome or "unknown",
        keywords=keywords,
        duration=duration,
        call=call,
    )


def validate_pytest_entry(entry: PytestTestEntry) -> None:
    """
    Validate required pytest entry fields.

    Raises
    ------
    ValueError
        When nodeid is missing.
    """
    if not entry.get("nodeid"):
        message = "Pytest entry missing nodeid"
        raise ValueError(message)


__all__ = [
    "GraphBackendConfig",
    "GraphFeatureFlags",
    "PytestCallEntry",
    "PytestTestEntry",
    "ScipDocument",
    "ScipOccurrence",
    "ScipRange",
    "SnapshotRef",
    "normalize_pytest_entry",
    "normalize_scip_document",
    "validate_pytest_entry",
    "validate_scip_document",
]
