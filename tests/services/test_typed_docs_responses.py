"""Ensure docs view responses return typed payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, cast

import pytest

from codeintel.serving.mcp.models import (
    FunctionSummaryResponse,
    SubsystemCoverageResponse,
    SubsystemProfileResponse,
)
from codeintel.serving.mcp.query_service import BackendLimits, DuckDBQueryService
from codeintel.storage.gateway import open_memory_gateway


class StubFunctionRow(TypedDict):
    """Typed function summary payload used in stub repositories."""

    repo: str
    commit: str
    rel_path: str
    function_goid_h128: int
    urn: str
    qualname: str
    risk_score: float


@dataclass
class _StubFunctions:
    """Minimal function repository stub for typed response tests."""

    row: StubFunctionRow

    def resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        _ = (urn, rel_path, qualname)
        return goid_h128 or int(self.row["function_goid_h128"])

    def get_function_summary_by_goid(self, goid_h128: int) -> dict[str, object] | None:
        _ = goid_h128
        return cast("dict[str, object]", self.row)


class _StubQueryService(DuckDBQueryService):
    def __init__(self, row: StubFunctionRow) -> None:
        gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
        super().__init__(
            gateway=gateway,
            repo=str(row["repo"]),
            commit=str(row["commit"]),
            limits=BackendLimits(),
        )
        self._stub_functions = _StubFunctions(row=row)

    @property
    def functions(self) -> _StubFunctions:  # type: ignore[override]
        return self._stub_functions


@dataclass
class _StubSubsystems:
    """Minimal subsystem repository stub for typed response tests."""

    profile_row: dict[str, object]
    coverage_row: dict[str, object]

    def list_subsystem_profiles(self, *, limit: int) -> list[dict[str, object]]:
        _ = limit
        return [self.profile_row]

    def list_subsystem_coverage(self, *, limit: int) -> list[dict[str, object]]:
        _ = limit
        return [self.coverage_row]


class _StubSubsystemQueryService(DuckDBQueryService):
    def __init__(
        self,
        *,
        repo: str,
        commit: str,
        profile_row: dict[str, object],
        coverage_row: dict[str, object],
    ) -> None:
        gateway = open_memory_gateway(ensure_views=True, validate_schema=False)
        super().__init__(gateway=gateway, repo=repo, commit=commit, limits=BackendLimits())
        self._stub_subsystems = _StubSubsystems(profile_row=profile_row, coverage_row=coverage_row)

    @property
    def subsystems(self) -> _StubSubsystems:  # type: ignore[override]
        return self._stub_subsystems


def test_function_summary_response_uses_typed_row() -> None:
    """Function summaries should populate FunctionSummaryRow instances."""
    goid = 1234
    row: StubFunctionRow = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "rel_path": "pkg/mod.py",
        "function_goid_h128": goid,
        "urn": "urn:demo",
        "qualname": "pkg.mod:func",
        "risk_score": 0.5,
    }
    service = _StubQueryService(row=row)

    resp = service.get_function_summary(goid_h128=goid)

    if not isinstance(resp, FunctionSummaryResponse):
        pytest.fail("Expected FunctionSummaryResponse")
    if not resp.found or resp.summary is None:
        pytest.fail("Expected function summary to be present")
    if resp.summary.function_goid_h128 != goid:
        pytest.fail("Incorrect GOID propagated in summary")
    if resp.summary.repo != "demo/repo":
        pytest.fail("Incorrect repo propagated in summary")


def test_subsystem_profile_response_uses_typed_row() -> None:
    """Subsystem profile listing should emit typed rows."""
    expected_limit = 5
    row: dict[str, object] = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "subsystem_id": "subsysdemo",
        "name": "Subsystem Demo",
        "description": "Demo subsystem",
        "module_count": 3,
        "function_count": 10,
        "risk_level": "medium",
        "avg_risk_score": 0.4,
        "entrypoints_json": [],
    }
    service = _StubSubsystemQueryService(
        repo="demo/repo", commit="deadbeef", profile_row=row, coverage_row={}
    )

    resp = service.list_subsystem_profiles(limit=expected_limit)

    if not isinstance(resp, SubsystemProfileResponse):
        pytest.fail("Expected SubsystemProfileResponse")
    if not resp.profiles:
        pytest.fail("Expected subsystem profile rows")
    first = resp.profiles[0]
    if first.subsystem_id != "subsysdemo":
        pytest.fail("Incorrect subsystem_id propagated")
    if resp.meta.applied_limit != expected_limit:
        pytest.fail("Expected applied limit to be set on meta")


def test_subsystem_coverage_response_uses_typed_row() -> None:
    """Subsystem coverage listing should emit typed rows."""
    coverage_row: dict[str, object] = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "subsystem_id": "subsysdemo",
        "name": "Subsystem Demo",
        "test_count": 4,
        "passed_test_count": 3,
        "failed_test_count": 1,
        "function_coverage_ratio": 0.5,
    }
    service = _StubSubsystemQueryService(
        repo="demo/repo",
        commit="deadbeef",
        profile_row={},
        coverage_row=coverage_row,
    )

    expected_limit = 2
    resp = service.list_subsystem_coverage(limit=expected_limit)

    if not isinstance(resp, SubsystemCoverageResponse):
        pytest.fail("Expected SubsystemCoverageResponse")
    if not resp.coverage:
        pytest.fail("Expected subsystem coverage rows")
    first = resp.coverage[0]
    expected_test_count = 4
    if first.test_count != expected_test_count:
        pytest.fail("Incorrect test_count propagated")
    if resp.meta.applied_limit != expected_limit:
        pytest.fail("Expected applied limit to be set on meta")
