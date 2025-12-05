"""Ensure docs view responses return typed payloads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TypedDict, cast

import pytest

from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import (
    FunctionSummaryResponse,
    FunctionSummaryRow,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemProfileResponse,
    SubsystemProfileRow,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.functions import FunctionRepository
from codeintel.storage.repositories.subsystems import SubsystemRepository
from tests._helpers.gateway import gateway_with_macros


class StubFunctionRow(TypedDict):
    """Typed function summary payload used in stub repositories."""

    repo: str
    commit: str
    rel_path: str
    function_goid_h128: int
    urn: str
    qualname: str
    risk_score: float


class _StubFunctions(FunctionRepository):
    """Function repository stub overriding required methods."""

    def __init__(
        self,
        row: StubFunctionRow,
        *,
        gateway: StorageGateway,
        repo: str,
        commit: str,
    ) -> None:
        super().__init__(gateway=gateway, repo=repo, commit=commit)
        self._row = row

    def resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        _ = (urn, rel_path, qualname)
        return goid_h128 or int(self._row["function_goid_h128"])

    def get_function_summary_by_goid(self, goid_h128: int) -> dict[str, object] | None:
        _ = goid_h128
        return cast("dict[str, object]", self._row)


class _StubFunctionQuery:
    """Lightweight stub implementing the function summary surface."""

    def __init__(self, row: StubFunctionRow) -> None:
        gateway = gateway_with_macros(validate_schema=False)
        self.gateway = gateway
        self.limits = BackendLimits()
        self._functions = _StubFunctions(
            row=row,
            gateway=gateway,
            repo=str(row["repo"]),
            commit=str(row["commit"]),
        )
        self.functions = self._functions
        self.modules = self
        self.subsystems = self
        self.datasets = self

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> FunctionSummaryResponse:
        _ = (urn, rel_path, qualname)
        row = self._functions.get_function_summary_by_goid(goid_h128 or 0)
        return FunctionSummaryResponse(
            found=row is not None,
            summary=FunctionSummaryRow.model_validate(row) if row is not None else None,
            meta=ResponseMeta(),
        )


class _StubSubsystems(SubsystemRepository):
    """Minimal subsystem repository stub for typed response tests."""

    def __init__(
        self,
        *,
        profile_row: dict[str, object],
        coverage_row: dict[str, object],
        gateway: StorageGateway,
        repo: str,
        commit: str,
    ) -> None:
        super().__init__(gateway=gateway, repo=repo, commit=commit)
        self._profile_row = profile_row
        self._coverage_row = coverage_row

    def list_subsystem_profiles(self, *, limit: int) -> list[dict[str, object]]:
        _ = limit
        return [self._profile_row]

    def list_subsystem_coverage(self, *, limit: int) -> list[dict[str, object]]:
        _ = limit
        return [self._coverage_row]


class _StubSubsystemQueryService:
    # Explicit type annotations for attributes that get reassigned
    subsystems: _StubSubsystems | _StubSubsystemQueryService

    def __init__(
        self,
        *,
        repo: str,
        commit: str,
        profile_row: dict[str, object],
        coverage_row: dict[str, object],
    ) -> None:
        self.gateway = SimpleNamespace(datasets=SimpleNamespace(mapping={}))
        self.limits = BackendLimits()
        self.functions = self
        self.modules = self
        self.subsystems = self
        self.datasets = self
        gateway = gateway_with_macros(validate_schema=False)
        self._profile_row = profile_row
        self._coverage_row = coverage_row
        self._subsystems = _StubSubsystems(
            profile_row=profile_row,
            coverage_row=coverage_row,
            gateway=gateway,
            repo=repo,
            commit=commit,
        )
        self.subsystems = self._subsystems

    def list_subsystem_profiles(self, *, limit: int) -> SubsystemProfileResponse:
        _ = limit
        return SubsystemProfileResponse(
            profiles=[SubsystemProfileRow.model_validate(self._profile_row)],
            meta=ResponseMeta(requested_limit=limit, applied_limit=limit),
        )

    def list_subsystem_coverage(self, *, limit: int) -> SubsystemCoverageResponse:
        _ = limit
        return SubsystemCoverageResponse(
            coverage=[SubsystemCoverageRow.model_validate(self._coverage_row)],
            meta=ResponseMeta(requested_limit=limit, applied_limit=limit),
        )


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
    service = _StubFunctionQuery(row=row)

    resp = service.get_function_summary(goid_h128=goid)

    if not isinstance(resp, FunctionSummaryResponse):
        pytest.fail("Expected FunctionSummaryResponse")
    if not resp.found or resp.summary is None:
        pytest.fail("Expected function summary to be present")
    if not isinstance(resp.summary, FunctionSummaryRow):
        pytest.fail("Expected typed FunctionSummaryRow payload")
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
