"""Validate OperationSpec alignment with HTTP routers and MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from fastapi.routing import APIRoute
from mcp.server.fastmcp import FastMCP

from codeintel.config.datasets import DATASET_CONTRACTS, DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.routes.architecture import build_architecture_router
from codeintel.serving.http.routes.datasets import build_datasets_router
from codeintel.serving.http.routes.functions import build_functions_router
from codeintel.serving.http.routes.health import build_health_router
from codeintel.serving.http.routes.ide import build_ide_router
from codeintel.serving.http.routes.profiles import build_profiles_router
from codeintel.serving.http.routes.subsystems import build_subsystem_router
from codeintel.serving.mcp.tool_utils import QueryBackendOrService
from codeintel.serving.mcp.tools_base import register_tools
from codeintel.serving.registry import OperationSpec, get_operation_spec, iter_operation_specs


class _DummyModel:
    """Lightweight model stub returning dict payloads."""

    @staticmethod
    def model_dump() -> dict[str, object]:
        return {}


class _DummyBackend:
    """Backend stub that exposes callable attributes for all specs."""

    limits = BackendLimits()

    def __getattr__(self, _name: str) -> Callable[..., _DummyModel]:
        def _call(**_kwargs: object) -> _DummyModel:
            return _DummyModel()

        return _call


def _ensure_spec(spec_id: str) -> OperationSpec:
    spec = get_operation_spec(spec_id)
    if spec is None:
        pytest.fail(f"OperationSpec {spec_id} is not registered")
    if spec.http_path is None:
        pytest.fail(f"OperationSpec {spec_id} is missing http_path")
    return spec


def test_http_routes_match_operation_specs() -> None:
    """Ensure HTTP routers expose paths declared in OperationSpec."""
    router_specs = [
        (
            build_functions_router(),
            [
                "function.summary",
                "functions.high_risk",
                "functions.tests",
                "graph.call_neighbors",
                "graph.call_neighborhood",
                "graph.import_boundary",
                "file.summary",
            ],
        ),
        (build_profiles_router(), ["profiles.function", "profiles.file", "profiles.module"]),
        (
            build_datasets_router(),
            ["datasets.list", "datasets.specs", "datasets.rows", "datasets.schema"],
        ),
        (build_ide_router(), ["ide.hints"]),
        (build_architecture_router(), ["architecture.function", "architecture.module"]),
        (
            build_subsystem_router(),
            [
                "subsystems.list",
                "subsystems.profiles",
                "subsystems.coverage",
                "subsystems.module_memberships",
                "subsystems.detail",
            ],
        ),
        (build_health_router(), ["health.status"]),
    ]

    for router, spec_ids in router_specs:
        paths = {route.path for route in router.routes if isinstance(route, APIRoute)}
        for spec_id in spec_ids:
            spec = _ensure_spec(spec_id)
            if spec.http_path not in paths:
                pytest.fail(f"Path {spec.http_path} for {spec_id} not found in router")


def test_mcp_tool_names_match_operation_specs() -> None:
    """Ensure MCP registration exposes every OperationSpec.tool_name."""
    mcp = FastMCP("test")
    backend = _DummyBackend()
    register_tools(mcp, cast("QueryBackendOrService", backend))
    tools = cast("list[Any]", getattr(mcp, "tools", []))
    tool_names = {cast("str", getattr(tool, "name", "")) for tool in tools}
    tool_names.discard("")

    for spec in iter_operation_specs():
        if spec.tool_name is None:
            continue
        if spec.tool_name not in tool_names:
            pytest.fail(f"MCP tool {spec.tool_name} (spec {spec.id}) not registered")


def test_required_datasets_resolve_to_dataset_contracts() -> None:
    """Every OperationSpec.required_datasets entry must map to a DatasetContract."""
    dataset_names = set(DATASET_CONTRACTS.keys())
    table_keys = set(DATASET_CONTRACTS_BY_TABLE_KEY.keys())

    for spec in iter_operation_specs():
        for dataset_id in spec.required_datasets:
            if dataset_id in dataset_names or dataset_id in table_keys:
                continue
            pytest.fail(
                f"OperationSpec {spec.id} refers to unknown dataset identifier: {dataset_id}"
            )


def test_exposed_datasets_resolve_to_dataset_contracts() -> None:
    """Every OperationSpec.exposed_datasets entry must map to a DatasetContract."""
    dataset_names = set(DATASET_CONTRACTS.keys())
    table_keys = set(DATASET_CONTRACTS_BY_TABLE_KEY.keys())

    for spec in iter_operation_specs():
        for dataset_id in spec.exposed_datasets:
            if dataset_id == "*":
                continue
            if dataset_id in dataset_names or dataset_id in table_keys:
                continue
            pytest.fail(f"OperationSpec {spec.id} refers to unknown exposed dataset: {dataset_id}")
