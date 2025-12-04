"""Validate Operation alignment with HTTP routers and MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from fastapi.routing import APIRoute
from mcp.server.fastmcp import FastMCP

from codeintel.config.datasets import get_dataset_contracts, get_dataset_contracts_by_table_key
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
from codeintel.serving.operations import Operation, get_operation
from codeintel.serving.operations.catalog import iter_registry_operations


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


def _ensure_operation(op_id: str) -> Operation:
    op = get_operation(op_id)
    if op is None:
        pytest.fail(f"Operation {op_id} is not registered")
    if op.http_path is None:
        pytest.fail(f"Operation {op_id} is missing http_path")
    return op


def test_http_routes_match_operations() -> None:
    """Ensure HTTP routers expose paths declared in Operation."""
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

    for router, op_ids in router_specs:
        paths = {route.path for route in router.routes if isinstance(route, APIRoute)}
        for op_id in op_ids:
            op = _ensure_operation(op_id)
            if op.http_path not in paths:
                pytest.fail(f"Path {op.http_path} for {op_id} not found in router")


def test_mcp_tool_names_match_operations() -> None:
    """Ensure MCP registration exposes every Operation.tool_name."""
    mcp = FastMCP("test")
    backend = _DummyBackend()
    register_tools(mcp, cast("QueryBackendOrService", backend))
    tools = cast("list[Any]", getattr(mcp, "tools", []))
    tool_names = {cast("str", getattr(tool, "name", "")) for tool in tools}
    tool_names.discard("")

    for op in iter_registry_operations():
        if op.tool_name is None:
            continue
        if op.tool_name not in tool_names:
            pytest.fail(f"MCP tool {op.tool_name} (op {op.id}) not registered")


def test_required_datasets_resolve_to_dataset_contracts() -> None:
    """Every Operation.required_datasets entry must map to a DatasetContract."""
    dataset_names = set(get_dataset_contracts().keys())
    table_keys = set(get_dataset_contracts_by_table_key().keys())

    for op in iter_registry_operations():
        for dataset_id in op.required_datasets:
            if dataset_id in dataset_names or dataset_id in table_keys:
                continue
            pytest.fail(f"Operation {op.id} refers to unknown dataset identifier: {dataset_id}")


def test_exposed_datasets_resolve_to_dataset_contracts() -> None:
    """Every Operation.exposed_datasets entry must map to a DatasetContract."""
    dataset_names = set(get_dataset_contracts().keys())
    table_keys = set(get_dataset_contracts_by_table_key().keys())

    for op in iter_registry_operations():
        for dataset_id in op.exposed_datasets:
            if dataset_id == "*":
                continue
            if dataset_id in dataset_names or dataset_id in table_keys:
                continue
            pytest.fail(f"Operation {op.id} refers to unknown exposed dataset: {dataset_id}")
