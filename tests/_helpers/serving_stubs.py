"""Protocol-compliant stubs for serving-layer tests.

These stubs satisfy DuckDBQueryApi and QueryService protocols while allowing
tests to inject custom payload producers for each method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.backend.query_api import (
    DatasetQueriesApi,
    DuckDBQueryApi,
    FunctionQueriesApi,
    ProfileQueriesApi,
    SubsystemQueriesApi,
)
from codeintel.serving.mcp.models import DatasetSpecDescriptor, GraphScopePayload
from codeintel.serving.services.query_service import QueryService
from codeintel.storage.gateway import StorageGateway


def _dispatch[T](hooks: dict[str, Callable[..., object]], name: str, **kwargs: object) -> T:
    if name not in hooks:
        raise NotImplementedError(f"hook not provided for {name}")
    return cast("T", hooks[name](**kwargs))


@dataclass
class HookedFunctionQueries(FunctionQueriesApi):
    """FunctionQueriesApi stub driven by injected hooks."""

    hooks: dict[str, Callable[..., object]] = field(default_factory=dict)

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        return _dispatch(
            self.hooks,
            "get_function_summary",
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            scope=scope,
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        return _dispatch(
            self.hooks,
            "list_high_risk_functions",
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
            scope=scope,
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        return _dispatch(
            self.hooks,
            "get_callgraph_neighbors",
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
            scope=scope,
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        return _dispatch(
            self.hooks,
            "get_tests_for_function",
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
            scope=scope,
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        return _dispatch(
            self.hooks,
            "get_callgraph_neighborhood",
            goid_h128=goid_h128,
            radius=radius,
            max_nodes=max_nodes,
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        return _dispatch(
            self.hooks,
            "get_import_boundary",
            subsystem_id=subsystem_id,
            max_edges=max_edges,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphScopePayload | None = None,
    ) -> dm.FileSummaryResult:
        return _dispatch(
            self.hooks,
            "get_file_summary",
            rel_path=rel_path,
            scope=scope,
        )


@dataclass
class HookedProfileQueries(ProfileQueriesApi):
    """ProfileQueriesApi stub driven by hooks."""

    hooks: dict[str, Callable[..., object]] = field(default_factory=dict)

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        return _dispatch(self.hooks, "get_function_profile", goid_h128=goid_h128)

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        return _dispatch(self.hooks, "get_file_profile", rel_path=rel_path)

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        return _dispatch(self.hooks, "get_module_profile", module=module)

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        return _dispatch(self.hooks, "get_function_architecture", goid_h128=goid_h128)

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        return _dispatch(self.hooks, "get_module_architecture", module=module)

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        return _dispatch(self.hooks, "get_file_hints", rel_path=rel_path)

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphScopePayload | None = None,
    ) -> dm.FileSummaryResult:
        return _dispatch(self.hooks, "get_file_summary", rel_path=rel_path, scope=scope)


@dataclass
class HookedSubsystemQueries(SubsystemQueriesApi):
    """SubsystemQueriesApi stub driven by hooks."""

    hooks: dict[str, Callable[..., object]] = field(default_factory=dict)

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSummaryResult:
        return _dispatch(self.hooks, "list_subsystems", limit=limit, role=role, q=q)

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        return _dispatch(self.hooks, "get_module_subsystems", module=module)

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return _dispatch(
            self.hooks,
            "get_subsystem_modules",
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSearchResult:
        return _dispatch(self.hooks, "search_subsystems", limit=limit, role=role, q=q)

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return _dispatch(
            self.hooks,
            "summarize_subsystem",
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        return _dispatch(self.hooks, "list_subsystem_profiles", limit=limit)

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        return _dispatch(self.hooks, "list_subsystem_coverage", limit=limit)


@dataclass
class HookedDatasetQueries(DatasetQueriesApi):
    """DatasetQueriesApi stub driven by hooks."""

    hooks: dict[str, Callable[..., object]] = field(default_factory=dict)

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        return _dispatch(self.hooks, "list_datasets")

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        return _dispatch(self.hooks, "dataset_specs")

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        return _dispatch(
            self.hooks,
            "read_dataset_rows",
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        return _dispatch(
            self.hooks,
            "dataset_schema",
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )


class HookedDuckDBQueryApi(DuckDBQueryApi):
    """DuckDBQueryApi stub exposing configurable delegates."""

    def __init__(
        self,
        *,
        gateway: StorageGateway | None = None,
        limits: BackendLimits | None = None,
        function_hooks: dict[str, Callable[..., object]] | None = None,
        profile_hooks: dict[str, Callable[..., object]] | None = None,
        subsystem_hooks: dict[str, Callable[..., object]] | None = None,
        dataset_hooks: dict[str, Callable[..., object]] | None = None,
    ) -> None:
        self._gateway = gateway
        self._limits = limits or BackendLimits()
        self.function_hooks = function_hooks or {}
        self.profile_hooks = profile_hooks or {}
        self.subsystem_hooks = subsystem_hooks or {}
        self.dataset_hooks = dataset_hooks or {}

    @property
    def gateway(self) -> StorageGateway | None:
        return self._gateway

    @property
    def limits(self) -> BackendLimits:
        return self._limits

    @property
    def functions(self) -> HookedFunctionQueries:
        return HookedFunctionQueries(self.function_hooks)

    @property
    def modules(self) -> HookedProfileQueries:
        return HookedProfileQueries(self.profile_hooks)

    @property
    def subsystems(self) -> HookedSubsystemQueries:
        return HookedSubsystemQueries(self.subsystem_hooks)

    @property
    def datasets(self) -> HookedDatasetQueries:
        return HookedDatasetQueries(self.dataset_hooks)

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


@dataclass
class HookedQueryService(QueryService):
    """QueryService stub driven by hooks for each API surface."""

    limits: BackendLimits = field(default_factory=BackendLimits)
    hooks: dict[str, Callable[..., object]] = field(default_factory=dict)

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.FunctionSummaryResult:
        return _dispatch(
            self.hooks,
            "get_function_summary",
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
            scope=scope,
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> dm.HighRiskFunctionsResult:
        return _dispatch(
            self.hooks,
            "list_high_risk_functions",
            min_risk=min_risk,
            limit=limit,
            tested_only=tested_only,
            scope=scope,
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.CallGraphNeighbors:
        return _dispatch(
            self.hooks,
            "get_callgraph_neighbors",
            goid_h128=goid_h128,
            direction=direction,
            limit=limit,
            scope=scope,
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphScopePayload | None = None,
    ) -> dm.TestsForFunctionResult:
        return _dispatch(
            self.hooks,
            "get_tests_for_function",
            goid_h128=goid_h128,
            urn=urn,
            limit=limit,
            scope=scope,
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> dm.GraphNeighborhood:
        return _dispatch(
            self.hooks,
            "get_callgraph_neighborhood",
            goid_h128=goid_h128,
            radius=radius,
            max_nodes=max_nodes,
        )

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> dm.ImportBoundary:
        return _dispatch(
            self.hooks,
            "get_import_boundary",
            subsystem_id=subsystem_id,
            max_edges=max_edges,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphScopePayload | None = None,
    ) -> dm.FileSummaryResult:
        return _dispatch(
            self.hooks,
            "get_file_summary",
            rel_path=rel_path,
            scope=scope,
        )

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        return _dispatch(self.hooks, "get_function_profile", goid_h128=goid_h128)

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        return _dispatch(self.hooks, "get_file_profile", rel_path=rel_path)

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        return _dispatch(self.hooks, "get_module_profile", module=module)

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        return _dispatch(self.hooks, "get_function_architecture", goid_h128=goid_h128)

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        return _dispatch(self.hooks, "get_module_architecture", module=module)

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSummaryResult:
        return _dispatch(self.hooks, "list_subsystems", limit=limit, role=role, q=q)

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        return _dispatch(self.hooks, "get_module_subsystems", module=module)

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return _dispatch(
            self.hooks,
            "get_subsystem_modules",
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSearchResult:
        return _dispatch(self.hooks, "search_subsystems", limit=limit, role=role, q=q)

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return _dispatch(
            self.hooks,
            "summarize_subsystem",
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        return _dispatch(self.hooks, "list_subsystem_profiles", limit=limit)

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        return _dispatch(self.hooks, "list_subsystem_coverage", limit=limit)

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        return _dispatch(self.hooks, "list_datasets")

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        return _dispatch(self.hooks, "dataset_specs")

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dm.DatasetRows:
        return _dispatch(
            self.hooks,
            "read_dataset_rows",
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        return _dispatch(
            self.hooks,
            "dataset_schema",
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )


__all__ = [
    "HookedDatasetQueries",
    "HookedDuckDBQueryApi",
    "HookedFunctionQueries",
    "HookedProfileQueries",
    "HookedQueryService",
    "HookedSubsystemQueries",
]
