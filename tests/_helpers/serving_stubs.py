"""Protocol-compliant stubs for serving-layer tests.

These stubs satisfy DuckDBQueryApi and QueryService protocols while allowing
tests to inject custom payload producers for each method.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.backend.query_api import (
    DatasetQueriesApi,
    DuckDBQueryApi,
    FunctionQueriesApi,
    ProfileQueriesApi,
    SubsystemQueriesApi,
)
from codeintel.serving.mcp.models import DatasetSpecDescriptor
from codeintel.storage.gateway import StorageGateway, open_memory_gateway


def _dispatch(hooks: dict[str, Callable[..., object]], name: str, **kwargs: object) -> object:
    if name not in hooks:
        message = "hook not provided for requested operation"
        raise NotImplementedError(message)
    return hooks[name](**kwargs)


def _dispatch_typed[T](
    hooks: Mapping[str, Callable[..., object]],
    name: str,
    return_type: type[T],
    **kwargs: object,
) -> T:
    if name not in hooks:
        message = "hook not provided for requested operation"
        raise NotImplementedError(message)
    result = hooks[name](**kwargs)
    if isinstance(result, return_type):
        return result
    return cast("T", result)


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
        scope: GraphRunScope | None = None,
    ) -> dm.FunctionSummaryResult:
        return _dispatch_typed(
            self.hooks,
            "get_function_summary",
            dm.FunctionSummaryResult,
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
        scope: GraphRunScope | None = None,
    ) -> dm.HighRiskFunctionsResult:
        return _dispatch_typed(
            self.hooks,
            "list_high_risk_functions",
            dm.HighRiskFunctionsResult,
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
        scope: GraphRunScope | None = None,
    ) -> dm.CallGraphNeighbors:
        return _dispatch_typed(
            self.hooks,
            "get_callgraph_neighbors",
            dm.CallGraphNeighbors,
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
        scope: GraphRunScope | None = None,
    ) -> dm.TestsForFunctionResult:
        return _dispatch_typed(
            self.hooks,
            "get_tests_for_function",
            dm.TestsForFunctionResult,
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
        return _dispatch_typed(
            self.hooks,
            "get_callgraph_neighborhood",
            dm.GraphNeighborhood,
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
        return _dispatch_typed(
            self.hooks,
            "get_import_boundary",
            dm.ImportBoundary,
            subsystem_id=subsystem_id,
            max_edges=max_edges,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphRunScope | None = None,
    ) -> dm.FileSummaryResult:
        return _dispatch_typed(
            self.hooks,
            "get_file_summary",
            dm.FileSummaryResult,
            rel_path=rel_path,
            scope=scope,
        )

    def get_function_profile(self, goid_h128: int) -> dm.FunctionProfileResult:
        return _dispatch_typed(
            self.hooks, "get_function_profile", dm.FunctionProfileResult, goid_h128=goid_h128
        )

    def get_function_architecture(self, goid_h128: int) -> dm.FunctionArchitectureResult:
        return _dispatch_typed(
            self.hooks,
            "get_function_architecture",
            dm.FunctionArchitectureResult,
            goid_h128=goid_h128,
        )


@dataclass
class HookedProfileQueries(ProfileQueriesApi):
    """ProfileQueriesApi stub driven by hooks."""

    hooks: dict[str, Callable[..., object]] = field(default_factory=dict)

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        return _dispatch_typed(
            self.hooks,
            "get_function_profile",
            dm.FunctionProfileResult,
            goid_h128=goid_h128,
        )

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        return _dispatch_typed(
            self.hooks,
            "get_file_profile",
            dm.FileProfileResult,
            rel_path=rel_path,
        )

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        return _dispatch_typed(
            self.hooks,
            "get_module_profile",
            dm.ModuleProfileResult,
            module=module,
        )

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        return _dispatch_typed(
            self.hooks,
            "get_function_architecture",
            dm.FunctionArchitectureResult,
            goid_h128=goid_h128,
        )

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        return _dispatch_typed(
            self.hooks,
            "get_module_architecture",
            dm.ModuleArchitectureResult,
            module=module,
        )

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        return _dispatch_typed(
            self.hooks,
            "get_file_hints",
            dm.FileHintsResult,
            rel_path=rel_path,
        )

    def get_file_summary(
        self,
        *,
        rel_path: str,
        scope: GraphRunScope | None = None,
    ) -> dm.FileSummaryResult:
        return _dispatch_typed(
            self.hooks,
            "get_file_summary",
            dm.FileSummaryResult,
            rel_path=rel_path,
            scope=scope,
        )


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
        return _dispatch_typed(
            self.hooks,
            "list_subsystems",
            dm.SubsystemSummaryResult,
            limit=limit,
            role=role,
            q=q,
        )

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        return _dispatch_typed(
            self.hooks,
            "get_module_subsystems",
            dm.ModuleSubsystemResult,
            module=module,
        )

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return _dispatch_typed(
            self.hooks,
            "get_subsystem_modules",
            dm.SubsystemModulesResult,
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
        return _dispatch_typed(
            self.hooks,
            "search_subsystems",
            dm.SubsystemSearchResult,
            limit=limit,
            role=role,
            q=q,
        )

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return _dispatch_typed(
            self.hooks,
            "summarize_subsystem",
            dm.SubsystemModulesResult,
            subsystem_id=subsystem_id,
            module_limit=module_limit,
        )

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        return _dispatch_typed(
            self.hooks,
            "list_subsystem_profiles",
            dm.SubsystemProfileResult,
            limit=limit,
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        return _dispatch_typed(
            self.hooks,
            "list_subsystem_coverage",
            dm.SubsystemCoverageResult,
            limit=limit,
        )


@dataclass
class HookedDatasetQueries(DatasetQueriesApi):
    """DatasetQueriesApi stub driven by hooks."""

    hooks: dict[str, Callable[..., object]] = field(default_factory=dict)

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        """
        Return dataset descriptors from the configured hook.

        Returns
        -------
        list[DatasetDescriptorDomain]
            Dataset descriptors provided by the hook.
        """
        return _dispatch_typed(self.hooks, "list_datasets", list[dm.DatasetDescriptorDomain])

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Return dataset spec descriptors from the configured hook.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset specification descriptors.
        """
        return _dispatch_typed(self.hooks, "dataset_specs", list[DatasetSpecDescriptor])

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Mapping[str, object]]:
        return _dispatch_typed(
            self.hooks,
            "read_dataset_rows",
            Sequence[Mapping[str, object]],
            dataset_name=dataset_name,
            limit=limit,
            offset=offset,
        )

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        return _dispatch_typed(
            self.hooks,
            "dataset_schema",
            dm.DatasetSchema,
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
        hooks: dict[str, dict[str, Callable[..., object]]] | None = None,
    ) -> None:
        self._gateway = gateway or open_memory_gateway(
            apply_schema=True, ensure_views=False, validate_schema=False
        )
        self._limits = limits or BackendLimits()
        grouped = hooks or {}
        self.function_hooks = grouped.get("function_hooks", {})
        self.profile_hooks = grouped.get("profile_hooks", {})
        self.subsystem_hooks = grouped.get("subsystem_hooks", {})
        self.dataset_hooks = grouped.get("dataset_hooks", {})

    @property
    def gateway(self) -> StorageGateway:
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

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        """
        Delegate dataset listing through injected hooks.

        Returns
        -------
        list[DatasetDescriptorDomain]
            Dataset descriptors provided by the hook.
        """
        return self.datasets.list_datasets()

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Delegate dataset specs retrieval through injected hooks.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset spec descriptors provided by the hook.
        """
        return self.datasets.dataset_specs()

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


__all__ = [
    "HookedDatasetQueries",
    "HookedDuckDBQueryApi",
    "HookedFunctionQueries",
    "HookedProfileQueries",
    "HookedSubsystemQueries",
]
