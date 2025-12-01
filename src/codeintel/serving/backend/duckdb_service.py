"""DuckDB-backed query service shared by all serving surfaces.

All SQL queries against docs.* and analytics.* views/tables live here.
Other modules must call this service (via LocalQueryService/QueryService)
instead of issuing custom SELECTs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import networkx as nx

from codeintel.config.dataset_contract import DatasetContract
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.graphs.engine import GraphEngine
from codeintel.serving.backend.pagination import BackendLimits, clamp_limit_value
from codeintel.serving.backend.query_api import (
    DatasetQueriesApi,
    FunctionQueriesApi,
    ProfileQueriesApi,
    SubsystemQueriesApi,
)
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import (
    CallGraphEdgeRow,
    CallGraphNeighborsResponse,
    DatasetSchemaColumn,
    DatasetSchemaResponse,
    DatasetSpecDescriptor,
    FileHintsResponse,
    FileProfileResponse,
    FileProfileRow,
    FileSummaryResponse,
    FileSummaryRow,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    FunctionProfileRow,
    FunctionSummaryResponse,
    FunctionSummaryRow,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    Message,
    ModuleArchitectureResponse,
    ModuleArchitectureRow,
    ModuleProfileResponse,
    ModuleProfileRow,
    ModuleSubsystemResponse,
    ModuleWithSubsystemRow,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemProfileRow,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
    SubsystemSummaryRow,
    TestsForFunctionResponse,
    ViewRow,
)
from codeintel.storage.datasets import dataset_for_name, list_dataset_specs, load_dataset_registry
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.repositories import (
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)


def _fetch_duckdb_schema(con: DuckDBConnection, table_key: str) -> list[DatasetSchemaColumn]:
    """
    Return column descriptors for a DuckDB table/view.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Fully qualified table/view name.

    Returns
    -------
    list[DatasetSchemaColumn]
        Column descriptors derived from information_schema.
    """
    if "." not in table_key:
        return []
    schema_name, table_name = table_key.split(".", maxsplit=1)
    rows = con.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema_name, table_name],
    ).fetchall()
    return [
        DatasetSchemaColumn(
            name=str(col_name),
            type=str(col_type),
            nullable=str(nullable).upper() == "YES",
        )
        for col_name, col_type, nullable in rows
    ]


def _load_json_schema(ds: DatasetContract) -> dict[str, object] | None:
    """
    Load a JSON Schema document for a dataset if present on disk.

    Parameters
    ----------
    ds
        Dataset metadata entry from the registry.

    Returns
    -------
    dict[str, object] | None
        Parsed JSON Schema when available.
    """
    if ds.json_schema_id is None:
        return None
    schema_path = _schema_path(ds.json_schema_id)
    if not schema_path.exists():
        return None
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _schema_path(schema_id: str) -> Path:
    """
    Return the on-disk path for a dataset JSON Schema identifier.

    Parameters
    ----------
    schema_id:
        Identifier without the ``.json`` suffix.

    Returns
    -------
    Path
        Filesystem path to the JSON Schema document.
    """
    root = Path("src/codeintel/config/schemas/export")
    return root / f"{schema_id}.json"


def _normalize_validation_profile(
    value: str | None,
) -> Literal["strict", "lenient"] | None:
    """
    Restrict validation profile to supported literals.

    Returns
    -------
    Literal["strict", "lenient"] | None
        Normalized validation profile when valid.
    """
    if value == "strict":
        return "strict"
    if value == "lenient":
        return "lenient"
    return None


@dataclass(frozen=True)
class BackendContext:
    """Shared context for DuckDB query services."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    graph_engine: GraphEngine | None = None


@dataclass
class GraphEngineProvider:
    """Resolve and cache a graph engine for DuckDB query services."""

    context: BackendContext
    graph_engine: GraphEngine | None = None
    _engine: GraphEngine | None = field(default=None, init=False, repr=False)

    def require(self) -> GraphEngine:
        """
        Return a graph engine or raise when unavailable.

        Returns
        -------
        GraphEngine
            Resolved graph engine for the configured repo/commit.

        Raises
        ------
        backend_failure
            When no graph engine is available.
        """
        if self._engine is not None:
            return self._engine
        if self.graph_engine is not None:
            self._engine = self.graph_engine
            return self._engine
        if self.context.graph_engine is None:
            message = "Graph engine must be provided to DuckDBQueryService."
            raise errors.backend_failure(message)
        self._engine = self.context.graph_engine
        return self._engine


@dataclass
class DuckDBRepositories:
    """Lazily constructed repositories for DuckDB-backed services."""

    gateway: StorageGateway
    repo: str
    commit: str
    _functions: FunctionRepository | None = field(default=None, init=False, repr=False)
    _modules: ModuleRepository | None = field(default=None, init=False, repr=False)
    _subsystems: SubsystemRepository | None = field(default=None, init=False, repr=False)
    _tests: TestRepository | None = field(default=None, init=False, repr=False)
    _datasets: DatasetReadRepository | None = field(default=None, init=False, repr=False)
    _graphs: GraphRepository | None = field(default=None, init=False, repr=False)

    @property
    def functions(self) -> FunctionRepository:
        """Return a lazily constructed function repository."""
        if self._functions is None:
            self._functions = FunctionRepository(self.gateway, self.repo, self.commit)
        return self._functions

    @property
    def modules(self) -> ModuleRepository:
        """Return a lazily constructed module repository."""
        if self._modules is None:
            self._modules = ModuleRepository(self.gateway, self.repo, self.commit)
        return self._modules

    @property
    def subsystems(self) -> SubsystemRepository:
        """Return a lazily constructed subsystem repository."""
        if self._subsystems is None:
            self._subsystems = SubsystemRepository(self.gateway, self.repo, self.commit)
        return self._subsystems

    @property
    def tests(self) -> TestRepository:
        """Return a lazily constructed test repository."""
        if self._tests is None:
            self._tests = TestRepository(self.gateway, self.repo, self.commit)
        return self._tests

    @property
    def datasets(self) -> DatasetReadRepository:
        """Return a lazily constructed dataset repository."""
        if self._datasets is None:
            self._datasets = DatasetReadRepository(self.gateway, self.repo, self.commit)
        return self._datasets

    @property
    def graphs(self) -> GraphRepository:
        """Return a lazily constructed graph repository."""
        if self._graphs is None:
            self._graphs = GraphRepository(self.gateway, self.repo, self.commit)
        return self._graphs


@dataclass
class _FunctionQueries:
    context: BackendContext
    repositories: DuckDBRepositories
    engine_provider: GraphEngineProvider

    @property
    def con(self) -> DuckDBConnection:
        return self.context.gateway.con

    @property
    def functions(self) -> FunctionRepository:
        return self.repositories.functions

    @property
    def graphs(self) -> GraphRepository:
        return self.repositories.graphs

    def _require_graph_engine(self) -> GraphEngine:
        return self.engine_provider.require()

    def _resolve_function_goid(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
    ) -> int | None:
        try:
            return self.functions.resolve_function_goid(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
            )
        except ValueError as exc:
            message = str(exc)
            raise errors.backend_failure(message) from exc

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphRunScope | None = None,
    ) -> FunctionSummaryResponse:
        _ = scope
        meta = ResponseMeta()
        if goid_h128 is None and not (urn or (rel_path and qualname)):
            message = "Must provide urn or goid_h128 or (rel_path + qualname)."
            raise errors.invalid_argument(message)
        resolved = self._resolve_function_goid(
            urn=urn,
            goid_h128=goid_h128,
            rel_path=rel_path,
            qualname=qualname,
        )
        if resolved is None:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="Function not found",
                    context={
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                    },
                )
            )
            return FunctionSummaryResponse(found=False, summary=None, meta=meta)
        row = self.functions.get_function_summary_by_goid(resolved)
        if row is None:
            meta.messages.append(
                Message(
                    code="not_found",
                    severity="info",
                    detail="Function not found",
                    context={
                        "urn": urn,
                        "goid_h128": goid_h128,
                        "rel_path": rel_path,
                        "qualname": qualname,
                    },
                )
            )
            return FunctionSummaryResponse(found=False, summary=None, meta=meta)
        return FunctionSummaryResponse(
            found=True, summary=FunctionSummaryRow.model_validate(row), meta=meta
        )

    def list_high_risk_functions(
        self,
        *,
        min_risk: float = 0.7,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphRunScope | None = None,
    ) -> HighRiskFunctionsResponse:
        _ = scope
        limit_clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.functions.list_high_risk_functions(
            min_risk=min_risk,
            limit=limit_clamp.applied,
            tested_only=tested_only,
        )
        return HighRiskFunctionsResponse(
            functions=[
                ViewRow.model_validate(
                    {"repo": self.context.repo, "commit": self.context.commit, **r}
                )
                for r in rows
            ],
            meta=ResponseMeta(messages=limit_clamp.messages),
        )

    def get_callgraph_neighbors(
        self,
        *,
        goid_h128: int,
        direction: str = "both",
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> CallGraphNeighborsResponse:
        _ = scope
        limit_clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        outgoing_rows: list[CallGraphEdgeRow] = []
        incoming_rows: list[CallGraphEdgeRow] = []
        if direction in {"outgoing", "both"}:
            outgoing = self.graphs.get_outgoing_callgraph_neighbors(
                goid_h128, limit=limit_clamp.applied
            )
            outgoing_rows = [CallGraphEdgeRow.model_validate(edge) for edge in outgoing]
        if direction in {"incoming", "both"}:
            incoming = self.graphs.get_incoming_callgraph_neighbors(
                goid_h128, limit=limit_clamp.applied
            )
            incoming_rows = [CallGraphEdgeRow.model_validate(edge) for edge in incoming]
        return CallGraphNeighborsResponse(
            outgoing=outgoing_rows,
            incoming=incoming_rows,
            meta=ResponseMeta(messages=limit_clamp.messages),
        )

    def get_tests_for_function(
        self,
        *,
        goid_h128: int | None = None,
        urn: str | None = None,
        limit: int | None = None,
        scope: GraphRunScope | None = None,
    ) -> TestsForFunctionResponse:
        _ = scope
        resolved = goid_h128
        if resolved is None:
            resolved = self._resolve_function_goid(urn=urn)
        if resolved is None:
            return TestsForFunctionResponse(
                tests=[],
                meta=ResponseMeta(
                    messages=[
                        Message(
                            code="not_found",
                            severity="warning",
                            detail="Function not found",
                        )
                    ]
                ),
            )
        limit_clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        tests = self.repositories.tests.get_tests_for_function(resolved, limit=limit_clamp.applied)
        messages = list(limit_clamp.messages)
        if not tests:
            messages.append(
                Message(
                    code="not_found",
                    severity="warning",
                    detail="Tests not found for function",
                )
            )
        return TestsForFunctionResponse(
            tests=[ViewRow.model_validate(r) for r in tests],
            meta=ResponseMeta(messages=messages),
        )

    def get_callgraph_neighborhood(
        self,
        *,
        goid_h128: int,
        radius: int = 1,
        max_nodes: int | None = None,
    ) -> GraphNeighborhoodResponse:
        """
        Compute a bounded ego neighborhood in the call graph.

        Parameters
        ----------
        goid_h128
            GOID of the function to center the neighborhood on.
        radius
            Hop radius for ego graph computation.
        max_nodes
            Optional node cap; when provided, truncates result set.

        Returns
        -------
        GraphNeighborhoodResponse
            Nodes and edges in the neighborhood with truncation metadata.
        """
        engine = self._require_graph_engine()
        graph = engine.call_graph()
        if goid_h128 not in graph:
            meta = ResponseMeta(
                applied_limit=max_nodes,
                truncated=max_nodes is not None and max_nodes == 0,
            )
            return GraphNeighborhoodResponse(nodes=[], edges=[], meta=meta)
        subgraph = nx.ego_graph(graph, goid_h128, radius=radius, center=True)
        original_node_count = subgraph.number_of_nodes()
        truncated = False
        if max_nodes is not None and original_node_count > max_nodes:
            trimmed_nodes = list(subgraph.nodes)[:max_nodes]
            subgraph = subgraph.subgraph(trimmed_nodes).copy()
            truncated = True
        nodes: list[ViewRow] = []
        for node in subgraph.nodes:
            summary_row = self.functions.get_function_summary_by_goid(int(node))
            if summary_row is None:
                summary_row = {
                    "repo": self.context.repo,
                    "commit": self.context.commit,
                    "rel_path": "",
                    "function_goid_h128": int(node),
                }
            nodes.append(ViewRow.model_validate(summary_row))

        edges: list[ViewRow] = []
        for u, v, data in subgraph.edges(data=True):
            edges.append(
                ViewRow.model_validate(
                    {
                        "caller_goid_h128": int(u),
                        "caller_repo": self.context.repo,
                        "caller_commit": self.context.commit,
                        "callee_goid_h128": int(v),
                        "callee_repo": self.context.repo,
                        "callee_commit": self.context.commit,
                        "callsite_path": str(data.get("path")) if isinstance(data, dict) else None,
                        "callsite_line": int(data.get("line_number", 0))
                        if isinstance(data, dict)
                        else None,
                        "language": str(data.get("language", "python"))
                        if isinstance(data, dict)
                        else "python",
                        "kind": str(data.get("edge_type", "direct"))
                        if isinstance(data, dict)
                        else "direct",
                        "confidence": float(data.get("weight", 1.0))
                        if isinstance(data, dict)
                        else None,
                    }
                )
            )
        meta = ResponseMeta(
            applied_limit=max_nodes,
            truncated=truncated or (max_nodes is not None and max_nodes == 0),
        )
        return GraphNeighborhoodResponse(nodes=nodes, edges=edges, meta=meta)

    def get_import_boundary(
        self,
        *,
        subsystem_id: str,
        max_edges: int | None = None,
    ) -> ImportBoundaryResponse:
        """
        Return import edges crossing a subsystem boundary.

        Parameters
        ----------
        subsystem_id
            Subsystem identifier to find boundary edges for.
        max_edges
            Maximum edges to return; when provided, truncates result set.

        Returns
        -------
        ImportBoundaryResponse
            Boundary nodes and edges with truncation metadata.
        """
        limit_clamp = clamp_limit_value(
            max_edges,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        meta = ResponseMeta(
            messages=limit_clamp.messages,
            applied_limit=limit_clamp.applied,
            truncated=limit_clamp.applied == 0,
        )
        try:
            engine = self._require_graph_engine()
        except errors.McpError:
            return ImportBoundaryResponse(nodes=[], edges=[], meta=meta)
        import_graph = engine.import_graph()
        if subsystem_id not in import_graph:
            return ImportBoundaryResponse(nodes=[], edges=[], meta=meta)
        boundary_edges: list[dict[str, object]] = []
        boundary_nodes: set[str] = set()
        truncated = False
        for u, v, data in import_graph.out_edges(subsystem_id, data=True):
            if limit_clamp.applied is not None and len(boundary_edges) >= limit_clamp.applied:
                truncated = True
                break
            boundary_nodes.update({str(u), str(v)})
            boundary_edges.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "weight": float(data.get("weight", 1.0)) if isinstance(data, dict) else 1.0,
                }
            )
        for u, v, data in import_graph.in_edges(subsystem_id, data=True):
            if limit_clamp.applied is not None and len(boundary_edges) >= limit_clamp.applied:
                truncated = True
                break
            boundary_nodes.update({str(u), str(v)})
            boundary_edges.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "weight": float(data.get("weight", 1.0)) if isinstance(data, dict) else 1.0,
                }
            )
        nodes = [ViewRow.model_validate({"id": node}) for node in sorted(boundary_nodes)]
        edges = [ViewRow.model_validate(edge) for edge in boundary_edges]
        meta = ResponseMeta(
            messages=limit_clamp.messages,
            applied_limit=limit_clamp.applied,
            truncated=truncated or limit_clamp.applied == 0,
        )
        return ImportBoundaryResponse(nodes=nodes, edges=edges, meta=meta)

    def get_function_profile(self, goid_h128: int) -> FunctionProfileResponse:
        row = self.functions.get_function_profile(goid_h128)
        if row is None:
            message = f"Function profile not found: {goid_h128}"
            raise errors.not_found(message)
        return FunctionProfileResponse(
            found=True,
            profile=FunctionProfileRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def get_function_architecture(self, goid_h128: int) -> FunctionArchitectureResponse:
        """
        Fetch function architecture metrics by GOID.

        Queries the ``docs.v_function_architecture`` view which aggregates
        graph metrics from ``analytics.graph_metrics_functions`` and related
        tables. This follows the same repository pattern as module architecture.

        Parameters
        ----------
        goid_h128
            Function global object ID (128-bit hash).

        Returns
        -------
        FunctionArchitectureResponse
            Architecture metrics including call_fan_in, call_fan_out,
            pagerank, betweenness, closeness, and other graph centrality
            measures.

        Raises
        ------
        errors.not_found
            When no architecture record exists for the given GOID.
        """
        row = self.functions.get_function_architecture(goid_h128)
        if row is None:
            message = f"Function architecture not found: {goid_h128}"
            raise errors.not_found(message)
        return FunctionArchitectureResponse(
            found=True,
            architecture=ViewRow.model_validate(row),
            meta=ResponseMeta(),
        )


@dataclass
class _ModuleQueries:
    context: BackendContext
    repositories: DuckDBRepositories
    engine_provider: GraphEngineProvider

    @property
    def con(self) -> DuckDBConnection:
        return self.context.gateway.con

    @property
    def modules(self) -> ModuleRepository:
        return self.repositories.modules

    @property
    def subsystems(self) -> SubsystemRepository:
        return self.repositories.subsystems

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        row = self.modules.get_file_profile(rel_path)
        if row is None:
            return FileProfileResponse(
                found=False,
                profile=None,
                meta=ResponseMeta(
                    messages=[
                        Message(
                            code="not_found",
                            severity="warning",
                            detail=f"File profile not found: {rel_path}",
                        )
                    ]
                ),
            )
        return FileProfileResponse(
            found=True, profile=FileProfileRow.model_validate(row), meta=ResponseMeta()
        )

    def get_file_summary(
        self, *, rel_path: str, scope: GraphRunScope | None = None
    ) -> FileSummaryResponse:
        _ = scope
        row = self.modules.get_file_summary(rel_path)
        if row is None:
            message = f"File summary not found: {rel_path}"
            raise errors.not_found(message)
        return FileSummaryResponse(
            found=True, file=FileSummaryRow.model_validate(row), meta=ResponseMeta()
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        row = self.modules.get_module_profile(module)
        if row is None:
            message = f"Module profile not found: {module}"
            raise errors.not_found(message)
        return ModuleProfileResponse(
            found=True,
            profile=ModuleProfileRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        row = self.modules.get_module_architecture(module)
        if row is None:
            message = f"Module architecture not found: {module}"
            raise errors.not_found(message)
        return ModuleArchitectureResponse(
            found=True,
            architecture=ModuleArchitectureRow.model_validate(row),
            meta=ResponseMeta(),
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        """
        Fetch IDE-focused hints for a file by relative path.

        Queries the ``docs.v_ide_hints`` view which aggregates module metrics,
        subsystem context, and risk indicators for IDE consumption.

        Parameters
        ----------
        rel_path
            Relative file path within the repository.

        Returns
        -------
        FileHintsResponse
            Hint rows including subsystem_name, risk levels, and fan metrics.
        """
        hints = self.modules.get_file_hints(rel_path)
        return FileHintsResponse(
            found=True,
            hints=[ViewRow.model_validate(hint) for hint in hints],
            meta=ResponseMeta(),
        )


@dataclass
class _SubsystemQueries:
    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def subsystems(self) -> SubsystemRepository:
        return self.repositories.subsystems

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSummaryResponse:
        limit_clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.list_subsystems(limit=limit_clamp.applied, role=role, query=q)
        return SubsystemSummaryResponse(
            subsystems=[SubsystemSummaryRow.model_validate(r) for r in rows],
            meta=ResponseMeta(messages=limit_clamp.messages),
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        rows = self.subsystems.list_subsystems_for_module(module)
        return ModuleSubsystemResponse(
            found=True,
            memberships=[ModuleWithSubsystemRow.model_validate(r) for r in rows],
            meta=ResponseMeta(),
        )

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        """
        Fetch subsystem detail and member modules by subsystem identifier.

        Queries the ``docs.v_subsystem_summary`` view for subsystem metadata
        and ``docs.v_subsystem_modules`` for membership rows. This follows the
        repository pattern consistent with other architecture endpoints.

        Parameters
        ----------
        subsystem_id
            Unique subsystem identifier.
        module_limit
            Maximum modules to return (defaults to backend limit).

        Returns
        -------
        SubsystemModulesResponse
            Subsystem summary and module membership list.
        """
        limit_clamp = clamp_limit_value(
            module_limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        subsystem_row = self.subsystems.get_subsystem_summary(subsystem_id)
        if subsystem_row is None:
            return SubsystemModulesResponse(
                found=False,
                subsystem=None,
                modules=[],
                meta=ResponseMeta(messages=limit_clamp.messages),
            )
        rows = self.subsystems.list_subsystem_modules(subsystem_id)[: limit_clamp.applied]
        return SubsystemModulesResponse(
            found=True,
            subsystem=SubsystemSummaryRow.model_validate(subsystem_row),
            modules=[ModuleWithSubsystemRow.model_validate(r) for r in rows],
            meta=ResponseMeta(messages=limit_clamp.messages),
        )

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        limit_clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.search_subsystems(limit=limit_clamp.applied, role=role, query=q)
        return SubsystemSearchResponse(
            subsystems=[SubsystemSummaryRow.model_validate(r) for r in rows],
            meta=ResponseMeta(messages=limit_clamp.messages),
        )

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        return self.get_subsystem_modules(subsystem_id=subsystem_id, module_limit=module_limit)

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        limit_clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.list_subsystem_profiles(limit=limit_clamp.applied)
        return SubsystemProfileResponse(
            profiles=[SubsystemProfileRow.model_validate(r) for r in rows],
            meta=ResponseMeta(messages=limit_clamp.messages),
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        limit_clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        rows = self.subsystems.list_subsystem_coverage(limit=limit_clamp.applied)
        messages = list(limit_clamp.messages)
        if not rows:
            messages.append(
                Message(
                    code="not_found",
                    severity="warning",
                    detail="No subsystem coverage found",
                )
            )
        return SubsystemCoverageResponse(
            coverage=[SubsystemCoverageRow.model_validate(r) for r in rows],
            meta=ResponseMeta(messages=messages),
        )


@dataclass
class _DatasetQueries:
    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def datasets(self) -> DatasetReadRepository:
        return self.repositories.datasets

    @property
    def gateway(self) -> StorageGateway:
        return self.context.gateway

    def list_datasets(self) -> list[DatasetSpecDescriptor]:
        registry = load_dataset_registry(self.gateway.con)
        specs = list_dataset_specs(registry)
        return [DatasetSpecDescriptor.model_validate(dict(spec)) for spec in specs]

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        registry = load_dataset_registry(self.gateway.con)
        specs = list_dataset_specs(registry)
        sorted_specs = sorted(specs, key=lambda spec: cast("str", spec["name"]))
        results: list[DatasetSpecDescriptor] = []
        for spec in sorted_specs:
            normalized: dict[str, object] = dict(spec)
            normalized["schema_columns"] = list(cast("list[str]", spec["schema_columns"]))
            normalized["upstream_dependencies"] = list(
                cast("list[str]", spec.get("upstream_dependencies", []))
            )
            normalized["capabilities"] = dict(cast("dict[str, bool]", spec.get("capabilities", {})))
            normalized["validation_profile"] = _normalize_validation_profile(
                cast("str | None", spec.get("validation_profile"))
            )
            results.append(DatasetSpecDescriptor.model_validate(normalized))
        return results

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Mapping[str, object]]:
        registry = load_dataset_registry(self.gateway.con)
        ds = dataset_for_name(registry, dataset_name)
        clamp = clamp_limit_value(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        return self.datasets.read_dataset_rows(
            table_key=ds.table_key,
            limit=clamp.applied,
            offset=offset,
        )

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> DatasetSchemaResponse:
        registry = load_dataset_registry(self.gateway.con)
        try:
            ds = dataset_for_name(registry, dataset_name)
        except KeyError as exc:
            message = f"Unknown dataset: {dataset_name}"
            raise errors.not_found(message) from exc
        duckdb_schema = _fetch_duckdb_schema(self.gateway.con, ds.table_key)
        sample_rows = self.datasets.read_dataset_rows(
            table_key=ds.table_key,
            limit=sample_limit,
            offset=0,
        )
        return DatasetSchemaResponse(
            dataset=dataset_name,
            table_key=ds.table_key,
            duckdb_schema=duckdb_schema,
            json_schema=_load_json_schema(ds),
            sample_rows=[ViewRow.model_validate(row) for row in sample_rows],
            capabilities=ds.capabilities(),
            owner=ds.owner,
            freshness_sla=ds.freshness_sla,
            retention_policy=ds.retention_policy,
            schema_version=ds.schema_version,
            stable_id=ds.stable_id,
            validation_profile=_normalize_validation_profile(ds.validation_profile),
        )


@dataclass
class DuckDBQueryService:
    """Shared query runner facade delegating to internal helpers."""

    context: BackendContext
    repositories: DuckDBRepositories
    engine_provider: GraphEngineProvider
    _functions: _FunctionQueries = field(init=False, repr=False)
    _modules: _ModuleQueries = field(init=False, repr=False)
    _subsystems: _SubsystemQueries = field(init=False, repr=False)
    _datasets: _DatasetQueries = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Construct helper delegates backed by shared context/repos."""
        self._functions = _FunctionQueries(self.context, self.repositories, self.engine_provider)
        self._modules = _ModuleQueries(self.context, self.repositories, self.engine_provider)
        self._subsystems = _SubsystemQueries(self.context, self.repositories)
        self._datasets = _DatasetQueries(self.context, self.repositories)

    def __getattr__(self, name: str) -> object:
        """
        Delegate attribute lookups to the internal helpers.

        Returns
        -------
        object
            Attribute fetched from a helper when available.

        Raises
        ------
        AttributeError
            When the attribute is not found on any helper.
        """
        for helper in (self._functions, self._modules, self._subsystems, self._datasets):
            if hasattr(helper, name):
                return getattr(helper, name)
        raise AttributeError(name)

    @property
    def con(self) -> DuckDBConnection:
        """Return the underlying DuckDB connection."""
        return self.context.gateway.con

    @property
    def gateway(self) -> StorageGateway:
        """Expose the storage gateway for callers needing direct access."""
        return self.context.gateway

    @property
    def limits(self) -> BackendLimits:
        """Expose backend limits for services consuming the query facade."""
        return self.context.limits

    @property
    def graph_engine(self) -> GraphEngine:
        """Expose the configured graph engine for reuse across services."""
        return self.engine_provider.require()

    def __dir__(self) -> list[str]:
        """
        Expose combined attributes for better introspection.

        Returns
        -------
        list[str]
            Sorted attribute names across helpers and facade.
        """
        attrs = {
            "context",
            "repositories",
            "engine_provider",
            "con",
            "gateway",
            "limits",
            "functions",
            "modules",
            "subsystems",
            "datasets",
        }
        for helper in (self._functions, self._modules, self._subsystems, self._datasets):
            attrs.update(dir(helper))
        return sorted(attrs)

    @property
    def functions(self) -> FunctionQueriesApi:
        """Helper for function queries."""
        return self._functions

    @property
    def modules(self) -> ProfileQueriesApi:
        """Helper for module/file queries."""
        return self._modules

    @property
    def subsystems(self) -> SubsystemQueriesApi:
        """Helper for subsystem queries."""
        return self._subsystems

    @property
    def datasets(self) -> DatasetQueriesApi:
        """Helper for dataset queries."""
        return self._datasets


__all__ = ["BackendContext", "DuckDBQueryService", "DuckDBRepositories", "GraphEngineProvider"]
