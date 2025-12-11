"""Helpers for constructing domain payloads from repository rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Literal, Protocol, cast, runtime_checkable

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.view_utils import normalize_entrypoints_rows


@runtime_checkable
class _SupportsModelDump(Protocol):
    def model_dump(self) -> dict[str, object]:
        """Return a dictionary representation of the object.

        Returns
        -------
        dict[str, object]
            Dictionary of model fields.
        """
        ...


RowDict = Mapping[str, object]


def _ensure_meta(meta: dm.ResponseMeta | None) -> dm.ResponseMeta:
    """Return provided metadata or an empty instance.

    Returns
    -------
    dm.ResponseMeta
        Provided metadata or a new empty metadata container.
    """
    return meta or dm.ResponseMeta()


def _to_dict(row: RowDict | _SupportsModelDump | object) -> dict[str, object]:
    """Convert mapping-like inputs to plain dictionaries.

    Returns
    -------
    dict[str, object]
        Dictionary representation of the row.
    """
    if isinstance(row, Mapping):
        return dict(row)
    if isinstance(row, _SupportsModelDump):
        return row.model_dump()
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    return cast("dict[str, object]", row)


def build_function_summary(
    row: RowDict | None,
    meta: dm.ResponseMeta | None = None,
) -> dm.FunctionSummaryResult:
    """Construct a FunctionSummaryResult from an optional row.

    Returns
    -------
    dm.FunctionSummaryResult
        Domain function summary payload.
    """
    meta = _ensure_meta(meta)
    if row is None:
        return dm.FunctionSummaryResult(found=False, summary=None, meta=meta)
    return dm.FunctionSummaryResult(found=True, summary=_to_dict(row), meta=meta)


def build_high_risk_functions(
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.HighRiskFunctionsResult:
    """Construct HighRiskFunctionsResult from repository rows.

    Returns
    -------
    dm.HighRiskFunctionsResult
        Domain high-risk listing.
    """
    meta = _ensure_meta(meta)
    return dm.HighRiskFunctionsResult(
        functions=[_to_dict(row) for row in rows],
        truncated=bool(meta.truncated),
        meta=meta,
    )


def build_callgraph_neighbors(
    outgoing: Sequence[RowDict],
    incoming: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.CallGraphNeighbors:
    """Build call graph neighbor payload with shared metadata.

    Returns
    -------
    dm.CallGraphNeighbors
        Domain call graph neighbors.
    """
    meta = _ensure_meta(meta)
    return dm.CallGraphNeighbors(
        outgoing=[_to_dict(row) for row in outgoing],
        incoming=[_to_dict(row) for row in incoming],
        meta=meta,
    )


def build_tests_for_function(
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.TestsForFunctionResult:
    """Build TestsForFunctionResult from test rows.

    Returns
    -------
    dm.TestsForFunctionResult
        Domain tests-for-function payload.
    """
    meta = _ensure_meta(meta)
    return dm.TestsForFunctionResult(tests=[_to_dict(row) for row in rows], meta=meta)


def build_graph_neighborhood(
    nodes: Sequence[RowDict],
    edges: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.GraphNeighborhood:
    """Build GraphNeighborhood payload with nodes and edges.

    Returns
    -------
    dm.GraphNeighborhood
        Domain graph neighborhood.
    """
    meta = _ensure_meta(meta)
    return dm.GraphNeighborhood(
        nodes=[_to_dict(node) for node in nodes],
        edges=[_to_dict(edge) for edge in edges],
        meta=meta,
    )


def build_import_boundary(
    nodes: Sequence[str],
    edges: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.ImportBoundary:
    """Build ImportBoundary payload for boundary graphs.

    Returns
    -------
    dm.ImportBoundary
        Domain import boundary payload.
    """
    meta = _ensure_meta(meta)
    return dm.ImportBoundary(
        nodes=[{"id": node} for node in nodes],
        edges=[_to_dict(edge) for edge in edges],
        meta=meta,
    )


def build_file_summary(
    row: RowDict | None,
    rel_path: str,
    meta: dm.ResponseMeta | None = None,
) -> dm.FileSummaryResult:
    """Build FileSummaryResult for a single file path.

    Returns
    -------
    dm.FileSummaryResult
        Domain file summary payload.
    """
    meta = _ensure_meta(meta)
    if row is None:
        meta.messages.append(
            dm.Message(
                code="file_not_found",
                severity="info",
                detail=f"No summary for {rel_path}",
            )
        )
        return dm.FileSummaryResult(found=False, file=None, meta=meta)
    file_summary = _to_dict(row)
    if "rel_path" not in file_summary:
        file_summary["rel_path"] = rel_path
    return dm.FileSummaryResult(found=True, file=file_summary, meta=meta)


def build_function_profile(
    row: RowDict | None,
    meta: dm.ResponseMeta | None = None,
) -> dm.FunctionProfileResult:
    """Build FunctionProfileResult from an optional profile row.

    Returns
    -------
    dm.FunctionProfileResult
        Domain function profile payload.
    """
    meta = _ensure_meta(meta)
    if row is None:
        return dm.FunctionProfileResult(found=False, profile=None, meta=meta)
    return dm.FunctionProfileResult(found=True, profile=_to_dict(row), meta=meta)


def build_file_profile(
    row: RowDict | None,
    meta: dm.ResponseMeta | None = None,
) -> dm.FileProfileResult:
    """Build FileProfileResult from an optional profile row.

    Returns
    -------
    dm.FileProfileResult
        Domain file profile payload.
    """
    meta = _ensure_meta(meta)
    if row is None:
        return dm.FileProfileResult(found=False, profile=None, meta=meta)
    return dm.FileProfileResult(found=True, profile=_to_dict(row), meta=meta)


def build_module_profile(
    row: RowDict | None,
    meta: dm.ResponseMeta | None = None,
) -> dm.ModuleProfileResult:
    """Build ModuleProfileResult from an optional profile row.

    Returns
    -------
    dm.ModuleProfileResult
        Domain module profile payload.
    """
    meta = _ensure_meta(meta)
    if row is None:
        return dm.ModuleProfileResult(found=False, profile=None, meta=meta)
    return dm.ModuleProfileResult(found=True, profile=_to_dict(row), meta=meta)


def build_function_architecture(
    row: RowDict | None,
    meta: dm.ResponseMeta | None = None,
) -> dm.FunctionArchitectureResult:
    """Build FunctionArchitectureResult from an optional row.

    Returns
    -------
    dm.FunctionArchitectureResult
        Domain function architecture payload.
    """
    meta = _ensure_meta(meta)
    if row is None:
        return dm.FunctionArchitectureResult(found=False, architecture=None, meta=meta)
    return dm.FunctionArchitectureResult(found=True, architecture=_to_dict(row), meta=meta)


def build_module_architecture(
    row: RowDict | None,
    meta: dm.ResponseMeta | None = None,
) -> dm.ModuleArchitectureResult:
    """Build ModuleArchitectureResult from an optional row.

    Returns
    -------
    dm.ModuleArchitectureResult
        Domain module architecture payload.
    """
    meta = _ensure_meta(meta)
    if row is None:
        return dm.ModuleArchitectureResult(found=False, architecture=None, meta=meta)
    return dm.ModuleArchitectureResult(found=True, architecture=_to_dict(row), meta=meta)


def build_subsystem_summary(
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.SubsystemSummaryResult:
    """Build SubsystemSummaryResult from summary rows.

    Returns
    -------
    dm.SubsystemSummaryResult
        Domain subsystem summaries.
    """
    meta = _ensure_meta(meta)
    subsystems = [_to_dict(row) for row in rows]
    normalize_entrypoints_rows(subsystems)
    return dm.SubsystemSummaryResult(subsystems=subsystems, meta=meta)


def build_module_subsystems(
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.ModuleSubsystemResult:
    """Build ModuleSubsystemResult from membership rows.

    Returns
    -------
    dm.ModuleSubsystemResult
        Domain module-to-subsystem memberships.
    """
    meta = _ensure_meta(meta)
    return dm.ModuleSubsystemResult(
        found=True,
        memberships=[_to_dict(row) for row in rows],
        meta=meta,
    )


def build_file_hints(
    rows: Sequence[RowDict],
    rel_path: str,
    meta: dm.ResponseMeta | None = None,
) -> dm.FileHintsResult:
    """Build FileHintsResult from IDE hint rows.

    Returns
    -------
    dm.FileHintsResult
        Domain file hints payload.
    """
    meta = _ensure_meta(meta)
    hints = [_to_dict(row) for row in rows]
    if not hints:
        meta.messages.append(
            dm.Message(
                code="hints_not_found",
                severity="info",
                detail=f"No IDE hints found for {rel_path}",
            )
        )
        return dm.FileHintsResult(found=False, hints=[], meta=meta)
    return dm.FileHintsResult(found=True, hints=hints, meta=meta)


def build_subsystem_modules(
    subsystem: RowDict | None,
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.SubsystemModulesResult:
    """Build SubsystemModulesResult from subsystem and module rows.

    Returns
    -------
    dm.SubsystemModulesResult
        Domain subsystem modules payload.
    """
    meta = _ensure_meta(meta)
    if subsystem is None:
        return dm.SubsystemModulesResult(
            found=False,
            subsystem=None,
            modules=[],
            meta=meta,
        )
    return dm.SubsystemModulesResult(
        found=True,
        subsystem=_to_dict(subsystem),
        modules=[_to_dict(row) for row in rows],
        meta=meta,
    )


def build_subsystem_search(
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.SubsystemSearchResult:
    """Build SubsystemSearchResult from search result rows.

    Returns
    -------
    dm.SubsystemSearchResult
        Domain subsystem search payload.
    """
    meta = _ensure_meta(meta)
    return dm.SubsystemSearchResult(subsystems=[_to_dict(row) for row in rows], meta=meta)


def build_subsystem_profile(
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.SubsystemProfileResult:
    """Build SubsystemProfileResult from profile rows.

    Returns
    -------
    dm.SubsystemProfileResult
        Domain subsystem profiles payload.
    """
    meta = _ensure_meta(meta)
    profiles: list[RowDict] = [_to_dict(row) for row in rows]
    normalize_entrypoints_rows(profiles)
    return dm.SubsystemProfileResult(profiles=cast("list[object]", profiles), meta=meta)


def build_subsystem_coverage(
    rows: Sequence[RowDict],
    meta: dm.ResponseMeta | None = None,
) -> dm.SubsystemCoverageResult:
    """Build SubsystemCoverageResult from coverage rows.

    Returns
    -------
    dm.SubsystemCoverageResult
        Domain subsystem coverage payload.
    """
    meta = _ensure_meta(meta)
    return dm.SubsystemCoverageResult(coverage=[_to_dict(row) for row in rows], meta=meta)


@dataclass(frozen=True)
class DatasetSchemaInput:
    """Structured input for dataset schema builders."""

    dataset_name: str
    table_key: str
    duckdb_schema: Sequence[RowDict | object]
    json_schema: dict[str, object] | None
    sample_rows: Sequence[RowDict | object]
    capabilities: Mapping[str, bool]
    owner: str | None
    freshness_sla: str | None
    retention_policy: str | None
    schema_version: str | None
    stable_id: str | None
    validation_profile: Literal["strict", "lenient"] | None
    meta: dm.ResponseMeta | None = None


def build_dataset_schema(data: DatasetSchemaInput) -> dm.DatasetSchema:
    """Build DatasetSchema domain payload from structured components.

    Returns
    -------
    dm.DatasetSchema
        Domain dataset schema payload.
    """
    resolved_meta = _ensure_meta(data.meta)
    return dm.DatasetSchema(
        dataset_name=data.dataset_name,
        table_key=data.table_key,
        duckdb_schema=[_to_dict(column) for column in data.duckdb_schema],
        json_schema=data.json_schema,
        sample_rows=[_to_dict(row) for row in data.sample_rows],
        capabilities=dict(data.capabilities),
        owner=data.owner,
        freshness_sla=data.freshness_sla,
        retention_policy=data.retention_policy,
        schema_version=data.schema_version,
        stable_id=data.stable_id,
        validation_profile=data.validation_profile,
        meta=resolved_meta,
    )


__all__ = [
    "DatasetSchemaInput",
    "RowDict",
    "build_callgraph_neighbors",
    "build_dataset_schema",
    "build_file_hints",
    "build_file_profile",
    "build_file_summary",
    "build_function_architecture",
    "build_function_profile",
    "build_function_summary",
    "build_graph_neighborhood",
    "build_high_risk_functions",
    "build_import_boundary",
    "build_module_architecture",
    "build_module_profile",
    "build_module_subsystems",
    "build_subsystem_coverage",
    "build_subsystem_modules",
    "build_subsystem_profile",
    "build_subsystem_search",
    "build_subsystem_summary",
    "build_tests_for_function",
]
