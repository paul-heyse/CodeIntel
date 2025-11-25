"""Typed builders and insert helpers for seeded DuckDB test data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.storage.gateway import StorageGateway


def _iso(dt: datetime | None = None) -> str:
    """
    Return an ISO-8601 timestamp with timezone.

    Returns
    -------
    str
        Timestamp in ISO format with timezone.
    """
    return (dt or datetime.now(UTC)).isoformat()


@dataclass(frozen=True)
class RepoMapRow:
    """Row for core.repo_map."""

    repo: str
    commit: str
    modules: dict[str, str]
    overlays: dict[str, str] | None = None
    generated_at: datetime | None = None

    def to_tuple(self) -> tuple[str, str, str, str, str]:
        return (
            self.repo,
            self.commit,
            json.dumps(self.modules),
            json.dumps(self.overlays or {}),
            _iso(self.generated_at),
        )


@dataclass(frozen=True)
class ModuleRow:
    """Row for core.modules."""

    module: str
    path: str
    repo: str
    commit: str
    tags: str = "[]"
    owners: str = "[]"

    def to_tuple(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.module,
            self.path,
            self.repo,
            self.commit,
            "python",
            self.tags,
            self.owners,
        )


@dataclass(frozen=True)
class GoidRow:
    """Row for core.goids."""

    goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    kind: str
    qualname: str
    start_line: int
    end_line: int
    language: str = "python"
    created_at: datetime | None = None

    def to_tuple(self) -> tuple[int, str, str, str, str, str, str, str, int, int, str]:
        return (
            self.goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.language,
            self.kind,
            self.qualname,
            self.start_line,
            self.end_line,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class GoidCrosswalkRow:
    """Row for core.goid_crosswalk."""

    repo: str
    commit: str
    goid: str
    lang: str
    module_path: str
    file_path: str
    start_line: int
    end_line: int
    scip_symbol: str
    ast_qualname: str
    cst_node_id: str | None
    chunk_id: str | None
    symbol_id: str | None
    updated_at: datetime | None = None

    def to_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        str,
        str,
        str | None,
        str | None,
        str | None,
        str,
    ]:
        return (
            self.repo,
            self.commit,
            self.goid,
            self.lang,
            self.module_path,
            self.file_path,
            self.start_line,
            self.end_line,
            self.scip_symbol,
            self.ast_qualname,
            self.cst_node_id,
            self.chunk_id,
            self.symbol_id,
            _iso(self.updated_at),
        )


@dataclass(frozen=True)
class CallGraphNodeRow:
    """Row for graph.call_graph_nodes."""

    goid_h128: int
    language: str
    kind: str
    arity: int
    is_public: bool
    rel_path: str

    def to_tuple(self) -> tuple[int, str, str, int, bool, str]:
        return (
            self.goid_h128,
            self.language,
            self.kind,
            self.arity,
            self.is_public,
            self.rel_path,
        )


@dataclass(frozen=True)
class CallGraphEdgeRow:
    """Row for graph.call_graph_edges."""

    repo: str
    commit: str
    caller_goid_h128: int
    callee_goid_h128: int | None
    callsite_path: str
    callsite_line: int
    callsite_col: int
    language: str
    kind: str
    resolved_via: str
    confidence: float
    evidence: dict[str, object] | str | None = None

    def to_tuple(
        self,
    ) -> tuple[str, str, int, int | None, str, int, int, str, str, str, float, str]:
        evidence_json: str
        if self.evidence is None:
            evidence_json = "{}"
        elif isinstance(self.evidence, str):
            evidence_json = self.evidence
        else:
            evidence_json = json.dumps(self.evidence)
        return (
            self.repo,
            self.commit,
            self.caller_goid_h128,
            self.callee_goid_h128,
            self.callsite_path,
            self.callsite_line,
            self.callsite_col,
            self.language,
            self.kind,
            self.resolved_via,
            self.confidence,
            evidence_json,
        )


@dataclass(frozen=True)
class ImportGraphEdgeRow:
    """Row for graph.import_graph_edges."""

    repo: str
    commit: str
    src_module: str
    dst_module: str
    src_fan_out: int
    dst_fan_in: int
    cycle_group: int
    module_layer: int | None = None

    def to_tuple(self) -> tuple[str, str, str, str, int, int, int]:
        return (
            self.repo,
            self.commit,
            self.src_module,
            self.dst_module,
            self.src_fan_out,
            self.dst_fan_in,
            self.cycle_group,
        )


@dataclass(frozen=True)
class SymbolUseEdgeRow:
    """Row for graph.symbol_use_edges with GOID detail."""

    symbol: str
    def_path: str
    use_path: str
    same_file: bool
    same_module: bool
    def_goid_h128: int | None = None
    use_goid_h128: int | None = None

    def to_basic_tuple(self) -> tuple[str, str, str, bool, bool]:
        return (self.symbol, self.def_path, self.use_path, self.same_file, self.same_module)

    def to_detailed_tuple(
        self,
    ) -> tuple[str, str, str, bool, bool, int | None, int | None]:
        return (
            self.symbol,
            self.def_path,
            self.use_path,
            self.same_file,
            self.same_module,
            self.def_goid_h128,
            self.use_goid_h128,
        )


@dataclass(frozen=True)
class CFGBlockRow:
    """Row for graph.cfg_blocks."""

    function_goid_h128: int
    block_idx: int
    block_id: str
    label: str
    file_path: str
    start_line: int
    end_line: int
    kind: str
    stmts_json: str
    in_degree: int
    out_degree: int

    def to_tuple(
        self,
    ) -> tuple[int, int, str, str, str, int, int, str, str, int, int]:
        return (
            self.function_goid_h128,
            self.block_idx,
            self.block_id,
            self.label,
            self.file_path,
            self.start_line,
            self.end_line,
            self.kind,
            self.stmts_json,
            self.in_degree,
            self.out_degree,
        )


@dataclass(frozen=True)
class CFGEdgeRow:
    """Row for graph.cfg_edges."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    edge_kind: str | None

    def to_tuple(self) -> tuple[int, str, str, str | None]:
        return (
            self.function_goid_h128,
            self.src_block_id,
            self.dst_block_id,
            self.edge_kind,
        )


@dataclass(frozen=True)
class DFGEdgeRow:
    """Row for graph.dfg_edges."""

    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    src_var: str | None
    dst_var: str | None
    edge_kind: str | None
    via_phi: bool
    use_kind: str | None

    def to_tuple(
        self,
    ) -> tuple[int, str, str, str | None, str | None, str | None, bool, str | None]:
        return (
            self.function_goid_h128,
            self.src_block_id,
            self.dst_block_id,
            self.src_var,
            self.dst_var,
            self.edge_kind,
            self.via_phi,
            self.use_kind,
        )


@dataclass(frozen=True)
class DocstringRow:
    """Row for core.docstrings."""

    repo: str
    commit: str
    rel_path: str
    module: str
    qualname: str
    kind: str
    lineno: int
    end_lineno: int
    raw_docstring: str
    style: str
    short_desc: str
    long_desc: str
    params_json: str
    returns_json: str
    raises_json: str
    examples_json: str
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
    ]:
        return (
            self.repo,
            self.commit,
            self.rel_path,
            self.module,
            self.qualname,
            self.kind,
            self.lineno,
            self.end_lineno,
            self.raw_docstring,
            self.style,
            self.short_desc,
            self.long_desc,
            self.params_json,
            self.returns_json,
            self.raises_json,
            self.examples_json,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class FunctionMetricsRow:
    """Row for analytics.function_metrics."""

    function_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int
    end_line: int
    loc: int
    logical_loc: int
    param_count: int
    positional_params: int
    keyword_only_params: int
    has_varargs: bool
    has_varkw: bool
    is_async: bool
    is_generator: bool
    return_count: int
    yield_count: int
    raise_count: int
    cyclomatic_complexity: int
    max_nesting_depth: int
    stmt_count: int
    decorator_count: int
    has_docstring: bool
    complexity_bucket: str
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        bool,
        bool,
        bool,
        bool,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        bool,
        str,
        str,
    ]:
        return (
            self.function_goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.language,
            self.kind,
            self.qualname,
            self.start_line,
            self.end_line,
            self.loc,
            self.logical_loc,
            self.param_count,
            self.positional_params,
            self.keyword_only_params,
            self.has_varargs,
            self.has_varkw,
            self.is_async,
            self.is_generator,
            self.return_count,
            self.yield_count,
            self.raise_count,
            self.cyclomatic_complexity,
            self.max_nesting_depth,
            self.stmt_count,
            self.decorator_count,
            self.has_docstring,
            self.complexity_bucket,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class FunctionTypesRow:
    """Row for analytics.function_types."""

    function_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int
    end_line: int
    total_params: int
    annotated_params: int
    unannotated_params: int
    param_typed_ratio: float
    has_return_annotation: bool
    return_type: str
    return_type_source: str
    type_comment: str | None
    param_types_json: str
    fully_typed: bool
    partial_typed: bool
    untyped: bool
    typedness_bucket: str
    typedness_source: str
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        int,
        int,
        int,
        float,
        bool,
        str,
        str,
        str | None,
        str,
        bool,
        bool,
        bool,
        str,
        str,
        str,
    ]:
        return (
            self.function_goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.language,
            self.kind,
            self.qualname,
            self.start_line,
            self.end_line,
            self.total_params,
            self.annotated_params,
            self.unannotated_params,
            self.param_typed_ratio,
            self.has_return_annotation,
            self.return_type,
            self.return_type_source,
            self.type_comment,
            self.param_types_json,
            self.fully_typed,
            self.partial_typed,
            self.untyped,
            self.typedness_bucket,
            self.typedness_source,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class CoverageFunctionRow:
    """Row for analytics.coverage_functions."""

    function_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    start_line: int
    end_line: int
    executable_lines: int
    covered_lines: int
    coverage_ratio: float
    tested: bool
    untested_reason: str | None
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        int,
        int,
        float,
        bool,
        str | None,
        str,
    ]:
        return (
            self.function_goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.language,
            self.kind,
            self.qualname,
            self.start_line,
            self.end_line,
            self.executable_lines,
            self.covered_lines,
            self.coverage_ratio,
            self.tested,
            self.untested_reason,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class RiskFactorRow:
    """Row for analytics.goid_risk_factors."""

    function_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    language: str
    kind: str
    qualname: str
    loc: int
    logical_loc: int
    cyclomatic_complexity: int
    complexity_bucket: str
    typedness_bucket: str
    typedness_source: str
    hotspot_score: float
    file_typed_ratio: float
    static_error_count: int
    has_static_errors: bool
    executable_lines: int
    covered_lines: int
    coverage_ratio: float
    tested: bool
    test_count: int
    failing_test_count: int
    last_test_status: str
    risk_score: float
    risk_level: str
    tags: str
    owners: str
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        int,
        str,
        str,
        str,
        float,
        float,
        int,
        bool,
        int,
        int,
        float,
        bool,
        int,
        int,
        str,
        float,
        str,
        str,
        str,
        str,
    ]:
        return (
            self.function_goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.language,
            self.kind,
            self.qualname,
            self.loc,
            self.logical_loc,
            self.cyclomatic_complexity,
            self.complexity_bucket,
            self.typedness_bucket,
            self.typedness_source,
            self.hotspot_score,
            self.file_typed_ratio,
            self.static_error_count,
            self.has_static_errors,
            self.executable_lines,
            self.covered_lines,
            self.coverage_ratio,
            self.tested,
            self.test_count,
            self.failing_test_count,
            self.last_test_status,
            self.risk_score,
            self.risk_level,
            self.tags,
            self.owners,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class TestCatalogRow:
    """Row for analytics.test_catalog."""

    test_id: str
    repo: str
    commit: str
    rel_path: str
    qualname: str
    status: str
    created_at: datetime
    kind: str = "unit"
    test_goid_h128: int | None = None
    urn: str | None = None
    duration_ms: int | None = None
    markers: str = "[]"
    parametrized: bool = False
    flaky: bool = False

    def to_tuple(
        self,
    ) -> tuple[
        str,
        int | None,
        str | None,
        str,
        str,
        str,
        str,
        str,
        str,
        int | None,
        str,
        bool,
        bool,
        str,
    ]:
        return (
            self.test_id,
            self.test_goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.qualname,
            self.kind,
            self.status,
            self.duration_ms,
            self.markers,
            self.parametrized,
            self.flaky,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class TestCoverageEdgeRow:
    """Row for analytics.test_coverage_edges."""

    test_id: str
    function_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    qualname: str
    covered_lines: int
    executable_lines: int
    coverage_ratio: float
    last_status: str
    created_at: datetime
    test_goid_h128: int | None = None

    def to_tuple(
        self,
    ) -> tuple[
        str,
        int | None,
        int,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        float,
        str,
        str,
    ]:
        return (
            self.test_id,
            self.test_goid_h128,
            self.function_goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.qualname,
            self.covered_lines,
            self.executable_lines,
            self.coverage_ratio,
            self.last_status,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class TypednessRow:
    """Row for analytics.typedness."""

    repo: str
    commit: str
    path: str
    type_error_count: int
    annotation_ratio: str
    untyped_defs: int
    overlay_needed: bool

    def to_tuple(self) -> tuple[str, str, str, int, str, int, bool]:
        return (
            self.repo,
            self.commit,
            self.path,
            self.type_error_count,
            self.annotation_ratio,
            self.untyped_defs,
            self.overlay_needed,
        )


@dataclass(frozen=True)
class StaticDiagnosticsRow:
    """Row for analytics.static_diagnostics."""

    repo: str
    commit: str
    rel_path: str
    pyrefly_errors: int
    pyright_errors: int
    ruff_errors: int
    total_errors: int
    has_errors: bool

    def to_tuple(self) -> tuple[str, str, str, int, int, int, int, bool]:
        return (
            self.repo,
            self.commit,
            self.rel_path,
            self.pyrefly_errors,
            self.pyright_errors,
            self.ruff_errors,
            self.total_errors,
            self.has_errors,
        )


@dataclass(frozen=True)
class HotspotRow:
    """Row for analytics.hotspots."""

    rel_path: str
    commit_count: int
    author_count: int
    lines_added: int
    lines_deleted: int
    complexity: float
    score: float

    def to_tuple(self) -> tuple[str, int, int, int, int, float, float]:
        return (
            self.rel_path,
            self.commit_count,
            self.author_count,
            self.lines_added,
            self.lines_deleted,
            self.complexity,
            self.score,
        )


@dataclass(frozen=True)
class AstMetricsRow:
    """Row for core.ast_metrics."""

    rel_path: str
    node_count: int
    function_count: int
    class_count: int
    avg_depth: float
    max_depth: int
    complexity: float
    generated_at: datetime

    def to_tuple(
        self,
    ) -> tuple[str, int, int, int, float, int, float, str]:
        return (
            self.rel_path,
            self.node_count,
            self.function_count,
            self.class_count,
            self.avg_depth,
            self.max_depth,
            self.complexity,
            _iso(self.generated_at),
        )


@dataclass(frozen=True)
class FunctionValidationRow:
    """Row for analytics.function_validation."""

    repo: str
    commit: str
    rel_path: str
    qualname: str
    issue: str
    detail: str
    created_at: datetime

    def to_tuple(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.repo,
            self.commit,
            self.rel_path,
            self.qualname,
            self.issue,
            self.detail,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class ConfigValueRow:
    """Row for analytics.config_values."""

    repo: str
    commit: str
    config_path: str
    format: str
    key: str
    reference_paths: list[str]
    reference_modules: list[str]
    reference_count: int

    def to_tuple(self) -> tuple[str, str, str, str, str, str, int]:
        return (
            self.repo,
            self.commit,
            self.config_path,
            self.format,
            self.key,
            json.dumps(self.reference_paths),
            json.dumps(self.reference_modules),
            self.reference_count,
        )


@dataclass(frozen=True)
class GraphMetricsModulesExtRow:
    """Row for analytics.graph_metrics_modules_ext."""

    repo: str
    commit: str
    module: str
    import_betweenness: float
    import_closeness: float
    import_eigenvector: float
    import_harmonic: float
    import_k_core: int
    import_constraint: float
    import_effective_size: float
    import_community_id: int
    import_component_id: int
    import_component_size: int
    import_scc_id: int
    import_scc_size: int
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        float,
        float,
        float,
        float,
        int,
        float,
        float,
        int,
        int,
        int,
        int,
        int,
        str,
    ]:
        return (
            self.repo,
            self.commit,
            self.module,
            self.import_betweenness,
            self.import_closeness,
            self.import_eigenvector,
            self.import_harmonic,
            self.import_k_core,
            self.import_constraint,
            self.import_effective_size,
            self.import_community_id,
            self.import_component_id,
            self.import_component_size,
            self.import_scc_id,
            self.import_scc_size,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class SymbolGraphMetricsModulesRow:
    """Row for analytics.symbol_graph_metrics_modules."""

    repo: str
    commit: str
    module: str
    symbol_betweenness: float
    symbol_closeness: float
    symbol_eigenvector: float
    symbol_harmonic: float
    symbol_k_core: int
    symbol_constraint: float
    symbol_effective_size: float
    symbol_community_id: int
    symbol_component_id: int
    symbol_component_size: int
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        float,
        float,
        float,
        float,
        int,
        float,
        float,
        int,
        int,
        int,
        str,
    ]:
        return (
            self.repo,
            self.commit,
            self.module,
            self.symbol_betweenness,
            self.symbol_closeness,
            self.symbol_eigenvector,
            self.symbol_harmonic,
            self.symbol_k_core,
            self.symbol_constraint,
            self.symbol_effective_size,
            self.symbol_community_id,
            self.symbol_component_id,
            self.symbol_component_size,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class SubsystemModuleRow:
    """Row for analytics.subsystem_modules."""

    repo: str
    commit: str
    subsystem_id: str
    module: str
    role: str

    def to_tuple(self) -> tuple[str, str, str, str, str]:
        return (self.repo, self.commit, self.subsystem_id, self.module, self.role)


@dataclass(frozen=True)
class SubsystemRow:
    """Row for analytics.subsystems."""

    repo: str
    commit: str
    subsystem_id: str
    name: str
    description: str
    module_count: int
    modules_json: str
    entrypoints_json: str
    internal_edge_count: int
    external_edge_count: int
    fan_in: int
    fan_out: int
    function_count: int
    avg_risk_score: float | None
    max_risk_score: float | None
    high_risk_function_count: int
    risk_level: str
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        str,
        str,
        int,
        str,
        str,
        int,
        int,
        int,
        int,
        int,
        float | None,
        float | None,
        int,
        str,
        str,
    ]:
        return (
            self.repo,
            self.commit,
            self.subsystem_id,
            self.name,
            self.description,
            self.module_count,
            self.modules_json,
            self.entrypoints_json,
            self.internal_edge_count,
            self.external_edge_count,
            self.fan_in,
            self.fan_out,
            self.function_count,
            self.avg_risk_score,
            self.max_risk_score,
            self.high_risk_function_count,
            self.risk_level,
            _iso(self.created_at),
        )


def insert_repo_map(gateway: StorageGateway, rows: Iterable[RepoMapRow]) -> None:
    """Insert repo_map rows via gateway helper."""
    gateway.core.insert_repo_map(row.to_tuple() for row in rows)


def insert_modules(gateway: StorageGateway, rows: Iterable[ModuleRow]) -> None:
    """Insert modules via gateway helper."""
    gateway.con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_goids(gateway: StorageGateway, rows: Iterable[GoidRow]) -> None:
    """Insert GOIDs via gateway helper."""
    gateway.core.insert_goids(row.to_tuple() for row in rows)


def insert_call_graph_nodes(gateway: StorageGateway, rows: Iterable[CallGraphNodeRow]) -> None:
    """Insert call graph nodes via gateway helper."""
    gateway.graph.insert_call_graph_nodes(row.to_tuple() for row in rows)


def insert_call_graph_edges(gateway: StorageGateway, rows: Iterable[CallGraphEdgeRow]) -> None:
    """Insert call graph edges via gateway helper."""
    gateway.graph.insert_call_graph_edges(row.to_tuple() for row in rows)


def insert_import_graph_edges(gateway: StorageGateway, rows: Iterable[ImportGraphEdgeRow]) -> None:
    """Insert import graph edges, handling optional module layers."""
    layer_rows = [row for row in rows if row.module_layer is not None]
    base_rows = [row for row in rows if row.module_layer is None]
    if base_rows:
        gateway.graph.insert_import_graph_edges(row.to_tuple() for row in base_rows)
    if layer_rows:
        gateway.con.executemany(
            """
            INSERT INTO graph.import_graph_edges (
                repo, commit, src_module, dst_module, src_fan_out, dst_fan_in,
                cycle_group, module_layer
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.repo,
                    row.commit,
                    row.src_module,
                    row.dst_module,
                    row.src_fan_out,
                    row.dst_fan_in,
                    row.cycle_group,
                    row.module_layer,
                )
                for row in layer_rows
            ],
        )


def insert_symbol_use_edges(gateway: StorageGateway, rows: Iterable[SymbolUseEdgeRow]) -> None:
    """Insert symbol use edges with or without GOIDs."""
    detailed = [
        row for row in rows if row.def_goid_h128 is not None or row.use_goid_h128 is not None
    ]
    basic = [row for row in rows if row not in detailed]
    if basic:
        gateway.graph.insert_symbol_use_edges(row.to_basic_tuple() for row in basic)
    if detailed:
        gateway.con.executemany(
            """
            INSERT INTO graph.symbol_use_edges (
                symbol, def_path, use_path, same_file, same_module, def_goid_h128, use_goid_h128
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [row.to_detailed_tuple() for row in detailed],
        )


def insert_cfg_blocks(gateway: StorageGateway, rows: Iterable[CFGBlockRow]) -> None:
    """Insert CFG blocks via gateway helper."""
    gateway.graph.insert_cfg_blocks(row.to_tuple() for row in rows)


def insert_cfg_edges(gateway: StorageGateway, rows: Iterable[CFGEdgeRow]) -> None:
    """Insert CFG edges via gateway helper."""
    gateway.graph.insert_cfg_edges(row.to_tuple() for row in rows)


def insert_dfg_edges(gateway: StorageGateway, rows: Iterable[DFGEdgeRow]) -> None:
    """Insert DFG edges matching schema column order."""
    gateway.con.executemany(
        """
        INSERT INTO graph.dfg_edges (
            function_goid_h128, src_block_id, dst_block_id, src_var, dst_var, edge_kind,
            via_phi, use_kind
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_docstrings(gateway: StorageGateway, rows: Iterable[DocstringRow]) -> None:
    """Insert docstrings with standardized ordering."""
    gateway.con.executemany(
        """
        INSERT INTO core.docstrings (
            repo, commit, rel_path, module, qualname, kind, lineno, end_lineno,
            raw_docstring, style, short_desc, long_desc, params, returns, raises,
            examples, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_function_metrics(gateway: StorageGateway, rows: Iterable[FunctionMetricsRow]) -> None:
    """Insert function metrics via gateway helper."""
    gateway.analytics.insert_function_metrics(row.to_tuple() for row in rows)


def insert_function_types(gateway: StorageGateway, rows: Iterable[FunctionTypesRow]) -> None:
    """Insert function types via connection to avoid drift in gateway helpers."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.function_types (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, total_params, annotated_params, unannotated_params,
            param_typed_ratio, has_return_annotation, return_type, return_type_source,
            type_comment, param_types, fully_typed, partial_typed, untyped,
            typedness_bucket, typedness_source, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_coverage_functions(gateway: StorageGateway, rows: Iterable[CoverageFunctionRow]) -> None:
    """Insert coverage functions via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.coverage_functions (
            function_goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, executable_lines, covered_lines, coverage_ratio,
            tested, untested_reason, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_risk_factors(gateway: StorageGateway, rows: Iterable[RiskFactorRow]) -> None:
    """Insert GOID risk factors via gateway helper."""
    gateway.analytics.insert_goid_risk_factors(row.to_tuple() for row in rows)


def insert_test_catalog(gateway: StorageGateway, rows: Iterable[TestCatalogRow]) -> None:
    """Insert test_catalog rows via gateway helper."""
    gateway.analytics.insert_test_catalog(row.to_tuple() for row in rows)


def insert_test_coverage_edges(
    gateway: StorageGateway, rows: Iterable[TestCoverageEdgeRow]
) -> None:
    """Insert test_coverage_edges rows via gateway helper."""
    gateway.analytics.insert_test_coverage_edges(row.to_tuple() for row in rows)


def insert_typedness(gateway: StorageGateway, rows: Iterable[TypednessRow]) -> None:
    """Insert typedness rows via gateway helper."""
    gateway.analytics.insert_typedness(row.to_tuple() for row in rows)


def insert_static_diagnostics(
    gateway: StorageGateway, rows: Iterable[StaticDiagnosticsRow]
) -> None:
    """Insert static_diagnostics rows via gateway helper."""
    gateway.analytics.insert_static_diagnostics(row.to_tuple() for row in rows)


def insert_hotspots(gateway: StorageGateway, rows: Iterable[HotspotRow]) -> None:
    """Insert hotspots rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.hotspots (
            rel_path, commit_count, author_count, lines_added, lines_deleted, complexity, score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_ast_metrics(gateway: StorageGateway, rows: Iterable[AstMetricsRow]) -> None:
    """Insert ast_metrics rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO core.ast_metrics (
            rel_path, node_count, function_count, class_count, avg_depth, max_depth,
            complexity, generated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_function_validation(
    gateway: StorageGateway, rows: Iterable[FunctionValidationRow]
) -> None:
    """Insert function_validation rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.function_validation (
            repo, commit, rel_path, qualname, issue, detail, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_config_values(gateway: StorageGateway, rows: Iterable[ConfigValueRow]) -> None:
    """Insert config_values rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.config_values (
            repo, commit, config_path, format, key, reference_paths, reference_modules, reference_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_graph_metrics_modules_ext(
    gateway: StorageGateway, rows: Iterable[GraphMetricsModulesExtRow]
) -> None:
    """Insert graph_metrics_modules_ext rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.graph_metrics_modules_ext (
            repo, commit, module, import_betweenness, import_closeness, import_eigenvector,
            import_harmonic, import_k_core, import_constraint, import_effective_size,
            import_community_id, import_component_id, import_component_size,
            import_scc_id, import_scc_size, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_symbol_graph_metrics_modules(
    gateway: StorageGateway, rows: Iterable[SymbolGraphMetricsModulesRow]
) -> None:
    """Insert symbol_graph_metrics_modules rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.symbol_graph_metrics_modules (
            repo, commit, module, symbol_betweenness, symbol_closeness,
            symbol_eigenvector, symbol_harmonic, symbol_k_core, symbol_constraint,
            symbol_effective_size, symbol_community_id, symbol_component_id,
            symbol_component_size, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_subsystem_modules(gateway: StorageGateway, rows: Iterable[SubsystemModuleRow]) -> None:
    """Insert subsystem_modules rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.subsystem_modules (repo, commit, subsystem_id, module, role)
        VALUES (?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_subsystems(gateway: StorageGateway, rows: Iterable[SubsystemRow]) -> None:
    """Insert subsystems rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO analytics.subsystems (
            repo, commit, subsystem_id, name, description, module_count, modules_json,
            entrypoints_json, internal_edge_count, external_edge_count, fan_in, fan_out,
            function_count, avg_risk_score, max_risk_score, high_risk_function_count,
            risk_level, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )


def insert_goid_crosswalk(gateway: StorageGateway, rows: Iterable[GoidCrosswalkRow]) -> None:
    """Insert goid_crosswalk rows via connection."""
    gateway.con.executemany(
        """
        INSERT INTO core.goid_crosswalk (
            repo, commit, goid, lang, module_path, file_path, start_line, end_line,
            scip_symbol, ast_qualname, cst_node_id, chunk_id, symbol_id, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [row.to_tuple() for row in rows],
    )
