"""Row dataclasses for analytics.* schema tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from tests._helpers.builders._common import _iso

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "ConfigValueRow",
    "CoverageFunctionRow",
    "CoverageLineRow",
    "FunctionMetricsRow",
    "FunctionTypesRow",
    "FunctionValidationRow",
    "GraphMetricsModulesExtRow",
    "HotspotRow",
    "RiskFactorRow",
    "StaticDiagnosticsRow",
    "SubsystemModuleRow",
    "SubsystemRow",
    "SymbolGraphMetricsModulesRow",
    "TestCatalogRow",
    "TestCoverageEdgeRow",
    "TypednessRow",
]


@dataclass(frozen=True)
class FunctionMetricsRow:
    """Row for analytics.function_metrics."""

    __table__: ClassVar[str] = "analytics.function_metrics"
    __columns__: ClassVar[tuple[str, ...]] = (
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "loc",
        "logical_loc",
        "param_count",
        "positional_params",
        "keyword_only_params",
        "has_varargs",
        "has_varkw",
        "is_async",
        "is_generator",
        "return_count",
        "yield_count",
        "raise_count",
        "cyclomatic_complexity",
        "max_nesting_depth",
        "stmt_count",
        "decorator_count",
        "has_docstring",
        "complexity_bucket",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.function_types"
    __columns__: ClassVar[tuple[str, ...]] = (
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "total_params",
        "annotated_params",
        "unannotated_params",
        "param_typed_ratio",
        "has_return_annotation",
        "return_type",
        "return_type_source",
        "type_comment",
        "param_types",
        "fully_typed",
        "partial_typed",
        "untyped",
        "typedness_bucket",
        "typedness_source",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.coverage_functions"
    __columns__: ClassVar[tuple[str, ...]] = (
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "executable_lines",
        "covered_lines",
        "coverage_ratio",
        "tested",
        "untested_reason",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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
class CoverageLineRow:
    """Row for analytics.coverage_lines."""

    __table__: ClassVar[str] = "analytics.coverage_lines"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "rel_path",
        "line",
        "is_executable",
        "is_covered",
        "hits",
        "context_count",
        "created_at",
    )

    repo: str
    commit: str
    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool
    hits: int
    context_count: int
    created_at: datetime | None = None

    def to_tuple(
        self,
    ) -> tuple[str, str, str, int, bool, bool, int, int, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.repo,
            self.commit,
            self.rel_path,
            self.line,
            self.is_executable,
            self.is_covered,
            self.hits,
            self.context_count,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class RiskFactorRow:
    """Row for analytics.goid_risk_factors."""

    __table__: ClassVar[str] = "analytics.goid_risk_factors"
    __columns__: ClassVar[tuple[str, ...]] = (
        "function_goid_h128",
        "repo",
        "commit",
        "risk_score",
        "risk_level",
        "cyclomatic_complexity",
        "fan_in_count",
        "fan_out_count",
        "has_tests",
    )

    function_goid_h128: int
    repo: str
    commit: str
    risk_score: int = 0
    risk_level: str = "low"
    cyclomatic_complexity: int = 0
    fan_in_count: int = 0
    fan_out_count: int = 0
    has_tests: bool = False

    def to_tuple(
        self,
    ) -> tuple[
        int,
        str,
        str,
        int,
        str,
        int,
        int,
        int,
        bool,
    ]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.function_goid_h128,
            self.repo,
            self.commit,
            self.risk_score,
            self.risk_level,
            self.cyclomatic_complexity,
            self.fan_in_count,
            self.fan_out_count,
            self.has_tests,
        )


@dataclass(frozen=True)
class TestCatalogRow:
    """Row for analytics.test_catalog."""

    __test__ = False
    __table__: ClassVar[str] = "analytics.test_catalog"
    __columns__: ClassVar[tuple[str, ...]] = (
        "test_id",
        "test_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "qualname",
        "kind",
        "status",
        "duration_ms",
        "markers",
        "parametrized",
        "flaky",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __test__ = False
    __table__: ClassVar[str] = "analytics.test_coverage_edges"
    __columns__: ClassVar[tuple[str, ...]] = (
        "test_id",
        "test_goid_h128",
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "qualname",
        "covered_lines",
        "executable_lines",
        "coverage_ratio",
        "last_status",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.typedness"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "path",
        "type_error_count",
        "annotation_ratio",
        "untyped_defs",
        "overlay_needed",
    )

    repo: str
    commit: str
    path: str
    type_error_count: int
    annotation_ratio: str
    untyped_defs: int
    overlay_needed: bool

    def to_tuple(self) -> tuple[str, str, str, int, str, int, bool]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.static_diagnostics"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "rel_path",
        "pyrefly_errors",
        "pyright_errors",
        "ruff_errors",
        "total_errors",
        "has_errors",
    )

    repo: str
    commit: str
    rel_path: str
    pyrefly_errors: int
    pyright_errors: int
    ruff_errors: int
    total_errors: int
    has_errors: bool

    def to_tuple(self) -> tuple[str, str, str, int, int, int, int, bool]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.hotspots"
    __columns__: ClassVar[tuple[str, ...]] = (
        "rel_path",
        "commit_count",
        "author_count",
        "lines_added",
        "lines_deleted",
        "complexity",
        "score",
    )

    rel_path: str
    commit_count: int
    author_count: int
    lines_added: int
    lines_deleted: int
    complexity: float
    score: float

    def to_tuple(self) -> tuple[str, int, int, int, int, float, float]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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
class FunctionValidationRow:
    """Row for analytics.function_validation."""

    __table__: ClassVar[str] = "analytics.function_validation"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "function_goid_h128",
        "rel_path",
        "qualname",
        "issue",
        "detail",
        "created_at",
    )

    repo: str
    commit: str
    function_goid_h128: int
    rel_path: str
    qualname: str
    issue: str
    detail: str
    created_at: datetime

    def to_tuple(self) -> tuple[str, str, int, str, str, str, str, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.repo,
            self.commit,
            self.function_goid_h128,
            self.rel_path,
            self.qualname,
            self.issue,
            self.detail,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class ConfigValueRow:
    """Row for analytics.config_values."""

    __table__: ClassVar[str] = "analytics.config_values"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "config_path",
        "format",
        "key",
        "reference_paths",
        "reference_modules",
        "reference_count",
    )

    repo: str
    commit: str
    config_path: str
    format: str
    key: str
    reference_paths: list[str]
    reference_modules: list[str]
    reference_count: int

    def to_tuple(self) -> tuple[str, str, str, str, str, str, str, int]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.graph_metrics_modules_ext"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "module",
        "import_betweenness",
        "import_closeness",
        "import_eigenvector",
        "import_harmonic",
        "import_k_core",
        "import_constraint",
        "import_effective_size",
        "import_community_id",
        "import_component_id",
        "import_component_size",
        "import_scc_id",
        "import_scc_size",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.symbol_graph_metrics_modules"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "module",
        "symbol_betweenness",
        "symbol_closeness",
        "symbol_eigenvector",
        "symbol_harmonic",
        "symbol_k_core",
        "symbol_constraint",
        "symbol_effective_size",
        "symbol_community_id",
        "symbol_component_id",
        "symbol_component_size",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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

    __table__: ClassVar[str] = "analytics.subsystem_modules"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "subsystem_id",
        "module",
        "role",
    )

    repo: str
    commit: str
    subsystem_id: str
    module: str
    role: str

    def to_tuple(self) -> tuple[str, str, str, str, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (self.repo, self.commit, self.subsystem_id, self.module, self.role)


@dataclass(frozen=True)
class SubsystemRow:
    """Row for analytics.subsystems."""

    __table__: ClassVar[str] = "analytics.subsystems"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "subsystem_id",
        "name",
        "description",
        "module_count",
        "modules_json",
        "entrypoints_json",
        "internal_edge_count",
        "external_edge_count",
        "fan_in",
        "fan_out",
        "function_count",
        "avg_risk_score",
        "max_risk_score",
        "high_risk_function_count",
        "risk_level",
        "created_at",
    )

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
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
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
