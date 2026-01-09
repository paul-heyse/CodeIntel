"""Row dataclasses for analytics.* schema tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from tests._helpers.builders._common import _iso

if TYPE_CHECKING:
    from datetime import datetime

    from codeintel.core.serialization.converters import JsonValue

__all__ = [
    "ConfigValueRow",
    "FunctionTypesRow",
    "FunctionValidationRow",
    "GraphMetricsModulesExtRow",
    "StaticDiagnosticsRow",
    "SubsystemModuleRow",
    "SubsystemRow",
    "SymbolGraphMetricsModulesRow",
    "TestCatalogRow",
]


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
        "return_type",
        "type_comment",
        "extras",
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
    return_type: str | None
    type_comment: str | None
    param_types: JsonValue | None
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
        str | None,
        str | None,
        dict[str, JsonValue | None],
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
            self.return_type,
            self.type_comment,
            {"param_types": self.param_types},
            _iso(self.created_at),
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
        "extras",
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
    markers: list[str] = field(default_factory=list)
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
        dict[str, list[str]],
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
            {"markers": list(self.markers)},
            self.parametrized,
            self.flaky,
            _iso(self.created_at),
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
        "extras",
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

    def to_tuple(self) -> tuple[str, str, str, str, str, dict[str, list[str]], int]:
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
            {
                "reference_paths": list(self.reference_paths),
                "reference_modules": list(self.reference_modules),
            },
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
        "extras",
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
    modules_json: list[str]
    entrypoints_json: list[str]
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
        dict[str, list[str]],
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
            {
                "modules": list(self.modules_json),
                "entrypoints": list(self.entrypoints_json),
            },
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
