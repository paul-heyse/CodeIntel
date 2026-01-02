"""Unified row factory utilities for tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypedDict, cast

from codeintel.core.catalog import FunctionSpan
from codeintel.core.columnar.rows import ColumnarRowBuffer, ColumnarRows
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsFileProfileRow as FileProfileRowModel,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsFunctionProfileRow as FunctionProfileRowModel,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsModuleProfileRow as ModuleProfileRowModel,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsTestProfileRow as ProfileRowModel,
)
from codeintel.core.schemas.primitives import Column, ColumnType, TableSchema
from codeintel.core.schemas.row_models import normalize_row_value_for_type
from tests._helpers.builders import (
    AstMetricsRow,
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    ConfigValueRow,
    DatasetDataflowEdgeRow,
    DatasetDataflowNodeRow,
    DFGEdgeRow,
    DocstringRow,
    FunctionContextBuilder,
    FunctionMetricsRow,
    FunctionTypesRow,
    FunctionValidationRow,
    GoidCrosswalkRow,
    GoidRow,
    GraphMetricsModulesExtRow,
    HotspotRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    SubsystemModuleRow,
    SubsystemRow,
    SymbolEdgeOptions,
    SymbolGraphMetricsModulesRow,
    SymbolUseEdgeRow,
    TestCatalogRow,
    TypednessRow,
    insert_rows,
    insert_symbol_use_edges,
    make_symbol_use_edge_row,
)
from tests._helpers.builders.row_protocol import InsertableRow
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT
from tests._helpers.schemas import ensure_schema_service

if TYPE_CHECKING:
    from collections.abc import Sequence


@lru_cache(maxsize=256)
def _table_schema_for_key(table_key: str) -> TableSchema:
    schema = ensure_schema_service().get_table_schema(table_key)
    if schema is None:
        msg = f"Unknown table schema: {table_key}"
        raise KeyError(msg)
    return schema


@dataclass(frozen=True)
class RowCoercions:
    """Common coercion helpers for row fields."""

    @staticmethod
    def to_int(value: object, *, default: int = 0) -> int:
        """Coerce a value to int with a default fallback.

        Returns
        -------
        int
            Coerced integer value.
        """
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def to_bool(value: object, *, default: bool = False) -> bool:
        """Coerce a value to bool with a default fallback.

        Returns
        -------
        bool
            Coerced boolean value.
        """
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def to_timestamp(value: object | None = None) -> datetime:
        """Coerce a value to a UTC timestamp.

        Returns
        -------
        datetime
            Timestamp value.
        """
        if isinstance(value, datetime):
            return value
        return datetime.now(tz=UTC)


_JSON_LIST_COLUMNS: set[str] = {
    "entrypoints_json",
    "examples",
    "examples_json",
    "includes",
    "markers",
    "matches",
    "modules_json",
    "owners",
    "params",
    "params_json",
    "raises",
    "raises_json",
    "reference_modules",
    "reference_paths",
    "returns",
    "returns_json",
    "role_sources_json",
    "stmts_json",
    "tags",
    "usage_modes",
}
_JSON_DICT_COLUMNS: set[str] = {
    "annotation_ratio",
    "evidence_json",
    "metadata",
    "modules",
    "overlays",
    "span",
}


def _looks_like_json(value: str) -> bool:
    raw = value.strip()
    if not raw:
        return False
    if raw[0] not in {"{", "[", '"'}:
        return False
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False
    return True


def _guard_json_stringification(schema: TableSchema, row: Mapping[str, object]) -> None:
    json_columns = {column.name for column in schema.columns if column.type == "JSON"}
    for name in json_columns:
        value = row.get(name)
        if isinstance(value, str) and _looks_like_json(value):
            msg = f"JSON stringification detected for column {name}; pass dict/list instead."
            raise ValueError(msg)


def _json_default_for_column(column_name: str) -> object:
    if column_name in _JSON_DICT_COLUMNS:
        return {}
    if column_name in _JSON_LIST_COLUMNS or column_name.endswith("_json"):
        return []
    return {}


def _default_for_column(column: Column) -> object:
    column_type = column.type
    if column_type == "BOOLEAN":
        value: object = False
    elif column_type in {"INTEGER", "BIGINT"}:
        value = 0
    elif column_type == "DOUBLE":
        value = 0.0
    elif column_type in {"DECIMAL", "DECIMAL(38,0)"}:
        value = Decimal(0)
    elif column_type == "VARCHAR":
        value = ""
    elif column_type in {"TIMESTAMP", "TIMESTAMPTZ"}:
        value = datetime(1970, 1, 1, tzinfo=UTC)
    elif column_type == "JSON":
        value = _json_default_for_column(column.name)
    else:
        value = None
    return value


def _row_defaults(
    schema: TableSchema,
    *,
    fill_non_nullable: bool,
) -> dict[str, object]:
    row: dict[str, object] = {column.name: None for column in schema.columns}
    if not fill_non_nullable:
        return row
    for column in schema.columns:
        if not column.nullable:
            row[column.name] = _default_for_column(column)
    return row


def _normalize_row(schema: TableSchema, row: Mapping[str, object]) -> dict[str, object]:
    column_types: dict[str, ColumnType] = {column.name: column.type for column in schema.columns}
    return {
        name: normalize_row_value_for_type(value, column_types.get(name))
        for name, value in row.items()
    }


class RowFactory:
    """Create schema-aligned row dictionaries for tests."""

    @staticmethod
    def blank_row(table_key: str, *, fill_non_nullable: bool = True) -> Mapping[str, object]:
        """Create a blank row mapping for a table.

        Returns
        -------
        Mapping[str, object]
            Mapping populated with table columns set to None.
        """
        schema = _table_schema_for_key(table_key)
        return cast(
            "Mapping[str, object]",
            _row_defaults(schema, fill_non_nullable=fill_non_nullable),
        )

    @staticmethod
    def row_for(
        table_key: str,
        *,
        fill_non_nullable: bool = True,
        **fields: object,
    ) -> Mapping[str, object]:
        """Create a row mapping with provided field overrides.

        Returns
        -------
        Mapping[str, object]
            Row mapping with overrides applied.
        """
        schema = _table_schema_for_key(table_key)
        row = _row_defaults(schema, fill_non_nullable=fill_non_nullable)
        row.update(fields)
        _guard_json_stringification(schema, row)
        return _normalize_row(schema, row)

    @staticmethod
    def rows_for(
        table_key: str,
        count: int,
        *,
        fill_non_nullable: bool = True,
        **overrides: object,
    ) -> list[Mapping[str, object]]:
        """Create multiple row mappings for a table.

        Returns
        -------
        list[Mapping[str, object]]
            List of row mappings with overrides applied.
        """
        return [
            RowFactory.row_for(
                table_key,
                fill_non_nullable=fill_non_nullable,
                **overrides,
            )
            for _ in range(count)
        ]

    @staticmethod
    def columnar_rows(
        table_key: str,
        rows: Sequence[Mapping[str, object]],
        *,
        fill_non_nullable: bool = True,
    ) -> ColumnarRows:
        """Create columnar row payloads for a table.

        Returns
        -------
        ColumnarRows
            Column-oriented mapping aligned with the table schema.
        """
        schema = _table_schema_for_key(table_key)
        buffer = ColumnarRowBuffer(
            table_key=table_key,
            columns=tuple(schema.column_names()),
            column_types=tuple(column.type for column in schema.columns),
            data={column.name: [] for column in schema.columns},
        )
        for row in rows:
            resolved = _row_defaults(schema, fill_non_nullable=fill_non_nullable)
            resolved.update(row)
            _guard_json_stringification(schema, resolved)
            buffer.append(resolved)
        return buffer.data


def list_public_exports(module: object) -> tuple[str, ...]:
    """Return a stable tuple of public export names for a module.

    Returns
    -------
    tuple[str, ...]
        Publicly exported attribute names.
    """
    exports_attr: Any = getattr(module, "__all__", None)
    if isinstance(exports_attr, (list, tuple)):
        return tuple(exports_attr)
    return tuple(sorted(name for name in dir(module) if not name.startswith("_")))


def blank_row(table_key: str, *, fill_non_nullable: bool = True) -> Mapping[str, object]:
    """Return a blank row mapping for a table key.

    Returns
    -------
    Mapping[str, object]
        Mapping populated with table columns set to None.
    """
    return RowFactory.blank_row(table_key, fill_non_nullable=fill_non_nullable)


def row_for(
    table_key: str,
    *,
    fill_non_nullable: bool = True,
    **fields: object,
) -> Mapping[str, object]:
    """Return a row mapping with provided field overrides.

    Returns
    -------
    Mapping[str, object]
        Row mapping with overrides applied.
    """
    return RowFactory.row_for(table_key, fill_non_nullable=fill_non_nullable, **fields)


def row_list_for(
    table_key: str,
    count: int,
    *,
    fill_non_nullable: bool = True,
    **overrides: object,
) -> list[Mapping[str, object]]:
    """Return multiple row mappings with overrides applied.

    Returns
    -------
    list[Mapping[str, object]]
        List of row mappings.
    """
    return RowFactory.rows_for(
        table_key,
        count,
        fill_non_nullable=fill_non_nullable,
        **overrides,
    )


def columnar_rows_for(
    table_key: str,
    rows: Sequence[Mapping[str, object]],
    *,
    fill_non_nullable: bool = True,
) -> ColumnarRows:
    """Return columnar rows for a table key.

    Returns
    -------
    ColumnarRows
        Column-oriented mapping aligned with schema.
    """
    return RowFactory.columnar_rows(
        table_key,
        rows,
        fill_non_nullable=fill_non_nullable,
    )


def _row_defaults_for(row_type: type[InsertableRow]) -> Mapping[str, object]:
    defaults = getattr(row_type, "__row_defaults__", None)
    if isinstance(defaults, Mapping):
        return defaults
    return {}


def dataclass_row[RowType: InsertableRow](
    row_type: type[RowType],
    *,
    fill_non_nullable: bool = True,
    **fields: object,
) -> RowType:
    """Create an InsertableRow dataclass from RowFactory defaults.

    Returns
    -------
    RowType
        Dataclass row instantiated with defaulted table values.
    """
    row_defaults = _row_defaults_for(row_type)
    row = RowFactory.row_for(
        row_type.__table__,
        fill_non_nullable=fill_non_nullable,
        **{**row_defaults, **fields},
    )
    if not is_dataclass(row_type):
        return row_type(**row)
    field_defs = [field_def for field_def in dataclass_fields(row_type) if field_def.init]
    row_values: dict[str, object] = {}
    for field_def in field_defs:
        name = field_def.name
        if name in fields:
            row_values[name] = row.get(name)
            continue
        value = row.get(name)
        if value is not None:
            row_values[name] = value
            continue
        if field_def.default is not MISSING or field_def.default_factory is not MISSING:
            continue
        row_values[name] = value
    return row_type(**row_values)


def blank_file_profile_row() -> FileProfileRowModel:
    """Return a blank analytics.file_profile row.

    Returns
    -------
    FileProfileRowModel
        Blank row payload.
    """
    return cast("FileProfileRowModel", RowFactory.blank_row("analytics.file_profile"))


def blank_module_profile_row() -> ModuleProfileRowModel:
    """Return a blank analytics.module_profile row.

    Returns
    -------
    ModuleProfileRowModel
        Blank row payload.
    """
    return cast("ModuleProfileRowModel", RowFactory.blank_row("analytics.module_profile"))


def blank_test_profile_row() -> ProfileRowModel:
    """Return a blank analytics.test_profile row.

    Returns
    -------
    ProfileRowModel
        Blank row payload.
    """
    return cast("ProfileRowModel", RowFactory.blank_row("analytics.test_profile"))


def blank_function_profile_row() -> FunctionProfileRowModel:
    """Return a blank analytics.function_profile row.

    Returns
    -------
    FunctionProfileRowModel
        Blank row payload.
    """
    return cast("FunctionProfileRowModel", RowFactory.blank_row("analytics.function_profile"))


def sample_function_profile_rows(repo: str, commit: str) -> list[FunctionProfileRowModel]:
    """Return sample analytics.function_profile rows.

    Returns
    -------
    list[FunctionProfileRowModel]
        Sample row payloads.
    """
    return [
        cast(
            "FunctionProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "function_goid_h128": 101,
                "urn": "urn:fn:alpha::helper",
                "rel_path": "pkg/alpha.py",
                "language": "python",
                "kind": "function",
                "qualname": "pkg.alpha.helper",
                "tags": ["io", "auth"],
                "owners": ["team-data"],
                "created_at": None,
            },
        ),
        cast(
            "FunctionProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "function_goid_h128": 202,
                "urn": "urn:fn:beta::process",
                "rel_path": "pkg/beta.py",
                "language": "python",
                "kind": "method",
                "qualname": "pkg.beta.B.process",
                "tags": [],
                "owners": None,
                "created_at": None,
            },
        ),
        cast(
            "FunctionProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "function_goid_h128": 303,
                "urn": "urn:fn:unicode::delta",
                "rel_path": "pkg/unicode/delta.py",
                "language": "python",
                "kind": "function",
                "qualname": "pkg.unicode.delta.fn",
                "tags": ["unicode", "core"],
                "owners": ["team-delta"],
                "created_at": None,
            },
        ),
    ]


def sample_file_profile_rows(repo: str, commit: str) -> list[FileProfileRowModel]:
    """Return sample analytics.file_profile rows.

    Returns
    -------
    list[FileProfileRowModel]
        Sample row payloads.
    """
    return [
        cast(
            "FileProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "rel_path": "pkg/alpha.py",
                "module": "pkg.alpha_mod",
                "tags": ["core", "io"],
                "owners": ["team-analytics"],
                "created_at": None,
            },
        ),
        cast(
            "FileProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "rel_path": "pkg/beta.py",
                "module": "pkg.beta",
                "tags": [],
                "owners": None,
                "created_at": None,
            },
        ),
        cast(
            "FileProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "rel_path": "pkg/unicode/delta.py",
                "module": "pkg.unicode.delta",
                "tags": ["unicode"],
                "owners": None,
                "created_at": None,
            },
        ),
    ]


def sample_module_profile_rows(repo: str, commit: str) -> list[ModuleProfileRowModel]:
    """Return sample analytics.module_profile rows.

    Returns
    -------
    list[ModuleProfileRowModel]
        Sample row payloads.
    """
    return [
        cast(
            "ModuleProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "module": "pkg.alpha_mod",
                "rel_path": "pkg/alpha.py",
                "language": "python",
                "loc": 120,
                "created_at": None,
            },
        ),
        cast(
            "ModuleProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "module": "pkg.beta",
                "rel_path": "pkg/beta.py",
                "language": "python",
                "loc": 75,
                "created_at": None,
            },
        ),
    ]


def sample_test_profile_rows(repo: str, commit: str) -> list[ProfileRowModel]:
    """Return sample analytics.test_profile rows.

    Returns
    -------
    list[ProfileRowModel]
        Sample row payloads.
    """
    return [
        cast(
            "ProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/test_mod_a.py::test_func_a",
                "rel_path": "tests/test_mod_a.py",
                "qualname": "test_func_a",
                "status": "passed",
                "kind": "unit",
                "duration_ms": 150,
                "markers": [],
                "uses_parametrize": False,
                "flaky": False,
                "created_at": None,
            },
        ),
        cast(
            "ProfileRowModel",
            {
                "repo": repo,
                "commit": commit,
                "test_id": "tests/test_mod_b.py::test_func_b",
                "rel_path": "tests/test_mod_b.py",
                "qualname": "test_func_b",
                "status": "passed",
                "kind": "unit",
                "duration_ms": 200,
                "markers": ["slow"],
                "uses_parametrize": False,
                "flaky": False,
                "created_at": None,
            },
        ),
    ]


def function_meta(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: tuple[str, str] = (DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit),
    line_span: tuple[int, int] = (1, 1),
) -> FunctionSpan:
    """Build a FunctionSpan with consistent URN formatting.

    Returns
    -------
    FunctionMeta
        Catalog entry with normalized URN and line span.
    """
    repo, commit = snapshot
    start_line, end_line = line_span
    urn = f"urn:{repo}:{commit}:{rel_path}#{qualname}"
    return FunctionSpan(
        goid=goid,
        urn=urn,
        rel_path=rel_path,
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
    )


def module_row(
    *,
    module: str,
    path: str,
    snapshot: tuple[str, str] = (DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit),
) -> ModuleRow:
    """Create a ModuleRow for analytics modules.

    Returns
    -------
    ModuleRow
        Row ready for insertion into analytics.modules.
    """
    repo, commit = snapshot
    return ModuleRow(module=module, path=path, repo=repo, commit=commit)


def function_metrics_row(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: tuple[str, str] = (DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit),
    metrics: Mapping[str, int | str | bool | datetime | float | None] | None = None,
) -> FunctionMetricsRow:
    """Create a FunctionMetricsRow with sensible defaults and override support.

    Returns
    -------
    FunctionMetricsRow
        Row with defaulted metrics for analytics.function_metrics.
    """
    repo, commit = snapshot
    urn = f"urn:{repo}:{commit}:{rel_path}#{qualname}"
    defaults: dict[str, object] = {
        "language": "python",
        "kind": "function",
        "start_line": 1,
        "end_line": 1,
        "loc": 2,
        "logical_loc": 2,
        "param_count": 0,
        "positional_params": 0,
        "keyword_only_params": 0,
        "has_varargs": False,
        "has_varkw": False,
        "is_async": False,
        "is_generator": False,
        "return_count": 0,
        "yield_count": 0,
        "raise_count": 0,
        "cyclomatic_complexity": 1,
        "max_nesting_depth": 1,
        "stmt_count": 2,
        "decorator_count": 0,
        "has_docstring": False,
        "complexity_bucket": "low",
        "created_at": datetime.now(tz=UTC),
    }
    if metrics:
        defaults.update(metrics)

    def _as_int(key: str) -> int:
        return RowCoercions.to_int(defaults.get(key, 0))

    def _as_bool(key: str) -> bool:
        return RowCoercions.to_bool(defaults.get(key, False))

    def _as_str(key: str) -> str:
        value = defaults.get(key)
        if value is None:
            return str(defaults[key])
        return value if isinstance(value, str) else str(value)

    created_at_value = defaults.get("created_at")
    created_at = (
        created_at_value if isinstance(created_at_value, datetime) else datetime.now(tz=UTC)
    )
    return FunctionMetricsRow(
        function_goid_h128=goid,
        urn=urn,
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        language=_as_str("language"),
        kind=_as_str("kind"),
        qualname=qualname,
        start_line=_as_int("start_line"),
        end_line=_as_int("end_line"),
        loc=_as_int("loc"),
        logical_loc=_as_int("logical_loc"),
        param_count=_as_int("param_count"),
        positional_params=_as_int("positional_params"),
        keyword_only_params=_as_int("keyword_only_params"),
        has_varargs=_as_bool("has_varargs"),
        has_varkw=_as_bool("has_varkw"),
        is_async=_as_bool("is_async"),
        is_generator=_as_bool("is_generator"),
        return_count=_as_int("return_count"),
        yield_count=_as_int("yield_count"),
        raise_count=_as_int("raise_count"),
        cyclomatic_complexity=_as_int("cyclomatic_complexity"),
        max_nesting_depth=_as_int("max_nesting_depth"),
        stmt_count=_as_int("stmt_count"),
        decorator_count=_as_int("decorator_count"),
        has_docstring=_as_bool("has_docstring"),
        complexity_bucket=_as_str("complexity_bucket"),
        created_at=created_at,
    )


@dataclass
class DependencyCallSeed:
    library: str
    service_name: str
    qualname: str
    callsite_count: int
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def dependency_call_row(seed: DependencyCallSeed) -> tuple[object, ...]:
    """Row for analytics.external_dependency_calls.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.library,
        seed.service_name,
        seed.qualname,
        seed.callsite_count,
    )


def compute_dep_id(repo: str, commit: str, library: str) -> str:
    """Compute unique dependency identifier used in analytics dependency tables.

    Returns
    -------
    str
        SHA-1 hash prefix as dependency ID.
    """
    raw = f"{repo}:{commit}:{library}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class DependencyCallRow:
    """Row model for analytics.external_dependency_calls."""

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    function_goid_h128: Decimal
    function_urn: str
    rel_path: str
    module: str
    qualname: str
    callsite_count: int
    modes: list[str]
    evidence_json: list[dict[str, object]]
    created_at: datetime


@dataclass(frozen=True)
class DependencyAggregateRow:
    """Row model for analytics.external_dependencies."""

    repo: str
    commit: str
    dep_id: str
    library: str
    service_name: str
    category: str | None
    language: str
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    modules_json: list[str]
    usage_modes: list[str]
    config_keys: list[str]
    risk_level: str
    created_at: datetime


@dataclass
class DependencyCallPayloadSeed:
    library: str
    service_name: str
    qualname: str
    rel_path: str
    module: str
    callsite_count: int
    function_goid: int
    modes: Sequence[str] = ()
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    evidence_json: list[dict[str, object]] | None = None


class DependencyCallPayload(TypedDict):
    """Typed payload for analytics.external_dependency_calls."""

    repo: str
    commit: str
    library: str
    service_name: str
    qualname: str
    rel_path: str
    module: str
    callsite_count: int
    modes: list[str]
    function_goid_h128: Decimal
    function_urn: str
    evidence_json: list[dict[str, object]]
    created_at: datetime


def dependency_call_payload(seed: DependencyCallPayloadSeed) -> DependencyCallPayload:
    """Payload dict for analytics.external_dependency_calls.

    Returns
    -------
    DependencyCallPayload
        Dict matching adapter fields for external_dependency_calls.
    """
    created_at = datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "library": seed.library,
        "service_name": seed.service_name,
        "qualname": seed.qualname,
        "rel_path": seed.rel_path,
        "module": seed.module,
        "callsite_count": seed.callsite_count,
        "modes": list(seed.modes),
        "function_goid_h128": Decimal(seed.function_goid),
        "function_urn": f"urn:{seed.qualname}",
        "evidence_json": seed.evidence_json or [{"type": "call", "line": 1}],
        "created_at": created_at,
    }


@dataclass
class DependencyAggregateSeed:
    library: str
    service_name: str
    category: str | None
    severity: str | None
    criticality: str | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def dependency_aggregate_row(seed: DependencyAggregateSeed) -> tuple[object, ...]:
    """Row for analytics.external_dependencies.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.library,
        seed.service_name,
        seed.category,
        seed.severity,
        seed.criticality,
        seed.risk_score,
        seed.function_count,
        seed.callsite_count,
    )


@dataclass
class DependencyAggregatePayloadSeed:
    library: str
    service_name: str
    category: str | None
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    risk_level: str
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    modules_json: list[str] | None = None
    usage_modes: list[str] | None = None
    config_keys: list[str] | None = None


class DependencyAggregatePayload(TypedDict):
    """Typed payload for analytics.external_dependencies."""

    repo: str
    commit: str
    library: str
    service_name: str
    category: str | None
    language: str
    severity: str | None
    criticality: float | None
    risk_score: float | None
    function_count: int
    callsite_count: int
    modules_json: list[str]
    usage_modes: list[str]
    config_keys: list[str]
    risk_level: str
    created_at: datetime


def dependency_aggregate_payload(
    seed: DependencyAggregatePayloadSeed,
) -> DependencyAggregatePayload:
    """Payload dict for analytics.external_dependencies.

    Returns
    -------
    DependencyAggregatePayload
        Dict matching adapter fields for external_dependencies.
    """
    created_at = datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "library": seed.library,
        "service_name": seed.service_name,
        "category": seed.category,
        "language": "python",
        "severity": seed.severity,
        "criticality": seed.criticality,
        "risk_score": seed.risk_score,
        "function_count": seed.function_count,
        "callsite_count": seed.callsite_count,
        "modules_json": seed.modules_json or ["module.default"],
        "usage_modes": seed.usage_modes or ["read"],
        "config_keys": seed.config_keys or [],
        "risk_level": seed.risk_level,
        "created_at": created_at,
    }


def dependency_call_row_from_payload(seed: DependencyCallPayloadSeed) -> DependencyCallRow:
    """Build DependencyCallRow from the canonical payload seed.

    Returns
    -------
    DependencyCallRow
        Row ready for persistence.
    """
    payload = dependency_call_payload(seed)
    repo = str(payload["repo"])
    commit = str(payload["commit"])
    library = str(payload["library"])
    dep_id = compute_dep_id(repo, commit, library)
    return DependencyCallRow(
        repo=repo,
        commit=commit,
        dep_id=dep_id,
        library=library,
        service_name=str(payload["service_name"]),
        function_goid_h128=payload["function_goid_h128"],
        function_urn=str(payload["function_urn"]),
        rel_path=str(payload["rel_path"]),
        module=str(payload["module"]),
        qualname=str(payload["qualname"]),
        callsite_count=payload["callsite_count"],
        modes=payload["modes"],
        evidence_json=payload["evidence_json"],
        created_at=payload["created_at"],
    )


def dependency_aggregate_row_from_payload(
    seed: DependencyAggregatePayloadSeed,
) -> DependencyAggregateRow:
    """Build DependencyAggregateRow from the canonical payload seed.

    Returns
    -------
    DependencyAggregateRow
        Row ready for persistence.
    """
    payload = dependency_aggregate_payload(seed)
    repo = str(payload["repo"])
    commit = str(payload["commit"])
    library = str(payload["library"])
    dep_id = compute_dep_id(repo, commit, library)
    return DependencyAggregateRow(
        repo=repo,
        commit=commit,
        dep_id=dep_id,
        library=library,
        service_name=str(payload["service_name"]),
        category=payload["category"],
        language=str(payload["language"]),
        severity=payload["severity"],
        criticality=payload["criticality"],
        risk_score=payload["risk_score"],
        function_count=payload["function_count"],
        callsite_count=payload["callsite_count"],
        modules_json=payload["modules_json"],
        usage_modes=payload["usage_modes"],
        config_keys=payload["config_keys"],
        risk_level=str(payload["risk_level"]),
        created_at=payload["created_at"],
    )


@dataclass
class EntrypointSeed:
    entrypoint_id: str
    handler_qualname: str
    kind: str = "api_function"
    command_name: str | None = None
    http_method: str | None = "GET"
    path: str | None = "/"
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def entrypoint_row(seed: EntrypointSeed) -> tuple[object, ...]:
    """Row for analytics.entrypoints.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.entrypoint_id,
        seed.handler_qualname,
        seed.kind,
        seed.command_name,
        seed.http_method,
        seed.path,
    )


@dataclass
class EntrypointPayloadSeed:
    entrypoint_id: str
    handler_qualname: str
    kind: str = "http_endpoint"
    framework: str = "fastapi"
    handler_goid_h128: Decimal | int = Decimal(0)
    handler_rel_path: str = "src/api.py"
    handler_module: str = "api"
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    http_method: str | None = "GET"
    route_path: str | None = "/"
    status_codes: list[int] | None = None
    auth_required: bool = True
    command_name: str | None = None
    arguments_schema: dict[str, object] | None = None
    schedule: str | None = None
    trigger: str | None = None
    extra: dict[str, object] | None = None
    subsystem_id: str | None = None
    subsystem_name: str | None = None
    tags: list[str] | None = None
    owners: list[str] | None = None
    tests_touching: int = 0
    failing_tests: int = 0
    slow_tests: int = 0
    flaky_tests: int = 0
    entrypoint_coverage_ratio: float = 0.0
    last_test_status: str = "unknown"
    created_at: datetime | None = None


def entrypoint_payload(seed: EntrypointPayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.entrypoints adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for entrypoints.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "entrypoint_id": seed.entrypoint_id,
        "kind": seed.kind,
        "framework": seed.framework,
        "handler_goid_h128": Decimal(seed.handler_goid_h128),
        "handler_urn": f"urn:{seed.handler_module}:{seed.handler_qualname}",
        "handler_rel_path": seed.handler_rel_path,
        "handler_module": seed.handler_module,
        "handler_qualname": seed.handler_qualname,
        "http_method": seed.http_method,
        "route_path": seed.route_path,
        "status_codes": seed.status_codes or [200],
        "auth_required": seed.auth_required,
        "command_name": seed.command_name,
        "arguments_schema": seed.arguments_schema,
        "schedule": seed.schedule,
        "trigger": seed.trigger,
        "extra": seed.extra or {},
        "subsystem_id": seed.subsystem_id,
        "subsystem_name": seed.subsystem_name,
        "tags": seed.tags or [],
        "owners": seed.owners or [],
        "tests_touching": seed.tests_touching,
        "failing_tests": seed.failing_tests,
        "slow_tests": seed.slow_tests,
        "flaky_tests": seed.flaky_tests,
        "entrypoint_coverage_ratio": seed.entrypoint_coverage_ratio,
        "last_test_status": seed.last_test_status,
        "created_at": created_at,
    }


@dataclass
class EntrypointTestPayloadSeed:
    entrypoint_id: str
    test_id: str
    test_goid_h128: Decimal | int
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    coverage_ratio: float = 0.0
    status: str = "passed"
    duration_ms: float = 0.0
    created_at: datetime | None = None


def entrypoint_test_payload(seed: EntrypointTestPayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.entrypoint_tests adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for entrypoint_tests.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "entrypoint_id": seed.entrypoint_id,
        "test_id": seed.test_id,
        "test_goid_h128": Decimal(seed.test_goid_h128),
        "coverage_ratio": seed.coverage_ratio,
        "status": seed.status,
        "duration_ms": seed.duration_ms,
        "created_at": created_at,
    }


@dataclass
class EntrypointTestSeed:
    entrypoint_id: str
    test_qualname: str
    status: str
    coverage_ratio: float
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def entrypoint_test_row(seed: EntrypointTestSeed) -> tuple[object, ...]:
    """Row for analytics.entrypoint_tests.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.entrypoint_id,
        seed.test_qualname,
        seed.status,
        seed.coverage_ratio,
    )


@dataclass
class SemanticRoleFunctionSeed:
    goid: int
    role: str
    confidence: float
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def semantic_role_function_row(seed: SemanticRoleFunctionSeed) -> tuple[object, ...]:
    """Row for analytics.semantic_roles_functions.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.goid, seed.role, seed.confidence)


@dataclass
class SemanticRoleModuleSeed:
    module: str
    role: str
    confidence: float
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def semantic_role_module_row(seed: SemanticRoleModuleSeed) -> tuple[object, ...]:
    """Row for analytics.semantic_roles_modules.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.module, seed.role, seed.confidence)


@dataclass
class SubsystemPayloadSeed:
    subsystem_id: str
    name: str
    description: str
    module_count: int
    modules_json: Sequence[str]
    entrypoints_json: Sequence[str]
    internal_edge_count: int
    external_edge_count: int
    fan_in: int
    fan_out: int
    function_count: int
    risk_level: str
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    avg_risk_score: float | None = None
    max_risk_score: float | None = None
    high_risk_function_count: int = 0
    created_at: datetime | None = None


def subsystem_payload(seed: SubsystemPayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.subsystems adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for subsystems.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "subsystem_id": seed.subsystem_id,
        "name": seed.name,
        "description": seed.description,
        "module_count": seed.module_count,
        "modules_json": list(seed.modules_json),
        "entrypoints_json": list(seed.entrypoints_json),
        "internal_edge_count": seed.internal_edge_count,
        "external_edge_count": seed.external_edge_count,
        "fan_in": seed.fan_in,
        "fan_out": seed.fan_out,
        "function_count": seed.function_count,
        "avg_risk_score": seed.avg_risk_score,
        "max_risk_score": seed.max_risk_score,
        "high_risk_function_count": seed.high_risk_function_count,
        "risk_level": seed.risk_level,
        "created_at": created_at,
    }


@dataclass
class SubsystemModulePayloadSeed:
    subsystem_id: str
    module: str
    role: str | None = None
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def subsystem_module_payload(seed: SubsystemModulePayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.subsystem_modules adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for subsystem_modules.
    """
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "subsystem_id": seed.subsystem_id,
        "module": seed.module,
        "role": seed.role,
    }


@dataclass
class SubsystemSeed:
    subsystem_id: str
    name: str
    risk_level: str
    function_count: int
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    description: str | None = None
    module_count: int = 0
    entrypoints_json: Sequence[str] = field(default_factory=list)
    modules_json: Sequence[str] = field(default_factory=list)
    internal_edge_count: int = 0
    external_edge_count: int = 0
    fan_in: int = 0
    fan_out: int = 0
    avg_risk_score: float | None = None
    max_risk_score: float | None = None
    high_risk_function_count: int = 0
    created_at: datetime | None = None


def subsystem_row(
    seed: SubsystemSeed,
) -> tuple[
    str,
    str,
    str,
    str,
    str | None,
    int,
    list[str],
    list[str],
    int,
    int,
    int,
    int,
    int,
    float | None,
    float | None,
    int,
    str,
    datetime | None,
]:
    """Row for analytics.subsystems.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.subsystem_id,
        seed.name,
        seed.description,
        seed.module_count,
        list(seed.modules_json),
        list(seed.entrypoints_json),
        seed.internal_edge_count,
        seed.external_edge_count,
        seed.fan_in,
        seed.fan_out,
        seed.function_count,
        seed.avg_risk_score,
        seed.max_risk_score,
        seed.high_risk_function_count,
        seed.risk_level,
        seed.created_at or datetime.now(tz=UTC),
    )


@dataclass
class SubsystemModuleSeed:
    subsystem_id: str
    module: str
    role: str | None = None
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def subsystem_module_row(seed: SubsystemModuleSeed) -> tuple[str, str, str, str, str | None]:
    """Row for analytics.subsystem_modules.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.subsystem_id, seed.module, seed.role)


@dataclass
class DataModelUsagePayloadSeed:
    model_id: str
    goid: Decimal | int
    usage_kinds: Sequence[str]
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    evidence_json: list[dict[str, object]] | None = None
    context_json: dict[str, object] | None = None
    created_at: datetime | None = None


def data_model_usage_payload(seed: DataModelUsagePayloadSeed) -> dict[str, object]:
    """Payload dict for analytics.data_model_usage adapter tests.

    Returns
    -------
    dict[str, object]
        Dict matching adapter fields for data_model_usage.
    """
    created_at = seed.created_at or datetime.now(tz=UTC)
    return {
        "repo": seed.repo,
        "commit": seed.commit,
        "model_id": seed.model_id,
        "function_goid_h128": Decimal(seed.goid),
        "usage_kinds_json": list(seed.usage_kinds),
        "evidence_json": seed.evidence_json
        or [{"type": "attribute_access", "attr": "field", "line": 1}],
        "context_json": seed.context_json
        or {"file_path": "src/services/service.py", "function_name": "fn"},
        "created_at": created_at,
    }


@dataclass
class DataModelUsageSeed:
    model_id: str
    goid: int
    usage_kinds: Sequence[str]
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit


def data_model_usage_row(seed: DataModelUsageSeed) -> tuple[object, ...]:
    """Row for analytics.data_model_usage.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (seed.repo, seed.commit, seed.model_id, seed.goid, list(seed.usage_kinds))


@dataclass
class DataModelSeed:
    model_id: str
    model_name: str
    module: str
    rel_path: str
    model_kind: str
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    goid: int | None = None
    base_classes_json: list[dict[str, str]] | list[str] | None = None
    doc_short: str | None = None
    doc_long: str | None = None
    created_at: datetime | None = None


def data_model_row(seed: DataModelSeed) -> tuple[object, ...]:
    """Row for analytics.data_models.

    Returns
    -------
    tuple[object, ...]
        Row values ordered for analytics.data_models.
    """
    return (
        seed.repo,
        seed.commit,
        seed.model_id,
        seed.goid,
        seed.model_name,
        seed.module,
        seed.rel_path,
        seed.model_kind,
        seed.base_classes_json or [],
        seed.doc_short,
        seed.doc_long,
        seed.created_at or datetime.now(tz=UTC),
    )


@dataclass
class DataModelFieldSeed:
    model_id: str
    field_name: str
    field_type: str | None
    required: bool
    has_default: bool
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    default_expr: str | None = None
    constraints_json: dict[str, object] | None = None
    source: str = "tests"
    rel_path: str = "models.py"
    lineno: int | None = 1
    created_at: datetime | None = None


def data_model_field_row(seed: DataModelFieldSeed) -> tuple[object, ...]:
    """Row for analytics.data_model_fields.

    Returns
    -------
    tuple[object, ...]
        Row values ordered for analytics.data_model_fields.
    """
    return (
        seed.repo,
        seed.commit,
        seed.model_id,
        seed.field_name,
        seed.field_type,
        seed.required,
        seed.has_default,
        seed.default_expr,
        seed.constraints_json or {},
        seed.source,
        seed.rel_path,
        seed.lineno,
        seed.created_at or datetime.now(tz=UTC),
    )


@dataclass
class DataModelRelationshipSeed:
    source_model_id: str
    target_model_id: str
    field_name: str
    relationship_kind: str
    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    target_module: str | None = None
    target_model_name: str | None = None
    multiplicity: str | None = None
    via: str | None = None
    evidence_json: dict[str, object] | list[object] | None = None
    rel_path: str = "models.py"
    lineno: int | None = 1
    created_at: datetime | None = None


def data_model_relationship_row(seed: DataModelRelationshipSeed) -> tuple[object, ...]:
    """Row for analytics.data_model_relationships.

    Returns
    -------
    tuple[object, ...]
        Row values ordered for analytics.data_model_relationships.
    """
    return (
        seed.repo,
        seed.commit,
        seed.source_model_id,
        seed.target_model_id,
        seed.target_module,
        seed.target_model_name,
        seed.field_name,
        seed.relationship_kind,
        seed.multiplicity,
        seed.via,
        seed.evidence_json or {},
        seed.rel_path,
        seed.lineno,
        seed.created_at or datetime.now(tz=UTC),
    )


@dataclass
class AstMetricSeed:
    rel_path: str
    complexity: float
    node_count: int = 100
    function_count: int = 10
    class_count: int = 2
    avg_depth: float = 2.5
    max_depth: int = 5
    created_at: datetime | None = None


def ast_metric_row(seed: AstMetricSeed) -> tuple[str, int, int, int, float, int, float, datetime]:
    """Row for analytics.ast_metrics.

    Returns
    -------
    tuple[object, ...]
        Row values in repository schema order.
    """
    return (
        seed.rel_path,
        seed.node_count,
        seed.function_count,
        seed.class_count,
        seed.avg_depth,
        seed.max_depth,
        seed.complexity,
        seed.created_at or datetime.now(tz=UTC),
    )


@dataclass
class TestCatalogSeed:
    test_id: str
    test_goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    qualname: str
    kind: str
    status: str
    duration_ms: int
    markers: list[str]
    parametrized: bool
    flaky: bool
    created_at: datetime


def test_catalog_row(
    seed: TestCatalogSeed,
) -> tuple[
    str,
    int,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    int,
    list[str],
    bool,
    bool,
    datetime,
]:
    """Row for analytics.test_catalog.

    Returns
    -------
    tuple[
        str, int, str, str, str, str, str, str, str, int, list[str], bool, bool, datetime
    ]
        Row values in schema order.
    """
    return (
        seed.test_id,
        seed.test_goid_h128,
        seed.urn,
        seed.repo,
        seed.commit,
        seed.rel_path,
        seed.qualname,
        seed.kind,
        seed.status,
        seed.duration_ms,
        seed.markers,
        seed.parametrized,
        seed.flaky,
        seed.created_at,
    )


@dataclass
class TypednessSeed:
    repo: str
    commit: str
    path: str
    type_error_count: int
    annotation_ratio_json: dict[str, float]
    untyped_defs: int
    overlay_needed: bool


def typedness_row(seed: TypednessSeed) -> tuple[str, str, str, int, dict[str, float], int, bool]:
    """Row for analytics.typedness.

    Returns
    -------
    tuple[str, str, str, int, str, int, bool]
        Row values in schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.path,
        seed.type_error_count,
        seed.annotation_ratio_json,
        seed.untyped_defs,
        seed.overlay_needed,
    )


@dataclass
class StaticDiagnosticsSeed:
    repo: str
    commit: str
    rel_path: str
    pyrefly_errors: int
    pyright_errors: int
    ruff_errors: int
    total_errors: int
    has_errors: bool


def static_diagnostics_row(
    seed: StaticDiagnosticsSeed,
) -> tuple[str, str, str, int, int, int, int, bool]:
    """Row for analytics.static_diagnostics.

    Returns
    -------
    tuple[str, str, str, int, int, int, int, bool]
        Row values in schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.rel_path,
        seed.pyrefly_errors,
        seed.pyright_errors,
        seed.ruff_errors,
        seed.total_errors,
        seed.has_errors,
    )


# =============================================================================
# Profile adapter row helpers
# =============================================================================


def function_profile_row(
    *,
    goid: Decimal = Decimal(12345),
    qualname: str = "module.function_name",
    rel_path: str = "src/module.py",
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
    **overrides: object,
) -> dict[str, Any]:
    """Build a function_profile row mapping with sensible defaults.

    Returns
    -------
    dict[str, Any]
        Row mapping aligned with analytics.function_profile schema.
    """
    base: dict[str, Any] = {
        "function_goid_h128": goid,
        "urn": f"urn:{repo}::{qualname}",
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "module": rel_path.replace("/", ".").replace(".py", ""),
        "language": "python",
        "kind": "function",
        "qualname": qualname,
        "start_line": 10,
        "end_line": 30,
        "loc": 50,
        "logical_loc": 40,
        "cyclomatic_complexity": 5,
        "complexity_bucket": "low",
        "param_count": 3,
        "positional_params": 2,
        "keyword_params": 1,
        "vararg": False,
        "kwarg": False,
        "max_nesting_depth": 2,
        "stmt_count": 15,
        "decorator_count": 1,
        "has_docstring": True,
        "total_params": 3,
        "annotated_params": 3,
        "return_type": "str",
        "param_types": ["int", "str", "bool"],
        "fully_typed": True,
        "partial_typed": False,
        "untyped": False,
        "typedness_bucket": "fully_typed",
        "typedness_source": "annotations",
        "file_typed_ratio": 0.95,
        "static_error_count": 0,
        "has_static_errors": False,
        "executable_lines": 40,
        "covered_lines": 35,
        "coverage_ratio": 0.875,
        "tested": True,
        "untested_reason": None,
        "tests_touching": 5,
        "failing_tests": 0,
        "slow_tests": 0,
        "flaky_tests": 0,
        "last_test_status": "passed",
        "dominant_test_status": "passed",
        "slow_test_threshold_ms": 1000.0,
        "created_in_commit": commit,
        "created_at_history": datetime.now(tz=UTC),
        "last_modified_commit": commit,
        "last_modified_at": datetime.now(tz=UTC),
        "age_days": 30,
        "commit_count": 10,
        "author_count": 3,
        "lines_added": 100,
        "lines_deleted": 20,
        "churn_score": 0.5,
        "stability_bucket": "stable",
        "call_fan_in": 5,
        "call_fan_out": 3,
        "call_edge_in_count": 5,
        "call_edge_out_count": 3,
        "call_is_leaf": False,
        "call_is_entrypoint": False,
        "call_is_public": True,
        "risk_score": 0.25,
        "risk_level": "low",
        "risk_component_coverage": 0.1,
        "risk_component_complexity": 0.05,
        "risk_component_static": 0.0,
        "risk_component_hotspot": 0.1,
        "is_pure": True,
        "uses_io": False,
        "touches_db": False,
        "uses_time": False,
        "uses_randomness": False,
        "modifies_globals": False,
        "modifies_closure": False,
        "spawns_threads_or_tasks": False,
        "has_transitive_effects": False,
        "purity_confidence": 0.95,
        "param_nullability_json": [],
        "return_nullability": "non_null",
        "has_preconditions": False,
        "has_postconditions": False,
        "has_raises": False,
        "contract_confidence": 0.9,
        "role": "helper",
        "framework": None,
        "role_confidence": 0.85,
        "role_sources_json": ["path_hint"],
        "tags": [],
        "owners": [],
        "doc_short": "Test function.",
        "doc_long": "A test function for unit tests.",
        "doc_params": {"param1": "int", "param2": "str"},
        "doc_returns": {"type": "str", "description": "A string result"},
        "created_at": datetime.now(tz=UTC),
    }
    base.update(overrides)
    return base


def file_profile_row(
    *,
    rel_path: str = "src/services/api.py",
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
    **overrides: object,
) -> dict[str, Any]:
    """Build a file_profile row mapping with sensible defaults.

    Returns
    -------
    dict[str, Any]
        Row mapping aligned with analytics.file_profile schema.
    """
    base: dict[str, Any] = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "module": rel_path.replace("/", ".").replace(".py", ""),
        "language": "python",
        "node_count": 100,
        "function_count": 10,
        "class_count": 2,
        "avg_depth": 2.5,
        "max_depth": 5,
        "ast_complexity": 15.0,
        "hotspot_score": 0.75,
        "commit_count": 50,
        "author_count": 5,
        "lines_added": 500,
        "lines_deleted": 200,
        "annotation_ratio": 0.85,
        "untyped_defs": 2,
        "overlay_needed": False,
        "type_error_count": 0,
        "static_error_count": 0,
        "has_static_errors": False,
        "total_functions": 10,
        "public_functions": 8,
        "avg_loc": 25.0,
        "max_loc": 100,
        "avg_cyclomatic_complexity": 3.5,
        "max_cyclomatic_complexity": 8,
        "high_risk_function_count": 1,
        "medium_risk_function_count": 3,
        "max_risk_score": 0.65,
        "file_coverage_ratio": 0.85,
        "tested_function_count": 8,
        "untested_function_count": 2,
        "tests_touching": 15,
        "tags": [],
        "owners": [],
        "created_at": datetime.now(tz=UTC),
    }
    base.update(overrides)
    return base


def module_profile_row(
    *,
    module: str = "services.api",
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
    **overrides: object,
) -> dict[str, Any]:
    """Build a module_profile row mapping with sensible defaults.

    Returns
    -------
    dict[str, Any]
        Row mapping aligned with analytics.module_profile schema.
    """
    base: dict[str, Any] = {
        "repo": repo,
        "commit": commit,
        "module": module,
        "path": module.replace(".", "/"),
        "language": "python",
        "file_count": 5,
        "total_loc": 500,
        "total_logical_loc": 400,
        "function_count": 25,
        "class_count": 5,
        "avg_file_complexity": 10.0,
        "max_file_complexity": 25.0,
        "high_risk_function_count": 2,
        "medium_risk_function_count": 5,
        "low_risk_function_count": 18,
        "max_risk_score": 0.85,
        "avg_risk_score": 0.35,
        "module_coverage_ratio": 0.75,
        "tested_function_count": 20,
        "untested_function_count": 5,
        "import_fan_in": 10,
        "import_fan_out": 15,
        "cycle_group": None,
        "in_cycle": False,
        "role": "service",
        "role_confidence": 0.9,
        "role_sources_json": ["path_hint", "decorator"],
        "tags": [],
        "owners": [],
        "created_at": datetime.now(tz=UTC),
    }
    base.update(overrides)
    return base


@dataclass
class ConfigValueSeed:
    repo: str
    commit: str
    config_path: str
    format: str
    key: str
    value: str | None
    section: str | None
    seq_no: int


def config_value_row(
    seed: ConfigValueSeed,
) -> tuple[str, str, str, str, str, str | None, str | None, int]:
    """Row for analytics.config_values.

    Returns
    -------
    tuple[str, str, str, str, str, str | None, str | None, int]
        Row values in schema order.
    """
    return (
        seed.repo,
        seed.commit,
        seed.config_path,
        seed.format,
        seed.key,
        seed.value,
        seed.section,
        seed.seq_no,
    )


__all__ = [
    "AstMetricSeed",
    "AstMetricsRow",
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "ConfigValueRow",
    "ConfigValueSeed",
    "DFGEdgeRow",
    "DataModelFieldSeed",
    "DataModelRelationshipSeed",
    "DataModelSeed",
    "DataModelUsagePayloadSeed",
    "DataModelUsageSeed",
    "DatasetDataflowEdgeRow",
    "DatasetDataflowNodeRow",
    "DependencyAggregatePayloadSeed",
    "DependencyAggregateRow",
    "DependencyAggregateSeed",
    "DependencyCallPayloadSeed",
    "DependencyCallRow",
    "DependencyCallSeed",
    "DocstringRow",
    "EntrypointPayloadSeed",
    "EntrypointSeed",
    "EntrypointTestPayloadSeed",
    "EntrypointTestSeed",
    "FileProfileRowModel",
    "FunctionContextBuilder",
    "FunctionMetricsRow",
    "FunctionProfileRowModel",
    "FunctionTypesRow",
    "FunctionValidationRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "GraphMetricsModulesExtRow",
    "HotspotRow",
    "ImportGraphEdgeRow",
    "ModuleProfileRowModel",
    "ModuleRow",
    "ProfileRowModel",
    "RepoMapRow",
    "RiskFactorRow",
    "RowCoercions",
    "RowFactory",
    "SemanticRoleFunctionSeed",
    "SemanticRoleModuleSeed",
    "StaticDiagnosticsRow",
    "StaticDiagnosticsSeed",
    "SubsystemModulePayloadSeed",
    "SubsystemModuleRow",
    "SubsystemModuleSeed",
    "SubsystemPayloadSeed",
    "SubsystemRow",
    "SubsystemSeed",
    "SymbolEdgeOptions",
    "SymbolGraphMetricsModulesRow",
    "SymbolUseEdgeRow",
    "TestCatalogRow",
    "TestCatalogSeed",
    "TypednessRow",
    "TypednessSeed",
    "ast_metric_row",
    "blank_file_profile_row",
    "blank_function_profile_row",
    "blank_module_profile_row",
    "blank_row",
    "blank_test_profile_row",
    "columnar_rows_for",
    "compute_dep_id",
    "config_value_row",
    "data_model_field_row",
    "data_model_relationship_row",
    "data_model_row",
    "data_model_usage_payload",
    "data_model_usage_row",
    "dataclass_row",
    "dependency_aggregate_payload",
    "dependency_aggregate_row",
    "dependency_aggregate_row_from_payload",
    "dependency_call_payload",
    "dependency_call_row",
    "dependency_call_row_from_payload",
    "entrypoint_payload",
    "entrypoint_row",
    "entrypoint_test_payload",
    "entrypoint_test_row",
    "file_profile_row",
    "function_meta",
    "function_metrics_row",
    "function_profile_row",
    "insert_rows",
    "insert_symbol_use_edges",
    "list_public_exports",
    "make_symbol_use_edge_row",
    "module_profile_row",
    "module_row",
    "row_for",
    "row_list_for",
    "sample_file_profile_rows",
    "sample_function_profile_rows",
    "sample_module_profile_rows",
    "sample_test_profile_rows",
    "semantic_role_function_row",
    "semantic_role_module_row",
    "static_diagnostics_row",
    "subsystem_module_payload",
    "subsystem_module_row",
    "subsystem_payload",
    "subsystem_row",
    "test_catalog_row",
    "typedness_row",
]
