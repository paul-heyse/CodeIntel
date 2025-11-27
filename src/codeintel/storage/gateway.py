"""Composition root types and dataset registry helpers for DuckDB access."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import duckdb

from codeintel.storage.datasets import Dataset, load_dataset_registry
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets
from codeintel.storage.schemas import apply_all_schemas, assert_schema_alignment
from codeintel.storage.views import create_all_views

DuckDBConnection = duckdb.DuckDBPyConnection
DuckDBRelation = duckdb.DuckDBPyRelation
DuckDBError = duckdb.Error


def _macro_insert_rows(
    con: DuckDBConnection,
    table_key: str,
    rows: Iterable[Sequence[object]],
) -> None:
    """
    Insert rows via ingest macro using table schema as ground truth.

    Pads missing trailing columns with NULLs; raises if rows exceed schema width.

    Raises
    ------
    ValueError
        If the table name is invalid or rows exceed the column count.
    """
    rows_list = list(rows)
    if not rows_list:
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", table_key):
        message = f"Invalid table key: {table_key}"
        raise ValueError(message)
    schema_name, table_name = table_key.split(".", maxsplit=1)
    table_sql = f'"{schema_name}"."{table_name}"'
    pragma_sql = f"PRAGMA table_info({table_sql})"
    columns = [row[1] for row in con.execute(pragma_sql).fetchall()]
    if not columns:
        message = f"Table {table_key} missing"
        raise ValueError(message)
    col_count = len(columns)
    normalized: list[tuple[object, ...]] = []
    for row in rows_list:
        if len(row) > col_count:
            message = (
                f"Row for {table_key} has {len(row)} values, exceeds column count {col_count}"
            )
            raise ValueError(message)
        padded = tuple(row) + (None,) * (col_count - len(row))
        normalized.append(padded)
    view_name = f"temp_ingest_values_{table_name}"
    con.execute(f"DROP TABLE IF EXISTS {view_name}")
    con.execute(
        f"CREATE TEMP TABLE {view_name} AS SELECT * FROM {table_sql} WHERE 0=1"  # noqa: S608
    )
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    con.executemany(
        f"INSERT INTO {view_name} ({column_list}) VALUES ({placeholders})",  # noqa: S608 - validated
        normalized,
    )
    macro_name = f"metadata.ingest_{table_name}"
    con.execute(
        f"INSERT INTO {table_sql} SELECT * FROM {macro_name}(?)", [view_name]  # noqa: S608 - validated
    )
    con.execute(f"DROP TABLE IF EXISTS {view_name}")


@dataclass(frozen=True)
class StorageConfig:
    """Define configuration for opening a CodeIntel DuckDB database."""

    db_path: Path
    read_only: bool = False
    apply_schema: bool = False
    ensure_views: bool = False
    validate_schema: bool = False
    attach_history: bool = False
    history_db_path: Path | None = None
    repo: str | None = None
    commit: str | None = None

    @classmethod
    def for_ingest(
        cls,
        db_path: Path,
        *,
        history_db_path: Path | None = None,
    ) -> StorageConfig:
        """
        Build a write-capable configuration used by ingestion and analytics runs.

        Parameters
        ----------
        db_path
            Primary DuckDB database path.
        history_db_path
            Optional history database to attach for cross-commit analytics.

        Returns
        -------
        StorageConfig
            Preconfigured ingest-ready storage configuration.
        """
        return cls(
            db_path=db_path,
            read_only=False,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
            attach_history=history_db_path is not None,
            history_db_path=history_db_path,
        )

    @classmethod
    def for_readonly(cls, db_path: Path) -> StorageConfig:
        """
        Build a read-only configuration for serving/inspection surfaces.

        Parameters
        ----------
        db_path
            DuckDB database path to open read-only.

        Returns
        -------
        StorageConfig
            Preconfigured read-only storage configuration.
        """
        return cls(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=True,
            validate_schema=True,
        )


@dataclass(frozen=True)
class DatasetRegistry:
    """Track known table and view dataset names and export metadata."""

    mapping: Mapping[str, str]
    tables: tuple[str, ...]
    views: tuple[str, ...]
    meta: Mapping[str, Dataset] | None = None
    jsonl_mapping: Mapping[str, str] | None = None
    parquet_mapping: Mapping[str, str] | None = None

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """
        Return all registered dataset identifiers.

        Returns
        -------
        tuple[str, ...]
            Combined table and view names.
        """
        return self.tables + self.views

    def resolve(self, name: str) -> str:
        """
        Return a validated dataset table or view key.

        Parameters
        ----------
        name
            Dataset identifier to validate.

        Returns
        -------
        str
            Fully qualified dataset name.

        Raises
        ------
        KeyError
            If the dataset name is unknown.
        """
        if name not in self.mapping:
            message = f"Unknown dataset: {name}"
            raise KeyError(message)
        return self.mapping[name]


class StorageGateway(Protocol):
    """Expose DuckDB access along with dataset registry metadata."""

    config: StorageConfig
    datasets: DatasetRegistry
    core: CoreTables
    graph: GraphTables
    docs: DocsViews
    analytics: AnalyticsTables

    @property
    def con(self) -> DuckDBConnection:
        """
        Return an open DuckDB connection.

        Returns
        -------
        DuckDBConnection
            Live connection bound to the configured database.
        """
        ...

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        ...

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection:
        """Execute SQL against the underlying connection."""
        ...

    def table(self, name: str) -> DuckDBRelation:
        """Return a relation for a fully qualified table or view name."""
        ...


SnapshotGatewayResolver = Callable[[str], StorageGateway]
"""Callable returning a StorageGateway for a given commit."""


def build_dataset_registry(
    con: DuckDBConnection,
    *,
    include_views: bool = True,
) -> DatasetRegistry:
    """
    Build a dataset registry from DuckDB's metadata.datasets catalog.

    Parameters
    ----------
    con
        Active DuckDB connection with metadata tables initialized.
    include_views
        When True, include docs views alongside base tables.

    Returns
    -------
    DatasetRegistry
        Registry containing dataset metadata and export filenames.
    """
    ds_registry = load_dataset_registry(con)
    mapping: dict[str, str] = {name: ds.table_key for name, ds in ds_registry.by_name.items()}
    table_names = tuple(name for name, ds in ds_registry.by_name.items() if not ds.is_view)
    view_names = (
        tuple(name for name, ds in ds_registry.by_name.items() if ds.is_view)
        if include_views
        else ()
    )
    return DatasetRegistry(
        mapping=mapping,
        tables=table_names,
        views=view_names,
        meta=ds_registry.by_name,
        jsonl_mapping=ds_registry.jsonl_datasets,
        parquet_mapping=ds_registry.parquet_datasets,
    )


@dataclass(frozen=True)
class CoreTables:
    """Accessors for core schema tables."""

    con: DuckDBConnection

    def goids(self) -> DuckDBRelation:
        """
        Return relation for core.goids.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.goids.
        """
        return self.con.table("core.goids")

    def modules(self) -> DuckDBRelation:
        """
        Return relation for core.modules.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.modules.
        """
        return self.con.table("core.modules")

    def repo_map(self) -> DuckDBRelation:
        """
        Return relation for core.repo_map.

        Returns
        -------
        DuckDBRelation
            Relation selecting core.repo_map.
        """
        return self.con.table("core.repo_map")

    def insert_repo_map(
        self,
        rows: Iterable[tuple[str, str, str, str, str]],
    ) -> None:
        """
        Insert rows into core.repo_map.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, modules_json, overlays_json, generated_at_iso).
        """
        _macro_insert_rows(self.con, "core.repo_map", rows)

    def insert_modules(
        self,
        rows: Iterable[tuple[str, str, str, str]],
    ) -> None:
        """
        Insert rows into core.modules.

        Parameters
        ----------
        rows
            Iterable of (module, path, repo, commit).
        """
        normalized = [
            (module, path, repo, commit, "python", "[]", "[]") for module, path, repo, commit in rows
        ]
        _macro_insert_rows(self.con, "core.modules", normalized)

    def insert_goids(
        self,
        rows: Iterable[
            tuple[
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
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into core.goids.

        Parameters
        ----------
        rows
            Iterable of (goid_h128, urn, repo, commit, rel_path, language, kind,
            qualname, start_line, end_line, created_at_iso).
        """
        _macro_insert_rows(self.con, "core.goids", rows)


@dataclass(frozen=True)
class GraphTables:
    """Accessors for graph schema tables."""

    con: DuckDBConnection

    def call_graph_edges(self) -> DuckDBRelation:
        """
        Return relation for graph.call_graph_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.call_graph_edges.
        """
        return self.con.table("graph.call_graph_edges")

    def insert_call_graph_edges(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                int,
                int | None,
                str,
                int,
                int,
                str,
                str,
                str,
                float,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into graph.call_graph_edges.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, caller_goid_h128, callee_goid_h128,
            callsite_path, callsite_line, callsite_col, language, kind,
            resolved_via, confidence, evidence_json).
        """
        _macro_insert_rows(self.con, "graph.call_graph_edges", rows)

    def call_graph_nodes(self) -> DuckDBRelation:
        """
        Return relation for graph.call_graph_nodes.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.call_graph_nodes.
        """
        return self.con.table("graph.call_graph_nodes")

    def insert_call_graph_nodes(
        self,
        rows: Iterable[tuple[int, str, str, int, bool, str]],
    ) -> None:
        """
        Insert rows into graph.call_graph_nodes.

        Parameters
        ----------
        rows
            Iterable of (goid_h128, language, kind, arity, is_public, rel_path).
        """
        _macro_insert_rows(self.con, "graph.call_graph_nodes", rows)

    def import_graph_edges(self) -> DuckDBRelation:
        """
        Return relation for graph.import_graph_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.import_graph_edges.
        """
        return self.con.table("graph.import_graph_edges")

    def insert_import_graph_edges(
        self,
        rows: Iterable[tuple[str, str, str, str, int, int, int]],
    ) -> None:
        """
        Insert rows into graph.import_graph_edges.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, src_module, dst_module, src_fan_out,
            dst_fan_in, cycle_group).
        """
        _macro_insert_rows(self.con, "graph.import_graph_edges", rows)

    def symbol_use_edges(self) -> DuckDBRelation:
        """
        Return relation for graph.symbol_use_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting graph.symbol_use_edges.
        """
        return self.con.table("graph.symbol_use_edges")

    def insert_symbol_use_edges(
        self,
        rows: Iterable[Sequence[object]],
    ) -> None:
        """
        Insert rows into graph.symbol_use_edges.

        Parameters
        ----------
        rows
            Iterable of (symbol, def_path, use_path, same_file, same_module, def_goid_h128, use_goid_h128).

        Raises
        ------
        ValueError
            If a row is not length 5 or 7.
        """
        expected_basic_len = 5
        expected_full_len = 7
        normalized_rows = []
        for row in rows:
            if len(row) == expected_basic_len:
                symbol, def_path, use_path, same_file, same_module = row
                normalized_rows.append(
                    (symbol, def_path, use_path, same_file, same_module, None, None)
                )
            elif len(row) == expected_full_len:
                normalized_rows.append(tuple(row))
            else:
                message = (
                    "symbol_use_edges rows must have 5 or 7 fields, "
                    f"got {len(row)}: {row}"
                )
                raise ValueError(message)
        _macro_insert_rows(self.con, "graph.symbol_use_edges", normalized_rows)

    def insert_cfg_blocks(
        self,
        rows: Iterable[tuple[int, int, str, str, str, int, int, str, str, int, int]],
    ) -> None:
        """
        Insert rows into graph.cfg_blocks.

        Parameters
        ----------
        rows
            Iterable of values matching cfg_blocks columns.
        """
        _macro_insert_rows(self.con, "graph.cfg_blocks", rows)

    def insert_cfg_edges(
        self,
        rows: Iterable[tuple[int, str, str, str | None]],
    ) -> None:
        """
        Insert rows into graph.cfg_edges.

        Parameters
        ----------
        rows
            Iterable of (function_goid_h128, src_block_id, dst_block_id, edge_kind).
        """
        _macro_insert_rows(self.con, "graph.cfg_edges", rows)

    def insert_dfg_edges(
        self,
        rows: Iterable[tuple[int, str, str, str | None, str | None, str | None]],
    ) -> None:
        """
        Insert rows into graph.dfg_edges.

        Parameters
        ----------
        rows
            Iterable of values matching dfg_edges columns.
        """
        _macro_insert_rows(self.con, "graph.dfg_edges", rows)


@dataclass(frozen=True)
class DocsViews:
    """Accessors for docs schema views."""

    con: DuckDBConnection

    def function_summary(self) -> DuckDBRelation:
        """
        Return relation for docs.v_function_summary.

        Returns
        -------
        DuckDBRelation
            Relation selecting docs.v_function_summary.
        """
        return self.con.table("docs.v_function_summary")

    def call_graph_enriched(self) -> DuckDBRelation:
        """
        Return relation for docs.v_call_graph_enriched.

        Returns
        -------
        DuckDBRelation
            Relation selecting docs.v_call_graph_enriched.
        """
        return self.con.table("docs.v_call_graph_enriched")

    def function_profile(self) -> DuckDBRelation:
        """
        Return relation for docs.v_function_profile.

        Returns
        -------
        DuckDBRelation
            Relation selecting docs.v_function_profile.
        """
        return self.con.table("docs.v_function_profile")


@dataclass(frozen=True)
class AnalyticsTables:
    """Accessors for analytics schema tables."""

    con: DuckDBConnection

    def function_metrics(self) -> DuckDBRelation:
        """
        Return relation for analytics.function_metrics.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.function_metrics.
        """
        return self.con.table("analytics.function_metrics")

    def function_types(self) -> DuckDBRelation:
        """
        Return relation for analytics.function_types.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.function_types.
        """
        return self.con.table("analytics.function_types")

    def coverage_functions(self) -> DuckDBRelation:
        """
        Return relation for analytics.coverage_functions.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.coverage_functions.
        """
        return self.con.table("analytics.coverage_functions")

    def insert_coverage_functions(
        self,
        rows: Iterable[
            tuple[
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
                str,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.coverage_functions.

        Parameters
        ----------
        rows
            Iterable of values matching coverage_functions columns.
        """
        _macro_insert_rows(self.con, "analytics.coverage_functions", rows)

    def coverage_lines(self) -> DuckDBRelation:
        """
        Return relation for analytics.coverage_lines.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.coverage_lines.
        """
        return self.con.table("analytics.coverage_lines")

    def insert_coverage_lines(
        self,
        rows: Iterable[tuple[str, str, str, int, bool, bool, int, int, str]],
    ) -> None:
        """
        Insert rows into analytics.coverage_lines.

        Parameters
        ----------
        rows
            Iterable of values matching coverage_lines columns.
        """
        _macro_insert_rows(self.con, "analytics.coverage_lines", rows)

    def test_catalog(self) -> DuckDBRelation:
        """
        Return relation for analytics.test_catalog.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.test_catalog.
        """
        return self.con.table("analytics.test_catalog")

    def insert_test_catalog(
        self,
        rows: Iterable[
            tuple[
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
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.test_catalog.

        Parameters
        ----------
        rows
            Iterable of values matching test_catalog columns.
        """
        _macro_insert_rows(self.con, "analytics.test_catalog", rows)

    def test_coverage_edges(self) -> DuckDBRelation:
        """
        Return relation for analytics.test_coverage_edges.

        Returns
        -------
        DuckDBRelation
            Relation selecting analytics.test_coverage_edges.
        """
        return self.con.table("analytics.test_coverage_edges")

    def insert_test_coverage_edges(
        self,
        rows: Iterable[
            tuple[
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
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.test_coverage_edges.

        Parameters
        ----------
        rows
            Iterable of values matching test_coverage_edges columns.
        """
        _macro_insert_rows(self.con, "analytics.test_coverage_edges", rows)

    def insert_function_metrics(
        self,
        rows: Iterable[
            tuple[
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
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.function_metrics.

        Parameters
        ----------
        rows
            Iterable of values matching function_metrics columns.
        """
        _macro_insert_rows(self.con, "analytics.function_metrics", rows)

    def insert_goid_risk_factors(
        self,
        rows: Iterable[
            tuple[
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
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.goid_risk_factors.

        Parameters
        ----------
        rows
            Iterable of values matching goid_risk_factors columns.
        """
        _macro_insert_rows(self.con, "analytics.goid_risk_factors", rows)

    def insert_config_values(
        self,
        rows: Iterable[tuple[str, str, str, str, str | None, str | None, str | None, int]],
    ) -> None:
        """
        Insert rows into analytics.config_values.

        Parameters
        ----------
        rows
            Iterable of (repo, commit, config_path, format, key, reference_paths,
            reference_modules, reference_modules_json, reference_count).
        """
        _macro_insert_rows(self.con, "analytics.config_values", rows)

    def insert_typedness(
        self,
        rows: Iterable[tuple[str, str, str, int, str, int, bool]],
    ) -> None:
        """
        Insert rows into analytics.typedness.

        Parameters
        ----------
        rows
            Iterable of values matching typedness columns.
        """
        _macro_insert_rows(self.con, "analytics.typedness", rows)

    def insert_static_diagnostics(
        self,
        rows: Iterable[tuple[str, str, str, int, int, int, int, bool]],
    ) -> None:
        """
        Insert rows into analytics.static_diagnostics.

        Parameters
        ----------
        rows
            Iterable of values matching static_diagnostics columns.
        """
        _macro_insert_rows(self.con, "analytics.static_diagnostics", rows)

    def insert_graph_metrics_functions(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                int,
                int,
                int,
                int,
                int,
                float | None,
                float | None,
                float | None,
                bool,
                int | None,
                int | None,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.graph_metrics_functions.

        Parameters
        ----------
        rows
            Iterable of values matching graph_metrics_functions columns.
        """
        _macro_insert_rows(self.con, "analytics.graph_metrics_functions", rows)

    def insert_graph_metrics_modules(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                str,
                int,
                int,
                int,
                int,
                float | None,
                float | None,
                float | None,
                bool,
                int | None,
                int | None,
                int,
                int,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.graph_metrics_modules.

        Parameters
        ----------
        rows
            Iterable of values matching graph_metrics_modules columns.
        """
        _macro_insert_rows(self.con, "analytics.graph_metrics_modules", rows)

    def insert_subsystems(
        self,
        rows: Iterable[
            tuple[
                str,
                str,
                str,
                str,
                str | None,
                int,
                str,
                str | None,
                int,
                int,
                int,
                int,
                int,
                float | None,
                float | None,
                int,
                str | None,
                str,
            ]
        ],
    ) -> None:
        """
        Insert rows into analytics.subsystems.

        Parameters
        ----------
        rows
            Iterable of values matching subsystems columns.
        """
        _macro_insert_rows(self.con, "analytics.subsystems", rows)

    def insert_subsystem_modules(
        self,
        rows: Iterable[tuple[str, str, str, str, str | None]],
    ) -> None:
        """
        Insert rows into analytics.subsystem_modules.

        Parameters
        ----------
        rows
            Iterable of values matching subsystem_modules columns.
        """
        _macro_insert_rows(self.con, "analytics.subsystem_modules", rows)


@dataclass
class _DuckDBGateway:
    """Concrete StorageGateway implementation."""

    config: StorageConfig
    datasets: DatasetRegistry
    con: DuckDBConnection
    core: CoreTables = field(init=False)
    graph: GraphTables = field(init=False)
    docs: DocsViews = field(init=False)
    analytics: AnalyticsTables = field(init=False)

    def __post_init__(self) -> None:
        self.core = CoreTables(self.con)
        self.graph = GraphTables(self.con)
        self.docs = DocsViews(self.con)
        self.analytics = AnalyticsTables(self.con)

    def close(self) -> None:
        """Close the underlying connection."""
        self.con.close()

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection:
        """
        Execute a SQL statement using the active DuckDB connection.

        Returns
        -------
        DuckDBConnection
            Connection representing the executed query.
        """
        return self.con.execute(sql, params)

    def table(self, name: str) -> DuckDBRelation:
        """
        Return a relation object for the specified table or view.

        Returns
        -------
        DuckDBRelation
            Relation bound to the requested table/view.
        """
        return self.con.table(name)


def _connect(config: StorageConfig) -> DuckDBConnection:
    """
    Open a DuckDB connection using the provided configuration.

    Parameters
    ----------
    config
        Storage configuration controlling path, schema application, and validation.

    Returns
    -------
    DuckDBConnection
        Live DuckDB connection with optional schema/views applied.

    Raises
    ------
    ValueError
        Raised when attach_history is requested without a history path.
    FileNotFoundError
        Raised when the configured history database does not exist.
    """
    if not config.read_only and config.db_path != Path(":memory:"):
        config.db_path.parent.mkdir(parents=True, exist_ok=True)

    con: DuckDBConnection
    if not config.read_only and config.db_path != Path(":memory:") and not config.db_path.exists():
        con = duckdb.connect(str(Path(":memory:")))
        db_path_str = str(config.db_path).replace("'", "''")
        con.execute(f"ATTACH DATABASE '{db_path_str}' AS main_db (STORAGE_VERSION 'latest')")
        con.execute("USE main_db")
    else:
        con = duckdb.connect(str(config.db_path), read_only=config.read_only)
    if config.attach_history:
        if config.history_db_path is None:
            message = "attach_history requires history_db_path"
            raise ValueError(message)
        if not config.history_db_path.exists():
            message = f"History database not found: {config.history_db_path}"
            raise FileNotFoundError(message)
        con.execute(
            "ATTACH DATABASE ? AS history",
            [str(config.history_db_path)],
        )
    if config.apply_schema and not config.read_only:
        apply_all_schemas(con)
    if config.ensure_views and not config.read_only:
        create_all_views(con)
    if config.validate_schema:
        assert_schema_alignment(con, strict=True)
    return con


def open_gateway(config: StorageConfig) -> StorageGateway:
    """
    Create a StorageGateway bound to a DuckDB database.

    Parameters
    ----------
    config
        Storage configuration describing connection options.

    Returns
    -------
    StorageGateway
        Gateway exposing typed accessors and dataset registry.
    """
    con = _connect(config)
    if not config.read_only:
        bootstrap_metadata_datasets(con)
    datasets = build_dataset_registry(con)
    return _DuckDBGateway(config=config, datasets=datasets, con=con)


def build_snapshot_gateway_resolver(
    *,
    db_dir: Path,
    repo: str | None = None,
    primary_gateway: StorageGateway | None = None,
) -> SnapshotGatewayResolver:
    """
    Build a resolver that opens per-commit snapshot databases as StorageGateways.

    Parameters
    ----------
    db_dir:
        Directory containing per-commit DuckDB snapshots, named
        ``codeintel-<commit>.duckdb``.
    repo:
        Optional repository slug to record in the StorageConfig for observability.
    primary_gateway:
        Optional gateway to reuse when the requested commit resolves to the same
        database path, avoiding duplicate connections with conflicting settings.

    Returns
    -------
    SnapshotGatewayResolver
        Callable that returns a read-only StorageGateway for the given commit.
    """

    def _resolve(commit: str) -> StorageGateway:
        db_path = db_dir / f"codeintel-{commit}.duckdb"
        if (
            primary_gateway is not None
            and db_path.resolve() == primary_gateway.config.db_path.resolve()
        ):
            return primary_gateway
        if not db_path.is_file():
            message = f"Missing snapshot database for commit {commit}: {db_path}"
            raise FileNotFoundError(message)
        cfg = StorageConfig(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
            repo=repo,
            commit=commit,
        )
        return open_gateway(cfg)

    return _resolve


def open_memory_gateway(
    *,
    apply_schema: bool = True,
    ensure_views: bool = False,
    validate_schema: bool = True,
) -> StorageGateway:
    """
    Create an in-memory StorageGateway for tests.

    Parameters
    ----------
    apply_schema
        When True, apply all table schemas to the in-memory database.
    ensure_views
        When True, create docs views after schema application.
    validate_schema
        When True, validate schema alignment after setup.

    Returns
    -------
    StorageGateway
        Gateway backed by an in-memory DuckDB connection.
    """
    cfg = StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=apply_schema,
        ensure_views=ensure_views,
        validate_schema=validate_schema,
    )
    return open_gateway(cfg)
