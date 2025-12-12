"""Bootstrap DuckDB metadata catalog for datasets and apply metadata macros."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    build_contract_dataflow_graph,
)
from codeintel.storage.macros.generation import render_macro
from codeintel.storage.sql import safe_macro_call

if TYPE_CHECKING:
    from collections.abc import Mapping

    from duckdb import DuckDBPyConnection


def _canonical_type(type_str: str) -> str:
    upper = type_str.upper()
    if upper in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMPTZ"
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


def _is_dataset_rows_only(table_key: str) -> bool:
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    return contract is not None and "dataset_rows_only" in contract.tags


def _expected_schema_hash(table_key: str) -> str:
    schema = DATASET_CONTRACTS_BY_TABLE_KEY[table_key].schema
    if schema is None:
        message = f"Cannot compute schema hash for view or missing schema: {table_key}"
        raise KeyError(message)
    parts: list[str] = []
    for column in schema.columns:
        canonical_type = _canonical_type(column.type)
        parts.append(f"{column.name}:{canonical_type}")
    normalized = "|".join(parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _assert_macro_coverage() -> None:
    datasets = {
        key
        for key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
        if contract.schema is not None
    }
    macro_backed = set(NORMALIZED_MACROS)
    dataset_rows_only = {table_key for table_key in datasets if _is_dataset_rows_only(table_key)}
    unexpected_dataset_rows = datasets - macro_backed - dataset_rows_only
    if unexpected_dataset_rows:
        message = "Datasets missing normalized macros or allowlist entries: " + ", ".join(
            sorted(unexpected_dataset_rows)
        )
        raise RuntimeError(message)


def dataset_rows_only_entries() -> list[str]:
    """Return datasets explicitly allowlisted for dataset_rows-only reads.

    Returns
    -------
    list[str]
        Sorted dataset identifiers permitted to use dataset_rows-only reads.
    """
    return sorted(
        table_key
        for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
        if contract.schema is not None and _is_dataset_rows_only(table_key)
    )


def load_dataset_schema_registry(con: DuckDBPyConnection) -> dict[str, str]:
    """
    Return the dataset schema hashes recorded in metadata.dataset_schema_registry.

    Parameters
    ----------
    con :
        Active DuckDB connection.

    Returns
    -------
    dict[str, str]
        Mapping of table key -> schema hash string.
    """
    rows = con.execute(
        "SELECT table_key, schema_hash FROM metadata.dataset_schema_registry"
    ).fetchall()
    return {str(table_key): str(schema_hash) for table_key, schema_hash in rows}


def load_macro_registry(con: DuckDBPyConnection) -> dict[str, tuple[str | None, str, str | None]]:
    """
    Return macro registry entries keyed by macro name.

    Parameters
    ----------
    con :
        Active DuckDB connection.

    Returns
    -------
    dict[str, tuple[str | None, str, str | None]]
        Mapping of macro_name -> (dataset_table_key, ddl_hash, schema_hash).
    """
    rows = con.execute(
        """
        SELECT macro_name, dataset_table_key, ddl_hash, schema_hash
        FROM metadata.macro_registry
        """
    ).fetchall()
    registry: dict[str, tuple[str | None, str, str | None]] = {}
    for macro_name, dataset_table_key, ddl_hash, schema_hash in rows:
        registry[str(macro_name)] = (
            str(dataset_table_key) if dataset_table_key is not None else None,
            str(ddl_hash),
            str(schema_hash) if schema_hash is not None else None,
        )
    return registry


def _register_dataset_schema_hashes(con: DuckDBPyConnection) -> None:
    entries = {
        table_key: _expected_schema_hash(table_key)
        for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
        if contract.schema is not None
    }
    con.execute("DELETE FROM metadata.dataset_schema_registry")
    con.executemany(
        """
        INSERT INTO metadata.dataset_schema_registry (table_key, schema_hash)
        VALUES (?, ?)
        """,
        list(entries.items()),
    )


NORMALIZED_MACROS: dict[str, str] = {
    "graph.call_graph_edges": "metadata.normalized_call_graph_edges",
    "graph.symbol_use_edges": "metadata.normalized_symbol_use_edges",
    "analytics.test_coverage_edges": "metadata.normalized_test_coverage_edges",
    "analytics.function_metrics": "metadata.normalized_function_metrics",
    "analytics.function_profile": "metadata.normalized_function_profile",
    "analytics.function_history": "metadata.normalized_function_history",
    "analytics.function_types": "metadata.normalized_function_types",
    "analytics.test_catalog": "metadata.normalized_test_catalog",
    "analytics.coverage_functions": "metadata.normalized_coverage_functions",
    "analytics.goid_risk_factors": "metadata.normalized_goid_risk_factors",
    "analytics.function_validation": "metadata.normalized_function_validation",
    "analytics.graph_validation": "metadata.normalized_graph_validation",
    "graph.cfg_blocks": "metadata.normalized_cfg_blocks",
    "graph.cfg_edges": "metadata.normalized_cfg_edges",
    "graph.dfg_edges": "metadata.normalized_dfg_edges",
    "analytics.cfg_block_metrics": "metadata.normalized_cfg_block_metrics",
    "analytics.cfg_function_metrics": "metadata.normalized_cfg_function_metrics",
    "analytics.cfg_function_metrics_ext": "metadata.normalized_cfg_function_metrics_ext",
    "analytics.config_data_flow": "metadata.normalized_config_data_flow",
    "analytics.data_model_usage": "metadata.normalized_data_model_usage",
    "analytics.dfg_block_metrics": "metadata.normalized_dfg_block_metrics",
    "analytics.dfg_function_metrics": "metadata.normalized_dfg_function_metrics",
    "analytics.dfg_function_metrics_ext": "metadata.normalized_dfg_function_metrics_ext",
    "analytics.entrypoint_tests": "metadata.normalized_entrypoint_tests",
    "analytics.entrypoints": "metadata.normalized_entrypoints",
    "analytics.external_dependency_calls": "metadata.normalized_external_dependency_calls",
    "analytics.function_contracts": "metadata.normalized_function_contracts",
    "analytics.function_effects": "metadata.normalized_function_effects",
    "analytics.graph_metrics_functions": "metadata.normalized_graph_metrics_functions",
    "analytics.graph_metrics_functions_ext": "metadata.normalized_graph_metrics_functions_ext",
    "analytics.history_timeseries": "metadata.normalized_history_timeseries",
    "analytics.semantic_roles_functions": "metadata.normalized_semantic_roles_functions",
    "analytics.symbol_graph_metrics_functions": "metadata.normalized_symbol_graph_metrics_functions",
    "analytics.test_graph_metrics_functions": "metadata.normalized_test_graph_metrics_functions",
    "analytics.test_profile": "metadata.normalized_test_profile",
    "analytics.behavioral_coverage": "metadata.normalized_behavioral_coverage",
}


AUTO_NORMALIZED_MACRO_DDLS: list[str] = []
for _table_key, _contract in sorted(DATASET_CONTRACTS_BY_TABLE_KEY.items()):
    if _table_key.startswith("metadata."):
        continue
    if _contract.schema is None:
        continue
    if _is_dataset_rows_only(_table_key):
        continue
    if _table_key in NORMALIZED_MACROS:
        continue
    rendered_macro = render_macro(_table_key)
    NORMALIZED_MACROS[_table_key] = rendered_macro.macro_name
    AUTO_NORMALIZED_MACRO_DDLS.append(rendered_macro.ddl)
INGEST_MACROS: dict[str, str] = {
    table_key: f"metadata.ingest_{table_key.split('.', maxsplit=1)[1]}"
    for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
    if not table_key.startswith("metadata.") and contract.schema is not None
}


def _build_ingest_macro_ddl(macro: str) -> str:
    """
    Build DDL for an ingest macro using a validated macro name.

    Parameters
    ----------
    macro :
        Fully qualified macro name (e.g., metadata.ingest_function_metrics).

    Returns
    -------
    str
        CREATE OR REPLACE MACRO statement.

    Raises
    ------
    ValueError
        If the macro name contains unsupported characters.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", macro):
        message = f"Invalid macro name: {macro}"
        raise ValueError(message)
    template = """
    CREATE OR REPLACE MACRO {macro_name}(input_view TEXT) AS TABLE
    SELECT * FROM query_table(input_view);
    """
    return template.format(macro_name=macro)


INGEST_MACRO_DDLS: tuple[str, ...] = tuple(
    _build_ingest_macro_ddl(macro) for macro in sorted(INGEST_MACROS.values())
)


METADATA_SCHEMA_DDL_BASE: tuple[str, ...] = (
    """
    CREATE SCHEMA IF NOT EXISTS metadata;
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.macro_registry (
        macro_name TEXT PRIMARY KEY,
        dataset_table_key TEXT,
        ddl_hash TEXT NOT NULL,
        schema_hash TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_schema_registry (
        table_key TEXT PRIMARY KEY,
        schema_hash TEXT NOT NULL
    );
    """,
)

METADATA_SCHEMA_DDL_REST: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS metadata.datasets (
        table_key        TEXT PRIMARY KEY,
        name             TEXT NOT NULL,
        is_view          BOOLEAN NOT NULL,
        jsonl_filename   TEXT,
        parquet_filename TEXT,
        family           TEXT,
        description      TEXT,
        schema_version   TEXT,
        deprecated       BOOLEAN DEFAULT FALSE
    );
    """,
    """
    CREATE OR REPLACE MACRO metadata.dataset_rows(
        table_key TEXT,
        row_limit BIGINT := 100,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT *
    FROM query_table(table_key)
    LIMIT row_limit OFFSET row_offset;
    """,
    """
    CREATE OR REPLACE MACRO metadata.dataset_exists(ds_name) AS
        EXISTS (
            SELECT 1
            FROM metadata.datasets d
        WHERE d.name = ds_name
        );
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_call_graph_edges(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        repo,
        commit,
        CAST(caller_goid_h128 AS BIGINT) AS caller_goid_h128,
        CAST(callee_goid_h128 AS BIGINT) AS callee_goid_h128,
        * EXCLUDE (repo, commit, caller_goid_h128, callee_goid_h128)
    FROM metadata.dataset_rows(table_key, row_limit, row_offset);
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_symbol_use_edges(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    WITH defaults AS (
        SELECT repo, commit FROM core.repo_map LIMIT 1
    )
    SELECT
        defaults.repo AS repo,
        defaults.commit AS commit,
        ds.symbol,
        ds.def_path,
        ds.use_path,
        ds.same_file,
        ds.same_module,
        CAST(ds.def_goid_h128 AS BIGINT) AS def_goid_h128,
        CAST(ds.use_goid_h128 AS BIGINT) AS use_goid_h128
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds
    CROSS JOIN defaults;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_test_coverage_edges(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        repo,
        commit,
        test_id,
        CAST(test_goid_h128 AS BIGINT) AS test_goid_h128,
        CAST(function_goid_h128 AS BIGINT) AS function_goid_h128,
        urn,
        rel_path,
        qualname,
        covered_lines,
        executable_lines,
        coverage_ratio,
        last_status,
        CAST(created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset);
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_function_metrics(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_function_profile(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_function_history(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_function_types(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_test_catalog(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.test_goid_h128 AS BIGINT) AS test_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_coverage_functions(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_goid_risk_factors(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_function_validation(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_graph_validation(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_cfg_blocks(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_cfg_edges(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_dfg_edges(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_cfg_block_metrics(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_cfg_function_metrics(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_cfg_function_metrics_ext(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_config_data_flow(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_data_model_usage(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_dfg_block_metrics(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_dfg_function_metrics(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_dfg_function_metrics_ext(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_entrypoint_tests(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.test_goid_h128 AS BIGINT) AS test_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_entrypoints(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.handler_goid_h128 AS BIGINT) AS handler_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_external_dependency_calls(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_function_contracts(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_function_effects(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_graph_metrics_functions(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_graph_metrics_functions_ext(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_history_timeseries(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at_row AS VARCHAR) AS created_at_row
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_semantic_roles_functions(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_symbol_graph_metrics_functions(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_test_graph_metrics_functions(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.function_goid_h128 AS BIGINT) AS function_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_test_profile(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.test_goid_h128 AS BIGINT) AS test_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at,
        CAST(ds.last_run_at AS VARCHAR) AS last_run_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
    """
    CREATE OR REPLACE MACRO metadata.normalized_behavioral_coverage(
        table_key TEXT,
        row_limit BIGINT := 9223372036854775807,
        row_offset BIGINT := 0
    ) AS TABLE
    SELECT
        ds.*,
        CAST(ds.test_goid_h128 AS BIGINT) AS test_goid_h128,
        CAST(ds.created_at AS VARCHAR) AS created_at
    FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
    """,
)

METADATA_SCHEMA_DDL_REST += tuple(AUTO_NORMALIZED_MACRO_DDLS)

METADATA_SCHEMA_DDL_REST += (
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_nodes (
        id            TEXT PRIMARY KEY,
        kind          TEXT NOT NULL,
        family        TEXT,
        owner_package TEXT,
        description   TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata.dataset_dataflow_edges (
        src       TEXT NOT NULL,
        dst       TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        PRIMARY KEY (src, dst, edge_type)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_src
        ON metadata.dataset_dataflow_edges (src);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dataset_dataflow_edges_dst
        ON metadata.dataset_dataflow_edges (dst);
    """,
)

METADATA_SCHEMA_DDL: tuple[str, ...] = (
    METADATA_SCHEMA_DDL_BASE + INGEST_MACRO_DDLS + METADATA_SCHEMA_DDL_REST
)


PIPELINE_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS metadata.pipeline_runs (
    run_id              TEXT PRIMARY KEY,
    repo                TEXT NOT NULL,
    commit              TEXT NOT NULL,
    kind                TEXT NOT NULL,
    trigger             TEXT NOT NULL,
    requested_operation TEXT,
    requested_datasets  JSON,
    started_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ,
    status              TEXT NOT NULL,
    error_summary       TEXT,
    pipeline_name       TEXT
);
"""

PIPELINE_STEPS_DDL = """
CREATE TABLE IF NOT EXISTS metadata.pipeline_steps (
    run_id          TEXT NOT NULL,
    module          TEXT NOT NULL,
    stage           TEXT NOT NULL,
    name            TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL,
    row_counts      JSON,
    extra           JSON,
    PRIMARY KEY (run_id, module, name)
);
"""

PIPELINE_INDEXES_DDL = """
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_repo_commit
    ON metadata.pipeline_runs (repo, commit, started_at);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
    ON metadata.pipeline_runs (status, repo, commit);

CREATE INDEX IF NOT EXISTS idx_pipeline_steps_run
    ON metadata.pipeline_steps (run_id, module, stage);
"""


def apply_metadata_ddl(con: DuckDBPyConnection) -> None:
    """Create metadata schema, datasets catalog, and helper macros."""
    repo_map_exists_row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'core' AND table_name = 'repo_map'
        LIMIT 1
        """
    ).fetchone()
    repo_map_exists = repo_map_exists_row is not None

    for stmt in METADATA_SCHEMA_DDL:
        if (
            not repo_map_exists
            and "CREATE OR REPLACE MACRO metadata.normalized_symbol_use_edges" in stmt
        ):
            con.execute(
                """
                CREATE OR REPLACE MACRO metadata.normalized_symbol_use_edges(
                    table_key TEXT,
                    row_limit BIGINT := 9223372036854775807,
                    row_offset BIGINT := 0
                ) AS TABLE
                SELECT
                    CAST(NULL AS VARCHAR) AS repo,
                    CAST(NULL AS VARCHAR) AS commit,
                    ds.symbol,
                    ds.def_path,
                    ds.use_path,
                    ds.same_file,
                    ds.same_module,
                    CAST(ds.def_goid_h128 AS BIGINT) AS def_goid_h128,
                    CAST(ds.use_goid_h128 AS BIGINT) AS use_goid_h128
                FROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;
                """
            )
            continue
        con.execute(stmt)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata.dataset_schema_registry (
            table_key TEXT PRIMARY KEY,
            schema_hash TEXT NOT NULL
        )
        """
    )

    con.execute(PIPELINE_RUNS_DDL)
    con.execute(PIPELINE_STEPS_DDL)
    for index_stmt in PIPELINE_INDEXES_DDL.strip().split(";"):
        stripped_stmt = index_stmt.strip()
        if stripped_stmt:
            con.execute(stripped_stmt)


def _canonicalize_ddl(stmt: str) -> str:
    return " ".join(stmt.split())


def _collect_macro_hashes() -> dict[str, str]:
    macro_hashes: dict[str, str] = {}
    for stmt in METADATA_SCHEMA_DDL:
        match = re.search(r"CREATE\s+OR\s+REPLACE\s+MACRO\s+([\w\.]+)", stmt, re.IGNORECASE)
        if match is None:
            continue
        macro_name = match.group(1)
        normalized = _canonicalize_ddl(stmt)
        macro_hashes[macro_name] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return macro_hashes


def _load_existing_macro_hashes(con: DuckDBPyConnection) -> dict[str, tuple[str, str | None]]:
    existing: dict[str, tuple[str, str | None]] = {}
    rows = con.execute(
        "SELECT macro_name, ddl_hash, schema_hash FROM metadata.macro_registry"
    ).fetchall()
    for name, ddl_hash, schema_hash in rows:
        existing[str(name)] = (str(ddl_hash), str(schema_hash) if schema_hash is not None else None)
    return existing


def _detect_macro_drift(
    macro_hashes: Mapping[str, str],
    schema_hashes: Mapping[str, str],
    existing: Mapping[str, tuple[str, str | None]],
) -> tuple[list[str], list[str]]:
    mismatched: list[str] = []
    schema_mismatched: list[str] = []
    for name, expected_hash in macro_hashes.items():
        if name in existing and existing[name][0] != expected_hash:
            mismatched.append(name)
    for name, expected_schema_hash in schema_hashes.items():
        if name in existing:
            existing_schema_hash = existing[name][1]
            if existing_schema_hash is not None and existing_schema_hash != expected_schema_hash:
                schema_mismatched.append(name)
    return mismatched, schema_mismatched


def _upsert_macro_registry(
    con: DuckDBPyConnection,
    macro_hashes: Mapping[str, str],
    schema_hashes: Mapping[str, str],
    macro_to_dataset: Mapping[str, str],
) -> None:
    for name, ddl_hash in macro_hashes.items():
        con.execute(
            """
            INSERT INTO metadata.macro_registry (macro_name, dataset_table_key, ddl_hash, schema_hash)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(macro_name) DO UPDATE SET
                dataset_table_key = excluded.dataset_table_key,
                ddl_hash = excluded.ddl_hash,
                schema_hash = excluded.schema_hash;
            """,
            [name, macro_to_dataset.get(name), ddl_hash, schema_hashes.get(name)],
        )


def _register_macros(con: DuckDBPyConnection) -> None:
    macro_hashes = _collect_macro_hashes()
    macro_to_dataset = {v: k for k, v in NORMALIZED_MACROS.items()}
    macro_to_dataset.update({v: k for k, v in INGEST_MACROS.items()})
    schema_hashes = {
        macro: _expected_schema_hash(dataset) for macro, dataset in macro_to_dataset.items()
    }
    existing = _load_existing_macro_hashes(con)
    mismatched, schema_mismatched = _detect_macro_drift(macro_hashes, schema_hashes, existing)
    if mismatched or schema_mismatched:
        parts = []
        if mismatched:
            parts.append(f"DDL drift: {', '.join(sorted(mismatched))}")
        if schema_mismatched:
            parts.append(f"Schema drift: {', '.join(sorted(schema_mismatched))}")
        raise RuntimeError("Macro registry hash drift detected: " + "; ".join(parts))
    _upsert_macro_registry(con, macro_hashes, schema_hashes, macro_to_dataset)


def validate_macro_registry(con: DuckDBPyConnection) -> None:
    """
    Validate macro registry hashes without mutating the database.

    Raises
    ------
    RuntimeError
        When DDL or schema hashes drift from expected values.
    """
    macro_hashes = _collect_macro_hashes()
    macro_to_dataset = {v: k for k, v in NORMALIZED_MACROS.items()}
    macro_to_dataset.update({v: k for k, v in INGEST_MACROS.items()})
    schema_hashes = {
        macro: _expected_schema_hash(dataset) for macro, dataset in macro_to_dataset.items()
    }

    existing = _load_existing_macro_hashes(con)

    mismatched, schema_mismatched = _detect_macro_drift(macro_hashes, schema_hashes, existing)

    missing = [name for name in macro_hashes if name not in existing]
    if mismatched or schema_mismatched or missing:
        parts = []
        if mismatched:
            parts.append(f"DDL drift: {', '.join(sorted(mismatched))}")
        if schema_mismatched:
            parts.append(f"Schema drift: {', '.join(sorted(schema_mismatched))}")
        if missing:
            parts.append(f"Missing registry rows: {', '.join(sorted(missing))}")
        raise RuntimeError("Macro registry validation failed: " + "; ".join(parts))


def ingest_macro_coverage(con: DuckDBPyConnection) -> tuple[list[str], list[str]]:
    """
    Return missing and present ingest macros for visibility/logging.

    Returns
    -------
    tuple[list[str], list[str]]
        (missing_macro_names, present_macro_names)
    """
    rows = con.execute(
        """
        SELECT CONCAT_WS('.', schema_name, function_name)
        FROM duckdb_functions()
        WHERE function_type IN ('macro', 'table_macro')
        """
    ).fetchall()
    available = {str(row[0]).lower() for row in rows}
    missing: list[str] = []
    present: list[str] = []
    for macro in sorted(INGEST_MACROS.values()):
        if macro.lower() in available or macro.split(".")[-1].lower() in available:
            present.append(macro)
        else:
            missing.append(macro)
    return missing, present


def validate_dataset_schema_registry(con: DuckDBPyConnection) -> None:
    """
    Validate dataset schema hashes for both macro-backed and dataset_rows-only tables.

    Raises
    ------
    RuntimeError
        When schema hashes drift or registry entries are missing.
    """
    expected = {
        table_key: _expected_schema_hash(table_key)
        for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
        if contract.schema is not None
    }
    rows = con.execute(
        "SELECT table_key, schema_hash FROM metadata.dataset_schema_registry"
    ).fetchall()
    actual = {str(table_key): str(schema_hash) for table_key, schema_hash in rows}

    missing = sorted(set(expected) - set(actual))
    mismatched = sorted(
        table_key for table_key, hash_val in expected.items() if actual.get(table_key) != hash_val
    )
    if missing or mismatched:
        parts = []
        if missing:
            parts.append(f"Missing dataset schema registry entries: {', '.join(missing)}")
        if mismatched:
            parts.append(f"Dataset schema drift: {', '.join(mismatched)}")
        raise RuntimeError("; ".join(parts))


def _macro_schema_differences(con: DuckDBPyConnection) -> list[str]:
    failures: list[str] = []
    for table_key, macro in sorted(NORMALIZED_MACROS.items()):
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None or contract.schema is None:
            continue
        schema = contract.schema
        sql, params = safe_macro_call(macro, [table_key, 0, 0])
        rel = con.sql(sql, params=params)
        actual: dict[str, str] = {}
        for name, dtype in zip(rel.columns, rel.dtypes, strict=False):
            if name.endswith("_1"):
                continue
            actual[name] = _canonical_type(str(dtype))

        expected = {col.name: _canonical_type(col.type) for col in schema.columns}

        missing = expected.keys() - actual.keys()
        if missing:
            failures.append(f"{table_key}: missing columns {sorted(missing)}")
            continue
        for col_name, expected_type in expected.items():
            actual_type = actual[col_name]
            if expected_type in {"TIMESTAMP", "DATE"} and actual_type == "VARCHAR":
                continue
            if actual_type != expected_type:
                failures.append(f"{table_key}.{col_name}: {actual_type} != {expected_type}")
    return failures


def validate_normalized_macro_schemas(con: DuckDBPyConnection) -> None:
    """
    Validate normalized macros against TABLE_SCHEMAS and fail fast on drift.

    Raises
    ------
    RuntimeError
        When column names or types diverge from TABLE_SCHEMAS expectations.
    """
    failures = _macro_schema_differences(con)
    if failures:
        message = "Normalized macro schema drift: " + "; ".join(failures)
        raise RuntimeError(message)


@dataclass(frozen=True)
class _DatasetUpsert:
    table_key: str
    name: str
    is_view: bool
    jsonl_filename: str | None
    parquet_filename: str | None
    family: str | None
    description: str | None
    schema_version: str | None
    deprecated: bool


def _upsert_dataset_row(con: DuckDBPyConnection, payload: _DatasetUpsert) -> None:
    jsonl_filename = payload.jsonl_filename
    parquet_filename = payload.parquet_filename
    con.execute(
        """
        INSERT INTO metadata.datasets (
            table_key,
            name,
            is_view,
            jsonl_filename,
            parquet_filename,
            family,
            description,
            schema_version,
            deprecated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(table_key) DO UPDATE SET
            name             = excluded.name,
            is_view          = excluded.is_view,
            jsonl_filename   = excluded.jsonl_filename,
            parquet_filename = excluded.parquet_filename,
            family           = excluded.family,
            description      = excluded.description,
            schema_version   = excluded.schema_version,
            deprecated       = excluded.deprecated;
        """,
        [
            payload.table_key,
            payload.name,
            payload.is_view,
            jsonl_filename,
            parquet_filename,
            payload.family,
            payload.description,
            payload.schema_version,
            payload.deprecated,
        ],
    )


def sync_dataset_dataflow_graph(con: DuckDBPyConnection) -> None:
    """Refresh dataset-level dataflow graph metadata tables based on static contracts."""
    nodes, edges = build_contract_dataflow_graph()

    con.execute("DELETE FROM metadata.dataset_dataflow_nodes")
    con.execute("DELETE FROM metadata.dataset_dataflow_edges")

    if nodes:
        con.executemany(
            """
            INSERT INTO metadata.dataset_dataflow_nodes (
                id,
                kind,
                family,
                owner_package,
                description
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (node.id, node.kind, node.family, node.owner_package, node.description)
                for node in nodes
            ],
        )

    if edges:
        con.executemany(
            """
            INSERT INTO metadata.dataset_dataflow_edges (
                src,
                dst,
                edge_type
            )
            VALUES (?, ?, ?)
            """,
            [(edge.src, edge.dst, edge.edge_type) for edge in edges],
        )


def bootstrap_metadata_datasets(
    con: DuckDBPyConnection,
    *,
    jsonl_filenames: Mapping[str, str] | None = None,
    parquet_filenames: Mapping[str, str] | None = None,
    include_views: bool = True,
    validate_macros: bool = True,
) -> None:
    """
    Populate metadata.datasets from Python schemas and default filename mappings.

    Safe to run repeatedly; uses idempotent upserts to refresh filenames and view flags.
    """
    _assert_macro_coverage()
    apply_metadata_ddl(con)
    _register_macros(con)
    _register_dataset_schema_hashes(con)
    if validate_macros:
        validate_normalized_macro_schemas(con)
    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        warning_message = "dataset_rows-only datasets (no normalized macro): " + ", ".join(
            dataset_rows_list
        )

        con.execute("SELECT ? AS warning", [warning_message])

    jsonl_mapping = dict(jsonl_filenames or {})
    parquet_mapping = dict(parquet_filenames or {})

    for name, contract in sorted(DATASET_CONTRACTS.items(), key=lambda item: item[1].table_key):
        if contract.is_view and not include_views:
            continue

        table_key = contract.table_key
        schema_prefix, _ = table_key.split(".", maxsplit=1)
        jsonl_filename = jsonl_mapping.get(table_key) or contract.jsonl_filename
        parquet_filename = parquet_mapping.get(table_key) or contract.parquet_filename

        _upsert_dataset_row(
            con,
            _DatasetUpsert(
                table_key=table_key,
                name=name,
                is_view=contract.is_view,
                jsonl_filename=jsonl_filename,
                parquet_filename=parquet_filename,
                family=contract.family or schema_prefix,
                description=contract.description,
                schema_version=contract.schema_version,
                deprecated=contract.deprecated,
            ),
        )

    sync_dataset_dataflow_graph(con)
