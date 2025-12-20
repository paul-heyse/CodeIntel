"""Orchestration functions for history and timeseries testing."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.build.schemas import iter_contracts_by_table_key
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.schema import apply_all_schemas

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.configs.history_config import SnapshotSpec

_FUNCTION_PROFILE_TABLE_KEY = "analytics.function_profile"
_MODULE_PROFILE_TABLE_KEY = "analytics.module_profile"
_FUNCTION_HISTORY_TABLE_KEY = "analytics.function_history"


@lru_cache(maxsize=1)
def _contracts_by_table() -> dict[str, DatasetContract]:
    return dict(iter_contracts_by_table_key())


@lru_cache(maxsize=16)
def _columns_for_table_key(table_key: str) -> tuple[str, ...]:
    contract = _contracts_by_table().get(table_key)
    schema = getattr(contract, "schema", None)
    if schema is None:
        return ()
    return tuple(schema.column_names())


def _require_columns(table_key: str) -> tuple[str, ...]:
    columns = _columns_for_table_key(table_key)
    if not columns:
        msg = f"Missing schema columns for {table_key}"
        raise ValueError(msg)
    return columns


def _function_profile_row(spec: SnapshotSpec) -> tuple[object, ...]:
    """Build a function profile row from a snapshot spec.

    Returns
    -------
    tuple[object, ...]
        Row tuple for function_profile table.
    """
    columns = _require_columns(_FUNCTION_PROFILE_TABLE_KEY)
    defaults: dict[str, object | None] = dict.fromkeys(columns, None)
    defaults.update(
        {
            "function_goid_h128": spec.goid,
            "urn": f"goid:{spec.repo}/{spec.rel_path}#{spec.qualname}",
            "repo": spec.repo,
            "commit": spec.commit,
            "rel_path": spec.rel_path,
            "module": spec.module,
            "language": "python",
            "kind": "function",
            "qualname": spec.qualname,
            "start_line": 1,
            "end_line": spec.loc,
            "loc": spec.loc,
            "cyclomatic_complexity": spec.cyclomatic_complexity,
            "coverage_ratio": spec.coverage_ratio,
            "risk_score": spec.risk_score,
            "risk_level": spec.risk_level,
        }
    )
    return tuple(defaults[col] for col in columns)


def _module_profile_row(spec: SnapshotSpec) -> tuple[object, ...]:
    """Build a module profile row from a snapshot spec.

    Returns
    -------
    tuple[object, ...]
        Row tuple for module_profile table.
    """
    columns = _require_columns(_MODULE_PROFILE_TABLE_KEY)
    defaults: dict[str, object | None] = dict.fromkeys(columns, None)
    defaults.update(
        {
            "repo": spec.repo,
            "commit": spec.commit,
            "module": spec.module,
            "path": spec.rel_path,
            "language": "python",
            "file_count": 1,
            "total_loc": spec.loc,
            "module_coverage_ratio": spec.coverage_ratio,
            "max_risk_score": spec.risk_score,
            "avg_risk_score": spec.risk_score,
        }
    )
    return tuple(defaults[col] for col in columns)


def create_snapshot_db(base_dir: Path, spec: SnapshotSpec) -> Path:
    """
    Create a minimal snapshot DuckDB with function/module profile rows.

    Parameters
    ----------
    base_dir
        Directory to place the database file.
    spec
        Snapshot specification.

    Returns
    -------
    Path
        Path to the created DuckDB file.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / f"codeintel-{spec.commit}.duckdb"
    cfg = StorageConfig.for_ingest(db_path)
    gateway = open_gateway(cfg)
    con = gateway.con
    apply_all_schemas(con)
    fp_columns = _require_columns(_FUNCTION_PROFILE_TABLE_KEY)
    mp_columns = _require_columns(_MODULE_PROFILE_TABLE_KEY)
    fp_df = pd.DataFrame([_function_profile_row(spec)], columns=pd.Index(fp_columns))
    mp_df = pd.DataFrame([_module_profile_row(spec)], columns=pd.Index(mp_columns))
    con.register("fp_df", fp_df)
    con.register("mp_df", mp_df)
    con.execute("INSERT INTO analytics.function_profile BY NAME SELECT * FROM fp_df")
    con.execute("INSERT INTO analytics.module_profile BY NAME SELECT * FROM mp_df")
    gateway.close()
    return db_path


def insert_function_history_row(
    gateway: StorageGateway,
    spec: SnapshotSpec,
) -> None:
    """Insert a minimal function_history row for validation helpers."""
    con = gateway.con
    fh_columns = _require_columns(_FUNCTION_HISTORY_TABLE_KEY)
    defaults: dict[str, object | None] = dict.fromkeys(fh_columns, None)
    now = datetime.now(tz=UTC)
    defaults.update(
        {
            "repo": spec.repo,
            "commit": spec.commit,
            "function_goid_h128": spec.goid,
            "urn": f"goid:{spec.repo}/{spec.rel_path}#{spec.qualname}",
            "rel_path": spec.rel_path,
            "module": spec.module,
            "qualname": spec.qualname,
            "created_in_commit": spec.commit,
            "created_at": now,
            "last_modified_commit": spec.commit,
            "last_modified_at": now,
            "age_days": 0,
            "commit_count": 1,
            "author_count": 1,
            "lines_added": 3,
            "lines_deleted": 0,
            "churn_score": 0.3,
            "stability_bucket": "new_hot",
            "created_at_row": now,
        }
    )
    fh_df = pd.DataFrame(
        [tuple(defaults[col] for col in fh_columns)],
        columns=pd.Index(fh_columns),
    )
    con.register("fh_df", fh_df)
    con.execute("INSERT INTO analytics.function_history BY NAME SELECT * FROM fh_df")


__all__ = ["create_snapshot_db", "insert_function_history_row"]
