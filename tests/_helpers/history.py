"""Helpers for history and timeseries testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.schemas import apply_all_schemas


@dataclass(frozen=True)
class SnapshotSpec:
    """Specification for a minimal function snapshot."""

    repo: str
    commit: str
    goid: int
    rel_path: str
    module: str
    qualname: str
    risk_score: float = 0.5
    coverage_ratio: float = 0.5
    risk_level: str = "medium"
    cyclomatic_complexity: int = 1
    loc: int = 10


_FP_COLUMNS = TABLE_SCHEMAS["analytics.function_profile"].column_names()
_MP_COLUMNS = TABLE_SCHEMAS["analytics.module_profile"].column_names()
_FUNCTION_HISTORY_COLUMNS = TABLE_SCHEMAS["analytics.function_history"].column_names()


def _function_profile_row(spec: SnapshotSpec) -> tuple[object, ...]:
    columns = TABLE_SCHEMAS["analytics.function_profile"].column_names()
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
    columns = TABLE_SCHEMAS["analytics.module_profile"].column_names()
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
    fp_df = pd.DataFrame([_function_profile_row(spec)], columns=pd.Index(_FP_COLUMNS))
    mp_df = pd.DataFrame([_module_profile_row(spec)], columns=pd.Index(_MP_COLUMNS))
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
    defaults: dict[str, object | None] = dict.fromkeys(_FUNCTION_HISTORY_COLUMNS, None)
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
        [tuple(defaults[col] for col in _FUNCTION_HISTORY_COLUMNS)],
        columns=pd.Index(_FUNCTION_HISTORY_COLUMNS),
    )
    con.register("fh_df", fh_df)
    con.execute("INSERT INTO analytics.function_history BY NAME SELECT * FROM fh_df")
