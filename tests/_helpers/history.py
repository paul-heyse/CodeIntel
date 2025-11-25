"""Helpers for history and timeseries testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from codeintel.config.schemas.tables import TABLE_SCHEMAS
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
    con = duckdb.connect(str(db_path))
    apply_all_schemas(con)
    fp_columns = TABLE_SCHEMAS["analytics.function_profile"].column_names()
    mp_columns = TABLE_SCHEMAS["analytics.module_profile"].column_names()
    fp_cols = ", ".join(fp_columns)
    mp_cols = ", ".join(mp_columns)
    fp_placeholders = ", ".join("?" for _ in fp_columns)
    mp_placeholders = ", ".join("?" for _ in mp_columns)
    con.execute(
        "INSERT INTO analytics.function_profile (" + fp_cols + ") VALUES (" + fp_placeholders + ")",
        _function_profile_row(spec),
    )
    con.execute(
        "INSERT INTO analytics.module_profile (" + mp_cols + ") VALUES (" + mp_placeholders + ")",
        _module_profile_row(spec),
    )
    con.close()
    return db_path


def insert_function_history_row(
    con: duckdb.DuckDBPyConnection,
    spec: SnapshotSpec,
) -> None:
    """Insert a minimal function_history row for validation helpers."""
    columns = TABLE_SCHEMAS["analytics.function_history"].column_names()
    defaults: dict[str, object | None] = dict.fromkeys(columns, None)
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
    con.execute(
        f"INSERT INTO analytics.function_history ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        [tuple(defaults[col] for col in columns)],
    )
