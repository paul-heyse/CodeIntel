"""Build GOIDs and crosswalk entries from AST nodes."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from codeintel.config.models import GoidBuilderConfig
from codeintel.graphs.function_catalog import load_function_catalog
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import GoidCrosswalkRow, GoidRow, goid_crosswalk_to_tuple, goid_to_tuple
from codeintel.utils.paths import relpath_to_module

log = logging.getLogger(__name__)
DECIMAL_38_MAX = 10**38 - 1


@dataclass(frozen=True)
class GoidDescriptor:
    """Descriptor for a single code entity."""

    repo: str
    commit: str
    language: str
    rel_path: str
    kind: str
    qualname: str
    start_line: int
    end_line: int | None


def _relpath_to_module(path: str | Path) -> str:
    """
    Convert a repository-relative path to a dotted module path.

    Parameters
    ----------
    path:
        Repository-relative path to a Python file.

    Returns
    -------
    str
        Dotted module path derived from the relative path.
    """
    return relpath_to_module(path)


def _safe_int(value: object, default: int | None = None) -> int | None:
    """
    Convert optional values to int, treating pandas nulls as missing.

    Returns
    -------
    int | None
        Integer value when conversion succeeds; otherwise the provided default.
    """
    if value is None or isinstance(value, (pd.Series, pd.DataFrame)):
        return default

    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        return default

    if not isinstance(value, (int, float, str, bool)):
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compute_goid(descriptor: GoidDescriptor) -> int:
    """
    Compute a stable 128-bit GOID integer from an entity descriptor.

    Parameters
    ----------
    descriptor:
        Metadata describing the code entity.

    Returns
    -------
    int
        Stable 128-bit integer representation of the GOID.
    """
    payload = (
        f"{descriptor.repo}:{descriptor.commit}:{descriptor.language}:"
        f"{descriptor.rel_path}:{descriptor.kind}:{descriptor.qualname}:"
        f"{descriptor.start_line}:{descriptor.end_line}"
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big") % DECIMAL_38_MAX


def _build_urn(descriptor: GoidDescriptor) -> str:
    """
    Build a GOID URN that matches the README format.

    Parameters
    ----------
    descriptor:
        Metadata describing the code entity.

    Returns
    -------
    str
        GOID URN encoding repo, path, kind, and span.
    """
    base = (
        f"goid:{descriptor.repo}/{descriptor.rel_path}"
        f"#{descriptor.language}:{descriptor.kind}:{descriptor.qualname}"
    )
    if descriptor.end_line is None:
        return f"{base}?s={descriptor.start_line}"
    return f"{base}?s={descriptor.start_line}&e={descriptor.end_line}"


def _build_goid_entries(
    row: pd.Series, cfg: GoidBuilderConfig, now: datetime, module_by_path: dict[str, str]
) -> tuple[GoidRow, GoidCrosswalkRow]:
    rel_path = str(row["path"]).replace("\\", "/")
    node_type = str(row["node_type"])
    qualname = str(row["qualname"])
    parent_qualname_raw = row["parent_qualname"]
    parent_qualname = str(parent_qualname_raw) if parent_qualname_raw is not None else None

    if node_type == "Module":
        kind = "module"
    elif node_type == "ClassDef":
        kind = "class"
    elif parent_qualname and parent_qualname != _relpath_to_module(Path(rel_path)):
        kind = "method"
    else:
        kind = "function"

    start_line = _safe_int(row["lineno"], default=1) or 1
    decorator_start = _safe_int(row.get("decorator_start_line"))
    if decorator_start is not None and decorator_start > 0:
        start_line = min(start_line, decorator_start)
    end_line = _safe_int(row["end_lineno"])

    descriptor = GoidDescriptor(
        repo=cfg.repo,
        commit=cfg.commit,
        language=cfg.language,
        rel_path=rel_path,
        kind=kind,
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
    )

    goid_h128 = _compute_goid(descriptor)
    urn = _build_urn(descriptor)

    goid_row = GoidRow(
        goid_h128=goid_h128,
        urn=urn,
        repo=cfg.repo,
        commit=cfg.commit,
        rel_path=rel_path,
        language=cfg.language,
        kind=kind,
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
        created_at=now,
    )

    module_name = module_by_path.get(rel_path, _relpath_to_module(Path(rel_path)))
    xwalk_row = GoidCrosswalkRow(
        repo=cfg.repo,
        commit=cfg.commit,
        goid=urn,
        lang=cfg.language,
        module_path=module_name,
        file_path=rel_path,
        start_line=start_line,
        end_line=end_line,
        scip_symbol=None,
        ast_qualname=qualname,
        cst_node_id=None,
        chunk_id=None,
        symbol_id=None,
        updated_at=now,
    )

    return goid_row, xwalk_row


def build_goids(con: duckdb.DuckDBPyConnection, cfg: GoidBuilderConfig) -> None:
    """
    Populate core.goids and core.goid_crosswalk from core.ast_nodes.

    Parameters
    ----------
    con:
        DuckDB connection to populate.
    cfg:
        Configuration identifying the repo and commit to hash.
    """
    df = con.execute(
        """
        SELECT
            an.path,
            an.node_type,
            an.name,
            an.qualname,
            an.lineno,
            an.end_lineno,
            an.parent_qualname
        FROM core.ast_nodes an
        JOIN core.modules m
          ON m.path = an.path
        WHERE m.repo = ? AND m.commit = ?
          AND an.node_type IN ('Module', 'ClassDef', 'FunctionDef', 'AsyncFunctionDef')
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()

    if df.empty:
        log.warning("No AST nodes found in core.ast_nodes; cannot build GOIDs.")
        return

    goid_rows: list[GoidRow] = []
    xwalk_rows: list[GoidCrosswalkRow] = []

    now = datetime.now(UTC)

    module_by_path = load_function_catalog(con, repo=cfg.repo, commit=cfg.commit).module_by_path

    for _, row in df.iterrows():
        goid_row, xwalk_row = _build_goid_entries(row, cfg, now, module_by_path)
        goid_rows.append(goid_row)
        xwalk_rows.append(xwalk_row)

    run_batch(
        con,
        "core.goids",
        [goid_to_tuple(row) for row in goid_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    run_batch(
        con,
        "core.goid_crosswalk",
        [goid_crosswalk_to_tuple(row) for row in xwalk_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "GOID build complete for repo=%s commit=%s: %d entities",
        cfg.repo,
        cfg.commit,
        len(goid_rows),
    )
