"""Build GOIDs and crosswalk entries from AST nodes."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)
DECIMAL_38_MAX = 10**38 - 1


@dataclass(frozen=True)
class GoidBuilderConfig:
    """Configuration for building GOIDs."""

    repo: str
    commit: str
    language: str = "python"


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
    rel_path = Path(path).with_suffix("")
    return ".".join(rel_path.parts)


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
            path,
            node_type,
            name,
            qualname,
            lineno,
            end_lineno,
            parent_qualname
        FROM core.ast_nodes
        WHERE node_type IN ('Module', 'ClassDef', 'FunctionDef', 'AsyncFunctionDef')
        """
    ).fetch_df()

    if df.empty:
        log.warning("No AST nodes found in core.ast_nodes; cannot build GOIDs.")
        return

    con.execute("DELETE FROM core.goids WHERE repo = ? AND commit = ?", [cfg.repo, cfg.commit])
    con.execute("DELETE FROM core.goid_crosswalk")

    goid_rows: list[tuple] = []
    xwalk_rows: list[tuple] = []

    now = datetime.now(UTC)

    for _, row in df.iterrows():
        rel_path = str(row["path"]).replace("\\", "/")
        node_type = str(row["node_type"])
        qualname = str(row["qualname"])
        parent_qualname_raw = row["parent_qualname"]
        parent_qualname = (
            str(parent_qualname_raw) if parent_qualname_raw is not None else None
        )

        if node_type == "Module":
            kind = "module"
        elif node_type == "ClassDef":
            kind = "class"
        elif parent_qualname and parent_qualname != _relpath_to_module(Path(rel_path)):
            kind = "method"
        else:
            kind = "function"

        start_line = int(row["lineno"]) if row["lineno"] is not None else 1
        end_line = int(row["end_lineno"]) if row["end_lineno"] is not None else None

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

        goid_rows.append(
            (
                goid_h128,
                urn,
                cfg.repo,
                cfg.commit,
                rel_path,
                cfg.language,
                kind,
                qualname,
                start_line,
                end_line,
                now,
            )
        )

        xwalk_rows.append(
            (
                urn,
                cfg.language,
                _relpath_to_module(Path(rel_path)),
                rel_path,
                start_line,
                end_line,
                None,
                qualname,
                None,
                None,
                None,
                now,
            )
        )

    if goid_rows:
        con.executemany(
            """
            INSERT INTO core.goids
              (goid_h128, urn, repo, commit, rel_path, language, kind,
               qualname, start_line, end_line, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            goid_rows,
        )

    if xwalk_rows:
        con.executemany(
            """
            INSERT INTO core.goid_crosswalk
              (goid, lang, module_path, file_path, start_line, end_line,
               scip_symbol, ast_qualname, cst_node_id, chunk_id, symbol_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            xwalk_rows,
        )

    log.info(
        "GOID build complete for repo=%s commit=%s: %d entities",
        cfg.repo,
        cfg.commit,
        len(goid_rows),
    )
