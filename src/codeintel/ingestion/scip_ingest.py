"""Run SCIP indexing and register outputs for downstream analytics."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from asyncio.subprocess import PIPE
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb

from codeintel.config.models import ScipIngestConfig

log = logging.getLogger(__name__)
MISSING_BINARY_EXIT_CODE = 127


@dataclass(frozen=True)
class ScipIngestResult:
    """Outcome of SCIP ingestion."""

    status: Literal["success", "unavailable", "failed"]
    index_scip: Path | None
    index_json: Path | None
    reason: str | None = None


def ingest_scip(con: duckdb.DuckDBPyConnection, cfg: ScipIngestConfig) -> ScipIngestResult:
    """
    Run scip-python + scip print, register view, and backfill SCIP symbols.

    Returns
    -------
    ScipIngestResult
        Status and artifact paths. Unavailable/failed states do not raise.
    """
    repo_root = cfg.repo_root.resolve()
    if not (repo_root / ".git").is_dir():
        reason = "SCIP ingestion requires a git repository (.git missing)"
        log.warning(reason)
        return ScipIngestResult(status="unavailable", index_scip=None, index_json=None, reason=reason)

    probe_result = _probe_binaries(cfg)
    if probe_result is not None:
        return probe_result

    scip_dir = cfg.build_dir.resolve() / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = cfg.document_output_dir.resolve()
    doc_dir.mkdir(parents=True, exist_ok=True)

    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.scip.json"

    if not _run_scip_python(cfg.scip_python_bin, repo_root, index_scip):
        message = f"SCIP indexing failed for {cfg.repo}@{cfg.commit}"
        log.warning(message)
        return ScipIngestResult(status="failed", index_scip=None, index_json=None, reason=message)
    if not _run_scip_print(cfg.scip_bin, index_scip, index_json):
        message = f"SCIP JSON export failed for {cfg.repo}@{cfg.commit}"
        log.warning(message)
        return ScipIngestResult(status="failed", index_scip=index_scip, index_json=None, reason=message)

    shutil.copy2(index_scip, doc_dir / "index.scip")
    shutil.copy2(index_json, doc_dir / "index.scip.json")

    con.execute(
        """
        CREATE OR REPLACE VIEW scip_index_view AS
        SELECT * FROM read_json(?);
        """,
        [str(index_json)],
    )

    _update_scip_symbols(con, index_json)
    log.info("SCIP index ingested for %s@%s", cfg.repo, cfg.commit)
    return ScipIngestResult(status="success", index_scip=index_scip, index_json=index_json)


def _probe_binaries(cfg: ScipIngestConfig) -> ScipIngestResult | None:
    """
    Ensure SCIP binaries exist and respond to --version.

    Returns
    -------
    ScipIngestResult | None
        Unavailable result when probes fail; otherwise None.
    Returns None when binaries look usable; otherwise an unavailable result.
    """
    missing: list[str] = []
    missing = [binary for binary in (cfg.scip_python_bin, cfg.scip_bin) if shutil.which(binary) is None]
    if missing:
        reason = f"Missing SCIP binaries: {', '.join(missing)}"
        log.warning(reason)
        return ScipIngestResult(status="unavailable", index_scip=None, index_json=None, reason=reason)

    for binary in (cfg.scip_python_bin, cfg.scip_bin):
        code, _, stderr = _run_command([binary, "--version"], cwd=cfg.repo_root)
        if code != 0:
            reason = f"{binary} --version failed (code {code}): {stderr.strip()}"
            log.warning(reason)
            return ScipIngestResult(status="unavailable", index_scip=None, index_json=None, reason=reason)
    return None


def _run_scip_python(binary: str, repo_root: Path, output_path: Path) -> bool:
    code, stdout, stderr = _run_command(
        [binary, "index", str(repo_root), "--output", str(output_path)],
        cwd=repo_root,
    )
    if code == 0:
        if stderr:
            log.debug("scip-python stderr: %s", stderr.strip())
        return True
    if code == MISSING_BINARY_EXIT_CODE:
        log.warning("scip-python binary %r not found; skipping SCIP indexing", binary)
        return False
    log.warning("scip-python index failed (code %s): %s", code, stderr.strip() or stdout.strip())
    return False


def _run_scip_print(binary: str, index_scip: Path, output_json: Path) -> bool:
    code, stdout, stderr = _run_command(
        [binary, "print", "--format", "json", str(index_scip)],
        cwd=index_scip.parent,
    )
    if code == 0:
        output_json.write_text(stdout or "", encoding="utf8")
        if stderr:
            log.debug("scip print stderr: %s", stderr.strip())
        return True
    if code == MISSING_BINARY_EXIT_CODE:
        log.warning("scip binary %r not found; skipping SCIP JSON export", binary)
        return False
    log.warning("scip print failed (code %s): %s", code, stderr.strip() or stdout.strip())
    return False


def _update_scip_symbols(con: duckdb.DuckDBPyConnection, index_json: Path) -> None:
    """Populate core.goid_crosswalk.scip_symbol by matching SCIP definitions to GOIDs."""
    try:
        docs = json.loads(index_json.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to parse %s for SCIP symbols: %s", index_json, exc)
        return

    def_map: dict[tuple[str, int], str] = {}
    for doc in docs:
        rel_path = doc.get("relative_path")
        for occ in doc.get("occurrences", []):
            roles = occ.get("symbol_roles", 0)
            if roles & 1 == 0:
                continue
            rng = occ.get("range") or []
            if rng:
                start_line = int(rng[0][0]) + 1
                def_map[rel_path, start_line] = occ.get("symbol")

    if not def_map:
        return

    goids = con.execute(
        """
        SELECT urn, rel_path, start_line
        FROM core.goids
        """
    ).fetchall()

    updates: list[tuple[str, str]] = []
    for urn, rel_path, start_line in goids:
        symbol = def_map.get((rel_path, int(start_line)))
        if symbol:
            updates.append((symbol, urn))

    if updates:
        con.executemany(
            """
            UPDATE core.goid_crosswalk
            SET scip_symbol = ?
            WHERE goid = ?
            """,
            updates,
        )
        log.info("Updated SCIP symbols for %d GOIDs", len(updates))


def _run_command(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    async def _exec() -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd) if cwd is not None else None,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return (
            proc.returncode if proc.returncode is not None else 1,
            stdout_b.decode(),
            stderr_b.decode(),
        )

    try:
        return asyncio.run(_exec())
    except FileNotFoundError as exc:
        return MISSING_BINARY_EXIT_CODE, "", str(exc)
