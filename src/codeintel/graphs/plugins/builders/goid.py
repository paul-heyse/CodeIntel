"""Global object identifier builder plugin.

This module provides the GOID builder as a build target plugin.

Architecture
------------
The GOID plugin performs the following steps:

1. Parse source files to extract modules, classes, and functions
2. Compute stable GOIDs and URNs for each entity
3. Build GOID rows and crosswalk rows
4. Persist to core.goids and core.goid_crosswalk
"""

from __future__ import annotations

import ast
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import GoidBuilderStepConfig
from codeintel.graphs.compute import goid as goid_compute
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.ingestion.infrastructure.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.graphs.compute.goid import GoidCrosswalkRow, GoidRow
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _get_source_root(gateway: StorageGateway, repo: str, commit: str) -> Path | None:
    """Retrieve source root from core.snapshots.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    Path | None
        Absolute path to source root, or None if not found.
    """
    con = gateway.con
    try:
        row = con.execute(
            "SELECT source_root FROM core.snapshots WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        if row and row[0]:
            return Path(row[0])
    except Exception as e:  # noqa: BLE001
        log.debug("goid_builder: Could not get source root: %s", e)
    return None


def _get_tracked_files(gateway: StorageGateway, repo: str, commit: str) -> list[str]:
    """Get list of tracked Python files from core.modules.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    list[str]
        List of relative paths to Python files.
    """
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT DISTINCT path
            FROM core.modules
            WHERE repo = ? AND commit = ?
            ORDER BY path
            """,
            [repo, commit],
        ).fetchall()
        return [str(row[0]) for row in rows]
    except Exception:  # noqa: BLE001
        return []


def _path_to_module_name(rel_path: str) -> str:
    """Convert relative path to module name.

    Parameters
    ----------
    rel_path
        Relative file path.

    Returns
    -------
    str
        Module name.
    """
    path = Path(rel_path)
    parts = list(path.parts)
    if path.suffix == ".py":
        parts[-1] = path.stem
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _extract_entities_from_file(
    file_path: Path,
    rel_path: str,
    repo: str,
    commit: str,
    now: datetime,
) -> tuple[list[GoidRow], list[GoidCrosswalkRow]]:
    """Extract entities from a Python file and compute GOIDs.

    Parameters
    ----------
    file_path
        Absolute path to the file.
    rel_path
        Relative path to the file.
    repo
        Repository identifier.
    commit
        Commit SHA.
    now
        Timestamp for created_at/updated_at.

    Returns
    -------
    tuple[list[GoidRow], list[GoidCrosswalkRow]]
        GOID rows and crosswalk rows.
    """
    if not file_path.exists():
        return [], []

    try:
        source = file_path.read_text(encoding="utf8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return [], []

    goid_rows: list[GoidRow] = []
    crosswalk_rows: list[GoidCrosswalkRow] = []

    module_name = _path_to_module_name(rel_path)
    normalized_path = normalize_rel_path(rel_path)

    # Process module itself
    module_descriptor = goid_compute.GoidDescriptor(
        repo=repo,
        commit=commit,
        language="python",
        rel_path=normalized_path,
        kind="module",
        qualname=module_name,
        start_line=1,
        end_line=len(source.splitlines()) if source else 1,
    )
    module_result = goid_compute.compute_goid_result(module_descriptor)
    goid_rows.append(
        goid_compute.build_goid_row(
            module_descriptor, module_result.goid_h128, module_result.urn, now
        )
    )
    crosswalk_rows.append(
        goid_compute.build_crosswalk_row(module_descriptor, module_result.urn, module_name, now)
    )

    # Process all nodes
    def process_node(
        node: ast.AST,
        parent_qualname: str | None,
    ) -> None:
        """Process an AST node recursively.

        Parameters
        ----------
        node
            AST node to process.
        parent_qualname
            Qualified name of the parent node.
        """
        name: str | None = None
        start_line: int = 0
        end_line: int | None = None

        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", node.lineno)

        if name is not None:
            qualname = f"{parent_qualname}.{name}" if parent_qualname else f"{module_name}.{name}"
            kind = goid_compute.determine_kind(
                type(node).__name__, parent_qualname, normalized_path, module_name
            )

            descriptor = goid_compute.GoidDescriptor(
                repo=repo,
                commit=commit,
                language="python",
                rel_path=normalized_path,
                kind=kind,
                qualname=qualname,
                start_line=start_line,
                end_line=end_line,
            )
            result = goid_compute.compute_goid_result(descriptor)
            goid_rows.append(
                goid_compute.build_goid_row(descriptor, result.goid_h128, result.urn, now)
            )
            crosswalk_rows.append(
                goid_compute.build_crosswalk_row(descriptor, result.urn, module_name, now)
            )

            # Process children with updated parent
            for child in ast.iter_child_nodes(node):
                process_node(child, qualname)
        else:
            # Process children with same parent
            for child in ast.iter_child_nodes(node):
                process_node(child, parent_qualname)

    # Start processing from module level
    for child in ast.iter_child_nodes(tree):
        process_node(child, module_name)

    return goid_rows, crosswalk_rows


def _persist_goid_rows(
    gateway: StorageGateway,
    rows: list[GoidRow],
    repo: str,
    commit: str,
) -> int:
    """Persist GOID rows.

    Parameters
    ----------
    gateway
        Storage gateway.
    rows
        Rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of rows persisted.
    """
    if not rows:
        return 0

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "core.goids",
        [row.to_tuple() for row in rows],
        delete_params=[repo, commit],
        scope="goids",
    )
    return len(rows)


def _persist_crosswalk_rows(
    gateway: StorageGateway,
    rows: list[GoidCrosswalkRow],
    repo: str,
    commit: str,
) -> int:
    """Persist crosswalk rows.

    Parameters
    ----------
    gateway
        Storage gateway.
    rows
        Rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of rows persisted.
    """
    if not rows:
        return 0

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "core.goid_crosswalk",
        [row.to_tuple() for row in rows],
        delete_params=[repo, commit],
        scope="goid_crosswalk",
    )
    return len(rows)


class GoidBuilderPlugin(TargetPlugin):
    """Build global object identifiers.

    This plugin performs full GOID construction:
    1. Parses source files to extract entities
    2. Computes stable GOIDs and URNs
    3. Persists to core.goids and core.goid_crosswalk

    Outputs
    -------
    - core.goids: GOID records
    - core.goid_crosswalk: GOID crosswalk records
    """

    plugin_name: ClassVar[str] = "goid_builder"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build global object identifiers."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute GOID construction.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result with row counts.
        """
        _ = self  # Protocol method requires instance

        # Build config
        cfg = GoidBuilderStepConfig(snapshot=ctx.snapshot)

        gateway = ctx.gateway
        repo = cfg.repo
        commit = cfg.commit

        try:
            # Step 1: Get source root (prefer snapshot, then db, then cwd)
            source_root = ctx.snapshot.repo_root
            if not source_root:
                source_root = _get_source_root(gateway, repo, commit)
            if not source_root:
                source_root = Path.cwd()
                log.warning("goid_builder: No source root found, using current directory")

            # Step 2: Get tracked files
            tracked_files = _get_tracked_files(gateway, repo, commit)

            if not tracked_files:
                log.info("goid_builder: No tracked files found, skipping")
                return TargetResult.succeeded(
                    row_counts={
                        "core.goids": 0,
                        "core.goid_crosswalk": 0,
                    }
                )

            # Step 3: Process each file
            now = datetime.now(UTC)
            all_goid_rows: list[GoidRow] = []
            all_crosswalk_rows: list[GoidCrosswalkRow] = []

            for rel_path in tracked_files:
                file_path = source_root / rel_path
                goid_rows, crosswalk_rows = _extract_entities_from_file(
                    file_path, rel_path, repo, commit, now
                )
                all_goid_rows.extend(goid_rows)
                all_crosswalk_rows.extend(crosswalk_rows)

            log.info(
                "goid_builder: Extracted %d GOIDs and %d crosswalk entries from %d files",
                len(all_goid_rows),
                len(all_crosswalk_rows),
                len(tracked_files),
            )

            # Step 4: Persist
            goid_count = _persist_goid_rows(gateway, all_goid_rows, repo, commit)
            crosswalk_count = _persist_crosswalk_rows(gateway, all_crosswalk_rows, repo, commit)

            log.info(
                "goid_builder: Persisted %d GOIDs and %d crosswalk entries",
                goid_count,
                crosswalk_count,
            )

            return TargetResult.succeeded(
                row_counts={
                    "core.goids": goid_count,
                    "core.goid_crosswalk": crosswalk_count,
                }
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"GOID build failed: {e}")


__all__ = ["GoidBuilderPlugin"]
