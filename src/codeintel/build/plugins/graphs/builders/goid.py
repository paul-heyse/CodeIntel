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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.build.plugins._helpers import filter_paths, get_source_root
from codeintel.build.plugins.graphs.builders.goid_options import GoidBuilderOptions
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.graphs.compute import goid as goid_compute
from codeintel.ingestion.infrastructure.paths import normalize_rel_path
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.graphs.compute.goid import GoidCrosswalkRow, GoidRow
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


GOID_BUILDER_METADATA = CorePluginMetadata(
    name="graphs.goid_builder",
    version="3.0.0",
    description="Build global object identifiers.",
    domain=PluginDomain.GRAPH,
    kind="builder",
    stage="goid",
    provides=("core.goids", "core.goid_crosswalk"),
    requires=("core.modules",),
    produces_tables=("core.goids", "core.goid_crosswalk"),
    consumes_tables=("core.modules",),
    supports_incremental=False,
    scope_aware=True,
    options_model=GoidBuilderOptions,
    extra={"graph_kinds": ("goid",)},
)


@dataclass(frozen=True)
class GoidExtractionContext:
    """Context for GOID extraction."""

    repo: str
    commit: str
    now: datetime
    options: GoidBuilderOptions
    module_name: str
    normalized_path: str


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
    try:
        modules = gateway.ibis.table("core.modules")
        expr = (
            modules.filter(
                cast("Any", modules.repo == repo) & cast("Any", modules.commit == commit)
            )
            .select(modules.path)
            .distinct()
            .order_by(modules.path)
        )
        df = expr.execute()
        return [str(path) for (path,) in df.itertuples(index=False, name=None)]
    except DuckDBError:
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


def _process_ast_node(
    node: ast.AST,
    parent_qualname: str | None,
    *,
    context: GoidExtractionContext,
    goid_rows: list[GoidRow],
    crosswalk_rows: list[GoidCrosswalkRow],
) -> None:
    """Process an AST node recursively."""
    options = context.options
    name: str | None = None
    start_line: int = 0
    end_line: int | None = None

    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        name = node.name
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

    if name is not None:
        module_name = context.module_name
        if not options.include_private and name.startswith("_") and name != "__init__":
            for child in ast.iter_child_nodes(node):
                _process_ast_node(
                    child,
                    parent_qualname,
                    context=context,
                    goid_rows=goid_rows,
                    crosswalk_rows=crosswalk_rows,
                )
            return
        qualname = f"{parent_qualname}.{name}" if parent_qualname else f"{module_name}.{name}"
        kind = goid_compute.determine_kind(
            type(node).__name__, parent_qualname, context.normalized_path, module_name
        )

        descriptor = goid_compute.GoidDescriptor(
            repo=context.repo,
            commit=context.commit,
            language="python",
            rel_path=context.normalized_path,
            kind=kind,
            qualname=qualname,
            start_line=start_line,
            end_line=end_line,
        )
        result = goid_compute.compute_goid_result(descriptor)
        goid_rows.append(
            goid_compute.build_goid_row(descriptor, result.goid_h128, result.urn, context.now)
        )
        crosswalk_rows.append(
            goid_compute.build_crosswalk_row(descriptor, result.urn, module_name, context.now)
        )

        for child in ast.iter_child_nodes(node):
            _process_ast_node(
                child,
                qualname,
                context=context,
                goid_rows=goid_rows,
                crosswalk_rows=crosswalk_rows,
            )
    else:
        for child in ast.iter_child_nodes(node):
            _process_ast_node(
                child,
                parent_qualname,
                context=context,
                goid_rows=goid_rows,
                crosswalk_rows=crosswalk_rows,
            )


def _extract_entities_from_file(
    file_path: Path,
    context: GoidExtractionContext,
) -> tuple[list[GoidRow], list[GoidCrosswalkRow]]:
    """Extract entities from a Python file and compute GOIDs.

    Parameters
    ----------
    file_path
        Absolute path to the file.
    context
        Extraction context with repo, commit, module metadata, and options.

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

    module_name = context.module_name
    normalized_path = context.normalized_path

    module_descriptor = goid_compute.GoidDescriptor(
        repo=context.repo,
        commit=context.commit,
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
            module_descriptor, module_result.goid_h128, module_result.urn, context.now
        )
    )
    crosswalk_rows.append(
        goid_compute.build_crosswalk_row(
            module_descriptor, module_result.urn, module_name, context.now
        )
    )

    for child in ast.iter_child_nodes(tree):
        _process_ast_node(
            child,
            module_name,
            context=context,
            goid_rows=goid_rows,
            crosswalk_rows=crosswalk_rows,
        )

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

    gateway.policy.ensure_table("core.goids")
    gateway.policy.delete_for_snapshot("core.goids", repo=repo, commit=commit)
    gateway.policy.bulk_insert("core.goids", [row.to_tuple() for row in rows])
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

    gateway.policy.ensure_table("core.goid_crosswalk")
    gateway.policy.delete_for_snapshot("core.goid_crosswalk", repo=repo, commit=commit)
    gateway.policy.bulk_insert("core.goid_crosswalk", [row.to_tuple() for row in rows])
    return len(rows)


class GoidBuilderPlugin(MetadataPlugin):
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

    _core_metadata: ClassVar[CorePluginMetadata] = GOID_BUILDER_METADATA

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
        _ = self

        opts = self.resolve_options(GoidBuilderOptions)

        repo = ctx.snapshot.repo
        commit = ctx.snapshot.commit

        try:
            source_root = ctx.snapshot.repo_root or get_source_root(ctx.gateway, repo, commit)

            tracked_files = filter_paths(
                _get_tracked_files(ctx.gateway, repo, commit),
                scope_paths=opts.scope_paths,
                include_tests=opts.include_tests,
            )

            if not tracked_files:
                log.info("goid_builder: No tracked files found, skipping")
                return TargetResult.succeeded(
                    row_counts={
                        "core.goids": 0,
                        "core.goid_crosswalk": 0,
                    }
                )

            now = datetime.now(UTC)
            all_goid_rows: list[GoidRow] = []
            all_crosswalk_rows: list[GoidCrosswalkRow] = []

            for rel_path in tracked_files:
                rows = _extract_entities_from_file(
                    source_root / rel_path,
                    GoidExtractionContext(
                        repo=repo,
                        commit=commit,
                        now=now,
                        options=opts,
                        module_name=_path_to_module_name(rel_path),
                        normalized_path=normalize_rel_path(rel_path),
                    ),
                )
                all_goid_rows.extend(rows[0])
                all_crosswalk_rows.extend(rows[1])

            log.info(
                "goid_builder: Extracted %d GOIDs and %d crosswalk entries from %d files",
                len(all_goid_rows),
                len(all_crosswalk_rows),
                len(tracked_files),
            )

            goid_count = _persist_goid_rows(ctx.gateway, all_goid_rows, repo, commit)
            crosswalk_count = _persist_crosswalk_rows(ctx.gateway, all_crosswalk_rows, repo, commit)

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
