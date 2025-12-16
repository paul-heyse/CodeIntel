"""Native Hamilton implementation for goids target.

This module implements GOID (Global Object IDentifier) construction as a
native Hamilton pipeline with:
- t__goids__extract: Parse source files and compute GOIDs
- t__goids: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import filter_paths, get_source_root, persist_rows
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.options.graphs import GoidBuilderOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import goid as goid_compute
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.graphs.compute.goid import GoidCrosswalkRow, GoidRow
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class GoidExtractionContext:
    """Context for GOID extraction.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    now
        Timestamp for row creation.
    options
        GOID builder options.
    module_name
        Module name for the current file.
    normalized_path
        Normalized relative path for the current file.
    """

    repo: str
    commit: str
    now: datetime
    options: GoidBuilderOptions
    module_name: str
    normalized_path: str


@dataclass(frozen=True)
class GoidExtractResult:
    """Result from GOID extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    goid_count
        Number of GOIDs extracted.
    crosswalk_count
        Number of crosswalk entries extracted.
    table_counts
        Row counts per produced table.
    error
        Fatal error message if extraction failed.
    """

    success: bool
    goid_count: int = 0
    crosswalk_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
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


@tag(node_type="helper")
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


@tag(node_type="helper")
def _process_ast_node(
    node: ast.AST,
    parent_qualname: str | None,
    *,
    context: GoidExtractionContext,
    goid_rows: list[GoidRow],
    crosswalk_rows: list[GoidCrosswalkRow],
) -> None:
    """Process an AST node recursively.

    Parameters
    ----------
    node
        The AST node to process.
    parent_qualname
        Qualified name of the parent node.
    context
        GOID extraction context.
    goid_rows
        List to append GOID rows to.
    crosswalk_rows
        List to append crosswalk rows to.
    """
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


@tag(node_type="helper")
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


@tag(domain="graphs", target="goids", node_type="compute")
def t__goids__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> GoidExtractResult:
    """Execute GOID extraction on repository modules.

    This is the compute node for the goids target. It parses Python source
    files, extracts modules, classes, and functions, and computes stable
    GOIDs for each entity.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    GoidExtractResult
        Result containing GOID and crosswalk row counts.

    Notes
    -----
    Produces:
    - core.goids: GOID records for all entities
    - core.goid_crosswalk: GOID crosswalk records
    """
    if t__modules.status != "succeeded":
        return GoidExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = GoidBuilderOptions()

        source_root = env.snapshot.repo_root or get_source_root(env.gateway, repo, commit)

        tracked_files = filter_paths(
            _get_tracked_files(env.gateway, repo, commit),
            scope_paths=opts.scope_paths,
            include_tests=opts.include_tests,
        )

        if not tracked_files:
            log.info("goids: No tracked files found, skipping")
            return GoidExtractResult(
                success=True,
                goid_count=0,
                crosswalk_count=0,
                table_counts={
                    "core.goids": 0,
                    "core.goid_crosswalk": 0,
                },
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
                    normalized_path=normalize_path(rel_path),
                ),
            )
            all_goid_rows.extend(rows[0])
            all_crosswalk_rows.extend(rows[1])

        log.info(
            "goids: Extracted %d GOIDs and %d crosswalk entries from %d files",
            len(all_goid_rows),
            len(all_crosswalk_rows),
            len(tracked_files),
        )

        goid_count = persist_rows(
            env.gateway, "core.goids", all_goid_rows, repo=repo, commit=commit
        )
        crosswalk_count = persist_rows(
            env.gateway, "core.goid_crosswalk", all_crosswalk_rows, repo=repo, commit=commit
        )

        log.info(
            "goids: Persisted %d GOIDs and %d crosswalk entries",
            goid_count,
            crosswalk_count,
        )

        return GoidExtractResult(
            success=True,
            goid_count=goid_count,
            crosswalk_count=crosswalk_count,
            table_counts={
                "core.goids": goid_count,
                "core.goid_crosswalk": crosswalk_count,
            },
        )

    except Exception as exc:
        log.exception("GOID extraction failed")
        return GoidExtractResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="goids", node_type="materialize")
def t__goids(
    env: BuildEnv,
    graph: TargetGraph,
    t__goids__extract: GoidExtractResult,
) -> TargetRunRecord:
    """Materialize GOIDs target with validation.

    This is the entry point for the goids target. It orchestrates
    GOID extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__goids__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "goids")

    if executor.should_skip():
        return executor.skip()

    if not t__goids__extract.success:
        return executor.fail(RuntimeError(t__goids__extract.error or "GOID extraction failed"))

    def compute() -> dict[str, int]:
        return dict(t__goids__extract.table_counts)

    return executor.execute(compute)


__all__ = [
    "GoidExtractResult",
    "GoidExtractionContext",
    "t__goids",
    "t__goids__extract",
]
