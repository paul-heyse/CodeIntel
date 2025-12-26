"""Native Hamilton implementation for import_graph target.

This module implements import graph construction as a native Hamilton pipeline with:
- t__import_graph__run: Parse source files and extract imports
- t__import_graph__ingest: Package row payloads for materialization
- t__import_graph: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

import ibis.expr.types as ir

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.helpers import filter_mapping, get_source_root
from codeintel.build.hamilton.native.options.graphs import ImportGraphOptions
from codeintel.build.hamilton.native.patterns import (
    IngestStep,
    SaverContext,
    TableSaveSpec,
    ToolFinalizeContext,
    ToolRunContext,
    finalize_target_from_materializations,
    run_tool_step,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord, options_hash_for_target
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.ibis_typing import filter_by
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import imports as imports_compute
from codeintel.storage.gateway import DuckDBError

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

IMPORT_GRAPH_TARGET_NAME = "import_graph"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
IMPORT_GRAPH_EDGES_TABLE_KEY = "graph.import_graph_edges"
IMPORT_GRAPH_TABLE_KEYS = (
    IMPORT_MODULES_TABLE_KEY,
    IMPORT_GRAPH_EDGES_TABLE_KEY,
)

IMPORT_GRAPH_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=IMPORT_GRAPH_TARGET_NAME,
    hash_options_node="import_graph__hash_options",
)


@dataclass(frozen=True)
class ImportGraphToolOutput(ToolStepOutput):
    """Tool step output for import graph extraction."""

    module_rows: tuple[tuple[object, ...], ...] = ()
    edge_rows: tuple[tuple[object, ...], ...] = ()


@tag_helper(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def import_graph__hash_options(
    env: BuildEnv,
    modules__hash_options: InputHashOptions,
) -> InputHashOptions:
    """Build hash options for import graph materialization."""
    options_hash = options_hash_for_target(env, IMPORT_GRAPH_TARGET_NAME)
    file_state_hash = modules__hash_options.file_state_hash
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=file_state_hash,
    )


@tag_helper(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def import_graph__source_root(env: BuildEnv) -> Path | None:
    """Resolve the repository source root for import graph extraction."""
    repo_root = env.snapshot.repo_root
    if repo_root is not None:
        return repo_root
    try:
        return get_source_root(env.gateway, env.snapshot.repo, env.snapshot.commit)
    except (OSError, RuntimeError, ValueError):
        return None


@tag_helper(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def import_graph__module_map(
    env: BuildEnv,
    q__core__modules: ir.Table,
) -> dict[str, str]:
    """Build a mapping of module path to module name for import graph extraction."""
    module_by_path = _load_modules(q__core__modules, env.snapshot.repo, env.snapshot.commit)
    opts = load_target_options(
        env,
        target_name=IMPORT_GRAPH_TARGET_NAME,
        options_type=ImportGraphOptions,
    )
    return filter_mapping(module_by_path, scope_paths=opts.scope_paths)


@tag_helper(domain="graphs")
def _load_modules(
    q__core__modules: ir.Table,
    repo: str,
    commit: str,
) -> dict[str, str]:
    """Load module information from core.modules.

    Parameters
    ----------
    q__core__modules
        Ibis table expression for core.modules.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    dict[str, str]
        Mapping of relative path to module name.
    """
    try:
        expr = filter_by(
            q__core__modules,
            q__core__modules.repo == repo,
            q__core__modules.commit == commit,
        ).select(q__core__modules.path, q__core__modules.module)
        df = expr.execute()
        return {
            normalize_path(str(path)): str(module)
            for path, module in df.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper(domain="graphs")
def _extract_imports_from_file(file_path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Extract imports from a Python file.

    Parameters
    ----------
    file_path
        Absolute path to the file.

    Returns
    -------
    list[tuple[str, tuple[str, ...]]]
        List of (module_name, imported_names) tuples.
    """
    if not file_path.exists():
        return []

    try:
        source = file_path.read_text(encoding="utf8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    imports: list[tuple[str, tuple[str, ...]]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names = tuple(alias.name for alias in node.names)
                imports.append((module, names))

    return imports


def _collect_import_edges(
    *,
    source_root: Path,
    module_by_path: dict[str, str],
) -> list[imports_compute.ImportEdge]:
    edges: list[imports_compute.ImportEdge] = []
    for rel_path, module_name in module_by_path.items():
        edges.extend(
            imports_compute.collect_import_edges(
                module_name,
                _extract_imports_from_file(source_root / rel_path),
            )
        )
    return edges


def _coerce_import_graph_output(output: ToolStepOutput) -> ImportGraphToolOutput:
    if isinstance(output, ImportGraphToolOutput):
        return output
    return ImportGraphToolOutput(result=output.result)


@tag_tool(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def t__import_graph__run(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    import_graph__source_root: Path | None,
    import_graph__module_map: dict[str, str],
    import_graph__hash_options: InputHashOptions,
) -> ImportGraphToolOutput:
    """Execute import graph extraction on repository modules.

    Returns
    -------
    ImportGraphToolOutput
        Tool output with row tuples for import graph tables.
    """
    context = ToolRunContext(
        env=env,
        graph=graph,
        target_name=IMPORT_GRAPH_TARGET_NAME,
        hash_options=import_graph__hash_options,
        skip_reason="import_graph skipped",
    )

    def _execute() -> ImportGraphToolOutput:
        if t__modules.status != "succeeded":
            return ImportGraphToolOutput(
                result=ExecutionResult.failed(f"Upstream modules target failed: {t__modules.error}")
            )

        if import_graph__source_root is None:
            return ImportGraphToolOutput(
                result=ExecutionResult.failed("Import graph source root could not be resolved")
            )

        module_by_path = import_graph__module_map
        if not module_by_path:
            return ImportGraphToolOutput(
                result=ExecutionResult.ok(
                    table_counts={
                        IMPORT_MODULES_TABLE_KEY: 0,
                        IMPORT_GRAPH_EDGES_TABLE_KEY: 0,
                    }
                )
            )

        edges = _collect_import_edges(
            source_root=import_graph__source_root,
            module_by_path=module_by_path,
        )
        analysis = imports_compute.analyze_imports(edges, set(module_by_path.values()))
        log.info(
            "import_graph: %d edges, %d SCCs",
            len(edges),
            len(set(analysis.scc_map.values())),
        )

        module_rows = tuple(
            row.to_tuple()
            for row in imports_compute.build_import_module_rows(
                env.snapshot.repo,
                env.snapshot.commit,
                analysis,
            )
        )
        edge_rows = tuple(
            row.to_tuple()
            for row in imports_compute.build_import_edge_rows(
                env.snapshot.repo,
                env.snapshot.commit,
                analysis,
            )
        )

        return ImportGraphToolOutput(
            result=ExecutionResult.ok(
                table_counts={
                    IMPORT_MODULES_TABLE_KEY: len(module_rows),
                    IMPORT_GRAPH_EDGES_TABLE_KEY: len(edge_rows),
                }
            ),
            module_rows=module_rows,
            edge_rows=edge_rows,
        )

    return _coerce_import_graph_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def t__import_graph__ingest(
    t__import_graph__run: ImportGraphToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package import graph rows for table materialization."""
    result = t__import_graph__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "Import graph skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "Import graph failed",
                warnings=result.warnings,
            )
        )

    payload = {
        IMPORT_MODULES_TABLE_KEY: t__import_graph__run.module_rows,
        IMPORT_GRAPH_EDGES_TABLE_KEY: t__import_graph__run.edge_rows,
    }
    table_counts = {
        IMPORT_MODULES_TABLE_KEY: len(t__import_graph__run.module_rows),
        IMPORT_GRAPH_EDGES_TABLE_KEY: len(t__import_graph__run.edge_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=IMPORT_GRAPH_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=IMPORT_MODULES_TABLE_KEY),
)
@tag_compute(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME, target_="import_graph__modules_rows")
def import_graph__modules_rows(
    t__import_graph__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.import_modules."""
    if t__import_graph__ingest.result.skipped or not t__import_graph__ingest.result.success:
        return None
    payload = t__import_graph__ingest.payload
    if payload is None:
        msg = "Missing import graph payload"
        raise ValueError(msg)
    rows = payload.get(IMPORT_MODULES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {IMPORT_MODULES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=IMPORT_GRAPH_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=IMPORT_GRAPH_EDGES_TABLE_KEY),
)
@tag_compute(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME, target_="import_graph__edges_rows")
def import_graph__edges_rows(
    t__import_graph__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.import_graph_edges."""
    if t__import_graph__ingest.result.skipped or not t__import_graph__ingest.result.success:
        return None
    payload = t__import_graph__ingest.payload
    if payload is None:
        msg = "Missing import graph payload"
        raise ValueError(msg)
    rows = payload.get(IMPORT_GRAPH_EDGES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {IMPORT_GRAPH_EDGES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def import_graph__table_materializations(
    m__graph__import_modules: MaterializationMetadata,
    m__graph__import_graph_edges: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect materialization metadata for import graph tables."""
    return {
        IMPORT_MODULES_TABLE_KEY: m__graph__import_modules,
        IMPORT_GRAPH_EDGES_TABLE_KEY: m__graph__import_graph_edges,
    }


@tag_helper(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def import_graph__finalize_context(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    graph: TargetGraph,
    import_graph__hash_options: InputHashOptions,
) -> ToolFinalizeContext:
    """Build finalization context for import graph."""
    return ToolFinalizeContext(
        env=env,
        graph=graph,
        target_name=IMPORT_GRAPH_TARGET_NAME,
        hash_options=import_graph__hash_options,
    )


@codeintel_target(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def t__import_graph(
    import_graph__finalize_context: ToolFinalizeContext,
    t__import_graph__run: ImportGraphToolOutput,
    t__import_graph__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    import_graph__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Construct a module import graph."""
    return finalize_target_from_materializations(
        context=import_graph__finalize_context,
        tool_step=t__import_graph__run,
        ingest_step=t__import_graph__ingest,
        artifact_materializations=None,
        table_materializations=import_graph__table_materializations,
    )


__all__ = [
    "t__import_graph",
    "t__import_graph__ingest",
    "t__import_graph__run",
]
