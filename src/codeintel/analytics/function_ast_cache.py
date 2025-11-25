"""Shared helpers for resolving function GOIDs to AST nodes."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

from codeintel.analytics.function_parsing import (
    FunctionParserRegistry,
    ParsedFile,
    get_parsed_file,
)
from codeintel.analytics.span_resolver import resolve_span
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionAst:
    """Resolved AST node and metadata for a function GOID."""

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int
    node: ast.FunctionDef | ast.AsyncFunctionDef
    lines: list[str]


@dataclass(frozen=True)
class FunctionAstLoadRequest:
    """Inputs required to resolve GOIDs to ASTs."""

    repo: str
    commit: str
    repo_root: Path
    catalog_provider: FunctionCatalogProvider | None = None
    max_functions: int | None = None


def load_function_asts(
    gateway: StorageGateway,
    request: FunctionAstLoadRequest,
) -> tuple[dict[int, FunctionAst], set[int]]:
    """
    Build a mapping of GOID -> FunctionAst for a repository snapshot.

    Parameters
    ----------
    gateway:
        Storage gateway providing access to DuckDB.
    request:
        Details describing the target snapshot and budget constraints.

    Returns
    -------
    tuple[dict[int, FunctionAst], set[int]]
        Mapping of GOID to resolved AST details and a set of GOIDs that could
        not be resolved due to parse failures or missing spans.
    """
    provider = request.catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=request.repo, commit=request.commit
    )
    catalog = provider.catalog()
    functions_by_path = catalog.functions_by_path

    parser = FunctionParserRegistry().get(None)
    parsed_cache: dict[str, ParsedFile | None] = {}

    ast_by_goid: dict[int, FunctionAst] = {}
    missing: set[int] = set()

    allowed_goids: set[int] | None = None
    if request.max_functions is not None and request.max_functions > 0:
        sorted_metas = sorted(
            ((path, meta) for path, metas in functions_by_path.items() for meta in metas),
            key=lambda item: (
                item[0],
                getattr(item[1], "start_line", 0),
                getattr(item[1], "end_line", 0),
                getattr(item[1], "qualname", ""),
            ),
        )
        allowed_goids = {meta.goid for _, meta in sorted_metas[: request.max_functions]}

    for rel_path, metas in functions_by_path.items():
        normalized_path = normalize_rel_path(rel_path)
        abs_path = (request.repo_root / normalized_path).resolve()
        metas_for_path = [
            meta for meta in metas if allowed_goids is None or meta.goid in allowed_goids
        ]
        if allowed_goids is not None and not metas_for_path:
            missing.update(meta.goid for meta in metas)
            continue
        parsed = get_parsed_file(normalized_path, abs_path, parsed_cache, parser)
        if parsed is None:
            log.debug("Skipping %s; failed to parse file", abs_path)
            missing.update(meta.goid for meta in metas_for_path)
            continue

        for meta in metas_for_path:
            resolution = resolve_span(parsed.index, meta.start_line, meta.end_line)
            node = resolution.node
            if node is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                missing.add(meta.goid)
                continue
            ast_by_goid[meta.goid] = FunctionAst(
                goid=meta.goid,
                rel_path=normalized_path,
                qualname=meta.qualname,
                start_line=meta.start_line,
                end_line=meta.end_line,
                node=node,
                lines=parsed.lines,
            )

    return ast_by_goid, missing
