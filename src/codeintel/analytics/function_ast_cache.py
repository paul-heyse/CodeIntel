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


def load_function_asts(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    repo_root: Path,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> tuple[dict[int, FunctionAst], set[int]]:
    """
    Build a mapping of GOID -> FunctionAst for a repository snapshot.

    Parameters
    ----------
    gateway:
        Storage gateway providing access to DuckDB.
    repo:
        Repository identifier.
    commit:
        Commit hash anchoring this analysis run.
    repo_root:
        Filesystem root of the repository (used to load source files).
    catalog_provider:
        Optional pre-loaded function catalog; when omitted a catalog is built from DuckDB.

    Returns
    -------
    tuple[dict[int, FunctionAst], set[int]]
        Mapping of GOID to resolved AST details and a set of GOIDs that could
        not be resolved due to parse failures or missing spans.
    """
    provider = catalog_provider or FunctionCatalogService.from_db(gateway, repo=repo, commit=commit)
    catalog = provider.catalog()
    functions_by_path = catalog.functions_by_path

    parser = FunctionParserRegistry().get(None)
    parsed_cache: dict[str, ParsedFile | None] = {}

    ast_by_goid: dict[int, FunctionAst] = {}
    missing: set[int] = set()

    for rel_path, metas in functions_by_path.items():
        normalized_path = normalize_rel_path(rel_path)
        abs_path = (repo_root / normalized_path).resolve()
        parsed = get_parsed_file(normalized_path, abs_path, parsed_cache, parser)
        if parsed is None:
            log.debug("Skipping %s; failed to parse file", abs_path)
            missing.update(meta.goid for meta in metas)
            continue

        for meta in metas:
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
