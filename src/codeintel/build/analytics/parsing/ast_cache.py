"""Shared helpers for resolving function GOIDs to AST nodes."""

from __future__ import annotations

import ast
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pyarrow as pa

from codeintel.build.analytics.functions.parsing import parse_python_file
from codeintel.core.columnar.conversion import table_to_reader
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.paths import normalize_path

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionAst:
    """Resolved AST node and metadata for a function GOID."""

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    lines: list[str]


@dataclass(frozen=True)
class FunctionAstLoadRequest:
    """Inputs required to resolve GOIDs to ASTs."""

    repo: str
    commit: str
    repo_root: Path
    catalog_provider: FunctionCatalogProvider | None = None
    max_functions: int | None = None
    worklist: pa.Table | pa.RecordBatchReader | None = None


class _SpanMeta(Protocol):
    @property
    def goid(self) -> int: ...

    @property
    def qualname(self) -> str: ...

    @property
    def start_line(self) -> int: ...

    @property
    def end_line(self) -> int: ...


@dataclass(frozen=True)
class _WorklistMeta:
    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int


def load_function_asts(request: FunctionAstLoadRequest) -> tuple[dict[int, FunctionAst], set[int]]:
    """
    Build a mapping of GOID -> FunctionAst for a repository snapshot.

    Parameters
    ----------
    request:
        Details describing the target snapshot and budget constraints.

    Returns
    -------
    tuple[dict[int, FunctionAst], set[int]]
        Mapping of GOID to resolved AST details and a set of GOIDs that could
        not be resolved due to parse failures or missing spans.

    Raises
    ------
    ValueError
        If the catalog provider is missing from the request.
    """
    functions_by_path: Mapping[str, Sequence[_SpanMeta]] = {}
    if request.worklist is not None:
        functions_by_path = _functions_by_path_from_worklist(request.worklist)
    else:
        if request.catalog_provider is None:
            msg = "FunctionAstLoadRequest.catalog_provider is required for AST loading."
            raise ValueError(msg)
        provider = request.catalog_provider
        catalog = provider.catalog()
        functions_by_path = catalog.functions_by_path

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
        normalized_path = normalize_path(rel_path)
        abs_path = (request.repo_root / normalized_path).resolve()
        metas_for_path = [
            meta for meta in metas if allowed_goids is None or meta.goid in allowed_goids
        ]
        if allowed_goids is not None and not metas_for_path:
            missing.update(meta.goid for meta in metas)
            continue
        try:
            parsed = parse_python_file(abs_path)
        except (OSError, ValueError):
            log.debug("Skipping %s; failed to parse file", abs_path)
            missing.update(meta.goid for meta in metas_for_path)
            continue

        for meta in metas_for_path:
            node = parsed.span_index.lookup(meta.start_line, meta.end_line)
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
                lines=list(parsed.lines),
            )

    return ast_by_goid, missing


def _functions_by_path_from_worklist(
    worklist: pa.Table | pa.RecordBatchReader,
) -> dict[str, list[_WorklistMeta]]:
    reader = _worklist_reader(worklist)
    functions_by_path: dict[str, list[_WorklistMeta]] = {}
    for goid, rel_path, qualname, start_line, end_line in iter_tuples(
        reader,
        columns=("goid_h128", "rel_path", "qualname", "start_line", "end_line"),
    ):
        goid_id = normalize_decimal_id(goid)
        if goid_id is None:
            continue
        if not isinstance(rel_path, str) or not isinstance(qualname, str):
            continue
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            continue
        normalized_path = normalize_path(rel_path)
        functions_by_path.setdefault(normalized_path, []).append(
            _WorklistMeta(
                goid=goid_id,
                rel_path=normalized_path,
                qualname=qualname,
                start_line=start_line,
                end_line=end_line,
            )
        )
    return functions_by_path


def _worklist_reader(
    worklist: pa.Table | pa.RecordBatchReader,
) -> pa.RecordBatchReader:
    if isinstance(worklist, pa.RecordBatchReader):
        return worklist
    return table_to_reader(worklist)
