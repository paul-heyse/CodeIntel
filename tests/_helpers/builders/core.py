"""Row dataclasses for core.* schema tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from tests._helpers.builders._common import _iso

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "AstMetricsRow",
    "DocstringRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "ModuleRow",
    "RepoMapRow",
]


@dataclass(frozen=True)
class RepoMapRow:
    """Row for core.repo_map."""

    __table__: ClassVar[str] = "core.repo_map"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "modules",
        "overlays",
        "generated_at",
    )

    repo: str
    commit: str
    modules: dict[str, str]
    overlays: dict[str, str] | None = None
    generated_at: datetime | None = None

    def to_tuple(self) -> tuple[str, str, str, str, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.repo,
            self.commit,
            json.dumps(self.modules),
            json.dumps(self.overlays or {}),
            _iso(self.generated_at),
        )


@dataclass(frozen=True)
class ModuleRow:
    """Row for core.modules."""

    __table__: ClassVar[str] = "core.modules"
    __columns__: ClassVar[tuple[str, ...]] = (
        "module",
        "path",
        "repo",
        "commit",
        "language",
        "tags",
        "owners",
    )

    module: str
    path: str
    repo: str
    commit: str
    tags: str = "[]"
    owners: str = "[]"

    def to_tuple(self) -> tuple[str, str, str, str, str, str, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.module,
            self.path,
            self.repo,
            self.commit,
            "python",
            self.tags,
            self.owners,
        )


@dataclass(frozen=True)
class GoidRow:
    """Row for core.goids."""

    __table__: ClassVar[str] = "core.goids"
    __columns__: ClassVar[tuple[str, ...]] = (
        "goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "created_at",
    )

    goid_h128: int
    urn: str
    repo: str
    commit: str
    rel_path: str
    kind: str
    qualname: str
    start_line: int
    end_line: int
    language: str = "python"
    created_at: datetime | None = None

    def to_tuple(self) -> tuple[int, str, str, str, str, str, str, str, int, int, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.goid_h128,
            self.urn,
            self.repo,
            self.commit,
            self.rel_path,
            self.language,
            self.kind,
            self.qualname,
            self.start_line,
            self.end_line,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class GoidCrosswalkRow:
    """Row for core.goid_crosswalk."""

    __table__: ClassVar[str] = "core.goid_crosswalk"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "goid",
        "lang",
        "module_path",
        "file_path",
        "start_line",
        "end_line",
        "scip_symbol",
        "ast_qualname",
        "cst_node_id",
        "chunk_id",
        "symbol_id",
        "updated_at",
    )

    repo: str
    commit: str
    goid: str
    lang: str
    module_path: str
    file_path: str
    start_line: int
    end_line: int
    scip_symbol: str
    ast_qualname: str
    cst_node_id: str | None
    chunk_id: str | None
    symbol_id: str | None
    updated_at: datetime | None = None

    def to_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        str,
        str,
        str | None,
        str | None,
        str | None,
        str,
    ]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.repo,
            self.commit,
            self.goid,
            self.lang,
            self.module_path,
            self.file_path,
            self.start_line,
            self.end_line,
            self.scip_symbol,
            self.ast_qualname,
            self.cst_node_id,
            self.chunk_id,
            self.symbol_id,
            _iso(self.updated_at),
        )


@dataclass(frozen=True)
class DocstringRow:
    """Row for core.docstrings."""

    __table__: ClassVar[str] = "core.docstrings"
    __columns__: ClassVar[tuple[str, ...]] = (
        "repo",
        "commit",
        "rel_path",
        "module",
        "qualname",
        "kind",
        "lineno",
        "end_lineno",
        "raw_docstring",
        "style",
        "short_desc",
        "long_desc",
        "params",
        "returns",
        "raises",
        "examples",
        "created_at",
    )

    repo: str
    commit: str
    rel_path: str
    module: str
    qualname: str
    kind: str
    lineno: int
    end_lineno: int
    raw_docstring: str
    style: str
    short_desc: str
    long_desc: str
    params_json: str
    returns_json: str
    raises_json: str
    examples_json: str
    created_at: datetime

    def to_tuple(
        self,
    ) -> tuple[
        str,
        str,
        str,
        str,
        str,
        str,
        int,
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
    ]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.repo,
            self.commit,
            self.rel_path,
            self.module,
            self.qualname,
            self.kind,
            self.lineno,
            self.end_lineno,
            self.raw_docstring,
            self.style,
            self.short_desc,
            self.long_desc,
            self.params_json,
            self.returns_json,
            self.raises_json,
            self.examples_json,
            _iso(self.created_at),
        )


@dataclass(frozen=True)
class AstMetricsRow:
    """Row for core.ast_metrics."""

    __table__: ClassVar[str] = "core.ast_metrics"
    __columns__: ClassVar[tuple[str, ...]] = (
        "rel_path",
        "node_count",
        "function_count",
        "class_count",
        "avg_depth",
        "max_depth",
        "complexity",
        "generated_at",
    )

    rel_path: str
    node_count: int
    function_count: int
    class_count: int
    avg_depth: float
    max_depth: int
    complexity: float
    generated_at: datetime

    def to_tuple(
        self,
    ) -> tuple[str, int, int, int, float, int, float, str]:
        """Serialize row to database insert order.

        Returns
        -------
        tuple
            Values in column order for INSERT.
        """
        return (
            self.rel_path,
            self.node_count,
            self.function_count,
            self.class_count,
            self.avg_depth,
            self.max_depth,
            self.complexity,
            _iso(self.generated_at),
        )
