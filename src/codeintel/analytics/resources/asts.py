"""AST resource provider for parsed AST access.

This module provides `AstProvider` for lazy loading of parsed AST
maps used in function analytics.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.resources.protocol import LazyResource

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionAstInfo:
    """AST information for a single function.

    Attributes
    ----------
    goid
        Global object identifier for the function.
    node
        The AST node for the function.
    lines
        Source lines for the function's file.
    rel_path
        Relative path to the source file.
    """

    goid: int
    node: ast.FunctionDef | ast.AsyncFunctionDef
    lines: list[str]
    rel_path: str


@dataclass
class AstMap:
    """Container for function AST data.

    Attributes
    ----------
    functions
        Mapping from GOID to AST info.
    missing_goids
        Set of GOIDs that could not be parsed.
    files_parsed
        Number of files successfully parsed.
    parse_errors
        Number of files with parse errors.
    """

    functions: dict[int, FunctionAstInfo]
    missing_goids: set[int]
    files_parsed: int = 0
    parse_errors: int = 0

    def get(self, goid: int) -> FunctionAstInfo | None:
        """Get AST info for a GOID.

        Parameters
        ----------
        goid
            The function GOID.

        Returns
        -------
        FunctionAstInfo | None
            The AST info, or None if not available.
        """
        return self.functions.get(goid)

    def __contains__(self, goid: int) -> bool:
        """Check if a GOID has AST info.

        Parameters
        ----------
        goid
            The function GOID.

        Returns
        -------
        bool
            True if AST info is available.
        """
        return goid in self.functions


class AstProvider(LazyResource[AstMap]):
    """Provider for function ASTs with lazy loading.

    Parses source files and builds a map from function GOIDs to their
    AST nodes. Files are parsed on demand.

    Example
    -------
    >>> provider = AstProvider(gateway, snapshot)
    >>> ast_map = provider.get()
    >>> func_ast = ast_map.get(function_goid)
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        max_files: int | None = None,
    ) -> None:
        """Initialize the AST provider.

        Parameters
        ----------
        gateway
            Storage gateway for GOID queries.
        snapshot
            Repository snapshot reference.
        max_files
            Maximum number of files to parse (for resource limits).
        """
        super().__init__("AstMap")
        self._gateway = gateway
        self._snapshot = snapshot
        self._max_files = max_files

    def _load(self) -> AstMap:
        """Load and parse function ASTs.

        Returns
        -------
        AstMap
            Map of function GOIDs to AST info.
        """
        # Load GOIDs from database
        goids_by_file = self._load_goids_by_file()

        functions: dict[int, FunctionAstInfo] = {}
        missing_goids: set[int] = set()
        files_parsed = 0
        parse_errors = 0

        repo_root = self._snapshot.repo_root

        for rel_path, goid_list in goids_by_file.items():
            abs_path = (repo_root / rel_path).resolve()

            try:
                source = abs_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(rel_path))
                lines = source.splitlines()
                files_parsed += 1
            except (OSError, SyntaxError) as e:
                log.debug("Failed to parse %s: %s", rel_path, e)
                parse_errors += 1
                missing_goids.update(g["goid_h128"] for g in goid_list)
                continue

            # Build line-to-function index
            func_nodes: dict[tuple[int, int], ast.FunctionDef | ast.AsyncFunctionDef] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = node.lineno
                    end = getattr(node, "end_lineno", start) or start
                    func_nodes[start, end] = node

            # Match GOIDs to AST nodes
            for goid_info in goid_list:
                goid = int(goid_info["goid_h128"])
                start = int(goid_info["start_line"])
                end_raw = goid_info.get("end_line")
                end = int(end_raw) if end_raw is not None else start

                node = func_nodes.get((start, end))
                if node is None:
                    # Try fuzzy match
                    node = self._fuzzy_match(func_nodes, start, end)

                if node is not None:
                    functions[goid] = FunctionAstInfo(
                        goid=goid,
                        node=node,
                        lines=lines,
                        rel_path=rel_path,
                    )
                else:
                    missing_goids.add(goid)

        return AstMap(
            functions=functions,
            missing_goids=missing_goids,
            files_parsed=files_parsed,
            parse_errors=parse_errors,
        )

    def _load_goids_by_file(self) -> dict[str, list[dict[str, object]]]:
        """Load GOIDs grouped by file.

        Returns
        -------
        dict[str, list[dict[str, object]]]
            GOIDs grouped by relative path.
        """
        query = """
            SELECT
                goid_h128,
                rel_path,
                start_line,
                end_line
            FROM core.goids
            WHERE repo = ? AND commit = ?
              AND kind IN ('function', 'method')
        """
        result = self._gateway.con.execute(
            query,
            [self._snapshot.repo, self._snapshot.commit],
        )

        by_file: dict[str, list[dict[str, object]]] = {}
        files_seen = 0

        for row in result.fetchall():
            rel_path = str(row[1]).replace("\\", "/")

            # Honor file limit
            if self._max_files is not None:
                if rel_path not in by_file:
                    files_seen += 1
                    if files_seen > self._max_files:
                        continue

            goid_info: dict[str, object] = {
                "goid_h128": row[0],
                "start_line": row[2],
                "end_line": row[3],
            }
            by_file.setdefault(rel_path, []).append(goid_info)

        return by_file

    def _fuzzy_match(
        self,
        func_nodes: dict[tuple[int, int], ast.FunctionDef | ast.AsyncFunctionDef],
        target_start: int,
        target_end: int,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """Try to fuzzy match a function node.

        Parameters
        ----------
        func_nodes
            Available function nodes indexed by span.
        target_start
            Target start line.
        target_end
            Target end line.

        Returns
        -------
        ast.FunctionDef | ast.AsyncFunctionDef | None
            Matched node, or None.
        """
        # Try nodes starting at the same line
        for (start, end), node in func_nodes.items():
            if start == target_start:
                return node

        # Try nodes containing the target span
        for (start, end), node in func_nodes.items():
            if start <= target_start and end >= target_end:
                return node

        return None


__all__ = [
    "AstMap",
    "AstProvider",
    "FunctionAstInfo",
]
