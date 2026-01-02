"""Context container for advanced query execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intervaltree import IntervalTree

from tools.advanced_query_engine.backends.astgrep_backend import AstGrepRoot, parse_ast_grep_source
from tools.advanced_query_engine.backends.libcst_backend import LibCSTIndex, build_def_index
from tools.advanced_query_engine.backends.treesitter_backend import (
    TreeSitterParsed,
    parse_tree_sitter_source,
)
from tools.advanced_query_engine.contracts import QueryBudget
from tools.advanced_query_engine.packs.catalog import PackCatalog
from tools.advanced_query_engine.util.line_index import LineIndex
from tools.advanced_query_engine.util.repo_cache import RepoCache
from tools.advanced_query_engine.util.snippets import SnippetConfig


@dataclass
class SearchContext:
    """Shared context for query handlers."""

    repo_root: Path
    query_catalog: PackCatalog
    wiring_catalog: PackCatalog
    cache: RepoCache
    snippet_config: SnippetConfig
    default_budget: QueryBudget
    _def_indexes: dict[str, LibCSTIndex]
    _ast_grep_roots: dict[str, AstGrepRoot]
    _tree_sitter_parses: dict[tuple[str, str], TreeSitterParsed]

    def __init__(
        self,
        *,
        repo_root: Path,
        query_catalog: PackCatalog,
        wiring_catalog: PackCatalog,
        snippet_config: SnippetConfig,
        default_budget: QueryBudget,
    ) -> None:
        self.repo_root = repo_root
        self.query_catalog = query_catalog
        self.wiring_catalog = wiring_catalog
        self.cache = RepoCache(repo_root)
        self.snippet_config = snippet_config
        self.default_budget = default_budget
        self._def_indexes = {}
        self._ast_grep_roots = {}
        self._tree_sitter_parses = {}

    def read_text(self, rel_path: str) -> str:
        """Read a file as UTF-8 text, replacing invalid bytes.

        Returns
        -------
        str
            Decoded text content.
        """
        data = self.cache.read_bytes(rel_path)
        return data.decode("utf-8", errors="replace")

    def line_index(self, rel_path: str) -> LineIndex:
        """Return a cached LineIndex for a file.

        Returns
        -------
        LineIndex
            Line index for the file.
        """
        return self.cache.line_index(rel_path)

    def def_index(self, rel_path: str) -> LibCSTIndex:
        """Return a cached LibCST definition index for a file.

        Returns
        -------
        LibCSTIndex
            Definition index for the file.
        """
        if rel_path not in self._def_indexes:
            self._def_indexes[rel_path] = build_def_index(rel_path, self.cache.read_bytes(rel_path))
        return self._def_indexes[rel_path]

    def span_tree(self, rel_path: str) -> IntervalTree:
        """Return a cached interval tree for definition spans.

        Returns
        -------
        IntervalTree
            Interval tree for definition spans.
        """
        return self.def_index(rel_path).span_tree()

    def ast_grep_root(self, rel_path: str) -> AstGrepRoot:
        """Return a cached ast-grep root for a file.

        Returns
        -------
        AstGrepRoot
            Parsed ast-grep root.
        """
        if rel_path not in self._ast_grep_roots:
            source = self.read_text(rel_path)
            self._ast_grep_roots[rel_path] = parse_ast_grep_source(source)
        return self._ast_grep_roots[rel_path]

    def tree_sitter_parse(self, rel_path: str, language: str) -> TreeSitterParsed:
        """Return a cached tree-sitter parse for a file.

        Returns
        -------
        TreeSitterParsed
            Parsed tree and line index.
        """
        key = (rel_path, language)
        if key not in self._tree_sitter_parses:
            source_bytes = self.cache.read_bytes(rel_path)
            self._tree_sitter_parses[key] = parse_tree_sitter_source(language, source_bytes)
        return self._tree_sitter_parses[key]


__all__ = ["SearchContext"]
