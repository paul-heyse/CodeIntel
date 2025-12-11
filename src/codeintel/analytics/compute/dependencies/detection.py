"""Pure computation for dependency detection in AST.

This module provides functions to detect dependency calls in Python AST,
build import alias maps, and group calls by library.

All functions are pure and side-effect-free.

Examples
--------
>>> import ast
>>> source = "import requests; requests.get('http://example.com')"
>>> tree = ast.parse(source)
>>> alias_map = build_alias_map(tree)
>>> alias_map.get("requests")
'requests'
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.compute.dependencies.classification import (
    classify_modes,
    risk_score,
)
from codeintel.analytics.utilities.ast import resolve_call_target, safe_unparse, snippet_from_lines

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.analytics.compute.dependencies.classification import (
        LibraryPattern,
    )


@dataclass(frozen=True)
class DependencyCall:
    """A single call into an external library.

    Attributes
    ----------
    library
        Library identifier (e.g., "requests").
    target
        Full call target string.
    modes
        List of usage modes (e.g., ["read", "query"]).
    severity
        Severity level of the call.
    criticality
        Criticality score.
    matched_pattern
        Name of the matched pattern (if any).
    risk_score
        Computed risk score.
    lineno
        Starting line number in source.
    end_lineno
        Ending line number in source.
    snippet
        Code snippet containing the call.
    """

    library: str
    target: str
    modes: list[str]
    severity: str | None
    criticality: float | None
    matched_pattern: str | None = None
    risk_score: float | None = None
    lineno: int | None = None
    end_lineno: int | None = None
    snippet: str = ""


class DependencyCallVisitor(ast.NodeVisitor):
    """Walk a function AST and collect dependency calls.

    This visitor identifies calls to external libraries based on import
    aliases and classifies them according to library patterns.

    Attributes
    ----------
    alias_map
        Mapping of local names to library names.
    patterns
        Library patterns for classification.
    calls
        Collected dependency calls.

    Examples
    --------
    >>> import ast
    >>> source = '''
    ... import requests
    ... requests.get("http://example.com")
    ... '''
    >>> tree = ast.parse(source)
    >>> patterns = {
    ...     "requests": LibraryPattern(
    ...         library="requests",
    ...         service_name="HTTP",
    ...         category="http",
    ...         matchers=[],
    ...     )
    ... }
    >>> visitor = DependencyCallVisitor(
    ...     alias_map={"requests": "requests"},
    ...     patterns=patterns,
    ...     rel_path="example.py",
    ...     lines=source.splitlines(),
    ... )
    >>> visitor.visit(tree)
    >>> len(visitor.calls)
    1
    """

    def __init__(
        self,
        alias_map: dict[str, str],
        patterns: dict[str, LibraryPattern],
        rel_path: str,
        lines: Sequence[str],
    ) -> None:
        """Initialize the visitor.

        Parameters
        ----------
        alias_map
            Mapping of local names to library names.
        patterns
            Library patterns for classification.
        rel_path
            Relative path of the source file.
        lines
            Source lines for snippet extraction.
        """
        self.alias_map = alias_map
        self.patterns = patterns
        self._rel_path = rel_path
        self._lines = lines
        self.calls: list[DependencyCall] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a Call node and classify if it's a dependency call.

        Parameters
        ----------
        node
            AST Call node to visit.
        """
        target = resolve_call_target(node.func, self.alias_map)
        library = target.library
        method = target.attribute or target.base
        if library is None or library not in self.patterns:
            self.generic_visit(node)
            return

        pattern = self.patterns[library]
        target_text = safe_unparse(node) or ""
        modes, matcher = classify_modes(pattern, method, target_text)
        severity = (matcher.severity if matcher else None) or pattern.severity
        criticality = (matcher.criticality if matcher else None) or pattern.criticality
        score = risk_score(severity, criticality)
        matched_pattern = matcher.name if matcher is not None else method
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", lineno)
        snippet = snippet_from_lines(self._lines, lineno, end_lineno)

        self.calls.append(
            DependencyCall(
                library=library,
                target=target_text or (method or ""),
                modes=modes,
                severity=severity,
                criticality=criticality,
                matched_pattern=matched_pattern,
                risk_score=score,
                lineno=lineno,
                end_lineno=end_lineno,
                snippet=snippet,
            )
        )
        self.generic_visit(node)


def build_alias_map(tree: ast.AST) -> dict[str, str]:
    """Build a mapping of local names to library names from imports.

    Parameters
    ----------
    tree
        Parsed AST to analyze.

    Returns
    -------
    dict[str, str]
        Mapping of local alias to library name.

    Examples
    --------
    >>> import ast
    >>> source = '''
    ... import requests
    ... import pandas as pd
    ... from sqlalchemy import create_engine
    ... '''
    >>> tree = ast.parse(source)
    >>> alias_map = build_alias_map(tree)
    >>> alias_map["requests"]
    'requests'
    >>> alias_map["pd"]
    'pandas'
    >>> alias_map["create_engine"]
    'sqlalchemy'
    """
    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = alias.asname or alias.name
                alias_map[key] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            base = node.module.split(".")[0]
            for alias in node.names:
                key = alias.asname or alias.name
                alias_map[key] = base
    return alias_map


def build_alias_maps(
    repo_root: Path,
    module_map: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Build alias maps for all modules in the repository.

    Parameters
    ----------
    repo_root
        Repository root directory.
    module_map
        Mapping of relative paths to module names.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of relative paths to their alias maps.
    """
    alias_maps: dict[str, dict[str, str]] = {}
    for rel_path in module_map:
        abs_path = repo_root / rel_path
        if not abs_path.is_file():
            continue
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=rel_path)
            alias_maps[rel_path] = build_alias_map(tree)
        except (SyntaxError, ValueError):
            continue
    return alias_maps


def group_calls_by_library(
    calls: list[DependencyCall],
) -> dict[str, list[DependencyCall]]:
    """Group dependency calls by library.

    Parameters
    ----------
    calls
        List of dependency calls to group.

    Returns
    -------
    dict[str, list[DependencyCall]]
        Mapping of library names to their calls.

    Examples
    --------
    >>> calls = [
    ...     DependencyCall(
    ...         library="requests", target="get", modes=["read"], severity=None, criticality=None
    ...     ),
    ...     DependencyCall(
    ...         library="requests", target="post", modes=["write"], severity=None, criticality=None
    ...     ),
    ...     DependencyCall(
    ...         library="pandas", target="read_csv", modes=["read"], severity=None, criticality=None
    ...     ),
    ... ]
    >>> grouped = group_calls_by_library(calls)
    >>> len(grouped["requests"])
    2
    >>> len(grouped["pandas"])
    1
    """
    grouped: dict[str, list[DependencyCall]] = defaultdict(list)
    for call in calls:
        grouped[call.library].append(call)
    return dict(grouped)


__all__ = [
    "DependencyCall",
    "DependencyCallVisitor",
    "build_alias_map",
    "build_alias_maps",
    "group_calls_by_library",
]
