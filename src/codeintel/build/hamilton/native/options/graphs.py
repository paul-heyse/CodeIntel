"""Configuration options for graph Hamilton native modules.

These dataclasses configure the behavior of graph-building targets such as
GOID construction, call graphs, import graphs, and control/data flow graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CallGraphOptions",
    "CfgDfgOptions",
    "CpgOptions",
    "GoidBuilderOptions",
    "ImportGraphOptions",
    "SymbolUsesOptions",
]


@dataclass(frozen=True)
class GoidBuilderOptions:
    """Configuration options for GOID construction.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    include_tests
        Whether to include test files.
    include_private
        Whether to include symbols whose names start with an underscore.
    """

    scope_paths: list[str] | None = None
    include_tests: bool = True
    include_private: bool = True


@dataclass(frozen=True)
class CallGraphOptions:
    """Configuration options for call graph construction.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    max_edges_per_file
        Maximum number of edges to emit per file.
    include_external_calls
        Whether to include calls to external (non-local) functions.
    resolve_imports
        Whether to resolve import statements to target modules.
    use_libcst
        Whether to use LibCST for parsing (vs AST).
    """

    scope_paths: list[str] | None = None
    max_edges_per_file: int = 10000
    include_external_calls: bool = True
    resolve_imports: bool = True
    use_libcst: bool = True


@dataclass(frozen=True)
class ImportGraphOptions:
    """Configuration options for import graph construction.

    Attributes
    ----------
    scope_paths
        If set, only process files within these paths.
    include_stdlib
        Whether to include stdlib imports in the graph.
    include_third_party
        Whether to include third-party imports.
    resolve_dynamic
        Whether to attempt resolution of dynamic imports.
    """

    scope_paths: list[str] | None = None
    include_stdlib: bool = False
    include_third_party: bool = True
    resolve_dynamic: bool = False


@dataclass(frozen=True)
class SymbolUsesOptions:
    """Configuration options for symbol use graph construction.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    include_tests
        Whether to include test files when building symbol use edges.
    """

    scope_paths: list[str] | None = None
    include_tests: bool = True


@dataclass(frozen=True)
class CfgDfgOptions:
    """Configuration options for CFG/DFG construction.

    Attributes
    ----------
    scope_paths
        Optional prefixes to limit processing to matching files.
    include_test_files
        Whether to include test files when building graphs.
    """

    scope_paths: list[str] | None = None
    include_test_files: bool = True


@dataclass(frozen=True)
class CpgOptions:
    """Configuration options for CPG construction.

    Attributes
    ----------
    enable_reaches
        Whether to emit REACHES edges for bytecode dataflow reachability.
    """

    enable_reaches: bool = True
