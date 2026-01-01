"""Extraction utilities for per-function AST feature vectors."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.ast_features.patterns import DEFAULT_PATTERNS
from codeintel.build.analytics.utilities.ast import resolve_call_target, safe_unparse
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from codeintel.build.analytics.ast_features.patterns import AstFeaturePatterns
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.build.analytics.utilities.ast import CallTarget


def build_import_map(tree: ast.AST) -> dict[str, str]:
    """
    Build alias -> module mapping from import statements.

    This is a shared primitive for both function and test AST analyses.

    Returns
    -------
    dict[str, str]
        Mapping of alias to fully qualified module path.
    """
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    mapping[alias.asname] = alias.name
                else:
                    root = alias.name.split(".", maxsplit=1)[0]
                    mapping[root] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module = node.module
            for alias in node.names:
                if alias.asname:
                    mapping[alias.asname] = module
                else:
                    root = alias.name.split(".", maxsplit=1)[0]
                    mapping[root] = f"{module}.{alias.name}"
    return mapping


def io_flags_from_call(
    node: ast.Call,
    import_map: Mapping[str, str],
    existing: IoFlags,
    *,
    patterns: AstFeaturePatterns,
) -> IoFlags:
    """
    Update IoFlags based on a single Call node.

    Returns
    -------
    IoFlags
        Updated IO flags reflecting the call target.
    """
    func = node.func
    root_name: str | None = None
    attr: str | None = None
    if isinstance(func, ast.Name):
        root_name = func.id
        attr = func.id
    elif isinstance(func, ast.Attribute):
        attr = func.attr
        value = func.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name):
            root_name = value.id

    if root_name is None:
        return existing

    module = import_map.get(root_name, root_name)
    module_root = module.split(".", maxsplit=1)[0]
    attr_lower = attr.lower() if attr is not None else None

    uses_network = existing.uses_network
    uses_db = existing.uses_db
    uses_filesystem = existing.uses_filesystem
    uses_subprocess = existing.uses_subprocess

    network_spec = patterns.io_spec["network"]
    db_spec = patterns.io_spec["db"]
    filesystem_spec = patterns.io_spec["filesystem"]
    subprocess_spec = patterns.io_spec["subprocess"]

    if module_root in network_spec["libs"] or (
        attr_lower is not None and attr_lower in network_spec["funcs"]
    ):
        uses_network = True
    if module_root in db_spec["libs"] or (
        attr_lower is not None and attr_lower in db_spec["funcs"]
    ):
        uses_db = True
    if module_root in filesystem_spec["libs"] or (
        attr_lower is not None and attr_lower in filesystem_spec["funcs"]
    ):
        uses_filesystem = True
    if module_root in subprocess_spec["libs"] or (
        attr_lower is not None and attr_lower in subprocess_spec["funcs"]
    ):
        uses_subprocess = True

    return IoFlags(
        uses_network=uses_network,
        uses_db=uses_db,
        uses_filesystem=uses_filesystem,
        uses_subprocess=uses_subprocess,
    )


@dataclass
class _FunctionFeatureState:
    decorators: list[str]
    libraries_used: set[str]
    io_flags: IoFlags
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool
    http_client_libs: set[str]
    http_server_libs: set[str]
    db_libs: set[str]
    message_libs: set[str]
    config_read_count: int
    feature_flag_count: int


class FunctionFeatureVisitor(ast.NodeVisitor):
    """
    Walk a function node and compute semantic features.

    Uses import_map + patterns; should be run inside a module-level context.
    """

    def __init__(
        self,
        import_map: Mapping[str, str],
        patterns: AstFeaturePatterns,
    ) -> None:
        self.import_map = dict(import_map)
        self.patterns = patterns
        self.state = _FunctionFeatureState(
            decorators=[],
            libraries_used=set(),
            io_flags=IoFlags(),
            uses_concurrency_lib=False,
            uses_threading=False,
            uses_asyncio_lib=False,
            http_client_libs=set(),
            http_server_libs=set(),
            db_libs=set(),
            message_libs=set(),
            config_read_count=0,
            feature_flag_count=0,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            dec_str = safe_unparse(decorator)
            if dec_str:
                self.state.decorators.append(dec_str)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            dec_str = safe_unparse(decorator)
            if dec_str:
                self.state.decorators.append(dec_str)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = resolve_call_target(node.func, self.import_map)
        self._record_library_use(target.library)
        self._update_io_flags(node)
        lib_root = self._library_root(target.library)
        if lib_root is None:
            self.generic_visit(node)
            return
        self._update_concurrency_flags(lib_root)
        self._classify_frameworks(lib_root)
        self._update_config_flags(target)
        self.generic_visit(node)

    def _record_library_use(self, library: str | None) -> None:
        if library:
            self.state.libraries_used.add(library)

    def _update_io_flags(self, node: ast.Call) -> None:
        self.state.io_flags = io_flags_from_call(
            node,
            self.import_map,
            self.state.io_flags,
            patterns=self.patterns,
        )

    @staticmethod
    def _library_root(library: str | None) -> str | None:
        if library is None:
            return None
        return library.split(".", maxsplit=1)[0]

    def _update_concurrency_flags(self, lib_root: str) -> None:
        if lib_root in self.patterns.concurrency_libs:
            self.state.uses_concurrency_lib = True
        if lib_root == "threading":
            self.state.uses_threading = True
        if lib_root == "asyncio":
            self.state.uses_asyncio_lib = True

    def _classify_frameworks(self, lib_root: str) -> None:
        if lib_root in self.patterns.http_client_libs:
            self.state.http_client_libs.add(lib_root)
        if lib_root in self.patterns.http_server_libs:
            self.state.http_server_libs.add(lib_root)
        if lib_root in self.patterns.db_libs:
            self.state.db_libs.add(lib_root)
        if lib_root in self.patterns.message_libs:
            self.state.message_libs.add(lib_root)

    def _update_config_flags(self, target: CallTarget) -> None:
        dotted = target.attribute or ""
        if target.base and target.attribute:
            dotted = f"{target.base}.{target.attribute}"
        if "feature_flag" in dotted or ".flag(" in dotted:
            self.state.feature_flag_count += 1
        if ".config" in dotted or ".settings" in dotted:
            self.state.config_read_count += 1


def compute_function_features(
    fn: FunctionAst,
    *,
    repo_root: Path | None = None,
    patterns: AstFeaturePatterns = DEFAULT_PATTERNS,
) -> FunctionAstFeatures:
    """
    Compute FunctionAstFeatures from a FunctionAst instance.

    Returns
    -------
    FunctionAstFeatures
        Derived feature vector for the provided function AST.
    """
    if repo_root is not None:
        module_path = (repo_root / fn.rel_path).resolve()
        parsed = parse_python_module(module_path)
        if parsed is None:
            module_tree = ast.parse("".join(fn.lines), filename=str(module_path))
        else:
            _, module_tree = parsed
    else:
        module_tree = ast.parse("".join(fn.lines), filename=fn.rel_path)

    import_map = build_import_map(module_tree)

    visitor = FunctionFeatureVisitor(import_map=import_map, patterns=patterns)
    visitor.visit(fn.node)

    state = visitor.state
    decorators = tuple(state.decorators)

    return FunctionAstFeatures(
        goid=fn.goid,
        rel_path=fn.rel_path,
        qualname=fn.qualname,
        is_async=isinstance(fn.node, ast.AsyncFunctionDef),
        decorators=decorators,
        imports=import_map,
        libraries_used=frozenset(state.libraries_used),
        io_flags=state.io_flags,
        uses_concurrency_lib=state.uses_concurrency_lib,
        uses_threading=state.uses_threading,
        uses_asyncio_lib=state.uses_asyncio_lib,
        http_client_libs=frozenset(state.http_client_libs),
        http_server_libs=frozenset(state.http_server_libs),
        db_libs=frozenset(state.db_libs),
        message_libs=frozenset(state.message_libs),
        config_read_count=state.config_read_count,
        feature_flag_count=state.feature_flag_count,
        extra={},
    )


__all__ = [
    "build_import_map",
    "compute_function_features",
    "io_flags_from_call",
]
