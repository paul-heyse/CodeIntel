"""Tests for AST helper lookups used in sample repo builders."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

from tests._helpers.assertions.expectation_assertions import (
    expect_in,
    expect_is_instance,
)
from tests._helpers.graphs import build_ast_map

if TYPE_CHECKING:
    from pathlib import Path


def test_build_ast_map_handles_functions_and_classes(tmp_path: Path) -> None:
    """build_ast_map should resolve both functions and classes."""
    source = "\n".join(
        [
            "def foo():",
            "    return 1",
            "",
            "class MyClass:",
            "    def method(self):",
            "        return 2",
        ]
    )
    module_path = tmp_path / "mod.py"
    module_path.write_text(source, encoding="utf-8")

    paths = {"mod": module_path}
    goids = {"foo": 1, "MyClass": 2}
    ast_map = build_ast_map(
        paths,
        goids,
        tmp_path,
        target_names={"mod": ("foo", "MyClass")},
    )

    expect_in(goids["foo"], ast_map, label="foo goid present")
    expect_in(goids["MyClass"], ast_map, label="class goid present")
    expect_is_instance(ast_map[goids["MyClass"]].node, ast.ClassDef, label="class node")
    expect_is_instance(ast_map[goids["foo"]].node, ast.FunctionDef, label="function node")


def test_build_ast_map_raises_for_missing_target(tmp_path: Path) -> None:
    """Missing targets should surface clear errors."""
    module_path = tmp_path / "mod.py"
    module_path.write_text("def foo():\n    return 1\n", encoding="utf-8")

    paths = {"mod": module_path}
    goids = {"foo": 1}
    # target_names requests a non-existent function
    with pytest.raises(ValueError, match="Function bar not found"):
        build_ast_map(paths, goids, tmp_path, target_names={"mod": ("bar",)})
