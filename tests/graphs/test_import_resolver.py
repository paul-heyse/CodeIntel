"""Unit tests for shared import resolution helpers."""

from __future__ import annotations

import libcst as cst
import pytest

from codeintel.graphs.import_resolver import collect_aliases, collect_import_edges


def test_collect_aliases_import_and_from_import() -> None:
    """Alias collector maps asnames and default names to targets."""
    source = "\n".join(
        [
            "import pkg.mod as m",
            "import pkg.other",
            "from pkg.sub import foo as bar",
            "from pkg.sub import baz",
        ]
    )
    module = cst.parse_module(source)
    aliases = collect_aliases(module)
    if aliases.get("m") != "pkg.mod":
        pytest.fail("Alias m did not resolve to pkg.mod")
    if aliases.get("other") != "pkg.other":
        pytest.fail("Alias other did not resolve to pkg.other")
    if aliases.get("bar") != "pkg.sub.foo":
        pytest.fail("Alias bar did not resolve to pkg.sub.foo")
    if aliases.get("baz") != "pkg.sub.baz":
        pytest.fail("Alias baz did not resolve to pkg.sub.baz")


def test_collect_import_edges_relative_resolution() -> None:
    """Import edges include relative-from resolved to the current package."""
    source = "\n".join(
        [
            "import os",
            "from .sub import helper",
            "from pkg.external import thing",
        ]
    )
    module = cst.parse_module(source)
    edges = collect_import_edges("pkg.module", module)
    if ("pkg.module", "os") not in edges:
        pytest.fail("Expected os import edge")
    if ("pkg.module", "pkg.sub") not in edges:
        pytest.fail("Expected relative import edge to pkg.sub")
    if ("pkg.module", "pkg.external") not in edges:
        pytest.fail("Expected absolute import edge to pkg.external")


def test_collect_import_edges_deep_relative_and_multi_targets() -> None:
    """Relative imports with multiple dots and bare from-imports resolve to parent packages."""
    source = "\n".join(
        [
            "from ..utils import helpers",
            "from . import sibling, other as alias_other",
            "from ..subpackage.child import leaf",
        ]
    )
    module = cst.parse_module(source)
    edges = collect_import_edges("pkg.subpkg.module", module)
    expected = {
        ("pkg.subpkg.module", "pkg.utils"),
        ("pkg.subpkg.module", "pkg.subpkg.sibling"),
        ("pkg.subpkg.module", "pkg.subpkg.other"),
        ("pkg.subpkg.module", "pkg.subpackage.child"),
    }
    missing = expected.difference(edges)
    if missing:
        pytest.fail(f"Missing expected import edges: {missing}")


def test_collect_aliases_handles_from_import_aliases_and_packages() -> None:
    """Aliases cover from-import renames and default package names."""
    source = "\n".join(
        [
            "from pkg import sub as alias_sub",
            "from pkg.mod import thing",
            "import pkg.another as alt",
        ]
    )
    module = cst.parse_module(source)
    aliases = collect_aliases(module)
    if aliases.get("alias_sub") != "pkg.sub":
        pytest.fail("alias_sub did not map to pkg.sub")
    if aliases.get("thing") != "pkg.mod.thing":
        pytest.fail("thing did not map to pkg.mod.thing")
    if aliases.get("alt") != "pkg.another":
        pytest.fail("alt did not map to pkg.another")
