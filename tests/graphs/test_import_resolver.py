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
