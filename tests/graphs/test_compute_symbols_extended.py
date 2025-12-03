"""Extended tests for symbol computation module.

This module provides additional test coverage for the symbols module
from `codeintel.graphs.compute.symbols`, including:

- SymbolOccurrence role detection
- Definition map building
- Use edge construction
- Same file/module detection
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Final

import pytest

from codeintel.graphs.compute.symbols import (
    SymbolOccurrence,
    SymbolUseEdge,
    SymbolUseRow,
    build_def_map,
    build_use_edges,
)
from tests._helpers.frozen_test import try_setattr

# Constants - SCIP Role Bitmasks
ROLE_DEFINITION: Final = 1
ROLE_REFERENCE: Final = 2
ROLE_CALL: Final = 4
ROLE_IMPORT: Final = 8
DEF_MAP_EXPECTED_COUNT: Final = 2
USE_EDGE_COUNT: Final = 3
SYMBOL_DEF_GOID: Final = 12345
SYMBOL_USE_GOID: Final = 67890
ROW_DEF_GOID: Final = 111
ROW_USE_GOID: Final = 222


def test_symbol_occurrence_is_definition() -> None:
    """SymbolOccurrence correctly identifies definitions."""
    occ = SymbolOccurrence(
        symbol="MyClass",
        rel_path="src/module.py",
        line=10,
        roles=ROLE_DEFINITION,
    )

    assert occ.is_definition
    assert not occ.is_reference


def test_symbol_occurrence_is_reference() -> None:
    """SymbolOccurrence correctly identifies references."""
    occ = SymbolOccurrence(
        symbol="my_function",
        rel_path="src/caller.py",
        line=25,
        roles=ROLE_REFERENCE,
    )

    assert not occ.is_definition
    assert occ.is_reference


def test_symbol_occurrence_is_call() -> None:
    """SymbolOccurrence recognizes call references."""
    occ = SymbolOccurrence(
        symbol="helper",
        rel_path="src/main.py",
        line=50,
        roles=ROLE_CALL,
    )

    assert not occ.is_definition
    assert occ.is_reference


def test_symbol_occurrence_is_import() -> None:
    """SymbolOccurrence recognizes import references."""
    occ = SymbolOccurrence(
        symbol="os.path",
        rel_path="src/utils.py",
        line=1,
        roles=ROLE_IMPORT,
    )

    assert occ.is_reference


def test_symbol_occurrence_combined_roles() -> None:
    """SymbolOccurrence handles combined role bits."""
    occ = SymbolOccurrence(
        symbol="reexported",
        rel_path="src/__init__.py",
        line=5,
        roles=ROLE_DEFINITION | ROLE_IMPORT,
    )

    assert occ.is_definition
    assert occ.is_reference


def test_symbol_occurrence_no_roles() -> None:
    """SymbolOccurrence with no roles is neither def nor ref."""
    occ = SymbolOccurrence(
        symbol="unknown",
        rel_path="src/unknown.py",
        line=1,
        roles=0,
    )

    assert not occ.is_definition
    assert not occ.is_reference


def test_symbol_occurrence_frozen() -> None:
    """SymbolOccurrence is frozen (immutable)."""
    occ = SymbolOccurrence(
        symbol="test",
        rel_path="test.py",
        line=1,
        roles=1,
    )

    with pytest.raises(FrozenInstanceError):
        try_setattr(occ, "symbol", "changed")


def test_build_def_map_simple() -> None:
    """Build definition map from simple occurrences."""
    occurrences = [
        SymbolOccurrence("ClassA", "src/a.py", 10, ROLE_DEFINITION),
        SymbolOccurrence("ClassB", "src/b.py", 20, ROLE_DEFINITION),
        SymbolOccurrence("ClassA", "src/c.py", 5, ROLE_REFERENCE),
    ]

    def_map = build_def_map(occurrences)

    assert def_map["ClassA"] == "src/a.py"
    assert def_map["ClassB"] == "src/b.py"
    assert len(def_map) == DEF_MAP_EXPECTED_COUNT  # Only definitions


def test_build_def_map_first_definition_wins() -> None:
    """First definition for a symbol is used."""
    occurrences = [
        SymbolOccurrence("Symbol", "first.py", 1, ROLE_DEFINITION),
        SymbolOccurrence("Symbol", "second.py", 2, ROLE_DEFINITION),
    ]

    def_map = build_def_map(occurrences)

    assert def_map["Symbol"] == "first.py"


def test_build_def_map_ignores_references() -> None:
    """Definition map only includes definitions."""
    occurrences = [
        SymbolOccurrence("Used", "user.py", 10, ROLE_REFERENCE),
        SymbolOccurrence("Called", "caller.py", 20, ROLE_CALL),
    ]

    def_map = build_def_map(occurrences)

    assert def_map == {}


def test_build_def_map_empty() -> None:
    """Definition map from empty occurrences is empty."""
    def_map = build_def_map([])

    assert def_map == {}


def test_build_use_edges_simple() -> None:
    """Build use edges from simple occurrences."""
    occurrences = [
        SymbolOccurrence("func", "src/def.py", 10, ROLE_DEFINITION),
        SymbolOccurrence("func", "src/use.py", 20, ROLE_REFERENCE),
    ]
    def_map = {"func": "src/def.py"}
    module_by_path = {
        "src/def.py": "mypackage.def",
        "src/use.py": "mypackage.use",
    }

    edges = build_use_edges(occurrences, def_map, module_by_path)

    assert len(edges) == 1
    edge = edges[0]
    assert edge.symbol == "func"
    assert edge.def_path == "src/def.py"
    assert edge.use_path == "src/use.py"
    assert not edge.same_file
    assert not edge.same_module


def test_build_use_edges_same_file() -> None:
    """Build use edges detects same file usage."""
    occurrences = [
        SymbolOccurrence("local", "src/module.py", 10, ROLE_DEFINITION),
        SymbolOccurrence("local", "src/module.py", 50, ROLE_REFERENCE),
    ]
    def_map = {"local": "src/module.py"}
    module_by_path = {"src/module.py": "mypackage.module"}

    edges = build_use_edges(occurrences, def_map, module_by_path)

    assert len(edges) == 1
    assert edges[0].same_file


def test_build_use_edges_same_module() -> None:
    """Build use edges detects same module usage."""
    occurrences = [
        SymbolOccurrence("shared", "src/pkg/a.py", 10, ROLE_DEFINITION),
        SymbolOccurrence("shared", "src/pkg/b.py", 20, ROLE_REFERENCE),
    ]
    def_map = {"shared": "src/pkg/a.py"}
    module_by_path = {
        "src/pkg/a.py": "mypackage.pkg",
        "src/pkg/b.py": "mypackage.pkg",
    }

    edges = build_use_edges(occurrences, def_map, module_by_path)

    assert len(edges) == 1
    assert edges[0].same_module


def test_build_use_edges_no_definition_skipped() -> None:
    """References without definitions are skipped."""
    occurrences = [
        SymbolOccurrence("undefined", "src/use.py", 10, ROLE_REFERENCE),
    ]
    def_map = {}  # No definitions
    module_by_path = {"src/use.py": "mypackage"}

    edges = build_use_edges(occurrences, def_map, module_by_path)

    assert edges == []


def test_build_use_edges_multiple_uses() -> None:
    """Multiple uses of same symbol create multiple edges."""
    occurrences = [
        SymbolOccurrence("helper", "src/helper.py", 5, ROLE_DEFINITION),
        SymbolOccurrence("helper", "src/a.py", 10, ROLE_REFERENCE),
        SymbolOccurrence("helper", "src/b.py", 15, ROLE_REFERENCE),
        SymbolOccurrence("helper", "src/c.py", 20, ROLE_CALL),
    ]
    def_map = {"helper": "src/helper.py"}
    module_by_path = {
        "src/helper.py": "mypackage.helper",
        "src/a.py": "mypackage.a",
        "src/b.py": "mypackage.b",
        "src/c.py": "mypackage.c",
    }

    edges = build_use_edges(occurrences, def_map, module_by_path)

    assert len(edges) == USE_EDGE_COUNT
    use_paths = {e.use_path for e in edges}
    assert use_paths == {"src/a.py", "src/b.py", "src/c.py"}


def test_build_use_edges_empty() -> None:
    """Empty occurrences produce no edges."""
    edges = build_use_edges([], {}, {})

    assert edges == []


def test_symbol_use_edge_attributes() -> None:
    """SymbolUseEdge has all expected attributes."""
    edge = SymbolUseEdge(
        symbol="MyClass",
        def_path="src/model.py",
        use_path="src/service.py",
        same_file=False,
        same_module=True,
        def_goid=SYMBOL_DEF_GOID,
        use_goid=SYMBOL_USE_GOID,
    )

    assert edge.symbol == "MyClass"
    assert edge.def_path == "src/model.py"
    assert edge.use_path == "src/service.py"
    assert not edge.same_file
    assert edge.same_module
    assert edge.def_goid == SYMBOL_DEF_GOID
    assert edge.use_goid == SYMBOL_USE_GOID


def test_symbol_use_edge_optional_goids() -> None:
    """SymbolUseEdge goids are optional."""
    edge = SymbolUseEdge(
        symbol="func",
        def_path="def.py",
        use_path="use.py",
        same_file=False,
        same_module=False,
    )

    assert edge.def_goid is None
    assert edge.use_goid is None


def test_symbol_use_edge_frozen() -> None:
    """SymbolUseEdge is frozen (immutable)."""
    edge = SymbolUseEdge(
        symbol="test",
        def_path="def.py",
        use_path="use.py",
        same_file=False,
        same_module=False,
    )

    with pytest.raises(FrozenInstanceError):
        try_setattr(edge, "symbol", "changed")


def test_symbol_use_row_attributes() -> None:
    """SymbolUseRow has all expected attributes."""
    row = SymbolUseRow(
        symbol="helper",
        def_path="src/helper.py",
        use_path="src/main.py",
        same_file=False,
        same_module=False,
        def_goid_h128=ROW_DEF_GOID,
        use_goid_h128=ROW_USE_GOID,
    )

    assert row.symbol == "helper"
    assert row.def_goid_h128 == ROW_DEF_GOID
    assert row.use_goid_h128 == ROW_USE_GOID


def test_symbol_use_row_frozen() -> None:
    """SymbolUseRow is frozen (immutable)."""
    row = SymbolUseRow(
        symbol="test",
        def_path="def.py",
        use_path="use.py",
        same_file=False,
        same_module=False,
        def_goid_h128=None,
        use_goid_h128=None,
    )

    with pytest.raises(FrozenInstanceError):
        try_setattr(row, "symbol", "changed")
