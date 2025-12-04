"""Extended integration tests for import graph builder plugin.

This module provides integration test coverage for the import graph
builder plugin from `codeintel.graphs.plugins.builders.import_graph`,
focusing on `build_import_graph()` - the main orchestration function.

Tests exercise:
- Real file I/O with actual Python files
- Real LibCST parsing
- Real DuckDB persistence
- SCC and layer computation through NetworkX
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from codeintel.config import ImportGraphStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.plugins.builders.import_graph import build_import_graph
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_builder import QueryBuilder, SafeTable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO: Final = "test/import-graph"
COMMIT: Final = "import123"
LANGUAGE: Final = "python"

# Table references
TABLE_IMPORT_MODULES: Final = SafeTable("graph.import_modules")
TABLE_IMPORT_EDGES: Final = SafeTable("graph.import_graph_edges")

# Expected counts
EXPECTED_ZERO: Final = 0
EXPECTED_ONE: Final = 1
EXPECTED_TWO: Final = 2
EXPECTED_THREE: Final = 3
EXPECTED_FOUR: Final = 4

# Layer values
LAYER_ZERO: Final = 0
LAYER_ONE: Final = 1
LAYER_TWO: Final = 2


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _create_python_file(tmp_path: Path, rel_path: str, content: str) -> None:
    """Create a Python file at the given relative path.

    Parameters
    ----------
    tmp_path
        Root directory.
    rel_path
        Relative path for the file.
    content
        Python source content.
    """
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def _seed_module_catalog(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    modules: dict[str, str],
) -> None:
    """Seed core.modules table with module->path mappings.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit hash.
    modules
        Mapping of module name to relative path.
    """
    service = IngestStorageService.from_gateway(gateway)
    rows = [
        (module_name, path, repo, commit, LANGUAGE, "[]", "[]")
        for module_name, path in modules.items()
    ]
    service.run_batch(
        "core.modules",
        rows,
        delete_params=[repo, commit],
        scope=f"{repo}@{commit}",
    )


def _make_import_graph_config(
    repo: str,
    commit: str,
    repo_root: Path,
) -> ImportGraphStepConfig:
    """Create config for build_import_graph.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit hash.
    repo_root
        Repository root path.

    Returns
    -------
    ImportGraphStepConfig
        Configured step config.
    """
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
    return ImportGraphStepConfig(snapshot=snapshot)


def _count_import_modules(gateway: StorageGateway, repo: str, commit: str) -> int:
    """Count rows in graph.import_modules for given repo/commit.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    int
        Row count.
    """
    query, params = QueryBuilder.count(
        TABLE_IMPORT_MODULES,
        where={"repo": repo, "commit": commit},
    )
    result = gateway.con.execute(query, params).fetchone()
    return int(result[0]) if result else 0


def _count_import_edges(gateway: StorageGateway, repo: str, commit: str) -> int:
    """Count rows in graph.import_graph_edges for given repo/commit.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    int
        Row count.
    """
    query, params = QueryBuilder.count(
        TABLE_IMPORT_EDGES,
        where={"repo": repo, "commit": commit},
    )
    result = gateway.con.execute(query, params).fetchone()
    return int(result[0]) if result else 0


def _get_module_layers(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Get module -> layer mapping from graph.import_modules.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    dict[str, int]
        Module name to layer mapping.
    """
    rows = gateway.con.execute(
        """
        SELECT module, layer
        FROM graph.import_modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    return {str(row[0]): int(row[1]) if row[1] is not None else 0 for row in rows}


def _get_module_scc_ids(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Get module -> scc_id mapping from graph.import_modules.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    dict[str, int]
        Module name to SCC ID mapping.
    """
    rows = gateway.con.execute(
        """
        SELECT module, scc_id
        FROM graph.import_modules
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


# ---------------------------------------------------------------------------
# Section 1: Basic Integration Tests
# ---------------------------------------------------------------------------


def test_build_import_graph_empty_catalog(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Empty module catalog produces no rows."""
    # Don't seed any modules
    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)

    build_import_graph(graph_gateway, cfg)

    assert _count_import_modules(graph_gateway, REPO, COMMIT) == EXPECTED_ZERO
    assert _count_import_edges(graph_gateway, REPO, COMMIT) == EXPECTED_ZERO


def test_build_import_graph_single_module_no_imports(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Single module with no imports produces module row only."""
    # Create a module with no imports
    _create_python_file(
        tmp_path,
        "pkg/standalone.py",
        '''"""Standalone module with no imports."""

def hello() -> str:
    """Return greeting."""
    return "hello"
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {"pkg.standalone": "pkg/standalone.py"},
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # Should have one module row, no edges
    assert _count_import_modules(graph_gateway, REPO, COMMIT) == EXPECTED_ONE
    assert _count_import_edges(graph_gateway, REPO, COMMIT) == EXPECTED_ZERO


def test_build_import_graph_simple_chain(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Linear A->B->C chain produces correct edges and layers."""
    # Create chain: module_a imports module_b imports module_c
    # Using `import pkg.module_b` style to get direct edges
    _create_python_file(
        tmp_path,
        "pkg/module_a.py",
        '''"""Module A imports B."""
import pkg.module_b
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/module_b.py",
        '''"""Module B imports C."""
import pkg.module_c
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/module_c.py",
        '''"""Module C is the leaf."""

def leaf() -> int:
    """Return a value."""
    return 42
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {
            "pkg.module_a": "pkg/module_a.py",
            "pkg.module_b": "pkg/module_b.py",
            "pkg.module_c": "pkg/module_c.py",
        },
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # Should have at least 3 seeded modules (plus any added from edges)
    module_count = _count_import_modules(graph_gateway, REPO, COMMIT)
    assert module_count >= EXPECTED_THREE

    # Verify layer ordering: A (0) -> B (1) -> C (2)
    layers = _get_module_layers(graph_gateway, REPO, COMMIT)
    assert layers["pkg.module_a"] < layers["pkg.module_b"]
    assert layers["pkg.module_b"] < layers["pkg.module_c"]


# ---------------------------------------------------------------------------
# Section 2: Complex Graph Tests
# ---------------------------------------------------------------------------


def test_build_import_graph_cycle_detection(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Cycle A->B->C->A produces single SCC for all cycle members."""
    # Create cycle: A -> B -> C -> A
    # Using `import pkg.module` style for direct edges
    _create_python_file(
        tmp_path,
        "pkg/cycle_a.py",
        '''"""Cycle A imports B."""
import pkg.cycle_b
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/cycle_b.py",
        '''"""Cycle B imports C."""
import pkg.cycle_c
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/cycle_c.py",
        '''"""Cycle C imports A, completing the cycle."""
import pkg.cycle_a
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {
            "pkg.cycle_a": "pkg/cycle_a.py",
            "pkg.cycle_b": "pkg/cycle_b.py",
            "pkg.cycle_c": "pkg/cycle_c.py",
        },
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # All modules in same SCC
    scc_ids = _get_module_scc_ids(graph_gateway, REPO, COMMIT)
    assert scc_ids["pkg.cycle_a"] == scc_ids["pkg.cycle_b"]
    assert scc_ids["pkg.cycle_b"] == scc_ids["pkg.cycle_c"]

    # All modules have same layer
    layers = _get_module_layers(graph_gateway, REPO, COMMIT)
    assert layers["pkg.cycle_a"] == layers["pkg.cycle_b"]
    assert layers["pkg.cycle_b"] == layers["pkg.cycle_c"]


def test_build_import_graph_diamond_pattern(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Diamond pattern A->{B,C}->D produces correct layers."""
    # Create diamond: A imports B and C, both import D
    # Using `import pkg.module` style for direct edges
    _create_python_file(
        tmp_path,
        "pkg/diamond_a.py",
        '''"""Diamond A imports B and C."""
import pkg.diamond_b
import pkg.diamond_c
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/diamond_b.py",
        '''"""Diamond B imports D."""
import pkg.diamond_d
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/diamond_c.py",
        '''"""Diamond C imports D."""
import pkg.diamond_d
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/diamond_d.py",
        '''"""Diamond D is the leaf."""

def leaf() -> int:
    """Return value."""
    return 1
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {
            "pkg.diamond_a": "pkg/diamond_a.py",
            "pkg.diamond_b": "pkg/diamond_b.py",
            "pkg.diamond_c": "pkg/diamond_c.py",
            "pkg.diamond_d": "pkg/diamond_d.py",
        },
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # At least 4 seeded modules
    module_count = _count_import_modules(graph_gateway, REPO, COMMIT)
    assert module_count >= EXPECTED_FOUR

    # Verify layer structure
    layers = _get_module_layers(graph_gateway, REPO, COMMIT)
    # A at layer 0 (root)
    assert layers["pkg.diamond_a"] == LAYER_ZERO
    # B and C at same layer (both downstream of A)
    assert layers["pkg.diamond_b"] == layers["pkg.diamond_c"]
    assert layers["pkg.diamond_b"] == LAYER_ONE
    # D at deepest layer
    assert layers["pkg.diamond_d"] == LAYER_TWO


def test_build_import_graph_mixed_imports(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Handle relative, absolute, and from imports correctly."""
    # Create modules with various import styles
    _create_python_file(
        tmp_path,
        "pkg/__init__.py",
        '''"""Package init."""
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/main.py",
        '''"""Main module with mixed imports."""
import os
from . import helper
from pkg.utils import util_func
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/helper.py",
        '''"""Helper module."""

def help_me() -> str:
    """Return help."""
    return "help"
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/utils.py",
        '''"""Utils module."""

def util_func() -> int:
    """Return utility value."""
    return 99
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {
            "pkg": "pkg/__init__.py",
            "pkg.main": "pkg/main.py",
            "pkg.helper": "pkg/helper.py",
            "pkg.utils": "pkg/utils.py",
        },
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # Should have modules and edges
    module_count = _count_import_modules(graph_gateway, REPO, COMMIT)
    edge_count = _count_import_edges(graph_gateway, REPO, COMMIT)

    # At least the 4 seeded modules plus possibly os
    assert module_count >= EXPECTED_FOUR
    # At least edges from main to helper and utils
    assert edge_count >= EXPECTED_TWO


# ---------------------------------------------------------------------------
# Section 3: Error Handling Tests
# ---------------------------------------------------------------------------


def test_build_import_graph_missing_file_skipped(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Missing file is skipped, other modules still processed."""
    # Create only one of two modules
    _create_python_file(
        tmp_path,
        "pkg/exists.py",
        '''"""Module that exists."""

def present() -> bool:
    """Return True."""
    return True
''',
    )
    # Seed both modules but only one file exists
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {
            "pkg.exists": "pkg/exists.py",
            "pkg.missing": "pkg/missing.py",  # File doesn't exist
        },
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # Should still process the existing module
    assert _count_import_modules(graph_gateway, REPO, COMMIT) >= EXPECTED_ONE


def test_build_import_graph_parse_error_skipped(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Syntax error in file is skipped, other modules still processed."""
    # Create a valid module
    _create_python_file(
        tmp_path,
        "pkg/valid.py",
        '''"""Valid module."""

def valid() -> str:
    """Return valid."""
    return "valid"
''',
    )
    # Create an invalid module with syntax error
    _create_python_file(
        tmp_path,
        "pkg/invalid.py",
        '''"""Invalid module with syntax error."""
def broken(
    # Missing closing parenthesis and body
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {
            "pkg.valid": "pkg/valid.py",
            "pkg.invalid": "pkg/invalid.py",
        },
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # Should still process the valid module
    assert _count_import_modules(graph_gateway, REPO, COMMIT) >= EXPECTED_ONE


# ---------------------------------------------------------------------------
# Section 4: Persistence Verification Tests
# ---------------------------------------------------------------------------


def test_build_import_graph_persists_modules_table(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Verify rows written to graph.import_modules with correct schema."""
    _create_python_file(
        tmp_path,
        "pkg/persist_test.py",
        '''"""Module for persistence test."""
import os
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {"pkg.persist_test": "pkg/persist_test.py"},
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # Query and verify schema
    rows = graph_gateway.con.execute(
        """
        SELECT repo, commit, module, scc_id, component_size, layer, cycle_group
        FROM graph.import_modules
        WHERE repo = ? AND commit = ?
        """,
        [REPO, COMMIT],
    ).fetchall()

    assert len(rows) >= EXPECTED_ONE

    # Find our module
    our_row = next((r for r in rows if r[2] == "pkg.persist_test"), None)
    assert our_row is not None

    # Verify schema fields are populated
    repo_val, commit_val, module_val, scc_id, comp_size, _layer, _cycle_group = our_row
    assert repo_val == REPO
    assert commit_val == COMMIT
    assert module_val == "pkg.persist_test"
    assert isinstance(scc_id, int)
    assert isinstance(comp_size, int)
    assert comp_size >= EXPECTED_ONE


def test_build_import_graph_persists_edges_table(
    graph_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Verify rows written to graph.import_graph_edges with fan-in/fan-out."""
    # Create modules with edges to verify fan metrics
    # Using `import pkg.target` style for direct edge
    _create_python_file(
        tmp_path,
        "pkg/source.py",
        '''"""Source module imports target."""
import pkg.target
''',
    )
    _create_python_file(
        tmp_path,
        "pkg/target.py",
        '''"""Target module."""

def target_func() -> int:
    """Return value."""
    return 1
''',
    )
    _seed_module_catalog(
        graph_gateway,
        REPO,
        COMMIT,
        {
            "pkg.source": "pkg/source.py",
            "pkg.target": "pkg/target.py",
        },
    )

    cfg = _make_import_graph_config(REPO, COMMIT, tmp_path)
    build_import_graph(graph_gateway, cfg)

    # Query and verify edge schema
    rows = graph_gateway.con.execute(
        """
        SELECT repo, commit, src_module, dst_module, src_fan_out, dst_fan_in,
               cycle_group, module_layer
        FROM graph.import_graph_edges
        WHERE repo = ? AND commit = ?
        """,
        [REPO, COMMIT],
    ).fetchall()

    assert len(rows) >= EXPECTED_ONE

    # Find our edge
    our_edge = next(
        (r for r in rows if r[2] == "pkg.source" and r[3] == "pkg.target"),
        None,
    )
    assert our_edge is not None

    # Verify schema fields
    (
        repo_val,
        commit_val,
        src_mod,
        dst_mod,
        fan_out,
        fan_in,
        _cycle_grp,
        _mod_layer,
    ) = our_edge
    assert repo_val == REPO
    assert commit_val == COMMIT
    assert src_mod == "pkg.source"
    assert dst_mod == "pkg.target"
    assert isinstance(fan_out, int)
    assert fan_out >= EXPECTED_ONE  # source has at least 1 outgoing edge
    assert isinstance(fan_in, int)
    assert fan_in >= EXPECTED_ONE  # target has at least 1 incoming edge
