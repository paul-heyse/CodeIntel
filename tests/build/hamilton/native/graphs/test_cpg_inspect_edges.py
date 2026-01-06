"""Tests for inspect-derived CPG edges."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.build.hamilton.native.graphs.cpg2.planes.overlays_inspect import (
    cpg2_edges__inspect_to_ast,
    cpg2_edges__py_inspect_unwrap,
)
from codeintel.core.serialization.payload import decode_payload

pytestmark = pytest.mark.no_runtime_env

EXPECTED_UNWRAP_EDGE_COUNT = 2


def test_inspect_to_ast_uses_source_span_match() -> None:
    """Ensure inspect->AST anchors prefer source span matches."""
    inspect_objects = pa.Table.from_pylist(
        [
            {
                "repo": "demo",
                "commit": "abc123",
                "object_id": "obj1",
                "module_name": "pkg.mod",
                "qualname": "foo",
            }
        ]
    )
    inspect_source = pa.Table.from_pylist(
        [
            {
                "object_id": "obj1",
                "file_name": "/tmp/repo/pkg/mod.py",
                "start_line": 1,
                "line_count": 3,
            }
        ]
    )
    ast_nodes = pa.Table.from_pylist(
        [
            {
                "path": "pkg/mod.py",
                "hash": "h1",
                "node_type": "FunctionDef",
                "qualname": "pkg.mod.foo",
                "lineno": 1,
                "end_lineno": 3,
                "decorator_start_line": 1,
                "decorator_end_line": 1,
            }
        ]
    )

    edges = cpg2_edges__inspect_to_ast(inspect_objects, inspect_source, ast_nodes)
    assert edges.num_rows == 1
    row = edges.to_pylist()[0]
    assert row["edge_kind"] == "INSPECT_ANCHORS_AST"
    extras = decode_payload(row.get("extras_json"))
    assert isinstance(extras, dict)
    assert extras.get("match_kind") == "SOURCE_SPAN"
    assert extras.get("match_confidence") is not None


def test_py_inspect_unwrap_edges_distinguish_decorates() -> None:
    """Ensure unwrap edges label the first hop as DECORATES."""
    unwrap_hops = pa.Table.from_pylist(
        [
            {
                "repo": "demo",
                "commit": "abc123",
                "root_object_id": "obj1",
                "hop": 0,
                "object_id": "obj1",
                "has_wrapped": True,
                "has_signature_override": False,
                "stop_reason": None,
            },
            {
                "repo": "demo",
                "commit": "abc123",
                "root_object_id": "obj1",
                "hop": 1,
                "object_id": "obj2",
                "has_wrapped": True,
                "has_signature_override": False,
                "stop_reason": None,
            },
            {
                "repo": "demo",
                "commit": "abc123",
                "root_object_id": "obj1",
                "hop": 2,
                "object_id": "obj3",
                "has_wrapped": False,
                "has_signature_override": False,
                "stop_reason": "no_wrapped",
            },
        ]
    )

    edges = cpg2_edges__py_inspect_unwrap(unwrap_hops)
    assert edges.num_rows == EXPECTED_UNWRAP_EDGE_COUNT
    kinds = {row["edge_kind"] for row in edges.to_pylist()}
    assert "DECORATES" in kinds
    assert "WRAPS" in kinds
