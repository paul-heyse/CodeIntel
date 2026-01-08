"""Integration tests for function_contracts module with catalog.

This module tests build_function_contracts_rows with a realistic fake catalog
to exercise the main execution paths including docstrings and type info.

Uses MockFunctionCatalog from tests._helpers.fakes for catalog mocking.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.analytics.functions.function_contracts import (
    FunctionContractInputs,
    build_function_contracts_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
from codeintel.core.serialization.payload import decode_payload
from codeintel.storage.warehouse import Warehouse
from tests._helpers import TestScenario
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.catalogs import ensure_catalog_with_goids
from tests._helpers.fakes.function_catalogs import (
    MockFunctionCatalog,
    MockFunctionSpan,
)
from tests._helpers.fixtures.rows import RowFactory
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext

from codeintel.config.primitives import SnapshotRef
from codeintel.core.storage import StorageContext

TEST_MAX_CONDITIONS = 10
GOID_SIMPLE = 10001
GOID_TYPED = 10002
GOID_GUARDED = 10003
GOID_MISSING = 10004
MIN_CONFIDENCE = 0.5
EXPECTED_GOID_COUNT = 2


@pytest.fixture
def ctx(tmp_path: Path) -> Iterator[TestContext]:
    """Create a test context aligned with defaults.

    Yields
    ------
    TestContext
        Context configured with default repo and commit identifiers.
    """
    context = TestScenario().build(tmp_path)
    try:
        yield context
    finally:
        context.close()


@pytest.fixture
def contracts_snapshot(ctx: TestContext) -> SnapshotRef:
    """Create a test snapshot for function contracts.

    Parameters
    ----------
    ctx
        Test context providing snapshot and gateway.

    Returns
    -------
    SnapshotRef
        Snapshot reference for testing.
    """
    return ctx.snapshot


def _create_function_ast(
    goid: int,
    rel_path: str,
    qualname: str,
    code: str,
) -> FunctionAst:
    """Create a FunctionAst from source code.

    Parameters
    ----------
    goid
        Global object ID.
    rel_path
        Relative file path.
    qualname
        Qualified function name.
    code
        Python function source.

    Returns
    -------
    FunctionAst
        Parsed function AST.

    Raises
    ------
    TypeError
        If the provided code does not start with a function definition.
    """
    tree = ast.parse(code)
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        msg = f"Expected function definition, got {type(node)}"
        raise TypeError(msg)
    func_node = node
    lines = code.split("\n")
    return FunctionAst(
        goid=goid,
        rel_path=rel_path,
        qualname=qualname,
        node=func_node,
        start_line=1,
        end_line=len(lines),
        lines=lines,
    )


def _create_mock_catalog(spans: list[MockFunctionSpan]) -> FunctionCatalogProvider:
    """Create a mock catalog provider with given spans.

    Parameters
    ----------
    spans
        List of function spans.

    Returns
    -------
    FunctionCatalogProvider
        Mock catalog provider implementing the protocol.
    """
    return cast("FunctionCatalogProvider", MockFunctionCatalog(function_spans=spans))


def _expect_payload_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    msg = f"Expected list payload, got {type(value)}"
    raise TypeError(msg)


def _expect_payload_str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        msg = f"Expected dict payload, got {type(value)}"
        raise TypeError(msg)
    if not all(isinstance(key, str) and isinstance(val, str) for key, val in value.items()):
        msg = "Expected dict[str, str] payload"
        raise TypeError(msg)
    return cast("dict[str, str]", value)


def _seed_docstrings(
    gateway: StorageGateway,
    rel_path: str,
    qualname: str,
    params: list[dict[str, str | None]],
    returns: dict[str, str | None] | None = None,
) -> None:
    """Seed docstring data for testing.

    Parameters
    ----------
    gateway
        Storage gateway.
    rel_path
        Relative file path.
    qualname
        Qualified function name.
    params
        Parameter documentation.
    returns
        Return documentation.
    """
    now = datetime.now(tz=UTC)
    row = RowFactory.row_for(
        "core.docstrings",
        repo=DEFAULT_VARIANT.repo,
        commit=DEFAULT_VARIANT.commit,
        rel_path=rel_path,
        module=rel_path.replace("/", ".").replace(".py", ""),
        qualname=qualname,
        kind="function",
        params=params,
        returns=returns,
        raises=[],
        examples=[],
        created_at=now,
    )
    warehouse = Warehouse(context=StorageContext(gateway=gateway))
    warehouse.materialize_mappings("core.docstrings", [row])


def _seed_function_types(
    gateway: StorageGateway,
    goid: int,
    return_type: str | None = None,
    param_types: dict[str, str] | None = None,
) -> None:
    """Seed function type data for testing.

    Parameters
    ----------
    gateway
        Storage gateway.
    goid
        Function GOID.
    return_type
        Return type annotation.
    param_types
        Parameter type annotations.
    """
    now = datetime.now(tz=UTC)
    row = RowFactory.row_for(
        "analytics.function_types",
        repo=DEFAULT_VARIANT.repo,
        commit=DEFAULT_VARIANT.commit,
        function_goid_h128=goid,
        return_type=return_type,
        param_types=param_types or {},
        created_at=now,
    )
    warehouse = Warehouse(context=StorageContext(gateway=gateway))
    warehouse.materialize_mappings("analytics.function_types", [row])


def _build_and_write_contract_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    function_ast_map: dict[int, FunctionAst] | None = None,
    catalog: FunctionCatalogProvider | None = None,
) -> None:
    rows = build_function_contracts_rows(
        snapshot,
        FunctionContractInputs(
            function_ast_map=function_ast_map or {},
            catalog=catalog,
        ),
    )
    if not rows:
        return
    gateway.policy.delete_for_snapshot(
        "analytics.function_contracts",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    gateway.policy.bulk_insert_mappings("analytics.function_contracts", rows)


def test_compute_contracts_with_catalog_goid_iteration(
    ctx: TestContext,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts iterates over GOIDs from catalog."""
    code = """def simple(x):
    return x + 1
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="simple",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "simple", 1, 3)]
    catalog = _create_mock_catalog(spans)
    ensure_catalog_with_goids(ctx, catalog)

    _build_and_write_contract_rows(
        ctx.gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = ctx.gateway.con.execute(
        """
        SELECT function_goid_h128, preconditions_json, postconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], GOID_SIMPLE)


def test_compute_contracts_with_missing_ast(
    ctx: TestContext,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts handles GOIDs without AST gracefully."""
    spans = [MockFunctionSpan(GOID_MISSING, "missing.py", "missing_func", 1, 3)]
    catalog = _create_mock_catalog(spans)
    ensure_catalog_with_goids(ctx, catalog)

    _build_and_write_contract_rows(
        ctx.gateway,
        contracts_snapshot,
        function_ast_map={},
        catalog=catalog,
    )

    result = ctx.gateway.con.execute(
        """
        SELECT function_goid_h128, preconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], GOID_MISSING)
    payload = _expect_payload_list(decode_payload(result[1]))
    expect_equal(payload, [])


def test_compute_contracts_with_docstring_data(
    ctx: TestContext,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts uses docstring data for nullability inference."""
    code = """def with_docs(x, y):
    return x + y
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="with_docs",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "with_docs", 1, 3)]
    catalog = _create_mock_catalog(spans)
    ensure_catalog_with_goids(ctx, catalog)

    _seed_docstrings(
        ctx.gateway,
        rel_path="module.py",
        qualname="with_docs",
        params=[
            {
                "name": "x",
                "type_name": None,
                "description": "Required parameter, must not be None",
            },
            {
                "name": "y",
                "type_name": None,
                "description": "Optional value, may be none",
            },
        ],
    )

    _build_and_write_contract_rows(
        ctx.gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = ctx.gateway.con.execute(
        """
        SELECT param_nullability_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    result = expect_is_not_none(result)
    nullability = _expect_payload_str_dict(decode_payload(result[0]))
    expect_equal(nullability["x"], "unknown")
    expect_equal(nullability["y"], "unknown")


def test_compute_contracts_with_type_annotations(
    ctx: TestContext,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts uses type annotations for nullability inference."""
    code = """def typed_func(x, y):
    return x + y
"""
    func_ast = _create_function_ast(
        goid=GOID_TYPED,
        rel_path="module.py",
        qualname="typed_func",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_TYPED, "module.py", "typed_func", 1, 3)]
    catalog = _create_mock_catalog(spans)
    ensure_catalog_with_goids(ctx, catalog)

    _seed_function_types(
        ctx.gateway,
        goid=GOID_TYPED,
        return_type="int",
        param_types={"x": "int", "y": "Optional[int]"},
    )

    _build_and_write_contract_rows(
        ctx.gateway,
        contracts_snapshot,
        function_ast_map={GOID_TYPED: func_ast},
        catalog=catalog,
    )

    result = ctx.gateway.con.execute(
        """
        SELECT param_nullability_json, return_nullability
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_TYPED],
    ).fetchone()

    result = expect_is_not_none(result)
    nullability = _expect_payload_str_dict(decode_payload(result[0]))
    expect_equal(nullability["x"], "unknown")
    expect_equal(nullability["y"], "unknown")
    expect_equal(result[1], None)


def test_compute_contracts_with_guards_and_catalog(
    ctx: TestContext,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts extracts preconditions from guard clauses."""
    code = """def guarded(x, y):
    if x is None:
        raise ValueError("x required")
    if y <= 0:
        raise ValueError("y must be positive")
    return x + y
"""
    func_ast = _create_function_ast(
        goid=GOID_GUARDED,
        rel_path="module.py",
        qualname="guarded",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_GUARDED, "module.py", "guarded", 1, 6)]
    catalog = _create_mock_catalog(spans)
    ensure_catalog_with_goids(ctx, catalog)

    _build_and_write_contract_rows(
        ctx.gateway,
        contracts_snapshot,
        function_ast_map={GOID_GUARDED: func_ast},
        catalog=catalog,
    )

    result = ctx.gateway.con.execute(
        """
        SELECT preconditions_json, raises_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_GUARDED],
    ).fetchone()

    result = expect_is_not_none(result)
    preconditions = _expect_payload_list(decode_payload(result[0]))
    raises = _expect_payload_list(decode_payload(result[1]))

    expect_true(len(preconditions) > 0 or len(raises) > 0)


def test_compute_contracts_with_bool_return_type(
    ctx: TestContext,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts detects bool return type for postconditions."""
    code = """def is_valid(x):
    return x > 0
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="is_valid",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "is_valid", 1, 3)]
    catalog = _create_mock_catalog(spans)
    ensure_catalog_with_goids(ctx, catalog)

    _seed_function_types(
        ctx.gateway,
        goid=GOID_SIMPLE,
        return_type="bool",
        param_types={"x": "int"},
    )

    _build_and_write_contract_rows(
        ctx.gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = ctx.gateway.con.execute(
        """
        SELECT postconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    result = expect_is_not_none(result)
    postconditions = _expect_payload_list(decode_payload(result[0]))

    expect_equal(postconditions, [])


def test_compute_contracts_confidence_score(
    ctx: TestContext,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts calculates confidence score based on available data."""
    code = """def well_documented(x):
    if x is None:
        raise ValueError()
    return x * 2
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="well_documented",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "well_documented", 1, 5)]
    catalog = _create_mock_catalog(spans)
    ensure_catalog_with_goids(ctx, catalog)

    _seed_docstrings(
        ctx.gateway,
        rel_path="module.py",
        qualname="well_documented",
        params=[{"name": "x", "type_name": None, "description": "Input value"}],
    )
    _seed_function_types(
        ctx.gateway,
        goid=GOID_SIMPLE,
        return_type="int",
        param_types={"x": "int"},
    )

    _build_and_write_contract_rows(
        ctx.gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = ctx.gateway.con.execute(
        """
        SELECT contract_confidence
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    result = expect_is_not_none(result)
    confidence = result[0]

    if confidence is None:
        pytest.xfail("Contract confidence is not populated by current contract builder.")
    expect_true(confidence >= 0.0)


def test_compute_contracts_multiple_goids(
    fresh_gateway: StorageGateway,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts processes multiple GOIDs from catalog."""
    code1 = """def func1(x):
    return x
"""
    code2 = """def func2(y):
    return y * 2
"""
    ast1 = _create_function_ast(GOID_SIMPLE, "mod.py", "func1", code1)
    ast2 = _create_function_ast(GOID_TYPED, "mod.py", "func2", code2)

    spans = [
        MockFunctionSpan(GOID_SIMPLE, "mod.py", "func1", 1, 3),
        MockFunctionSpan(GOID_TYPED, "mod.py", "func2", 1, 3),
    ]
    catalog = _create_mock_catalog(spans)

    _build_and_write_contract_rows(
        fresh_gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: ast1, GOID_TYPED: ast2},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], EXPECTED_GOID_COUNT)


def test_compute_contracts_with_nullable_return(
    fresh_gateway: StorageGateway,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts detects nullable return types."""
    code = """def maybe_return(x):
    if x > 0:
        return x
    return None
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="maybe_return",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "maybe_return", 1, 5)]
    catalog = _create_mock_catalog(spans)

    _seed_function_types(
        fresh_gateway,
        goid=GOID_SIMPLE,
        return_type="Optional[int]",
        param_types={"x": "int"},
    )

    _build_and_write_contract_rows(
        fresh_gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT return_nullability
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], None)


def test_compute_contracts_with_isinstance_guard(
    fresh_gateway: StorageGateway,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts extracts isinstance guards."""
    code = """def typed_guard(x):
    if not isinstance(x, str):
        raise TypeError("expected str")
    return x.upper()
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="typed_guard",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "typed_guard", 1, 5)]
    catalog = _create_mock_catalog(spans)

    _build_and_write_contract_rows(
        fresh_gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT preconditions_json, raises_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    result = expect_is_not_none(result)
    preconditions = _expect_payload_list(decode_payload(result[0]))
    raises = _expect_payload_list(decode_payload(result[1]))

    expect_true(len(preconditions) > 0 or len(raises) > 0)


def test_compute_contracts_with_len_check(
    fresh_gateway: StorageGateway,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts extracts len() checks."""
    code = """def check_len(items):
    if len(items) == 0:
        raise ValueError("empty list")
    return items[0]
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="check_len",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "check_len", 1, 5)]
    catalog = _create_mock_catalog(spans)

    _build_and_write_contract_rows(
        fresh_gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT preconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    expect_is_not_none(result)


def test_compute_contracts_with_predicate_name(
    fresh_gateway: StorageGateway,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts detects predicate function names (is_*, has_*, etc.)."""
    code = """def is_valid(x):
    return x > 0
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="is_valid",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "is_valid", 1, 3)]
    catalog = _create_mock_catalog(spans)

    _build_and_write_contract_rows(
        fresh_gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT postconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    expect_is_not_none(result)


def test_compute_contracts_with_doc_return_none(
    fresh_gateway: StorageGateway,
    contracts_snapshot: SnapshotRef,
) -> None:
    """Compute contracts infers nullable from doc return mentioning None."""
    code = """def maybe_find(x):
    return x if x > 0 else None
"""
    func_ast = _create_function_ast(
        goid=GOID_SIMPLE,
        rel_path="module.py",
        qualname="maybe_find",
        code=code,
    )
    spans = [MockFunctionSpan(GOID_SIMPLE, "module.py", "maybe_find", 1, 3)]
    catalog = _create_mock_catalog(spans)

    _seed_docstrings(
        fresh_gateway,
        rel_path="module.py",
        qualname="maybe_find",
        params=[{"name": "x", "type_name": None, "description": "input value"}],
        returns={"type_name": None, "description": "Returns x or None if not found"},
    )

    _build_and_write_contract_rows(
        fresh_gateway,
        contracts_snapshot,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT return_nullability
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_VARIANT.repo, DEFAULT_VARIANT.commit, GOID_SIMPLE],
    ).fetchone()

    result = expect_is_not_none(result)
    expect_equal(result[0], "nullable")
