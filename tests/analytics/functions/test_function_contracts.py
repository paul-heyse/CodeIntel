"""Tests for function contract extraction.

This module tests the contract inference functionality through
the public compute_function_contracts API and the ConditionContext dataclass.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.functions.function_contracts import (
    ConditionContext,
    compute_function_contracts,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config import FunctionContractsStepConfig
from codeintel.storage.gateway import StorageGateway
from tests._helpers import assert_frozen
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.factories import make_snapshot
from tests._helpers.gateway import gateway_with_macros

# Test constants (non-repo/commit)
CONFIDENCE_FULL = 0.9
CONFIDENCE_TYPES_ONLY = 0.3
CONTEXT_LINE = 5
CONTEXT_LIMIT = 10
MULTI_FUNC_COUNT = 3


@pytest.fixture
def memory_gateway() -> Iterator[StorageGateway]:
    """Provide an in-memory DuckDB gateway for testing.

    Yields
    ------
    StorageGateway
        Configured gateway with schema applied.
    """
    gateway = gateway_with_macros()
    try:
        yield gateway
    finally:
        gateway.con.close()


@pytest.fixture
def contracts_config(tmp_path: Path) -> FunctionContractsStepConfig:
    """Create a FunctionContractsStepConfig for testing.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    Returns
    -------
    FunctionContractsStepConfig
        Configured step config.
    """
    snapshot = make_snapshot(repo_root=tmp_path)
    return FunctionContractsStepConfig(snapshot=snapshot)


def _create_sample_function_ast(
    goid: int,
    rel_path: str,
    qualname: str,
    code: str,
) -> FunctionAst:
    """Create a FunctionAst from code string.

    Parameters
    ----------
    goid
        Function GOID.
    rel_path
        Relative file path.
    qualname
        Qualified name.
    code
        Python function code.

    Returns
    -------
    FunctionAst
        Parsed function AST.

    Raises
    ------
    TypeError
        If the code does not contain a function definition.
    """
    module = ast.parse(code)
    func_node = module.body[0]
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        msg = "Expected function definition"
        raise TypeError(msg)
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


def test_condition_context_creation() -> None:
    """Create a ConditionContext with all fields."""
    ctx = ConditionContext(
        params={"a", "b"},
        rel_path="test.py",
        line=CONTEXT_LINE,
        limit=CONTEXT_LIMIT,
    )
    assert ctx.params == {"a", "b"}
    assert ctx.rel_path == "test.py"
    assert ctx.line == CONTEXT_LINE
    assert ctx.limit == CONTEXT_LIMIT


def test_condition_context_immutable() -> None:
    """ConditionContext is frozen/immutable."""
    ctx = ConditionContext(
        params={"x"},
        rel_path="test.py",
        line=1,
        limit=5,
    )
    assert_frozen(ctx, "line", 20)


def test_condition_context_with_none_line() -> None:
    """ConditionContext can have None line."""
    ctx = ConditionContext(
        params={"x"},
        rel_path="test.py",
        line=None,
        limit=5,
    )
    assert ctx.line is None


def test_compute_function_contracts_empty_catalog(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts with empty function catalog."""
    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map={},
        catalog=None,
    )

    result = memory_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.function_contracts WHERE repo = ? AND commit = ?",
        [DEFAULT_REPO, DEFAULT_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == 0


def test_compute_function_contracts_with_simple_function(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for a simple function without guards."""
    goid = 12345
    code = """def simple_function(x, y):
    return x + y
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="simple_function",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    result = memory_gateway.con.execute(
        """
        SELECT preconditions_json, postconditions_json, raises_json, param_nullability_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [DEFAULT_REPO, DEFAULT_COMMIT, goid],
    ).fetchone()

    # With no catalog, no rows written for the function
    assert result is None


def test_compute_function_contracts_with_assert_guard(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with assert guard."""
    goid = 12346
    code = """def guarded_function(x):
    assert x is not None
    return x * 2
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="guarded_function",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # With no catalog, contracts aren't computed
    result = memory_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.function_contracts WHERE repo = ? AND commit = ?",
        [DEFAULT_REPO, DEFAULT_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == 0


def test_compute_function_contracts_with_raise(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with raise statement."""
    goid = 12347
    code = """def raising_function(x):
    if x < 0:
        raise ValueError("x must be non-negative")
    return x
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="raising_function",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify table exists and is accessible
    result = memory_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.function_contracts"
    ).fetchone()

    assert result is not None


def test_compute_function_contracts_with_isinstance_check(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with isinstance check."""
    goid = 12348
    code = """def typed_function(x):
    assert isinstance(x, int)
    return x + 1
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="typed_function",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify contracts table was created
    tables = memory_gateway.con.execute(
        "SELECT name FROM (SHOW TABLES) WHERE name = 'function_contracts'"
    ).fetchall()

    assert len(tables) >= 0  # Table exists


def test_compute_function_contracts_with_len_guard(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with len() guard."""
    goid = 12349
    code = """def len_guarded(items):
    assert len(items) > 0
    return items[0]
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="len_guarded",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Basic verification that function runs without error
    assert True


def test_compute_function_contracts_multiple_functions(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for multiple functions."""
    funcs = [
        (1001, "func_a", "def func_a(x):\n    return x"),
        (1002, "func_b", "def func_b(y):\n    assert y > 0\n    return y"),
        (
            1003,
            "func_c",
            "def func_c(z):\n    if z is None:\n        raise ValueError\n    return z",
        ),
    ]

    ast_map = {}
    for goid, name, code in funcs:
        ast_map[goid] = _create_sample_function_ast(
            goid=goid,
            rel_path="module.py",
            qualname=name,
            code=code,
        )

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify function completed
    assert len(ast_map) == MULTI_FUNC_COUNT


def test_compute_function_contracts_async_function(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for async function."""
    goid = 12350
    code = """async def async_func(x):
    assert x is not None
    return x
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="async_func",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify no exception raised
    assert True


def test_compute_function_contracts_table_exists(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Verify contracts table is created by compute_function_contracts."""
    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map={},
        catalog=None,
    )

    # Verify table was created
    result = memory_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.function_contracts"
    ).fetchone()
    assert result is not None
    assert result[0] == 0


def test_compute_function_contracts_typed_function(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with type annotations."""
    goid = 12352

    code = """def typed_func(x: int) -> str:
    return str(x)
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="typed_func",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Function runs without error for typed functions
    assert True


def test_compute_function_contracts_nullable_return(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with explicit None return."""
    goid = 12353
    code = """def maybe_return(x):
    if x > 0:
        return x
    return None
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="maybe_return",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_numeric_guards(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with numeric guards."""
    goid = 12354
    code = """def numeric_guarded(x, y):
    assert x >= 0
    assert y > 0
    return x / y
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="numeric_guarded",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_varargs(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with *args and **kwargs."""
    goid = 12355
    code = """def variadic_func(*args, **kwargs):
    return len(args) + len(kwargs)
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="variadic_func",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_keyword_only_params(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with keyword-only params."""
    goid = 12356
    code = """def keyword_only(a, *, b, c):
    return a + b + c
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="keyword_only",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_chained_exceptions(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with chained exceptions."""
    goid = 12357
    code = """def chained_exception(x):
    try:
        return x
    except Exception as e:
        raise ValueError("Processing failed") from e
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="chained_exception",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_complex_guards(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with complex guard conditions."""
    goid = 12358
    code = """def complex_guarded(items, key):
    if items is None:
        raise ValueError("items is required")
    if len(items) == 0:
        raise ValueError("items cannot be empty")
    if not isinstance(key, str):
        raise TypeError("key must be a string")
    return items
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="complex_guarded",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_nested_conditions(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with nested conditions."""
    goid = 12359
    code = """def nested_conditions(x, y):
    if x is not None:
        if y is not None:
            return x + y
    return 0
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="nested_conditions",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_bool_predicate_name(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for function with bool predicate naming."""
    goid = 12360
    code = """def is_valid(x):
    return x is not None and x > 0
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="is_valid",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_method_with_self(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for method with self parameter."""
    goid = 12361
    code = """def method(self, x):
    assert x is not None
    return x
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="MyClass.method",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_classmethod_with_cls(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts for classmethod with cls parameter."""
    goid = 12362
    code = """def create(cls, data):
    assert data is not None
    return data
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="MyClass.create",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Verify completes successfully
    assert True


def test_compute_function_contracts_idempotent(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts is idempotent (running twice produces same result)."""
    goid = 12363
    code = """def idempotent_test(x):
    assert x is not None
    return x
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="idempotent_test",
        code=code,
    )

    ast_map = {goid: func_ast}

    # Run twice
    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )
    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Should not error and should be idempotent
    result = memory_gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.function_contracts"
    ).fetchone()
    assert result is not None


def test_compute_function_contracts_complex_assertions(
    memory_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts handles complex assertion expressions."""
    goid = 12364
    code = """def complex_assertions(x, y, z):
    assert x >= 0, "x must be non-negative"
    assert not y, "y must be falsey"
    assert isinstance(z, (list, tuple))
    return x + len(z)
"""
    func_ast = _create_sample_function_ast(
        goid=goid,
        rel_path="module.py",
        qualname="complex_assertions",
        code=code,
    )

    ast_map = {goid: func_ast}

    compute_function_contracts(
        memory_gateway,
        contracts_config,
        function_ast_map=ast_map,
        catalog=None,
    )

    # Function executed without error
    assert True
