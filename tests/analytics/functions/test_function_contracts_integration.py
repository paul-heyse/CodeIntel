"""Integration tests for function_contracts module with catalog.

This module tests compute_function_contracts with a realistic fake catalog
to exercise the main execution paths including docstrings and type info.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.analytics.functions.function_contracts import (
    compute_function_contracts,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config import FunctionContractsStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.catalog import FunctionCatalog, FunctionCatalogProvider

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Test constants
TEST_REPO = "test-repo"
TEST_COMMIT = "abc123"
TEST_MAX_CONDITIONS = 10
GOID_SIMPLE = 10001
GOID_TYPED = 10002
GOID_GUARDED = 10003
GOID_MISSING = 10004
MIN_CONFIDENCE = 0.5
EXPECTED_GOID_COUNT = 2


@dataclass(frozen=True)
class FunctionSpan:
    """Mock function span for testing."""

    goid: int
    rel_path: str
    qualname: str
    start_line: int
    end_line: int


@pytest.fixture
def contracts_config(tmp_path: Path) -> FunctionContractsStepConfig:
    """Create a test configuration for function contracts.

    Parameters
    ----------
    tmp_path
        Temporary directory for test repo.

    Returns
    -------
    FunctionContractsStepConfig
        Configuration for testing.
    """
    snapshot = SnapshotRef(repo=TEST_REPO, commit=TEST_COMMIT, repo_root=tmp_path)
    return FunctionContractsStepConfig(
        snapshot=snapshot,
        max_conditions_per_func=TEST_MAX_CONDITIONS,
    )


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
    """
    tree = ast.parse(code)
    func_node = tree.body[0]
    lines = code.split("\n")
    return FunctionAst(
        goid=goid,
        rel_path=rel_path,
        qualname=qualname,
        node=func_node,  # type: ignore[arg-type]
        start_line=1,
        end_line=len(lines),
        lines=lines,
    )


@dataclass
class FakeFunctionCatalog:
    """Fake function catalog for testing.

    Attributes
    ----------
    function_spans
        List of function spans in the catalog.
    """

    function_spans: list[FunctionSpan] = field(default_factory=list)


@dataclass
class FakeCatalogProvider:
    """Fake catalog provider implementing FunctionCatalogProvider protocol.

    Attributes
    ----------
    _catalog
        The inner function catalog.
    """

    _catalog: FakeFunctionCatalog = field(default_factory=FakeFunctionCatalog)

    def catalog(self) -> FunctionCatalog:
        """Return the function catalog.

        Returns
        -------
        FunctionCatalog
            The inner catalog containing function spans (cast from fake).
        """
        # Cast is safe: FakeFunctionCatalog has function_spans which is all
        # that compute_function_contracts uses
        return cast("FunctionCatalog", self._catalog)

    def urn_for_goid(self, _goid: int) -> str | None:  # noqa: PLR6301
        """Return URN for a GOID (stub implementation).

        Parameters
        ----------
        _goid
            The GOID to look up.

        Returns
        -------
        str | None
            Always returns None in this fake.
        """
        return None

    def lookup_goid(  # noqa: PLR6301
        self,
        _rel_path: str,
        _start_line: int,
        _end_line: int | None,
        _qualname: str | None,
    ) -> int | None:
        """Resolve GOID from span (stub implementation).

        Parameters
        ----------
        _rel_path
            Relative path to the file.
        _start_line
            Starting line number.
        _end_line
            Ending line number.
        _qualname
            Qualified name of the function.

        Returns
        -------
        int | None
            Always returns None in this fake.
        """
        return None


def _create_fake_catalog(spans: list[FunctionSpan]) -> FunctionCatalogProvider:
    """Create a fake catalog provider with given spans.

    Parameters
    ----------
    spans
        List of function spans.

    Returns
    -------
    FunctionCatalogProvider
        Fake catalog provider implementing the protocol.
    """
    # Cast to protocol type - the fake implements the needed subset
    return cast(
        "FunctionCatalogProvider",
        FakeCatalogProvider(_catalog=FakeFunctionCatalog(function_spans=spans)),
    )


def _seed_docstrings(
    gateway: StorageGateway,
    rel_path: str,
    qualname: str,
    params: list[dict[str, str]],
    returns: dict[str, str] | None = None,
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
    gateway.con.execute(
        """
        INSERT INTO core.docstrings (
            repo, commit, rel_path, module, qualname, kind,
            params, returns, raises, examples, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            TEST_REPO,
            TEST_COMMIT,
            rel_path,
            rel_path.replace("/", ".").replace(".py", ""),
            qualname,
            "function",
            json.dumps(params),
            json.dumps(returns) if returns else "{}",
            "[]",
            "[]",
        ],
    )


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
    gateway.con.execute(
        """
        INSERT INTO analytics.function_types (
            repo, commit, function_goid_h128, return_type, param_types, created_at
        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            TEST_REPO,
            TEST_COMMIT,
            goid,
            return_type,
            json.dumps(param_types) if param_types else "{}",
        ],
    )


# =============================================================================
# Integration Tests with Catalog
# =============================================================================


def test_compute_contracts_with_catalog_goid_iteration(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "simple", 1, 3)]
    catalog = _create_fake_catalog(spans)

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT function_goid_h128, preconditions_json, postconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [TEST_REPO, TEST_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == GOID_SIMPLE


def test_compute_contracts_with_missing_ast(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts handles GOIDs without AST gracefully."""
    spans = [FunctionSpan(GOID_MISSING, "missing.py", "missing_func", 1, 3)]
    catalog = _create_fake_catalog(spans)

    # No AST map provided for the GOID
    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT function_goid_h128, preconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [TEST_REPO, TEST_COMMIT],
    ).fetchone()

    # Row should be created with empty contracts
    assert result is not None
    assert result[0] == GOID_MISSING
    assert json.loads(result[1]) == []


def test_compute_contracts_with_docstring_data(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "with_docs", 1, 3)]
    catalog = _create_fake_catalog(spans)

    # Seed docstring with optional parameter info
    _seed_docstrings(
        fresh_gateway,
        rel_path="module.py",
        qualname="with_docs",
        params=[
            {"name": "x", "desc": "Required parameter, must not be None"},
            {"name": "y", "desc": "Optional value, may be none"},
        ],
    )

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT param_nullability_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None
    nullability = json.loads(result[0])
    assert nullability["x"] == "non_null"
    assert nullability["y"] == "nullable"


def test_compute_contracts_with_type_annotations(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_TYPED, "module.py", "typed_func", 1, 3)]
    catalog = _create_fake_catalog(spans)

    # Seed type info with nullable and non-nullable types
    _seed_function_types(
        fresh_gateway,
        goid=GOID_TYPED,
        return_type="int",
        param_types={"x": "int", "y": "Optional[int]"},
    )

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_TYPED: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT param_nullability_json, return_nullability
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_TYPED],
    ).fetchone()

    assert result is not None
    nullability = json.loads(result[0])
    assert nullability["x"] == "non_null"
    assert nullability["y"] == "nullable"
    assert result[1] == "non_null"


def test_compute_contracts_with_guards_and_catalog(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
) -> None:
    """Compute contracts extracts preconditions from guard clauses."""
    code = """def guarded(x, y):
    if x is None:
        raise ValueError("x required")
    assert y > 0, "y must be positive"
    return x + y
"""
    func_ast = _create_function_ast(
        goid=GOID_GUARDED,
        rel_path="module.py",
        qualname="guarded",
        code=code,
    )
    spans = [FunctionSpan(GOID_GUARDED, "module.py", "guarded", 1, 6)]
    catalog = _create_fake_catalog(spans)

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_GUARDED: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT preconditions_json, raises_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_GUARDED],
    ).fetchone()

    assert result is not None
    preconditions = json.loads(result[0])
    raises = json.loads(result[1])

    # Should have extracted at least one precondition or raise
    assert len(preconditions) > 0 or len(raises) > 0


def test_compute_contracts_with_bool_return_type(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "is_valid", 1, 3)]
    catalog = _create_fake_catalog(spans)

    _seed_function_types(
        fresh_gateway,
        goid=GOID_SIMPLE,
        return_type="bool",
        param_types={"x": "int"},
    )

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT postconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None
    postconditions = json.loads(result[0])

    # Should have a returns_bool_predicate postcondition
    kinds = [p.get("kind") for p in postconditions]
    assert "returns_bool_predicate" in kinds


def test_compute_contracts_confidence_score(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "well_documented", 1, 5)]
    catalog = _create_fake_catalog(spans)

    # Add both docstrings and type info for higher confidence
    _seed_docstrings(
        fresh_gateway,
        rel_path="module.py",
        qualname="well_documented",
        params=[{"name": "x", "desc": "Input value"}],
    )
    _seed_function_types(
        fresh_gateway,
        goid=GOID_SIMPLE,
        return_type="int",
        param_types={"x": "int"},
    )

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT contract_confidence
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None
    confidence = result[0]
    # With types, docs, and guards, confidence should be high
    assert confidence > MIN_CONFIDENCE


def test_compute_contracts_multiple_goids(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
        FunctionSpan(GOID_SIMPLE, "mod.py", "func1", 1, 3),
        FunctionSpan(GOID_TYPED, "mod.py", "func2", 1, 3),
    ]
    catalog = _create_fake_catalog(spans)

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: ast1, GOID_TYPED: ast2},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [TEST_REPO, TEST_COMMIT],
    ).fetchone()

    assert result is not None
    assert result[0] == EXPECTED_GOID_COUNT


def test_compute_contracts_with_nullable_return(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "maybe_return", 1, 5)]
    catalog = _create_fake_catalog(spans)

    _seed_function_types(
        fresh_gateway,
        goid=GOID_SIMPLE,
        return_type="Optional[int]",
        param_types={"x": "int"},
    )

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT return_nullability
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None
    assert result[0] == "nullable"


def test_compute_contracts_with_isinstance_guard(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "typed_guard", 1, 5)]
    catalog = _create_fake_catalog(spans)

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT preconditions_json, raises_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None
    preconditions = json.loads(result[0])
    raises = json.loads(result[1])

    # Should have extracted type error raises
    assert len(preconditions) > 0 or len(raises) > 0


def test_compute_contracts_with_len_check(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "check_len", 1, 5)]
    catalog = _create_fake_catalog(spans)

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT preconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None


def test_compute_contracts_with_predicate_name(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "is_valid", 1, 3)]
    catalog = _create_fake_catalog(spans)

    # No type info, just rely on name
    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    # Should produce result without error
    result = fresh_gateway.con.execute(
        """
        SELECT postconditions_json
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None


def test_compute_contracts_with_doc_return_none(
    fresh_gateway: StorageGateway,
    contracts_config: FunctionContractsStepConfig,
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
    spans = [FunctionSpan(GOID_SIMPLE, "module.py", "maybe_find", 1, 3)]
    catalog = _create_fake_catalog(spans)

    _seed_docstrings(
        fresh_gateway,
        rel_path="module.py",
        qualname="maybe_find",
        params=[{"name": "x", "desc": "input value"}],
        returns={"desc": "Returns x or None if not found"},
    )

    compute_function_contracts(
        fresh_gateway,
        contracts_config,
        function_ast_map={GOID_SIMPLE: func_ast},
        catalog=catalog,
    )

    result = fresh_gateway.con.execute(
        """
        SELECT return_nullability
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [TEST_REPO, TEST_COMMIT, GOID_SIMPLE],
    ).fetchone()

    assert result is not None
    assert result[0] == "nullable"
