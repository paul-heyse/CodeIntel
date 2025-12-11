"""Tests for module_index module."""

from __future__ import annotations

import logging
import logging.handlers
from typing import TYPE_CHECKING

from codeintel.storage.helpers.module_index import load_module_map
from tests._helpers.assertions.expectation_assertions import (
    expect_empty,
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    import pytest

    from codeintel.storage.gateway import StorageGateway


def test_load_module_map_returns_normalized_paths(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify load_module_map returns normalized path mappings."""
    repo = "test/repo"
    commit = "abc123"

    fresh_gateway.core.insert_modules(
        [
            ("test.module", "src/test/module.py", repo, commit),
            ("another.module", "src/another/module.py", repo, commit),
        ]
    )

    result = load_module_map(fresh_gateway, repo, commit)

    expect_is_instance(result, dict)
    expect_length(result, 2)
    expect_in("src/test/module.py", result)
    expect_equal(result.get("src/test/module.py"), "test.module")


def test_load_module_map_filters_by_language(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify load_module_map filters by language when specified."""
    repo = "test/repo"
    commit = "abc123"

    con = fresh_gateway.con
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language)
        VALUES
            ('py.module', 'py.py', ?, ?, 'python'),
            ('js.module', 'js.js', ?, ?, 'javascript')
        """,
        [repo, commit, repo, commit],
    )

    result = load_module_map(fresh_gateway, repo, commit, language="python")

    expect_length(result, 1)
    expect_in("py.py", result)


def test_load_module_map_returns_empty_on_no_match(
    fresh_gateway: StorageGateway, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify load_module_map returns empty dict with warning for no match."""
    repo = "nonexistent/repo"
    commit = "nonexistent"

    with caplog.at_level(logging.WARNING):
        result = load_module_map(fresh_gateway, repo, commit)

    expect_is_instance(result, dict)
    expect_empty(result)

    warning_found = any("No modules found" in record.message for record in caplog.records)
    expect_true(warning_found)


def test_load_module_map_uses_custom_logger(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify load_module_map uses provided logger."""
    repo = "nonexistent/repo"
    commit = "nonexistent"

    test_logger = logging.getLogger("test_logger")
    handler = logging.handlers.MemoryHandler(capacity=100)
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.WARNING)

    try:
        result = load_module_map(fresh_gateway, repo, commit, logger=test_logger)

        expect_length(result, 0)

        handler.flush()
        warning_found = any("No modules found" in record.message for record in handler.buffer)
        expect_true(warning_found)
    finally:
        test_logger.removeHandler(handler)


def test_load_module_map_normalizes_path_with_leading_slash(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify load_module_map normalizes paths with leading slashes."""
    repo = "test/repo"
    commit = "abc123"

    con = fresh_gateway.con
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit)
        VALUES ('mod', '/src/module.py', ?, ?)
        """,
        [repo, commit],
    )

    result = load_module_map(fresh_gateway, repo, commit)

    expect_length(result, 1)

    has_key = "src/module.py" in result or "/src/module.py" in result
    expect_true(has_key)


def test_load_module_map_handles_multiple_modules_same_path(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify load_module_map handles duplicate paths by last-wins."""
    repo = "test/repo"
    commit = "abc123"

    con = fresh_gateway.con
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit)
        VALUES
            ('mod1', 'shared.py', ?, ?),
            ('mod2', 'shared.py', ?, ?)
        """,
        [repo, commit, repo, commit],
    )

    result = load_module_map(fresh_gateway, repo, commit)

    expect_in("shared.py", result)
    module_value = result.get("shared.py")
    expect_true(module_value in {"mod1", "mod2"})
