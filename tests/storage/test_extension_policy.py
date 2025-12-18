"""Tests for storage-owned DuckDB extension loading policy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.storage.gateway.extensions import load_extensions_from_env, parse_extensions_env
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.env_vars import temporary_env, unset_env


@dataclass
class _ExecRecorder:
    executed: list[str]

    def execute(self, query: str, parameters: object | None = None) -> None:
        _ = parameters
        self.executed.append(query)


def test_parse_extensions_env_empty() -> None:
    """parse_extensions_env returns empty when unset."""
    with unset_env("CODEINTEL_DUCKDB_EXTENSIONS"):
        expect_equal(parse_extensions_env(), (), label="extensions")


def test_load_extensions_from_env_loads_and_installs() -> None:
    """load_extensions_from_env issues INSTALL+LOAD when allowed."""
    with temporary_env("CODEINTEL_DUCKDB_EXTENSIONS", "fts,json"):
        recorder = _ExecRecorder(executed=[])

        load_extensions_from_env(recorder, allow_install=True)
        expect_true("INSTALL fts" in recorder.executed, message="install fts")
        expect_true("LOAD fts" in recorder.executed, message="load fts")
        expect_true("INSTALL json" in recorder.executed, message="install json")
        expect_true("LOAD json" in recorder.executed, message="load json")


def test_load_extensions_from_env_skips_install_when_disallowed() -> None:
    """load_extensions_from_env skips INSTALL when allow_install is False."""
    with temporary_env("CODEINTEL_DUCKDB_EXTENSIONS", "fts"):
        recorder = _ExecRecorder(executed=[])

        load_extensions_from_env(recorder, allow_install=False)
        expect_true("INSTALL fts" not in recorder.executed, message="install skipped")
        expect_true("LOAD fts" in recorder.executed, message="load fts")


def test_load_extensions_from_env_rejects_invalid_extension_name() -> None:
    """load_extensions_from_env rejects invalid extension tokens."""
    with temporary_env("CODEINTEL_DUCKDB_EXTENSIONS", "bad-name"):
        recorder = _ExecRecorder(executed=[])

        with pytest.raises(ValueError, match="Invalid DuckDB extension name"):
            load_extensions_from_env(recorder, allow_install=True)
