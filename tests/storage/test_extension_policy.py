"""Tests for storage-owned DuckDB extension loading policy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.storage.gateway.extensions import load_extensions_from_env, parse_extensions_env
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


@dataclass
class _ExecRecorder:
    executed: list[str]

    def execute(self, sql: str) -> None:
        self.executed.append(sql)


def test_parse_extensions_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEINTEL_DUCKDB_EXTENSIONS", raising=False)
    expect_equal(parse_extensions_env(), (), label="extensions")


def test_load_extensions_from_env_loads_and_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEINTEL_DUCKDB_EXTENSIONS", "fts,json")
    recorder = _ExecRecorder(executed=[])

    load_extensions_from_env(recorder, allow_install=True)
    expect_true("INSTALL fts" in recorder.executed, message="install fts")
    expect_true("LOAD fts" in recorder.executed, message="load fts")
    expect_true("INSTALL json" in recorder.executed, message="install json")
    expect_true("LOAD json" in recorder.executed, message="load json")


def test_load_extensions_from_env_skips_install_when_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEINTEL_DUCKDB_EXTENSIONS", "fts")
    recorder = _ExecRecorder(executed=[])

    load_extensions_from_env(recorder, allow_install=False)
    expect_true("INSTALL fts" not in recorder.executed, message="install skipped")
    expect_true("LOAD fts" in recorder.executed, message="load fts")


def test_load_extensions_from_env_rejects_invalid_extension_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEINTEL_DUCKDB_EXTENSIONS", "bad-name")
    recorder = _ExecRecorder(executed=[])

    with pytest.raises(ValueError, match="Invalid DuckDB extension name"):
        load_extensions_from_env(recorder, allow_install=True)

