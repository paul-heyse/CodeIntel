"""Tests for lifecycle-safe staging helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.storage.staging import registered_temp_relation
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


@dataclass
class _FakeDuckDBConn:
    registered: dict[str, object]
    unregistered: list[str]

    def register(self, name: str, obj: object) -> None:
        self.registered[name] = obj

    def unregister(self, name: str) -> None:
        self.unregistered.append(name)
        self.registered.pop(name, None)


def test_registered_temp_relation_unregisters_on_success() -> None:
    con = _FakeDuckDBConn(registered={}, unregistered=[])
    payload = object()
    with registered_temp_relation(con, payload, prefix="ci_test_") as name:
        expect_true(name.startswith("ci_test_"), message="prefix applied")
        expect_true(name in con.registered, message="registered")
        expect_true(con.registered[name] is payload, message="payload registered")
    expect_true(name in con.unregistered, message="unregistered after context")
    expect_equal(con.registered, {}, label="registered map empty")


def test_registered_temp_relation_unregisters_on_exception() -> None:
    con = _FakeDuckDBConn(registered={}, unregistered=[])
    payload = object()
    with pytest.raises(RuntimeError, match="boom"):
        with registered_temp_relation(con, payload, prefix="ci_test_") as name:
            expect_true(name in con.registered, message="registered")
            raise RuntimeError("boom")
    expect_true(con.registered == {}, message="cleaned up after exception")
    expect_true(len(con.unregistered) == 1, message="unregister called exactly once")

