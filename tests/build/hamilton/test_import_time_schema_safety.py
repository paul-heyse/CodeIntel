"""Import-time schema safety checks."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import codeintel.build.schemas as build_schemas

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pytest

    from codeintel.core.schemas.contract_primitives import DatasetContract


def test_history_helper_does_not_resolve_contracts_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify history helpers do not resolve contracts at import time."""

    def _fail_on_contract_resolution() -> Iterable[tuple[str, DatasetContract]]:
        msg = "iter_contracts_by_table_key called during import"
        raise AssertionError(msg)

    monkeypatch.setattr(build_schemas, "iter_contracts_by_table_key", _fail_on_contract_resolution)
    history_module = importlib.import_module("tests._helpers.orchestration.history")
    importlib.reload(history_module)
