"""Tests for retired SQLGlot view schema inference."""

from __future__ import annotations

from types import ModuleType

import pytest

from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.views.schema_inference import derive_view_schemas

pytestmark = pytest.mark.no_runtime_env


def _view_module() -> ModuleType:
    module = ModuleType("tests.view_schema_module")

    def v_demo() -> str:
        return "SELECT * FROM analytics.demo"

    v_demo.__module__ = module.__name__
    module.__dict__["v_demo"] = v_demo
    return module


def test_derive_view_schema_raises_for_retired_sqlglot() -> None:
    """SQLGlot schema inference should be retired."""
    provider = MappingSchemaProvider({})
    view_module = _view_module()

    with pytest.raises(RuntimeError, match="SQLGlot view schema inference has been retired"):
        derive_view_schemas(provider=provider, modules=(view_module,))
