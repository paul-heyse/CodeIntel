"""Tests for saver-tag-derived output inventory."""

from __future__ import annotations

from types import ModuleType

import hamilton.driver as h_driver
import pytest
from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.dag_catalog_compiler import compile_dag_catalog
from codeintel.build.hamilton.materializers import DuckDBRowsSaver, FileArtifactSaver
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator


def _build_driver(module: ModuleType) -> h_driver.Driver:
    return h_driver.Builder().with_modules(module).allow_module_overrides().build()


def _module_with_saver_outputs() -> ModuleType:
    module = ModuleType("saver_inventory_module")

    @SaveToObjectMetadataDecorator(
        [DuckDBRowsSaver],
        output_name_="m__core__alpha",
        env=source("env"),
        catalog=source("catalog"),
        target_name=value("alpha"),
        table_key=value("core.alpha"),
        columns=value(("id",)),
        json_schema_id=value("schema:alpha"),
    )
    def alpha_rows() -> tuple[tuple[int, ...], ...]:
        return ()

    @SaveToObjectMetadataDecorator(
        [FileArtifactSaver],
        output_name_="m__artifact__alpha_meta",
        env=source("env"),
        catalog=source("catalog"),
        target_name=value("alpha"),
        artifact_name=value("alpha_meta"),
        path_template=value("{build_dir}/alpha_meta.json"),
    )
    def alpha_meta(alpha_rows: tuple[tuple[int, ...], ...]) -> bytes:
        _ = alpha_rows
        return b"ok"

    @codeintel_target(domain="analytics", target="alpha")
    def t__alpha(
        alpha_rows: tuple[tuple[int, ...], ...],
        alpha_meta: bytes,
    ) -> int:
        _ = (alpha_rows, alpha_meta)
        return 1

    module.__dict__.update(
        {
            "alpha_rows": alpha_rows,
            "alpha_meta": alpha_meta,
            "t__alpha": t__alpha,
        }
    )
    return module


def test_catalog_outputs_derived_from_savers() -> None:
    """Catalog outputs should mirror saver tags."""
    driver = _build_driver(_module_with_saver_outputs())
    catalog = compile_dag_catalog(driver, strict=True)

    table_output = catalog.table_outputs.get("core.alpha")
    if table_output is None:
        pytest.fail("Expected table output core.alpha")
    if table_output.producer_target != "alpha":
        pytest.fail("Table output has wrong producer target")
    if table_output.tags.get("ci.json_schema_id") != "schema:alpha":
        pytest.fail("Table output tags missing metadata")

    artifact_output = catalog.artifact_outputs.get("alpha_meta")
    if artifact_output is None:
        pytest.fail("Expected artifact output alpha_meta")
    if artifact_output.artifact_path_template != "{build_dir}/alpha_meta.json":
        pytest.fail("Artifact output missing path template")
