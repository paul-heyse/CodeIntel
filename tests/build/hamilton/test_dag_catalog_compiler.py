"""Tests for DagCatalog compilation and IO surface derivation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType

import hamilton.driver as h_driver
import polars as pl
import pytest
from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.dag_catalog import IOSurface
from codeintel.build.hamilton.dag_catalog_compiler import compile_dag_catalog
from codeintel.build.hamilton.materializers import FileArtifactSaver, IcebergDatasetSaver
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_loader_query


def _build_driver(module: ModuleType) -> h_driver.Driver:
    return h_driver.Builder().with_modules(module).allow_module_overrides().build()


def _register_module_functions(
    module: ModuleType,
    *,
    functions: Mapping[str, Callable[..., object]],
) -> None:
    for name, fn in functions.items():
        fn.__module__ = module.__name__
        module.__dict__[name] = fn


def _module_with_duplicate_anchors() -> ModuleType:
    module = ModuleType("dup_anchor_module")

    @codeintel_target(domain="analytics", target="dup")
    def t__dup_one() -> int:
        return 1

    @codeintel_target(domain="analytics", target="dup")
    def t__dup_two() -> int:
        return 2

    _register_module_functions(
        module,
        functions={
            "t__dup_one": t__dup_one,
            "t__dup_two": t__dup_two,
        },
    )
    return module


def _module_with_branching_chain() -> ModuleType:
    module = ModuleType("branching_chain_module")

    @codeintel_target(domain="analytics", target="beta")
    def t__beta() -> int:
        return 1

    @codeintel_target(domain="analytics", target="gamma")
    def t__gamma() -> int:
        return 1

    @codeintel_target(domain="analytics", target="alpha")
    def t__alpha(t__beta: int, t__gamma: int) -> int:
        return t__beta + t__gamma

    _register_module_functions(
        module,
        functions={
            "t__beta": t__beta,
            "t__gamma": t__gamma,
            "t__alpha": t__alpha,
        },
    )
    return module


def _module_with_duplicate_outputs() -> ModuleType:
    module = ModuleType("dup_outputs_module")

    @SaveToObjectMetadataDecorator(
        [IcebergDatasetSaver],
        output_name_="m__core__dup_one",
        env=source("env"),
        catalog=source("catalog"),
        target_name=value("dup_target"),
        table_key=value("core.dup"),
    )
    def dup_rows_one() -> pl.LazyFrame:
        return pl.DataFrame({"id": [1]}).lazy()

    @SaveToObjectMetadataDecorator(
        [IcebergDatasetSaver],
        output_name_="m__core__dup_two",
        env=source("env"),
        catalog=source("catalog"),
        target_name=value("dup_target"),
        table_key=value("core.dup"),
    )
    def dup_rows_two() -> pl.LazyFrame:
        return pl.DataFrame({"id": [2]}).lazy()

    @codeintel_target(domain="analytics", target="dup_target")
    def t__dup_target(
        m__core__dup_one: object,
        m__core__dup_two: object,
    ) -> int:
        _ = (m__core__dup_one, m__core__dup_two)
        return 1

    _register_module_functions(
        module,
        functions={
            "dup_rows_one": dup_rows_one,
            "dup_rows_two": dup_rows_two,
            "t__dup_target": t__dup_target,
        },
    )
    return module


def _module_with_io_surface() -> ModuleType:
    module = ModuleType("io_surface_module")

    @tag_loader_query(domain="analytics", table_key="core.source")
    def source_rows() -> pl.LazyFrame:
        return pl.DataFrame({"id": [1]}).lazy()

    @SaveToObjectMetadataDecorator(
        [IcebergDatasetSaver],
        output_name_="m__analytics__alpha_out",
        env=source("env"),
        catalog=source("catalog"),
        target_name=value("alpha"),
        table_key=value("analytics.alpha_out"),
    )
    def alpha_rows(source_rows: pl.LazyFrame) -> pl.LazyFrame:
        return source_rows

    @SaveToObjectMetadataDecorator(
        [FileArtifactSaver],
        output_name_="m__artifact__alpha_meta",
        env=source("env"),
        catalog=source("catalog"),
        target_name=value("alpha"),
        artifact_name=value("alpha_meta"),
        path_template=value("{build_dir}/alpha_meta.json"),
    )
    def alpha_meta(alpha_rows: pl.LazyFrame) -> bytes:
        _ = alpha_rows
        return b"ok"

    @codeintel_target(domain="analytics", target="alpha")
    def t__alpha(alpha_rows: pl.LazyFrame) -> int:
        _ = alpha_rows
        return 1

    _register_module_functions(
        module,
        functions={
            "source_rows": source_rows,
            "alpha_rows": alpha_rows,
            "alpha_meta": alpha_meta,
            "t__alpha": t__alpha,
        },
    )
    return module


def test_duplicate_anchor_nodes_rejected() -> None:
    """Duplicate materialize anchors for the same target should fail."""
    driver = _build_driver(_module_with_duplicate_anchors())
    with pytest.raises(RuntimeError, match="Duplicate materialize nodes"):
        compile_dag_catalog(driver, strict=True)


def test_closure_deterministic_order() -> None:
    """Dependency closure should be stable and deterministic."""
    driver = _build_driver(_module_with_branching_chain())
    catalog = compile_dag_catalog(driver, strict=True)
    closure = catalog.closure(("alpha",))
    repeat = catalog.closure(("alpha",))
    assert closure == repeat
    assert closure == ("beta", "gamma", "alpha")


def test_duplicate_contract_outputs_rejected() -> None:
    """Duplicate contract outputs should raise a compile error."""
    driver = _build_driver(_module_with_duplicate_outputs())
    with pytest.raises(RuntimeError, match="Duplicate contract table output"):
        compile_dag_catalog(driver, strict=True)


def test_io_surfaces_include_reads_and_writes() -> None:
    """IO surfaces should include derived reads and writes."""
    driver = _build_driver(_module_with_io_surface())
    catalog = compile_dag_catalog(driver, strict=True)

    surface = catalog.io_surfaces.get("alpha")
    if surface is None:
        pytest.fail("Expected IO surface for alpha target")

    assert isinstance(surface, IOSurface)
    assert {read.table_key for read in surface.reads} == {"core.source"}
    assert {write.table_key for write in surface.table_writes} == {"analytics.alpha_out"}
    assert {write.artifact_name for write in surface.artifact_writes} == {"alpha_meta"}
