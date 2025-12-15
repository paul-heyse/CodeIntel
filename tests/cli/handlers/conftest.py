"""Shared fixtures for CLI handler tests.

These fixtures use real gateways via ``CliTestContext`` and typed handler harnesses
from ``tests/_helpers/harnesses``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._helpers.cli_context import create_cli_test_context
from tests._helpers.harnesses.cli import CliHandlerHarness
from tests._helpers.harnesses.datasets import dataset_handler_harness
from tests._helpers.harnesses.docs import docs_handler_harness
from tests._helpers.harnesses.storage import storage_macro_harness
from tests._helpers.seeds import CORE_PACK, GRAPH_PACK, SUBSYSTEM_PACK

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tests._helpers.cli_context import CliTestContext
    from tests._helpers.harnesses.datasets import DatasetHandlerHarness
    from tests._helpers.harnesses.docs import DocsHandlerHarness
    from tests._helpers.harnesses.storage import StorageHandlerHarness


@pytest.fixture
def cli_handler_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide a CliTestContext with core seeds for handler tests.

    Yields
    ------
    CliTestContext
        CLI test context with a seeded gateway and runtime.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def graph_cli_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide a CliTestContext with graph seeds for handler tests.

    Yields
    ------
    CliTestContext
        CLI test context with core and graph seeds.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK, GRAPH_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def subsystem_cli_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide a CliTestContext with subsystem seeds for handler tests.

    Yields
    ------
    CliTestContext
        CLI test context with core and subsystem seeds.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK, SUBSYSTEM_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def cli_handler_harness_fixture(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Provide a CliHandlerHarness with core seeds.

    Yields
    ------
    CliHandlerHarness
        Handler harness bound to a seeded CLI test context.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    harness = CliHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()


@pytest.fixture
def dataset_handler_harness_fixture(tmp_path: Path) -> Iterator[DatasetHandlerHarness]:
    """Provide dataset handler harness with real dependencies.

    Yields
    ------
    DatasetHandlerHarness
        Dataset harness with seeded runtime and gateway dependencies.
    """
    with dataset_handler_harness(tmp_path) as harness:
        yield harness


@pytest.fixture
def docs_handler_harness_fixture(tmp_path: Path) -> Iterator[DocsHandlerHarness]:
    """Provide docs handler harness with runtime stub and gateway.

    Yields
    ------
    DocsHandlerHarness
        Docs harness with a stub runtime and gateway.
    """
    with docs_handler_harness(tmp_path) as harness:
        yield harness


@pytest.fixture
def storage_macro_harness_fixture(tmp_path: Path) -> Iterator[StorageHandlerHarness]:
    """Provide storage handler harness with seeded macros/profiles.

    Yields
    ------
    StorageHandlerHarness
        Storage harness with seeded macro and profile artifacts.
    """
    with storage_macro_harness(tmp_path) as harness:
        yield harness
