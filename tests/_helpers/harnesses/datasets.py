"""Harness for dataset handler tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.schemas.dataset_service import DocsFilterMode, ReadOnlyFilterMode
from codeintel.cli.handlers.datasets import DatasetDependencies
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.storage.validation import collect_contract_issues
from tests._helpers.cli_context import create_cli_test_context
from tests._helpers.seeds import CORE_PACK

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from codeintel.cli.context import CommandContext
    from tests._helpers.cli_context import CliTestContext


@dataclass
class DatasetHandlerHarness:
    """Harness encapsulating context and dependencies for dataset handlers."""

    ctx: CliTestContext
    deps: DatasetDependencies

    @contextmanager
    def command_context(self, params: Mapping[str, object]) -> Iterator[CommandContext]:
        """Yield a CommandContext bound to the underlying gateway.

        Yields
        ------
        CommandContext
            CommandContext configured for dataset handlers.
        """
        with self.ctx.command_context(dict(params)) as cmd_ctx:
            yield cmd_ctx


@contextmanager
def dataset_handler_harness(tmp_path: Path) -> Iterator[DatasetHandlerHarness]:
    """Provide dataset handler harness with real dependencies.

    Yields
    ------
    DatasetHandlerHarness
        Harness exposing command context and dependencies.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)

    contract = DatasetContract(
        table_key="test.table",
        name="test_dataset",
        schema=None,
        description="Test dataset for handler harness",
        owner_package="core",
    )

    def _list_datasets(
        *,
        docs_view: DocsFilterMode = "include",
        read_only: ReadOnlyFilterMode = "include",
    ) -> list[DatasetContract]:
        _ = (docs_view, read_only)
        return [contract]

    deps = DatasetDependencies(
        list_datasets=_list_datasets,
        issue_collector=collect_contract_issues,
    )
    harness = DatasetHandlerHarness(ctx=ctx, deps=deps)
    try:
        yield harness
    finally:
        ctx.close()
