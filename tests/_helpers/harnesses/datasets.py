"""Harness for dataset handler tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from codeintel.cli.handlers.datasets import DatasetDependencies
from codeintel.config.datasets.contracts import DatasetContract
from codeintel.storage.validation import collect_contract_issues
from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.seeds import CORE_PACK


@dataclass
class _Runtime:
    """Lightweight runtime stub exposing gateway and root."""

    gateway: object
    root: object

    @property
    def runtime(self) -> _Runtime:
        """Return self to satisfy CommandContext.runtime access."""
        return self


@dataclass
class DatasetHandlerHarness:
    """Harness encapsulating context and dependencies for dataset handlers."""

    ctx: CliTestContext
    deps: DatasetDependencies

    @contextmanager
    def command_context(self, params: dict[str, object]) -> Iterator[object]:
        """Yield a CommandContext bound to the underlying gateway.

        Yields
        ------
        object
            CommandContext configured for dataset handlers.
        """
        with self.ctx.command_context(params) as cmd_ctx:
            # Attach runtime stub for handlers that expect ctx.runtime
            cmd_ctx.__dict__["_runtime"] = _Runtime(
                gateway=self.ctx.gateway,
                root=self.ctx.repo_root,
            )
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

    def _build_runtime(_ctx: object) -> _Runtime:
        return _Runtime(gateway=ctx.gateway, root=ctx.repo_root)

    deps = DatasetDependencies(
        runtime_builder=_build_runtime,
        contracts_provider=lambda: {contract.name: contract},
        issue_collector=collect_contract_issues,
    )
    harness = DatasetHandlerHarness(ctx=ctx, deps=deps)
    try:
        yield harness
    finally:
        ctx.close()
