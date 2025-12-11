"""Harness for dataset handler tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from codeintel.cli.handlers.datasets import DatasetDependencies
from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.storage.validation import collect_contract_issues
from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.seeds import CORE_PACK


@dataclass
class _Runtime:
    """Lightweight runtime stub exposing gateway and root."""

    gateway: object
    root: object


@dataclass
class DatasetHandlerHarness:
    """Harness encapsulating context and dependencies for dataset handlers."""

    ctx: CliTestContext
    deps: DatasetDependencies

    @contextmanager
    def command_context(self, params: dict[str, object]) -> Iterator[object]:
        """Yield a CommandContext bound to the underlying gateway."""
        with self.ctx.command_context(params) as cmd_ctx:
            # Attach runtime stub for handlers that expect ctx.runtime
            cmd_ctx._runtime = _Runtime(  # type: ignore[attr-defined]
                gateway=self.ctx.gateway,
                root=self.ctx.repo_root,
            )
            yield cmd_ctx


@contextmanager
def dataset_handler_harness(tmp_path: Path) -> Iterator[DatasetHandlerHarness]:
    """Provide dataset handler harness with real dependencies."""
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    deps = DatasetDependencies(
        runtime_builder=lambda _ctx: _Runtime(gateway=ctx.gateway, root=ctx.repo_root),
        contracts_provider=get_dataset_contracts_by_table_key,
        issue_collector=lambda con: collect_contract_issues(con),
    )
    harness = DatasetHandlerHarness(ctx=ctx, deps=deps)
    try:
        yield harness
    finally:
        ctx.close()
