"""Pipeline support for CLI operations.

Enable chaining operations, streaming output, and batch execution
from files for shell pipeline integration.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import yaml

from codeintel.cli.errors import ProblemDetail
from codeintel.cli.execution import get_executor
from codeintel.cli.introspection import get_operation_registry
from codeintel.cli.core import CliResult

if TYPE_CHECKING:
    from codeintel.cli.execution import ExecutionResult


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Parameters
    ----------
    stream_output
        Emit results as JSON Lines.
    fail_fast
        Stop on first error.
    continue_on_error
        Continue batch on error.
    max_parallel
        Maximum parallel executions.
    """

    stream_output: bool = False
    fail_fast: bool = False
    continue_on_error: bool = True
    max_parallel: int = 1


class StreamingRenderer:
    """Render results as JSON Lines for streaming.

    Parameters
    ----------
    output
        Output stream.
    """

    def __init__(self, output: TextIO | None = None) -> None:
        """Initialize renderer."""
        self._output = output or sys.stdout

    def emit(self, result: CliResult[Any]) -> None:
        """Emit result as JSON line.

        Parameters
        ----------
        result
            Result to emit.
        """
        data = result.to_dict()
        self._output.write(json.dumps(data))
        self._output.write("\n")
        self._output.flush()

    def emit_progress(self, index: int, total: int, operation_id: str) -> None:
        """Emit progress indicator.

        Parameters
        ----------
        index
            Current index.
        total
            Total items.
        operation_id
            Current operation.
        """
        data = {
            "type": "progress",
            "index": index,
            "total": total,
            "operation_id": operation_id,
        }
        self._output.write(json.dumps(data))
        self._output.write("\n")
        self._output.flush()

    def emit_summary(self, summary: dict[str, Any]) -> None:
        """Emit summary at end of batch.

        Parameters
        ----------
        summary
            Summary data.
        """
        data = {"type": "summary", **summary}
        self._output.write(json.dumps(data))
        self._output.write("\n")
        self._output.flush()


@dataclass
class BatchOperation:
    """Single operation in a batch.

    Parameters
    ----------
    operation_id
        Operation to execute.
    params
        Operation parameters.
    name
        Optional name for tracking.
    """

    operation_id: str
    params: dict[str, Any]
    name: str | None = None


@dataclass
class BatchItemResult:
    """Result of a single batch item.

    Parameters
    ----------
    operation
        The batch operation.
    result
        The execution result.
    index
        Position in batch.
    """

    operation: BatchOperation
    result: CliResult[Any]
    index: int


@dataclass
class BatchResult:
    """Result of batch execution.

    Parameters
    ----------
    total
        Total operations.
    succeeded
        Successful operations.
    failed
        Failed operations.
    skipped
        Skipped operations (after fail_fast).
    items
        Individual results.
    """

    total: int
    succeeded: int
    failed: int
    skipped: int
    items: list[BatchItemResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.succeeded / self.total if self.total > 0 else 0.0,
        }


def load_batch(path: Path) -> list[BatchOperation]:
    """Load batch operations from file.

    Parameters
    ----------
    path
        Path to batch file (YAML or JSON).

    Returns
    -------
    list[BatchOperation]
        Operations to execute.
    """
    content = path.read_text(encoding="utf-8")

    data = yaml.safe_load(content) if path.suffix in {".yaml", ".yml"} else json.loads(content)

    if not isinstance(data, dict):
        return []

    operations: list[BatchOperation] = []
    raw_ops = data.get("operations", [])
    if not isinstance(raw_ops, list):
        return []

    for item in raw_ops:
        if not isinstance(item, dict):
            continue
        op_id = item.get("operation")
        if not isinstance(op_id, str):
            continue

        params = item.get("params", {})
        if not isinstance(params, dict):
            params = {}

        name = item.get("name")
        operations.append(
            BatchOperation(
                operation_id=op_id,
                params=params,
                name=str(name) if name is not None else None,
            )
        )
    return operations


def execute_batch(
    operations: list[BatchOperation],
    config: PipelineConfig | None = None,
) -> BatchResult:
    """Execute batch of operations.

    Parameters
    ----------
    operations
        Operations to execute.
    config
        Pipeline configuration.

    Returns
    -------
    BatchResult
        Batch execution result.
    """
    config = config or PipelineConfig()
    executor = get_executor()
    registry = get_operation_registry()
    renderer = StreamingRenderer() if config.stream_output else None

    items: list[BatchItemResult] = []
    succeeded = 0
    failed = 0
    skipped = 0
    stopped = False

    for i, batch_op in enumerate(operations):
        if stopped:
            skipped += 1
            continue

        # Emit progress
        if renderer:
            renderer.emit_progress(i, len(operations), batch_op.operation_id)

        # Get operation spec
        spec = registry.get(batch_op.operation_id)
        if spec is None:
            result: CliResult[Any] = CliResult.fail(
                ProblemDetail(
                    type="urn:codeintel:cli:operation/not-found",
                    title="Operation Not Found",
                    detail=f"Unknown operation: {batch_op.operation_id}",
                    status=404,
                )
            )
            exec_result: ExecutionResult[Any] | None = None
        else:
            exec_result = executor.execute(
                spec,
                batch_op.params,
                render=False,
            )
            result = exec_result.result

        if result.success:
            succeeded += 1
        else:
            failed += 1
            if config.fail_fast:
                stopped = True

        items.append(BatchItemResult(operation=batch_op, result=result, index=i))

        if renderer:
            renderer.emit(result)

    batch_result = BatchResult(
        total=len(operations),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        items=items,
    )

    if renderer:
        renderer.emit_summary(batch_result.to_dict())

    return batch_result


def read_stdin_operations() -> Iterator[BatchOperation]:
    """Read operations from stdin (JSON Lines).

    Yield operations one at a time from stdin.

    Yields
    ------
    BatchOperation
        Operations from stdin.
    """
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        data = json.loads(stripped)
        if not isinstance(data, dict):
            continue
        op_id = data.get("operation")
        if not isinstance(op_id, str):
            continue

        params = data.get("params", {})
        if not isinstance(params, dict):
            params = {}

        name = data.get("name")
        yield BatchOperation(
            operation_id=op_id,
            params=params,
            name=str(name) if name is not None else None,
        )


def stream_results(
    results: Iterator[CliResult[Any]],
    output: TextIO | None = None,
) -> int:
    """Stream results to output as JSON Lines.

    Parameters
    ----------
    results
        Results to stream.
    output
        Output stream (defaults to stdout).

    Returns
    -------
    int
        Number of results streamed.
    """
    renderer = StreamingRenderer(output=output)
    count = 0
    for result in results:
        renderer.emit(result)
        count += 1
    return count


__all__ = [
    "BatchItemResult",
    "BatchOperation",
    "BatchResult",
    "PipelineConfig",
    "StreamingRenderer",
    "execute_batch",
    "load_batch",
    "read_stdin_operations",
    "stream_results",
]
