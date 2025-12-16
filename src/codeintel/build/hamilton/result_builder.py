"""Custom ResultBuilder for structured build output.

This module provides a BuildResultBuilder that aggregates build execution
outputs into a structured format suitable for reporting, serialization,
and downstream processing.

The ResultBuilder integrates with Hamilton's lifecycle API to transform
raw node outputs into a coherent build result.

Examples
--------
Using BuildResultBuilder for structured output:

>>> from codeintel.build.hamilton.result_builder import BuildResultBuilder
>>> result_builder = BuildResultBuilder()
>>> dr = driver.Builder().with_modules(modules).with_adapters(result_builder).build()
>>> result = dr.execute(["t__metrics__materialize"])
>>> print(result.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from hamilton.lifecycle import ResultBuilder

__all__ = [
    "BuildExecutionResult",
    "BuildResultBuilder",
    "NodeResult",
    "ResultStatus",
]


class ResultStatus(Enum):
    """Status of a build result.

    Attributes
    ----------
    SUCCESS
        All nodes executed successfully.
    PARTIAL
        Some nodes failed but the build continued.
    FAILED
        Build failed due to critical error.
    SKIPPED
        Build was skipped (all targets up-to-date).
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result for a single node execution.

    Attributes
    ----------
    node_name
        Name of the executed node.
    value
        The output value from the node.
    status
        Whether the node succeeded or failed.
    error_message
        Error message if the node failed.
    duration_seconds
        Execution time in seconds (if available).
    """

    node_name: str
    value: object | None
    status: ResultStatus = ResultStatus.SUCCESS
    error_message: str | None = None
    duration_seconds: float | None = None

    @property
    def is_success(self) -> bool:
        """Check if this node result is successful."""
        return self.status == ResultStatus.SUCCESS


@dataclass
class BuildExecutionResult:
    """Structured result from a build execution.

    Aggregates results from all executed nodes into a single
    object suitable for reporting and serialization.

    Attributes
    ----------
    status
        Overall build status.
    node_results
        Individual results for each executed node.
    requested_outputs
        List of output names that were requested.
    total_duration_seconds
        Total execution time.
    start_time
        Unix timestamp when execution started.
    end_time
        Unix timestamp when execution ended.
    metadata
        Additional metadata about the execution.
    """

    status: ResultStatus
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    requested_outputs: list[str] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        """Count of successful nodes."""
        return sum(1 for r in self.node_results.values() if r.is_success)

    @property
    def failure_count(self) -> int:
        """Count of failed nodes."""
        return sum(1 for r in self.node_results.values() if not r.is_success)

    def get_output(self, name: str) -> object | None:
        """Get output value by name.

        Parameters
        ----------
        name
            Name of the output to retrieve.

        Returns
        -------
        object | None
            The output value (or None when values are not included).

        Raises
        ------
        KeyError
            If the requested output name is not found.
        """
        if name not in self.node_results:
            msg = f"Output '{name}' not found in results"
            raise KeyError(msg)
        return self.node_results[name].value

    def get_outputs(self) -> dict[str, object | None]:
        """Get all output values as a dictionary.

        Returns
        -------
        dict[str, object | None]
            Mapping from node name to output value.
        """
        return {name: r.value for name, r in self.node_results.items()}

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns
        -------
        str
            Summary of the build execution.
        """
        lines = [
            f"Build Status: {self.status.value.upper()}",
            f"Nodes: {self.success_count} succeeded, {self.failure_count} failed",
            f"Duration: {self.total_duration_seconds:.2f}s",
        ]
        if self.requested_outputs:
            lines.append(f"Requested: {', '.join(self.requested_outputs)}")

        if self.failure_count > 0:
            lines.append("\nFailed nodes:")
            for name, result in self.node_results.items():
                if not result.is_success:
                    lines.append(f"  - {name}: {result.error_message}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, object]
            Dictionary suitable for JSON serialization.
        """
        return {
            "status": self.status.value,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_duration_seconds": self.total_duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "requested_outputs": self.requested_outputs,
            "node_results": {
                name: {
                    "status": r.status.value,
                    "error_message": r.error_message,
                    "duration_seconds": r.duration_seconds,
                    # Note: value not serialized - may not be JSON-safe
                }
                for name, r in self.node_results.items()
            },
            "metadata": self.metadata,
        }


class BuildResultBuilder(ResultBuilder):
    """ResultBuilder that aggregates outputs into BuildExecutionResult.

    Transforms raw Hamilton execution outputs into a structured
    BuildExecutionResult with metadata about the execution.

    Parameters
    ----------
    include_values
        If True, include node output values in results.
        Set to False for large outputs to reduce memory.
    metadata
        Additional metadata to include in the result.

    Examples
    --------
    >>> builder = BuildResultBuilder(include_values=True)
    >>> dr = driver.Builder().with_adapters(builder).build()
    >>> result = dr.execute(["output1", "output2"])
    >>> assert isinstance(result, BuildExecutionResult)
    >>> print(result.summary())
    """

    def __init__(
        self,
        *,
        include_values: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialize the result builder."""
        self.include_values = include_values
        self.metadata = metadata or {}
        self._start_time = 0.0

    def build_result(self, **outputs: object) -> BuildExecutionResult:
        """Build the structured result from outputs.

        Parameters
        ----------
        **outputs
            Keyword arguments with node outputs.

        Returns
        -------
        BuildExecutionResult
            Structured build result.
        """
        end_time = time.time()

        # Create node results
        node_results: dict[str, NodeResult] = {}
        has_failures = False

        for name, value in outputs.items():
            # Check if value indicates an error
            is_error = isinstance(value, Exception)
            if is_error:
                has_failures = True
                node_results[name] = NodeResult(
                    node_name=name,
                    value=None,
                    status=ResultStatus.FAILED,
                    error_message=str(value),
                )
            else:
                node_results[name] = NodeResult(
                    node_name=name,
                    value=value if self.include_values else None,
                    status=ResultStatus.SUCCESS,
                )

        # Determine overall status
        if not node_results:
            status = ResultStatus.SKIPPED
        elif has_failures:
            status = ResultStatus.PARTIAL if len(node_results) > 1 else ResultStatus.FAILED
        else:
            status = ResultStatus.SUCCESS

        # Calculate duration
        total_duration = end_time - self._start_time if self._start_time else 0.0

        return BuildExecutionResult(
            status=status,
            node_results=node_results,
            requested_outputs=list(outputs.keys()),
            total_duration_seconds=total_duration,
            start_time=self._start_time,
            end_time=end_time,
            metadata=self.metadata,
        )

    @staticmethod
    def input_types() -> list[type]:
        """Return accepted input types.

        Returns
        -------
        list[type]
            List of accepted input types.
        """
        return [object]

    @staticmethod
    def output_type() -> type:
        """Return the output type.

        Returns
        -------
        type
            BuildExecutionResult type.
        """
        return BuildExecutionResult

    def start_timing(self) -> None:
        """Record start time for duration calculation.

        Call this before driver.execute() to enable
        accurate duration tracking.
        """
        self._start_time = time.time()


class DictResultBuilder(ResultBuilder):
    """Simple ResultBuilder that returns outputs as a dictionary.

    A minimal result builder for cases where structured results
    aren't needed.

    Examples
    --------
    >>> builder = DictResultBuilder()
    >>> dr = driver.Builder().with_adapters(builder).build()
    >>> result = dr.execute(["output"])
    >>> assert isinstance(result, dict)
    """

    @staticmethod
    def build_result(**outputs: object) -> dict[str, object]:
        """Return outputs as a dictionary.

        Parameters
        ----------
        **outputs
            Keyword arguments with node outputs.

        Returns
        -------
        dict[str, object]
            Dictionary of outputs.
        """
        return dict(outputs)

    @staticmethod
    def input_types() -> list[type]:
        """Return accepted input types.

        Returns
        -------
        list[type]
            Types accepted as inputs for this result builder.
        """
        return [object]

    @staticmethod
    def output_type() -> type:
        """Return the output type.

        Returns
        -------
        type
            The concrete output container type.
        """
        return dict
