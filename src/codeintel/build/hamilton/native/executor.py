"""Native target executor for Hamilton Phase 3.

This module provides NativeTargetExecutor, which consolidates the boilerplate
patterns found across native Hamilton targets:
- Input hash computation
- Skip check logic
- Timing and error handling
- Record creation and manifest persistence
- Async execution support

Example
-------
>>> executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
>>> if executor.should_skip():
...     return executor.skip()
>>> return executor.execute(lambda: compute_and_materialize())

For async targets:
>>> async def async_example():
...     executor = NativeTargetExecutor.for_target(env, graph, "scip")
...     if executor.should_skip():
...         return executor.skip()
...     return await executor.execute_async(async_compute)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.errors import TargetNotFoundError
from codeintel.build.hamilton.run_records import (
    NativeRunInfo,
    RunRecordInputs,
    compute_target_input_hash,
    create_run_record,
    options_hash_for_target,
    save_manifest,
    should_skip_native_target,
)
from codeintel.build.hashing import InputHashOptions
from codeintel.core.errors import CodeIntelError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.run_records import TargetRunRecord
    from codeintel.build.targets import OutputTarget, TargetGraph


# Exceptions that should be caught and converted to failed records
_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
    CodeIntelError,
)


@dataclass
class NativeTargetExecutor:
    """Handle skip-check, timing, and record creation for native targets.

    This class consolidates the common patterns found across native Hamilton
    targets, reducing boilerplate and ensuring consistent behavior. It supports
    both synchronous and asynchronous execution patterns.

    Attributes
    ----------
    env
        Build environment with gateway, snapshot, and configuration.
    target
        The target being executed.
    input_hash
        Computed input hash for cache invalidation.
    options_hash
        Optional configuration options hash.

    Examples
    --------
    Synchronous execution:

    >>> executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
    >>> if executor.should_skip():
    ...     return executor.skip()
    >>> return executor.execute(lambda: {"analytics.table": 100})

    Asynchronous execution:

    >>> async def run_async():
    ...     executor = NativeTargetExecutor.for_target(env, graph, "scip")
    ...     if executor.should_skip():
    ...         return executor.skip()
    ...     return await executor.execute_async(async_index_fn)
    """

    env: BuildEnv
    target: OutputTarget
    input_hash: str
    options_hash: str | None = None
    _start_time: float = field(default=0.0, repr=False)

    @classmethod
    def for_target(
        cls,
        env: BuildEnv,
        graph: TargetGraph,
        target_name: str,
        *,
        options_hash: str | None = None,
    ) -> NativeTargetExecutor:
        """Create an executor for a named target.

        Parameters
        ----------
        env
            Build environment with gateway and snapshot.
        graph
            Target graph for looking up the target.
        target_name
            Name of the target to execute.
        options_hash
            Optional configuration options hash.

        Returns
        -------
        NativeTargetExecutor
            Configured executor ready for skip check or execution.

        Raises
        ------
        TargetNotFoundError
            If target is not found in the graph.

        Examples
        --------
        >>> executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
        >>> executor.target.name
        'risk_factors'
        """
        target = graph.get(target_name)
        if target is None:
            raise TargetNotFoundError(target_name, list(graph))

        resolved_options_hash = options_hash
        if resolved_options_hash is None:
            resolved_options_hash = options_hash_for_target(env, target_name)

        hash_options = InputHashOptions(
            options_hash=resolved_options_hash,
            manifests=env.manifest_index,
        )
        input_hash = compute_target_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )

        return cls(
            env=env,
            target=target,
            input_hash=input_hash,
            options_hash=resolved_options_hash,
        )

    def should_skip(self) -> bool:
        """Check if the target can be skipped based on manifest.

        Returns
        -------
        bool
            True if target can be skipped (manifest matches), False otherwise.

        Examples
        --------
        >>> if executor.should_skip():
        ...     return executor.skip()
        """
        return should_skip_native_target(
            self.env,
            self.target,
            self.input_hash,
            options_hash=self.options_hash,
        )

    def skip(self) -> TargetRunRecord:
        """Create a skipped record for this target.

        Call this when `should_skip()` returns True.

        Returns
        -------
        TargetRunRecord
            Record with status="skipped".

        Examples
        --------
        >>> if executor.should_skip():
        ...     return executor.skip()
        """
        run = NativeRunInfo(
            input_hash=self.input_hash,
            options_hash=self.options_hash,
            duration_ms=0.0,
        )
        return create_run_record(
            self.target,
            "skipped",
            self.input_hash,
            inputs=RunRecordInputs(env=self.env, run=run),
        )

    def execute(
        self,
        compute_fn: Callable[[], dict[str, int]],
    ) -> TargetRunRecord:
        """Execute with timing, error handling, and manifest persistence.

        This method:
        1. Records the start time
        2. Calls the compute function
        3. Handles any exceptions
        4. Creates the appropriate record (success or failed)
        5. Persists the manifest on success

        Parameters
        ----------
        compute_fn
            Function that performs the computation and returns row counts.
            Should return a dict mapping table keys to row counts.

        Returns
        -------
        TargetRunRecord
            Record with status="succeeded" or status="failed".

        Examples
        --------
        >>> def compute() -> dict[str, int]:
        ...     ref = materialize_table(ctx, "analytics.table", expr)
        ...     return {ref.table_key: ref.row_count}
        >>> return executor.execute(compute)

        Raises
        ------
        KeyboardInterrupt
            Propagated if execution is interrupted by the user.
        SystemExit
            Propagated if the interpreter is exiting.
        GeneratorExit
            Propagated if a generator close is requested.
        """
        start = time.perf_counter()
        try:
            row_counts = compute_fn()
        except _RECOVERABLE_EXCEPTIONS as exc:
            return self._create_failed_record(start, exc)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise

        return self._create_success_record(start, row_counts)

    async def execute_async(
        self,
        compute_fn: Callable[[], Awaitable[dict[str, int]]],
    ) -> TargetRunRecord:
        """Execute async compute function with timing and error handling.

        This method is the async equivalent of execute(). Use this for targets
        that need to perform I/O operations that benefit from async execution,
        such as calling external tools (SCIP, Pyright) or network operations.

        Hamilton supports async execution via AsyncGraphAdapter. This method
        allows native targets to leverage async patterns while maintaining
        the same skip-check, timing, and manifest persistence behavior.

        Parameters
        ----------
        compute_fn
            Async function that performs the computation and returns row counts.
            Should return a dict mapping table keys to row counts.

        Returns
        -------
        TargetRunRecord
            Record with status="succeeded" or status="failed".

        Examples
        --------
        >>> async def async_compute() -> dict[str, int]:
        ...     result = await scip_indexer.index(repo_root)
        ...     ref = persist_result(result)
        ...     return {ref.table_key: ref.row_count}
        >>> return await executor.execute_async(async_compute)

        Raises
        ------
        KeyboardInterrupt
            Propagated if execution is interrupted by the user.
        SystemExit
            Propagated if the interpreter is exiting.
        GeneratorExit
            Propagated if a generator close is requested.
        """
        start = time.perf_counter()
        try:
            row_counts = await compute_fn()
        except _RECOVERABLE_EXCEPTIONS as exc:
            return self._create_failed_record(start, exc)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise

        return self._create_success_record(start, row_counts)

    def fail(self, error: Exception) -> TargetRunRecord:
        """Create a failed record for an error that occurred before execution.

        Use this for errors that happen during setup or validation,
        before the main compute function is called.

        Parameters
        ----------
        error
            The exception that caused the failure.

        Returns
        -------
        TargetRunRecord
            Record with status="failed".

        Examples
        --------
        >>> try:
        ...     validate_inputs()
        ... except ValueError as e:
        ...     return executor.fail(e)
        """
        run = NativeRunInfo(
            input_hash=self.input_hash,
            options_hash=self.options_hash,
            duration_ms=0.0,
        )
        return create_run_record(
            self.target,
            "failed",
            self.input_hash,
            inputs=RunRecordInputs(run=run, error=error),
        )

    def _create_failed_record(
        self,
        start: float,
        exc: Exception,
    ) -> TargetRunRecord:
        """Create a failed record from an exception.

        Parameters
        ----------
        start
            Start time from time.perf_counter().
        exc
            The exception that caused the failure.

        Returns
        -------
        TargetRunRecord
            Record with status="failed".
        """
        duration_ms = (time.perf_counter() - start) * 1000
        run = NativeRunInfo(
            input_hash=self.input_hash,
            options_hash=self.options_hash,
            duration_ms=duration_ms,
        )
        return create_run_record(
            self.target,
            "failed",
            self.input_hash,
            inputs=RunRecordInputs(run=run, error=exc),
        )

    def _create_success_record(
        self,
        start: float,
        row_counts: dict[str, int],
    ) -> TargetRunRecord:
        """Create a success record and persist manifest.

        Parameters
        ----------
        start
            Start time from time.perf_counter().
        row_counts
            Dict mapping table keys to row counts.

        Returns
        -------
        TargetRunRecord
            Record with status="succeeded".
        """
        expected = set(self.target.contract.table_keys)
        observed = set(row_counts)
        if observed != expected:
            missing = tuple(sorted(expected - observed))
            extra = tuple(sorted(observed - expected))
            parts: list[str] = []
            if missing:
                parts.append(f"missing table_counts for {missing}")
            if extra:
                parts.append(f"unexpected table_counts for {extra}")
            message = "; ".join(parts) if parts else "Invalid table_counts for target"
            duration_ms = (time.perf_counter() - start) * 1000
            run = NativeRunInfo(
                input_hash=self.input_hash,
                options_hash=self.options_hash,
                duration_ms=duration_ms,
                row_counts=None,
            )
            return create_run_record(
                self.target,
                "failed",
                self.input_hash,
                inputs=RunRecordInputs(env=self.env, run=run, error=ValueError(message)),
            )

        duration_ms = (time.perf_counter() - start) * 1000
        run = NativeRunInfo(
            input_hash=self.input_hash,
            options_hash=self.options_hash,
            duration_ms=duration_ms,
            row_counts=row_counts,
        )
        record = create_run_record(
            self.target,
            "succeeded",
            self.input_hash,
            inputs=RunRecordInputs(env=self.env, run=run),
        )

        save_manifest(self.env, record)
        return record


__all__ = [
    "NativeTargetExecutor",
]
