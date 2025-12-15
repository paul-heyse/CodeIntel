"""Native target executor for Hamilton Phase 3.

This module provides NativeTargetExecutor, which consolidates the boilerplate
patterns found across native Hamilton targets:
- Input hash computation
- Skip check logic
- Timing and error handling
- Record creation and manifest persistence

Example
-------
>>> executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
>>> if executor.should_skip():
...     return executor.skip()
>>> return executor.execute(lambda: compute_and_materialize())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.errors import TargetNotFoundError
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    RunRecordInputs,
    create_run_record,
    save_manifest,
    should_skip_native_target,
)
from codeintel.build.hashing import compute_input_hash
from codeintel.core.errors import CodeIntelError

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import OutputTarget, TargetGraph


@dataclass
class NativeTargetExecutor:
    """Handle skip-check, timing, and record creation for native targets.

    This class consolidates the common patterns found across native Hamilton
    targets, reducing boilerplate and ensuring consistent behavior.

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
    >>> executor = NativeTargetExecutor.for_target(env, graph, "risk_factors")
    >>> if executor.should_skip():
    ...     return executor.skip()
    >>> return executor.execute(lambda: {"analytics.table": 100})
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

        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=options_hash,
            manifests=env.manifest_index,
        )

        return cls(
            env=env,
            target=target,
            input_hash=input_hash,
            options_hash=options_hash,
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
        return should_skip_native_target(self.env, self.target, self.input_hash)

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
        """
        start = time.perf_counter()
        try:
            row_counts = compute_fn()
        except (
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            OSError,
            CodeIntelError,
        ) as exc:
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
        except BaseException as exc:
            # Re-raise system exceptions (KeyboardInterrupt, SystemExit, GeneratorExit)
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
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
                inputs=RunRecordInputs(
                    run=run,
                    error=exc if isinstance(exc, Exception) else RuntimeError(str(exc)),
                ),
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


__all__ = [
    "NativeTargetExecutor",
]
