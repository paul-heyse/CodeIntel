"""Pipeline scheduler for DAG-based dataset execution.

This module provides `PipelineScheduler` which plans and executes
dataset computations respecting their dependency relationships.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.analytics.pipeline.lineage import DatasetLineage, LineageStore, compute_table_hash
from codeintel.analytics.pipeline.protocol import (
    DatasetComputation,
    DatasetResult,
    DatasetSpec,
    PipelineContext,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionStep:
    """A single step in the execution plan.

    Attributes
    ----------
    dataset
        Name of the dataset to compute.
    spec
        Dataset specification.
    computation
        The computation implementation.
    level
        Topological level (0 = no dependencies).
    dependencies
        Names of datasets this step depends on.
    """

    dataset: str
    spec: DatasetSpec[Any]
    computation: DatasetComputation[Any]
    level: int
    dependencies: tuple[str, ...]


@dataclass
class ExecutionPlan:
    """Plan for executing a set of datasets.

    The plan contains steps organized by topological level,
    allowing parallel execution within each level.

    Attributes
    ----------
    target_datasets
        The datasets that were requested.
    steps
        Ordered list of execution steps.
    levels
        Steps grouped by topological level.
    total_datasets
        Total number of datasets to compute.
    """

    target_datasets: tuple[str, ...]
    steps: list[ExecutionStep] = field(default_factory=list)
    levels: dict[int, list[ExecutionStep]] = field(default_factory=dict)
    total_datasets: int = 0

    def add_step(self, step: ExecutionStep) -> None:
        """Add a step to the plan.

        Parameters
        ----------
        step
            Step to add.
        """
        self.steps.append(step)
        if step.level not in self.levels:
            self.levels[step.level] = []
        self.levels[step.level].append(step)
        self.total_datasets += 1

    @property
    def max_level(self) -> int:
        """Return the maximum topological level.

        Returns
        -------
        int
            Highest level number.
        """
        return max(self.levels.keys()) if self.levels else 0

    def iter_by_level(self) -> list[list[ExecutionStep]]:
        """Return steps grouped by level in order.

        Returns
        -------
        list[list[ExecutionStep]]
            Steps for each level, in execution order.
        """
        return [self.levels.get(i, []) for i in range(self.max_level + 1)]


@dataclass
class PipelineReport:
    """Report from executing a pipeline.

    Attributes
    ----------
    run_id
        Unique identifier for this run.
    plan
        The execution plan that was run.
    results
        Results for each dataset.
    success
        Whether all datasets succeeded.
    total_rows
        Total rows computed across all datasets.
    total_duration_ms
        Total execution time in milliseconds.
    started_at
        When execution started.
    completed_at
        When execution completed.
    errors
        Error messages from failed datasets.
    """

    run_id: str
    plan: ExecutionPlan
    results: dict[str, DatasetResult[Any]] = field(default_factory=dict)
    success: bool = True
    total_rows: int = 0
    total_duration_ms: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)

    def record_result(self, result: DatasetResult[Any]) -> None:
        """Record a dataset result.

        Parameters
        ----------
        result
            Result to record.
        """
        self.results[result.spec.name] = result
        self.total_rows += result.row_count
        self.total_duration_ms += result.duration_ms

        if not result.success:
            self.success = False
            if result.error:
                self.errors.append(f"{result.spec.name}: {result.error}")


class PipelineScheduler:
    """Schedules and executes dataset computations.

    The scheduler builds an execution plan by analyzing dataset
    dependencies and executes them in topological order.

    Example
    -------
    >>> scheduler = PipelineScheduler()
    >>> scheduler.register(my_dataset_computation)
    >>> plan = scheduler.plan(["analytics.function_metrics"])
    >>> report = scheduler.execute(plan, ctx)
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        self._computations: dict[str, DatasetComputation[Any]] = {}
        self._specs: dict[str, DatasetSpec[Any]] = {}

    def register(self, computation: DatasetComputation[Any]) -> None:
        """Register a dataset computation.

        Parameters
        ----------
        computation
            Computation to register.
        """
        spec = computation.spec
        self._computations[spec.name] = computation
        self._specs[spec.name] = spec
        log.debug("Registered dataset: %s", spec.name)

    def unregister(self, name: str) -> None:
        """Unregister a dataset computation.

        Parameters
        ----------
        name
            Name of the dataset to unregister.
        """
        self._computations.pop(name, None)
        self._specs.pop(name, None)

    def plan(
        self,
        targets: list[str],
        *,
        include_dependencies: bool = True,
    ) -> ExecutionPlan:
        """Build an execution plan for target datasets.

        Parameters
        ----------
        targets
            Names of datasets to compute.
        include_dependencies
            Whether to include transitive dependencies.

        Returns
        -------
        ExecutionPlan
            Plan for computing the targets.

        Raises
        ------
        ValueError
            If a target or dependency is not registered.
        """
        # Collect all datasets to compute
        to_compute: set[str] = set(targets)
        if include_dependencies:
            to_compute = self._collect_dependencies(targets)

        # Validate all datasets exist
        for name in to_compute:
            if name not in self._computations:
                message = f"Dataset not registered: {name}"
                raise ValueError(message)

        # Compute topological levels
        levels = self._compute_levels(to_compute)

        # Build plan
        plan = ExecutionPlan(target_datasets=tuple(targets))

        for level, datasets in sorted(levels.items()):
            for name in sorted(datasets):
                spec = self._specs[name]
                computation = self._computations[name]
                deps = tuple(d for d in spec.inputs if d in to_compute)

                step = ExecutionStep(
                    dataset=name,
                    spec=spec,
                    computation=computation,
                    level=level,
                    dependencies=deps,
                )
                plan.add_step(step)

        log.info(
            "Built execution plan: %d datasets, %d levels",
            plan.total_datasets,
            plan.max_level + 1,
        )
        return plan

    def execute(
        self,
        plan: ExecutionPlan,
        ctx: PipelineContext,
        *,
        fail_fast: bool = True,
        skip_unchanged: bool = False,
    ) -> PipelineReport:
        """Execute a pipeline plan.

        Parameters
        ----------
        plan
            Execution plan to run.
        ctx
            Pipeline context with gateway and config.
        fail_fast
            Whether to stop on first failure.
        skip_unchanged
            Whether to skip datasets with unchanged inputs.

        Returns
        -------
        PipelineReport
            Report with results for all datasets.
        """
        report = PipelineReport(run_id=ctx.run_id, plan=plan)
        lineage_store = LineageStore(ctx.gateway)

        # Cache of computed outputs for passing between datasets
        outputs: dict[str, Any] = {}

        # Execute level by level
        for level_steps in plan.iter_by_level():
            for step in level_steps:
                if not report.success and fail_fast:
                    # Skip remaining steps
                    result = DatasetResult(
                        spec=step.spec,
                        success=False,
                        error="Skipped due to previous failure",
                    )
                    report.record_result(result)
                    continue

                # Check if recomputation needed
                if skip_unchanged:
                    input_hashes = self._compute_input_hashes(
                        ctx.gateway, step.spec, ctx.snapshot.repo, ctx.snapshot.commit
                    )
                    if not lineage_store.needs_recompute(step.dataset, input_hashes):
                        log.info("Skipping unchanged dataset: %s", step.dataset)
                        continue

                # Execute computation
                result = self._execute_step(step, ctx, outputs)
                report.record_result(result)

                # Record lineage
                if result.success:
                    self._record_lineage(ctx, lineage_store, step, result, outputs)

        report.completed_at = datetime.now(tz=UTC)
        log.info(
            "Pipeline completed: %d datasets, %d rows, %.1fms",
            len(report.results),
            report.total_rows,
            report.total_duration_ms,
        )
        return report

    def _collect_dependencies(self, targets: list[str]) -> set[str]:
        """Collect all transitive dependencies for targets.

        Parameters
        ----------
        targets
            Target dataset names.

        Returns
        -------
        set[str]
            All datasets needed (including targets).
        """
        result: set[str] = set()
        queue = list(targets)

        while queue:
            name = queue.pop(0)
            if name in result:
                continue
            result.add(name)

            if name in self._specs:
                queue.extend(
                    dep
                    for dep in self._specs[name].inputs
                    if dep not in result and dep in self._specs
                )

        return result

    def _build_adjacency(
        self,
        datasets: set[str],
    ) -> tuple[dict[str, int], dict[str, list[str]]]:
        """Build in-degree and dependents maps for datasets.

        Returns
        -------
        tuple[dict[str, int], dict[str, list[str]]]
            Tuple of (in_degree, dependents) mappings.
        """
        in_degree: dict[str, int] = defaultdict(int)
        dependents: dict[str, list[str]] = defaultdict(list)

        for name in datasets:
            if name not in self._specs:
                continue
            for dep in self._specs[name].inputs:
                if dep in datasets:
                    in_degree[name] += 1
                    dependents[dep].append(name)

        return in_degree, dependents

    @staticmethod
    def _process_levels(
        datasets: set[str],
        in_degree: dict[str, int],
        dependents: dict[str, list[str]],
    ) -> dict[int, set[str]]:
        """Process datasets into topological levels.

        Returns
        -------
        dict[int, set[str]]
            Mapping of level to dataset names at that level.
        """
        # Initialize level 0 with no dependencies
        levels: dict[int, set[str]] = {0: {name for name in datasets if in_degree[name] == 0}}

        # Process each level
        current_level = 0
        while levels.get(current_level):
            next_level = current_level + 1
            levels[next_level] = set()

            for name in levels[current_level]:
                for dep in dependents[name]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        levels[next_level].add(dep)

            current_level = next_level

        # Remove empty final level
        levels.pop(current_level, None)
        return levels

    def _compute_levels(self, datasets: set[str]) -> dict[int, set[str]]:
        """Compute topological levels for datasets.

        Parameters
        ----------
        datasets
            Set of dataset names.

        Returns
        -------
        dict[int, set[str]]
            Mapping of level to dataset names at that level.
        """
        in_degree, dependents = self._build_adjacency(datasets)
        return self._process_levels(datasets, in_degree, dependents)

    @staticmethod
    def _execute_step(
        step: ExecutionStep,
        ctx: PipelineContext,
        outputs: dict[str, Any],
    ) -> DatasetResult[Any]:
        """Execute a single computation step.

        Parameters
        ----------
        step
            Step to execute.
        ctx
            Pipeline context.
        outputs
            Cached outputs from previous steps.

        Returns
        -------
        DatasetResult
            Result of the computation.
        """
        log.info("Computing dataset: %s", step.dataset)
        start = time.perf_counter()

        # Gather inputs
        inputs: dict[str, Any] = {}
        for dep in step.dependencies:
            if dep in outputs:
                inputs[dep] = outputs[dep]

        try:
            # Execute computation
            rows = list(step.computation.compute(ctx, inputs))
            row_count = len(rows)

            # Store output for dependents
            outputs[step.dataset] = rows

            duration = (time.perf_counter() - start) * 1000

            log.info(
                "Computed %s: %d rows in %.1fms",
                step.dataset,
                row_count,
                duration,
            )

            return DatasetResult(
                spec=step.spec,
                row_count=row_count,
                duration_ms=duration,
                success=True,
            )

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            log.exception("Failed to compute %s", step.dataset)

            return DatasetResult(
                spec=step.spec,
                duration_ms=duration,
                success=False,
                error=str(e),
            )

    @staticmethod
    def _compute_input_hashes(
        gateway: StorageGateway,
        spec: DatasetSpec[Any],
        repo: str,
        commit: str,
    ) -> dict[str, str]:
        """Compute hashes for input datasets.

        Parameters
        ----------
        gateway
            Storage gateway.
        spec
            Dataset specification.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        dict[str, str]
            Mapping of input names to their hashes.
        """
        hashes: dict[str, str] = {}
        for inp in spec.inputs:
            hashes[inp] = compute_table_hash(gateway, inp, repo=repo, commit=commit)
        return hashes

    def _record_lineage(
        self,
        ctx: PipelineContext,
        store: LineageStore,
        step: ExecutionStep,
        result: DatasetResult[Any],
        _outputs: dict[str, Any],
    ) -> None:
        """Record lineage for a completed computation.

        Parameters
        ----------
        ctx
            Pipeline context.
        store
            Lineage store.
        step
            Completed step.
        result
            Computation result.
        _outputs
            Cached outputs (reserved for future use).
        """
        input_hashes = self._compute_input_hashes(ctx.gateway, step.spec, ctx.repo, ctx.commit)

        output_hash = compute_table_hash(
            ctx.gateway,
            step.spec.primary_output,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        lineage = DatasetLineage(
            dataset=step.dataset,
            run_id=ctx.run_id,
            input_datasets=tuple(input_hashes.keys()),
            input_hashes=tuple(input_hashes.values()),
            output_hash=output_hash,
            row_count=result.row_count,
            computed_at=datetime.now(tz=UTC),
            duration_ms=result.duration_ms,
            version=step.spec.version,
        )

        store.record(lineage)


__all__ = [
    "ExecutionPlan",
    "ExecutionStep",
    "PipelineReport",
    "PipelineScheduler",
]
