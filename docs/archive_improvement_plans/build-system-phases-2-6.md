# Build System Implementation Plan: Phases 2–6

> **Status**: Planning Document  
> **Created**: 2025-01-06  
> **Depends On**: Phase 1 (Output Target Graph) and Phase 7 (Schema & Storage) — both complete

---

## Vision & Motivation

### The Problem We're Solving

CodeIntel analyzes codebases by running a series of **pipelines** that extract, transform, and compute insights: AST extraction, call graph construction, function metrics, risk scoring, and dozens more. Each pipeline produces outputs that downstream pipelines consume. Today, orchestrating these pipelines requires users to:

1. **Know the dependency order** — "Before I can compute function metrics, I need GOIDs, which needs AST extraction, which needs module scanning"
2. **Track what's already computed** — "Did I already run AST extraction for this commit? Is it still valid?"
3. **Manually decide what to rerun** — "I changed my coverage data; which downstream analytics are now stale?"
4. **Handle partial failures** — "Call graph failed; what can I still compute? What's blocked?"

This manual orchestration is **error-prone**, **wasteful** (recomputing things unnecessarily), and **opaque** (hard to know why something is stale or what will happen next).

### The Solution: A Build System for Code Intelligence

We're building what is essentially **`make` for code intelligence outputs**. Like `make`:

- **Declares outputs and their dependencies** — "Function profiles depend on metrics, coverage, semantic roles, and risk factors"
- **Tracks what's been built** — Content-addressable hashes prove whether an output is current
- **Computes minimal work** — Only rebuild what's actually stale
- **Handles the dependency cascade** — If AST changes, automatically invalidate everything downstream

Unlike `make`, our system:

- **Understands database state** — Outputs are DuckDB tables, not files
- **Integrates with typed pipelines** — Each output is produced by a specific plugin with known configuration
- **Provides rich introspection** — "Why is this stale? What will be recomputed? How long will it take?"

### Functional Outcome

When complete, users interact with CodeIntel through a simple, declarative interface:

```bash
# "I want function profiles to be current"
codeintel build function_profile

# System responds:
# - Checks database state
# - Finds that AST is stale (source files changed)
# - Computes the cascade: AST → GOIDs → call_graph → function_metrics → ...
# - Shows a plan: "Will compute 8 targets, skip 15 (already current), ~2 min"
# - Executes only what's needed
# - Records manifests for everything computed
```

The user doesn't need to know:
- The 12 transitive dependencies of `function_profile`
- That `goids` depends on both `ast` and `scip`
- That `risk_factors` requires 5 different inputs
- The correct order to run ingestion → graphs → analytics

**The system knows the graph and does the right thing.**

### Why This Architecture?

We chose a **layered build system** rather than extending the existing pipeline infrastructure because:

1. **Separation of concerns** — Pipelines know *how* to compute; the build system knows *what* and *when*

2. **Correctness by construction** — The dependency graph is declared once, validated at startup, and enforced everywhere. No ad-hoc "run these steps in this order" scattered across CLI commands.

3. **Incremental by default** — Every computation records its inputs' hashes. Staleness detection is automatic, not opt-in.

4. **Observable** — Dry-run shows exactly what will happen. Status shows exactly what state you're in. No "run it and see."

5. **Composable** — The same infrastructure works for:
   - Full rebuilds (`codeintel build --all`)
   - Targeted updates (`codeintel build function_profile`)
   - Forced refreshes (`codeintel build call_graph --force`)
   - CI pipelines (JSON output for automation)

### What This Enables (Future Possibilities)

The build system foundation opens doors we're not implementing now but could pursue:

**Incremental Table Updates**
- Today: Rebuild entire `function_metrics` table
- Future: Detect which functions changed, update only those rows
- Requires: Row-level provenance tracking

**Parallel Execution**
- Today: Execute targets sequentially within a stage
- Future: Run independent targets in parallel (e.g., `typing` and `coverage` have no dependency)
- Requires: Intra-stage dependency analysis

**Distributed Builds**
- Today: Single-machine execution
- Future: Farm out stages to workers, aggregate results
- Requires: Serializable plans, remote manifest storage

**Cross-Commit Analysis**
- Today: Each commit is independent
- Future: "What changed between these commits? What broke?"
- Requires: Manifest diffing, cross-commit dependency tracking

**Smart Caching**
- Today: Cache invalidated by any input change
- Future: Content-addressed caching ("I've seen this exact AST before")
- Requires: Normalized content hashing

**Predictive Scheduling**
- Today: Estimated durations are static
- Future: Learn actual durations, predict completion times
- Requires: Historical manifest analysis

**IDE Integration**
- Today: CLI-only
- Future: VS Code extension showing "these outputs are stale" with one-click rebuild
- Requires: Language server protocol integration

### Design Principles

1. **Explicit over implicit** — Every dependency is declared, every decision is logged
2. **Fail fast, fail clearly** — Validation before execution, actionable error messages
3. **Incremental correctness** — Partial execution is safe; resume where you left off
4. **Composition over configuration** — Small, focused components that combine
5. **Backwards compatible** — Existing `run_pipeline` usage continues to work

---

## Executive Summary

This document details the implementation plan for Phases 2–6 of the **Recipe-Driven PipelineSpec with Build Resolution** system. The goal is to create a unified build system that:

1. **Knows all outputs** and their dependencies (Phase 1 ✓)
2. **Tracks what has been computed** and when (Phase 7 ✓)
3. **Validates database state** before execution (Phase 2)
4. **Computes minimal work** needed to reach a goal (Phase 3)
5. **Generates executable plans** that respect dependencies (Phase 4)
6. **Integrates with existing pipelines** seamlessly (Phase 5)
7. **Exposes CLI/API for "build X"** requests (Phase 6)

---

## Completed Foundation: Phases 1 & 7

Before detailing Phases 2–6, here's what's already built and working.

### Phase 1: Output Target Graph

The target graph declares every output the system can produce, what tables it writes, which plugin produces it, and what it depends on.

| File | Purpose |
|------|---------|
| [`src/codeintel/core/build/__init__.py`](../../src/codeintel/core/build/__init__.py) | Package init with public exports |
| [`src/codeintel/core/build/targets.py`](../../src/codeintel/core/build/targets.py) | **`OutputTarget`** dataclass (name, module, plugin, tables, dependencies) and **`TargetGraph`** class with registration, lookup, dependency traversal, topological sorting, and cycle detection |
| [`src/codeintel/core/build/registry.py`](../../src/codeintel/core/build/registry.py) | Declares all ~43 targets: 9 ingestion (modules, ast, scip, typing, etc.), 6 graphs (goids, call_graph, cfg, dfg, etc.), and 28 analytics (function_metrics, profiles, subsystems, etc.). Provides `get_target_graph()` singleton. |
| [`src/codeintel/core/build/hashing.py`](../../src/codeintel/core/build/hashing.py) | **`compute_input_hash()`** creates content-addressable hashes from target definition + snapshot + dependency hashes. **`compute_options_hash()`** hashes plugin configuration. |
| [`tests/core/build/test_targets.py`](../../tests/core/build/test_targets.py) | Unit tests for target creation, graph operations, topological sort, cycle detection |
| [`tests/core/build/test_registry.py`](../../tests/core/build/test_registry.py) | Integration tests verifying all targets register correctly, no cycles exist, dependencies resolve |

**Key capabilities:**
- `graph.get("function_profile")` → returns the target with its metadata
- `graph.transitive_deps("function_profile")` → returns all 12+ upstream dependencies
- `graph.topological_order(["function_profile"])` → returns execution order respecting dependencies
- `graph.validate()` → checks for missing dependencies and cycles

### Phase 7: Schema & Storage

The storage layer persists build manifests (records of completed computations) and run tracking (records of build executions).

| File | Purpose |
|------|---------|
| [`src/codeintel/core/build/manifest.py`](../../src/codeintel/core/build/manifest.py) | **`OutputManifest`** dataclass (target, repo, commit, plugin, computed_at, input_hash, output_hash, row_count) and **`BuildRunRecord`** dataclass (run_id, requested/computed/skipped targets, status, timing) |
| [`src/codeintel/config/datasets/schemas.py`](../../src/codeintel/config/datasets/schemas.py) | Added `build.output_manifests` and `build.runs` table schemas to the dataset registry |
| [`src/codeintel/storage/schema/ddl.py`](../../src/codeintel/storage/schema/ddl.py) | Added `"build"` to the `SCHEMAS` tuple for database bootstrap |
| [`src/codeintel/storage/tracking/build_tracking.py`](../../src/codeintel/storage/tracking/build_tracking.py) | **`BuildTracking`** accessor class with `save_manifest()`, `load_manifest()`, `list_manifests()`, `delete_manifests()`, `start_run()`, `complete_run()`, `fetch_run()`, `list_runs()` |
| [`src/codeintel/storage/gateway/accessors.py`](../../src/codeintel/storage/gateway/accessors.py) | Added `build: BuildTracking` property to `DuckDBGateway` |
| [`src/codeintel/storage/gateway/protocol.py`](../../src/codeintel/storage/gateway/protocol.py) | Added `build: BuildTracking` to `StorageGateway` protocol |
| [`tests/storage/tracking/test_build_tracking.py`](../../tests/storage/tracking/test_build_tracking.py) | Unit tests for manifest CRUD, run lifecycle, listing/filtering |

**Key capabilities:**
- `gateway.build.save_manifest(manifest)` → records that a target was computed
- `gateway.build.load_manifest("ast", repo, commit)` → retrieves the manifest if it exists
- `gateway.build.list_manifests(repo, commit)` → all manifests for a snapshot
- `gateway.build.start_run(record)` / `complete_run(...)` → tracks build execution lifecycle

### How They Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                    get_target_graph()                           │
│     Returns singleton TargetGraph with all 43 OutputTargets     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ "What are the dependencies of X?"
                              │ "What's the execution order for Y?"
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    compute_input_hash()                         │
│   Combines target definition + snapshot + dependency hashes     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ "Is this hash the same as what's stored?"
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    gateway.build.*                              │
│   Persists/retrieves OutputManifest and BuildRunRecord          │
└─────────────────────────────────────────────────────────────────┘
```

Phase 2 will use all three: query the graph for dependencies, compute current hashes, compare against stored manifests.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI / API Layer                                 │
│                          codeintel build <target>                            │
│                                  (Phase 6)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Pipeline Integration                               │
│              Bridges BuildPlan → PipelineSpec execution                      │
│                                  (Phase 5)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Plan Generation                                   │
│              BuildPlan with ordered steps & skip decisions                   │
│                                  (Phase 4)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Minimal Work Resolver                               │
│              Determines what needs recomputation                             │
│                                  (Phase 3)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           State Validator                                    │
│              Checks DB state, detects staleness                              │
│                                  (Phase 2)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                     Target Graph & Tracking (Complete)                       │
│              OutputTarget, TargetGraph, BuildTracking                        │
│                            (Phases 1 & 7)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: State Validation Layer

### Purpose

Before computing anything, we must understand the **current state** of the database:
- Which targets have been computed for this repo/commit?
- Are any targets **stale** (dependencies changed since computation)?
- Are there **missing dependencies** that block computation?
- Are there **schema mismatches** (target was computed with an older schema)?

### Why It's Necessary

The `BuildTracking` layer (Phase 7) stores manifests recording *when* and *how* each target was computed. Phase 2 reads those manifests and compares them against:

1. **Current input hashes** — If a dependency's `input_hash` changed, downstream targets are stale
2. **Target graph** — If a dependency isn't computed, the dependent can't be computed
3. **Schema versions** — If the table schema evolved, recomputation may be required

Without this validation, we'd either:
- Recompute everything (wasteful)
- Skip recomputation when stale (incorrect results)
- Fail mid-pipeline due to missing dependencies

### Key Interfaces

```python
# core/build/state.py

@dataclass(frozen=True)
class TargetState:
    """Current state of a single target."""
    
    name: str
    status: Literal["missing", "computed", "stale", "blocked"]
    manifest: OutputManifest | None  # None if missing
    staleness_reason: str | None  # Why it's stale, if applicable
    blocking_deps: tuple[str, ...]  # Missing/stale deps that block this


@dataclass(frozen=True)
class DatabaseState:
    """Snapshot of all target states for a repo/commit."""
    
    repo: str
    commit: str
    targets: Mapping[str, TargetState]
    
    def get(self, name: str) -> TargetState: ...
    def missing_targets(self) -> tuple[str, ...]: ...
    def stale_targets(self) -> tuple[str, ...]: ...
    def computed_targets(self) -> tuple[str, ...]: ...
    def blocked_targets(self) -> tuple[str, ...]: ...


class StateValidator:
    """Validates database state against the target graph."""
    
    def __init__(
        self,
        graph: TargetGraph,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None: ...
    
    def validate(self) -> DatabaseState:
        """Scan all targets and determine their state."""
        ...
    
    def is_target_stale(self, name: str) -> tuple[bool, str | None]:
        """Check if a specific target needs recomputation."""
        ...
    
    def check_input_hash(self, target: OutputTarget) -> bool:
        """Compare stored input_hash against current computation."""
        ...
```

### Implementation Details

1. **Load all manifests** for the current `repo/commit` via `gateway.build.list_manifests()`

2. **For each target in the graph**:
   - If no manifest exists → `status = "missing"`
   - If manifest exists, check `input_hash`:
     - Recompute `compute_input_hash(target, snapshot, gateway)`
     - If hashes differ → `status = "stale"`, record reason
   - Check all dependencies:
     - If any dep is `missing` or `stale` → `status = "blocked"`

3. **Return `DatabaseState`** with all findings

### Outputs

- `DatabaseState` dataclass with per-target status
- Methods to query missing/stale/blocked targets
- Foundation for Phase 3's minimal work computation

### Files to Create

| File | Description |
|------|-------------|
| `src/codeintel/core/build/state.py` | `TargetState`, `DatabaseState`, `StateValidator` |
| `tests/core/build/test_state.py` | Unit tests for state validation |

---

## Phase 3: Minimal Work Resolver

### Purpose

Given:
- A **goal** (one or more targets the user wants up-to-date)
- The **current database state** (from Phase 2)

Compute the **minimal set of targets** that must be (re)computed to achieve the goal.

### Why It's Necessary

Users will request high-level outputs like:
- "I want `function_profile` up to date"
- "Rebuild everything for analytics"
- "Just update `call_graph`"

The resolver must:
1. **Expand dependencies** — `function_profile` requires 12+ upstream targets
2. **Prune already-computed** — Skip targets with valid, current manifests
3. **Detect forced recomputation** — If a mid-graph target is stale, all downstream must recompute
4. **Respect the DAG** — Never skip a dependency that a needed target requires

### Key Interfaces

```python
# core/build/resolver.py

@dataclass(frozen=True)
class ResolutionResult:
    """Result of resolving what work needs to be done."""
    
    requested: tuple[str, ...]  # What user asked for
    to_compute: tuple[str, ...]  # Targets that need computation (in order)
    to_skip: tuple[str, ...]  # Targets already up-to-date
    blocked: tuple[str, ...]  # Targets that cannot be computed (missing external deps)
    reasons: Mapping[str, str]  # Why each target is in its bucket


class BuildResolver:
    """Resolves minimal work needed to achieve goal targets."""
    
    def __init__(
        self,
        graph: TargetGraph,
        state: DatabaseState,
    ) -> None: ...
    
    def resolve(
        self,
        goals: Iterable[str],
        force_recompute: Iterable[str] | None = None,
    ) -> ResolutionResult:
        """Compute minimal work to make goals up-to-date.
        
        Parameters
        ----------
        goals
            Target names that must be up-to-date after execution.
        force_recompute
            Optional targets to recompute even if not stale.
        """
        ...
    
    def resolve_all(self, module: TargetModule | None = None) -> ResolutionResult:
        """Resolve work needed for all targets (optionally filtered by module)."""
        ...
```

### Algorithm

```
RESOLVE(goals, force_recompute):
    # 1. Expand goals to include all transitive dependencies
    all_needed = expand_transitive(goals)
    
    # 2. Mark forced targets as needing recomputation
    must_compute = set(force_recompute or [])
    
    # 3. Walk dependencies in reverse topological order
    for target in topological_order(all_needed):
        state = database_state.get(target)
        
        if target in must_compute:
            continue  # Already marked
        
        if state.status == "missing":
            must_compute.add(target)
        
        elif state.status == "stale":
            must_compute.add(target)
        
        elif state.status == "blocked":
            # Check if blocking deps are in must_compute
            if all(dep in must_compute for dep in state.blocking_deps):
                must_compute.add(target)
            else:
                # Truly blocked by external constraint
                blocked.add(target)
        
        else:  # computed and current
            # Check if any dependency is being recomputed
            if any(dep in must_compute for dep in target.dependencies):
                must_compute.add(target)  # Cascade invalidation
    
    # 4. Compute skip set
    to_skip = all_needed - must_compute - blocked
    
    # 5. Return in topological order
    return ResolutionResult(
        requested=goals,
        to_compute=topological_order(must_compute),
        to_skip=tuple(to_skip),
        blocked=tuple(blocked),
        reasons=compute_reasons(...),
    )
```

### Cascade Invalidation

This is critical: if `ast` is recomputed, then `goids` must recompute (even if its manifest shows "current"), because `goids` depends on `ast`. The resolver tracks this cascade.

### Outputs

- `ResolutionResult` with ordered computation list
- Clear reasons for each decision (useful for `--dry-run` output)
- Foundation for Phase 4's plan generation

### Files to Create

| File | Description |
|------|-------------|
| `src/codeintel/core/build/resolver.py` | `ResolutionResult`, `BuildResolver` |
| `tests/core/build/test_resolver.py` | Unit tests including cascade scenarios |

---

## Phase 4: Plan Generation

### Purpose

Transform a `ResolutionResult` into an **executable `BuildPlan`** that:
- Groups targets by pipeline module (ingestion → graphs → analytics)
- Provides estimated duration and resource hints
- Can be serialized for dry-run display or async execution
- Tracks execution progress

### Why It's Necessary

The resolver tells us *what* to compute; the plan tells us *how*:
- **Ordering**: Respects module boundaries (all ingestion before graphs)
- **Batching**: Groups targets that can run in the same pipeline stage
- **Estimation**: Provides time/resource estimates for progress bars
- **Tracking**: Records which steps completed, failed, or were skipped

### Key Interfaces

```python
# core/build/plan.py

@dataclass(frozen=True)
class PlanStep:
    """A single step in the build plan."""
    
    target: str
    module: TargetModule
    plugin: str
    estimated_duration_ms: int | None
    dependencies: tuple[str, ...]
    reason: str  # Why this step is included


@dataclass(frozen=True)
class PlanStage:
    """A group of steps that execute together."""
    
    module: TargetModule
    steps: tuple[PlanStep, ...]
    
    @property
    def estimated_duration_ms(self) -> int | None:
        """Sum of step estimates, or None if any unknown."""
        ...


@dataclass(frozen=True)
class BuildPlan:
    """Complete execution plan for a build request."""
    
    requested_targets: tuple[str, ...]
    stages: tuple[PlanStage, ...]
    skipped_targets: tuple[str, ...]
    blocked_targets: tuple[str, ...]
    
    @property
    def total_steps(self) -> int: ...
    
    @property
    def estimated_duration_ms(self) -> int | None: ...
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output (dry-run display)."""
        ...


class PlanGenerator:
    """Generates executable plans from resolution results."""
    
    def __init__(self, graph: TargetGraph) -> None: ...
    
    def generate(self, resolution: ResolutionResult) -> BuildPlan:
        """Generate a build plan from resolution result."""
        ...
```

### Stage Grouping Logic

```python
def generate(self, resolution: ResolutionResult) -> BuildPlan:
    # Group targets by module
    by_module: dict[TargetModule, list[str]] = {
        "ingestion": [],
        "graphs": [],
        "analytics": [],
    }
    
    for target_name in resolution.to_compute:
        target = self.graph.get(target_name)
        by_module[target.module].append(target_name)
    
    # Create stages in order: ingestion → graphs → analytics
    stages = []
    for module in ["ingestion", "graphs", "analytics"]:
        if by_module[module]:
            steps = [
                PlanStep(
                    target=name,
                    module=module,
                    plugin=self.graph.get(name).plugin,
                    estimated_duration_ms=self.graph.get(name).estimated_duration_ms,
                    dependencies=self.graph.get(name).dependencies,
                    reason=resolution.reasons.get(name, ""),
                )
                for name in by_module[module]
            ]
            stages.append(PlanStage(module=module, steps=tuple(steps)))
    
    return BuildPlan(
        requested_targets=resolution.requested,
        stages=tuple(stages),
        skipped_targets=resolution.to_skip,
        blocked_targets=resolution.blocked,
    )
```

### Dry-Run Output

The plan's `to_dict()` enables output like:

```
Build Plan for: function_profile
═══════════════════════════════════════════════════════════════

Stage 1: Ingestion (3 steps, ~15s)
  ✓ modules     [skip: already computed]
  → ast         (reason: stale, input changed)
  → typing      (reason: cascade from ast)

Stage 2: Graphs (2 steps, ~30s)
  → goids       (reason: cascade from ast)
  → call_graph  (reason: cascade from goids)

Stage 3: Analytics (5 steps, ~45s)
  → function_metrics    (reason: cascade from goids)
  → coverage_functions  (reason: missing)
  → risk_factors        (reason: cascade from function_metrics)
  → semantic_roles      (reason: cascade from function_metrics)
  → function_profile    (reason: requested)

Total: 10 steps, ~90s estimated
Skipped: 8 targets (already current)
```

### Files to Create

| File | Description |
|------|-------------|
| `src/codeintel/core/build/plan.py` | `PlanStep`, `PlanStage`, `BuildPlan`, `PlanGenerator` |
| `tests/core/build/test_plan.py` | Unit tests for plan generation |

---

## Phase 5: Pipeline Integration

### Purpose

Bridge the `BuildPlan` to actual execution via existing pipeline infrastructure:
- Map `PlanStage` to `PipelineSpec` stages
- Execute using `run_pipeline()` with proper configuration
- Record manifests after successful target completion
- Handle failures gracefully (partial completion)

### Why It's Necessary

We have:
- **Build system** (Phases 1-4): Knows what to compute
- **Pipeline system** (`PipelineSpec`, `run_pipeline`): Knows how to compute

Phase 5 connects them:
1. Translate `BuildPlan` → `PipelineSpec` + `PipelinePlanOptions`
2. Hook into plugin completion to record manifests
3. Propagate failures without losing partial progress

### Key Interfaces

```python
# core/build/executor.py

@dataclass(frozen=True)
class BuildResult:
    """Result of executing a build plan."""
    
    run_id: str
    plan: BuildPlan
    completed_targets: tuple[str, ...]
    failed_targets: tuple[str, ...]
    skipped_targets: tuple[str, ...]
    duration_ms: float
    error_summary: str | None


class BuildExecutor:
    """Executes build plans via the pipeline system."""
    
    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        config: CodeIntelConfig,
    ) -> None: ...
    
    def execute(
        self,
        plan: BuildPlan,
        dry_run: bool = False,
    ) -> BuildResult:
        """Execute a build plan.
        
        Parameters
        ----------
        plan
            The build plan to execute.
        dry_run
            If True, validate but don't execute.
        """
        ...
    
    def _execute_stage(
        self,
        stage: PlanStage,
        run_id: str,
    ) -> tuple[list[str], list[str]]:
        """Execute a single stage, return (completed, failed)."""
        ...
    
    def _record_manifest(
        self,
        target: OutputTarget,
        duration_ms: float,
        row_count: int | None,
    ) -> None:
        """Record successful target completion."""
        ...
```

### Execution Flow

```python
def execute(self, plan: BuildPlan, dry_run: bool = False) -> BuildResult:
    run_id = generate_run_id()
    
    # 1. Record run start
    self.gateway.build.start_run(BuildRunRecord(
        run_id=run_id,
        repo=self.snapshot.repo,
        commit=self.snapshot.commit,
        requested_targets=plan.requested_targets,
        computed_targets=(),
        skipped_targets=plan.skipped_targets,
        started_at=datetime.now(UTC),
        status="running",
    ))
    
    if dry_run:
        return BuildResult(run_id=run_id, plan=plan, ...)
    
    completed = []
    failed = []
    
    try:
        # 2. Execute each stage in order
        for stage in plan.stages:
            stage_completed, stage_failed = self._execute_stage(stage, run_id)
            completed.extend(stage_completed)
            failed.extend(stage_failed)
            
            # Stop on failure (dependencies broken)
            if stage_failed:
                break
        
        status = "succeeded" if not failed else "failed"
    
    except Exception as e:
        status = "failed"
        error_summary = str(e)
    
    # 3. Record run completion
    self.gateway.build.complete_run(
        run_id=run_id,
        status=status,
        completed_targets=tuple(completed),
        skipped_targets=plan.skipped_targets,
        error_summary=error_summary if failed else None,
    )
    
    return BuildResult(...)
```

### Stage Execution

```python
def _execute_stage(self, stage: PlanStage, run_id: str) -> tuple[list[str], list[str]]:
    # Map stage to PipelineSpec
    if stage.module == "ingestion":
        spec = self._build_ingestion_spec(stage)
        result = run_ingestion_pipeline(spec, self.config)
    elif stage.module == "graphs":
        spec = self._build_graphs_spec(stage)
        result = run_graphs_pipeline(spec, self.config)
    else:  # analytics
        spec = self._build_analytics_spec(stage)
        result = run_analytics_pipeline(spec, self.config)
    
    # Record manifests for completed targets
    for step in stage.steps:
        if step.target in result.completed:
            self._record_manifest(
                target=self.graph.get(step.target),
                duration_ms=result.step_durations.get(step.target),
                row_count=result.row_counts.get(step.target),
            )
    
    return result.completed, result.failed
```

### Plugin-to-Target Mapping

Each pipeline plugin maps to an `OutputTarget`. The mapping is defined in `core/build/registry.py`:

```python
# Already exists from Phase 1
AST_TARGET = OutputTarget(
    name="ast",
    module="ingestion",
    plugin=AST_EXTRACT_PLUGIN.metadata.name,  # "ast_extract"
    tables=("core.ast_nodes", "core.ast_metrics"),
    dependencies=("modules",),
)
```

The executor uses `plugin` to configure which plugins run in each stage.

### Files to Create/Modify

| File | Description |
|------|-------------|
| `src/codeintel/core/build/executor.py` | `BuildResult`, `BuildExecutor` |
| `src/codeintel/core/build/pipeline_bridge.py` | `PlanStage` → `PipelineSpec` translation |
| `tests/core/build/test_executor.py` | Integration tests with mock pipelines |

---

## Phase 6: CLI and API Integration

### Purpose

Expose the build system to users via:
- **CLI**: `codeintel build <target>` command
- **API**: Programmatic access for automation

### Why It's Necessary

All the infrastructure is internal without user-facing commands. Phase 6 provides:
- Simple "build X" interface
- Dry-run mode for planning
- Progress reporting
- Integration with existing CLI infrastructure

### CLI Interface

```bash
# Build a specific target
codeintel build function_profile

# Build multiple targets
codeintel build function_profile test_profile

# Build all analytics
codeintel build --module analytics

# Dry-run (show plan without executing)
codeintel build function_profile --dry-run

# Force recomputation of specific targets
codeintel build function_profile --force ast goids

# Show current state
codeintel build --status

# Output as JSON (for automation)
codeintel build function_profile --json
```

### Implementation

```python
# cli/commands/build.py

@app.command()
def build(
    targets: Annotated[list[str] | None, typer.Argument()] = None,
    module: Annotated[str | None, typer.Option("--module", "-m")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n")] = False,
    force: Annotated[list[str] | None, typer.Option("--force", "-f")] = None,
    status: Annotated[bool, typer.Option("--status")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build targets with automatic dependency resolution.
    
    The build system computes the minimal work needed to bring
    requested targets up-to-date, respecting dependencies and
    detecting stale data.
    
    Examples
    --------
    Build function profiles:
        codeintel build function_profile
    
    Show what would be built:
        codeintel build function_profile --dry-run
    
    Force rebuild from AST:
        codeintel build function_profile --force ast
    """
    # 1. Resolve project config
    config = resolve_project_config()
    snapshot = resolve_snapshot(config)
    
    # 2. Open gateway
    with open_gateway(config) as gateway:
        graph = get_target_graph()
        
        # 3. Handle --status
        if status:
            validator = StateValidator(graph, gateway, snapshot)
            state = validator.validate()
            display_status(state, json_output)
            return
        
        # 4. Resolve goals
        if module:
            goals = [t.name for t in graph.targets_for_module(module)]
        elif targets:
            goals = targets
        else:
            raise typer.BadParameter("Specify targets or --module")
        
        # 5. Validate state and resolve work
        validator = StateValidator(graph, gateway, snapshot)
        state = validator.validate()
        
        resolver = BuildResolver(graph, state)
        resolution = resolver.resolve(goals, force_recompute=force)
        
        # 6. Generate plan
        generator = PlanGenerator(graph)
        plan = generator.generate(resolution)
        
        # 7. Display or execute
        if dry_run or json_output:
            display_plan(plan, json_output)
            return
        
        # 8. Execute
        executor = BuildExecutor(gateway, snapshot, config)
        result = executor.execute(plan)
        
        display_result(result)
        
        if result.failed_targets:
            raise typer.Exit(1)
```

### Status Output

```
Database State for demo/repo @ abc123
═══════════════════════════════════════════════════════════════

Computed (23):
  ✓ modules, ast, cst, scip, typing, coverage, tests, docstrings
  ✓ goids, call_graph, import_graph, cfg, dfg, symbol_uses
  ✓ function_metrics, hotspots, ...

Stale (2):
  ⚠ ast              (input hash changed)
  ⚠ function_metrics (cascade from ast)

Missing (3):
  ✗ config_ingest
  ✗ config_data_flow
  ✗ external_deps

Blocked (1):
  ⊘ entrypoints      (requires: config_data_flow)
```

### API Access

```python
# Programmatic usage
from codeintel.core.build import (
    get_target_graph,
    StateValidator,
    BuildResolver,
    PlanGenerator,
    BuildExecutor,
)

graph = get_target_graph()
validator = StateValidator(graph, gateway, snapshot)
state = validator.validate()

resolver = BuildResolver(graph, state)
resolution = resolver.resolve(["function_profile"])

generator = PlanGenerator(graph)
plan = generator.generate(resolution)

executor = BuildExecutor(gateway, snapshot, config)
result = executor.execute(plan)
```

### Files to Create/Modify

| File | Description |
|------|-------------|
| `src/codeintel/cli/commands/build.py` | `build` command implementation |
| `src/codeintel/cli/main.py` | Register `build` command |
| `tests/cli/test_build_command.py` | CLI integration tests |

---

## Dependency Graph

```
Phase 1 (Complete)          Phase 7 (Complete)
    │                           │
    └──────────┬────────────────┘
               │
               ▼
         Phase 2: State Validation
               │
               ▼
         Phase 3: Minimal Work Resolver
               │
               ▼
         Phase 4: Plan Generation
               │
               ▼
         Phase 5: Pipeline Integration
               │
               ▼
         Phase 6: CLI/API
```

---

## Testing Strategy

### Unit Tests

Each phase has isolated unit tests:
- **Phase 2**: Mock manifests, verify state detection
- **Phase 3**: Various graph configurations, cascade scenarios
- **Phase 4**: Plan structure, stage grouping
- **Phase 5**: Mock pipeline execution
- **Phase 6**: CLI argument parsing, output formatting

### Integration Tests

End-to-end scenarios:
1. **Fresh build**: Empty database → build `function_profile` → verify all deps computed
2. **Incremental update**: Change source → verify cascade invalidation
3. **Partial failure**: Plugin fails → verify partial manifest recording
4. **Dry-run accuracy**: Plan matches actual execution

### Golden File Tests

- Expected plan output for standard scenarios
- Consistent `--json` output format

---

## Migration Path

### Existing `run_pipeline` Users

The build system wraps existing pipelines, so:
- **No breaking changes** to `PipelineSpec` or `run_pipeline`
- Existing direct pipeline usage continues to work
- Build system is opt-in via `codeintel build`

### Gradual Adoption

1. Start with `codeintel build --dry-run` to see plans
2. Use `codeintel build --status` to understand state
3. Replace manual pipeline invocations with `codeintel build`
4. Eventually, all pipeline execution flows through build system

---

## Open Questions

1. **Schema Evolution**: How do we handle schema version tracking?
   - Option A: Store schema hash in manifest
   - Option B: Semantic versioning in table definitions

2. **Parallel Execution**: Can targets within a stage run in parallel?
   - Current plan: Sequential within stage
   - Future: Parallel based on intra-stage dependencies

3. **External Inputs**: How do we track non-database inputs (files, git history)?
   - Current: File hashes via `SnapshotRef`
   - Future: Extended input hash computation

4. **Partial Rebuild**: Can we rebuild part of a table (e.g., new modules only)?
   - Current: Full table rebuild
   - Future: Incremental table updates

---

## Timeline Estimate

| Phase | Estimated Effort | Dependencies |
|-------|------------------|--------------|
| Phase 2 | 2-3 days | Phase 1, 7 |
| Phase 3 | 2-3 days | Phase 2 |
| Phase 4 | 1-2 days | Phase 3 |
| Phase 5 | 3-4 days | Phase 4 |
| Phase 6 | 1-2 days | Phase 5 |
| **Total** | **~2 weeks** | |

---

## Summary

The build system transforms CodeIntel from "run these plugins" to "make this output current." By understanding the complete dependency graph and tracking computation history, we can:

- **Minimize work**: Only compute what's needed
- **Ensure correctness**: Never serve stale data
- **Provide visibility**: Show exactly what will happen
- **Enable automation**: Simple "build X" interface

Phases 1 and 7 established the foundation. Phases 2-6 build the intelligence and user interface on top.

