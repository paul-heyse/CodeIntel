
## 1) Modify existing test helpers to amplify the realism of testing conditions

### A. `tests/_helpers/hamilton_execution.py`: finish the “Hamilton-native” execution path (right now it’s a stub)

Right now `HamiltonTestBuilder.execute_target()` routes through `execute_hamilton_target_async()` and `NativeTargetExecutor` with a `compute()` that returns `{}` (stub), which **does not** execute the real DAG and will never validate “production-like” behavior (skip checks, real dependencies, contract hooks, dataset/artifact materialization, etc.).

**What to change**

1. **Stop calling `NativeTargetExecutor` directly in the test helper.**
   In production, targets are materialized by the *Hamilton* materialize node (e.g. `t__modules`, `t__call_graph`, …), which internally uses `executor_materialize()` and therefore `NativeTargetExecutor` correctly.

2. **Build and reuse a real `HamiltonRuntime` via `codeintel.build.hamilton.driver_factory.build_driver()`.**
   That runtime gives you:

   * the *actual* executable Hamilton `Driver`
   * the *Hamilton-derived* `TargetGraph` (avoids drift vs. `get_target_metadata_service().system.graph`)

3. **Execute the *materialize node* for a target via the Hamilton Driver**:

   * map `target_name -> node_name` using `runtime.target_to_node[target]` or `target_to_node_name(target, runtime=runtime)`
   * run: `runtime.dr.execute([node_name], inputs={"env": env, "graph": runtime.graph})`
   * return the `TargetRunRecord` produced by the node

4. Add higher-level helpers that mirror production usage:

   * `execute_targets([...]) -> dict[str, TargetRunRecord]` (multi-output driver execution is *more* realistic and faster because shared deps compute once)
   * `execute_target(target, *, force=False, profile=None)` (use `dataclasses.replace(env, force_targets=...)` since `BuildEnv` is frozen)

**Why this materially improves realism**

* You’re exercising the *same orchestration mechanism* as prod: “DAG-derived outputs determine execution.”
* You get actual dependency execution, skip behavior, artifact saving, dataset writes, row count validation, etc.
* You can write “pipeline realism tests” without falling back to mocked internal nodes.

---

### B. `tests/_helpers/hamilton_execution.py` and fixtures: stop depending on `get_target_metadata_service()` graphs and make tool behavior deterministic

`tests/_helpers/hamilton_fixtures.py` constructs env/gateway/paths OK, but the graph lookup in `tests/_helpers/hamilton_execution.py` still leans on:

* `create_default_providers(ToolsConfig.default())` (often means “real tools must exist on PATH”)
* drift risk if you use `get_target_metadata_service().system.graph` elsewhere

**What to change**

1. Provide a fixture that returns a **session-scoped** `HamiltonRuntime` built via `build_driver()`:

   * `@pytest.fixture(scope="session") def hamilton_runtime(): return build_driver(config=...)`
   * This avoids repeated DAG builds (huge speed win) and avoids graph drift.

2. Update env-building fixtures to accept **providers/tool config injection** cleanly:

   * “realistic execution” doesn’t mean “must call real scip/pyright binaries” in every unit/integration test
   * but it *does* mean you should go through the real `ToolService` plugin path when you can (see tool helper changes below)

---

### C. `tests/_helpers/fakes/tools.py`: upgrade `FakeToolRunner` so it produces realistic artifacts for **all** tool plugins you exercise

Today `FakeToolRunner.run_async()` only writes `output_path` payloads for `{coverage, pyrefly}`.
But your ingestion tool plugins also rely on output artifacts for:

* **pytest** (`pytest-json-report` writes a JSON report file)
* **scip-python** (writes `index.scip`)
* sometimes others, depending on execution path

**What to change**

1. Make `FakeToolRunner` *always* create the `options.output_path` file when provided, unless a tool is configured to “not produce output”.

   * Default “file exists” is closer to real tool behavior.
2. Add per-tool file payloads:

   * `payloads["pytest_json"]` used to write the json-report file
   * `payloads["scip_binary"]` (or just a string) written to the `.scip` output path
   * `payloads["scip_print_stdout"]` used as stdout for `scip print --json` (the plugin writes this to `output_json`)
   * keep the existing `coverage_json` / `pyrefly_json` style payloads
3. Add per-tool failure controls (realism needs negative paths):

   * `returncodes[tool]`, `raise_on[tool]`, `not_found[tool]`
   * that lets you test “tool missing”, “tool fails”, “tool output malformed”, etc. *through the real plugin code*.

**Why this improves realism**

* Your DAG nodes that call `env.providers.tool_service.run_*()` now exercise:

  * the real tool plugin parsing
  * the “file exists?” logic
  * realistic failure modes
* And you can run those tests in CI without requiring the full toolchain installed.

> If you want to go **even more production-parity**, the next step is a “tool sandbox” that creates stub executables and uses the real `ToolRunner` subprocess path (I outline that in section 2).

---

### D. `tests/_helpers/gateway.py`: add a clear on-disk gateway helper (critical for threadpool/parallel realism)

A lot of “realistic Hamilton execution” eventually wants to test:

* parallel backend (`threadpool`)
* multiple connections / gateway reopen behavior
* persistence across runs (skip logic + manifests)

`GatewayFactory` already supports file-backed gateways, but there is no small helper that standardizes on-disk setup for tests. That blocks realistic testing of threadpool mode because in-memory DuckDB connections can’t be reopened the same way.

**What to change**

* Add something like:

  * `GatewayFactory.open_on_disk(db_path: Path) -> StorageGateway`
  * or `GatewayFactory.open_disk(tmp_path: Path, name="codeintel.duckdb")`
* Use `StorageConfig.for_ingest(db_path)` and `open_gateway(cfg)` (you already do this pattern elsewhere, e.g. `tests/_helpers/cli_project.py`).

**Why this improves realism**

* Lets you run Hamilton with `threadpool` adapter (which may reopen gateways per worker)
* Enables “run twice → skip via manifests” without relying on the same in-memory connection

---

### E. Orchestration helpers that call internal compute nodes directly: prefer executing targets via the DAG

I found multiple helpers that call internal `t__*_extract` functions directly (e.g. `tests/_helpers/orchestration/graph_orchestration.py`, and `tests/_helpers/orchestration/provisioning.py` calls `t__call_graph__extract`).

That pattern bypasses:

* target-level skip checks
* target run records
* artifacts/datasets bookkeeping

**What to change**

* For “realistic pipeline” tests: call the *target materialize node* via your Hamilton harness (`execute_target("call_graph")`, etc.)
* Keep direct compute-node calls only for narrow algorithm/unit tests, and label them explicitly as such.

---

### F. Refinements and guardrails for production-parity helpers

These are smaller refinements that make the above changes safer and closer to real execution:

1. **Use canonical node naming instead of assuming `t__{target}`.**
   Prefer `target_to_node_name(target, runtime=runtime)` or `runtime.target_to_node[target]`
   over hand-constructed names. This avoids breakage when targets use non-trivial names.

2. **Attach real adapters/hooks when executing via Hamilton Driver.**
   If you call `build_driver()` directly, you do not get the telemetry hooks, parallel adapter,
   or contract enforcement that `HamiltonBuildExecutor` wires up. For production parity, either:
   * execute through `HamiltonBuildExecutor`, or
   * pass the same adapters/hook factory into `build_driver()`.

3. **Model `BuildEnv` as immutable.**
   `BuildEnv` is frozen, so set `force_targets`/`profile` via `dataclasses.replace`
   (or rebuild the env) rather than mutating in place.

4. **`FakeToolRunner` should mirror plugin expectations (stdout + files).**
   The SCIP plugin writes JSON from `stdout` of `scip print --json`, so you need a payload
   for `stdout` in addition to creating the output file. Pytest parsing likewise needs a
   valid JSON payload at the report path.

5. **Align new harness locations with existing patterns.**
   There are already harness helpers under `tests/_helpers/harnesses/`.
   Place a Hamilton build harness there to keep discovery and imports consistent.

---

## 2) Additional test helpers you should create (currently missing)

### A. A first-class “Hamilton Build Harness” (the main missing piece)

Create a helper whose entire purpose is:

> “Given a repo fixture + gateway + tool behavior, execute one or more Hamilton targets through the real DAG and return run records + make assertions easy.”

**Recommended shape**

* `tests/_helpers/harnesses/hamilton_build.py`
* Core API:

  * `build_harness(tmp_path, *, gateway_mode="memory|disk", macros=True, views=True)`
  * `.with_repo(writer_fn | repo_tree)`
  * `.with_tool_payloads(...)` (feeds FakeToolRunner OR tool sandbox)
  * `.with_profile("default")`
  * `.with_force_targets({"x","y"})` (implemented via `dataclasses.replace` on `BuildEnv`)
  * `.run_targets(["modules","goids","call_graph"]) -> dict[target, TargetRunRecord]`
  * `.assert_row_counts_match_records(records)` convenience
  * `.snapshot_tables([...])` for regression testing

**Why you need it**

* Right now, realistic DAG testing is scattered:

  * a little in CLI smoke tests
  * a little in direct-node tests
  * a stubbed-out `hamilton_execution.py`
* A harness gives you one unified “production-like” interface for most tests.

---

### B. A “Tool Sandbox” helper (optional, but the highest realism)

If you want to go *max realism* while still being deterministic:

**Goal:** run the real `ToolRunner` (subprocess path) but provide stub binaries in a temp `bin/` directory.

**Helper**

* `tests/_helpers/tool_sandbox.py`
* Provides:

  * `ToolSandbox.create(tmp_path)`
  * `.install_stub("pyright", stdout=json_payload, returncode=1)`
  * `.install_stub("scip-python", creates="--output")`
  * `.install_stub("scip", stdout=scip_json)`
  * `.install_stub("pytest", writes="--json-report-file=...")`
  * `.tools_config()` that points to these stub binaries
  * context manager that prepends sandbox bin dir to `PATH`

**Why this is worth it**

* You exercise the real subprocess execution, environment propagation, cwd handling, timeouts, etc.
* Still deterministic and CI-friendly.
* It’s closer to production than `FakeToolRunner`, but heavier—so you’d use it for `@pytest.mark.integration` tests.

---

### C. Manifest + skip-test helper (fills an explicit gap in your suite)

You already have placeholder skip integration tests in `tests/build/hamilton/native/test_skip_logic.py` that skip because “requires full integration infra”.

Add:

* `tests/_helpers/manifests.py`

  * `load_manifest_index(gateway, repo, commit) -> dict[target, OutputManifest]`
  * `assert_skipped(record)`
  * `assert_succeeded(record)`
  * `run_twice_and_assert_skip(harness, target, ...)`

This lets you write realistic tests for:

* “first run computes + persists manifest”
* “second run skips”
* “force overrides skip”
* “input hash change recomputes”

---

### D. Target record assertion helpers (small, but they make tests cleaner and more realistic)

Create:

* `tests/_helpers/assertions/target_record_assertions.py`

  * `assert_target_ok(record, *, expected_status="succeeded")`
  * `assert_record_row_counts(record, expected: dict[table_key,int])`
  * `assert_record_has_datasets(record, keys=[...])`
  * `assert_record_has_artifacts(record, names=[...])`

---

## 3) Modules-first expectations and golden diffs (implemented)

These helpers are now available and should be the default for module inventory tests:

- `modules_expected_from_repo_tree(...)`, `module_paths_expected_from_repo_tree(...)`,
  `modules_expected_from_env(...)` in `tests/_helpers/modules_expectations.py`
- `ModulesAssertions` for repo_map/modules parity and module inventory checks.
- `format_missing_extra(...)`, `format_module_map_diff(...)`, and
  `module_map_from_path_map(...)` for high-signal diffs.

Hamilton build tests should prefer `HamiltonBuildHarness` with:

- `harness.run_targets([...])` for execution.
- `harness.artifacts` for output paths.
- `harness.priming` when manifests are needed without full runs.

Additional expanded usage patterns:

- SCIP ingestion tests can add a harness-based real-tools path when SCIP binaries are present,
  while retaining deterministic artifact-based execution for tool-free runs.
- Serving/export tests that write schema manifests, buildspecs, or semantic registries should
  prefer `HarnessArtifacts` writers to keep payloads consistent and avoid hand-rolled JSON.
- Skip-logic tests for serving/export targets can use `ManifestPriming` to seed manifests
  (for example `serving_artifacts`) without paying for a full upstream execution.
- Graph loader tests that depend on module catalogs should call
  `ModulesAssertions.inventory_consistent()` after module/repo_map seeding.
- Analytics tests comparing module/path mappings should use golden diff helpers for clearer
  mismatch diagnostics.

This keeps tests asserting what production cares about: **TargetRunRecord + persisted outputs**, not just “some table has rows”.

---

### E. Table snapshotting utilities for regression tests

You already have snapshot-style tests in some areas, but a generalized helper that can:

* dump selected tables deterministically (sorted rows)
* store/compare snapshots

…is extremely effective once you start running more of the DAG.

Put it in:

* `tests/_helpers/snapshots/tables.py`

---

## 3) Where to deploy these within your pytest scope

### A. What stays in `tests/_helpers/…` (library code)

Put reusable, non-pytest-specific code here:

* `tests/_helpers/hamilton_execution.py` (updated to real DAG execution)
* `tests/_helpers/harnesses/hamilton_build.py` (new)
* `tests/_helpers/tool_sandbox.py` (new, optional)
* `tests/_helpers/manifests.py` (new)
* `tests/_helpers/assertions/target_record_assertions.py` (new)
* upgrades to `tests/_helpers/fakes/tools.py` and `tests/_helpers/gateway.py`

This keeps helpers importable without pytest side effects (you already enforce this via `tests/_helpers/__init__.py`).

---

### B. What becomes pytest fixtures in `conftest.py` (wiring + lifecycle)

**In `tests/conftest.py`** (globally useful fixtures):

* `hamilton_runtime` → `scope="session"`
  Build once per worker via `build_driver()`.
* `tool_sandbox` or `fake_tools` → likely `scope="function"`
  Because payloads differ per test.
* `build_harness` → `scope="function"`
  Because it owns tmp_path, gateway, repo root.

**Why these scopes**

* Session-scoped runtime: heavy to build, safe to reuse if it’s stateless (it should be).
* Function-scoped env/gateway/repos: keeps tests isolated and xdist-safe.

---

### C. Domain-specific fixtures live in domain conftests

For example:

* `tests/analytics/conftest.py`:

  * `analytics_harness` that defaults to the analytics profile and seeds common analytics inputs
* `tests/build/hamilton/native/conftest.py`:

  * fixtures for running native target migrations, maybe with stricter contract enforcement
* `tests/ingestion/conftest.py`:

  * ingestion harness that provides a repo tree + tool payload defaults

This avoids overloading global fixtures with domain-specific defaults.

---

### D. Marking strategy (to preserve speed while increasing realism)

You already have markers in `pytest.ini`. Use them deliberately:

* **default (fast) tests**:

  * use `FakeToolRunner` + real `ToolService` plugin path
  * memory gateway ok
* **`@pytest.mark.integration`**:

  * use tool sandbox (stub binaries) OR real binaries if available
  * on-disk gateway
  * run multiple targets together (more realistic)
* **`@pytest.mark.e2e` / `smoke`**:

  * CLI-based invocations (you already do this)
  * minimal assertions, production-like runtime

---

## Practical “next step” recommendation

If you do only one thing first: **complete `tests/_helpers/hamilton_execution.py` so it truly executes targets through `build_driver().dr.execute(...)`** and returns real `TargetRunRecord`s. That one change unlocks:

* real DAG coverage,
* real skip logic tests,
* real manifest persistence tests,
* and a smooth path to refactoring older helpers that call internal nodes directly.

If you want, I can also propose a concrete API sketch for `HamiltonBuildHarness` that plugs directly into your current `GatewayFactory`, `repo_writers`, and seed packs so you can migrate tests incrementally without breaking the suite.

---

## 4) Expanded helper rollout (graph, analytics, serving, fixtures, snapshots)

### A) Target-family harness wrappers

New Hamilton-driven wrappers reduce boilerplate when exercising higher-level targets:

- `tests/_helpers/harnesses/graph_harness.py`
  - Default targets: `call_graph`, `import_graph`
  - Helpers: run targets + assert graph tables/datasets
- `tests/_helpers/harnesses/analytics_harness.py`
  - Default targets: `function_metrics`
  - Helpers: run targets + assert metrics tables
- `tests/_helpers/harnesses/serving_harness.py`
  - Default targets: `serving_artifacts`
  - Helpers: publish serving snapshot via `publish_serving_snapshot(...)`

### B) Deterministic repo fixture writers

Repo writers now cover monorepo layouts, generated noise, large files, and scope paths:

- `write_monorepo_fixture(...)`
- `write_generated_noise_fixture(...)`
- `write_large_file_fixture(...)`
- `write_scoped_paths_fixture(...)`

Each writer returns a `RepoFixture` with `module_map` computed via
`modules_expected_from_repo_tree(...)` so tests can assert expectations directly.

### C) ToolSandbox default stubs

`ToolSandbox.install_default_stubs()` installs deterministic stub binaries for:

- `pytest`, `coverage`, `scip-python`, `scip`, `pyright`, `pyrefly`, `ruff`, `git`

This keeps integration tests deterministic while exercising the real subprocess path.

### D) Table snapshot diffs

`tests/_helpers/snapshots/tables.py` now includes `diff_table_snapshot(...)` to
produce unified diffs for snapshot mismatches, keeping regression failures readable.

### E) New helper expansions (plan/status, guards, payloads, registry)

Additional helpers are now available for broader target coverage:

- Upstream status guards: `require_upstream_ok(...)` to normalize dependency checks
  and optionally treat `skipped` as cached success.
- Plan/status harness helpers: compute plan/status summaries and format readable diffs.
- Tool payload fixtures: minimal valid payloads for pytest, coverage, and SCIP.
- Config override helpers: write build config sections and reload BuildConfig into a harness.
- Repo fixture registry: request fixtures by intent (e.g., `monorepo`, `generated`).
- Failure scenario assertions: consistent checks for partial/failed TargetRunRecord bundles.
- Snapshot hashing/column subsets for large tables: stable diffs without full dumps.


# concrete API sketch for HamiltonBuildHarness #

Below is a concrete **API sketch** for a `HamiltonBuildHarness` that:

* **Uses your production Hamilton path** (`HamiltonBuildExecutor` + `BuildEnv`)
* **Wraps your existing pytest “world”** (`TestContext` + `GatewayFactory` + seed packs)
* **Optionally writes realistic repos** via `tests/_helpers/orchestration/repo_writers.py`
* Lets you **migrate incrementally** (seed-first tests can start running Hamilton targets without rewriting everything)

---

## Proposed module: `tests/_helpers/harnesses/hamilton_build.py`

### Goals baked into the API

1. **One object** that owns the canonical “build run boundary”: `TestContext` + `BuildEnv` + `HamiltonBuildExecutor`.
2. **Repo realism is opt-in** and pluggable (`repo_writers`-style functions).
3. **Seed packs remain first-class** (`ctx.require(...)`) so you can adopt Hamilton target-by-target.
4. **Minimal breakage**: you can wrap existing `TestContext` fixtures instead of creating new ones.

---

## Core types and the harness class (sketch)

```python
# tests/_helpers/hamilton_build_harness.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from codeintel.build.config import BuildConfig, load_build_config
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor, HamiltonBuildResult
from codeintel.build.providers import Providers, create_default_providers
from codeintel.config.models import ToolsConfig

from tests._helpers.build import TEST_BUILD_SETTINGS
from tests._helpers.context import SeedPack, TestContext, create_test_context
from tests._helpers.env_options import EnvOptions, GatewayOptions

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence, Callable

type RepoWriter = Callable[[Path], list[Path]]
type RepoStrategy = Literal["canonical", "writer", "none"]
type ToolingMode = Literal["default", "overrides"]


@dataclass(frozen=True)
class HarnessConfig:
    """Small config surface for stable call-sites."""

    repo: str
    commit: str
    profile: str | None = None
    file_backed_db: bool = False

    # realism knobs
    strict_contracts: bool = False
    validate_outputs: bool = False

    # executor knobs
    parallel_backend: str = "sequential"
    max_workers: int | None = None
    enable_hamilton_cache: bool = False
    cache_dir: Path | None = None


@dataclass
class HamiltonBuildHarness:
    """
    Production-parity Hamilton execution harness for tests.

    - Owns a TestContext (gateway + snapshot + build_paths)
    - Owns a frozen BuildEnv
    - Runs HamiltonBuildExecutor against real native modules
    """

    ctx: TestContext
    env: BuildEnv
    executor: HamiltonBuildExecutor
    config: HarnessConfig

    repo_files: tuple[Path, ...] = ()
    last_result: HamiltonBuildResult | None = None

    _owns_ctx: bool = True

    # ---------------------------
    # Constructors
    # ---------------------------

    @classmethod
    def open(
        cls,
        tmp_path: Path,
        *,
        harness: HarnessConfig | None = None,
        repo_strategy: RepoStrategy = "canonical",
        repo_writer: RepoWriter | None = None,
        seed_packs: Sequence[SeedPack] = (),
        gateway_options: GatewayOptions | None = None,
        tools_config: ToolsConfig | None = None,
        providers: Providers | None = None,
        build_config: BuildConfig | None = None,
    ) -> HamiltonBuildHarness:
        """
        Create an isolated harness.

        This is the “new default” entry point:
        - creates TestContext using GatewayFactory-backed helpers
        - optionally writes repo content
        - applies seed packs
        - constructs BuildEnv + HamiltonBuildExecutor
        """
        cfg = harness or HarnessConfig(repo="test_repo", commit="deadbeef")

        # Make build_dir live *inside* repo_root for realism unless caller overrides.
        repo_root = tmp_path / "repo"
        build_dir = repo_root / "build"
        db_path = build_dir / "db" / "codeintel.duckdb"

        env_opts = EnvOptions(
            repo=cfg.repo,
            commit=cfg.commit,
            file_backed=cfg.file_backed_db,
            repo_root=repo_root,
            build_dir=build_dir,
            db_path=db_path if cfg.file_backed_db else None,
        )

        ctx = create_test_context(tmp_path, options=env_opts, gateway_options=gateway_options)

        # Repo wiring
        written: list[Path] = []
        if repo_strategy == "canonical":
            ctx.ensure_canonical_repo()
        elif repo_strategy == "writer":
            if repo_writer is None:
                raise ValueError("repo_strategy='writer' requires repo_writer=...")
            written = repo_writer(ctx.repo_root)
        elif repo_strategy == "none":
            pass
        else:
            raise ValueError(f"Unknown repo_strategy: {repo_strategy}")

        # Seed wiring
        if seed_packs:
            # NOTE: seed packs today assume canonical repo in many cases.
            # Keeping this explicit makes migration safer.
            ctx.require(*seed_packs)

        # Tools + providers
        resolved_tools = tools_config or ToolsConfig.default()
        resolved_providers = providers or create_default_providers(resolved_tools)

        # BuildConfig
        resolved_build_config = build_config or load_build_config(ctx.repo_root)

        env = BuildEnv(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.build_paths,
            providers=resolved_providers,
            config=resolved_build_config,
            settings=TEST_BUILD_SETTINGS,
            profile=cfg.profile,
            validate_outputs=cfg.validate_outputs,
            strict_contracts=cfg.strict_contracts,
        )

        executor = HamiltonBuildExecutor(
            profile=cfg.profile,
            parallel_backend=cfg.parallel_backend,
            max_workers=cfg.max_workers,
            enable_cache=cfg.enable_hamilton_cache,
            cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        )

        return cls(
            ctx=ctx,
            env=env,
            executor=executor,
            config=cfg,
            repo_files=tuple(written),
            _owns_ctx=True,
        )

    @classmethod
    def wrap(
        cls,
        ctx: TestContext,
        *,
        harness: HarnessConfig | None = None,
        tools_config: ToolsConfig | None = None,
        providers: Providers | None = None,
        build_config: BuildConfig | None = None,
    ) -> HamiltonBuildHarness:
        """
        Wrap an existing TestContext fixture without owning its lifecycle.
        Ideal for incremental migration: keep existing fixtures, add Hamilton.
        """
        cfg = harness or HarnessConfig(repo=ctx.snapshot.repo, commit=ctx.snapshot.commit)

        resolved_tools = tools_config or ToolsConfig.default()
        resolved_providers = providers or create_default_providers(resolved_tools)
        resolved_build_config = build_config or load_build_config(ctx.repo_root)

        env = BuildEnv(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            paths=ctx.build_paths,
            providers=resolved_providers,
            config=resolved_build_config,
            settings=TEST_BUILD_SETTINGS,
            profile=cfg.profile,
            validate_outputs=cfg.validate_outputs,
            strict_contracts=cfg.strict_contracts,
        )

        executor = HamiltonBuildExecutor(
            profile=cfg.profile,
            parallel_backend=cfg.parallel_backend,
            max_workers=cfg.max_workers,
            enable_cache=cfg.enable_hamilton_cache,
            cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
        )

        return cls(
            ctx=ctx,
            env=env,
            executor=executor,
            config=cfg,
            _owns_ctx=False,
        )

    # ---------------------------
    # Lifecycle
    # ---------------------------

    def close(self) -> None:
        if self._owns_ctx:
            self.ctx.close()

    def __enter__(self) -> HamiltonBuildHarness:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # ---------------------------
    # Convenience: env mutations
    # ---------------------------

    def with_force_targets(self, *targets: str) -> HamiltonBuildHarness:
        self.env = replace(self.env, force_targets=frozenset(targets))
        return self

    def with_profile(self, profile: str | None) -> HamiltonBuildHarness:
        self.env = replace(self.env, profile=profile)
        return self

    def with_build_config(self, config: BuildConfig) -> HamiltonBuildHarness:
        self.env = replace(self.env, config=config)
        return self

    # ---------------------------
    # Execution API
    # ---------------------------

    def run(self, *targets: str) -> HamiltonBuildResult:
        """
        Execute the full dependency closure for requested targets.
        """
        result = self.executor.run(env=self.env, targets=list(targets))
        self.last_result = result
        return result

    def record(self, target: str, *, result: HamiltonBuildResult | None = None):
        """
        Get TargetRunRecord for `target` from a result (or last_result).
        """
        resolved = result or self.last_result
        if resolved is None:
            raise RuntimeError("No HamiltonBuildResult available yet; call run() first.")
        return resolved.get_record(target)

    # ---------------------------
    # Optional bridging helpers (migration)
    # ---------------------------

    def require(self, *packs: SeedPack) -> HamiltonBuildHarness:
        """
        Apply seed packs after construction (useful in tests that branch).
        """
        self.ctx.require(*packs)
        return self
```

---

## Why this plugs into your existing helpers cleanly

### GatewayFactory

* `create_test_context(...)` already routes through `build_test_gateway(...)` which routes through `GatewayFactory` and schema/view setup.
* Harness doesn’t reinvent gateway creation; it just **standardizes build_dir placement** (inside repo root by default for realism).

### repo_writers

You can pass any existing writer from `tests/_helpers/orchestration/repo_writers.py`:

```python
from tests._helpers.hamilton_build_harness import HamiltonBuildHarness, HarnessConfig
from tests._helpers.orchestration.repo_writers import write_sample_repo

def test_modules_end_to_end(tmp_path):
    with HamiltonBuildHarness.open(
        tmp_path,
        harness=HarnessConfig(repo="r", commit="c"),
        repo_strategy="writer",
        repo_writer=write_sample_repo,
    ) as h:
        result = h.run("modules")
        assert result.success
        assert h.record("modules").status == "succeeded"
```

### seed packs

You can keep your current seed-first tests and start swapping assertions to “run the DAG”:

```python
from tests._helpers import CORE_PACK, GRAPH_PACK
from tests._helpers.hamilton_build_harness import HamiltonBuildHarness

def test_function_metrics_from_seeded_graph(tmp_path):
    with HamiltonBuildHarness.open(
        tmp_path,
        repo_strategy="canonical",
        seed_packs=(CORE_PACK, GRAPH_PACK),
    ) as h:
        # Now run an actual Hamilton target that consumes those tables.
        result = h.run("function_metrics")
        assert result.success
```

---

## Incremental migration patterns this API enables

### Pattern A: Wrap existing fixtures (lowest risk)

If you already have a `graph_ctx` / `full_context` fixture returning `TestContext`, don’t rebuild the world:

```python
def test_something(graph_ctx):
    h = HamiltonBuildHarness.wrap(graph_ctx)
    result = h.run("risk_factors")
    assert result.success
```

### Pattern B: Seed-first now, repo_writers later

Start with canonical + seed packs to stabilize semantics; switch to repo_writers when you’re ready to validate ingestion realism:

* Phase 1: `repo_strategy="canonical" + seed_packs=(...)`
* Phase 2: `repo_strategy="writer" + repo_writer=write_sample_repo` and remove packs as upstream Hamilton targets take over.

### Pattern C: Config realism (options hashes + behavior toggles)

Because harness loads `config/codeintel.build.toml` by default, you can easily do:

1. write a config file into `repo_root`
2. run a target
3. assert behavior changes

You already have `tests._helpers.build.write_build_config(...)`; wire it like:

```python
from tests._helpers.build import write_build_config
from codeintel.build.config import load_build_config

def test_options_change_affects_run(tmp_path):
    with HamiltonBuildHarness.open(tmp_path) as h:
        write_build_config(h.ctx.repo_root, {"analytics.function_history": {"max_history_days": 7}})
        h.with_build_config(load_build_config(h.ctx.repo_root))

        result = h.run("function_history")
        assert result.success
```

---

## Where I would “hook” this into pytest ergonomically

Even though you only asked for the API sketch, here’s the cleanest **drop-in** way to use it suite-wide:

1. Put the harness in:
   **`tests/_helpers/hamilton_build_harness.py`**

2. Re-export lazily via `tests/_helpers/__init__.py` by adding:

```python
"tests._helpers.hamilton_build_harness": (
    "HarnessConfig",
    "HamiltonBuildHarness",
),
```

3. Optional fixture in `tests/conftest.py`:

```python
import pytest
from tests._helpers.hamilton_build_harness import HamiltonBuildHarness

@pytest.fixture
def hamilton_harness(tmp_path):
    with HamiltonBuildHarness.open(tmp_path) as h:
        yield h
```

This gives you a “standard harness” while still allowing per-test customization.

---

## If you want the next step

If you tell me which targets you most want to migrate first (e.g., `modules`, `typing`, `scip`, `call_graph`, `function_metrics`), I can extend this sketch with **two concrete bridging helpers** that usually matter immediately:

1. `HarnessArtifacts`: a small helper to generate `.coverage`, `pytest-report.json`, and dummy SCIP artifacts in the exact paths your targets search.
2. `ManifestPriming`: a helper to “pretend upstream targets already ran” (by inserting manifests and/or minimal run records) so downstream targets can be tested without paying the upstream compute cost—while still exercising skip logic.

But the class above is the minimal, production-aligned nucleus that cleanly plugs into what you already have.
