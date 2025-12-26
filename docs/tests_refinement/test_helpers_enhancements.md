Below is a “next wave” set of **test-helper upgrades** and **new helpers** that would materially increase **production realism** (while still keeping things inside pytest and avoiding true end-to-end deployments).

I’m basing this on what you already have in Phase4, especially:

* `tests/_helpers/harnesses/hamilton_build.py` (HamiltonBuildHarness)
* `tests/_helpers/hamilton_harness_artifacts.py` (HarnessArtifacts)
* `tests/_helpers/hamilton_manifest_priming.py` (ManifestPriming)
* `tests/_helpers/tool_sandbox.py` + `tests/_helpers/orchestration/tooling.py` (to be replaced by real-tool invocation recording + strict ToolRunner wiring)
* `tests/_helpers/orchestration/provisioning.py` (non-Hamilton ingestion provisioning)
* the various harness wrappers (analytics/graph/serving) and assertion modules

---

## 1) Upgrades to existing helpers that most increase realism

### A. Make “production-parity harness defaults” the *default* harness config

Right now, the harnesses are very capable, but tests only get “realistic settings” if the author remembers to opt in (file-backed DB, real-ish tool behaviors, concurrency settings, etc.).

**Upgrade**: make production-parity the default for all harness-backed tests, with little to no opt-out.

Concretely:

#### Make `HarnessConfig` default to production-parity

In `tests/_helpers/harnesses/hamilton_build.py`:

* Set **defaults** (not just a preset) to production-parity values:

  * `file_backed_db=True` (most important realism bump)
  * `parallel_backend="threadpool"` and `max_workers` to something small-but-real (e.g. 4)
  * `enable_hamilton_cache=True` with a per-test cache dir
  * `validate_outputs=True` and `strict_contracts=True`
  * mirror production `BuildSettings` values that affect DAG realism
    (e.g., contract validation strategy and saver-derived support surfaces)

* If a parity override *must* exist, it should be **rare** and **explicit**, and only for tests
  that never touch the build/runtime pipeline (pure algorithmic unit tests). All tests that
  touch build, tool, or gateway surfaces should stay on parity defaults.

Why it matters:

* File-backed + parallel execution catches issues you’ll never see in memory mode (locks, path assumptions, pool behaviors, atomic writes, etc.).
* Cache-on catches “stale reads” and dependency tracking bugs early (especially once you lean into manifest/skip logic).

#### Add a preset on `HarnessOpenOptions`

Add something like `HarnessOpenOptions.production_repo(...)` that defaults to:

* `repo_strategy="canonical"` (or `"writer"` for suite-specific cases)
* `seed_packs` = minimal (or none) by default, so tests are naturally forced to rely on **Hamilton-derived outputs**, not hand inserts.

---

### B. Make real tool binaries a first-class harness default (no stubs)

We must always execute **real production tool binaries** in tests (no fakes, no stubs).

**Upgrade**: make `HamiltonBuildHarness.open(...)` always resolve real `ToolsConfig` and fail fast
if any tool is missing.

Concretely:

* Add a **mandatory** tool resolution step in `HamiltonBuildHarness.open(...)`:

  * resolve `ToolsConfig` from the environment or explicit paths
  * verify every required tool binary exists and is executable
  * raise a clear error when a tool is missing (do **not** fallback to stubs)

Why it matters:

* Tool realism becomes the default instead of an afterthought.
* You test “production wiring” more often: `ToolsConfig → ToolRunner → ToolService → ingestion/build nodes`.

---

### C. Replace `ToolSandbox` with a real-tool invocation recorder + contract validator

`ToolSandbox` is built for stubs, which we will not use. We still need the same
*contract enforcement* and *observability*, but against real binaries.

**Upgrade**: replace `tests/_helpers/tool_sandbox.py` with a real-tool recorder/validator that
wraps `ToolRunner`.

Capabilities:

1. **Invocation recording**
   Have the ToolRunner wrapper append a JSON line to a log file if an env var is set, e.g.:

* `CODEINTEL_TOOL_CALL_LOG=/tmp/.../tool_calls.jsonl`

Each line records:

* tool name
* argv
* cwd
* selected env keys (or a hash)
* timestamp
* return code
* elapsed time

This lets you add realism assertions like:

* “modules target should not call pyright”
* “typing target must call pyright with `--outputjson`” (or whatever your contract is)

Per-test isolation:

* ensure each test gets its own log file (e.g., under `tmp_path`) to avoid interleaved
  calls when `threadpool` execution is enabled.

2. **Argument-sensitive behavior**
   Extend the recorder to support per-tool **expected args** assertions:

* `expected_args_contains: list[str]` (or regex)
* `fail_if_missing: list[str]`

This makes failures meaningful and prevents tests from passing while the code is calling tools incorrectly.

3. **Add “version” handling**
   A lot of production code eventually starts caring about tool versions.
   Have the recorder capture:

* `--version` strings when invoked
* and allow tests to assert minimum/expected versions where necessary.

---

### D. Align provisioning flows with Hamilton to avoid “two pipelines” drift

Your provisioning path is currently realistic in spirit, but it is **not Hamilton-native**:

* `tests/_helpers/orchestration/provisioning.py` uses `RepoScanStep`, `materialize_repo_scan_result`, `_run_ingestion_steps`, etc.

If the design goal is “Hamilton DAG derived outputs are the orchestration truth,” then the most production-realistic provisioning fixture is:

> “write repo → run Hamilton targets → use gateway + artifacts produced by those targets”

**Upgrade**:

* Introduce `provision_hamilton_repo(...)` alongside (or replacing) `provision_ingested_repo(...)`.

It should:

* create a `HamiltonBuildHarness` with parity config
* run targets required for “baseline realism” (modules first, then typing/coverage/etc as needed)
* return the same `ProvisionedGateway` shape, but built from DAG outputs

Why it matters:

* This prevents regression where provisioning is “green” but the Hamilton pipeline is broken (or vice versa).
* It also forces you to keep DAG contracts stable.

---

### E. Add “incremental/skip realism” to `ManifestPriming` and harness APIs

You already have a good `ManifestPriming`, but you can push realism further by baking in the behaviors you care about long-term:

**Upgrades**:

1. Add `prime_target_as_upstream_success(...)` that writes *both*:

* the manifest
* a minimal run record row / run status evidence your execution layer expects
  (aligned with `BuildRunWriter` semantics where possible)

2. Add `assert_incremental_behavior(...)` helpers:

* run target once → expect `ran`
* run again without changes → expect `skipped` (or “cached”)
* touch repo file / change config input → expect `ran` again

Even if the underlying status labels change, encoding the *behavior contract* is a huge realism win.

Where: `tests/_helpers/hamilton_manifest_priming.py` + small convenience methods on `HamiltonBuildHarness`.

---

### F. Add DAG validation + output inventory drift checks as harness-first helpers

Given the DAG-first architecture, tests should not pass if DAG invariants or output inventories drift.

**Upgrade**:

* Add `HamiltonBuildHarness.validate_graph()` that calls `validate_graph()` and fails on any invariant
  issues (including compute I/O purity checks).
* Add `HamiltonBuildHarness.assert_output_inventory_consistent()` that compares saver-derived
  outputs to registry/contract inventories and fails on mismatches (with explicit strictness).

This makes DAG correctness a *default* preflight for tests that exercise build surfaces.

## 2) Additional helpers you don’t really have yet (and that will pay off fast)

### A. ServingSnapshotFactory (unifies all the “manual DB + registry + manifest + pointer” patterns)

Your serving tests repeatedly do some form of:

* create duckdb file
* write registry JSON
* write schema manifest JSON
* write buildspec JSON
* write `current.json`

You have pieces (`HarnessArtifacts`), but there’s no single “make me a serving snapshot exactly like production would expect” helper.
This should **replace** `tests/_helpers/serving_snapshots.py`, with the current
`setup_demo_snapshot` becoming a method/variant on the new factory.

**New helper**: `tests/_helpers/serving_snapshot_factory.py` (replaces `tests/_helpers/serving_snapshots.py`)

API sketch:

```py
@dataclass(frozen=True)
class ServingSnapshot:
    serve_dir: Path
    db_path: Path
    registry_path: Path
    schema_manifest_path: Path
    buildspec_path: Path
    pointer_path: Path

    repo: str
    commit: str
    run_id: str

class ServingSnapshotFactory:
    def __init__(self, tmp_path: Path) -> None:
        ...

    def demo_snapshot(
        self,
        *,
        repo: str = "demo/repo",
        commit: str = "deadbeef",
        run_id: str = "run-1",
        row_count: int = 3,
    ) -> ServingSnapshot:
        """Replacement for setup_demo_snapshot."""
        ...

    def make_snapshot(
        self,
        *,
        repo: str = "demo/repo",
        commit: str = "deadbeef",
        run_id: str = "run-1",
        views: list[dict[str, object]] | None = None,
        tables: list[dict[str, object]] | None = None,
        db_setup: Callable[[Path], None] | None = None,
        use_production_pointer_model: bool = True,
    ) -> ServingSnapshot:
        ...
```

Key realism boosts:

* Use `codeintel.serving.db.pointer.ServingSnapshotPointer` to write pointer JSON (instead of hand dicts).
* Optionally run the same publish function production uses (or `ServingTargetHarness.publish_snapshot(...)`)
  to exercise the Hamilton artifact pipeline and pointer atomics.

This single helper will make **HTTP tests**, **MCP tests**, and **hot-swap tests** consistent and production-aligned.

---

### B. ServingAppHarness and McpAppHarness (runtime-level realism, minimal boilerplate)

You already have `ServingTargetHarness` (build-side), but you don’t have a harness that says:

> “Given a snapshot on disk, build the app, start it, give me a client, and cleanly stop it.”

**New helper**: `tests/_helpers/harnesses/serving_app.py`

What it should provide:

* `serving_client(snapshot, settings_overrides=...) -> TestClient`
* `mcp_client(snapshot, settings_overrides=...) -> fastmcp.client.Client`

This harness should:

* create settings with realistic defaults (`pool_size`, `poll_interval_s`, `schema_enforcement`)
* start/stop properly (for HTTP, `TestClient` handles lifespan; for MCP you can manage `ServingDBManager` lifecycle like your tests already do)
* optionally support `hot_swap=True` test mode
* consume snapshots produced by `ServingSnapshotFactory` (including published mode via
  `ServingTargetHarness.publish_snapshot(...)`) to avoid divergent serving layouts

This is one of the biggest “realism per line of code removed” wins.

---

### C. Eventually / async wait helpers for hot-swap + polling tests

You have hot-swap, pollers, rate limiting, and streaming behavior. Those tests become flaky without a disciplined “eventually” helper.

**New helper**: `tests/_helpers/waiting.py`

* `eventually(assert_fn, timeout_s=..., interval_s=...)`
* `await eventually_async(assert_fn, timeout_s=..., interval_s=...)`

This lets you keep **real polling loops** enabled (realism) without writing bespoke sleeps everywhere.

Implementation details to keep tests stable and debuggable:

* use `time.monotonic()` for timeout tracking
* preserve the last assertion error/message (and diff where available) for failure output

---

### D. Golden dataset snapshot helpers for “regression-grade realism”

If your explicit goal is “unit tests should tell me if production behavior regressed,” you eventually need:

> “Run pipeline on canonical repo → compare key outputs to goldens.”

You already have a golden diff formatter for *path/module sets* (`tests/_helpers/assertions/golden_diffs.py`), but not for **tables** and **artifacts** broadly.

**New helper**: `tests/_helpers/goldens/table_goldens.py`

Capabilities:

* `dump_table(gateway, table_key) -> stable dataframe` (sorted, normalized types)
* `assert_table_matches_golden(..., golden_path=...)`
* good diff output (row count delta, key-based mismatch report)

Start with a tiny set of tables:

* `core.modules`
* `core.repo_map`
* `docs.schema_manifest`-equivalent artifact outputs (JSON)
* and one “graph edge” table

This is the most direct way to get “audit-grade regression detection” without a full E2E harness.

---

### E. Real tool execution policy + audit helpers (always on)

We will **always** run real tool binaries in tests. No stubs, no replay.

**New helper**: `tests/_helpers/tooling_audit.py`

* `require_tooling()` that verifies tool availability early (fail fast)
* `tool_call_log()` fixture that exposes the JSONL call log from the ToolRunner recorder
  (scoped per test to avoid threadpool interleaving)
* optional `assert_tool_called(...)` helpers for per-test expectations

---

## 3) Where to place these helpers in your pytest scope

Here’s a placement scheme that stays consistent with how your suite is already organized:

### Modify existing helpers in-place

* `tests/_helpers/harnesses/hamilton_build.py`

  * make parity defaults the standard config
  * add real tool resolution + validation hooks
  * add incremental/skip convenience methods
  * add DAG validation + output inventory helpers

* Replace `tests/_helpers/tool_sandbox.py`

  * with `tests/_helpers/tooling_audit.py` (real-tool invocation recorder + validator)

* `tests/_helpers/orchestration/provisioning.py`

  * add `provision_hamilton_repo(...)` (leave the old path for now, but migrate callers)
  * optionally make `ProvisioningConfig(run_ingestion=True)` route to Hamilton

* Replace `tests/_helpers/serving_snapshots.py`

  * with `tests/_helpers/serving_snapshot_factory.py` (including a `setup_demo_snapshot`-equivalent)

* `tests/_helpers/hamilton_manifest_priming.py`

  * add incremental assertions + richer “pretend upstream ran” records

### Add new helpers

* Snapshot + app harnesses:

  * `tests/_helpers/serving_snapshot_factory.py`
  * `tests/_helpers/harnesses/serving_app.py`
  * (optional) `tests/_helpers/harnesses/mcp_app.py` if you want it separate

* Async waiting utilities:

  * `tests/_helpers/waiting.py`

* Goldens:

  * `tests/_helpers/goldens/`

    * `__init__.py`
    * `table_goldens.py`
    * maybe `artifact_goldens.py` for JSON artifacts

* Tooling audit:

  * `tests/_helpers/tooling_audit.py`
  * (optional) `tests/_helpers/pytest_options.py` if you want centralized CLI options

### Wire into pytest fixtures

* Put **generic** fixtures in `tests/conftest.py`:

  * `require_tooling` (session, fail fast on missing binaries)
* `tool_call_log` (function-scoped, per-test log file)
  * `parity_harness_config` (function or session)
  * `serving_snapshot` factory fixture
  * `eventually` helpers (fixture or import)

* Put **domain-specific** fixtures in `tests/serving/conftest.py` (optional) to avoid global clutter:

  * `serving_app_client`
  * `mcp_client`
  * `hot_swap_runtime`

### Markers/options

You already have `integration`, `e2e`, `performance`, `load`, etc in `pytest.ini`.

Given parity is now the default, I would **not** add a `production_parity` marker.
If an escape hatch is absolutely necessary, add a narrowly scoped `non_parity` marker
that is allowed only for pure, non-I/O unit tests and should not be used for anything
touching build, tool, or gateway surfaces.

---

## The practical “max realism” path I’d actually execute

If your real intent is: “unit tests alone should certify production readiness,” then in practice you want:

1. **Production-parity defaults** on the core harness (`HamiltonBuildHarness`)
2. **ServingSnapshotFactory + ServingAppHarness** so serving tests stop hand-rolling snapshots
3. **Eventually helpers** to keep hot-swap/polling tests realistic but stable
4. **Golden tables/artifacts** for a small canonical repo to detect true regressions
5. **Real-tool execution + invocation recorder** so tooling behavior is always production-aligned

That combination will move you from “tests are thorough” to “tests are regression-grade.”

If you want, I can take one concrete slice (e.g., **serving tests** or **graph/analytics tests**) and outline exactly which existing tests become shorter and more production-realistic once `ServingSnapshotFactory` + `ServingAppHarness` exist—similar to the “before → after” planning we did earlier, but at the *helper architecture* level.


# serving focused helper-architecture plan to serve as a demonstration of the above plan #

Below is a **serving-focused (“serving slice”) helper-architecture plan** showing **exactly which existing tests** in your current suite get **shorter** *and* **more production-realistic** once **ServingSnapshotFactory** + **ServingAppHarness** exist.

I’m basing this on what’s in the latest zip you uploaded (Phase4): in `tests/serving/` there are **19 separate files** that each re-implement a `_write_pointer` + snapshot seeding pattern (and many also hand-roll DB + artifacts + `ServingDBManager` lifecycle).

---

## What’s repetitive today in the serving slice

Across `tests/serving/**` you currently have:

* **19** files that define their own `def _write_pointer(...)` (same shape, small variations).
* **61** occurrences of `HarnessArtifacts(...)` usage inside serving tests.
* **67** occurrences of `ServingDBManager` usage inside serving tests (many manually `await start()` / `await stop()`).

That repetition is the single biggest place where a “realistic harness” improves both:

1. **production parity** (same pointer location, same start/stop lifecycles, same snapshot layout), and
2. **test clarity** (tests focus on behavior, not scaffolding).

---

## The two helpers and what they change architecturally

### ServingSnapshotFactory

A factory that creates **realistic on-disk serving snapshot state** in the same shape production expects:

* **Always** creates a `serve_dir/` with:

  * `serve_dir/current.json`
  * `serve_dir/exports/` (for export artifacts)
  * optionally `serve_dir/snapshots/<run_id>/…` (if you choose “published snapshot mode”)
* Produces a single object you pass around, e.g. `snapshot.serve_dir`, `snapshot.pointer_path`, `snapshot.db_path`, `snapshot.registry_path`, etc.
* Supports *modes* so you can choose realism level per test:

  * **seeded**: fast, writes `db + semantic_registry + schema_manifest + buildspec + current.json`
  * **published**: runs `publish_serving_snapshot(...)` so you exercise:

    * atomic pointer semantics
    * snapshot directory layout
    * search index + lineage preflight requirements

### ServingAppHarness

A harness that wires **ServingSettings + runtime + app clients** in a production-aligned way:

* **FastAPI harness**: creates `create_serving_app(...)` and gives you a `TestClient` that correctly runs the app lifespan (so `ServingDBManager.start/stop` happens the production way).
* **FastMCP harness**: creates `build_mcp_app(...)` with a lifespan that starts/stops the same DB manager (so MCP tests don’t need to manually `await manager.start()` / `await manager.stop()`).
* Gives consistent affordances:

  * `.http_client()` context manager
  * `.mcp_client()` async context manager
  * `.settings` with predictable defaults + per-test overrides
  * optional toggles: `mount_mcp`, auth headers, rate limit settings, cache flags, etc.

The net effect: tests stop “constructing the world” and instead say **“give me a realistic world in state X”**.

---

## Exactly which existing tests get shorter and more production-realistic

I’ll group these by the *kind of realism* you gain and *which helper* does the work.

---

# 1) FastAPI HTTP integration tests (biggest immediate win from both helpers)

These tests currently:

* create a DuckDB file manually
* write registry/manifest/buildspec manually (often via `HarnessArtifacts`)
* write a pointer JSON by hand (often under `serve_dir/current.json`)
* instantiate `ServingSettings`
* call `create_serving_app`
* wrap in `TestClient`

Once **ServingSnapshotFactory + ServingAppHarness** exist, these become ~3–10 lines of setup + assertions.

**Files:**

1. `tests/serving/test_semantic_http_routes.py`
2. `tests/serving/http/test_export.py`
3. `tests/serving/test_http_mcp_integration.py`
4. `tests/serving/http/test_mcp_mount.py`

### What gets deleted/replaced in each file

* Delete per-file helpers like:

  * `_make_db`
  * `_write_registry`, `_write_schema_manifest`, `_write_buildspec`
  * `_write_pointer`
  * bespoke `_setup_serving_env(...)` functions
* Replace with:

  * `snapshot = ServingSnapshotFactory(tmp_path).demo_view_snapshot(...).publish_current(...)`
  * `with ServingAppHarness(snapshot, mount_mcp=...).http_client() as client: ...`

### Why this becomes more production-realistic

* **Pointer location** becomes consistent: `serve_dir/current.json` (instead of mixed patterns like `tmp_path/current.json`).
* **DB manager lifecycle** runs the same way production does: via the FastAPI lifespan.
* If you flip these tests to **published snapshot mode**, you also start exercising:

  * atomic pointer updates
  * snapshot directory layout under `serve_dir/snapshots/<run_id>/`
  * publisher preflight logic (search index + lineage requirements)

**Typical before → after change (architecture-level)**

* **Before:** each test file re-creates its own mini “serving install”.
* **After:** the harness produces a realistic serving install once, tests just drive behavior.

---

# 2) MCP tool + resource tests (ServingSnapshotFactory + ServingAppHarness removes manual DBManager wiring)

These tests currently:

* build a db + registry/manifest/buildspec + pointer
* construct `ServingDBManager(pointer_path=...)`
* manually `await manager.start()` / `await manager.stop()`
* construct `SemanticQueryKernel`
* construct `build_mcp_app(kernel=..., settings=...)`
* use `fastmcp.client.Client(...)`

Once **ServingSnapshotFactory + ServingAppHarness** exist:

* You stop manually starting/stopping the DB manager.
* You stop passing “random pointer paths”; everything is rooted at `serve_dir`.
* You get a consistent MCP runtime with production middleware stack applied the same way.

**Files:**

1. `tests/serving/test_semantic_mcp_tools.py`
2. `tests/serving/test_mcp_middleware_rate_limit.py`
3. `tests/serving/test_serving_meta_tooling_mismatch_warnings.py`
4. `tests/serving/mcp/test_resources.py`
5. `tests/serving/mcp/test_tasks.py`
6. `tests/serving/mcp/test_sampling.py`
7. `tests/serving/mcp/test_prompts_advanced.py`
8. `tests/serving/mcp/test_meta_sql_resources.py`
9. `tests/serving/mcp/test_sql_fingerprint.py`
10. `tests/serving/mcp/test_middleware_logging_smoke.py`

### What gets shorter

* All of these currently have local snapshot helper blocks (often identical).
* Many define `_setup_test_snapshot(...)` returning a pointer path; that disappears.
* You get a uniform pattern:

  * `snapshot = ServingSnapshotFactory(...).demo_view_snapshot(...).publish_current(...)`
  * `async with ServingAppHarness(snapshot).mcp_client() as client: ...`

### What becomes more production-realistic

* MCP app runs with the **same** settings-derived middleware stack and feature gating you use in production (you already do this; the harness just makes it consistent and eliminates the temptation to “mock around it”).
* DB readiness becomes “real” (Client lifespan + db manager start/stop), rather than manual ad-hoc sequencing.
* Export TTL tests become more realistic because the harness always creates `serve_dir/exports/` and you can centralize cleanup timing knobs.

---

# 3) Semantic kernel + compiler upgrade gate tests (ServingSnapshotFactory removes the snapshot boilerplate; ServingAppHarness can standardize lifecycle)

These tests don’t need FastAPI, but they still have heavy repeated snapshot setup and manager lifecycle code.

**Files:**

1. `tests/serving/semantic/test_kernel.py`
2. `tests/serving/semantic/test_compiler_upgrade_gates.py`
3. `tests/serving/semantic/test_pr87_allowed_columns_enforced.py`
4. `tests/serving/semantic/test_pr88_polars_execution_path.py`

### What gets shorter

* All their per-file DB + registry + manifest + buildspec + pointer creation becomes a one-liner factory call.
* Manual `ServingDBManager.start/stop` can be standardized into a harness context manager (even if you don’t build FastAPI/MCP apps).

### What becomes more production-realistic

* Your snapshot creation becomes consistent with:

  * the same pointer placement (`serve_dir/current.json`)
  * the same metadata requirements (if/when you choose “published snapshot” mode)
* The “compiler upgrade gate” tests become more meaningful if you choose to generate the registry/manifest/buildspec using the same production path (Hamilton artifacts or publisher), because you’re validating SQL stability against more production-shaped metadata.

---

# 4) DB manager hot-swap behavior tests (ServingSnapshotFactory enables realistic pointer writing + multi-snapshot rotation)

**File:**

1. `tests/serving/db/test_manager.py`

### What gets shorter

* This file currently re-implements:

  * `_write_registry`, `_write_schema_manifest`, `_write_buildspec`
  * `_write_pointer`
  * `_make_db`
* Snapshot factory replaces *all* of those.

### What becomes more production-realistic

This test suite is *specifically* about pointer hot swap, so it benefits strongly from snapshot factory supporting:

* `snapshot1 = factory.snapshot(run_id="run-1", db_value=1)`
* `snapshot2 = factory.snapshot(run_id="run-2", db_value=2)`
* `factory.point_current_to(snapshot1)` then `factory.point_current_to(snapshot2)`

If you run these in **published snapshot mode**, you also get production-true behavior for:

* pointer atomic writes
* snapshot folder structure
* retention behavior (`keep_last`)

---

## Tests that won’t materially change from these helpers (and that’s fine)

A few serving tests are already short and focused, and shouldn’t be forced onto the harness path:

* `tests/serving/db/test_pointer.py` (pure pointer parsing/roundtrip unit tests)
* `tests/serving/test_settings_mcp_worker_guardrails.py` (settings validation / app factory guardrails)
* `tests/serving/http/test_metrics.py` (metrics behavior unit tests)
* `tests/serving/test_streaming_ndjson.py` (streaming parser)

These can stay as-is; the goal is reducing bespoke scaffolding where realism matters.

---

## The “before → after” payoff summary for the serving slice

Once **ServingSnapshotFactory + ServingAppHarness** exist, you can realistically expect:

* The **19 pointer-seeding tests** to drop their local snapshot setup blocks.
* The **4 HTTP integration files** to converge on a single “production-shaped app + client” pattern.
* The **10 MCP files** to stop manually managing DB manager lifecycle and pointer placement.
* The semantic kernel/upgrade gate tests to stop being “DB + JSON glue tests” and become “semantic behavior tests”.

---

## Practical migration order for the serving slice

If you want the fastest win with minimal assertion churn:

1. Migrate **HTTP route tests** first:

   * `test_semantic_http_routes.py`
   * `http/test_export.py`
   * `test_http_mcp_integration.py`
   * `http/test_mcp_mount.py`

2. Then migrate **MCP tests** (they get *much* cleaner once `.mcp_client()` owns lifespan).

3. Then migrate the **kernel/compiler** tests (snapshot factory first; lifecycle harness second).

4. Then migrate **DB manager** tests to snapshot factory rotation / published snapshot mode.

---

If you want, I can take **one of the MCP-heavy files** (for example `tests/serving/mcp/test_resources.py`, which currently contains a lot of snapshot scaffolding) and outline a concrete **module-level refactor plan**: what fixtures you introduce, what local helpers you delete, and what the “new normal” test shape becomes when it runs entirely through `ServingSnapshotFactory` + `ServingAppHarness` without changing the behavioral assertions.
