Below is a “deploy what you already have” review of the **Phase4 zip** test suite, focused on places where tests still do bespoke setup that your existing helpers in `tests/_helpers` (and a couple of already-present fixtures in `tests/conftest.py`) can replace to make tests both **more realistic** and **less repetitive**.

I’m going to be concrete: **file paths → what to replace → which helper to use → what realism you gain**.

---

## Where you’re already leveraging helpers well (so we don’t waste effort)

You already have strong “production-aligned” patterns in several areas:

* **Ingestion**: e.g. `tests/ingestion/test_module_inventory.py` uses `module_inventory_context`, `modules_expected_from_repo_tree`, and the missing/extra formatter. Good realism + low bespoke code.
* **Hamilton target plumbing**: e.g. `tests/ingestion/test_runner_plumbing.py` uses `HamiltonBuildHarness`, `HarnessArtifacts`, and `ManifestPriming`.
* **Tool realism**: e.g. `tests/ingestion/test_scip_ingest.py` uses `ToolSandbox` + `HarnessArtifacts.write_dummy_scip_artifacts()` instead of mocking.
* **Graph tests**: often use `GraphTargetHarness` and networkx fakes.
* **Serving demo snapshots**: multiple serving tests already use
  `tests/_helpers/serving_snapshots.py::setup_demo_snapshot`, which is exactly the
  right “existing helper” to standardize on for demo.view-based tests.

So: the biggest remaining ROI is **Serving** (HTTP/MCP/semantic kernel tests) and **env var tests**.

---

## Biggest under-deployment: Serving tests are still hand-rolling “serving snapshots”

### What I observed

A large cluster of serving tests all implement their own versions of:

* `_make_db(...)` via raw `duckdb.connect()`
* `_write_registry / _write_schema_manifest / _write_buildspec` (sometimes via `HarnessArtifacts`, sometimes manual JSON)
* `_write_pointer(...)` writing `current.json` by hand
* `_setup_test_snapshot(...)` repeated per file

This shows up in (non-exhaustive, but it’s the main cluster):

**Snapshot + pointer scaffolding repeated**

* `tests/serving/test_http_mcp_integration.py`
* `tests/serving/test_mcp_middleware_rate_limit.py`
* `tests/serving/test_serving_meta_tooling_mismatch_warnings.py`
* `tests/serving/http/test_export.py`
* `tests/serving/mcp/test_sampling.py`
* `tests/serving/mcp/test_meta_sql_resources.py`
* `tests/serving/mcp/test_sql_fingerprint.py`
* `tests/serving/mcp/test_middleware_logging_smoke.py`
* plus several `tests/serving/semantic/*` tests that do similar snapshot setup
  (e.g., `test_kernel.py`, `test_pr87_allowed_columns_enforced.py`,
  `test_pr88_polars_execution_path.py`, `test_compiler_upgrade_gates.py`)

### Existing helpers you already have that should replace this

You already have two “right” options in `tests/_helpers`:

#### Option A (most realistic): Use the production-aligned snapshot path via `ServingTargetHarness`

Helper: `tests/_helpers/harnesses/serving_harness.py::ServingTargetHarness`

Why it’s a big realism win:

* Executes the **Hamilton DAG target** `serving_artifacts` (same path as prod).
* Publishes the snapshot using prod code: `codeintel.build.serving.publisher.publish_serving_snapshot()`.
* Produces the **exact disk layout** that Serving expects:

  * `build/serving/snapshots/<run_id>/codeintel.duckdb`
  * `build/serving/snapshots/<run_id>/{semantic_registry.json,schema_manifest.json,buildspec.json}`
  * `build/serving/current.json` written by prod code.

Where to apply it immediately:

* Any test that currently just wants “a serving snapshot exists + app starts + endpoints respond”

  * `test_http_mcp_integration.py`
  * `test_mcp_middleware_rate_limit.py`
  * `test_serving_meta_tooling_mismatch_warnings.py`
  * some MCP tool routing tests that don’t need deterministic demo rows

**Important constraints to document up front**

`ServingTargetHarness.publish_snapshot()` uses production publishing logic and will
fail fast if required tables are missing. In particular, it builds search documents
and requires lineage tables (`metadata.derived_lineage_edges` and
`metadata.derived_lineage_columns`). So it’s ideal for integration-style tests, but
for small “demo.view” tests without lineage/search data, Option B below is the right
fit.

**Concrete refactor pattern (what these files can converge on):**

```python
def test_smoke(serving_target_harness: ServingTargetHarness) -> None:
    # 1) Run the real DAG target that emits artifacts
    records = serving_target_harness.run_targets()
    assert_target_ok(records["serving_artifacts"])

    # 2) Publish a real snapshot (writes current.json)
    serving_target_harness.publish_snapshot(run_id="run-1")

    # 3) Point Serving at the real serve_dir
    serve_dir = serving_target_harness.harness.ctx.build_paths.build_dir / "serving"
    settings = ServingSettings(serve_dir=serve_dir, hot_swap=False, pool_size=1)

    app = create_serving_app(settings=settings)  # or build_mcp_app(...)
    client = TestClient(app)
    ...
```

**Important note about “demo.view” tests**
Several of your serving tests currently depend on a tiny `demo.view` registry + a handcrafted `docs.v_demo` table and then assert exact returned rows. If you migrate those tests to `ServingTargetHarness`, you’ll likely want to:

* stop asserting exact demo rows, and instead assert **endpoint correctness** on a *real registry view*, or
* seed a pack that makes a known real view non-empty, or
* keep a minimal “demo snapshot” fixture (see Option B below)

That’s not a blocker, just an expectation shift.

#### Option B (still more realistic, still uses existing helpers): keep your “demo.view” snapshot, but stop using raw DuckDB + stop re-implementing snapshot pieces in every file

Helpers you already have to do this:

* `tests/_helpers/serving_snapshots.py::setup_demo_snapshot` (already in use)
* `tests/_helpers/gateway.py::GatewayFactory` (instantiate directly in serving tests)
* `tests/_helpers/hamilton_harness_artifacts.py::HarnessArtifacts`
* `tests/_helpers/docs_views.py::create_bootstrapped_docs_db` (optional alternative to GatewayFactory)
* Production class (not in tests/_helpers, but important realism): `codeintel.serving.db.pointer.ServingSnapshotPointer`

What to change:

1. Prefer `setup_demo_snapshot(...)` wherever you just need a demo snapshot with
   deterministic rows. It already wires DB + artifacts + pointer and accepts `row_count`.

2. Where you need a custom mini-DB, replace raw `duckdb.connect()` with a
   file-backed `GatewayFactory().file_backed(db_path).open()` so schema + views are
   applied the same way prod does. Insert rows via `gw.con.execute(...)`, then close
   the gateway before Serving opens the snapshot.

3. Replace manual JSON writing with `HarnessArtifacts.write_*` **when the helper
   matches the required schema**. Note: `write_schema_manifest(...)` always writes
   `"views": []`, so tests that need view entries must keep manual JSON until helper
   changes are in scope.

4. Replace `_write_pointer()` dict dumps with `ServingSnapshotPointer(...).to_json()`
   (production class), or use `publish_serving_snapshot` via `ServingTargetHarness`
   if you’re ready.

**Where this is the best immediate fit**

* Tests that intentionally need a tiny schema + deterministic rows (your “demo view” tests):

  * already use `setup_demo_snapshot`: `tests/serving/test_semantic_http_routes.py`,
    `tests/serving/test_semantic_mcp_tools.py`, `tests/serving/mcp/test_resources.py`,
    `tests/serving/mcp/test_tasks.py`, `tests/serving/mcp/test_prompts_advanced.py`
  * should migrate to `setup_demo_snapshot`: `tests/serving/test_http_mcp_integration.py`,
    `tests/serving/test_mcp_middleware_rate_limit.py`,
    `tests/serving/test_serving_meta_tooling_mismatch_warnings.py`,
    `tests/serving/mcp/test_sampling.py`, `tests/serving/mcp/test_sql_fingerprint.py`,
    `tests/serving/mcp/test_middleware_logging_smoke.py`

**A very direct “deploy existing helpers” improvement**
In those files, you can keep your `docs.v_demo` data, but make `_make_db()` do:

* `gw = GatewayFactory().file_backed(db_path).open()`
* write/insert using `gw.con.execute(...)`
* `gw.close()` before Serving opens the snapshot

That buys you production schema bootstrapping and avoids subtly missing schemas/views.

---

## Serving tests also underuse your HTTP response assertion helpers

Helper exists: `tests/_helpers/assertions/http_responses.py`

Observed:

* Serving HTTP tests are doing lots of bespoke:

  * `expect_equal(resp.status_code, 200)`
  * manual JSON checks for problem details

Where to deploy it:

* `tests/serving/test_semantic_http_routes.py`
* `tests/serving/http/test_export.py`
* `tests/serving/test_http_mcp_integration.py`
* any route tests that inspect errors

What to replace:

* Replace GET success checks with `assert_http_success(client, "/path")`
  (note: this helper takes a client + path and performs the request).
* Replace manual error-body asserts with `assert_problem_detail_response(resp, status_code=...)`.
* Use `assert_ok_or_not_found(resp)` for endpoints where 200/404 are both valid.
* Use `assert_success_meta(payload, expect_limits=..., expect_offset=...)` for
  shared meta structures after parsing the JSON body.

This doesn’t just reduce code: it makes failures much clearer and consistent.

---

## `test_mcp_mount.py` is writing artifacts manually even though `HarnessArtifacts` exists

File:

* `tests/serving/http/test_mcp_mount.py`

Observed:

* It writes `semantic_registry.json` / `schema_manifest.json` / `buildspec.json` via direct `json.dumps` and `Path.write_text`.

Existing helper to use:

* `tests._helpers.hamilton_harness_artifacts.HarnessArtifacts`

Why it matters:

* You get canonical formatting and you stop duplicating “what does a registry look like?” shape.
* If the artifact format evolves, the helper is the one place to update.

So: this file is a quick win: swap the manual JSON writes to `HarnessArtifacts.write_*`
(the artifacts in this test don’t require view entries, so the helper is a clean fit).

---

## A lot of serving tests are duplicating env-var context managers, even though you already have cleanup

Files with repeated `_set_env(...)`:

* `tests/serving/test_settings.py`
* `tests/serving/test_auth_enforcement.py`
* `tests/serving/test_mcp_feature_flags.py`
* `tests/serving/test_mcp_metrics.py`
* `tests/serving/test_uvicorn_config.py`

Key thing you already have:

* `tests/conftest.py` defines a fixture `codeintel_env` that snapshots/restores all
  `CODEINTEL_` env vars per test (note: it is **not** autouse).

That means: in these tests you can usually delete `_set_env(...)` entirely and just:

```python
import os

def test_something(tmp_path, codeintel_env):
    os.environ["CODEINTEL_SERVE_DIR"] = str(tmp_path)
    os.environ["CODEINTEL_MCP_ENABLE_SEARCH"] = "0"
    settings = get_serving_settings()
    ...
```

The fixture restores environment after the test. For module-wide use, prefer
`@pytest.mark.usefixtures("codeintel_env")`.

If you do need scoped temporary env changes inside a single test:

* you also have `tests/_helpers/env_vars.py::{temporary_env, unset_env}`

  * (single-var granularity, but works; can be combined with `contextlib.ExitStack`)

This is pure “deploy what you already have” and it removes a *lot* of bespoke code.

---

## Medium ROI: a couple of Hamilton tests re-implement multi-env switching

Files:

* `tests/build/hamilton/adapters/test_parallel.py`
* `tests/build/hamilton/hooks/test_lifecycle.py`

Observed:

* both define a local `_temporary_env(values: dict[str, str | None])`.

Existing helper:

* `tests/_helpers/env_vars.py::{temporary_env, unset_env}`

You *can* already eliminate local helpers by nesting contexts (or `ExitStack`). It’s not as pretty as a multi-env helper, but it’s “use what you have today”.

(If/when you decide to add a multi-env helper later, these are the first two consumers.)

---

## Suggested “deployment plan” using only what you already have

This is ordered to maximize realism and reduce bespoke code quickly:

1. **Serving snapshot consolidation**

   * Pick 2–3 serving tests that are clearly “integration-like” (HTTP+MCP) and migrate them to:

     * `ServingTargetHarness.run_targets()` + `ServingTargetHarness.publish_snapshot()`
   * This instantly ensures you are testing the real artifact+pointer layout.
   * Document the lineage/search prerequisites so contributors know when this path
     will fail fast and to fall back to the demo snapshot helper instead.

2. **Serving demo snapshot consolidation**

   * For “demo.view needs deterministic rows” tests, stop using raw `duckdb.connect`:

     * use `GatewayFactory().file_backed(...)` to build the snapshot DB
     * keep `HarnessArtifacts` for registry/manifest/buildspec
   * Prefer `setup_demo_snapshot(...)` where possible; it already builds DB + artifacts + pointer.
   * Replace `_write_pointer` dict dumps with `ServingSnapshotPointer.to_json()` (production class).
   * Keep manual JSON for schema manifests that require view entries until helper changes are in scope.

3. **Use `assertions/http_responses` everywhere you touch FastAPI**

   * This shrinks bespoke asserts and gives consistent failure output.

4. **Delete `_set_env` helpers in serving settings/auth tests**

   * Rely on `codeintel_env` by explicitly requesting the fixture (or use `@pytest.mark.usefixtures`).
   * Use `temporary_env`/`unset_env` for scoped overrides.

5. **Standardize pointer JSON writes on `ServingSnapshotPointer`**

   * Anywhere `current.json` is built by hand, use the production class instead:
     `tests/serving/db/test_manager.py`, `tests/serving/test_http_mcp_integration.py`,
     `tests/serving/mcp/test_sampling.py`, `tests/serving/mcp/test_sql_fingerprint.py`,
     `tests/serving/test_mcp_middleware_rate_limit.py`,
     `tests/serving/test_serving_meta_tooling_mismatch_warnings.py`,
     `tests/serving/mcp/test_middleware_logging_smoke.py`,
     `tests/serving/semantic/test_pr88_polars_execution_path.py`,
     `tests/serving/semantic/test_kernel.py`,
     `tests/serving/semantic/test_compiler_upgrade_gates.py`,
     `tests/serving/semantic/test_pr87_allowed_columns_enforced.py`.

---

## First-pass concrete refactors (integration-like serving tests)

This is the first pass I would actually implement (small, mechanical, and easy to review).
It uses `ServingTargetHarness` wherever the test only needs “a real snapshot exists.”

### Shared skeleton (drop into each target file)

```python
def _publish_serving_snapshot(serving_target_harness: ServingTargetHarness) -> Path:
    records = serving_target_harness.run_targets()
    assert_target_ok(records["serving_artifacts"])
    serving_target_harness.publish_snapshot(run_id="run-1")
    serve_dir = serving_target_harness.harness.ctx.build_paths.build_dir / "serving"
    return serve_dir / "current.json"
```

If `publish_snapshot()` fails due to missing search/lineage tables, stop and revert to
Option B for that specific test (that is a legitimate outcome and should be recorded in
the PR summary).

### `tests/serving/test_http_mcp_integration.py`

**Refactor sketch**

```python
def test_fastapi_app_mounts_mcp(serving_target_harness: ServingTargetHarness) -> None:
    pointer_path = _publish_serving_snapshot(serving_target_harness)
    serve_dir = pointer_path.parent

    settings = ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)
    app = create_serving_app(settings=settings, mount_mcp=True)

    mount_paths = {route.path for route in app.routes if isinstance(route, Mount)}
    expect_in("/mcp", mount_paths)

    with TestClient(app) as client:
        resp = client.get("/v1/semantic/views")
        expect_equal(resp.status_code, status.HTTP_200_OK)
```

**Checklist**

* Remove `_make_db`, `_write_registry`, `_write_schema_manifest`, `_write_buildspec`, `_write_pointer`.
* Drop `duckdb`, `json`, `BuildPaths`, `HarnessArtifacts` imports.
* Add `ServingTargetHarness` parameter and import `assert_target_ok`.
* Create `serve_dir` from the harness build paths and use it in `ServingSettings`.
* Keep assertions focused on mount + response status (no demo row assumptions).

### `tests/serving/test_mcp_middleware_rate_limit.py`

**Refactor sketch**

```python
@pytest.mark.anyio
async def test_mcp_rate_limiting_applies_to_list_tools(
    serving_target_harness: ServingTargetHarness,
) -> None:
    pointer_path = _publish_serving_snapshot(serving_target_harness)
    serve_dir = pointer_path.parent

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=serve_dir,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            mcp_rate_limit_rps=0.001,
            mcp_rate_limit_burst=1,
            mcp_cache_listings=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            await client.list_tools()
            with pytest.raises(McpError):
                await client.list_tools()
    finally:
        await manager.stop()
```

**Checklist**

* Remove bespoke snapshot helpers (`_make_db`, `_write_*`, `_write_pointer`).
* Drop `duckdb`, `json`, `BuildPaths`, `HarnessArtifacts` imports.
* Add `ServingTargetHarness` parameter + `_publish_serving_snapshot` helper.
* Use `pointer_path = serve_dir / "current.json"` from harness output.
* Keep rate-limit assertions unchanged.

### `tests/serving/test_serving_meta_tooling_mismatch_warnings.py`

**Refactor sketch**

```python
@pytest.mark.anyio
async def test_serving_meta_includes_tool_version_mismatch_warning(
    serving_target_harness: ServingTargetHarness,
) -> None:
    try:
        runtime_sqlglot = get_package_version("sqlglot")
    except PackageNotFoundError:
        pytest.skip("sqlglot not installed in runtime environment")

    pointer_path = _publish_serving_snapshot(serving_target_harness)
    serve_dir = pointer_path.parent

    (serve_dir / "environment.json").write_text(
        json.dumps({"tools": {"sqlglot": "0.0.0"}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=serve_dir,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            payload = extract_payload(await client.call_tool("serving_meta", {}))
            warnings = payload.get("warnings")
            expected = f"tool-version-mismatch: sqlglot snapshot=0.0.0 runtime={runtime_sqlglot}"
            if not isinstance(warnings, list):
                pytest.fail("Expected serving_meta.warnings to be a list")
            expect_true(expected in warnings, message="Expected mismatch warning for sqlglot")
    finally:
        await manager.stop()
```

**Checklist**

* Remove `_make_db`, `_write_registry`, `_write_schema_manifest`, `_write_buildspec`, `_write_pointer`.
* Drop `duckdb`, `BuildPaths`, `HarnessArtifacts` imports.
* Use harness-derived `serve_dir` and write `environment.json` there (not `tmp_path`).
* Keep mismatch assertion focused on warnings list content.

---

## Second-pass concrete refactors (demo snapshot MCP tests)

These tests want deterministic demo rows and don’t need the full serving DAG.
Standardize them on `setup_demo_snapshot(...)` and remove the bespoke snapshot
builders. This keeps behavior stable while eliminating repetitive boilerplate.

### Shared skeleton (replace local snapshot builders)

```python
def _setup_demo_snapshot(tmp_path: Path, *, row_count: int = 3) -> Path:
    snapshot = setup_demo_snapshot(tmp_path, row_count=row_count)
    return snapshot.pointer_path
```

If a test uses a custom `serve_dir`, pass
`pointer_path=serve_dir / "current.json"` into `setup_demo_snapshot(...)` and
use `serve_dir = pointer_path.parent` for settings.

### `tests/serving/mcp/test_middleware_logging_smoke.py`

**Refactor sketch**

```python
@pytest.mark.anyio
async def test_mcp_middleware_logging_smoke(tmp_path: Path) -> None:
    pointer_path = _setup_demo_snapshot(tmp_path)

    manager = ServingDBManager(
        pointer_path=pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
    )
    await manager.start()
    try:
        settings = ServingSettings(
            serve_dir=tmp_path,
            hot_swap=False,
            pool_size=1,
            poll_interval_s=0.01,
            mcp_enable_structured_logging=True,
            mcp_cache_listings=False,
        )
        kernel = SemanticQueryKernel(db=manager, settings=settings)
        mcp = build_mcp_app(kernel=kernel, settings=settings)

        async with Client(mcp) as client:
            tools = await client.list_tools()
            expect_true(any(t.name == "semantic_catalog" for t in tools))
            _ = await client.call_tool("semantic_catalog", {})
    finally:
        await manager.stop()
```

**Checklist**

* Remove `_make_db`, `_write_registry`, `_write_schema_manifest`, `_write_buildspec`, `_write_pointer`.
* Drop `duckdb`, `BuildPaths`, `HarnessArtifacts`, `json`, `datetime` imports.
* Add `setup_demo_snapshot` import and `_setup_demo_snapshot` helper.
* Keep `ServingSettings(serve_dir=tmp_path, ...)` unchanged.

### `tests/serving/mcp/test_sampling.py`

**Refactor sketch**

```python
@pytest.mark.anyio
async def test_mcp_sampling_opt_in_adds_summary(tmp_path: Path) -> None:
    pointer_path = _setup_demo_snapshot(tmp_path, row_count=30)
    ...

@pytest.mark.anyio
async def test_mcp_sampling_disabled_does_not_sample(tmp_path: Path) -> None:
    pointer_path = _setup_demo_snapshot(tmp_path, row_count=30)
    ...
```

**Checklist**

* Replace `_setup_test_snapshot(...)` with `_setup_demo_snapshot(...)`.
* Use `row_count=30` to preserve the original “larger dataset” intent.
* Remove `_make_db`, `_write_registry`, `_write_schema_manifest`, `_write_buildspec`, `_write_pointer`.
* Drop `duckdb`, `BuildPaths`, `HarnessArtifacts`, `json`, `datetime` imports if unused.
* Keep the sampling handler logic and assertions unchanged.

### `tests/serving/mcp/test_sql_fingerprint.py`

**Refactor sketch**

```python
@pytest.mark.anyio
async def test_mcp_sql_fingerprint_is_stable_for_same_request(tmp_path: Path) -> None:
    pointer_path = _setup_demo_snapshot(tmp_path)
    ...

@pytest.mark.anyio
async def test_mcp_sql_fingerprint_changes_when_limit_changes(tmp_path: Path) -> None:
    pointer_path = _setup_demo_snapshot(tmp_path)
    ...
```

**Checklist**

* Replace `_setup_test_snapshot(...)` with `_setup_demo_snapshot(...)`.
* Remove bespoke snapshot helpers and related imports (`duckdb`, `BuildPaths`,
  `HarnessArtifacts`, `json`, `datetime`).
* Keep the fingerprint assertions untouched; only the snapshot setup changes.

---

## Third-pass concrete refactors (meta SQL resources)

These are still demo.view-based and can fully reuse `setup_demo_snapshot`.

### `tests/serving/mcp/test_meta_sql_resources.py`

**Refactor sketch**

```python
def _setup_demo_snapshot(tmp_path: Path) -> Path:
    snapshot = setup_demo_snapshot(tmp_path, row_count=1)
    return snapshot.pointer_path

@pytest.mark.anyio
async def test_mcp_meta_views_sql_resources_round_trip(tmp_path: Path) -> None:
    pointer_path = _setup_demo_snapshot(tmp_path)
    ...

@pytest.mark.anyio
async def test_mcp_meta_views_sql_rejects_unsafe_sql(tmp_path: Path) -> None:
    pointer_path = _setup_demo_snapshot(tmp_path)
    ...
```

**Checklist**

* Replace `_setup_test_snapshot(...)` with `_setup_demo_snapshot(...)`.
* Remove `_make_db`, `_write_registry`, `_write_schema_manifest`, `_write_buildspec`, `_write_pointer`.
* Drop `duckdb`, `BuildPaths`, `HarnessArtifacts`, `json`, `datetime` imports if unused.
* Keep the views_sql payload checks untouched.

---

## Fourth-pass concrete refactors (HTTP export tests with custom view)

These tests need a non-demo view (`export.test`) and stable row contents. Use
`GatewayFactory` to build a file-backed DB, `HarnessArtifacts` for the artifacts,
and `ServingSnapshotPointer` for the pointer.

### `tests/serving/http/test_export.py`

**Refactor sketch**

```python
def _setup_serving_env(tmp_path: Path) -> ServingSettings:
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "export_test.duckdb"
    gateway = GatewayFactory().file_backed(db_path).open()
    try:
        gateway.con.execute("CREATE SCHEMA docs")
        gateway.con.execute("CREATE TABLE docs.v_export_test (id INTEGER, name VARCHAR, value DOUBLE)")
        gateway.con.execute(
            """
            INSERT INTO docs.v_export_test VALUES
            (1, 'alpha', 1.1),
            (2, 'beta', 2.2),
            (3, 'gamma', 3.3),
            (4, 'delta', 4.4),
            (5, 'epsilon', 5.5)
            """
        )
    finally:
        gateway.close()

    artifacts = HarnessArtifacts(
        repo_root=tmp_path,
        paths=BuildPaths.from_explicit(build_dir=tmp_path / "build"),
    )
    registry_path = artifacts.write_semantic_registry(
        views=[{... export.test view payload ...}],
        path=tmp_path / "semantic_registry.json",
    )
    manifest_path = artifacts.write_schema_manifest(
        tables=[{... docs.v_export_test schema ...}],
        path=tmp_path / "schema_manifest.json",
    )
    buildspec_path = artifacts.write_buildspec(
        datasets=[{"table_key": "docs.v_export_test", "schema_hash": "schema_export_test"}],
        path=tmp_path / "buildspec.json",
    )

    pointer = ServingSnapshotPointer(
        db_path=db_path,
        semantic_registry_path=registry_path,
        schema_manifest_path=manifest_path,
        buildspec_path=buildspec_path,
        repo="test/export",
        commit="abc123",
        run_id="run-export-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v100",
    )
    (serve_dir / "current.json").write_text(pointer.to_json(), encoding="utf-8")

    return ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)
```

**Checklist**

* Replace `_make_db` with `GatewayFactory().file_backed(...).open()` and close the gateway.
* Keep `HarnessArtifacts.write_*` for registry/manifest/buildspec.
* Replace `_write_pointer` dict with `ServingSnapshotPointer.to_json()`.
* Ensure `serve_dir` (not `tmp_path`) remains the settings root and pointer location.

---

## Fifth-pass concrete refactors (MCP mount contract tests)

These only need an empty snapshot pointer, so use `HarnessArtifacts` and
`ServingSnapshotPointer` for canonical artifacts and JSON.

### `tests/serving/http/test_mcp_mount.py`

**Refactor sketch**

```python
def _write_pointer(tmp_path: Path, *, repo: str, commit: str) -> None:
    db_path = tmp_path / "codeintel.duckdb"
    gateway = GatewayFactory().file_backed(db_path).open()
    gateway.close()

    artifacts = HarnessArtifacts(
        repo_root=tmp_path,
        paths=BuildPaths.from_explicit(build_dir=tmp_path),
    )
    registry_path = artifacts.write_semantic_registry(path=tmp_path / "semantic_registry.json")
    manifest_path = artifacts.write_schema_manifest(path=tmp_path / "schema_manifest.json")
    buildspec_path = artifacts.write_buildspec(path=tmp_path / "buildspec.json")

    pointer = ServingSnapshotPointer(
        db_path=db_path,
        semantic_registry_path=registry_path,
        schema_manifest_path=manifest_path,
        buildspec_path=buildspec_path,
        repo=repo,
        commit=commit,
        run_id="run-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v1",
    )
    (tmp_path / "current.json").write_text(pointer.to_json(), encoding="utf-8")
```

**Checklist**

* Replace direct `json.dumps` artifact payloads with `HarnessArtifacts.write_*`.
* Replace raw `duckdb.connect` with `GatewayFactory().file_backed(...).open()` and close it.
* Replace pointer dict with `ServingSnapshotPointer.to_json()`.

---

## Sixth-pass concrete refactors (semantic kernel + compiler enforcement tests)

These tests need custom registry/manifest payloads (including `views` entries), so
keep manual JSON for those artifacts but standardize DB creation and pointer JSON.

### Shared helper for semantic tests that need custom payloads

```python
def _open_semantic_db(db_path: Path) -> None:
    gateway = GatewayFactory().file_backed(db_path).open()
    try:
        gateway.con.execute("CREATE SCHEMA docs")
        gateway.con.execute("CREATE TABLE docs.demo (id INTEGER, label VARCHAR)")
        gateway.con.execute("INSERT INTO docs.demo VALUES (1, 'one'), (2, 'two')")
        gateway.con.execute("CREATE VIEW docs.v_demo AS SELECT * FROM docs.demo")
    finally:
        gateway.close()
```

Use `_write_json(...)` (already present) for registry/manifest/buildspec, and use
`ServingSnapshotPointer` to create `current.json`.

### `tests/serving/semantic/test_pr87_allowed_columns_enforced.py`

**Checklist**

* Replace `_make_snapshot_db` with `_open_semantic_db` using `GatewayFactory`.
* Keep `_write_json` payloads for registry/manifest/buildspec (they include `views` entries).
* Replace `_write_pointer` with `ServingSnapshotPointer.to_json()`.

### `tests/serving/semantic/test_pr88_polars_execution_path.py`

**Checklist**

* Replace `_make_snapshot_db` with `_open_semantic_db` (or equivalent).
* Keep manual JSON payloads for registry/manifest/buildspec.
* Replace `_write_pointer` with `ServingSnapshotPointer.to_json()`.

### `tests/serving/semantic/test_compiler_upgrade_gates.py`

**Checklist**

* Replace `_make_snapshot_db` with `_open_semantic_db` using `GatewayFactory`.
* Keep `HarnessArtifacts.write_*` for registry/manifest/buildspec.
* Replace `_write_pointer` with `ServingSnapshotPointer.to_json()`.

### `tests/serving/semantic/test_kernel.py`

**Checklist**

* Replace `_make_snapshot_db` with `_open_semantic_db` using `GatewayFactory`.
* Keep `HarnessArtifacts.write_*` for registry/manifest/buildspec.
* Replace `_write_pointer` with `ServingSnapshotPointer.to_json()`.
* Preserve the additional metadata seeding logic in the lineage test (only the snapshot
  bootstrap should change).

---

## Single-sweep execution order (per-file runbook)

Use this runbook when you want to apply the entire migration in one pass without
bouncing between files. The sequence is ordered to minimize dependency churn and
make review easy.

1. `tests/serving/test_http_mcp_integration.py`
   - Convert to `ServingTargetHarness` (integration path).
2. `tests/serving/test_mcp_middleware_rate_limit.py`
   - Convert to `ServingTargetHarness` (integration path).
3. `tests/serving/test_serving_meta_tooling_mismatch_warnings.py`
   - Convert to `ServingTargetHarness` and write `environment.json` under `serve_dir`.
4. `tests/serving/mcp/test_middleware_logging_smoke.py`
   - Convert to `setup_demo_snapshot`.
5. `tests/serving/mcp/test_sampling.py`
   - Convert to `setup_demo_snapshot(row_count=30)`.
6. `tests/serving/mcp/test_sql_fingerprint.py`
   - Convert to `setup_demo_snapshot`.
7. `tests/serving/mcp/test_meta_sql_resources.py`
   - Convert to `setup_demo_snapshot(row_count=1)` and keep views_sql payload checks.
8. `tests/serving/http/test_export.py`
   - Convert DB creation to `GatewayFactory().file_backed(...)` and pointer to
     `ServingSnapshotPointer`.
9. `tests/serving/http/test_mcp_mount.py`
   - Convert artifacts to `HarnessArtifacts.write_*` and pointer to
     `ServingSnapshotPointer`.
10. `tests/serving/semantic/test_pr87_allowed_columns_enforced.py`
    - Convert DB creation to `GatewayFactory().file_backed(...)`, keep manual JSON,
      update pointer to `ServingSnapshotPointer`.
11. `tests/serving/semantic/test_pr88_polars_execution_path.py`
    - Convert DB creation to `GatewayFactory().file_backed(...)`, keep manual JSON,
      update pointer to `ServingSnapshotPointer`.
12. `tests/serving/semantic/test_compiler_upgrade_gates.py`
    - Convert DB creation to `GatewayFactory().file_backed(...)`, keep `HarnessArtifacts`,
      update pointer to `ServingSnapshotPointer`.
13. `tests/serving/semantic/test_kernel.py`
    - Convert DB creation to `GatewayFactory().file_backed(...)`, keep `HarnessArtifacts`,
      update pointer to `ServingSnapshotPointer`.
14. `tests/serving/db/test_manager.py`
    - Pointer JSON: replace dict writes with `ServingSnapshotPointer.to_json()`.

Notes:
* If any integration test fails due to missing search/lineage tables, swap that one
  back to the demo snapshot path and note it explicitly in the PR summary.
* Keep manual JSON for schema manifests that require `views` entries until helper
  changes are in scope.

---

## Where this leaves you before creating new helpers

After doing the above, the “remaining duplication” will be much clearer and will likely cluster into just two places:

* A tiny shared fixture/helper for **“demo serving snapshot (db+artifacts+pointer)”** reused across 8–12 serving tests.
* A multi-env helper for non-`CODEINTEL_` env var swapping (parallel/hook tests).

But the big point is: you can get a major realism boost **right now** by simply deploying:

* `ServingTargetHarness` where you’re currently faking snapshot layouts, and/or
* `GatewayFactory` in serving tests where you’re using raw DuckDB connections.
