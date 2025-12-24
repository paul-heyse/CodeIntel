Below is a “deploy what you already have” review of the **Phase4 zip** test suite, focused on places where tests still do bespoke setup that your existing helpers in `tests/_helpers` (and a couple of already-present fixtures in `tests/conftest.py`) can replace to make tests both **more realistic** and **less repetitive**.

I’m going to be concrete: **file paths → what to replace → which helper to use → what realism you gain**.

---

## Where you’re already leveraging helpers well (so we don’t waste effort)

You already have strong “production-aligned” patterns in several areas:

* **Ingestion**: e.g. `tests/ingestion/test_module_inventory.py` uses `module_inventory_context`, `modules_expected_from_repo_tree`, and the missing/extra formatter. Good realism + low bespoke code.
* **Hamilton target plumbing**: e.g. `tests/ingestion/test_runner_plumbing.py` uses `HamiltonBuildHarness`, `HarnessArtifacts`, and `ManifestPriming`.
* **Tool realism**: e.g. `tests/ingestion/test_scip_ingest.py` uses `ToolSandbox` + `HarnessArtifacts.write_dummy_scip_artifacts()` instead of mocking.
* **Graph tests**: often use `GraphTargetHarness` and networkx fakes.

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

* `tests/serving/test_semantic_http_routes.py`
* `tests/serving/test_http_mcp_integration.py`
* `tests/serving/test_mcp_middleware_rate_limit.py`
* `tests/serving/test_serving_meta_tooling_mismatch_warnings.py`
* `tests/serving/test_semantic_mcp_tools.py`
* `tests/serving/http/test_export.py`
* `tests/serving/mcp/test_resources.py`
* `tests/serving/mcp/test_tasks.py`
* `tests/serving/mcp/test_prompts_advanced.py`
* `tests/serving/mcp/test_sampling.py`
* `tests/serving/mcp/test_meta_sql_resources.py`
* `tests/serving/mcp/test_sql_fingerprint.py`
* `tests/serving/mcp/test_middleware_logging_smoke.py`
* plus several `tests/serving/semantic/*` tests that do similar snapshot setup

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
  * many MCP tool routing tests

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

* `tests/_helpers/gateway.py::GatewayFactory` (via fixture `gateway_factory`)
* `tests/_helpers/hamilton_harness_artifacts.py::HarnessArtifacts`
* `tests/_helpers/docs_views.py::create_bootstrapped_docs_db` (optional alternative to GatewayFactory)
* Production class (not in tests/_helpers, but important realism): `codeintel.serving.db.pointer.ServingSnapshotPointer`

What to change:

1. Replace every `duckdb.connect()` snapshot builder with **GatewayFactory file-backed bootstrap**, so the DB has the full schema/view bootstrapping that prod has.

2. Replace manual JSON writing (where it exists) with `HarnessArtifacts.write_*`.

3. Replace `_write_pointer()` dict dumps with `ServingSnapshotPointer(...).to_json()` or just use `publish_serving_snapshot` via `ServingTargetHarness` if you’re ready.

**Where this is the best immediate fit**

* Tests that intentionally need a tiny schema + deterministic rows (your “demo view” tests):

  * `tests/serving/test_semantic_http_routes.py`
  * `tests/serving/test_semantic_mcp_tools.py`
  * `tests/serving/mcp/test_prompts_advanced.py` (wizard prompt tests are much easier with a tiny view)
  * `tests/serving/http/test_export.py` (if you want stable exported row content)

**A very direct “deploy existing helpers” improvement**
In those files, you can keep your `docs.v_demo` data, but make `_make_db()` do:

* `gw = gateway_factory.file_backed(db_path)`
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

* Replace `expect_equal(resp.status_code, 200)` with `assert_http_success(resp)`
* Replace manual error-body asserts with `assert_problem_detail_response(...)`
* Replace your current “shape checks” with:

  * `assert_response_has_keys(...)`
  * `assert_paginated_response(...)`

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

So: this file is a quick win: swap the manual JSON writes to `HarnessArtifacts.write_*`.

---

## A lot of serving tests are duplicating env-var context managers, even though you already have cleanup

Files with repeated `_set_env(...)`:

* `tests/serving/test_settings.py`
* `tests/serving/test_auth_enforcement.py`
* `tests/serving/test_mcp_feature_flags.py`
* `tests/serving/test_mcp_metrics.py`
* `tests/serving/test_uvicorn_config.py`

Key thing you already have:

* `tests/conftest.py` defines an **autouse** fixture `codeintel_env` that snapshots/restores all `CODEINTEL_` env vars per test.

That means: in these tests you can usually delete `_set_env(...)` entirely and just:

```python
import os

def test_something(tmp_path):
    os.environ["CODEINTEL_SERVE_DIR"] = str(tmp_path)
    os.environ["CODEINTEL_MCP_ENABLE_SEARCH"] = "0"
    settings = get_serving_settings()
    ...
```

The autouse fixture will restore environment after the test.

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

## Small but important audit note: `tests/_helpers/__init__.py` exports missing modules

Your `tests/_helpers/__init__.py` currently imports:

```py
from tests._helpers.factories.http_clients import build_test_client, build_tool_http_client
```

…but there is **no** `tests/_helpers/factories/http_clients.py` in the zip.

Practically:

* Don’t try to “centralize imports” via `from tests._helpers import ...` until that’s corrected.
* It also means tests are currently *forced* to import submodules directly, which can make helper adoption feel more scattered than it needs to be.

Not required for the refactors above, but worth fixing because it affects “deploy helpers broadly” ergonomics.

---

## Suggested “deployment plan” using only what you already have

This is ordered to maximize realism and reduce bespoke code quickly:

1. **Serving snapshot consolidation**

   * Pick 2–3 serving tests that are clearly “integration-like” (HTTP+MCP) and migrate them to:

     * `ServingTargetHarness.run_targets()` + `ServingTargetHarness.publish_snapshot()`
   * This instantly ensures you are testing the real artifact+pointer layout.

2. **Serving demo snapshot consolidation**

   * For “demo.view needs deterministic rows” tests, stop using raw `duckdb.connect`:

     * use `gateway_factory.file_backed(...)` to build the snapshot DB
     * keep `HarnessArtifacts` for registry/manifest/buildspec
   * Optionally: replace `_write_pointer` dict dumps with `ServingSnapshotPointer.to_json()` (production class).

3. **Use `assertions/http_responses` everywhere you touch FastAPI**

   * This shrinks bespoke asserts and gives consistent failure output.

4. **Delete `_set_env` helpers in serving settings/auth tests**

   * Rely on the autouse `codeintel_env` fixture for cleanup.

---

## Where this leaves you before creating new helpers

After doing the above, the “remaining duplication” will be much clearer and will likely cluster into just two places:

* A tiny shared fixture/helper for **“demo serving snapshot (db+artifacts+pointer)”** reused across 8–12 serving tests.
* A multi-env helper for non-`CODEINTEL_` env var swapping (parallel/hook tests).

But the big point is: you can get a major realism boost **right now** by simply deploying:

* `ServingTargetHarness` where you’re currently faking snapshot layouts, and/or
* `GatewayFactory` in serving tests where you’re using raw DuckDB connections.


