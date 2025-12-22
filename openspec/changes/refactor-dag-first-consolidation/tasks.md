## 1. Implementation
- [ ] 1.1 Remove non-DAG analytics/graph/history orchestration entrypoints and keep Hamilton
      targets/materializers as the only execution path; update CLI/debug commands to read
      DAG-derived datasets or cached DAG artifacts.
      Status: Remaining. Non-DAG orchestration functions still exist and are invoked by
      Hamilton targets (e.g., graph metrics, symbol metrics, subsystem metrics, test coverage,
      semantic roles, function effects/contracts, and test profiles). Public APIs still
      export non-DAG entrypoints.
      Final design: Replace target implementations to consume row-producing functions only
      and materialize via Hamilton DataSavers; remove compute_* orchestration functions and
      public exports; CLI/debug paths should read DAG-produced tables or cached artifacts.
- [ ] 1.2 Introduce a shared, contract-backed writer for analytics persistence and delete
      ad-hoc Pandera validation/persistence helpers that bypass contracts.
      Status: Mostly complete. Shared writer exists and most analytics persistence paths
      now use it; remaining direct SQL cache refreshes in subsystem cache materialization
      still bypass the contract writer.
      Final design: Convert subsystem cache refresh to contract-backed writes or move
      cache materialization into Hamilton targets using the shared writer.
- [x] 1.3 Route all Ibis usage in non-storage modules through the storage Ibis gateway and
      remove direct ibis.duckdb.from_connection calls from analytics/history.
      Status: Complete. Analytics/history usage routes through the storage gateway/facade.
- [ ] 1.4 Consolidate runtime config loading for build/CLI/serving through the canonical
      runtime loader and remove bespoke env/path parsing modules, including observability
      and metrics auth gating.
      Status: Partially complete. Serving and CLI observability now use the runtime loader,
      but CLI config env parsing and serving_factory still read env directly.
      Final design: Replace serving_factory env parsing with runtime loader settings and
      collapse CLI env parsing into the canonical runtime loader.
- [x] 1.5 Consolidate ID normalization utilities to codeintel.core.data_models.ids and
      remove duplicate conversion helpers.
      Status: Complete. Duplicate ID conversion helpers removed in analytics graphs.
- [ ] 1.6 Update docs and tests to reflect DAG-first analytics execution and canonical
      runtime loading.
      Status: Remaining. Docs/specs/tests still reference non-DAG orchestration paths.
- [ ] 1.7 Run the quality report and targeted test suites for analytics and CLI surfaces.
      Status: Remaining. Quality report not run; Ruff still flags TRY300 in
      src/codeintel/serving/semantic/kernel.py.
