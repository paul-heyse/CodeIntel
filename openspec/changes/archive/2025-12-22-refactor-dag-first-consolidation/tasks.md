## 1. Implementation
- [x] 1.1 Remove non-DAG analytics/graph/history orchestration entrypoints and keep Hamilton
      targets/materializers as the only execution path; update CLI/debug commands to read
      DAG-derived datasets or cached DAG artifacts.
      Status: Complete. DAG targets consume row builders only, and public exports no longer
      surface non-DAG orchestration entrypoints.
- [x] 1.2 Introduce a shared, contract-backed writer for analytics persistence and delete
      ad-hoc Pandera validation/persistence helpers that bypass contracts.
      Status: Complete. Backfill paths route through DuckDBPolicyBackend and subsystem cache
      tables are materialized via Hamilton targets.
- [x] 1.3 Route all Ibis usage in non-storage modules through the storage Ibis gateway and
      remove direct ibis.duckdb.from_connection calls from analytics/history.
      Status: Complete. Analytics/history usage routes through the storage gateway/facade.
- [x] 1.4 Consolidate runtime config loading for build/CLI/serving through the canonical
      runtime loader and remove bespoke env/path parsing modules, including observability
      and metrics auth gating.
      Status: Complete. CLI/build/serving load settings via the runtime loader.
- [x] 1.5 Consolidate ID normalization utilities to codeintel.core.data_models.ids and
      remove duplicate conversion helpers.
      Status: Complete. Duplicate ID conversion helpers removed in analytics graphs.
- [x] 1.6 Update docs and tests to reflect DAG-first analytics execution and canonical
      runtime loading.
      Status: Complete. Tests now rely on DAG targets/row builders and public exports are
      DAG-first.
- [x] 1.7 Run the quality report and targeted test suites for analytics and CLI surfaces.
      Status: Complete. Quality report and pytest -q are green.
