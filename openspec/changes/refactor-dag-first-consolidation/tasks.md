## 1. Implementation
- [ ] 1.1 Remove non-DAG analytics/graph/history orchestration entrypoints and keep Hamilton
      targets/materializers as the only execution path; update CLI/debug commands to read
      DAG-derived datasets or cached DAG artifacts.
- [ ] 1.2 Introduce a shared, contract-backed writer for analytics persistence and delete
      ad-hoc Pandera validation/persistence helpers that bypass contracts.
- [ ] 1.3 Route all Ibis usage in non-storage modules through the storage Ibis gateway and
      remove direct ibis.duckdb.from_connection calls from analytics/history.
- [ ] 1.4 Consolidate runtime config loading for build/CLI/serving through the canonical
      runtime loader and remove bespoke env/path parsing modules, including observability
      and metrics auth gating.
- [ ] 1.5 Consolidate ID normalization utilities to codeintel.core.data_models.ids and
      remove duplicate conversion helpers.
- [ ] 1.6 Update docs and tests to reflect DAG-first analytics execution and canonical
      runtime loading.
- [ ] 1.7 Run the quality report and targeted test suites for analytics and CLI surfaces.
