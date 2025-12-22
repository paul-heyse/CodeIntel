## 1. Implementation
- [ ] 1.1 Remove non-DAG analytics/graph/history orchestration entrypoints and keep Hamilton
      targets/materializers as the only execution path; update CLI/debug commands to read
      DAG-derived datasets or cached DAG artifacts.
      Status: Partially complete. Most analytics targets are DAG-only, but remaining targets
      still call orchestration helpers instead of DAG-native compute + materialize nodes
      (graph metrics/ext/stats, symbol/subsystem graph metrics, semantic_roles, test_profile,
      function_effects/contracts, and profiles). Public APIs still re-export non-DAG
      entrypoints in analytics and ingestion packages.
      Remaining scope detail:
      - Replace target implementations to consume row-producing functions only and
        materialize via Hamilton DataSavers (no orchestration in compute nodes).
      - Remove or internalize non-DAG orchestration functions and public exports that
        surface them (analytics/history/semantic_roles/testing/profiles/subsystems,
        ingestion engine/ports/adapters).
      - Ensure CLI/debug flows read DAG-produced tables or cached artifacts only.
- [ ] 1.2 Introduce a shared, contract-backed writer for analytics persistence and delete
      ad-hoc Pandera validation/persistence helpers that bypass contracts.
      Status: Mostly complete. Shared writer exists and most analytics persistence paths
      now use it; remaining direct SQL writes still bypass the contract writer (notably
      test_catalog backfill during test_coverage_edges, and subsystem cache tables).
      Remaining scope detail:
      - Route test_catalog GOID/URN backfill through DuckDBPolicyBackend (or a dedicated
        Hamilton target) instead of direct executemany.
      - Materialize subsystem cache tables via Hamilton targets or a shared writer path.
- [x] 1.3 Route all Ibis usage in non-storage modules through the storage Ibis gateway and
      remove direct ibis.duckdb.from_connection calls from analytics/history.
      Status: Complete. Analytics/history usage routes through the storage gateway/facade.
- [ ] 1.4 Consolidate runtime config loading for build/CLI/serving through the canonical
      runtime loader and remove bespoke env/path parsing modules, including observability
      and metrics auth gating.
      Status: Partially complete. Serving + CLI observability are loader-driven, but CLI
      config path resolution still reads env directly and parallel adapter defaults still
      use os.getenv.
      Remaining scope detail:
      - Route CLI config path/env overrides through runtime settings (or a dedicated
        loader-owned settings object).
      - Replace ParallelConfig.from_env usage with runtime loader settings.
      - Keep metrics auth gating exclusively loader-driven.
- [x] 1.5 Consolidate ID normalization utilities to codeintel.core.data_models.ids and
      remove duplicate conversion helpers.
      Status: Complete. Duplicate ID conversion helpers removed in analytics graphs.
- [ ] 1.6 Update docs and tests to reflect DAG-first analytics execution and canonical
      runtime loading.
      Status: Remaining. Docs/specs/tests still reference non-DAG orchestration paths and
      legacy entrypoint exports.
      Remaining scope detail:
      - Replace doc references to compute_* / build_* orchestration helpers with DAG targets.
      - Update tests to execute DAG targets and assert on DAG-produced tables/artifacts.
- [ ] 1.7 Run the quality report and targeted test suites for analytics and CLI surfaces.
      Status: Remaining. Quality report not run; Ruff still flags TRY300 in
      src/codeintel/serving/semantic/kernel.py.
