## 1. Implementation
- [ ] 1.1 Add an idempotent schema seeding helper in tests/_helpers that uses the production
      schema provider (DuckDBPolicyBackend) to ensure schemas exist.
- [ ] 1.2 Replace ad-hoc docs schema creation in serving tests with the shared helper.
- [ ] 1.3 Update HamiltonBuildHarness to raise diagnostic errors when target records are missing
      (include build error, failed targets, missing targets).
- [ ] 1.4 Update ServingTargetHarness and related tests to use the error-aware record access.
- [ ] 1.5 Add/adjust tests that validate idempotent schema seeding and harness error reporting.
