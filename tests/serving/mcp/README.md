Shared MCP test fixtures
========================

- Reuse `mcp_backend_factory` (from `tests.serving.mcp.conftest`) to build a gateway-bound trio of `DuckDBQueryService`, `LocalQueryService`, and `DuckDBBackend`. This is the canonical way to stand up serving components in tests; avoid ad-hoc `_build_*` helpers.
- `mcp_backend_components` provides a ready-made snapshot for the default provisioned gateway. Prefer it for most tool/back-end tests; reach for `mcp_backend_factory` only when you must vary gateway, repo/commit, or limits.
- When HTTP or service-layer tests need a backend/service, delegate to `mcp_backend_factory` inside fixtures (e.g., `make_http_app`) so wiring stays consistent with MCP tool tests.
- Keep new fixtures and helpers colocated here if they construct MCP backends or services, and document parameters clearly. If you need a new variant, prefer adding a small, typed factory wrapper rather than repeating inline setup.
