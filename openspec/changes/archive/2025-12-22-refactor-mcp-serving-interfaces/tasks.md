## 1. Implementation
- [x] 1.1 Add readiness signaling + cached snapshot summary in ServingDBManager and update MCP
      health/ready routes to await readiness and return cached metadata.
- [x] 1.2 Introduce MCP request envelopes for semantic_query/semantic_explain/semantic_export and
      replace tool signatures with a single request object plus shared validation helper.
- [x] 1.3 Refactor MCP tool logic into use-case handlers/workflows with explicit dependencies and
      shared metrics/logging utilities.
- [x] 1.4 Add a public prompt registry/introspection API in codeintel.serving.mcp.prompts and
      update prompt tests to use it.
- [x] 1.5 Add tests/_helpers/mcp_payloads.py with Protocol/TypeGuard parsing and update MCP tests
      to use the helper.
- [x] 1.6 Add tests/_helpers/security_fixtures.py (public bind host + token generator) and update
      auth enforcement tests to use it.
- [x] 1.7 Update MCP tool docs/examples (if any) to reflect the request-envelope inputs.
- [x] 1.8 Run quality report + pytest.
