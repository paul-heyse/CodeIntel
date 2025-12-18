# FastMCP Serving Runbook (Single Worker / Sessionful)

This runbook documents how to run and validate CodeIntel’s FastMCP-based serving surface.

## Supported Deployment Modes

| Mode | Transport | Sessionful | `uvicorn_workers` | Elicitation (`ctx.elicit`) | Sampling (`ctx.sample`) |
|---|---|---:|---:|---:|---:|
| Local agent | stdio | Yes | N/A | Yes (client-dependent) | Yes (client-dependent) |
| HTTP (recommended) | streamable-http | Yes | **1 only** | Yes (client-dependent) | Yes (client-dependent) |

Notes:
- When MCP is mounted under FastAPI, **`uvicorn_workers` must be `1`** to preserve sessionful behavior.
- Multi-worker is intentionally out of scope for the current design basis.

## Local Development (stdio)

Run an MCP server over stdio for local agents/inspectors:

```bash
uv run python -m codeintel.serving.mcp.server --transport=stdio
```

## Local Development (HTTP / Streamable HTTP)

Run MCP over HTTP:

```bash
uv run python -m codeintel.serving.mcp.server --transport=http --host=127.0.0.1 --port=8000
```

If you mount MCP under the serving HTTP app, ensure `uvicorn_workers=1` (enforced at startup).

## Using Exports Safely (Chunked Resources)

MCP resources are not streaming; large exports should be fetched in chunks.

Typical flow:
1. Call `semantic_export(...)` to obtain an `export_id` and `meta_uri`.
2. Read `meta_uri` (`codeintel://exports/{export_id}/meta`) to discover safe retrieval URIs.
3. Fetch in chunks:
   - Text exports: `codeintel://exports/{export_id}/lines{?offset,limit}`
   - Binary exports: `codeintel://exports/{export_id}/bytes{?offset,limit}`

## Meta Resources (Snapshot Artifacts)

The server exposes snapshot-scoped meta resources to support agent workflows:
- `codeintel://meta/environment`
- `codeintel://meta/views_sql` (validated select-only)
- `codeintel://meta/views_sql_diff` (optional; if present in snapshot)

## Manifest Inspection

Generate an MCP manifest for debugging client-visible changes:

```bash
uv run fastmcp inspect src/codeintel/serving/mcp/server.py:create_mcp_server -o build/mcp-manifest.json
```

## Validation

Targeted serving checks:

```bash
uv run ruff check --fix src/codeintel/serving tests/serving
uv run pyright --warnings --pythonversion=3.13 src/codeintel/serving tests/serving
uv run pyrefly check src/codeintel/serving tests/serving
uv run pytest -q tests/serving
```

