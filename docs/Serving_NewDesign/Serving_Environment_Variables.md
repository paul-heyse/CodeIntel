# Serving Environment Variables

This document describes environment variables consumed by the semantic-first serving stack
(`codeintel.serving.settings.ServingSettings`).

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEINTEL_SERVE_DIR` | `.codeintel/serve` | Serving snapshot directory |
| `CODEINTEL_SERVE_HOTSWAP` | `1` | Enable hot-swap (0/1) |
| `CODEINTEL_SERVE_POOL_SIZE` | `4` | DuckDB connections per worker |
| `CODEINTEL_SERVE_POLL_INTERVAL` | `1.0` | Seconds between pointer checks |
| `CODEINTEL_MCP_TRANSPORT` | `stdio` | MCP transport (`stdio`/`http`) |
| `CODEINTEL_HOST` | `127.0.0.1` | HTTP bind address |
| `CODEINTEL_PORT` | `8000` | HTTP port |
| `CODEINTEL_AUTH_TOKEN` | (unset) | Optional bearer token |

