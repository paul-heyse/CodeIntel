# Serving

The serving module (`codeintel.serving`) provides HTTP APIs and MCP server
integration for external access to CodeIntel data.

## Responsibility

- Expose REST APIs for querying analytics data
- Provide MCP (Model Context Protocol) server for AI assistants
- Handle authentication and request routing
- Serve documentation and schema information

## Architecture

```
┌─────────────────────────────────────────┐
│           HTTP Layer                     │
│    (http/routers/*.py, FastAPI)          │
├─────────────────────────────────────────┤
│           MCP Layer                      │
│         (mcp/*.py)                       │
├─────────────────────────────────────────┤
│         Services Layer                   │
│       (services/*.py)                    │
├─────────────────────────────────────────┤
│         Backend Layer                    │
│        (backend/*.py)                    │
├─────────────────────────────────────────┤
│        Storage Gateway                   │
└─────────────────────────────────────────┘
```

## Key Entrypoints

- [`codeintel.serving.http`][codeintel.serving.http] - FastAPI application
- [`codeintel.serving.mcp`][codeintel.serving.mcp] - MCP server implementation
- [`codeintel.serving.bootstrap`][codeintel.serving.bootstrap] - Application bootstrap

## HTTP API

### Endpoints

| Path | Description |
|------|-------------|
| `/api/v1/functions` | Function metadata and metrics |
| `/api/v1/modules` | Module information |
| `/api/v1/graphs` | Graph queries |
| `/api/v1/analytics` | Analytics results |

### Starting the Server

```bash
codeintel serve --host 0.0.0.0 --port 8080
```

## MCP Server

The MCP server exposes CodeIntel as tools for AI assistants:

```bash
codeintel mcp serve
```

### Available Tools

- `search_functions` - Find functions by name or pattern
- `get_function_metrics` - Retrieve metrics for a function
- `query_call_graph` - Explore call relationships

## Dependencies

### Reads From

- [`codeintel.storage`][codeintel.storage] via gateway
- Analytics and graph results

### Writes To

- HTTP responses
- MCP tool responses

### Called By

- External HTTP clients
- AI assistants via MCP
- CLI commands

## Extension Points

### Adding an API Endpoint

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint():
    return {"data": "..."}
```

### Adding an MCP Tool

```python
from codeintel.serving.mcp.tools import register_tool

@register_tool
def my_tool(query: str) -> str:
    """My custom tool description."""
    return "result"
```

