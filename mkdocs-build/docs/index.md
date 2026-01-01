# CodeIntel

Code intelligence and analytics tooling for Python repositories.

## Overview

CodeIntel is a comprehensive code intelligence system that provides deep analysis
of Python codebases through ingestion, analytics, and graph-based insights.

## Key Subsystems

| Subsystem | Description |
|-----------|-------------|
| [`codeintel.ingestion`][codeintel.ingestion] | Build snapshots and datasets from source repositories |
| [`codeintel.build.analytics`][codeintel.build.analytics] | Analytics runtime, metrics, profiles, and risk assessment |
| [`codeintel.build.graphs`][codeintel.build.graphs] | Graph engines (call graph, import graph, CFG/DFG) |
| [`codeintel.storage`][codeintel.storage] | DuckDB persistence, Parquet datasets, and contracts |
| [`codeintel.serving`][codeintel.serving] | Backend services, HTTP APIs, and MCP integration |
| [`codeintel.pipeline`][codeintel.pipeline] | Unified orchestration for ingestion, graphs, and analytics |
| [`codeintel.runtime`][codeintel.runtime] | Runtime context and environment management |
| [`codeintel.core`][codeintel.core] | Core types and abstractions |
| [`codeintel.config`][codeintel.config] | Configuration models, primitives, and step configs |

## Quick Links

- [Architecture Overview](architecture/overview.md) - High-level system architecture
- [Code Reference](reference/) - Auto-generated API documentation

## Detailed Documentation

For in-depth architectural documentation, see the detailed guides in the
repository's `docs/` folder:

- **Analytics Architecture**: Comprehensive plugin-based analytics engine design
- **Ingestion Architecture**: Snapshot building and dataset extraction
- **Graphs Architecture**: Graph plugin system and runtime

## Getting Started

```bash
# Bootstrap the development environment
scripts/bootstrap.sh

# Run the full pipeline
codeintel pipeline run --mode full

# Start the API server
codeintel serve
```

