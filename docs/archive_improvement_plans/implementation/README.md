# Unified Data Abstraction Implementation Guide

> **Version**: 1.0.0
> **Status**: Implementation Ready
> **Total Duration**: 10-16 days

---

## Overview

This directory contains detailed implementation plans for the unified data abstraction architecture. The implementation is divided into 5 phases, each building on the previous, with clear deliverables and verification steps.

---

## Quick Reference

| Phase | Document | Duration | Risk | Deliverables |
|-------|----------|----------|------|--------------|
| 1 | [phase1-core-infrastructure.md](./phase1-core-infrastructure.md) | 1-2 days | Low | `CorePluginMetadata`, `ConfigSource`, `PluginOptionsResolver` |
| 2 | [phase2-spine-plugin-migration.md](./phase2-spine-plugin-migration.md) | 2-3 days | Low-Medium | Spine plugins with metadata + options |
| 3 | [phase3-full-rollout.md](./phase3-full-rollout.md) | 3-5 days | Medium | All plugins migrated, global registry |
| 4 | [phase4-profile-integration.md](./phase4-profile-integration.md) | 2-3 days | Medium | Profiles (fast/full/ci), CLI integration |
| 5 | [phase5-skip-manifest-integration.md](./phase5-skip-manifest-integration.md) | 2-3 days | Medium-High | Skip logic, manifest store, execution records |

---

## Architecture Document

The comprehensive architecture specification is available at:

- **[../unified-data-abstraction-architecture.md](../unified-data-abstraction-architecture.md)**

This document describes:
- End-state architecture design
- Core data models
- Integration patterns
- Module layout

---

## Phase Dependencies

```
Phase 1 (Core Infrastructure)
    │
    ▼
Phase 2 (Spine Plugin Migration)
    │
    ▼
Phase 3 (Full Rollout)
    │
    ▼
Phase 4 (Profile Integration)
    │
    ▼
Phase 5 (Skip/Manifest Integration)
```

Each phase must be completed before starting the next.

---

## Getting Started

### Prerequisites

```bash
# Bootstrap environment
scripts/bootstrap.sh

# Run quality checks to establish baseline
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest tests/core/plugins/ -q
```

### Starting Phase 1

1. Read [phase1-core-infrastructure.md](./phase1-core-infrastructure.md)
2. Create `CorePluginMetadata` type
3. Create options infrastructure
4. Create capability registry index
5. Run verification steps

---

## Key Patterns

### Metadata Declaration

```python
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

PLUGIN_METADATA = CorePluginMetadata(
    name="domain.plugin_name",
    version="X.Y.Z",
    description="Description.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=("capability.a",),
    requires=("capability.b",),
    produces_tables=("table.a",),
    options_model=PluginOptions,
)
```

### Options Resolution

```python
from codeintel.core.plugins.execution.options import PluginOptionsResolver

resolver = PluginOptionsResolver(config_source)
opts = resolver.get_options(metadata, OptionsModel)
```

### Skip Decision

```python
from codeintel.core.plugins.execution.skip import should_skip_plugin

decision = should_skip_plugin(
    run_context=ctx,
    manifest_store=store,
    repo=repo,
    commit=commit,
    scope_id=scope_id,
    variant=variant,
)

if decision.should_skip:
    # Reuse prior result
    ...
else:
    # Execute plugin
    ...
```

---

## Verification Commands

### Quality Checks

```bash
# Format and lint
uv run ruff format src/codeintel/core/plugins/
uv run ruff check --fix src/codeintel/core/plugins/

# Type checking
uv run pyright src/codeintel/core/plugins/
uv run pyrefly check src/codeintel/core/plugins/
```

### Test Suite

```bash
# Core plugin tests
uv run pytest tests/core/plugins/ -v

# Domain plugin tests
uv run pytest tests/analytics/plugins/ tests/graphs/plugins/ tests/ingestion/plugins/ -v

# Integration tests
uv run pytest tests/integration/ -v -k "metadata or options or skip"
```

---

## Rollback Strategy

Each phase includes a rollback plan. General strategy:

1. **Revert source files** to their pre-phase state
2. **Delete new files** created in the phase
3. **Revert `__init__.py` exports**
4. **Delete test files** for the phase
5. **Re-run existing tests** to verify no regression

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `CorePluginMetadata` type exists and is tested
- [ ] `ConfigSource` and `PluginOptionsResolver` work
- [ ] `PluginRegistryIndex` can build capability index
- [ ] All existing tests pass

### Phase 2 Complete When:
- [ ] Spine plugins (function_metrics, callgraph, scip_ingest) have metadata
- [ ] Spine plugins use `PluginOptionsResolver`
- [ ] `PluginRunContext` can prepare execution
- [ ] All plugin tests pass

### Phase 3 Complete When:
- [ ] All plugins have `CorePluginMetadata`
- [ ] `ALL_PLUGIN_METADATA` contains all plugins
- [ ] Global registry index works
- [ ] No duplicate plugin names

### Phase 4 Complete When:
- [ ] Builtin profiles (fast, full, ci) are defined
- [ ] Profile files can be loaded
- [ ] `BuildRunConfig` integrates profiles
- [ ] CLI `--profile` flag works

### Phase 5 Complete When:
- [ ] `DuckDBManifestStore` persists records
- [ ] Skip decisions work correctly
- [ ] Executor checks skip before execution
- [ ] CLI `--force` and `--dry-run` work

---

## Support

For questions or issues:

1. Review the architecture document
2. Check the phase-specific implementation plan
3. Review test files for usage examples
4. Consult AGENTS.md for coding standards

---

**Happy Implementing!** 🚀
