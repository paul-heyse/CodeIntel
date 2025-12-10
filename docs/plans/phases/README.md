# CLI Migration Implementation Phases

> **Parent Document:** [CLI_MIGRATION_PLAN.md](../CLI_MIGRATION_PLAN.md)  
> **Architecture Reference:** [CLI_UNIFIED_ARCHITECTURE.md](../CLI_UNIFIED_ARCHITECTURE.md)  
> **Total Duration:** 4-6 weeks (20-29 working days)

---

## Overview

This directory contains detailed implementation plans for each phase of the CLI architecture migration. Each phase document provides:

- Objectives and deliverables
- Step-by-step task breakdowns with effort estimates
- Code examples and patterns
- Testing requirements
- Verification checklists
- Exit criteria
- Rollback procedures

---

## Phase Documents

| Phase | Name | Duration | Risk | Document |
|-------|------|----------|------|----------|
| **0** | Preparation | 1-2 days | Low | [PHASE_0_PREPARATION.md](./PHASE_0_PREPARATION.md) |
| **1** | Foundation Layer | 3-4 days | Low | [PHASE_1_FOUNDATION.md](./PHASE_1_FOUNDATION.md) |
| **2** | Rendering Consolidation | 2-3 days | Medium | [PHASE_2_RENDERING.md](./PHASE_2_RENDERING.md) |
| **3** | Handler Migration | 5-7 days | Medium | [PHASE_3_HANDLERS.md](./PHASE_3_HANDLERS.md) |
| **4** | Registry Unification | 2-3 days | Low | [PHASE_4_REGISTRY.md](./PHASE_4_REGISTRY.md) |
| **5** | Command Decorator | 5-7 days | Medium | [PHASE_5_DECORATOR.md](./PHASE_5_DECORATOR.md) |
| **6** | Legacy Cleanup | 2-3 days | Low | [PHASE_6_CLEANUP.md](./PHASE_6_CLEANUP.md) |

---

## Phase Dependencies

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──┬──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6
   │           │           │      │
   │           │           │      └── (Can overlap with early Phase 3)
   │           │           │
   │           │           └── Rendering consolidated
   │           │
   │           └── HandlerContext + bootstrap_cli created
   │
   └── Baselines captured, inventories complete
```

---

## Quick Start

### Starting the Migration

1. **Read the architecture document:**
   - [CLI_UNIFIED_ARCHITECTURE.md](../CLI_UNIFIED_ARCHITECTURE.md)

2. **Read the high-level plan:**
   - [CLI_MIGRATION_PLAN.md](../CLI_MIGRATION_PLAN.md)

3. **Start with Phase 0:**
   - [PHASE_0_PREPARATION.md](./PHASE_0_PREPARATION.md)

### Tracking Progress

Use the migration tracking document created in Phase 0:
- `docs/plans/phases/MIGRATION_TRACKING.md` (created during Phase 0)

### Artifacts Directory

Phase 0 creates an artifacts directory for baselines and inventories:
- `docs/plans/phases/artifacts/`

---

## Key Deliverables by Phase

| Phase | Key Deliverable |
|-------|-----------------|
| 0 | Test baselines, handler/command inventories |
| 1 | `handlers/context.py`, `execution/bootstrap.py` |
| 2 | Consolidated `rendering/service.py`, deleted `renderers.py` |
| 3 | All handlers using `HandlerContext` |
| 4 | `execution/registry.py`, operations registered in handlers |
| 5 | `commands/decorators.py`, all commands using `@cli_command` |
| 6 | All legacy files deleted, clean architecture |

---

## Files Created

| Phase | Files Created |
|-------|---------------|
| 1 | `handlers/context.py`, `execution/bootstrap.py` |
| 2 | `rendering/specs.py` |
| 4 | `execution/registry.py` |
| 5 | `commands/decorators.py` |

## Files Deleted (Phase 6)

- `handlers/base.py`
- `handlers/protocol.py`
- `execution/context.py`
- `execution/adapter.py`
- `commands/context.py`
- `introspection/registry.py`
- `operations/*.py` (all operation files)

---

## Risk Summary

| Risk Level | Phases |
|------------|--------|
| **Low** | 0, 1, 4, 6 |
| **Medium** | 2, 3, 5 |
| **High** | None |

---

## Parallelization

| Phase | Can Parallelize? | Details |
|-------|------------------|---------|
| 0 | No | Must complete sequentially |
| 1 | Partially | P1-1 to P1-5 and P1-6 can run in parallel |
| 2 | No | Must complete sequentially |
| 3 | **Yes** | Handler files can be migrated in parallel |
| 4 | No | Must complete sequentially |
| 5 | **Yes** | Command files can be migrated in parallel |
| 6 | Partially | Deletions can be batched |

---

## Common Commands

```bash
# Run CLI tests
uv run pytest tests/cli/ -v

# Type checking
uv run pyright --warnings --pythonversion=3.13 src/codeintel/cli/
uv run pyrefly check src/codeintel/cli/

# Linting
uv run ruff check --fix src/codeintel/cli/

# Coverage
uv run pytest tests/cli/ --cov=src/codeintel/cli --cov-report=term-missing

# CLI smoke test
codeintel --help
codeintel jobs list
codeintel health check
```

---

## Contact

For questions about the migration, refer to:
- Architecture document: [CLI_UNIFIED_ARCHITECTURE.md](../CLI_UNIFIED_ARCHITECTURE.md)
- Project standards: [AGENTS.md](../../../AGENTS.md)
