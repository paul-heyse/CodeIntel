# Plugin Robustness Design Proposal

## Key Insight: Build System as Schema Authority

**The build system already defines what each target produces. It should also define the schema contracts.**

Current architecture (3 separate sources of truth):
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  OutputTarget   │    │  TABLE_SCHEMAS  │    │     Plugin      │
│  (registry.py)  │    │  (schemas.py)   │    │  output_tables  │
│                 │    │                 │    │                 │
│ output_tables:  │    │ "core.ast_nodes"│    │ output_tables:  │
│ ("core.ast...") │←?→ │   columns: [...]│←?→ │ ("core.ast...") │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↑                     ↑                      ↑
         └── Must be manually kept in sync ──────────┘
```

**Proposed architecture (build system as authority):**
```
┌─────────────────────────────────────────┐
│           OutputTarget                   │
│  (Build System - Source of Truth)        │
│                                          │
│  name: "ast"                             │
│  contract: OutputContract(               │
│    tables: [                             │
│      TableSchema("core.ast_nodes", [...])│
│      TableSchema("core.ast_metrics",[...])│
│    ],                                    │
│    artifacts: []                         │
│  )                                       │
│  plugin: "ast_extract"                   │
└─────────────────────────────────────────┘
           │
           │ derives
           ▼
┌─────────────────────────────────────────┐
│  TABLE_SCHEMAS (computed, not authored) │
│                                          │
│  = union of all target.contract.tables   │
└─────────────────────────────────────────┘
           │
           │ validates
           ▼
┌─────────────────────────────────────────┐
│           Plugin                         │
│  (Consumer, not declarer)                │
│                                          │
│  writes to tables defined by its target  │
│  runtime validates against contract      │
└─────────────────────────────────────────┘
```

---

## Problem Statement

The current plugin system has several structural weaknesses discovered during E2E testing:

### Issues Found

1. **Schema gaps** - Plugins declare tables that don't exist in `TABLE_SCHEMAS`:
   - `core.scip_symbols` - declared but no schema
   - `core.scip_occurrences` - written but not even declared
   - `index.scip` - declared as "output_table" but it's a file, not a table

2. **Column mismatches** - Plugins write rows with wrong column counts:
   - `analytics.static_diagnostics`: schema has 8 columns, plugin writes 6

3. **No compile-time validation** - Schema issues only surface at runtime when:
   - A plugin tries to write to a missing table
   - A plugin writes rows with wrong column count
   - pandas/DuckDB throws cryptic errors

4. **Resource contention** - Parallel plugin execution causes DuckDB internal errors:
   - `INTERNAL Error: Attempted to dereference unique_ptr that is NULL!`
   - Likely connection sharing issues

5. **Artifact/Table confusion** - No distinction between:
   - **Tables**: DuckDB tables with schemas
   - **Artifacts**: Files produced (SCIP index, coverage data, etc.)

---

## Proposed Architecture: Build-First Schema Model

### Core Principle

**The build system's `OutputTarget` becomes the single source of truth for:**
- What tables a target produces
- The schema of those tables
- What artifacts (files) a target produces
- Dependencies between targets

**Plugins become pure executors** - they don't declare what they produce, they receive a contract to fulfill.

### 1. OutputContract - The Build System's Schema Declaration

```python
from codeintel.config.datasets.primitives import TableSchema, Column

@dataclass(frozen=True)
class ArtifactSpec:
    """Specification for a file artifact."""
    name: str              # "scip_index"
    path_template: str     # "{scip_dir}/index.scip"
    required: bool = True


@dataclass(frozen=True)
class OutputContract:
    """Contract defining exactly what an OutputTarget produces.
    
    This is the source of truth - TABLE_SCHEMAS is derived from this.
    """
    tables: tuple[TableSchema, ...] = ()
    artifacts: tuple[ArtifactSpec, ...] = ()
    
    def table_keys(self) -> tuple[str, ...]:
        """Return fully qualified table keys."""
        return tuple(f"{t.schema}.{t.name}" for t in self.tables)


@dataclass(frozen=True)
class OutputTarget:
    """A discrete output in the build system with full schema contract."""
    name: str
    module: TargetModule
    plugin: str
    contract: OutputContract  # ← NEW: schemas defined here
    dependencies: tuple[str, ...] = ()
    estimated_duration_ms: int = 10000
    
    @property
    def output_tables(self) -> tuple[str, ...]:
        """Derived from contract for backward compatibility."""
        return self.contract.table_keys()
```

### 2. Example: AST Target Definition

```python
# In build/registry.py - THIS becomes the source of truth

AST_TARGET = OutputTarget(
    name="ast",
    module="ingestion",
    plugin="ast_extract",
    contract=OutputContract(
        tables=(
            TableSchema(
                schema="core",
                name="ast_nodes",
                columns=[
                    Column("path", "VARCHAR", nullable=False),
                    Column("node_type", "VARCHAR", nullable=False),
                    Column("name", "VARCHAR", nullable=True),
                    Column("start_line", "INTEGER", nullable=False),
                    Column("end_line", "INTEGER", nullable=False),
                    Column("parent_path", "VARCHAR", nullable=True),
                    Column("repo", "VARCHAR", nullable=False),
                    Column("commit", "VARCHAR", nullable=False),
                    Column("created_at", "TIMESTAMP", nullable=False),
                ],
            ),
            TableSchema(
                schema="core",
                name="ast_metrics",
                columns=[
                    Column("rel_path", "VARCHAR", nullable=False),
                    Column("node_count", "INTEGER", nullable=False),
                    Column("max_depth", "INTEGER", nullable=False),
                    Column("repo", "VARCHAR", nullable=False),
                    Column("commit", "VARCHAR", nullable=False),
                    Column("created_at", "TIMESTAMP", nullable=False),
                ],
            ),
        ),
    ),
    dependencies=("modules",),
)
```

### 3. SCIP Target - Tables + Artifacts

```python
SCIP_TARGET = OutputTarget(
    name="scip",
    module="ingestion",
    plugin="scip_ingest",
    contract=OutputContract(
        tables=(
            TableSchema(
                schema="core",
                name="scip_symbols",
                columns=[
                    Column("repo", "VARCHAR", nullable=False),
                    Column("commit", "VARCHAR", nullable=False),
                    Column("relative_path", "VARCHAR", nullable=False),
                    Column("symbol", "VARCHAR", nullable=False),
                    Column("documentation", "VARCHAR", nullable=True),
                    Column("created_at", "TIMESTAMP", nullable=False),
                ],
            ),
            TableSchema(
                schema="core",
                name="scip_occurrences",
                columns=[
                    Column("repo", "VARCHAR", nullable=False),
                    Column("commit", "VARCHAR", nullable=False),
                    Column("relative_path", "VARCHAR", nullable=False),
                    Column("symbol", "VARCHAR", nullable=False),
                    Column("range_start_line", "INTEGER", nullable=False),
                    Column("range_start_col", "INTEGER", nullable=False),
                    Column("range_end_line", "INTEGER", nullable=False),
                    Column("range_end_col", "INTEGER", nullable=False),
                    Column("symbol_roles", "INTEGER", nullable=False),
                    Column("created_at", "TIMESTAMP", nullable=False),
                ],
            ),
        ),
        artifacts=(
            ArtifactSpec("scip_index", "{scip_dir}/index.scip"),
            ArtifactSpec("scip_json", "{scip_dir}/index.json"),
        ),
    ),
    dependencies=("modules",),
)
```

### 4. Deriving TABLE_SCHEMAS from Build Targets

```python
# In config/datasets/__init__.py or build/schema_registry.py

def compute_table_schemas() -> dict[str, TableSchema]:
    """Derive TABLE_SCHEMAS from all registered build targets."""
    from codeintel.build.registry import get_target_graph
    
    schemas: dict[str, TableSchema] = {}
    graph = get_target_graph()
    
    for target in graph.all_targets:
        for table_schema in target.contract.tables:
            key = f"{table_schema.schema}.{table_schema.name}"
            if key in schemas:
                # Validate schemas match
                _validate_schema_consistency(schemas[key], table_schema)
            schemas[key] = table_schema
    
    return schemas


# TABLE_SCHEMAS becomes a computed property, not a hand-maintained dict
TABLE_SCHEMAS = compute_table_schemas()
```

### 5. Plugin Execution with Contract Validation

```python
class BuildExecutor:
    def _execute_plugin(
        self,
        target: OutputTarget,
        plugin: IngestPluginProtocol,
    ) -> StageExecutionResult:
        """Execute plugin with contract validation."""
        
        # Create storage adapter that validates against contract
        adapter = ContractValidatingStorageAdapter(
            connection=self._gateway.connection,
            contract=target.contract,
        )
        
        # Execute plugin - it receives the adapter
        result = plugin.execute(ctx_with_adapter)
        
        # Validate all promised tables were written
        written_tables = set(adapter.tables_written)
        expected_tables = set(target.contract.table_keys())
        
        if written_tables != expected_tables:
            missing = expected_tables - written_tables
            extra = written_tables - expected_tables
            raise ContractViolationError(
                f"Plugin {target.plugin} contract violation: "
                f"missing={missing}, unexpected={extra}"
            )
        
        return result
```

### Benefits of Build-First Schema Model

1. **Single source of truth** - No more syncing between 3 places
2. **Schemas are versioned with targets** - Change target, schema changes together
3. **Compile-time validation** - Can validate all schemas at import time
4. **Plugin simplification** - Plugins don't declare outputs, they just write
5. **Clear contracts** - Build target explicitly states what it produces
6. **Artifact tracking** - Files are first-class, not hidden in output_tables

---

## Extended Insight: ALL Plugin Settings Should Flow from Build

### Current Plugin Settings Audit

Plugins currently declare **12+ class variables**. Let's examine each:

| Setting | Current Location | Should Come From Build? | Rationale |
|---------|------------------|------------------------|-----------|
| `plugin_name` | Plugin | **No** - Identity | This is just the plugin's identity |
| `plugin_description` | Plugin | **No** - Docs | Documentation only |
| `plugin_version` | Plugin | **No** - Versioning | Plugin implementation version |
| `plugin_stage` | Plugin | **YES** - Implicit | Build graph determines order, not explicit stages |
| `output_tables` | Plugin | **YES** - Contract | Already discussed - schemas from build |
| `depends_on` | Plugin | **YES** - Graph | Build target `dependencies` already defines this |
| `requires` | Plugin | **YES** - Contract | Resources needed should be in target contract |
| `tool_dependencies` | Plugin | **YES** - Contract | Tools needed should be in target contract |
| `supports_incremental` | Plugin | **MAYBE** - Behavior | Could be target property |
| `isolation_kind` | Plugin | **YES** - Execution | Build system decides isolation |
| `tracker_required` | Plugin | **YES** - Contract | Resource requirement |
| `resource_hints` | Plugin | **YES** - Planning | Build system needs this for planning |

### The Radical Simplification

**Current plugin class (12+ declarations):**
```python
@dataclass
class AstExtractPlugin(TrackerRequiringPlugin, TableWriterIngestPlugin):
    plugin_name: ClassVar[str] = "ast_extract"
    plugin_description: ClassVar[str] = "Parse Python AST..."
    plugin_stage: ClassVar[IngestStage] = "parse"
    plugin_version: ClassVar[str] = "2.0.0"
    output_tables: ClassVar[tuple[str, ...]] = ("core.ast_nodes", "core.ast_metrics")
    depends_on: ClassVar[tuple[str, ...]] = ("repo_scan",)
    requires: ClassVar[tuple[str, ...]] = ("change_tracker",)
    supports_incremental: ClassVar[bool] = True
    isolation_kind: ClassVar[IngestIsolationKind] = "process"
    tracker_required: ClassVar[bool] = False
    resource_hints: ClassVar[PluginResourceHints] = PluginResourceHints(...)
    
    def compute(self, ctx): ...
```

**Proposed plugin class (pure execution):**
```python
class AstExtractPlugin:
    """Parse Python AST - all config comes from build target."""
    
    def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin with context from build system.
        
        ctx.contract  - tables to write, artifacts to produce
        ctx.resources - tracker, modules, gateway (pre-resolved)
        ctx.config    - any runtime parameters
        """
        modules = ctx.resources.modules
        
        for module in modules:
            ast_rows = parse_ast(module)
            ctx.write_table("core.ast_nodes", ast_rows)
        
        return TargetResult.success()
```

### The Complete OutputTarget (All Settings)

```python
@dataclass(frozen=True)
class TargetResources:
    """Resources a target requires from the build system."""
    tracker: bool = False           # Needs change tracker
    modules: bool = False           # Needs module list
    gateway: bool = True            # Needs DB access (almost always)
    tools: tuple[str, ...] = ()     # External tools needed ("scip", "pyright")


@dataclass(frozen=True)
class TargetExecution:
    """Execution hints for build planning."""
    cpu_intensive: bool = False
    io_intensive: bool = False
    max_runtime_ms: int = 60000
    isolation: Literal["thread", "process", "none"] = "thread"
    supports_incremental: bool = True


@dataclass(frozen=True)
class OutputTarget:
    """Complete build target with ALL settings."""
    
    # Identity
    name: str
    module: TargetModule
    plugin: str                     # Plugin class to execute
    
    # Contract (what it produces)
    contract: OutputContract        # Tables + artifacts with schemas
    
    # Dependencies (what must run first)  
    dependencies: tuple[str, ...] = ()
    
    # Resources (what it needs)
    resources: TargetResources = field(default_factory=TargetResources)
    
    # Execution (how to run it)
    execution: TargetExecution = field(default_factory=TargetExecution)
```

### Example: Complete AST Target

```python
AST_TARGET = OutputTarget(
    name="ast",
    module="ingestion",
    plugin="ast_extract",
    
    contract=OutputContract(
        tables=(
            TableSchema("core", "ast_nodes", [...]),
            TableSchema("core", "ast_metrics", [...]),
        ),
    ),
    
    dependencies=("modules",),  # Replaces plugin's depends_on
    
    resources=TargetResources(
        tracker=True,           # Replaces requires=("change_tracker",)
        modules=True,           # Needs module list
    ),
    
    execution=TargetExecution(
        cpu_intensive=True,     # Replaces resource_hints
        supports_incremental=True,
        isolation="process",    # Replaces isolation_kind
    ),
)
```

### Example: Complete SCIP Target

```python
SCIP_TARGET = OutputTarget(
    name="scip",
    module="ingestion",
    plugin="scip_ingest",
    
    contract=OutputContract(
        tables=(
            TableSchema("core", "scip_symbols", [...]),
            TableSchema("core", "scip_occurrences", [...]),
        ),
        artifacts=(
            ArtifactSpec("scip_index", "{scip_dir}/index.scip"),
            ArtifactSpec("scip_json", "{scip_dir}/index.json"),
        ),
    ),
    
    dependencies=("modules",),
    
    resources=TargetResources(
        tracker=True,
        modules=True,
        tools=("scip-python", "scip"),  # Replaces tool_dependencies
    ),
    
    execution=TargetExecution(
        cpu_intensive=True,
        io_intensive=True,
        max_runtime_ms=300000,
    ),
)
```

### What's Left in the Plugin?

**Only execution logic:**

```python
class ScipIngestPlugin:
    """SCIP indexing - receives everything from build context."""
    
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        # Tools are pre-validated by build system
        scip_python = ctx.tools["scip-python"]
        scip = ctx.tools["scip"]
        
        # Artifacts paths come from contract
        output_scip = ctx.artifact_path("scip_index")
        output_json = ctx.artifact_path("scip_json")
        
        # Run the tool
        await run_scip_python(scip_python, ctx.repo_root, output_scip)
        await run_scip_print(scip, output_scip, output_json)
        
        # Parse and write to tables (contract-validated)
        symbols, occurrences = parse_scip_json(output_json)
        ctx.write_table("core.scip_symbols", symbols)
        ctx.write_table("core.scip_occurrences", occurrences)
        
        return TargetResult.success()
```

### Why This is Better

1. **Build system has complete visibility** - Can plan, parallelize, check readiness
2. **Plugin is testable in isolation** - Just needs a mock context
3. **No declaration drift** - Can't declare one thing and do another
4. **Tool validation at plan time** - Know if scip is missing before running
5. **Resource pre-provisioning** - Build system can prepare what plugin needs
6. **Clear separation** - Build = what, Plugin = how

---

## Eliminating Explicit Stages

### Current: Explicit Stage Assignment

Plugins declare their stage (`"scan"`, `"parse"`, `"index"`, `"enrich"`):

```python
plugin_stage: ClassVar[IngestStage] = "parse"
```

This creates problems:
- Stage must be manually kept in sync with dependencies
- Stage doesn't actually control execution order (dependencies do)
- Recipes define stage order separately

### Proposed: Stages are Implicit from Dependencies

**The dependency graph IS the stage definition.**

```python
# These targets form implicit stages:

modules_target = OutputTarget(
    name="modules",
    dependencies=(),  # No deps = runs first (stage 0)
    ...
)

ast_target = OutputTarget(
    name="ast", 
    dependencies=("modules",),  # Needs modules = stage 1
    ...
)

scip_target = OutputTarget(
    name="scip",
    dependencies=("modules",),  # Also needs modules = stage 1 (parallel with ast)
    ...
)

typing_target = OutputTarget(
    name="typing",
    dependencies=("modules", "scip"),  # Needs scip = stage 2
    ...
)
```

**The build system computes stages from the graph:**

```python
def compute_stages(targets: list[OutputTarget]) -> list[list[OutputTarget]]:
    """Compute execution stages from dependency graph.
    
    Targets with no dependencies are stage 0.
    Targets whose deps are all in stage N are in stage N+1.
    Targets in the same stage can run in parallel.
    """
    # Topological sort with level assignment
    ...
```

### Benefits

1. **Single source of truth** - Dependencies define order
2. **Automatic parallelization** - Same-stage targets can run together
3. **No stage/dependency mismatch** - Impossible to declare wrong stage
4. **Simpler recipes** - Just specify which targets, order is automatic

---

## Summary: The Build-First Plugin Architecture

| Concern | Current (Plugin-Declared) | Proposed (Build-Declared) |
|---------|---------------------------|---------------------------|
| Output tables | `output_tables` ClassVar | `contract.tables` |
| Output artifacts | Mixed in output_tables | `contract.artifacts` |
| Schema | Separate TABLE_SCHEMAS | In contract |
| Dependencies | `depends_on` ClassVar | `dependencies` |
| Stage | `plugin_stage` ClassVar | Computed from deps |
| Tool requirements | `tool_dependencies` ClassVar | `resources.tools` |
| Resource hints | `resource_hints` ClassVar | `execution.*` |
| Isolation | `isolation_kind` ClassVar | `execution.isolation` |
| Incremental support | `supports_incremental` ClassVar | `execution.supports_incremental` |

**Plugin becomes pure execution:**
- Receives `TargetExecutionContext` with everything it needs
- Writes to tables validated against contract
- No declarations, just implementation

---

## Legacy Architecture Reference

The sections below describe incremental fixes to the current architecture.
For a cleaner implementation, prefer the Build-First Schema Model above.

---

## Incremental Fixes (Legacy Approach)

### 1. Plugin Contract Types

```python
@dataclass(frozen=True)
class TableOutput:
    """Declaration of a table a plugin writes to."""
    table_key: str  # e.g., "core.ast_nodes"
    
    def validate(self) -> list[str]:
        """Validate table exists in TABLE_SCHEMAS."""
        errors = []
        if self.table_key not in TABLE_SCHEMAS:
            errors.append(f"Table {self.table_key} not in TABLE_SCHEMAS")
        return errors


@dataclass(frozen=True)
class ArtifactOutput:
    """Declaration of a file artifact a plugin produces."""
    name: str           # e.g., "scip_index"
    extension: str      # e.g., ".scip"
    required: bool = True


@dataclass(frozen=True)
class PluginOutputContract:
    """Complete output contract for a plugin."""
    tables: tuple[TableOutput, ...] = ()
    artifacts: tuple[ArtifactOutput, ...] = ()
    
    def validate(self) -> list[str]:
        """Validate all outputs are properly defined."""
        errors = []
        for table in self.tables:
            errors.extend(table.validate())
        return errors
```

### 2. Row Schema Validation

```python
@dataclass(frozen=True)
class RowContract:
    """Contract for rows written to a table."""
    table_key: str
    columns: tuple[str, ...]  # Expected column order
    
    @classmethod
    def from_schema(cls, table_key: str) -> "RowContract":
        """Build contract from TABLE_SCHEMAS."""
        schema = TABLE_SCHEMAS.get(table_key)
        if schema is None:
            raise ValueError(f"No schema for {table_key}")
        return cls(
            table_key=table_key,
            columns=tuple(c.name for c in schema.columns),
        )
    
    def validate_row(self, row: Sequence[object]) -> list[str]:
        """Validate a row matches the expected schema."""
        errors = []
        if len(row) != len(self.columns):
            errors.append(
                f"{self.table_key}: expected {len(self.columns)} columns, "
                f"got {len(row)}"
            )
        return errors
```

### 3. Plugin Registration Validation

```python
def register_plugin(plugin: IngestPluginProtocol) -> None:
    """Register a plugin with compile-time validation."""
    # 1. Validate all output tables exist
    for table_key in plugin.output_tables:
        if table_key.startswith("index.") or "." not in table_key:
            # This is an artifact, not a table - should use output_artifacts
            raise PluginRegistrationError(
                f"Plugin {plugin.plugin_name} declares '{table_key}' as output_table "
                f"but it appears to be an artifact. Use output_artifacts instead."
            )
        
        if table_key not in TABLE_SCHEMAS:
            raise PluginRegistrationError(
                f"Plugin {plugin.plugin_name} declares output table '{table_key}' "
                f"which is not defined in TABLE_SCHEMAS"
            )
    
    # 2. Build row contracts for validation during writes
    plugin._row_contracts = {
        table_key: RowContract.from_schema(table_key)
        for table_key in plugin.output_tables
        if table_key in TABLE_SCHEMAS
    }
```

### 4. Connection Isolation

```python
class IsolatedStorageAdapter:
    """Storage adapter with isolated connection per plugin execution."""
    
    def __init__(self, db_path: Path, plugin_name: str):
        self._db_path = db_path
        self._plugin_name = plugin_name
        self._con: DuckDBPyConnection | None = None
    
    def __enter__(self) -> "IsolatedStorageAdapter":
        """Create isolated connection for this plugin."""
        self._con = duckdb.connect(str(self._db_path))
        return self
    
    def __exit__(self, *args) -> None:
        """Close connection when plugin completes."""
        if self._con:
            self._con.close()
            self._con = None
    
    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        contract: RowContract,
    ) -> BatchResult:
        """Write rows with contract validation."""
        # Validate all rows before writing
        for i, row in enumerate(rows):
            errors = contract.validate_row(row)
            if errors:
                raise RowValidationError(
                    f"Row {i} validation failed: {errors}"
                )
        
        # Now write (we know schema matches)
        ...
```

### 5. Execution Isolation

```python
class PluginExecutionContext:
    """Execution context with proper isolation."""
    
    @contextmanager
    def isolated_storage(self) -> Generator[IsolatedStorageAdapter, None, None]:
        """Create isolated storage adapter for plugin execution."""
        with IsolatedStorageAdapter(
            self.db_path, 
            self.plugin_name
        ) as adapter:
            yield adapter
```

---

## Migration Path

### Phase 1: Schema Audit (Immediate)

1. **Audit all plugins** - Document actual tables written vs declared
2. **Add missing schemas** - Create `core.scip_symbols`, `core.scip_occurrences`
3. **Fix column mismatches** - Update `analytics.static_diagnostics` or plugin

```bash
# Create audit script
python -m tools.audit_plugin_schemas
```

### Phase 2: Separate Tables from Artifacts

1. **New class attributes**:
   ```python
   class ScipIngestPlugin:
       output_tables: ClassVar[tuple[str, ...]] = (
           "core.scip_symbols",
           "core.goid_crosswalk",
       )
       output_artifacts: ClassVar[tuple[str, ...]] = (
           "index.scip",
           "index.json",
       )
   ```

2. **Update all plugins** with proper classification

### Phase 3: Registration-Time Validation

1. **Add validation to registry**:
   ```python
   def register(self, plugin: IngestPluginProtocol) -> None:
       errors = self._validate_plugin(plugin)
       if errors:
           raise PluginRegistrationError(f"Plugin {plugin.name}: {errors}")
       self._plugins[plugin.name] = plugin
   ```

2. **Fail fast** on invalid plugin declarations

### Phase 4: Runtime Row Validation

1. **Add row contracts** to storage adapter
2. **Validate before write** (catch column mismatches early)
3. **Clear error messages** pointing to exact problem

### Phase 5: Connection Isolation

1. **Create connection-per-plugin** model
2. **Remove shared connection** from parallel execution
3. **Add connection pool** for efficiency if needed

---

## Schema Definitions Needed

```python
# Add to TABLE_SCHEMAS in schemas.py

"core.scip_symbols": TableSchema(
    schema="core",
    name="scip_symbols",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("relative_path", "VARCHAR", nullable=False),
        Column("symbol", "VARCHAR", nullable=False),
        Column("documentation", "VARCHAR", nullable=True),
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    description="SCIP symbol definitions",
),

"core.scip_occurrences": TableSchema(
    schema="core",
    name="scip_occurrences",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("relative_path", "VARCHAR", nullable=False),
        Column("symbol", "VARCHAR", nullable=False),
        Column("range_start_line", "INTEGER", nullable=False),
        Column("range_start_col", "INTEGER", nullable=False),
        Column("range_end_line", "INTEGER", nullable=False),
        Column("range_end_col", "INTEGER", nullable=False),
        Column("symbol_roles", "INTEGER", nullable=False),
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    description="SCIP symbol occurrences (references and definitions)",
),
```

---

## Benefits

1. **Fail-fast** - Invalid plugins detected at registration, not runtime
2. **Clear errors** - "Table X not in TABLE_SCHEMAS" vs "INTERNAL Error: NULL pointer"
3. **Type safety** - Row contracts catch column mismatches before DuckDB
4. **Isolation** - Parallel plugins don't corrupt shared state
5. **Auditability** - Plugin → Table mapping is explicit and validated

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Add missing SCIP schemas | Low | Unblocks E2E |
| 2 | Fix column mismatch in static_diagnostics | Low | Unblocks E2E |
| 3 | Separate output_tables from output_artifacts | Medium | Clarity |
| 4 | Add registration-time validation | Medium | Prevention |
| 5 | Add row contracts | Medium | Better errors |
| 6 | Connection isolation | High | Stability |

---

## Questions for Discussion

1. Should artifacts be tracked in the build system like tables?
2. Should row validation be opt-in or mandatory?
3. How do we handle schema evolution / migrations?
4. Should plugins declare expected input tables too?

