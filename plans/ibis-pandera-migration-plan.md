# Ibis + Pandera Implementation Plan

## Executive Summary

This plan details the systematic migration from SQL/Python-driven data manipulation to Ibis and Pandera-driven processes across the CodeIntel codebase. The migration leverages existing foundations and extends them to achieve comprehensive coverage.

### Current State Analysis

**Already Implemented:**
- `IbisGateway` in `storage/ibis_adapter.py` - fully integrated with `StorageGateway`
- `pandera_schemas.py` with:
  - Dynamic schema generation from `DatasetContract`
  - Column-level checks for 15+ datasets
  - DataFrame-level checks for `core.goids` and `core.goid_crosswalk`
  - `validate_dataset_df()` validation function
  - `pandera_to_json_schema()` conversion
- `ibis_views.py` with 7 Ibis-defined views
- `FunctionRepository` and `GraphRepository` using Ibis with SQL fallbacks
- JSON Schema export in `dataset_backend.py`

**Remaining Work:**
- Complete Pandera schema coverage for all 85+ datasets
- Migrate remaining SQL to Ibis in repositories, backends, and plugins
- Add property-based tests with Hypothesis + Pandera
- Add cross-table invariant checks
- Migrate analytics/ingestion write paths to validate before insert

---

## Phase 1: Foundation Completion (Week 1-2)

### 1.1 Complete Pandera Schema Registry

**Goal:** Ensure every dataset has a Pandera schema with appropriate invariants.

**Files to modify:**
- `src/codeintel/storage/pandera_schemas.py`

**Tasks:**

1. **Audit existing schema coverage**
   ```python
   # Generate report of datasets without schemas
   from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
   from codeintel.storage.pandera_schemas import DATASET_SCHEMAS
   
   missing = set(DATASET_CONTRACTS_BY_TABLE_KEY.keys()) - set(DATASET_SCHEMAS.keys())
   ```

2. **Add column-level checks for remaining datasets**
   
   Priority datasets requiring column checks:
   
   | Dataset | Required Checks |
   |---------|-----------------|
   | `analytics.function_profile` | risk_score >= 0, coverage_ratio 0-1 |
   | `analytics.file_profile` | same as function_profile |
   | `analytics.module_profile` | import fan metrics >= 0 |
   | `analytics.test_catalog` | duration_seconds >= 0 |
   | `analytics.entrypoints` | confidence 0-1 |
   | `analytics.external_dependencies` | call_count >= 0 |
   | `analytics.subsystems` | module_count >= 0 |
   | `analytics.subsystem_modules` | N/A (composite key) |
   | `analytics.data_models` | field_count >= 0 |
   | `graph.cfg_blocks` | line numbers >= 1 |
   | `graph.cfg_edges` | N/A (edge table) |
   | `graph.dfg_edges` | N/A (edge table) |
   | `graph.symbol_use_edges` | line/col >= 1 |

3. **Add DataFrame-level checks for key datasets**

   ```python
   # Example: Add checks for analytics.function_profile
   _DATAFRAME_CHECKS["analytics.function_profile"] = [
       Check(
           lambda df: ~df.duplicated(subset=["repo", "commit", "function_goid_h128"]).any(),
           error="Duplicate primary key in analytics.function_profile",
       ),
       Check(
           lambda df: df["coverage_ratio"].isna() | ((df["coverage_ratio"] >= 0) & (df["coverage_ratio"] <= 1)),
           error="coverage_ratio must be between 0 and 1",
       ),
   ]
   ```

4. **Implement schema inheritance for related datasets**
   
   Create base column definitions that can be reused:
   ```python
   _BASE_REPO_COMMIT_COLUMNS = {
       "repo": Column(_STRING_DTYPE),
       "commit": Column(_STRING_DTYPE),
   }
   
   _BASE_GOID_COLUMNS = {
       **_BASE_REPO_COMMIT_COLUMNS,
       "function_goid_h128": Column(_INT_DTYPE, Check(lambda s: s.isna() | (s >= 0))),
   }
   ```

**Deliverables:**
- [ ] Complete `_COLUMN_CHECKS` for all 85+ datasets
- [ ] Complete `_DATAFRAME_CHECKS` for datasets with uniqueness constraints
- [ ] Unit tests validating all schemas load correctly

---

### 1.2 Extend Ibis Views Coverage

**Goal:** Define all compositive views using Ibis instead of raw SQL.

**Files to modify:**
- `src/codeintel/storage/views/ibis_views.py`

**Current Ibis Views (7):**
1. `docs.v_function_summary`
2. `analytics.v_function_summary`
3. `graph.v_call_graph_degree`
4. `docs.v_call_graph_enriched`
5. `core.v_goid_crosswalk_join`
6. `core.v_goid_crosswalk_mismatches`
7. `analytics.v_function_hotspots`
8. `graph.v_import_graph_degree`

**Views to add (prioritized):**

| View | Complexity | Dependencies |
|------|------------|--------------|
| `docs.v_module_with_subsystem` | Medium | subsystem_modules, modules, graph_metrics |
| `docs.v_subsystem_summary` | Medium | subsystems, subsystem_modules |
| `docs.v_subsystem_profile` | High | subsystems, subsystem_graph_metrics |
| `docs.v_subsystem_coverage` | High | subsystems, test_profile, coverage |
| `docs.v_file_summary` | Medium | modules, hotspots, typedness |
| `docs.v_function_architecture` | High | function_profile, graph_metrics, cfg/dfg |
| `docs.v_module_architecture` | High | module_profile, graph_metrics |
| `analytics.v_config_data_flow` | Medium | config_values, config_data_flow |

**Implementation template:**
```python
def create_module_with_subsystem_view(gateway: StorageGateway) -> None:
    """Create docs.v_module_with_subsystem using Ibis expressions."""
    con = gateway.ibis.con
    sm = con.table("analytics.subsystem_modules")
    subsys = con.table("analytics.subsystems")
    modules = con.table("core.modules")
    gm = con.table("analytics.graph_metrics_modules")
    
    joined = (
        sm.left_join(subsys, ["repo", "commit", "subsystem_id"])
        .left_join(modules, ["repo", "commit", sm.module == modules.module])
        .left_join(gm, ["repo", "commit", sm.module == gm.module])
    )
    
    view = joined.select(
        sm.repo,
        sm.commit,
        sm.subsystem_id,
        subsys.name.name("subsystem_name"),
        sm.module,
        sm.role,
        modules.rel_path,
        modules.tags,
        modules.owners,
        gm.import_fan_in,
        gm.import_fan_out,
        gm.symbol_fan_in,
        gm.symbol_fan_out,
        # Computed fields as needed
    )
    con.create_view("docs.v_module_with_subsystem", view, overwrite=True)
```

**Deliverables:**
- [ ] All `docs.*` views converted to Ibis
- [ ] Updated `create_all_ibis_views()` to include new views
- [ ] Tests verifying view schemas match expected Pandera schemas

---

## Phase 2: Repository Migration (Week 3-4)

### 2.1 Migrate Storage Repositories to Ibis

**Goal:** Replace raw SQL in repositories with Ibis expressions while maintaining SQL fallbacks for robustness.

**Files to modify:**
- `src/codeintel/storage/repositories/datasets.py`
- `src/codeintel/storage/repositories/functions.py`
- `src/codeintel/storage/repositories/graphs.py`
- `src/codeintel/storage/repositories/modules.py`
- `src/codeintel/storage/repositories/subsystems.py`
- `src/codeintel/storage/repositories/tests.py`
- `src/codeintel/storage/repositories/dataflow.py`

**Pattern to follow (already established in `FunctionRepository`):**

```python
@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    """Repository with Ibis-first query methods."""

    @staticmethod
    def _validated_records(table_key: str, expr: it.Table) -> list[RowDict]:
        """Execute Ibis expression and return Pandera-validated records."""
        df = pd.DataFrame(expr.execute())
        validated = validate_dataset_df(table_key, df)
        return validated.where(pd.notna(validated), None).to_dict(orient="records")

    def read_dataset_dataframe(
        self,
        *,
        table_key: str,
        limit: int,
        offset: int,
    ) -> pd.DataFrame:
        """Read dataset rows via Ibis with Pandera validation."""
        try:
            table = self.gateway.ibis.table(table_key)
            expr = table.filter(
                (table.repo == self.repo) & (table.commit == self.commit)
            ).limit(limit).offset(offset)
            df = expr.execute()
            return validate_dataset_df(table_key, df)
        except IbisError:
            # SQL fallback for edge cases
            return self._read_dataset_sql(table_key, limit, offset)
```

**Migration checklist per repository:**

| Repository | Methods to Migrate | Priority |
|------------|-------------------|----------|
| `datasets.py` | `read_dataset_rows`, `read_dataset_dataframe` | High |
| `modules.py` | `list_modules`, `get_module_by_path` | High |
| `subsystems.py` | `list_subsystems`, `get_subsystem_modules` | Medium |
| `tests.py` | `list_test_catalog`, `get_test_coverage` | Medium |
| `dataflow.py` | `get_config_data_flow` | Low |

**Deliverables:**
- [ ] All repository read methods use Ibis with SQL fallback
- [ ] All returned DataFrames pass through `validate_dataset_df()`
- [ ] Integration tests for each repository method

---

### 2.2 Migrate Serving Backends to Ibis

**Goal:** Replace raw SQL in serving backends with Ibis queries.

**Files to modify:**
- `src/codeintel/serving/backend/dataset_backend.py`
- `src/codeintel/serving/backend/function_backend.py`
- `src/codeintel/serving/backend/subsystem_backend.py`
- `src/codeintel/serving/backend/module_backend.py`
- `src/codeintel/serving/backend/duckdb_service.py`

**Priority methods to migrate:**

1. **DatasetQueryLayer** (already partially done)
   - `read_dataset_rows` - uses repository, needs Ibis path
   - `dataset_schema` - uses repository, already validates

2. **FunctionQueryLayer**
   - `get_function_summary` - migrate to Ibis
   - `list_high_risk_functions` - migrate to Ibis
   - `get_function_callers/callees` - migrate graph queries

3. **SubsystemQueryLayer**
   - `list_subsystems` - migrate to Ibis
   - `get_subsystem_profile` - migrate to Ibis
   - `get_subsystem_coverage` - migrate to Ibis

**Implementation pattern:**

```python
@dataclass
class SubsystemQueryLayer:
    context: BackendContext
    repositories: DuckDBRepositories

    def list_subsystems(self, *, limit: int = 100) -> list[SubsystemSummary]:
        """List subsystems using Ibis."""
        con = self.context.gateway.ibis.con
        subsys = con.table("analytics.subsystems")
        modules = con.table("analytics.subsystem_modules")
        
        expr = (
            subsys.filter(
                (subsys.repo == self.repo) & (subsys.commit == self.commit)
            )
            .left_join(
                modules.group_by(["repo", "commit", "subsystem_id"])
                .aggregate(module_count=modules.module.count()),
                ["repo", "commit", "subsystem_id"],
            )
            .limit(limit)
        )
        
        df = expr.execute()
        validated = validate_dataset_df("analytics.subsystems", df)
        return [SubsystemSummary.from_row(row) for row in validated.to_dict("records")]
```

**Deliverables:**
- [ ] All backend query methods use Ibis
- [ ] All returned data passes Pandera validation
- [ ] MCP/HTTP handlers receive validated data

---

## Phase 3: Analytics/Plugin Write Paths (Week 5-6)

### 3.1 Migrate Analytics Write Paths

**Goal:** Add Pandera validation before all database inserts in analytics plugins.

**Files to modify:**
- `src/codeintel/build/analytics/functions/metrics.py` ✓ (already done)
- `src/codeintel/build/analytics/utilities/datasets.py`
- `src/codeintel/build/analytics/adapters/*.py`
- `src/codeintel/build/analytics/plugins/**/*.py`

**Current state in `metrics.py`:**
```python
def _validated_records(table_key: str, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    validated = validate_dataset_df(table_key, df)
    return validated.where(pd.notna(validated), None).to_dict(orient="records")
```

**Plugins requiring migration:**

| Plugin Path | Output Tables | Priority |
|-------------|---------------|----------|
| `analytics/plugins/functions/metrics.py` | function_metrics, function_types | ✓ Done |
| `analytics/plugins/risk/factors.py` | goid_risk_factors | High |
| `analytics/plugins/profiles/build.py` | function_profile, file_profile, module_profile | High |
| `analytics/plugins/hotspots/build.py` | hotspots | High |
| `analytics/plugins/coverage/functions.py` | coverage_functions | Medium |
| `analytics/plugins/coverage/test_edges.py` | test_coverage_edges | Medium |
| `analytics/graphs/graph_metrics.py` | graph_metrics_functions, graph_metrics_modules | Medium |
| `analytics/graphs/graph_metrics_ext.py` | graph_metrics_*_ext | Medium |
| `analytics/graphs/subsystem_graph_metrics.py` | subsystem_graph_metrics | Medium |
| `analytics/subsystems/materialize.py` | subsystems, subsystem_modules | Medium |

**Implementation pattern for adapters:**

```python
# analytics/adapters/functions.py

from codeintel.storage.pandera_schemas import validate_dataset_df

@dataclass
class FunctionAdapter:
    """Database adapter for function analytics with Pandera validation."""
    
    def persist_function_metrics(
        self,
        gateway: StorageGateway,
        rows: Sequence[FunctionMetricsRow],
        *,
        scope: DeleteScope,
    ) -> int:
        """Persist function metrics with Pandera validation."""
        if not rows:
            return 0
        
        df = pd.DataFrame(rows)
        validated_df = validate_dataset_df("analytics.function_metrics", df)
        
        # Delete existing rows for scope
        gateway.con.execute(
            "DELETE FROM analytics.function_metrics WHERE repo = ? AND commit = ?",
            [scope.repo, scope.commit],
        )
        
        # Insert validated rows
        gateway.con.register("tmp_metrics", validated_df)
        gateway.con.execute(
            "INSERT INTO analytics.function_metrics SELECT * FROM tmp_metrics"
        )
        
        return len(validated_df)
```

**Deliverables:**
- [ ] All analytics adapters validate with Pandera before insert
- [ ] Consistent error handling for validation failures
- [ ] Metrics for validation pass/fail rates

---

### 3.2 Migrate Ingestion/Graph Write Paths

**Goal:** Add Pandera validation to ingestion and graph building plugins.

**Files to modify:**
- `src/codeintel/build/graphs/plugins/builders/callgraph.py`
- `src/codeintel/build/graphs/plugins/builders/importgraph.py`
- `src/codeintel/build/graphs/plugins/builders/cfg_dfg.py`
- `src/codeintel/ingestion/plugins/*.py`
- `src/codeintel/ingestion/adapters/duckdb_storage.py`

**Graph plugins requiring migration:**

| Plugin | Output Tables | Current Pattern |
|--------|---------------|-----------------|
| `callgraph.py` | call_graph_nodes, call_graph_edges | Direct tuple insert |
| `importgraph.py` | import_graph_edges, import_modules | Direct tuple insert |
| `cfg_dfg.py` | cfg_blocks, cfg_edges, dfg_edges | Direct tuple insert |
| `symboluse.py` | symbol_use_edges | Direct tuple insert |

**Implementation for callgraph builder:**

```python
# graphs/plugins/builders/callgraph.py

from codeintel.storage.pandera_schemas import validate_dataset_df

async def _persist_call_graph(
    gateway: StorageGateway,
    cfg: CallGraphStepConfig,
    nodes: list[CallGraphNodeRow],
    edges: list[CallGraphEdgeRow],
) -> dict[str, int]:
    """Persist call graph with Pandera validation."""
    
    # Validate nodes
    if nodes:
        nodes_df = pd.DataFrame(nodes)
        validated_nodes = validate_dataset_df("graph.call_graph_nodes", nodes_df)
        gateway.graph.insert_call_graph_nodes_df(validated_nodes)
    
    # Validate edges
    if edges:
        edges_df = pd.DataFrame(edges)
        validated_edges = validate_dataset_df("graph.call_graph_edges", edges_df)
        gateway.graph.insert_call_graph_edges_df(validated_edges)
    
    return {
        "graph.call_graph_nodes": len(nodes),
        "graph.call_graph_edges": len(edges),
    }
```

**Ingestion plugins requiring migration:**

| Plugin | Output Tables |
|--------|---------------|
| `scip_plugin.py` | core.goids, core.goid_crosswalk |
| `ast_extract.py` | core.ast_nodes, core.ast_metrics |
| `cst_extract.py` | core.cst_nodes |
| `docstrings_plugin.py` | core.docstrings |
| `coverage_plugin.py` | analytics.coverage_lines |

**Deliverables:**
- [ ] All graph builders validate with Pandera
- [ ] All ingestion plugins validate with Pandera
- [ ] Bulk insert helpers use DataFrame validation

---

## Phase 4: Testing & Validation (Week 7-8)

### 4.1 Property-Based Tests with Hypothesis + Pandera

**Goal:** Create comprehensive property-based tests for key datasets.

**Files to create:**
- `tests/storage/test_pandera_properties.py`
- `tests/analytics/test_function_metrics_properties.py`
- `tests/graphs/test_callgraph_properties.py`
- `tests/core/test_goids_properties.py`

**Pandera + Hypothesis integration:**

```python
# tests/storage/test_pandera_properties.py

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from pandera import hypothesis as pah

from codeintel.storage.pandera_schemas import (
    DATASET_SCHEMAS,
    get_dataset_schema,
)


@pytest.mark.parametrize("table_key", [
    "analytics.function_metrics",
    "analytics.function_types",
    "analytics.goid_risk_factors",
    "graph.call_graph_edges",
    "core.goids",
])
def test_schema_generates_valid_dataframes(table_key: str) -> None:
    """Verify Pandera schemas can generate valid DataFrames via Hypothesis."""
    schema = get_dataset_schema(table_key)
    assert schema is not None
    
    strategy = pah.dataframe_strategy(schema, size=10)
    
    @given(df=strategy)
    @settings(max_examples=5)
    def check_validates(df: pd.DataFrame) -> None:
        validated = schema.validate(df)
        assert len(validated) == len(df)
    
    check_validates()
```

**Domain-specific property tests:**

```python
# tests/analytics/test_function_metrics_properties.py

from hypothesis import given, settings
from pandera import hypothesis as pah

from codeintel.storage.pandera_schemas import get_dataset_schema
from codeintel.build.analytics.compute.hotspots.metrics import compute_hotspot_score


METRICS_SCHEMA = get_dataset_schema("analytics.function_metrics")


@settings(max_examples=20)
@given(df=pah.dataframe_strategy(METRICS_SCHEMA, size=50))
def test_hotspot_scores_never_negative(df: pd.DataFrame) -> None:
    """Hotspot scores should never be negative for valid metrics."""
    if df.empty:
        return
    
    scores = compute_hotspot_score(
        complexity=df["cyclomatic_complexity"],
        loc=df["loc"],
    )
    
    assert (scores >= 0).all(), "Hotspot scores must be non-negative"


@settings(max_examples=20)
@given(df=pah.dataframe_strategy(METRICS_SCHEMA, size=50))
def test_complexity_monotonicity(df: pd.DataFrame) -> None:
    """Higher complexity should not decrease hotspot score (fixed other factors)."""
    if len(df) < 2:
        return
    
    sorted_df = df.sort_values("cyclomatic_complexity")
    # For same LOC, higher complexity should not decrease score
    # (This is a simplified invariant check)
```

**Cross-table invariant tests:**

```python
# tests/core/test_goids_crosswalk_properties.py

def test_crosswalk_rows_have_matching_goids(test_gateway: StorageGateway) -> None:
    """Every crosswalk row must have a matching goid row."""
    con = test_gateway.ibis.con
    goids = con.table("core.goids")
    xwalk = con.table("core.goid_crosswalk")
    
    missing = xwalk.left_join(
        goids,
        [
            xwalk.repo == goids.repo,
            xwalk.commit == goids.commit,
            xwalk.goid == goids.urn,
        ],
    ).filter(goids.urn.isnull())
    
    missing_df = missing.execute()
    assert missing_df.empty, f"Found {len(missing_df)} crosswalk rows without matching goids"


def test_goid_urn_format_consistency(test_gateway: StorageGateway) -> None:
    """All GOID URNs must follow expected format."""
    con = test_gateway.ibis.con
    goids = con.table("core.goids").limit(1000).execute()
    
    for _, row in goids.iterrows():
        urn = row["urn"]
        assert urn.startswith("goid:"), f"URN must start with 'goid:': {urn}"
        assert "?" in urn, f"URN must contain query parameters: {urn}"
```

**Deliverables:**
- [ ] Property tests for all 15+ priority datasets
- [ ] Cross-table invariant tests for goids/crosswalk
- [ ] Integration with CI pipeline

---

### 4.2 View Consistency Tests

**Goal:** Verify Ibis views match expected Pandera schemas.

**Files to create:**
- `tests/storage/test_ibis_views.py`

```python
# tests/storage/test_ibis_views.py

import pytest
from codeintel.storage.views.ibis_views import create_all_ibis_views
from codeintel.storage.pandera_schemas import get_dataset_schema


@pytest.fixture
def gateway_with_views(test_gateway, seeded_data):
    """Gateway with Ibis views created."""
    create_all_ibis_views(test_gateway)
    return test_gateway


@pytest.mark.parametrize("view_key", [
    "docs.v_function_summary",
    "docs.v_call_graph_enriched",
    "docs.v_subsystem_summary",
    "docs.v_module_with_subsystem",
    "analytics.v_function_summary",
    "graph.v_call_graph_degree",
])
def test_view_matches_pandera_schema(
    gateway_with_views: StorageGateway,
    view_key: str,
) -> None:
    """Verify view data validates against its Pandera schema."""
    schema = get_dataset_schema(view_key)
    if schema is None:
        pytest.skip(f"No Pandera schema for {view_key}")
    
    df = gateway_with_views.ibis.table(view_key).limit(100).execute()
    if df.empty:
        pytest.skip(f"View {view_key} is empty")
    
    validated = schema.validate(df)
    assert len(validated) == len(df)
```

**Deliverables:**
- [ ] Tests for all Ibis-defined views
- [ ] Schema drift detection in CI

---

## Phase 5: Advanced Features (Week 9-10)

### 5.1 JSON Schema Export Enhancements

**Goal:** Generate comprehensive JSON Schemas from Pandera for API documentation and LLM tools.

**Files to modify:**
- `src/codeintel/storage/pandera_schemas.py`
- `src/codeintel/serving/schema_export.py` (new)

**Enhanced JSON Schema generation:**

```python
# storage/pandera_schemas.py - extend pandera_to_json_schema

def pandera_to_json_schema(
    df_schema: DataFrameSchema,
    *,
    include_examples: bool = False,
    include_descriptions: bool = False,
) -> dict[str, Any]:
    """Convert Pandera schema to JSON Schema with optional enhancements."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    
    for name, column in df_schema.columns.items():
        json_type, fmt = _json_type_for_dtype(column.dtype)
        types: list[str] = [json_type]
        if column.nullable:
            types.append("null")
        
        field_schema: dict[str, Any] = {"type": types}
        if fmt is not None:
            field_schema["format"] = fmt
        
        # Add constraints from checks
        for check in column.checks:
            if hasattr(check, "_name"):
                if check._name == "greater_than_or_equal_to":
                    field_schema["minimum"] = check._statistics["min_value"]
                elif check._name == "less_than_or_equal_to":
                    field_schema["maximum"] = check._statistics["max_value"]
        
        if include_descriptions and hasattr(column, "description"):
            field_schema["description"] = column.description
        
        properties[name] = field_schema
        if not column.nullable:
            required.append(name)
    
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
```

**MCP Tool Schema Integration:**

```python
# serving/mcp/tool_schemas.py

from codeintel.storage.pandera_schemas import dataset_json_schema

FUNCTION_METRICS_TOOL = {
    "name": "get_function_metrics",
    "description": "Get metrics for a function by GOID",
    "inputSchema": {
        "type": "object",
        "properties": {
            "goid_h128": {"type": "integer", "minimum": 0},
        },
        "required": ["goid_h128"],
    },
    "outputSchema": dataset_json_schema("analytics.function_metrics"),
}
```

**Deliverables:**
- [ ] Enhanced JSON Schema with constraints
- [ ] MCP tool schemas from Pandera
- [ ] OpenAPI schema generation

---

### 5.2 Ibis Query Optimization

**Goal:** Optimize common query patterns using Ibis features.

**Files to modify:**
- `src/codeintel/storage/repositories/*.py`
- `src/codeintel/serving/backend/*.py`

**Optimization patterns:**

1. **Lazy evaluation for chained queries:**
   ```python
   # Build expression without executing
   expr = (
       gateway.ibis.table("analytics.function_metrics")
       .filter(lambda t: t.repo == repo)
       .filter(lambda t: t.commit == commit)
       .filter(lambda t: t.risk_score >= threshold)
       .order_by(lambda t: t.risk_score.desc())
       .limit(100)
   )
   # Execute once at the end
   df = expr.execute()
   ```

2. **Projection pushdown:**
   ```python
   # Select only needed columns early
   expr = (
       gateway.ibis.table("docs.v_function_summary")
       .select("function_goid_h128", "qualname", "risk_score")
       .filter(...)
   )
   ```

3. **Common subexpression reuse:**
   ```python
   base = gateway.ibis.table("analytics.function_metrics").filter(...)
   
   high_risk = base.filter(lambda t: t.risk_score > 0.8)
   low_coverage = base.filter(lambda t: t.coverage_ratio < 0.5)
   
   # Both queries share the same base expression
   ```

**Deliverables:**
- [ ] Query optimization guidelines
- [ ] Performance benchmarks
- [ ] Caching layer for repeated queries

---

### 5.3 Validation Error Handling

**Goal:** Implement consistent error handling for Pandera validation failures.

**Files to create:**
- `src/codeintel/storage/validation/errors.py`

```python
# storage/validation/errors.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

if TYPE_CHECKING:
    from pandera import DataFrameSchema

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of Pandera validation with error details."""
    
    success: bool
    validated_df: pd.DataFrame | None
    errors: list[str]
    error_count: int
    
    @classmethod
    def from_validation(
        cls,
        schema: DataFrameSchema,
        df: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> ValidationResult:
        """Validate DataFrame and capture errors."""
        try:
            validated = schema.validate(df, lazy=True)
            return cls(
                success=True,
                validated_df=validated,
                errors=[],
                error_count=0,
            )
        except SchemaErrors as exc:
            errors = [str(e) for e in exc.failure_cases.itertuples()]
            if strict:
                raise
            log.warning(
                "Validation failed with %d errors for %s",
                len(errors),
                schema.name,
            )
            return cls(
                success=False,
                validated_df=None,
                errors=errors,
                error_count=len(errors),
            )


def validate_with_fallback(
    table_key: str,
    df: pd.DataFrame,
    *,
    on_error: str = "log",
) -> pd.DataFrame:
    """Validate with configurable error handling.
    
    Parameters
    ----------
    table_key
        Dataset identifier.
    df
        DataFrame to validate.
    on_error
        Error handling: "raise", "log", "ignore".
    
    Returns
    -------
    pd.DataFrame
        Validated DataFrame, or original on failure (if not raising).
    """
    from codeintel.storage.pandera_schemas import get_dataset_schema
    
    schema = get_dataset_schema(table_key)
    if schema is None:
        return df
    
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        if on_error == "raise":
            raise
        if on_error == "log":
            log.warning(
                "Validation failed for %s: %d errors",
                table_key,
                len(exc.failure_cases),
            )
        return df
```

**Deliverables:**
- [ ] Consistent validation error handling
- [ ] Error aggregation and reporting
- [ ] Configurable strictness levels

---

## Phase 6: Documentation & Tooling (Week 11-12)

### 6.1 Developer Documentation

**Files to create:**
- `docs/development/ibis-pandera-guide.md`

**Contents:**
1. Ibis query patterns and best practices
2. Pandera schema definition guidelines
3. Testing strategies with Hypothesis
4. Migration checklist for new datasets
5. Troubleshooting common issues

### 6.2 Schema Introspection Tools

**Files to create:**
- `tools/schema_audit.py`

```python
# tools/schema_audit.py

"""Audit Pandera schema coverage and generate reports."""

import json
from pathlib import Path

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.storage.pandera_schemas import (
    DATASET_SCHEMAS,
    pandera_to_json_schema,
)


def audit_schema_coverage() -> dict[str, object]:
    """Generate schema coverage report."""
    all_tables = set(DATASET_CONTRACTS_BY_TABLE_KEY.keys())
    covered = set(DATASET_SCHEMAS.keys())
    
    return {
        "total_tables": len(all_tables),
        "covered": len(covered),
        "missing": sorted(all_tables - covered),
        "coverage_pct": len(covered) / len(all_tables) * 100,
    }


def export_all_schemas(output_dir: Path) -> None:
    """Export all Pandera schemas as JSON Schema files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for table_key, schema in DATASET_SCHEMAS.items():
        json_schema = pandera_to_json_schema(schema)
        filename = table_key.replace(".", "_") + ".json"
        (output_dir / filename).write_text(
            json.dumps(json_schema, indent=2)
        )


if __name__ == "__main__":
    report = audit_schema_coverage()
    print(f"Schema coverage: {report['coverage_pct']:.1f}%")
    print(f"Missing schemas: {len(report['missing'])}")
    for table in report["missing"][:10]:
        print(f"  - {table}")
```

**Deliverables:**
- [ ] Developer guide
- [ ] Schema audit tool
- [ ] CI integration for coverage tracking

---

## Implementation Schedule

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1-2 | Foundation | Complete Pandera schemas, extend Ibis views |
| 3-4 | Repositories | Migrate storage repositories, serving backends |
| 5-6 | Write Paths | Analytics plugins, ingestion plugins |
| 7-8 | Testing | Property-based tests, view consistency tests |
| 9-10 | Advanced | JSON Schema export, query optimization |
| 11-12 | Documentation | Developer guide, tooling |

---

## Success Metrics

1. **Schema Coverage**: 100% of datasets have Pandera schemas
2. **Ibis Adoption**: 90%+ of read queries use Ibis expressions
3. **Validation Coverage**: All write paths validate with Pandera
4. **Test Coverage**: Property-based tests for 20+ key datasets
5. **Performance**: No regression in query latency (target: &lt;5% increase)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Ibis API changes | Pin ibis version, use stable APIs |
| Performance regression | Benchmark before/after, optimize hot paths |
| Schema validation overhead | Sample validation in production, full in tests |
| Breaking existing tests | Gradual migration with SQL fallbacks |

---

## Dependencies

- `ibis-framework[duckdb]>=9.0.0`
- `pandera>=0.20.0`
- `hypothesis>=6.100.0` (for property tests)

---

## Appendix: Dataset Priority Matrix

### Tier 1 (Critical - Week 1-2)
- `analytics.function_metrics`
- `analytics.function_types`
- `analytics.goid_risk_factors`
- `core.goids`
- `core.goid_crosswalk`
- `graph.call_graph_edges`
- `graph.call_graph_nodes`

### Tier 2 (High - Week 3-4)
- `analytics.function_profile`
- `analytics.file_profile`
- `analytics.module_profile`
- `analytics.hotspots`
- `analytics.typedness`
- `graph.import_graph_edges`
- `graph.import_modules`

### Tier 3 (Medium - Week 5-6)
- `analytics.coverage_functions`
- `analytics.test_catalog`
- `analytics.test_coverage_edges`
- `analytics.subsystems`
- `analytics.subsystem_modules`
- `graph.cfg_blocks`
- `graph.cfg_edges`
- `graph.dfg_edges`
- `graph.symbol_use_edges`

### Tier 4 (Lower Priority - Week 7+)
- All remaining datasets
- All views
