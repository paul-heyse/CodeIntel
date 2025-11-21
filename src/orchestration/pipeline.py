@dataclass
class PipelineContext:
    repo_root: Path
    db_path: Path
    repo: str
    commit: str
    config: PipelineConfig
    logger: logging.Logger

class PipelineStep(Protocol):
    name: str
    deps: list[str]

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        ...

PIPELINE_STEPS: dict[str, PipelineStep] = {
    "scan_repo": RepoScanStep(),
    "ast_cst": AstCstStep(),
    "scip": ScipStep(),
    "goids": GoidBuilderStep(),
    "callgraph": CallGraphStep(),
    "cfg_dfg": CfgDfgStep(),
    "imports": ImportGraphStep(),
    "symbol_uses": SymbolUsesStep(),
    "config_values": ConfigValuesStep(),
    "ast_metrics": AstMetricsStep(),
    "hotspots": HotspotsStep(),
    "typedness": TypednessStep(),
    "static_diagnostics": StaticDiagnosticsStep(),
    "function_metrics": FunctionMetricsStep(),
    "function_types": FunctionTypesStep(),
    "coverage_lines": CoverageLinesStep(),
    "coverage_functions": CoverageFunctionsStep(),
    "test_catalog": TestCatalogStep(),
    "test_edges": TestCoverageEdgesStep(),
    "risk_factors": RiskFactorsStep(),
    "export_docs": DocsExportStep(),
}
