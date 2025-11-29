.PHONY: catalog scaffold-demo contract-docs

catalog:
	@REPO_ROOT=$${REPO_ROOT:-$(PWD)} CODEINTEL_DB_PATH=$${CODEINTEL_DB_PATH:-build/db/db.duckdb} scripts/catalog.sh

scaffold-demo:
	@NAME=$${NAME:-demo_dataset} scripts/scaffold_dataset.sh $$NAME

contract-docs:
	@scripts/ci/contract_docs.sh
