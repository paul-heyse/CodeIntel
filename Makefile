.PHONY: catalog scaffold-demo contract-docs docs docs-fast docs-serve docs-diagrams docs-summary

catalog:
	@REPO_ROOT=$${REPO_ROOT:-$(PWD)} CODEINTEL_DB_PATH=$${CODEINTEL_DB_PATH:-build/db/codeintel.duckdb} scripts/catalog.sh

scaffold-demo:
	@NAME=$${NAME:-demo_dataset} scripts/scaffold_dataset.sh $$NAME

contract-docs:
	@scripts/ci/contract_docs.sh

docs-diagrams:
	@uv run python mkdocs_gen/gen_arch_diagrams.py

docs:
	@uv run python mkdocs_gen/build_docs.py

docs-fast:
	@uv run python mkdocs_gen/build_docs.py --skip-diagrams

docs-serve:
	@uv run mkdocs serve -f config/mkdocs.yml -a localhost:8000

docs-summary:
	@uv run python mkdocs_gen/build_single_markdown.py
