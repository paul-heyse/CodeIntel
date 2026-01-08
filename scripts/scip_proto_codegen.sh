#!/usr/bin/env bash
# scripts/scip_proto_codegen.sh
# Purpose: Locally generate scip_pb2.py using grpc_tools.protoc.

set -Eeuo pipefail

if [ ! -f "pyproject.toml" ]; then
  echo "Run scripts/scip_proto_codegen.sh from the repository root (pyproject.toml not found)." >&2
  exit 1
fi

repo_root="$(pwd)"
proto_path="${repo_root}/src/codeintel/ingestion/scip/proto/scip.proto"
proto_dir="$(dirname "${proto_path}")"
out_dir="${1:-"${repo_root}/build/scip/proto"}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "${out_dir}"
"${python_bin}" -m grpc_tools.protoc -I "${proto_dir}" --python_out "${out_dir}" \
  --pyi_out "${out_dir}" "${proto_path}"

echo "Generated ${out_dir}/scip_pb2.py and ${out_dir}/scip_pb2.pyi"
