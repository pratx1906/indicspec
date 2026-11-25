#!/usr/bin/env bash
# Install pinned engine environments inside WSL (PLAN P1, D-001, D-011).
# Usage: bash env/install_engines.sh [vllm|sglang|all]
set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.23}"
ROOT="${HOME}/.cache/indicspec"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT"
install_one() {
  local name="$1" spec="$2" venv="$ROOT/venv-$1"
  echo "== $name: $spec -> $venv"
  uv venv --python 3.12 "$venv"
  VIRTUAL_ENV="$venv" uv pip install "$spec"
  VIRTUAL_ENV="$venv" uv pip freeze > "$REPO/env/$name.requirements.lock"
  "$venv/bin/python" - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("arch_list", torch.cuda.get_arch_list())
PY
}
case "${1:-all}" in
  vllm) install_one vllm "vllm==${VLLM_VERSION}" ;;
  sglang) install_one sglang "sglang==${SGLANG_VERSION}" ;;
  all) install_one vllm "vllm==${VLLM_VERSION}"; install_one sglang "sglang==${SGLANG_VERSION}" ;;
esac
