set -e
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu
uv run python -m indicspec.analysis.graft_linmap --draft RedHatAI/Qwen3-8B-speculator.eagle3 --draft-revision 08610ffa01dd9f16731fe8f627b85905b6aa51c4 --out results/tables/graft_linmap_A3.json
uv run python -m indicspec.analysis.graft_linmap --draft SwitchXDDD/multilingual-eagle3-qwen3-8b --draft-revision 72072c4207ab5ee0aa0d65660107a7d52e46c80a --out results/tables/graft_linmap_A4.json
