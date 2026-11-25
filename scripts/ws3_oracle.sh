set -e
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu HF_HUB_OFFLINE=1
A3=$(ls -d ~/.cache/huggingface/hub/models--RedHatAI--Qwen3-8B-speculator.eagle3/snapshots/*)
A4=$(ls -d ~/.cache/huggingface/hub/models--SwitchXDDD--multilingual-eagle3-qwen3-8b/snapshots/*)
uv run python -m indicspec.analysis.oracle_bound --drafter "A3=$A3" --drafter "A4=$A4"
