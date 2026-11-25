set -e
cd "/mnt/c/Users/Pratyush Dixit/Desktop/Inference"
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/indicspec/venv-cpu
uv sync -q --extra data
grep DONE ~/.cache/indicspec/logs/download.log | awk '{print $2, $3}'
A3=$(ls -d ~/.cache/huggingface/hub/models--RedHatAI--Qwen3-8B-speculator.eagle3/snapshots/*)
A4=$(ls -d ~/.cache/huggingface/hub/models--SwitchXDDD--multilingual-eagle3-qwen3-8b/snapshots/*)
echo A3=$A3; echo A4=$A4
uv run python -m indicspec.analysis.graft_check --draft RedHatAI/Qwen3-8B-speculator.eagle3 --draft-revision 08610ffa01dd9f16731fe8f627b85905b6aa51c4 --out results/tables/graft_alignment_A3.json
uv run python -m indicspec.analysis.graft_check --draft SwitchXDDD/multilingual-eagle3-qwen3-8b --draft-revision "$(basename $A4)" --out results/tables/graft_alignment_A4.json
uv run python -m indicspec.analysis.tokentax --drafter "A3=$A3" --drafter "A4=$A4"
