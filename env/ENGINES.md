# Engine environments

vLLM and SGLang are installed in **separate** uv virtual environments inside WSL2 Ubuntu 24.04, on the ext4
filesystem (not `/mnt/c`), because their dependency pins conflict (PROJECT_BRIEF §6.2):

| Env | Path (WSL) | Version | Lock / requirements |
| --- | --- | --- | --- |
| CPU analysis | `~/.cache/indicspec/venv-cpu` (via `UV_PROJECT_ENVIRONMENT`, see Makefile) | — | `uv.lock` |
| vLLM | `~/.cache/indicspec/venv-vllm` | TBD (WS1) | `env/vllm.requirements.lock` (TBD) |
| SGLang | `~/.cache/indicspec/venv-sglang` | TBD (WS1) | `env/sglang.requirements.lock` (TBD) |

External paths used by this project (spec §0 rule 7):

* `~/.cache/indicspec/` (WSL) — virtual environments
* `~/.cache/huggingface/` (WSL) and `%USERPROFILE%\.cache\huggingface` (Windows) — HF hub cache (tokenizers, models)
* `%LOCALAPPDATA%\uv\cache`, `~/.cache/uv` — uv package caches
* `D:\indicspec-data\` (`/mnt/d/indicspec-data` in WSL) — datasets: `raw/` (junctioned from `data/raw`), `generated/` (D-012)
