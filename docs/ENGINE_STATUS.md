# Engine status matrix (arm × engine × version)

Each cell is either **works** (with the smoke-test command) or **fails** (exact command, version, error,
upstream issue links). Nothing is filled in until it has been run on this machine.

Hardware: NVIDIA GeForce RTX 5090 Laptop GPU (24 GB, `sm_120`), driver 610.88, WSL2 Ubuntu 24.04
(`env/probe.json`).

| Arm | Description | vLLM (version TBD) | SGLang (version TBD) |
| --- | --- | --- | --- |
| A0 | no speculation | not yet tried | not yet tried |
| A1 | engine n-gram / prompt lookup | not yet tried | not yet tried |
| A2 | byte-aware n-gram (ours) | not built | not built |
| A3 | `RedHatAI/Qwen3-8B-speculator.eagle3` | not yet tried | not yet tried |
| A4 | `SwitchXDDD/multilingual-eagle3-qwen3-8b` | not yet tried | not yet tried |
| A5 | `z-lab/Qwen3-{8B,4B}-DFlash-b16` | not yet tried | not yet tried |
| A6 | native MTP head (if any fits 24 GB) | not yet searched | not yet searched |
| A8 | stronger public drafter (if found) | — | — |

## Known incompatibilities (investigated, not generated)

- **A3 on SGLang 0.5.23 — unsupported as published.** `RedHatAI/Qwen3-8B-speculator.eagle3` is in the
  vLLM *speculators* format (`architectures: ["Eagle3Speculator"]`, no top-level `model_type`), so SGLang
  fails with `ValueError: Unrecognized model ... Should have a model_type key`
  (`results/raw/smoke/smoke-sglang-A3-qwen3-8b-20261125T185252`). Tensor names are convertible
  (SGLang maps `midlayer` → `layers.0`), **but** the config sets `norm_before_residual: true`, which
  vLLM implements (`vllm/model_executor/models/llama_eagle3.py`, `_norm_before_residual`: the residual
  carries the *normalised* hidden state) and SGLang does not (no occurrence of `norm_before_residual`
  in `sglang/` 0.5.23). A converted checkpoint would therefore run a different function than the one
  trained — still lossless, but with artificially lower acceptance. We do not convert it; A3 is
  measured on vLLM only. Patching SGLang is ask-first (spec §13.1.4).

## Smoke results (generated)

<!-- BEGIN GENERATED -->
| Engine | Arm | Model | Status | τ pooled (smoke) | τ by language (smoke) | Peak VRAM MiB | Latest run / error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sglang | A0 | qwen3-4b | works |  |  | 22382.0 | `smoke-sglang-A0-qwen3-4b-20261125T192046` |
| sglang | A0 | qwen3-8b | works |  |  | 22569.0 | `smoke-sglang-A0-qwen3-8b-20261125T192443` |
| sglang | A1 | qwen3-4b | works |  |  | 22517.0 | `smoke-sglang-A1-qwen3-4b-20261125T192207` |
| sglang | A3 | qwen3-8b | **fails** |  |  |  | `smoke-sglang-A3-qwen3-8b-20261125T185252` — ValueError: Unrecognized model in /home/pratyush/.cache/huggingface/hub/models--RedHatAI--Qwen3-8B-speculator.eagle3/snapshots/08610ffa01dd9f16731fe8f627b85905b6aa51c4. Should have a `model_type` key in its config.json. |
| sglang | A4 | qwen3-8b | works |  |  | 23935.0 | `smoke-sglang-A4-qwen3-8b-20261125T185307` |
| sglang | A5 | qwen3-4b | works |  |  | 23945.0 | `smoke-sglang-A5-qwen3-4b-20261125T192325` |
| sglang | A5 | qwen3-8b | works |  |  | 23837.0 | `smoke-sglang-A5-qwen3-8b-20261125T192641` |
| vllm | A0 | qwen3-4b | works |  |  | 22128.0 | `smoke-vllm-A0-qwen3-4b-20261125T183159` |
| vllm | A0 | qwen3-8b | works |  |  | 21627.0 | `smoke-vllm-A0-qwen3-8b-20261125T183625` |
| vllm | A1 | qwen3-4b | works | 1.40 |  | 21662.0 | `smoke-vllm-A1-qwen3-4b-20261125T183317` |
| vllm | A3 | qwen3-8b | works | 1.19 | en:2.82 hi:1.33 kn:1.02 ml:1.05 ta:1.07 te:1.03 | 20996.0 | `smoke-vllm-A3-qwen3-8b-20261125T191503` |
| vllm | A4 | qwen3-8b | works | 1.24 | en:1.50 hi:1.86 kn:1.09 ml:1.11 ta:1.05 te:1.19 | 21466.0 | `smoke-vllm-A4-qwen3-8b-20261125T191718` |
| vllm | A5 | qwen3-4b | works | 1.69 |  | 21159.0 | `smoke-vllm-A5-qwen3-4b-20261125T183444` |
| vllm | A5 | qwen3-8b | works | 1.46 | en:2.38 hi:1.57 kn:1.64 ml:1.15 ta:1.34 te:1.29 | 21122.0 | `smoke-vllm-A5-qwen3-8b-20261125T191854` |

Earlier failed attempts (kept for the record):

- `smoke-sglang-A0-qwen3-4b-20261125T184140`: No such file or directory: 'ninja'
- `smoke-sglang-A0-qwen3-4b-20261125T191210`: see status.json
- `smoke-sglang-A0-qwen3-8b-20261125T184613`: see status.json
- `smoke-sglang-A1-qwen3-4b-20261125T184229`: No such file or directory: 'ninja'
- `smoke-sglang-A5-qwen3-4b-20261125T184258`: see status.json
- `smoke-sglang-A5-qwen3-8b-20261125T185522`: see status.json
- `smoke-vllm-A0-qwen3-4b-20261125T182754`: KeyError('clocks.sm')
- `smoke-vllm-A3-qwen3-8b-20261125T183806`: ValueError: To serve at least one request with the model's max seq len (4096), (0.58 GiB KV cache is needed, which is larger than the available KV cache memory (0.35 GiB). Based on the available memory, the estimated maximum model length is 2496. The KV cache gets what is left after the model, its a
- `smoke-vllm-A5-qwen3-8b-20261125T183956`: ValueError: To serve at least one request with the model's max seq len (4096), (0.64 GiB KV cache is needed, which is larger than the available KV cache memory (0.23 GiB). Based on the available memory, the estimated maximum model length is 1472. The KV cache gets what is left after the model, its a

Smoke τ values come from 6 short prompts (one per language, 64 tokens) and are **not** results; they only show the speculative path is active.
<!-- END GENERATED -->
