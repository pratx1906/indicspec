# Engine research: vLLM and SGLang for IndicSpec (WS1, task 1)

Research date: 2026-11-25. Research only: nothing was installed, no model weights were downloaded, and no engine was run.

**Status labels**
- **OPENED**: I read the primary source myself, either a doc page, PyPI JSON, a release page, a model card, or engine source code at the named git tag. For source code, I made shallow clones of `vllm@v0.34.0` (commit `62d4bc3c`) and `sglang@v0.5.23` (commit `297e54fd`) and read the files at the paths cited.
- **EXCERPT-ONLY**: I saw the claim only in a search snippet or a summary, and did not verify it.
- **INFERENCE**: my own conclusion from OPENED sources. It still needs checking against a running install.

Host probe, read-only, run with `wsl -d Ubuntu-24.04` on 2026-11-25 (OPENED, local command output):
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`, compute capability `12.0`, 24463 MiB.
- Driver: KMD `610.88`, CUDA UMD `13.3`.
- Kernel: `6.6.114.1-microsoft-standard-WSL2`.
- Python `3.12.3` (system Python has no pip), `uv 0.13.1`.
- WSL sees 39 GiB RAM and 24 CPUs. 941 GB is free on `/`.

---

## 1. Versions, Python/PyTorch/CUDA requirements, sm_120 support, WSL2 issues

### 1.1 Latest stable releases on PyPI (as of 2026-11-25)

| Package | Latest | Uploaded | requires_python | Key pins | Wheels | Source / status |
|---|---|---|---|---|---|---|
| vllm | **0.34.0** | 2026-11-16 | `<3.15,>=3.11` | `torch==2.14.0`, `flashinfer-python==0.7.3`, `transformers<5.23.0,>=5.16.1`, `xgrammar==0.2.8` | `cp38-abi3-manylinux_2_28_x86_64` | https://pypi.org/pypi/vllm/json (OPENED) |
| vllm (previous) | 0.33.0 | 2026-11-02 | `<3.15,>=3.11` | `torch==2.14.0`, `nvidia-cutlass-dsl[cu13]==4.8.0` | abi3 | https://pypi.org/pypi/vllm/0.33.0/json (OPENED) |
| sglang | **0.5.23** | 2026-10-30 | `>=3.11` | `torch==2.14.1`, `flashinfer_python[cu13]==0.7.2`, `sglang-kernel==0.4.10`, `transformers==5.19.0` | cp311/cp312/cp313 `manylinux_2_34_x86_64` | https://pypi.org/pypi/sglang/json (OPENED) |
| sglang (previous) | 0.5.22 | 2026-10-16 | `>=3.11` | `torch==2.14.1` | cp311–313 | https://pypi.org/pypi/sglang/0.5.22/json (OPENED) |
| torch 2.14.0 / 2.14.1 | n/a | 2026-09-02 / 2026-09-30 | `>=3.10` | Linux deps: `cuda-toolkit[...]==13.0.3`, `nvidia-cudnn-cu13==9.24.0.43`, `nvidia-nccl-cu13==2.30.7`, `triton~=3.8.0` | n/a | https://pypi.org/pypi/torch/2.14.1/json (OPENED) |
| sglang-kernel | 0.4.10 | 2026-10-15 | `>=3.10` | `torch==2.14.1` | `cp310-abi3-manylinux2014_x86_64` | https://pypi.org/pypi/sglang-kernel/0.4.10/json (OPENED) |

Recent release cadence (PyPI upload times, OPENED):
- vLLM: 0.29.0 (09-09), 0.30.0 (09-22), 0.31.0 (10-05), 0.32.0 (10-18), 0.33.0 (11-02), 0.34.0 (11-16).
- SGLang: 0.5.18 (08-21), 0.5.19 (09-04), 0.5.20 (09-18), 0.5.21 (10-01), 0.5.22 (10-16), 0.5.23 (10-30).

**CUDA:**
- The PyPI torch 2.14.x Linux wheels depend on CUDA 13.0 user-space libraries (OPENED, torch JSON above).
- The vLLM 0.34.0 release notes list "CUDA 13.0 (PyPI)" for the default wheel and say "PyTorch stays at 2.14.0" (OPENED: https://github.com/vllm-project/vllm/releases/tag/v0.34.0).
- The SGLang 0.5.23 release notes list "NVIDIA (CUDA 13)" (OPENED: https://github.com/sgl-project/sglang/releases/tag/v0.5.23).
- The host driver reports CUDA UMD 13.3, which is at least 13.0. So **INFERENCE**: the driver is new enough for both default wheels.

### 1.2 Do the default wheels support sm_120?

- **vLLM 0.34.0**
  - Source build: `CMakeLists.txt` lists `12.0` in `CUDA_SUPPORTED_ARCHS` for CUDA ≥ 13.0, and the release Dockerfile sets `torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'` (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/CMakeLists.txt lines 118–134, https://github.com/vllm-project/vllm/blob/v0.34.0/docker/Dockerfile line 277). **INFERENCE**: the PyPI wheel includes sm_120 kernels. Confirm with `python -c "import torch; print(torch.cuda.get_arch_list())"` and a smoke run.
  - Attention backend priority on capability major 12, non-MLA models such as Qwen3: `FLASH_ATTN` → `FLASHINFER` → `TRITON_ATTN` → … (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/platforms/cuda.py around lines 153–175).
  - `FLASH_ATTN` falls back to **FA2** on sm_120. FA3 is chosen only when major == 9 and FA4 only when major == 10 (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/attention/backends/fa_utils.py lines 98–107).
  - The release notes mention Blackwell speedups and SM110 fixes but nothing specific to SM120 or the RTX 50 series (OPENED, release page).
- **SGLang 0.5.23**
  - The AOT kernel CMake adds `-gencode=arch=compute_120a,code=sm_120a` when CUDA ≥ 12.8 (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/kernels/aot/CMakeLists.txt line 222). **INFERENCE**: `sglang-kernel==0.4.10` wheels include sm_120a.
  - Default attention backend for MHA models (Qwen3): FA3 on Hopper, `trtllm_mha` on SM100. The docstring says "trtllm_mha does not support SM120, which will fall back to flashinfer". Otherwise the default is flashinfer, then triton (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/model_override_base.py lines 277–320).
  - **`--attention-backend fa3` hard-fails on sm_120.** It asserts `"FlashAttention v3 Backend requires SM>=80 and SM<=90. Please use --attention-backend flashinfer."` (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/layers/attention/attention_registry.py lines 238–246).
  - The release notes list SM120 items: "fp8_fa_sm120: Use the block-scaled MMA encoding" and "[SM120] quantization: Requantize block-FP8 MoE weights to UE8M0 for DeepGEMM". Neither matters for bf16 Qwen3 (OPENED, release page).
  - The docs on SM120 GDN hybrids say `triton` or `flashinfer` for prefill and `trtllm_mha` for decode only. These are not Qwen3-dense models (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/attention_backend.mdx line 390).

### 1.3 Known sm_120 issues and workarounds

- Community **sm120 field guide** (https://huggingface.co/datasets/witcheer/sm120-field-guide, CC BY 4.0, 80 entries from one desktop RTX 5090 on Ubuntu Server 26.04) (OPENED, as a page summary only; the full entries were not read). It says:
  - torch < 2.7 has no sm_120 kernels, and the error is `no kernel image is available`.
  - NVFP4 in vLLM does not run out of the box.
  - vLLM crashes on a system Python without development headers. **Use uv-managed Python.**
  - The speculative-decoding entries cover "six fixes" and a "/metrics measurement trap". These were not read in detail and remain EXCERPT-ONLY. Read the speculative-decoding file before WS1 bring-up.
  - It has no WSL entries.
- Quantisation and kernel gaps exist on SM120 (FP8 MoE CUTLASS, NVFP4 KV), but none affect bf16 dense Qwen3 (spec refs R-VLLM-43507 and R-VLLM-49011, already OPENED in the spec).
- **DFlash attention on sm_120** is covered in section 2.6.

### 1.4 WSL2-specific issues

- **vLLM pinned memory and UVA: likely blocker unless an environment variable is set.**
  - `CudaPlatform.is_pin_memory_available()` returns `envs.VLLM_WSL2_ENABLE_PIN_MEMORY` (default `0`) under WSL2 kernels ≥ 4.19.121. The code comment says: "Set to 1 when pinned memory or UVA is required (e.g. CPU offloading or v2 model runner)" (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/envs.py lines 2312–2318; https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/platforms/cuda.py lines 344–364).
  - `UvaBuffer` raises `RuntimeError("UVA is not available")` when UVA is unavailable (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/worker/gpu/buffer_utils.py line 42).
  - The 0.34.0 release notes say "Model Runner V1 is gone. Model Runner V2 is now the only GPU runner" and that `VLLM_USE_V2_MODEL_RUNNER=0` "now fails at startup" (OPENED, release page).
  - **INFERENCE**: on WSL2, vLLM 0.34.0 needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. The kernel here is 6.6.114, so the ≥ 4.19.121 gate passes.
- vLLM warns that `--cpu-offload-gb` combined with CUDA graphs on WSL2 hits a hard WDDM pinned-memory limit (OPENED: cuda.py lines 388–396). Do not use CPU offload.
- SGLang docs say: "Under WSL2, pinned memory is capped at about half of system RAM" (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/paged_experts.mdx line 74). WSL sees 39 GiB, so roughly 19 GiB can be pinned.
- NCCL is not relevant on a single GPU (TP = 1). I found no SGLang WSL-specific guidance beyond the note above (OPENED: grep of the v0.5.23 source and docs).
- NVIDIA's CUDA-on-WSL known-limitations page is referenced from vLLM's code (https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications). I did not open that page in this session (EXCERPT-ONLY).

---

## 2. Speculative-decoding CLI syntax per method

### 2.1 vLLM 0.34.0: `--speculative-config` (alias `-sc`) takes a JSON object

`method` values accepted in v0.34.0 (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/config/speculative.py lines 53–97):
- `ngram`, `draft_model`, `suffix`, `eagle`, `eagle3`, `extract_hidden_states`, `dflash`, `dspark`, `ngram_gpu`.
- `mtp` plus model-specific MTP types (`qwen3_next_mtp`, `qwen3_5_mtp`, `deepseek_mtp`, and others).
- Any name registered through the plugin API (section 6).
- `custom_class` was dropped in 0.34.0 (OPENED, release page).

SpeculativeConfig keys (OPENED, same file, lines 449–640):
- General: `method`, `model`, `num_speculative_tokens`, `draft_tensor_parallel_size`, `attention_backend` (drafter attention backend), `kv_cache_dtype`, `max_model_len`, `revision`, `quantization`, `enforce_eager`, `rejection_sample_method` (`standard` | `synthetic` | `block` | `fly`), `num_speculative_tokens_per_batch_size`.
- n-gram: `prompt_lookup_max`, `prompt_lookup_min`.
- Suffix: `suffix_decoding_max_tree_depth` (24), `suffix_decoding_max_cached_requests` (10000), `suffix_decoding_corpus_tokens` (1<<20).
- Other: `use_local_argmax_reduction`, `parallel_drafting` (deprecated), `speculator_options`.

| Arm | vLLM 0.34.0 command | Source / status |
|---|---|---|
| A1 n-gram / prompt lookup | `vllm serve <target> --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5}'`. Both lookup values default to 5 if omitted; if only one is set, the other mirrors it. A separate `ngram_gpu` method also exists. | https://docs.vllm.ai/en/stable/features/speculative_decoding/ (docs link to `v0.34.0`) (OPENED) |
| Suffix decoding | `--speculative-config '{"method": "suffix", "num_speculative_tokens": 8, "suffix_decoding_max_tree_depth": 24, "suffix_decoding_corpus_tokens": 1048576}'`. `suffix_decoding_max_cached_requests=0` disables the cross-request corpus. `suffix_decoding_max_spec_factor` and `suffix_decoding_min_token_prob` are deprecated no-ops. In 0.34.0 suffix decoding runs on MRV2 with a GPU speculator and "Arctic Inference is no longer needed". In 0.33.0 it still required `arctic-inference==0.1.1`. | Docs (OPENED); release notes (OPENED); https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/speculative.py line 1727 (OPENED) |
| A3/A4 EAGLE-3 | `--speculative-config '{"method": "eagle3", "model": "RedHatAI/Qwen3-8B-speculator.eagle3", "num_speculative_tokens": 3}'`. The docs example uses keys `model`, `draft_tensor_parallel_size`, `num_speculative_tokens`, `method`. Speculators-format checkpoints (`speculators_model_type`) are auto-converted, and `method` is filled in from `speculators_model_type`. | https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/ (OPENED); https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/transformers_utils/configs/speculators/base.py lines 120–137 (OPENED) |
| A5 DFlash | `vllm serve Qwen/Qwen3-8B --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 15}' --attention-backend flash_attn --max-num-batched-tokens 32768`. This is the z-lab model card command (it points to vLLM nightly). | https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16 (OPENED) |
| DFlash2 / DSpark | See 2.5. | n/a |
| A6 MTP | `--speculative-config '{"method": "mtp", "num_speculative_tokens": N}'`. `model` can often be omitted. This needs a target with native MTP weights. **Qwen3 dense (4B/8B) has no MTP type in the list**, so A6 needs a different target model. | Docs (OPENED); speculative.py MTPModelTypes (OPENED) |

### 2.2 SGLang 0.5.23

Speculative algorithms:
- `--speculative-algorithm` builtins: `EAGLE`, `EAGLE3`, `NEXTN` (alias of EAGLE), `STANDALONE`, `NGRAM`, `DFLASH`, `DSPARK`, `UNO`, plus names registered by plugins. The enum also contains `FROZEN_KV_MTP`.
- **There is no `SUFFIX` algorithm.**
- Sources (OPENED): https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/spec.py lines 31–34; https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/speculative/spec_info.py lines 32–47; https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/speculative/spec_registry.py line 183.

Related flags (OPENED, spec.py):
- General: `--speculative-draft-model-path` (alias `--speculative-draft-model`), `--speculative-draft-model-revision`, `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens`, `--speculative-dflash-block-size` (alias of num-draft-tokens), `--speculative-draft-window-size`.
- Acceptance and verification: `--speculative-accept-threshold-single`, `--speculative-accept-threshold-acc`, `--speculative-use-rejection-sampling`, `--speculative-use-block-verification`.
- Attention and vocabulary: `--speculative-token-map`, `--speculative-attention-mode {prefill,decode}` (default `prefill`), `--speculative-draft-attention-backend`.
- Other: `--speculative-draft-kv-cache-dtype`, `--speculative-adaptive`, `--speculative-adaptive-config`.
- n-gram: `--speculative-ngram-min-bfs-breadth` (1), `--speculative-ngram-max-bfs-breadth` (10), `--speculative-ngram-match-type {BFS,PROB}` (BFS), `--speculative-ngram-max-trie-depth` (18), `--speculative-ngram-capacity` (10,000,000), `--speculative-ngram-external-corpus-path`, `--speculative-ngram-external-sam-budget`, `--speculative-ngram-external-corpus-max-tokens`.

Allowed draft attention backends: `flashinfer`, `fa3`, `fa4`, `triton`, `ascend`, `trtllm_mha`, `intel_xpu` (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/choices.py lines 101–111).

| Arm | SGLang 0.5.23 command (`python -m sglang.launch_server --model-path <target> …`) | Source / status |
|---|---|---|
| A1 n-gram | `--speculative-algorithm NGRAM --speculative-num-draft-tokens 16 --speculative-ngram-max-bfs-breadth 10`. If num-draft-tokens is omitted it defaults to `min(max_trie_depth, 12)`. CUDA only. It disables the overlap scheduler and mixed chunked prefill. Breadth > 1 with `page_size` > 1 needs `--attention-backend flashinfer`. | https://docs.sglang.io/advanced_features/speculative_decoding.html (OPENED); spec.py (OPENED) |
| Suffix decoding | **Not available as an algorithm.** The closest is NGRAM's trie plus an external-corpus SAM (`--speculative-ngram-external-corpus-path`). | spec.py (OPENED) |
| A3/A4 EAGLE-3 | `--speculative-algorithm EAGLE3 --speculative-draft-model-path <head> --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. The docs example uses topk 4 and 16 draft tokens. The overlap scheduler (on by default) supports only `--speculative-eagle-topk 1`. `--speculative-token-map` is used for pruned-vocab heads that ship a token-map file. | Docs (OPENED) |
| A5 DFlash | `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 [--speculative-num-draft-tokens 16] [--speculative-draft-attention-backend triton\|flashinfer]`. DFLASH forces `speculative_num_steps=1` and `speculative_eagle_topk=1`, and requires `pp_size == 1`. Block size is inferred from the draft config, otherwise 16. | Docs (OPENED); https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/speculative_hook.py lines 321–360 (OPENED) |
| A6 MTP | `--speculative-algorithm EAGLE` (or `NEXTN`) on a model with built-in MTP. The docs example is `XiaomiMiMo/MiMo-7B-RL`. | Docs (OPENED) |

Further SGLang constraints:
- `--speculative-use-rejection-sampling` works only for `EAGLE`, `EAGLE3`, or `NEXTN` with topk = 1. It is rejected for NGRAM, DFLASH, and STANDALONE (OPENED: speculative_hook.py lines 1205–1225).
- z-lab's SGLang command for this card installs SGLang from PR 20547 and uses `--attention-backend fa3`, which fails on sm_120 (see 2.6) (OPENED: z-lab model card).

### 2.3 Extra arms (a): RedHatAI Qwen3 drafters

All four repositories exist, are licensed apache-2.0, and are in the vLLM Speculators format: they have `speculators_model_type` and a nested `transformer_layer_config` (OPENED: `https://huggingface.co/api/models/<id>` and `/resolve/main/config.json`). Revision SHAs are in the table.

| Repo (sha) | Architecture / type | vLLM 0.34.0 | SGLang 0.5.23 |
|---|---|---|---|
| `RedHatAI/Qwen3-4B-speculator.dflash2` (`27aa911d`) | `DFlash2DraftModel`, `speculators_model_type: dflash2`, block_size 9, full vocabulary (151936) | **Supported through method `dflash`.** The speculators loader rewrites `dflash2` to `"dflash"` and keeps the `DFlash2DraftModel` architecture (base.py lines 134–136; algos.py lines 97–117). The model card uses `"method": "dflash2"` and installs vLLM from PR 52816. `"dflash2"` is **not** in the 0.34.0 `SpeculativeMethod` literal. **INFERENCE**: use `{"method": "dflash", "model": "RedHatAI/Qwen3-4B-speculator.dflash2", "num_speculative_tokens": 8}`, or omit `method`. | SGLang has a `DFlash2DraftModel` class (models/dflash.py line 1264). It reads `dflash_config.selector_rank`, while the RedHat config has these at top level with no `dflash_config` block. **INFERENCE**: likely fails to load without config conversion. Untested. |
| `RedHatAI/Qwen3-8B-speculator.dflash2` (`e1c737a1`) | Same as the 4B. The card uses `num_speculative_tokens: 7` and `"method": "dflash2"`. | Same as the 4B. | Same as the 4B. |
| `RedHatAI/Qwen3-4B-speculator.dspark` (`bdffaa5f`) | `DSparkDraftModel`, `speculators_model_type: dspark`, block_size 7, `sample_from_anchor: true` | **Supported: `"method": "dspark"`** (in the method literal; speculators updater `dspark`; Qwen3 DSpark is re-mapped to `Qwen3DSparkModel`, speculative.py lines 788–795). Card: `{"model": …, "num_speculative_tokens": 8, "method": "dspark"}`. | `DSPARK` exists, and `is_dspark_draft` recognises `DSparkDraftModel` (configs/model_config.py line 256). Loading a speculators-format config is unverified. **INFERENCE**: likely needs conversion. |
| `RedHatAI/Qwen3-8B-speculator.peagle` (`9cb73539`) | `PEagleDraftModel`, `speculators_model_type: peagle` | **Not supported.** Both `update_peagle` and the speculators config loader raise `ValueError("P-EAGLE (parallel drafting EAGLE-3) is not supported.")` (algos.py lines 53–55; base.py lines 133–134). | No `PEagle` anywhere in the source. **Not supported.** |

Sources (OPENED):
- https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/transformers_utils/configs/speculators/algos.py
- https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/transformers_utils/configs/speculators/base.py
- https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/dflash.py
- The model cards at https://huggingface.co/RedHatAI/Qwen3-4B-speculator.dflash2, https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash2, https://huggingface.co/RedHatAI/Qwen3-4B-speculator.dspark and https://huggingface.co/RedHatAI/Qwen3-8B-speculator.peagle (README read with `curl`).

### 2.4 Extra arms (b): SuffixDecoding and SSSD

- **SuffixDecoding**
  - vLLM: yes, `method: "suffix"` (section 2.1).
  - SGLang: no algorithm. The NGRAM trie with an external-corpus SAM is the nearest equivalent.
- **SSSD** (arXiv 2411.05894; ACL 2026 long paper 1530)
  - Code: https://github.com/huawei-csl/sssd_speculator, licence BSD-3-Clause-Clear, last push 2026-06-29 (OPENED via GitHub API).
  - Its README says that for SGLang integration, readers should see https://github.com/huawei-csl/sglang-sssd. That repository is a fork of `sgl-project/sglang`, licensed Apache-2.0, last push 2026-05-09 (OPENED via GitHub API and README). The SGLang base version of the fork was not checked.
  - SSSD is **not in either upstream engine** (OPENED: grep of both trees).
  - I found no vLLM integration (EXCERPT-ONLY: search result summary).

### 2.5 Extra arm (c): DictSpec vLLM plugin

- Paper: "Dictionary-Based Speculative Decoding for Non-Latin-Script Languages", UNLP 2026, https://aclanthology.org/2026.unlp-1.15/ (EXCERPT-ONLY: search summary). I could not text-extract the PDF, so the vLLM version and GPU used in the paper are unknown.
- Code: **public.** https://github.com/osyvokon/dictspec, Apache-2.0, last push 2026-11-02 (OPENED via GitHub API). The `vllm-dictspec/` package (v0.1.0) registers under the `vllm.general_plugins` entry point `dictspec = "vllm_dictspec:register"`.
- How it works: it monkey-patches `vllm.v1.worker.gpu_model_runner.GPUModelRunner.__init__` and subclasses `vllm.v1.spec_decode.ngram_proposer.NgramProposer`. It is activated with `--speculative-config '{"method": "ngram", ...}'` plus `DICTSPEC_LOOKUP_PATH=<trie.marisa>`. Optional variables: `DICTSPEC_HYBRID=1`, `DICTSPEC_HYBRID_POLICY` (`dict-first` | `ngram-first` | `longest`), and `DICTSPEC_STATS` (OPENED: https://github.com/osyvokon/dictspec/blob/HEAD/vllm-dictspec/README.md, plus `patch.py` and `proposer.py`).
- **Target vLLM version: not pinned anywhere in the repo.** `vllm` is not even a declared dependency.
  - Both imported modules exist at tags v0.19.1 through v0.33.0 and are **absent in v0.34.0**, where Model Runner V1 was removed (OPENED: raw-file checks per tag, and the v0.34.0 release notes).
  - In 0.33.0 the V1 runner can be selected with `VLLM_USE_V2_MODEL_RUNNER=0` (OPENED: https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/vllm.py lines 748–756).
  - **INFERENCE**: DictSpec needs vLLM ≤ 0.33.0 with `VLLM_USE_V2_MODEL_RUNNER=0`. That makes it a third vLLM venv. It also makes it a good template for the A2 byte-aware drafter on old versions, but it is not portable to 0.34.

### 2.6 Extra item (d): DFlash attention backend on sm_120

- **SGLang**
  - `fa3` (the z-lab card) **cannot work on sm_120**: hard assert SM ≤ 90 (section 1.2).
  - The draft backend resolver works like this: use the user's `--speculative-draft-attention-backend` if it is in `{flashinfer, fa3, fa4, triton, trtllm_mha, …}`. Otherwise use the target's backend. Otherwise fall back to `flashinfer`, or to `triton` on XPU/ROCm (OPENED: speculative_hook.py lines 886–945).
  - The DFLASH verify path has a mask policy for `FlashInferAttnBackend`, `FlashAttentionBackend`, `TritonAttnBackend`, `TRTLLMHAAttnBackend`, and others (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/speculative/dflash_utils.py lines 32–40).
  - The Chutes card ran `--speculative-algorithm DFLASH --speculative-num-draft-tokens 16 --speculative-draft-attention-backend triton` on an RTX PRO 6000 Blackwell (sm_120 workstation part), with "SGLang >= v0.5.13" (benchmarked on 0.5.13.post1) (OPENED: https://huggingface.co/chutesai/Qwen3-32B-FP8-DFLASH).
  - **Recommendation**: on the 5090 Laptop, use `--attention-backend flashinfer` (or `triton`) and `--speculative-draft-attention-backend triton` first, then try `flashinfer`. **Do not use `fa3`.**
- **vLLM**
  - The DFlash speculator requests a non-causal attention backend when the draft config has non-causal layers (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py lines 68–72, 136).
  - `FLASH_ATTN`, `FLASHINFER` and `TRITON_ATTN` all return `supports_non_causal() = True` (OPENED: `vllm/v1/attention/backends/{flash_attn,flashinfer,triton_attn}.py`).
  - On sm_120, `flash_attn` means FA2 (section 1.2), so the z-lab `--attention-backend flash_attn` should be valid. **INFERENCE**: verify by smoke test.
  - Fallbacks: `--attention-backend TRITON_ATTN`, or set only the drafter's backend with the speculative-config key `"attention_backend"`.

---

## 3. Speculative metrics exposure

### 3.1 vLLM 0.34.0

- **Prometheus** (`/metrics`), counters (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/spec_decode/metrics.py lines 177–260):
  - `vllm:spec_decode_num_drafts_total`
  - `vllm:spec_decode_num_draft_tokens_total`
  - `vllm:spec_decode_num_accepted_tokens_total`
  - `vllm:spec_decode_num_accepted_tokens_per_pos_total`, with label `position`
- The documented PromQL formulas are:
  - acceptance rate = `rate(accepted_tokens_total)/rate(draft_tokens_total)`
  - mean acceptance length = `1 + rate(accepted_tokens_total)/rate(num_drafts_total)`
  - per-position rate = `accepted_tokens_per_pos_total / num_drafts_total`
  - (OPENED: metrics.py lines 180–196)
- **Log line** (periodic, when stats logging is on): `SpecDecoding metrics: Mean acceptance length: %.2f, Accepted throughput: %.2f tokens/s, Drafted throughput: %.2f tokens/s, Accepted: %d tokens, Drafted: %d tokens, Per-position acceptance rate: %s, Avg Draft acceptance rate: %.1f%%` (OPENED: metrics.py lines 121–129).
- **Per-request metrics**:
  - Server flag: `--per-request-spec-decode-metrics {none,summary,detailed}`, default `none`.
  - Response location: `metrics.speculative_decoding`, present when the flag is on, speculation is enabled and `n == 1`. Streaming needs `stream_options.include_usage: true` or `--enable-force-include-usage`.
  - Fields: `mean_acceptance_length` (includes the bonus token), `draft_acceptance_rate`, `acceptance_histogram`, `num_spec_steps`, `num_accepted_draft_tokens` (excludes bonus), `num_draft_tokens`, `num_spec_tokens`. With `detailed` you also get `per_step_accepted` and `per_step_drafted`.
  - Sources (OPENED): https://docs.vllm.ai/en/stable/features/speculative_decoding/acceptance_metrics/ and arg_utils.py line 1713.

### 3.2 SGLang 0.5.23

- **Prometheus**: needs `--enable-metrics` (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/observability.py line 70).
  - Gauges: `sglang:spec_accept_length` (accepted drafts + bonus per forward), `sglang:spec_accept_rate` (accepted / proposed in batch), `sglang:spec_cap_length`, `sglang:spec_block_accept_length`, `sglang:spec_num_steps`, `sglang:spec_num_draft_tokens`.
  - Counter: `sglang:spec_verify_calls_total`.
  - The gauges use `multiprocess_mode="mostrecent"`, so they are instantaneous, not cumulative. **Use per-request fields for exact sums.**
  - Source (OPENED): https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/observability/metrics_collector.py lines 508–545, 1866.
- **Per-request** `meta_info` on `/generate` (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/tokenizer_manager.py lines 3150–3210):
  - Fields: `spec_verify_ct`, `spec_accept_length` (= completion_tokens / verify_ct, includes bonus), `spec_accept_rate`, `spec_num_correct_drafts`, `spec_num_proposed_drafts`, `spec_correct_drafts_histogram`, plus the DSpark-only `spec_cap_length`, `spec_block_accept_length` and `spec_cap_lens_histogram`.
  - **Caveat**: `spec_num_proposed_drafts` is computed as `spec_verify_ct * (speculative_num_draft_tokens - 1)`. That is a nominal count, not the real number of tree or n-gram candidates. For NGRAM or topk > 1 it is only an approximation. Use `spec_accept_length` and `spec_verify_ct` as the primary metrics.
- On OpenAI-compatible endpoints, set request field `return_spec_tokens_details: true` to get `sglext.spec_tokens_details` with the same fields (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/entrypoints/openai/protocol.py lines 393, 456–482).
- **Log line** (scheduler decode log): `... accept len: {:.2f}, accept rate: {:.2f}, ...` (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/scheduler_components/metrics_reporter.py line 1005).
- Server-level `/get_server_info` → `internal_states[0].avg_spec_accept_length` is what bench_serving reads (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/benchmark/serving.py lines 1650–1660).

---

## 4. Built-in benchmark tools

### 4.1 `vllm bench serve` (0.34.0)

Source (OPENED): https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/benchmarks/serve.py and https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/benchmarks/datasets/datasets.py.

- **Datasets**: `--dataset-name {sharegpt,burstgpt,random,random-mm,random-rerank,hf,custom,custom_audio,custom_image,prefix_repetition,spec_bench,speed_bench,timed_trace}`, plus `--dataset-path`.
- **Custom dataset**: JSONL `{"prompt": "...", "output_tokens": N}`.
  - `--custom-output-len` (default 256; `-1` uses each row's `output_tokens`), `--skip-chat-template`, `--disable-shuffle`, `--no-oversample`.
  - **The prompt is text.** It is chat-templated unless `--skip-chat-template` is passed, then re-tokenised client-side. There is **no token-ID prompt input.**
- **SPEED-Bench**: `--dataset-name speed_bench --dataset-path <dir> --speed-bench-dataset-subset {qualitative,throughput_1k,throughput_2k,throughput_8k,throughput_16k,…} --speed-bench-category <cat> --speed-bench-output-len N`. The data is prepared with NVIDIA's `prepare.py` from NeMo-Skills (class `SpeedBench`, lines 4577ff).
- **Load control**:
  - `--max-concurrency N`
  - `--request-rate R` (default `inf`; Poisson arrivals when finite)
  - `--burstiness` (1.0 = Poisson; other values give gamma-distributed arrivals)
  - `--ramp-up-strategy`
  - `--num-warmups`
  - `--ignore-eos` (fixed output length when combined with the output-length settings)
  - `--temperature`, `--seed`, `--extra-body`
- **Results**: `--save-result`, `--save-detailed`, `--result-dir`, `--result-filename`, `--metadata`, `--percentile-metrics`, `--metric-percentiles`, `--goodput`.
- **JSON fields include**: `duration`, `completed`, `failed`, `total_input_tokens`, `total_output_tokens`, `request_throughput`, `request_goodput`, `output_throughput`, `total_token_throughput`, `input_lens`, `output_lens`, `ttfts`, `itls`, `latencies`, `start_times`, `generated_texts`, `errors`, `max_output_tokens_per_s`, `max_concurrent_requests`, plus percentile fields.
- **Speculative fields**: the tool scrapes `/metrics` before and after the run and adds `spec_decode_acceptance_rate`, `spec_decode_acceptance_length`, `spec_decode_num_drafts`, `spec_decode_draft_tokens`, `spec_decode_accepted_tokens` and `spec_decode_per_position_acceptance_rates` (serve.py lines 210–262, 1131–1386).

### 4.2 `python -m sglang.benchmark.serving` (0.5.23)

`sglang.bench_serving` still works but is deprecated (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/bench_serving.py and https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/benchmark/serving.py).

- **Datasets**: `--dataset-name {agentic-trace,sharegpt,custom,openai,random,random-ids,generated-shared-prefix,mmmu,image,mooncake,longbench_v2,speed-bench}`.
- **Custom dataset**: JSONL with `conversations` (or `conversation`). Each turn has `content` or `value`; the first user turn becomes the prompt. Output length is fixed with `--sharegpt-output-len`.
- **SPEED-Bench**: `--dataset-name speed-bench --dataset-path <throughput_1k.jsonl> --speed-bench-category {low_entropy,mixed,high_entropy} --speed-bench-output-len 512`. It applies the chat template "to match vllm bench behaviour".
- **Token-ID prompts**: `--tokenize-prompt` sends integer IDs, but only with `--backend sglang`, and it is **asserted off for the custom and sharegpt datasets**. It works for `random` and `generated-shared-prefix`.
- **Load control**:
  - `--max-concurrency`
  - `--request-rate` (default inf; Poisson when finite), `--use-trace-timestamps`
  - `ignore_eos` is **on by default**; `--disable-ignore-eos` turns it off
  - `--temperature`, `--top-p`, `--seed`, `--extra-request-body '{...}'`, `--warmup-requests`, `--flush-cache`
- **Results**: `--output-file`, `--output-details`, `--goodput`.
- **JSON fields**: `backend`, `dataset_name`, `request_rate`, `max_concurrency`, `duration`, `completed`, `total_input_tokens`, `total_output_tokens`, `total_output_tokens_retokenized`, `request_throughput`, `input_throughput`, `output_throughput`, `total_throughput`, `{mean,median,std,p90,p95,p99}_{e2e_latency,ttft,tpot,itl}_ms`, `concurrency`, **`accept_length`** (server-level average from `/get_server_info`), `max_output_tokens_per_s`, `max_concurrent_requests` (lines 1855–1908).

**INFERENCE for D2**:
- Neither built-in tool sends pre-tokenised **custom** prompts to both engines.
- The fair options are: (i) NVIDIA specdec_bench in-process mode (section 7), which passes token IDs to both engines; or (ii) our own async client that posts `prompt: [token ids]` to vLLM `/v1/completions` and `input_ids` to SGLang `/generate`, with `ignore_eos` and `max_tokens` set and per-request spec fields collected.
- Use the built-in tools as cross-checks.

---

## 5. Determinism and batch invariance (greedy-equality tests)

- **vLLM 0.34.0**
  - Setting: `VLLM_BATCH_INVARIANT=1` (beta). Hardware: "NVIDIA GPUs with compute capability 8.0 or higher".
  - Tested models include `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`.
  - On CUDA with tuned matmul tables it runs without `torch.compile` using breakable CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH`).
  - Source (OPENED): https://github.com/vllm-project/vllm/blob/v0.34.0/docs/features/batch_invariance.md (online: https://docs.vllm.ai/en/stable/features/batch_invariance/).
  - Tuned matmul configs include an `"sm120"` family selected when capability.major == 12 (OPENED: https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/model_executor/determinism/batch_invariant_configs.py lines 377, 1227–1238).
  - The spec-decode docs say greedy with speculation is validated to match greedy without it. They also state that vLLM does not guarantee stable logprobs and that batch-size changes can vary output probabilities (OPENED: speculative_decoding docs page).
  - Combining batch-invariant mode with each speculative method is **not documented**. **INFERENCE**: test it per arm.
- **SGLang 0.5.23**
  - Setting: `--enable-deterministic-inference` ("Enable deterministic inference mode with batch invariant ops"; OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/exec_.py line 1153).
  - Supported attention backends: `flashinfer`, `fa3`, `fa4`, `triton` (+ `ascend`, `intel_xpu`).
  - On sm_100 and sm_120 with no backend set, it auto-falls back to `flashinfer` (to `triton` for DeepSeek) (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/overrides.py lines 1093–1124).
  - The docs table says FlashInfer deterministic mode has **no radix cache**, while Triton supports CUDA graph, chunked prefill, radix cache and temperature > 0 (OPENED: https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/deterministic_inference.mdx).
  - **INFERENCE**: on sm_120 use `--attention-backend triton --enable-deterministic-inference`, or flashinfer with `--disable-radix-cache`.
  - UNO rejects deterministic mode (speculative_hook.py line 461).
  - The 0.5.23 release notes list "Seed rejection-sampling draft proposals for deterministic inference: #33395" (OPENED).

---

## 6. Custom drafter plug-in hooks (for A2: byte-aware n-gram)

- **vLLM 0.34.0: experimental "Custom Speculators" API.**
  - Call `register_speculative_method(name, config_cls)` from a `vllm.general_plugins` entry point.
  - The config class subclasses `SpeculatorConfig`; `speculator_class` names the class.
  - The speculator class subclasses `vllm.v1.worker.gpu.spec_decode.speculator.BaseSpeculator`. It is constructed as `(vllm_config, device)` and implements `propose(input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, …) -> torch.Tensor`, plus `init_cudagraph_manager` and `capture`.
  - Options are passed through `"speculator_options"` and the plugin is selected with `"method": "<name>"`.
  - "This is an experimental interface that may change between releases."
  - Built-in model-free speculators to copy from: `v1/worker/gpu/spec_decode/ngram/` and `.../suffix/`.
  - Sources (OPENED): https://docs.vllm.ai/en/stable/features/speculative_decoding/ (Custom Speculators section); https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/worker/gpu/spec_decode/speculator_config.py line 317; https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/worker/gpu/spec_decode/speculator.py lines 68–122; entry-point group `vllm.general_plugins` in vllm/plugins/__init__.py line 18.
  - The older V1 `NgramProposer` monkey-patch route (used by DictSpec) exists only in ≤ 0.33.0 (section 2.5).
- **SGLang 0.5.23**
  - Use the `@SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=..., validate_server_args=..., spec_class=...)` decorator on a factory that returns a worker class. Plugins load through the setuptools entry point group `sglang.srt.plugins`, which can be restricted with `SGLANG_PLUGINS`.
  - `supports_overlap=False` is deprecated: "the spec V1 worker path has been removed".
  - The built-in model-free reference is `speculative/ngram_worker.py` with the C++ trie in `speculative/cpp_ngram/`.
  - A no-code alternative for a corpus-level dictionary: `--speculative-ngram-external-corpus-path <jsonl>`, plus `POST /add_external_corpus` at runtime.
  - Sources (OPENED): https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/speculative/spec_info.py lines 69–90; https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/speculative/spec_registry.py; https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/plugins/__init__.py; spec.py lines 272–283.

---

## 7. NVIDIA SPEED-Bench harness

- **Dataset**: https://huggingface.co/datasets/nvidia/SPEED-Bench, licence "nvidia-evaluation-dataset-license" (NVIDIA Evaluation Dataset License Agreement; not OSI).
  - Configs: `qualitative` (880 rows), `throughput_1k/2k/8k/16k/32k` (+100k in YAML).
  - Qualitative categories include `multilingual`. Throughput categories: `high_entropy`, `mixed`, `low_entropy`.
  - Some rows are masked and must be rebuilt with the harness's `prepare_data.py`.
  - (OPENED: dataset README.) Paper: arXiv 2604.09557, ICML 2026 (EXCERPT-ONLY).
- **Harness**: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench. The repository licence is **Apache-2.0** (OPENED via GitHub API; repo pushed 2026-11-25).
  - Files: `run.py`, `prepare_data.py` (`python3 prepare_data.py --dataset speed --config all`), `specdec_bench/models/{vllm,sglang,trtllm_torch_api,auto_deploy,...}.py`.
  - Each run writes `configuration.json`, `timing.json` and `acceptance_rate.json`.
  - The README describes the Docker images `vllm/vllm-openai:v0.11.0` and `lmsysorg/sglang:v0.5.4.post2`, which are old relative to our pins (OPENED: README).
  - **How it runs engines**: in-process, not through HTTP.
    - vLLM uses `AsyncLLM` with `TokensPrompt(prompt_token_ids=...)`.
    - SGLang uses `sgl.Engine(skip_tokenizer_init=True, ...)` with `input_ids=prompt_ids`.
    - So prompts are pre-tokenised and identical across engines (OPENED: `specdec_bench/models/vllm.py` line 213 and `sglang.py` lines 52–60, 123 on `main`).
  - CLI flags (`run.py`): `--engine`, `--model_dir`, `--tokenizer`, `--draft_model_dir`, `--speculative_algorithm {EAGLE3,EAGLE,DRAFT_TARGET,NGRAM,MTP,DFLASH,DSPARK,NONE}`, `--draft_length`, `--dataset speed --dataset_path data/speed/<config>`, `--concurrency`, `--num_requests`, `--output_length`, `--ignore_eos`, `--temperature`, `--max_seq_len`, `--tp_size`, `--save_dir`, `--runtime_params`.
  - The README notes that requests are sent single-threaded, so concurrency above 256 can skew results, and suggests server mode plus NVIDIA AIPerf for heavy load.
- **Compatibility issues found in the harness's `main` branch (OPENED: source; INFERENCE: effect)**:
  - The SGLang wrapper maps `NGRAM` to `speculative_algorithm="LOOKAHEAD"`. SGLang 0.5.23 has no `LOOKAHEAD` builtin, and `from_string` raises `Unknown speculative algorithm name`. **Patch it to `NGRAM` locally.**
  - The SGLang DFLASH path may pass `speculative_dflash_draft_window_size`. The 0.5.23 field is named `speculative_draft_window_size`.
  - The vLLM NGRAM mapping sets only `prompt_lookup_max` (default 3), with the comment "No idea here".
  - **Plan**: vendor `specdec_bench` at a pinned commit, apply small documented patches, and add an IndicSpec dataset adapter (same `turns`/`messages` schema), or write our own client (section 4).

---

## Recommended pins and install plan

The plan uses separate uv venvs under `~/.cache/indicspec/` inside WSL, with Python 3.12 managed by uv (not system Python). The field guide warns about crashes with system Python lacking headers.

| venv | Pin (try first) | Fallback | Why |
|---|---|---|---|
| `~/.cache/indicspec/venv-vllm` | **vllm==0.34.0** (torch 2.14.0 cu13, flashinfer-python 0.7.3) | **vllm==0.33.0** | 0.34.0 is the latest stable release. It has DFlash (method `dflash`, which also loads RedHat DFlash2 checkpoints), DSpark, `suffix` without arctic-inference, per-request spec metrics, the `register_speculative_method` plugin API, and sm120 batch-invariant matmul tables. Risk: MRV2 is the only runner, and on WSL it needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. 0.33.0 still has the V1 runner (`VLLM_USE_V2_MODEL_RUNNER=0`) if MRV2 or UVA fails on WSL, but its `suffix` needs `arctic-inference==0.1.1`. |
| `~/.cache/indicspec/venv-vllm-v1` (optional) | **vllm==0.33.0** | an older 0.3x | Only for the DictSpec plugin (V1 `NgramProposer` monkey-patch) as an A2 comparison, with `VLLM_USE_V2_MODEL_RUNNER=0`. DictSpec pins no vLLM version, so record the exact one used. |
| `~/.cache/indicspec/venv-sglang` | **sglang==0.5.23** (torch 2.14.1 cu13, sglang-kernel 0.4.10, flashinfer 0.7.2) | **sglang==0.5.22**, then **0.5.13.post1** (the Chutes sm_120 DFlash datapoint) | 0.5.23 is the latest stable release. It has DFLASH/DSPARK, deterministic mode with sm_120 auto-fallback, per-request `spec_verify_ct` and accept length, `bench serving --dataset-name speed-bench`, and the algorithm plugin API. |

Install sketch for WS0/WS1. Not executed. Run inside WSL after owner approval; this is roughly 10+ GB of wheels and below the 10 GB-per-step stop only per package.

```bash
uv venv --python 3.12 ~/.cache/indicspec/venv-vllm
VIRTUAL_ENV=~/.cache/indicspec/venv-vllm uv pip install "vllm==0.34.0"
uv venv --python 3.12 ~/.cache/indicspec/venv-sglang
VIRTUAL_ENV=~/.cache/indicspec/venv-sglang uv pip install "sglang==0.5.23"
# verify: python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"
```

Runtime defaults to try first on sm_120 and WSL2:
- **vLLM**: `export VLLM_WSL2_ENABLE_PIN_MEMORY=1`. Keep the default `--attention-backend` (FLASH_ATTN = FA2 on sm_120), or set `flash_attn` explicitly for DFlash. Fallback: `TRITON_ATTN`. Add `--per-request-spec-decode-metrics detailed`. For equality tests, add `VLLM_BATCH_INVARIANT=1`.
- **SGLang**: `--attention-backend flashinfer` (never `fa3`). For DFlash add `--speculative-draft-attention-backend triton` (Chutes config), with `flashinfer` as the alternative. Add `--enable-metrics`. For equality tests, add `--enable-deterministic-inference --attention-backend triton`.
- **Arm matrix expectations**:

  | Arm | vLLM | SGLang |
  |---|---|---|
  | A1 n-gram | ✓ | ✓ |
  | Suffix | ✓ | ✗ (nearest: NGRAM + external corpus) |
  | A3/A4 EAGLE-3 | ✓ | ✓ |
  | A5 DFlash (z-lab) | ✓ | ✓ (non-fa3 backend) |
  | RedHat DFlash2 | ✓ (method `dflash`) | likely needs config conversion |
  | RedHat DSpark | ✓ (method `dspark`) | likely needs config conversion |
  | RedHat P-EAGLE | ✗ in 0.34.0 | ✗ |
  | A6 MTP | not applicable to Qwen3 dense | not applicable to Qwen3 dense |
  | SSSD | fork only (huawei-csl/sglang-sssd) | fork only (huawei-csl/sglang-sssd) |

  Confirm every cell with a smoke test and record it in `docs/ENGINE_STATUS.md`.
