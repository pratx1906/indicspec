# Decisions log

Short entries: decision · alternatives · reason · evidence. Newest last.

### D-001 (2026-11-25) Repository location and execution split
- **Decision:** the repository lives in the owner's Windows folder (`Desktop\Inference`); GPU engines
  (vLLM, SGLang) run inside WSL2 Ubuntu 24.04, reading the repo through `/mnt/c`. Engine and CPU
  venvs live on the WSL ext4 filesystem under `~/.cache/indicspec/`.
- **Alternatives:** repo inside WSL home (faster I/O, but the owner asked for this folder); native
  Windows engines (vLLM and SGLang do not support native Windows).
- **Reason:** owner's instruction + engines are Linux-only. Venvs on ext4 avoid slow `/mnt/c` I/O.
- **Evidence:** `env/probe.json`; `wsl -l -v` shows Ubuntu-24.04 on WSL 2; `nvidia-smi` in WSL sees the GPU
  with compute capability 12.0.

### D-002 (2026-11-25) Makefile runs under WSL/Linux
- **Decision:** `make` targets assume GNU make + bash (WSL/Linux/CI). Windows users can run the same
  commands with `uv run …` directly (documented in README).
- **Alternatives:** a Python task runner (`nox`, `invoke`); `make` for Windows.
- **Reason:** the spec names a Makefile; `make` is present in WSL and on CI runners, absent on Windows.

### D-003 (2026-11-25) CPU environment manager
- **Decision:** `uv` with `pyproject.toml` + `uv.lock`; Python ≥3.11 (host has 3.12 on both Windows and WSL).
  Engine environments are separate (D-001) and get their own lock/requirements files under `env/`.
- **Reason:** spec §6.2 asks for a lock-file-based manager; uv 0.13.1 is installed on both sides.

### D-004 (2026-11-25) Local git identity
- **Decision:** repo-local `user.name`/`user.email` set for the owner; no remote configured. Nothing is
  pushed without the owner's approval (spec §0 rule 6).

### D-005 (2026-11-25) Licence deferred
- **Decision:** no `LICENSE` file yet; the owner chooses after the dependency/data licence review (spec §6.1, §13.1.3).

### D-006 (2026-11-25) Token-class scheme refined
- **Decision:** classes are those of the spec plus `BYTE_INVALID` (bytes 0xC0, 0xC1, 0xF5–0xFF and other
  malformed sequences that never occur in well-formed UTF-8). A token that starts with continuation bytes
  is `BYTE_FRAGMENT_CONT` even if it also ends incomplete (flag `ends_incomplete` keeps that). `CHAR` /
  `MULTI_CHAR` count Unicode scalars after stripping whitespace (a leading BPE space does not make a
  token multi-character); combining vowel signs are separate scalars, as the spec's definition requires.
  Tokens whose non-whitespace scalars are all ASCII or Unicode punctuation/symbols (P*, S*) are
  `ASCII_OR_PUNCT` (so the Devanagari danda `।` is punctuation, not `CHAR`).
- **Alternatives:** grapheme-cluster-based CHAR (closer to what readers perceive as a "letter"); kept as a
  possible secondary analysis.
- **Reason:** single-byte tokens 0xF5–0xFF exist in any byte-level vocabulary but can never be generated
  in valid text; folding them into fragment classes would bias per-class rates.
- **Evidence:** `src/indicspec/tokens/classes.py`, tests in `tests/unit/test_token_classes.py`.

### D-007 (2026-11-25) Scope after novelty re-check — pending owner confirmation (G0 / §13.1.2)
- **Finding:** [R-PHRASE] (arXiv 2611.12464, 9 Nov 2026) reports speculative speed-ups on Qwen3-4B for
  Telugu, Kannada and Hindi (HF `generate()`, A5000/T4, a new phrase-level method). [R-SARVAM-EAGLE] is a
  public EAGLE-3 head trained on all four Dravidian languages, but for Sarvam-30B. [R-DICTSPEC] is an
  n-gram dictionary drafter for high-fertility scripts (Ukrainian/Crimean Tatar). See [NOVELTY_LOG.md](NOVELTY_LOG.md).
- **Proposed differentiation (unchanged core):** comparison of *existing* drafter families (pruned-vocab
  EAGLE-3, full-vocab DFlash, n-gram) in production engines under load, on all four Dravidian languages +
  Hindi + English with parallel content, plus the byte-fragment mechanism analysis. None of the found work
  covers that combination (as of 2026-11-25, search-based). Claims of "first" are dropped; R-PHRASE and
  R-DICTSPEC are cited and, where feasible, R-DICTSPEC becomes a comparison arm for A2.
- **Status:** awaiting owner confirmation before WS1 engine bring-up spends GPU time.

### D-008 (2026-11-25) FLORES leakage for arm A4
- **Decision:** the SwitchXDDD head (A4) self-reports training on FLORES-200 data (C27). A4 results on
  FLORES-derived prompts are reported separately and flagged; A4's headline comparison uses non-FLORES
  tasks (XQuAD-IN, Belebele, Aya, IN22).
- **Reason:** train/test overlap inflates acceptance.

### D-009 (2026-11-25) Initial data-source plan (from [research/data_sources.md](research/data_sources.md))
- Translation: FLORES devtest via `google/IndicGenBench_flores_in` (CC-BY-SA-4.0, ungated).
- QA: `google/IndicGenBench_xquad_in` (CC-BY-SA-4.0, human-translated, parallel).
- Summarisation: passages of `facebook/belebele` (CC-BY-SA-4.0, parallel; card: test use only).
- Chat: `CohereLabs/aya_evaluation_suite` `dolly_machine_translated` (Apache-2.0; NLLB-translated → translationese risk, flagged).
- Held-out domain: `ai4bharat/IN22-Conv` (CC-BY-4.0, **gated** → needs owner's HF token).
- Policy: ship builder scripts + checksummed MANIFEST, not raw text. Licences to be re-verified when builders run.

### D-010 (2026-11-25) Owner feedback on candidate additions N1–N4
- **Adopted:** (a) coverage-split acceptance (in- vs out-of-draft-vocab positions) is the primary,
  confirmatory mechanism test; out-of-vocab acceptance is 0 by construction, so only in-vocab acceptance
  across languages is informative. (b) Vocabulary grafting (N1) is exploratory only: EAGLE-3's `lm_head`
  is its own tensor after its own norm (A3 checkpoint has `lm_head [32000, 4096]`), so copied target rows
  may not align; a row-alignment pre-check runs first. (c) UTF-8 mask (N2) is replay-first; engine
  patching remains ask-first. (d) Per-character and words-per-KV metrics adopted; energy adopted after the
  counter was verified (C30). (e) #4/#5 folded into WS1/WS4.
- **Partly disagreed:** "for DFlash a mask can only truncate". DFlash produces logits for every block
  position in one pass, unconditioned on the drafted neighbour, so after picking position i−1 the mask
  can re-pick the best *valid* token at position i. That can add accepted tokens, not only save verify
  work. Both variants will be simulated; the replay decides.
- **Clarification:** the "32,000 of 151,936 undraftable" statement came from a search excerpt of
  R-PHRASE (arXiv 2611.12464, Qwen3-4B). It agrees with our own reading of the A3/A4 configs (C26).
- **Evidence:** this file; [PLAN.md](PLAN.md); C30–C31.

### D-011 (2026-11-25) Owner approval: model downloads and engine installs
- **Decision:** the owner gave explicit approval (chat, 2026-11-25) to download Qwen3-8B (~16 GB, above the
  10 GB single-step limit of spec §0 rule 6) and Qwen3-4B, and to install the vLLM and SGLang environments.
- **Where:** WSL HF cache `~/.cache/huggingface/hub`, at the revisions pinned in `configs/models.yaml`.

### D-012 (2026-11-25) Datasets stored on D:\ (owner instruction)
- **Decision:** passive dataset storage lives on `D:\indicspec-data` (WSL: `/mnt/d/indicspec-data`):
  `raw/` for downloaded source datasets (the repo's `data/raw` becomes a directory junction to it) and
  `generated/` for large generated data (target generations, hidden states for any WS5b training).
  Small frozen artefacts (`data/MANIFEST.json`, builders) stay in the repo; `data/bench/*.jsonl` stays
  in the repo folder (gitignored, small).
- **Not moved:** model weights stay in the WSL ext4 HF cache, because serving reads them at load time
  and `/mnt/*` I/O from WSL is slow. Engine venvs stay on ext4 for the same reason.
- **Reason:** owner instruction; D: has ~485 GB free.

### D-013 (2026-11-25) WS2 benchmark builders: what the real files required
- **Decision:** `python -m indicspec.data.build` builds `data/bench/{translation,translation_xx2en,qa,
  summarisation,chat}.jsonl` from four ungated sources pinned in `configs/data_sources.yaml`
  (Flores-IN `f8650438`, XQuAD-IN `32026f0b`, Belebele `7899cdfa`, Aya eval suite `e69f652e`).
  `data/bench/*.jsonl` is gitignored; `data/MANIFEST.json` (sha256, counts, source revisions, raw-file
  sha256, licences, tokenizer hashes) is committed. `--verify` rebuilds and compares checksums.
- **Deviations from the D-009 / research plan, found by inspecting the downloaded files:**
  1. Flores-IN files have **no row ids** and each `flores_{a}_{b}_test.json` has its own order. Rows are
     aligned by the whitespace-normalised English sentence (1012 unique, identical across all ten files).
     `en→xx` targets and `xx→en` sources differ only in whitespace (0 mismatches after normalisation).
  2. Translation items are not single sentences: devtest sentences are grouped by the Belebele passage
     that contains them (coherent 1–6 sentence text, longer outputs): 220 items, 767 of 1012 sentences;
     the 245 sentences in no Belebele passage are unused. Row `lang` = output language: Indic cells are
     en→L; the English cell is L→en with the source language rotated over items (44 each). Both directions:
     a second task `translation_xx2en` (5 Indic langs, same item ids and splits).
  3. Belebele passages come from FLORES dev (267) or devtest (220); one matched neither and is excluded.
     Summarisation uses only the **dev** passages, so it shares no sentence with translation (tested).
  4. Belebele `data/README` names a `split` column that the HF jsonl files lack; `link` is unique per
     passage in our six files and is used as the key.
  5. Aya `dolly_machine_translated` uses ISO 639-3 codes (`eng, hin, tam, tel, kan, mal`) plus a `script`
     column, not FLORES codes; 200 ids per language, `source_id` identical across languages.
  6. XQuAD-IN: test file only (1190 questions, 240 passages, ids identical in all six). One question per
     passage (seeded) → 240 items; prompt asks for the answer plus a 2–4 sentence explanation.
  7. **NFC:** the Qwen3 `tokenizer.json` normalizer is `NFC`, so token ids always encode NFC text.
     The builder stores NFC prompts/reference text (365 rows changed, all Indic: e.g. U+095B → U+091C
     U+093C) so `n_chars` counts exactly the tokenised text and `decode(ids) == text`.
- **Prompt language:** instructions are English for every cell (no human-made in-language templates;
  we do not machine-translate them ourselves). Content is in-language. Chat uses the Aya `inputs` as is.
- **Splits:** item-level, sha256-seeded (seed 20261125): pilot 10 %, tuning 20 %, report rest, cap 200
  (no task reached the cap). Pilot items carry all six languages. `heldout` is reserved for IN22-Conv.
- **Script sanity (tests):** letters/marks only (L*, M*); every non-English row ≥ 0.3 in its block, and
  per (task, lang) ≥ 98 % of rows ≥ 0.8 with median ≥ 0.9. A flat per-row 0.8 rejects correct rows:
  the lows are untranslated Latin named entities (e.g. "George W. Bush" in Telugu 0.70, wine grape names
  in Hindi chat 0.38). Observed: 14 of 5735 non-English rows < 0.8.
- **Flags:** `flores_derived` (translation, translation_xx2en, summarisation) for D-008; `translationese`
  for Aya Indic rows. Fixed-output-length variant = `generation_variants.fixed_length`
  (`fixed_output_tokens: 256`, `ignore_eos: true`) in the config; prompts are shared.
- **Observed lengths (Qwen3):** ~10 tokens per word in ta/te/kn/ml FLORES text; longest prompts are
  chat `dolly_mt:177` (11.3k tokens in te) and QA (≤ 5.0k). Engine `max_model_len` must allow this or
  the harness must filter by `n_prompt_tokens` (record which).
- **Owner items:** (a) IN22-Conv held-out (gated) → `build_heldout_in22_conv` TODO hook; (b) Belebele card
  says "test set, not for training or validation": our `tuning` split tunes decoding settings (draft
  length), not model weights; confirm acceptable or move tuning to non-Belebele tasks; (c) IndicGenBench
  canary asks no LLM pre-training use (we do none).
- **Tooling:** new optional extra `data` (transformers, jinja2, pyarrow); `make sync` installs it.
- **Evidence:** `data/MANIFEST.json` (`tasks.*` stats), `tests/data/test_bench_data.py`,
  `tests/unit/test_data_build.py`.

### D-013b (2026-11-25) Grafting (N1) after the pre-check
- **Decision:** naive row-copy grafting is dropped (C36). Grafting via a closed-form ridge map
  (target row → draft row space) stays as an **exploratory** replay experiment for A3 only (C37), with
  the caveat that the map is validated on in-vocab tokens and must extrapolate to Dravidian leads.
- **Primary mechanism test is unchanged:** coverage-split acceptance (D-010). C35 already predicts
  that A3/A4 acceptance runs on Dravidian text are cut at every character start.

### D-014 (2026-11-25) VRAM policy: full use, no OOM, no spill (owner rule)
- **Decision:** engines may use most of the GPU, but (1) a pre-flight check refuses to launch unless the
  engine budget fits in *currently free* VRAM minus 512 MiB; (2) runs whose telemetry peak reaches
  `total − 512 MiB` are flagged invalid (signature of Windows/WSL sysmem fallback); (3) no CPU offload
  or swap options are ever passed. Implementation: `src/indicspec/harness/gpu_guard.py`.
- **Not done:** changing the driver's "CUDA sysmem fallback policy" — a system-level GPU setting that
  is ask-first (spec §0 rule 6). Owner may choose to set it to "prefer no sysmem fallback".
- **Evidence:** idle desktop uses ~1.6 GB of the 24,463 MiB (nvidia-smi, 2026-11-25).

### D-015 (2026-11-25) Engine memory settings for 24 GB (from the first smoke failures)
- **Finding:** vLLM 0.34.0, Qwen3-8B + EAGLE-3 head (A3) at `gpu_memory_utilization=0.85` and default
  `max_num_seqs` left 0.35 GiB for KV cache; one 4,096-token request needs 0.58 GiB → engine refused
  to start (results/raw/smoke/smoke-vllm-A3-qwen3-8b-20261125T183806/server.log).
- **Decision:** for every arm (A0 included, so comparisons stay matched): `max_num_seqs` /
  `max-running-requests` = 32 (the highest concurrency in the design, §8.2), and the memory fraction
  defaults to the VRAM guard's maximum safe value (free VRAM − 512 MiB) instead of a fixed 0.85.
- **Also fixed:** A4 drafter revision pinned (`72072c42…`); unpinned revisions cannot resolve offline.

### D-016 (2026-11-25) Gate G0 (feasibility): PASS, with caveats
- **Evidence:** docs/ENGINE_STATUS.md (generated from results/raw/smoke). vLLM 0.34.0 runs A0, A1, A3,
  A4, A5 (Qwen3-8B; A0/A1/A5 also on 4B). SGLang 0.5.23 runs A0, A1, A4, A5 (4B and 8B); A3 is not
  faithfully supported on SGLang (speculators format + `norm_before_residual`, see ENGINE_STATUS).
  Both a pruned-vocabulary EAGLE-3 arm (A3, A4) and DFlash (A5) run on this laptop in at least one engine.
- **Novelty part of G0:** partial prior work found (R-PHRASE, R-SARVAM-EAGLE, R-DICTSPEC); owner was told and
  the differentiated scope (D-007) was accepted on 2026-11-25 ("go ahead").
- **Caveats to resolve before measurement:**
  1. SGLang A5 on 4B peaked at 23,945 MiB, 6 MiB under the guard ceiling — raise SGLang headroom for
     measured runs (DFlash adds draft buffers).
  2. SGLang spec metrics are not yet captured per request (only vLLM's are); add SGLang native metrics.
  3. Smoke English τ for A5 on Qwen3-8B is 2.38 (one prompt, 64 tokens) vs the authors' reported ~4.9 mean —
     WS1.6 must check DFlash settings (draft length, thinking mode, attention backend) on English before
     any comparison, so DFlash is not run mis-configured.
- **Smoke τ by language (one prompt each — not results):** A3 en 2.82 / hi 1.33 / ta 1.07 / te 1.03 /
  kn 1.02 / ml 1.05; A4 en 1.50 / hi 1.86 / Dravidian 1.05–1.19; A5-8B en 2.38 / hi 1.57 / Dravidian 1.15–1.64.

### D-017 (2026-11-25) Licence: Apache-2.0 (owner decision)
- Code under Apache-2.0 (official text from the GitHub licences API). Third-party benchmark text is not
  redistributed; builders + checksums only (D-009). Contributor-facing files removed: single-maintainer project.
