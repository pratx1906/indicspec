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

### D-018 (2026-11-25) Laptop power mode: discard throttled pilot, add a power guard
- **Finding:** with Armoury Crate / Windows plan "Silent" the enforced GPU power limit was 55 W; under load the
  SM clock median was 277 MHz of 3,090 (`sw_power_cap` on every busy sample). The limit changed to 150 W during
  the pilot, so arms would have run under different power states. After the owner switched modes, a 20 s bf16
  matmul load reached the GPU's 175 W maximum at 2.14–2.30 GHz (`scripts/gpu_load_test.sh`).
- **Decision:** the partial pilot is archived under `results/raw/_discarded/` (never used). Every sweep now
  records the enforced power limit, Windows power plan, clocks and throttle reasons at start and per cell, and
  refuses to start below `min_enforced_power_w` (140 W in all configs). Owner keeps the laptop on the AC
  adapter in the performance mode for all measured runs.

### D-019 (2026-11-25) Pilot complete; pre-registration frozen
- **Evidence:** `results/raw/pilot_vllm_4b` (A0, A1, A5; 162 cells) and `results/raw/pilot_vllm_8b` (A0, A3, A4, A5;
  144 cells), 0 failed requests, enforced power 175 W, max 87 °C. Noise tables: `results/tables/pilot_vllm_*_noise.csv`.
- **Decision:** `docs/PREREG.md` frozen with 100 prompts × 3 repeats per cell, H1–H4 thresholds, gates G1–G3,
  thermal/power/VRAM/failure rules and the draft-length tuning grid. It discloses that pilot effect sizes for all
  languages were visible when margins were chosen; confirmatory analysis reads only the `report` split.
- **Pilot observations (exploratory, not results):** A3 English 2.46 vs Dravidian 1.15–1.20; A4 Hindi 2.01,
  English 1.49 (matches its card's self-report); A5 English ~3.0 vs Dravidian 1.27–1.74 (4B: 1.46–2.14) and
  speed-up below 1.0 on Dravidian at 16 users (4B); A1 English 1.35 vs Dravidian 2.6–2.9.

### D-020 (2026-11-26) Draft-length tuning result (tuning split, frozen rule)
- **Evidence:** `results/raw/tune_vllm_{4b,8b}`; grids `results/tables/tune_vllm_{4b,8b}_grid.csv`.
- **Decision (mechanical, PREREG §5):** 4B: A1 k=8 (geometric-mean speed-up 2.71; k4 2.23, k2 1.83), A5 k=15
  (1.58; k7 1.46). 8B: A3 k=3 (1.14; k5 1.08), A4 k=3 (1.16; k5 1.08), A5 k=15 (1.36; k7 1.24).
  `configs/confirm_vllm_{4b,8b}.yaml` were generated by `analysis.tuning --write-config`.

### D-021 (2026-11-26) Greedy-equality gate failure explained as bf16 numerics; PREREG Amendment 2
- **Finding:** the overnight chain stopped at the 4B gate (02:24); the 8B gate also failed every drafter arm.
  Diagnostics show the failure is a property of the frozen threshold, not of the drafters: the 0.1 near-tie
  threshold is below one bf16 logit step (0.125); drafter divergences always pick the target's rank-2/3
  candidate; the plain target diverges just as often when only its batch shape changes (details in
  PREREG Amendment 2, claims C49–C51).
- **Decision (owner-approved):** PREREG Amendment 2 replaces the near-tie rule (top-3 and gap ≤ 1.0). All arms
  pass under it. Confirmatory sweeps may start; divergence rates are reported beside every speed-up. Follow-up:
  make `harness/equality_gate.py` record top-k and apply the amended rule directly.

### D-022 (2026-11-27) Qwen3-4B confirmatory sweep complete; verdicts recorded as computed
- **Evidence:** `results/raw/confirm_vllm_4b` (324 cells, 32,400 requests, 0 excluded, enforced power 175 W),
  run 03:26–10:16; verdicts `results/tables/confirm_vllm_4b_verdicts.json` (claims C52–C55).
- **Verdicts:** H2 inconclusive (n-gram confirmed, DFlash falsified), H3 inconclusive (same split), H4 confirmed,
  G1 YES, G2 NO. Verdicts are reported exactly as the frozen rules produce them, including G2's joint rule
  failing because the n-gram arm has no gap to explain; no rule is changed after seeing these data.
- **Operational:** the 11:00 hard stop fired at 11:10 because one long `sleep` drifted under WSL2;
  `scripts/run_until.sh` replaces it with a wall-clock check every 20 s. `confirm_vllm_8b` was stopped after 16
  of its 288 cells and resumes by cell (rows are written only when a cell completes).
- **Exploratory additions:** `analysis/breakdown.py` (by task, energy, token class; C56) and two README figures.

### D-023 (2026-11-27) Cross-engine check on SGLang; SGLang NGRAM drafts across requests
- **Evidence:** `results/raw/xengine_sglang_4b` (A0, A1, A5; tuning split, same 50 items per language as
  `tune_vllm_4b`; concurrency 1 and 16) and `results/raw/xengine_sglang_4b_flush` (A0, A1 with a cache flush
  before every cell); tables `results/tables/xengine_sglang_4b{,_flush}.csv` (claims C57–C58).
- **DFlash reproduces across engines:** SGLang acceptance length is within 0.04 of vLLM's in every language
  (en 2.72 vs 2.76, ml 1.39 vs 1.42), with the same language ordering and the same collapse under load.
- **Finding (benchmark pitfall):** SGLang's NGRAM drafter keeps one corpus of every earlier request on the server
  (`speculative/ngram_worker.py`, reset only by `POST /flush_cache`). Without a flush, a cell drafts from an earlier
  cell's outputs for the same prompts: English n-gram acceptance 4.70 and speed-up 3.80× vs 1.22 / 0.97× after a
  flush. Tamil, the only language whose 1-user cell ran before its 16-user cell, is unchanged by the flush
  (1.78× vs 1.77×), which isolates the cause. vLLM's n-gram proposer reads only the current request, so the
  vLLM results (pilot, tuning, confirmatory) are unaffected.
- **Decision:** the sweep now calls the engine's `flush_cache_endpoint` before every cell (SGLang default, recorded
  per cell as `cache_flushed`). The unflushed A1 numbers are kept as evidence of the pitfall and never used.

### D-024 (2026-11-27) Tooling for reproducibility, raw-log size and the 8B finish

- **Reproduce check:** `scripts/reproduce.sh` rebuilds every table derived from raw logs (pilot, tuning,
  confirmatory, breakdowns, cross-engine) and compares each with the committed copy via `git diff`. The
  cross-engine tables, first produced by a one-off script, now come from `indicspec.analysis.xengine`.
- **Raw-log size:** request logs reach 51–59 MB per block (GitHub warns above 50 MB). Readers now accept
  `requests.jsonl` and/or `requests.jsonl.gz` (`indicspec.rawio`). Writing gzip is opt-in per sweep
  (`compress_raw: true`) and is **not** enabled for `confirm_vllm_8b`, so that run stays in one format.
  `scripts/pack_raw.sh` can compress a finished run losslessly (SHA-256 checked before the plain file is
  removed); it has not been applied to any committed run.
- **8B finish:** `scripts/finish_8b.sh` runs the joint confirmatory analysis (4B + 8B), the 8B breakdowns,
  the figures and a Markdown digest (`indicspec.analysis.digest`, numbers copied from the verdicts, not
  recomputed). Updating CLAIMS/README and committing stay manual.
- **New exploratory figures:** `confirm-tasks` and `confirm-energy` (claims C56, C59). C59 also corrects C56's
  per-word n-gram energy range to 0.72–0.84×.
- **`run_until.sh` fix:** at the deadline the sweep exits first, and the script then killed its own watchdog
  while it was still inside `stop_sweeps.sh`, leaving the vLLM server holding 23.7 GB (seen on a 30-minute 8B
  resume test, 16:30). The script now waits for the watchdog after a deadline stop; verified with a dummy sweep.
  The test added 7 valid 8B cells (23/288); 1-user cells take 3.5-8.5 min, 8-user cells 0.5-1.5 min.
- **Graceful stop (owner request, 2026-11-27):** a deadline no longer kills the cell in progress. `run_until.sh`
  now waits for the running cell to finish (new line in any `cells.jsonl`; cap +20 min), then stops;
  `stop_after_cell.sh` does the same for an already-running sweep. Used at 20:15 (cell finished 29 s later; 69/288
  cells) and verified with a dummy sweep.

### D-025 (2026-11-28) Cross-model quick test (exploratory): the tokenizer drives the n-gram language pattern

- **Why:** every result so far used one tokenizer (Qwen3 = Qwen2.5). The owner asked whether the pattern
  holds for other open models, including Indian-built ones, and for FrontiersMind's Lumma/Nandi.
- **Protocol (not pre-registered; never used for verdicts):** tuning split; the same items in every
  language (first 10 per language that fit the model's context); greedy; 128 new tokens; one user;
  no speculation vs vLLM's n-gram drafter (k=8, lookup 2–5, as tuned arm A1.k8); a repetition score
  (share of repeated 4-grams) to flag looping models. Tokens per word on tuning-split reference texts.
  Answer quality (C63): 20 items per language, up to 1,024 new tokens, QA accuracy, chrF vs
  references, share of output in the expected script, truncation. Scripts `scripts/quick_models.py`,
  `quick_models_hf.py`, `quick_quality.py`, `crossmodel_summary.py`; raw rows `results/raw/quick_models/`.
- **Kept (8):** Qwen3-4B and Qwen3-0.6B (reference), Gemma 4 E4B, Sarvam-1, Navarasa-2.0 (Gemma-7B +
  Indic data), Tamil-LLaMA-7B, OpenHathi-7B, FrontiersMind Lumma-0.6B; Sarvam-105B/30B tokenizer only.
- **Excluded, with reasons** (raw rows kept; cached weights deleted): Bharat-Tiny-LLM-v2 (repetition
  0.54–0.95, so its speed-ups measure looping), Pragna-1B, Project Indus, Llama-2-7B base and
  Malayalam-LLaMA (repetition 0.3–0.7 in several languages), Ganga-2-1B (5x speed-up with tau 6.7: the
  output echoes the prompt), Gemma 4 E2B (same tokenizer as E4B), Qwen2.5-1.5B (its twin Bharat-Tiny
  is unusable, so the data-only comparison is lost), Nandi-Mini-150M (same tokenizer as Lumma, 150M),
  Krutrim-1 (its MPT config fails validation in transformers 5.22), BharatGen Param-1, Airavata and
  Gemma-7B (licence not accepted on the account).
- **FrontiersMind:** Lumma/Nandi do not load in vLLM 0.34 (its Transformers backend has no factorised
  output head `lm_head_proj`; SGLang's fallback is the same design). They ran in transformers with
  prompt-lookup decoding (draft 8, n-grams up to 5), with Qwen3-0.6B through the same path as the
  reference; only ratios within that engine are compared. Their remote code was read before running
  (imports torch and transformers only). Their public claim (Lumma 2.21 tokens per word vs
  Sarvam-105B 2.35 on 11 languages) is consistent with ours on 5 (2.12 vs 2.40).
- **Findings** (C60–C63): tokens per Dravidian word range from 2.3 (Lumma) to 15.7 (OpenHathi), with
  Qwen3 at 11.7; the n-gram drafter's Dravidian advantage over English shrinks as the tokenizer
  improves, including within single models (Tamil-LLaMA gains least on Tamil, OpenHathi least on
  Hindi). So the confirmatory n-gram result (C53) is largely a property of the Qwen tokenizer.
- **Answer quality changes what can be read from the speeds** (C63): only Qwen3-4B, Gemma 4 and Navarasa
  answer reliably in the right language. Lumma, Sarvam-1 (base), OpenHathi (base) and Tamil-LLaMA (outside
  Tamil) often echo the English prompt or switch script, which also inflates n-gram speed-ups. The
  tokenizer conclusion rests on the three reliable models (monotone: 11.7 -> 5.1 -> 3.0 tokens per word,
  Dravidian-vs-English n-gram advantage 2.16 -> 1.67 -> 1.07); Lumma's tokenizer result (C60) stands, its
  speed result (C62) does not without a fix to its answers.
- **Scope:** 10–20 items per language, one GPU, short outputs: a direction, not a confirmatory result.
