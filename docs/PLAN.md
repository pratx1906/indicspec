# IndicSpec working plan

Ordered by dependency, not by date (PROJECT_BRIEF has no schedule on purpose). Each phase has an exit
criterion; nothing confirmatory runs before `docs/PREREG.md` is frozen. Ask-first items (spec §0 rule 6,
§13.1) are marked **[ASK]**. Revised 2026-11-25 after owner feedback (D-010).

## Questions of interest, in priority order

1. **Coverage-split acceptance (primary mechanism test, replaces "grafting as proof").** For pruned-vocab
   EAGLE-3 (A3, A4), split per-position acceptance into positions where the target token is *inside* vs
   *outside* the draft vocabulary. Out-of-vocab acceptance is 0 by construction; the informative quantity
   is **in-vocab acceptance on Dravidian vs English**. If it still lags, the drafter is weak on these
   scripts beyond coverage. No new model needed (replay harness).
2. Language gap per drafter family (H1, H2, G1) in production engines, under load (H4).
3. Byte-fragment predictability (H3): target entropy/top-1 on fragment classes; class acceptance.
4. Per-character, per-word-per-KV-cache and per-joule efficiency per language (unit changes + NVML energy).
5. Exploratory only: vocabulary grafting (N1) and UTF-8 validity masking (N2), replay-first.

## P0 — CPU and small downloads (no asks needed; can start now)

| Task | Output | Exit |
| --- | --- | --- |
| Resume engine research agent (partial notes exist) | `docs/research/engine_research.md` | versions + flags + metric names with sources |
| WS2 builders for ungated sources (IndicGenBench FLORES-IN, XQuAD-IN, Belebele, Aya dolly-MT) | `src/indicspec/data/`, `data/MANIFEST.json` | `make data` reproduces checksums; `tests/data` alignment/dup/script/licence tests pass |
| Pilot / tuning / report / heldout split assignment (seeded, item-level, shared across languages) | split files in `data/bench/` | no item in two splits (test) |
| Qwen3 token tax: tokens per char / per word / parity vs English, per language; compare R-TAX's Qwen2.5 figures | `results/tables/token_tax.csv` | C22 → MEASURED |
| Fragmentation profile on reference text (class distribution per language) | `results/tables/fragmentation_ref.csv` | C19 → MEASURED for Qwen3 |
| Draft-vocab coverage of A3/A4 on reference text, per class (read `d2t`/`t2d` from checkpoints; verify structure) | `results/tables/coverage_ref.csv` | cross-check vs R-SPEED 22% (different data) and R-INDIC-EAGLE ~50% |
| Graft pre-check: cosine between A3 `lm_head` rows and Qwen3-8B `lm_head` rows for in-vocab tokens; compare A3 final norm to target norm | `results/tables/graft_alignment.csv` | decides whether N1 is worth running at all |

## P1 — WS1 engine bring-up (GPU)

**[ASK]** before: downloading Qwen3-8B (~16 GB, over the 10 GB single-step limit); engine installs are
several GB each (under the limit, recorded in `env/ENGINES.md`).

1. Separate uv venvs in WSL for vLLM and SGLang at pinned versions; lockfiles in `env/`.
2. A0 on Qwen3-4B, then Qwen3-8B: VRAM, max context, max concurrency, sanity throughput.
3. Arms in order of expected ease: A1 (n-gram) → A5 (DFlash) → A3 → A4; then candidates A8 (SSSD,
   DictSpec plugin, Red Hat DSpark/DFlash2; EAGLE-4 only if public Qwen3 weights exist).
   Resolve attention-backend questions (`fa3` on the z-lab card vs Triton on the Chutes card) against the
   installed versions on `sm_120`; document every failure with command, version, error, upstream issue.
4. Metrics scraper for each engine's speculative counters, verified against a running server.
5. Correctness: greedy-equality test per working arm at batch size 1 (first-divergence position + top-2
   logit gap on mismatch).
6. Telemetry: add NVML energy counter (`nvmlDeviceGetTotalEnergyConsumption`, verified working under WSL)
   alongside `nvidia-smi` sampling; record the 150 W enforced power limit in every manifest.

Exit: `docs/ENGINE_STATUS.md` complete; `make smoke` green for working arms; **Gate G0** reported to owner.

## P2 — Pilot and pre-registration freeze

1. Pilot on the `pilot` split only (English + small Indic slice): noise of τ, tokens/s, energy; thermal
   behaviour at the 150 W cap.
2. Set every threshold in `docs/PREREG.md`: H1–H4, G1–G3, prompts per cell, repeats, thermal rule,
   exclusions, replay tolerance, **coverage-split test (Q1) as confirmatory**, N1/N2 labelled exploratory.
3. Commit and record the freeze hash. Nothing below runs before this commit.

## P3 — Replay harness (HF Transformers) and mechanism

1. Offline draft-and-verify for A3/A4 (EAGLE-3) and A5 (DFlash) logging per position: proposed token,
   class, in/out of draft vocab, accepted?, target top-1 prob and entropy.
2. **Validate**: replay τ vs engine-reported τ on the same prompts within the pre-registered tolerance;
   otherwise no claims from it.
3. Q1 coverage-split analysis; H3 entropy/class acceptance; G2 test.
4. N2 measurement before any build: rate at which each drafter proposes a token that cannot complete
   the open UTF-8 character. If ≈0, drop N2. If not, simulate the mask:
   - sequential drafters (EAGLE-3, n-gram): mask then re-pick;
   - DFlash (parallel block): per-position logits are computed at once, so the mask can either truncate
     at the first violation or re-pick the best *valid* token at that position given the already-chosen
     prefix. Simulate both; truncation only saves verify work, re-picking can add accepted tokens.
5. N1 graft (exploratory, only if P0 pre-check shows alignment): extend A3's draft vocab with target rows;
   report as exploratory whatever the outcome; a failure is ambiguous and will be described as such.

## P4 — WS4 load sweeps and profiling

Closed-loop concurrency 1–32 (VRAM permitting) + Poisson; greedy primary, sampled secondary; per-arm
draft-length tuning on `tuning` split; block-randomised repeats; thermal rule; Nsight/torch-profiler
breakdown of draft/verify/overhead; cost-model fit; metrics incl. chars/s, words per GB of KV cache,
J per 100 words. Fold in the drafter-memory vs batch-capacity trade-off. Sampling-distribution test.

## P5 — Gates and the fix

- **G1 / G2** reported to owner with evidence.
- A2 byte-aware n-gram vs A1, DictSpec, SSSD, first in replay, then engine plugin hook if one exists.
- N2 mask as an engine plugin only if replay shows a gain **[ASK before patching vLLM/SGLang]**.
- WS5b trained drafter only if G1 and G2 justify it **[ASK before large hidden-state generation / renting GPUs]**.

## P6 — Release

`make reproduce`, benchmark card, report with honest limitations, final novelty re-check.
**[ASK]** before any push, HF upload, upstream issue/PR or author contact.

## Open owner decisions

1. Confirm scope D-007.
2. HF token for gated IN22-Conv (held-out domain).
3. Repository licence.
4. Native-speaker check of machine-translated prompts.
5. Approval to download Qwen3-8B (~16 GB) and install engine envs.
