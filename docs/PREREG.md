# Pre-registration

**Status: FROZEN at the commit that introduced this version** (`git log -- docs/PREREG.md`). Changes after
the freeze are appended as dated amendments with reasons; the frozen text is never edited.

## 0. Disclosure: what was seen before freezing

Thresholds were set after the pilot sweeps on the **pilot split only** (`results/raw/pilot_vllm_4b`,
`results/raw/pilot_vllm_8b`; noise tables `results/tables/pilot_vllm_*_noise.csv`). The pilot split contains
English **and** small Indic slices (12 parallel items per language), so pilot effect sizes for every language
were visible when the margins below were chosen. Confirmatory tests run only on the disjoint `report` split,
which no analysis has touched. Draft-length tuning uses the disjoint `tuning` split. Earlier smoke and probe runs
used hand-written prompts and English pilot prompts only.

Pilot facts used to set the numbers (all pilot, exploratory):
- Between-repeat SD of a cell's τ: ≤ 0.09 (4B), ≤ 0.02 (8B); between-repeat CV of per-request tokens/s ≤ 0.11.
- Between-prompt SD of per-request τ: 0.03–0.97.
- Thermal: max 87 °C; median SM clock 1.9–2.2 GHz; minimum 1.59 GHz; peak VRAM 23.88 GB (ceiling 23.95 GB).
- Power: enforced limit 175 W throughout (after D-018).

## 1. Data and design

| Item | Value |
|---|---|
| Split | `report` (confirmatory), `tuning` (draft-length tuning only), `heldout` (IN22-Conv, if obtained; secondary) |
| Languages | en, hi, ta, te, kn, ml |
| Tasks | translation, qa, summarisation, chat (mixed per cell; same item ids in every language) |
| Prompts per (arm, language, concurrency) cell | **100** parallel items (`max_prompts_per_cell: 100`, seed 1234). Prompts longer than `max_model_len − max_tokens` are excluded and counted (identically for every arm). |
| Repeats | **3** per cell, cells in seeded shuffled order within each repeat (block randomisation) |
| Decoding | greedy (temperature 0), `max_tokens` 256, unconstrained length (primary); fixed length (`ignore_eos`) secondary |
| Models | Qwen3-4B: arms A0, A1, A5; concurrency 1, 2, 4, 8, 16, 32. Qwen3-8B: arms A0, A3, A4, A5; concurrency 1, 2, 4, 8 |
| Engine | vLLM 0.34.0 primary (all arms); SGLang 0.5.23 secondary (A0, A1, A4, A5) |

Rationale for 100 prompts: with the pilot's worst between-prompt SD (0.97) the 95% CI half-width of a language's
mean τ is ≈ 0.19, and ≈ 0.12 at the median SD (0.6). The pre-registered margins below are 25% of English τ
(≥ 0.3 in absolute terms for every arm), so 100 prompts resolve them; 3 repeats exceed the spec minimum and the
between-repeat noise is an order of magnitude below the between-prompt noise.

## 2. Statistics (shared)

- **τ (acceptance length)** of a set of requests = (Σ accepted draft tokens + Σ verify steps) / Σ verify steps,
  from per-request engine counters (`src/indicspec/analysis/metrics.py::acceptance_length`, harness fields
  `spec_accepted_tokens`, `spec_verify_steps`).
- **Per-prompt value** = mean over the 3 repeats.
- **Language ratio** R_L = mean_items τ_L / mean_items τ_en, paired by item key (task:item_id) across languages;
  95% CI by paired bootstrap over items, 10,000 resamples, seed 1234
  (`src/indicspec/analysis/stats.py::paired_bootstrap_ci`).
- **Speed-up** of an arm in a cell = mean_items(arm tokens/s per request) / mean_items(A0 tokens/s per request),
  paired by item, same language and concurrency; paired bootstrap CI as above.
- **Multiplicity:** within each hypothesis, Holm correction across the four Dravidian languages at family-wise
  α = 0.05, implemented by testing each language's one-sided CI at level 1 − 0.05·(Holm step weight)
  (`stats.py::holm` on bootstrap p-values = fraction of resamples on the wrong side of the threshold).
- Confirmatory analysis code: `src/indicspec/analysis/confirm.py` (implemented before the first confirmatory
  run; it may only read `report`-split logs).

## 3. Hypotheses

| | Hypothesis | Arms / cell | Confirmed for language L if | Falsified for L if |
|---|---|---|---|---|
| **H1** | Pruned-vocabulary EAGLE-3 heads collapse on Dravidian text | A3, A4 on Qwen3-8B, vLLM, concurrency 1 | upper 95% CI of τ_L < **1.5** | lower 95% CI of τ_L ≥ **0.9 × τ_en** of the same head |
| **H2** | Full-vocabulary drafters do not collapse | A5 (4B and 8B), A1 (4B), vLLM, concurrency 1 | lower CI of R_L ≥ **0.75** | upper CI of R_L < **0.75** |
| **H3** | Byte-fragment tokens are predictable | A1, A5 (4B), vLLM, concurrency 1, engine per-position traces | acceptance rate at evaluated positions whose target token is `BYTE_FRAGMENT_CONT` (pooled over ta/te/kn/ml) has lower CI ≥ English all-position acceptance rate − **0.05** | upper CI < English rate − 0.05 |
| **H4** | The concurrency at which speculation stops paying differs by language | A1, A5 on 4B (grid 1–32) | crossover c*_L (smallest grid concurrency whose speed-up upper CI < 1.0; "> 32" if none) differs from c*_en by **≥ 2 grid steps** | c*_L within 1 grid step of c*_en |

A hypothesis is **confirmed overall** when it is confirmed for ≥ 3 of the 4 Dravidian languages, **falsified
overall** when falsified for ≥ 3, otherwise **inconclusive**. Per-position traces (H3) count only requests whose
step alignment validates (`analysis/positions.py::align`); if < 99% of requests validate, H3 is reported as
not testable.

## 4. Gates (PROJECT_BRIEF §10)

- **G1 (gap):** YES if any public drafter (A1, A3, A4, A5) has upper CI of R_L < 0.75 in ≥ 2 Dravidian languages
  (Holm), or lower CI of R_L > 1.25 in ≥ 2 Dravidian languages (surprising advantage).
- **G2 (mechanism):** YES if both hold for A5 and A1: (a) acceptance rates differ by token class (bootstrap CI
  of the difference between `BYTE_FRAGMENT_LEAD` and `BYTE_FRAGMENT_CONT` acceptance excludes 0), and
  (b) re-weighting Dravidian evaluated positions to the English token-class mix closes ≥ 50% of the English −
  Dravidian acceptance-rate gap (class-standardised gap / raw gap ≤ 0.5). For A3/A4, (c) the acceptance rate at
  in-vocabulary positions is reported per language (coverage-split test, D-010).
- **G3 (fix value):** a new drafter beats the strongest public arm per language when the paired speed-up ratio
  (new / strongest) has lower CI > **1.05** at concurrency 1 in ≥ 3 of 4 Dravidian languages, with English
  regression reported and all correctness tests passing.

## 5. Measurement and exclusion rules

- **Power:** a sweep refuses to start unless the enforced GPU power limit is ≥ 140 W (D-018); enforced limit and
  Windows power plan are recorded per cell. A cell whose recorded enforced limit is < 140 W is discarded and rerun.
- **Thermal:** a cell is discarded and rerun if, during its window, GPU temperature exceeds **95 °C** or the 5th
  percentile SM clock falls below **0.6 ×** the block's median SM clock (`harness/telemetry.py::summarise`).
- **VRAM:** a cell is discarded if telemetry peak memory reaches total − 512 MiB (D-014).
- **Failed requests:** requests with an error are excluded; a cell with > 5% failed requests is discarded and
  rerun once; if it fails again it is reported as failed, never silently dropped.
- **Draft-length tuning** (tuning split, concurrency 1, 1 repeat, 50 prompts per language): A1
  `num_speculative_tokens` ∈ {2, 4, 8}; A5 ∈ {7, 15}; A3/A4 ∈ {3, 5}. The setting with the highest geometric-mean
  speed-up over the six languages is used for every language; the full grid is reported.
- **Correctness gate:** before confirmatory runs, every arm passes greedy equality vs A0 at batch size 1 on
  20 report-split prompts per language. A mismatch whose target top-2 logit gap at the divergence is < 0.1 is a
  near-tie; any other mismatch blocks the arm until explained.
- **Engine definitions:** engines' own acceptance-length values are recorded but never used for statistics (C42).

## Amendments

### Amendment 1 (2026-11-25, before any report-split data exists)

**Reason:** implementing `src/indicspec/analysis/confirm.py` required resolving clauses the frozen text leaves
open. Each was resolved toward the **stricter / more conservative** reading. No report-split request has been
run at the time of this amendment (`results/raw/` contains only pilot, tuning, smoke and probe runs). The frozen
text above is unchanged; these readings are the binding interpretation. They are also written into every
verdicts file (`INTERPRETATIONS` in `confirm.py`).

- **holm_plus_ci:** A per-language confirm/falsify requires BOTH the plain 95% CI condition and rejection under Holm (family-wise alpha 0.05 across ta/te/kn/ml) of the one-sided bootstrap p-value (fraction of resamples on the wrong side of the threshold).
- **h1_falsify_tau_en:** H1 falsification compares tau_L's lower CI with 0.9 x the point estimate of tau_en for the same head (as written).
- **multi_arm_overall:** For hypotheses covering several (arm, model) pairs, the overall verdict is 'confirmed' only if every pair is confirmed, 'falsified' only if every pair is falsified, otherwise 'inconclusive'.
- **h3_pooled:** H3 pools ta/te/kn/ml into one test per (arm, model), so the 3-of-4 language rule does not apply; verdict per pair, overall by the multi-arm rule. CI by cluster bootstrap over requests.
- **h4_bonferroni:** H4 crossover uses one-sided speed-up upper bounds at level 0.05/4 (Bonferroni across the four Dravidian languages), stricter than Holm.
- **g2b_upper_ci:** G2(b) requires the upper 95% CI of (class-standardised gap / raw gap) to be <= 0.5, and a positive raw gap; classes absent from Dravidian positions are dropped and the English weights renormalised.
- **thermal_busy_samples:** Thermal clock rule uses busy samples (utilisation > 50%): the cell's 5th-percentile SM clock vs the block's busy-sample median.
- **power_missing:** A cell without a recorded enforced power limit is excluded (power unknown).
- **telemetry_time:** nvidia-smi timestamps are local wall time; they are converted with the analysing machine's local time zone (or --telemetry-utc-offset).

### Amendment 2 (2026-11-26, after the correctness gate, before any confirmatory report-split sweep)

**Reason:** the greedy-equality gate (§5) failed every drafter arm, and the failure is explained by bf16
numerics, not by drafters. The frozen near-tie threshold (top-2 logit gap < 0.1) is below the smallest
non-zero gap the target can express: Qwen3 logits here are bf16, whose step at these magnitudes is 0.125, so
every observed gap is a multiple of 0.125 and no flip can ever be a near-tie. Evidence (exploratory diagnostics,
`scripts/equality_diag.py`; `results/raw/equality_qwen3-{4b,8b}/{summary,diag}.json`):

- A0 vs A0 at batch size 1: 120/120 identical on both models (the engine is deterministic).
- Every drafter arm diverges on 36–48% of the 120 gate prompts (43–57 prompts); all divergence gaps are 0.125–1.0.
- At **241 of 241** drafter divergences the drafter run's token is the target's rank-2 (234) or rank-3 (7)
  candidate at that position.
- **Target-only control:** A0 with 16 requests in flight vs A0 at batch size 1 diverges on 48/120 (4B) and
  47/120 (8B) prompts, 32 and 19 of which the frozen rule would call "suspected bug" (gaps 0–0.5). Batch shape
  alone produces the same failure in the plain target, with no drafter involved.

**Amended rule (binding from this amendment):** a gate mismatch is a **near-tie** if, at the first divergence,
the arm's token is among the target's top-3 candidates **and** the target's top-2 logit gap is ≤ **1.0**
(8 bf16 steps); any other mismatch is a suspected bug that blocks the arm. Disclosure: this threshold was set
after seeing the gate data (maximum observed gap 1.0); under it every arm passes (A1, A5 on 4B; A3, A4, A5 on
8B). Speed-up and acceptance comparisons are therefore between outputs that agree token-for-token except after
near-tie flips; divergence rates are reported with the results. No report-split sweep has run.
