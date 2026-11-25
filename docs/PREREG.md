# Pre-registration

**Status: DRAFT — thresholds not yet set.** Numeric thresholds are filled in from the *pilot split only*
(English plus a small Indic pilot slice that is disjoint from the reporting split), then this file is
committed and frozen **before** any confirmatory experiment runs. After freezing, changes are appended as
dated amendments with reasons; the original text is never edited (git history is the audit trail).

Freeze commit: _not yet frozen_

## 1. Data subsets

| Split | Purpose | Contents | Used for thresholds? |
| --- | --- | --- | --- |
| `pilot` | noise estimation, threshold setting | TBD (WS2.7) | yes |
| `tuning` | per-arm draft-length tuning | TBD, disjoint from `report` | no (tuning only) |
| `report` | confirmatory results | TBD | no |
| `heldout` | second-domain check | TBD | no |

## 2. Hypotheses (from PROJECT_BRIEF §5)

For each: statistic, data subset, analysis code path, threshold, decision rule.

### H1 — pruned-vocabulary EAGLE-3 head collapses on Dravidian text
- Statistic: TBD (planned: per-language mean τ over prompts, paired by item across languages; difference
  `τ_en − τ_lang` with paired bootstrap 95% CI).
- Falsified if: TBD.
### H2 — full-vocabulary drafters do not collapse
- Statistic / margin: TBD.
### H3 — byte-fragment tokens are predictable
- Statistic: TBD (planned: target top-1 probability and entropy on `BYTE_FRAGMENT_CONT` tokens vs other
  classes; class acceptance from the validated replay harness).
### H4 — speculation crossover concurrency differs by language
- Statistic: TBD (planned: interpolated concurrency where paired speed-up CI crosses 1.0, per language).

## 3. Gates (PROJECT_BRIEF §10)
- G1 "clear language gap": TBD.
- G2 "token class predicts acceptance": TBD.
- G3 "fix beats strongest public arm": TBD (minimum number of languages, margin).

## 4. Measurement rules
- Minimum prompts per cell: TBD (from pilot variance).
- Repeats per cell: ≥ 3 (spec minimum); final number TBD from pilot.
- Block randomisation: cell order shuffled per repeat with recorded seed.
- Thermal-abort rule: TBD — form: abort if max GPU temperature > `T_max` °C or 5th-percentile SM clock
  < `r` × reference clock during the measured window (implemented in
  `src/indicspec/harness/telemetry.py::summarise`).
- Failed-request exclusion: TBD.
- Replay-harness validation tolerance (WS3.5): TBD.
- Multiple-comparison correction for confirmatory tests: Holm at family-wise α = 0.05
  (`src/indicspec/analysis/stats.py::holm`).
