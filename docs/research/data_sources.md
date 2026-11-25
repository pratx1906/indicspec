# IndicSpec-Bench: data source research (WS2)

Researched 2026-11-25. Research only. Nothing was downloaded, nothing was logged in to, and no gating terms were accepted.
Each fact comes from the official dataset page, the Hugging Face (HF) API (`https://huggingface.co/api/datasets/<id>`, which returns `gated`, `license` and the file list), the HF datasets-server metadata endpoints (`/splits`, `/size`, `/statistics`), the GitHub README or the paper. Those sources are named in the table.
Status: **OPENED** means the fact was read on the named source. **EXCERPT-ONLY** means it was seen only indirectly or partly, and it must be verified before we rely on it.
Our six languages in FLORES-style codes: `tam_Taml`, `tel_Telu`, `kan_Knda`, `mal_Mlym`, `hin_Deva`, `eng_Latn`.

## 1. Candidate table

| Dataset | URL | Licence (where verified) | Gated? | Our six covered | Parallel? | Sizes | Status |
|---|---|---|---|---|---|---|---|
| **FLORES+** (OLDI, v5.0) | https://huggingface.co/datasets/openlanguagedata/flores_plus | CC-BY-SA-4.0 (HF card tag + API `cardData.license`) | **Yes**: API `gated: "auto"`; page: "you have to accept the conditions to access its files". README.md itself returns 401 without login | all six; files `dev/<code>.jsonl`, `devtest/<code>.jsonl` (API siblings) | Yes, multi-way (human translations of the same English sentences) | dev 997, devtest 1012 per language (card); 231 varieties | OPENED (card + API). The card also says the data "should not be used as training data" and asks users not to redistribute files publicly "unless you protect it from automatic scrapping". Full terms text not seen (gated) |
| FLORES-200 (`facebook/flores`) | https://huggingface.co/datasets/facebook/flores ; https://github.com/facebookresearch/flores/blob/main/flores200/README.md | CC-BY-SA-4.0 (HF card text) | **Yes** (accept conditions) | all six (200 languages) | Yes | dev 997, devtest 1012, test hidden | OPENED. Deprecated: "This repository is no longer being updated", points to FLORES+. Original tarball link `https://tinyurl.com/flores200dataset` not followed (EXCERPT-ONLY) |
| **IndicGenBench: Flores-IN** | https://huggingface.co/datasets/google/IndicGenBench_flores_in | CC-BY-SA-4.0 (API; GitHub README; paper App. E) | **No** (API `gated: false`) | ta, te, kn, ml, hi paired with en: `flores_en_ta_{dev,test}.json`, `flores_ta_en_…` etc. | Yes (same English FLORES sentences; paper) | 997 dev / 1012 test per language (paper: "dev" and "test" = FLORES dev and devtest) | OPENED. An ungated copy of FLORES-200 text. It may differ from FLORES+ v5 corrections |
| **Belebele** | https://huggingface.co/datasets/facebook/belebele ; https://github.com/facebookresearch/belebele | CC-BY-SA-4.0 (GitHub `LICENSE_CC-BY-SA4.0`; HF API) | **No** (API `gated: false`) | all six (`data/<code>.jsonl`) | **Yes, fully parallel** (GitHub: "Being fully parallel") | 900 questions per language, test only; 488 distinct FLORES-200 passages, 1-2 questions each | OPENED. Card: "intended to be used only as a test set, and not for training or validation". Also on `dl.fbaipublicfiles.com/belebele/Belebele.zip` |
| **IndicGenBench: XQuAD-IN** | https://huggingface.co/datasets/google/IndicGenBench_xquad_in | CC-BY-SA-4.0 (API; README; paper) | **No** | all six (`xquad_{en,hi,kn,ml,ta,te}_{train,dev,test}.json`, among 12 languages) | Yes in effect: the same 240 XQuAD passages and 1190 QA pairs, professionally human-translated into each language (paper §3). The paper does not use the words "multi-way parallel" for this set | test 1190 QA / 240 passages per language; train and dev 100 QA / 20 passages each (paper) | OPENED (paper HTML, API) |
| IndicGenBench: XorQA-IN | https://huggingface.co/datasets/google/IndicGenBench_xorqa_in | MIT (API; README) | No | ta, te, kn, ml, hi (files listed). en file not checked | Aligned questions and answers across languages over **English** passages | 15.1k test total over 28 languages; test = 302 bn + 237 te Xor-TyDi examples | OPENED. Cross-lingual (English context), so it is a weaker fit |
| IndicGenBench: CrossSum-IN | https://huggingface.co/datasets/google/IndicGenBench_crosssum_in | **CC-BY-NC-SA-4.0** (API; README; paper) | No | `crosssum_english-{en,hi,ta,te,kn,ml}_{train,dev,test}.json` | **Partly**: English articles with human-translated target summaries, but for the 9 languages CrossSum already covered, the authors sampled their own data, so those are "not necessarily" the same articles (paper) | 100 train / 100 dev / 500 test per language (paper) | OPENED. **Non-commercial** |
| IndicGenBench (repo) | https://github.com/google-research-datasets/indic-gen-bench ; paper https://aclanthology.org/2024.acl-long.595/ (arXiv 2404.16816) | Per task, as above. The repo holds MIT, CC-BY-SA-4.0 and CC-BY-NC-SA-4.0 licence files | No | 29 Indic languages, incl. our five | "multi-way parallel benchmark" | see rows above | OPENED. Repo archived. Canary: "do not include the data … for LLM pre-training" |
| **Aya evaluation suite: dolly_machine_translated** | https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite | Apache-2.0 (HF card + API). Upstream `databricks/databricks-dolly-15k` is **CC-BY-SA-3.0** (API) | **No** | all six: tam/tel/kan/mal/hin/eng = **200 each** (datasets-server `/statistics`) | Yes: "parallel datasets (data instances can be mapped using their id column)". Columns: `id, inputs, language, script, source_id, targets` | 23,800 rows, test only, ~101 languages | OPENED. **Machine-translated with NLLB**, so there is translationese risk for ta/te/kn/ml/hi |
| Aya evaluation suite: dolly_human_edited | same | Apache-2.0 | No | **hi only** of our six (with ar, fr, ru, es, sr) | Yes, post-edits of the MT rows | 1.2k rows total | OPENED. Useful as a Hindi translationese check |
| Aya evaluation suite: aya_human_annotated | same | Apache-2.0 | No | **te, en only** of our six | **No**: different original prompts per language | 250 per language × 7 | OPENED |
| Aya Dataset | https://huggingface.co/datasets/CohereLabs/aya_dataset | Apache-2.0 (card + API) | No | train: ta 14,133; te 8,439; kn 334; ml 1,749; hi 1,153; en 3,944 (`/statistics`) | **No**: crowd-sourced originals and re-annotations per language | 202,362 train; test 1,750 (7 languages, not ta/kn/ml/hi) | OPENED. Natural, non-translated prompts, so it can serve as a non-parallel naturalness control |
| aya_redteaming | https://huggingface.co/datasets/CohereLabs/aya_redteaming | Apache-2.0 (card) | not stated | en, hi confirmed; others hidden ("+5") | not stated | 1K–10K | EXCERPT-ONLY for languages. Harmful-content prompts; **excluded** |
| **IN22-Gen** | https://huggingface.co/datasets/ai4bharat/IN22-Gen | CC-BY-4.0 (HF card + API; IndicTrans2 README licence table) | **Yes** (API `gated: "auto"`; accept conditions, login needed) | all six (`tam_Taml` … `eng_Latn`) | Yes: "n-way parallel"; "all sentences are aligned across configurations" | 1,024 sentences (512 Wikipedia + 512 web), 13 domains, India-centric | OPENED. Files: `data/train-00000-of-00001.parquet`, `IN22_benchmark.tar.gz` |
| **IN22-Conv** | https://huggingface.co/datasets/ai4bharat/IN22-Conv | CC-BY-4.0 (card; IndicTrans2 README) | **Yes** ("agree to share your Hugging Face information") | all six | Yes: "n-way parallel" | 1,503 sentences, 16 conversational domains; fields include `doc_id, sent_id, scenario, speaker, turn` | OPENED. Also as `ai4bharat/BPCC` `additional/IN22_testset.zip` (BPCC is also `gated: "auto"`) |
| IndicQA | https://huggingface.co/datasets/ai4bharat/IndicQA | CC-BY-4.0 (card tag) | not stated | hi, kn, ml, ta, te (no en) | Not stated; per-language SQuAD-style | n<1K category | OPENED (card). Uses a loading script. Not parallel and no English, so **not chosen** |
| XL-Sum | https://huggingface.co/datasets/csebuetnlp/xlsum | **CC-BY-NC-SA-4.0** ("non-commercial research purposes") | No | en, hi, ta, te; **no kn, ml** | **No** | test: en 11,535; hi 8,847; ta 2,027; te 1,302 | OPENED. Loading script (`xlsum.py`). **Excluded** (NC, missing kn and ml, not parallel) |
| Global-MMLU | https://huggingface.co/datasets/CohereLabs/Global-MMLU | Apache-2.0 (API) | No | en, hi, te only; **no ta, kn, ml** (API config list) | Probably (same row counts). Not verified | 14k test, 285 dev per language | OPENED. **Excluded** (coverage) |
| Okapi | https://github.com/nlp-uoregon/Okapi | **CC-BY-NC-4.0** ("only non-commercial use") | n/a (GitHub script) | 26 languages; our languages **not confirmed** on the page | ChatGPT-translated Alpaca. Parallelism not stated | 52K Alpaca × 26 | EXCERPT-ONLY for languages. **Excluded** (NC, MT by ChatGPT) |
| Indic-Instruct v0.1 | https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1 | not on card ("check … technical report", arXiv 2401.15006) | not stated | **en, hi only** | IndicTrans2-MT subsets | en 5.23k / hi 7.58k viewer rows | OPENED. **Excluded** (coverage) |
| IndicXNLI | https://huggingface.co/datasets/Divyanshu/indicxnli | Conflict: tag cc0-1.0 vs text "non-commercial research purposes" (CC-BY-NC-4.0) | No | hi, kn, ml, ta, te (no en) | MT of XNLI | test 5,010 per language | OPENED. **Excluded** (classification task with short outputs, licence conflict) |

## 2. Recommended design

### 2.1 Source per task (main domain)

| Task | Source | Prompt construction | Parallel? | Licence / ship policy |
|---|---|---|---|---|
| **Translation** | FLORES devtest via `google/IndicGenBench_flores_in` (ungated). FLORES+ v5 `devtest` is optional if the owner accepts its terms | Cell L (Indic): "Translate into L:" + English source. Cell en: Indic→English, rotating source language across items (or one fixed source, e.g. hin). Group 3-5 consecutive sentences per item so outputs are long enough for steady-state decoding | Yes, human (FLORES) | CC-BY-SA-4.0. Ship builder only (FLORES+ terms ask for no unprotected redistribution) |
| **Question answering** | `google/IndicGenBench_xquad_in` test (1190 QA per language, all six) | In-language passage + question, with the instruction "answer and explain in 2-4 sentences in L" (raw XQuAD answers are a few tokens, which is too short to measure speculative decoding) | Yes, professional human translation, same item IDs | CC-BY-SA-4.0. Builder script |
| **Summarisation** | Primary: **Belebele passages** (488 distinct FLORES-200 passages, in-language, all six): "Summarise this passage in L in 2-3 sentences". Alternative: CrossSum-IN test (English article → L summary), kept only if the builder confirms the shared article IDs across all six files | Belebele: fully parallel human text. CrossSum-IN: only partly parallel (see table) | Belebele CC-BY-SA-4.0 (test-only use; we do not train). CrossSum-IN **CC-BY-NC-SA-4.0**, builder only, flag non-commercial |
| **Open instruction / chat** | `CohereLabs/aya_evaluation_suite` `dolly_machine_translated` (200 per language, aligned by `id`) | Use `inputs` as the user turn | Parallel but **machine-translated (NLLB)**: translationese risk for all Indic cells. English is the Dolly original | Apache-2.0 (Dolly upstream CC-BY-SA-3.0). Builder |
| Non-parallel naturalness control (separate analysis) | `CohereLabs/aya_dataset` train, `annotation_type` = original annotations, sampled per language (kn has only 334 rows in total) | Native prompts | **Not parallel**; analyse separately | Apache-2.0 |
| Translationese sensitivity (Hindi only) | `dolly_human_edited` hi rows vs `dolly_machine_translated` hi rows (same IDs) | Same 200 prompts | Parallel | Apache-2.0 |

The summarisation source (Belebele) is the same FLORES text as the translation source, so the builder must use disjoint sentences. Belebele passages come from FLORES-200, but which FLORES split they come from was **not verified**. The builder should drop FLORES translation items whose sentences appear in any Belebele passage it uses.

### 2.2 Held-out second domain
**IN22-Conv** (conversational, 16 everyday domains, India-centric, n-way parallel, human-translated, CC-BY-4.0). It is clearly unlike the main domain: FLORES, Belebele and XQuAD are Wikipedia, Wikinews or encyclopedic text. Use it for (a) en→L translation of 3-5-turn dialogue windows (grouped by `doc_id`/`turn`) and (b) a chat task: "continue this conversation as <speaker>" in L. **IN22-Gen** (Wikipedia + web, India-centric, 1,024 sentences) is a backup held-out set for translation. Both are **gated (`auto`)**: the owner must log in to HF, accept the conditions, and give the builder an `HF_TOKEN` through `.env`.

### 2.3 Items per language-task cell
- Translation: 200 items per cell (multi-sentence groups from 1012 devtest sentences, after removing the Belebele overlap). Pilot: 40 items from FLORES **dev**.
- QA: 200 per cell from XQuAD-IN test (one question per passage, 200 of 240 passages). Pilot: 40 from XQuAD-IN dev and train (100 QA each).
- Summarisation: 200 distinct Belebele passages per cell. Pilot: 40 further passages.
- Chat: Dolly-MT has only 200 per language, so use 160 for reporting and 40 for the pilot.
- Held-out: 150 dialogue windows per cell from IN22-Conv (1,503 sentences).
- In total, about 200 × 4 tasks × 6 languages ≈ 4.8k reporting prompts, plus about 0.9k held-out prompts. Each cell has the same item IDs across all six languages.
- The fixed-output-length set (WS2 task 3) reuses the same items with forced output length. It needs no extra data.

### 2.4 Truly parallel vs machine-translated
- Human-translated and parallel: FLORES / Flores-IN, Belebele, XQuAD-IN, IN22-Gen, IN22-Conv.
- Machine-translated (translationese risk): Aya `dolly_machine_translated` (NLLB) for every Indic cell. Report chat results with this caveat and run the Hindi human-edited check.
- Not parallel: Aya Dataset originals; CrossSum-IN for languages CrossSum already covered.

### 2.5 Licence and redistribution policy
- Ship **builder scripts plus `data/MANIFEST.json` (sha256, source revision hash, licence)**, not the raw text, for every source. This keeps us consistent with FLORES+ terms (no unprotected public redistribution), Belebele's test-only statement and the IndicGenBench canary request.
- **Non-commercial:** CrossSum-IN (CC-BY-NC-SA-4.0), XL-Sum, Okapi, IndicXNLI. Use CrossSum-IN only through a builder, flagged NC. The rest are excluded.
- Share-alike (CC-BY-SA-4.0: FLORES, Belebele, XQuAD-IN, Flores-IN; CC-BY-SA-3.0: Dolly upstream): if any derived prompt text is ever published, it must carry the same licence and attribution.
- Pin HF dataset revisions (commit SHA) in the builder. FLORES+ was last modified on 2026-11-22 (API), so it is still changing.

## 3. Owner actions
1. Optional: accept FLORES+ terms on HF if you want v5 corrections instead of the ungated Flores-IN copy.
2. **Required for the held-out domain:** accept IN22-Conv (and IN22-Gen) conditions on HF and provide `HF_TOKEN` in a gitignored `.env`.
3. Decide whether non-commercial CrossSum-IN is acceptable even as a builder-only alternative.
4. Find a native-speaker check of the Dolly-MT prompts for ta/te/kn/ml (Section 12).

## 4. Not verified (to do before relying on)
- FLORES+ full gating terms text and changelog (the README is 401 without login).
- Which FLORES split the Belebele passages come from.
- Whether CrossSum-IN `english-{ta,te,kn,ml,hi,en}` test files share article IDs.
- Okapi language list; aya_redteaming language list; IndicQA gating.
- The FLORES-200 original tarball URL behind the tinyurl link.
