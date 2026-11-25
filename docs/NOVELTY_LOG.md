# Novelty log

Required by PROJECT_BRIEF Section 4.1. Each row records one query. A negative search is a negative *search result*, not proof of absence.

Conventions:
- **Engine/site:** `WebSearch` = the agent's general web search tool (US index); `HF API` = `https://huggingface.co/api/models?search=<q>` or `/api/datasets?search=<q>` (substring match on repo id, max 40 results).
- **Status of hits:** `OPENED` = page/card/config fetched and read; `EXCERPT-ONLY` = seen only in a search snippet.
- All queries run 2026-11-25.

## Query table

### A. Indic / Dravidian speculative-decoding evaluations

| # | Date | Engine/site | Exact query | Result summary | Relevant hits (status) |
|---|---|---|---|---|---|
| 1 | 2026-11-25 | WebSearch | `speculative decoding Tamil Telugu Kannada Malayalam evaluation` | No paper dedicated to these four languages. Returned Paudel et al. and Yi et al. (multilingual, language lists not in snippet). | https://arxiv.org/abs/2605.30580 (EXCERPT-ONLY here; spec lists it as R-PAUDEL OPENED); https://arxiv.org/abs/2406.16758 (OPENED abstract; languages not named in abstract); https://github.com/Kthyeon/Multilingual-SpecBench (EXCERPT-ONLY) |
| 2 | 2026-11-25 | WebSearch | `speculative decoding Indic languages Dravidian acceptance length drafter` | **Surfaced arXiv 2611.12464 (phrase-level drafting) with Telugu/Kannada/Hindi results on Qwen3-4B.** Also UniSpec (ACL 2026), AdaSpec (AAAI). | https://arxiv.org/abs/2611.12464 (OPENED abs + HTML); https://aclanthology.org/2026.acl-long.285/ (EXCERPT-ONLY); https://ojs.aaai.org/index.php/AAAI/article/view/40307 (fetch blocked by bot protection; EXCERPT-ONLY) |
| 3 | 2026-11-25 | WebSearch | `"Phrase-Level Speculative Decoding on a Frozen Target"` | Confirms title/author (Madhusudhana Naidu); related phrase-level work (Ouroboros). | https://arxiv.org/abs/2611.12464 (OPENED); https://arxiv.org/abs/2402.13720 (EXCERPT-ONLY) |
| 4 | 2026-11-25 | WebSearch | `UNISPEC multilingual speculative decoding ACL 2026 seven languages` | UniSpec / Multi-SpecBench covers EN, DE, FR, ES, ZH, JA, VI (snippet). No Indic language. | https://aclanthology.org/2026.acl-long.285/ (EXCERPT-ONLY) |
| 5 | 2026-11-25 | WebSearch | `Sarvam speculative decoding EAGLE Indic languages inference acceptance` | No published Sarvam acceptance numbers for Indic found. Sarvam-M blog mentions lookahead decoding (snippet). | https://www.sarvam.ai/blogs/sarvam-m (EXCERPT-ONLY) |
| 6 | 2026-11-25 | WebSearch | `tokenizer fertility speculative decoding low-resource script UTF-8 byte tokens drafter acceptance 2026` | Found DictSpec (UNLP 2026): fertility-linked speedups for Ukrainian/Crimean Tatar; phrase-level paper again; byte-LM paper. | https://aclanthology.org/2026.unlp-1.15/ (OPENED landing page/abstract); https://arxiv.org/abs/2611.12464 (OPENED); https://arxiv.org/abs/2610.05978 (EXCERPT-ONLY) |
| 7 | 2026-11-25 | WebSearch | `arxiv 2026 speculative decoding Hindi Tamil Bengali drafter vocabulary coverage EAGLE-3 multilingual` | Only Indic EAGLE-3 vocabulary-coverage source found is the SwitchXDDD card. SpecVocab is not multilingual-focused. | https://huggingface.co/SwitchXDDD/multilingual-eagle3-qwen3-8b (OPENED); https://arxiv.org/abs/2602.13836 (EXCERPT-ONLY) |
| 8 | 2026-11-25 | WebSearch | `"speculative decoding" Malayalam OR Kannada OR Tamil arxiv 2026` | Nothing new beyond rows 1–6. | (as above) |
| 9 | 2026-11-25 | WebSearch | `IndicSpec speculative decoding benchmark` | No project/paper named IndicSpec found. | none |
| 10 | 2026-11-25 | WebSearch | `"Cross-Lingual EAGLE-3 for Indic Languages"` (title cited as "link TBD" on the SwitchXDDD card) | No hit; the companion paper does not appear to be indexed yet. | none |

### B. DFlash on non-English / Indic text

| # | Date | Engine/site | Exact query | Result summary | Relevant hits (status) |
|---|---|---|---|---|---|
| 11 | 2026-11-25 | WebSearch | `DFlash speculative decoding multilingual non-English evaluation` | No dedicated DFlash-on-non-English study. One aggregate data point: NVIDIA blog reports DFlash vs EAGLE-3 on SPEED-Bench "multilingual" (Llama 3.1 8B, gpt-oss-120b), no per-language breakdown. | https://developer.nvidia.com/blog/?p=118866 (OPENED) |
| 12 | 2026-11-25 | WebSearch | `Qwen3 MTP head or DFlash drafter evaluated on Hindi or Indian languages acceptance` | No DFlash/MTP evaluation on Hindi or other Indian languages found. AngelSlim Qwen3-8B MTP/DFlash/DSpark/DFly tables are English/code only. | https://huggingface.co/AngelSlim/Qwen3-8B-MTP-TTT3 (OPENED card); https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2 (EXCERPT-ONLY) |
| 13 | 2026-11-25 | HF API models | `dflash tamil`, `dflash telugu`, `dflash hindi`, `dflash indic` | 0 results each. | none |
| 14 | 2026-11-25 | HF API models | `dflash multilingual` | 2 results, both ASR drafters (argmaxinc/asr-dflash-drafter-multilingual-*), not text LLM / not Indic. | argmaxinc repos (metadata only) |
| 15 | 2026-11-25 | HF API models | `sarvam dflash`, `sarvam speculator` | 0 results. | none |

### C. EAGLE-3 (or other) drafters for Tamil / Telugu / Kannada / Malayalam

| # | Date | Engine/site | Exact query | Result summary | Relevant hits (status) |
|---|---|---|---|---|---|
| 16 | 2026-11-25 | WebSearch | `EAGLE-3 draft head Tamil OR Telugu OR Kannada OR Malayalam huggingface` | No Dravidian EAGLE-3 head for **Qwen3** found; SwitchXDDD (Hindi/Gujarati) is closest; a Korean-only Gemma head shows language-specific heads exist elsewhere. | https://huggingface.co/SwitchXDDD/multilingual-eagle3-qwen3-8b (OPENED) |
| 17 | 2026-11-25 | HF API models | `eagle3 tamil`, `eagle3 telugu`, `eagle3 kannada`, `eagle3 malayalam`, `eagle3 indic`, `eagle3 hindi`, `eagle indic`, `hindi eagle`, `indic draft`, `speculator indic`, `indic speculative`, `tamil speculative`, `telugu speculative`, `kannada draft`, `malayalam draft` | 0 results each (substring search on repo ids only; cards/tags not searched). | none |
| 18 | 2026-11-25 | HF API models | `multilingual eagle3`, `eagle3 multilingual` | 1 result: SwitchXDDD/multilingual-eagle3-qwen3-8b. | (see row 16) |
| 19 | 2026-11-25 | HF API models | `sarvam eagle` | **3 results. `sulabhkatiyar/eagle3-sarvam-30b` is a public EAGLE-3 head whose training mix includes Tamil 11k, Telugu 11k, Kannada 8k, Malayalam 8k records** (target = sarvamai/sarvam-30b, not Qwen3; evaluated on Tamil but not Telugu/Kannada/Malayalam). Also `adarshxs/sarvam-30b-eagle3-draft(-v2)` (no card). | https://huggingface.co/sulabhkatiyar/eagle3-sarvam-30b (OPENED card); https://huggingface.co/adarshxs/sarvam-30b-eagle3-draft-v2 (metadata only, no README) |

### D. Byte-fragment or tokenizer-aware drafting

| # | Date | Engine/site | Exact query | Result summary | Relevant hits (status) |
|---|---|---|---|---|---|
| 20 | 2026-11-25 | WebSearch | `byte-level tokenizer aware speculative decoding drafter byte fragments` | No paper found that drafts byte-fragment tokens of a subword (byte-level BPE) target for under-merged scripts. Related: byte-level LMs with speculative decoding (MambaByte, BLT, byte-LM scaling), cross-vocabulary drafting (OmniDraft). | https://arxiv.org/abs/2401.13660 (EXCERPT-ONLY); https://arxiv.org/abs/2610.05978 (EXCERPT-ONLY); https://arxiv.org/abs/2605.08044 (EXCERPT-ONLY); https://arxiv.org/abs/2507.02659 (EXCERPT-ONLY) |
| 21 | 2026-11-25 | WebSearch | (row 6 query, re-used for this topic) | DictSpec: static n-gram dictionary per tokenizer+script, gains correlate with fertility; Ukrainian/Crimean Tatar only; vLLM plugin. Methodologically close to the project's planned byte-aware n-gram drafter (D6). | https://aclanthology.org/2026.unlp-1.15/ (OPENED landing page) |

### E. SPEED-Bench extensions to Indic languages

| # | Date | Engine/site | Exact query | Result summary | Relevant hits (status) |
|---|---|---|---|---|---|
| 22 | 2026-11-25 | WebSearch | `SPEED-Bench speculative decoding Indic languages extension Hindi` | No Indic extension of SPEED-Bench found. SPEED-Bench's own multilingual category (23 languages) includes BN and GU codes, no Dravidian codes (Appendix A, Table 2, read via arXiv HTML). | https://arxiv.org/html/2604.09557 (OPENED) |
| 23 | 2026-11-25 | HF API datasets | `speed-bench`, `speed bench indic`, `indic speculative`, `specbench multilingual` | `nvidia/SPEED-Bench` exists (licence field `other`, configs qualitative + throughput_1k…100k); several re-uploads/result dumps; 0 Indic-named datasets. | https://huggingface.co/datasets/nvidia/SPEED-Bench (metadata only) |

### F. Newer drafters for Qwen3-4B / Qwen3-8B with public weights

| # | Date | Engine/site | Exact query | Result summary | Relevant hits (status) |
|---|---|---|---|---|---|
| 24 | 2026-11-25 | WebSearch | `DSpark confidence-scheduled speculative decoding semi-autoregressive Qwen3 weights` | DSpark (arXiv 2607.05147, DeepSeek-AI + PKU per snippet) evaluates Qwen3-4B/8B/14B; SGLang integration blog; NeMo AutoModel recipe. | https://arxiv.org/abs/2607.05147 (EXCERPT-ONLY); https://lmsys.org/blog/2026-07-06-dspark-sglang (EXCERPT-ONLY); https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding (EXCERPT-ONLY) |
| 25 | 2026-11-25 | HF API models | `Qwen3-8B-DFlash`, `Qwen3-4B-DFlash` | z-lab repos exist plus many community DFlash drafters. | https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16, https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16 (OPENED) |
| 26 | 2026-11-25 | HF API models | `Qwen3-4B-speculator`, `Qwen3-8B-speculator` | Red Hat speculators: `RedHatAI/Qwen3-4B-speculator.dspark`, `RedHatAI/Qwen3-4B-speculator.dflash2`, `RedHatAI/Qwen3-8B-speculator.dflash`, `RedHatAI/Qwen3-8B-speculator.peagle`, `RedHatAI/Qwen3-8B-speculator.eagle3`, `RedHatAI/Qwen3-8B-Thinking-speculator.eagle3`. All Apache-2.0, not gated; training data English chat/code/math mixes. | https://huggingface.co/RedHatAI/Qwen3-4B-speculator.dspark, https://huggingface.co/RedHatAI/Qwen3-4B-speculator.dflash2, https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash, https://huggingface.co/RedHatAI/Qwen3-8B-speculator.peagle (card excerpts read via raw README; OPENED) |
| 27 | 2026-11-25 | HF API models | `Qwen3-8B dspark`, `Qwen3-8B-speculator.dspark`, `Qwen3-8B dflash2`, `dspark` | `deepseek-ai/dspark_qwen3_8b_block7`, `deepseek-ai/dspark_qwen3_4b_block7` (also `deepseek-ai/dflash_qwen3_4b_block7`, `deepseek-ai/eagle3_qwen3_{4b,8b}_ttt7`) — no README, no licence field; `RedHatAI/Qwen3-8B-speculator.dflash2` (Apache-2.0); `mgoin/Qwen3-8B-speculator.dspark`; many community checkpoints. | metadata only for deepseek-ai repos; RedHatAI dflash2 card read |
| 28 | 2026-11-25 | HF API models | `Qwen3-4B eagle3`, `Qwen3-8B eagle3`, `Qwen3-8B mtp` | Many EAGLE-3 heads (AngelSlim, Tengyunw, etc.); `AngelSlim/Qwen3-8B-MTP-TTT3` card describes MTP + DFly drafters for Qwen3-8B (vLLM PR 50246) with self-reported English/code AL table (MTP 3.24, DFlash 4.57, DSpark 5.32, DFly 5.41 avg). No Indic evaluation on any card read. | https://huggingface.co/AngelSlim/Qwen3-8B-MTP-TTT3 (OPENED) |

## Conclusions as of 2026-11-25

These conclusions reflect only the searches above. They are not proof of absence.

1. **Closest overlapping prior work: arXiv 2611.12464, "Phrase-Level Speculative Decoding on a Frozen Target" (Naidu, submitted 9 Nov 2026).** Per its HTML, it reports Indic speed-ups with a phrase-vocabulary drafter head: 1.82× Telugu, 1.49× Kannada, 1.29× Hindi on **Qwen3-4B** (Sec. 4 item 1, Sarvam vocabulary), and Gemma-based results that include Tamil and Telugu (Table 6, Gemma 4 E2B on T4, vs MTP). It also measures Qwen3-4B fertility (Hindi 5.00, Telugu 11.73 tokens/word; "253 Indic-script tokens of 151,669"). **Overlap with this project:** it measures speculative speed-ups on Dravidian languages with a Qwen3 target and discusses Qwen3 tokenizer fertility. **Apparent differences (to re-confirm against the full PDF):** it proposes a new phrase-level method rather than evaluating existing drafters; it does not appear to evaluate DFlash, EAGLE-3 with pruned vs full vocabulary, or n-gram drafters on Indic text; it uses HF `generate()` on one A5000/T4 with no vLLM/SGLang and no concurrency sweep; Malayalam does not appear; it does not discuss byte tokens; and it states "the high-fertility case is largely unrun." The project should cite it and position itself as an engine-level, multi-drafter, load-aware evaluation with a byte-level mechanism analysis. It should not claim "first speculative-decoding measurements on Telugu/Kannada with Qwen3."
2. **The C8 wording needs narrowing.** `sulabhkatiyar/eagle3-sarvam-30b` is a public Apache-2.0 EAGLE-3 head whose training mix includes Tamil, Telugu, Kannada and Malayalam, with self-reported Tamil speed-up 2.137× on vLLM (n=10 prompts). Its target is Sarvam-30B (vocab 262,144, draft vocab = full target vocab), not Qwen3. "No public EAGLE-3 drafter for Tamil/Telugu/Kannada/Malayalam" therefore appears **contradicted as stated**. The scoped claim "none for **Qwen3-4B/8B**" still holds as a negative search result.
3. **DFlash on Indic text:** no evaluation found. The only non-English DFlash data point is the aggregate SPEED-Bench "multilingual" row in an NVIDIA blog post (Llama 3.1 8B: EAGLE-3 1.4× vs DFlash 2.4×), with no per-language results. SPEED-Bench's multilingual category appears to contain no Dravidian language (BN and GU are the only Indic codes in Table 2). The project's core DFlash-vs-pruned-EAGLE-3-on-Dravidian comparison appears not to have been published yet, as far as these searches show.
4. **Byte-fragment drafting:** no prior work found that targets byte-fragment tokens of a byte-level-BPE model for under-merged scripts. Two adjacent lines exist: DictSpec (UNLP 2026, n-gram dictionaries for high-fertility non-Latin scripts, vLLM plugin), which is close in spirit to deliverable D6 and should be cited and compared, and byte-level LM speculative decoding (EXCERPT-ONLY).
5. **SPEED-Bench Indic extension:** none found.
6. **Stronger or newer public baselines for Qwen3-4B/8B exist** and the arm list (Section 8.1) should consider them: Red Hat DSpark (Qwen3-4B), DFlash2 (Qwen3-4B, Qwen3-8B), DFlash (Qwen3-8B, Red Hat retrain), P-EAGLE (Qwen3-8B), DeepSeek-released `dspark_qwen3_{4b,8b}_block7` (no card or licence), and AngelSlim MTP/DFly (Qwen3-8B, needs a vLLM PR). All were trained on English, code or math mixes, as far as the cards say. Record their licences, revisions and engine support before adding any as an arm.

## Round 2 (2026-11-25): candidate additions (owner asked "what else is novel")

All results below are from search-result excerpts (EXCERPT-ONLY) unless stated. Full texts not yet opened.

| # | Idea | Query (WebSearch, extended unless noted) | Closest hits | Verdict (search-based, as of 2026-11-25) |
| --- | --- | --- | --- | --- |
| N1 | Training-free draft-vocabulary *extension* of a pruned EAGLE-3 head (transplant target lm_head rows for missing tokens) | "training-free draft vocabulary extension EAGLE-3 copy target lm_head rows missing tokens speculative decoding"; "speculative decoding draft vocabulary expansion low-resource languages without retraining draft head missing tokens multilingual" | VocabTrim arXiv 2506.22694 (pruning, opposite direction); NanoSpec arXiv 2605.26444 (in-context vocab reduction); SlimSpec (Nebius blog; avoids cutting vocab); R-PHRASE 2611.12464 (trains a phrase head; notes 32,000/151,936 tokens undraftable on Qwen3-4B); R-INDIC-EAGLE (retrains head) | No source found doing training-free extension. NEGATIVE-SEARCH |
| N2 | UTF-8-validity mask on draft logits (lossless; drafts that cannot complete the open character are removed) | "UTF-8 byte validity constraint draft tokens speculative decoding byte-level BPE partial character"; "tokenizer-aware speculative decoding constrained drafting invalid tokens masked draft logits lossless acceptance improvement" | arXiv 2611.13607 (UTF-8 validity mask for *sampling* in a byte-level model; warns about mask lag under pipelined scheduling); arXiv 2606.14122 (UTF-8 validity DFA, evaluation only); vLLM PR #69360 (grammar mask on *draft* logits for structured output, lossless); Grammar-as-a-Draft arXiv 2611.28349 (grammar as drafter, not lossless) | Mask concept exists for sampling and for grammar-constrained drafting; no source applies it to byte-fragment natural-language text in speculative decoding. NEGATIVE-SEARCH for the specific combination; prior art must be cited |
| N3 | Energy per word/char per language and arm, consumer GPU | "speculative decoding energy per token consumer GPU multilingual joules per word measurement"; "\"Only Accepted Tokens Count\" energy accounting speculative decoding latency targets" (standard) | arXiv 2610.40287 (energy accounting under latency targets, production engines, two GPUs); arXiv 2602.09113 (EACL Findings 2026, energy savings of spec. decoding); arXiv 2611.16714 (memory-based drafting, consumer hw, legal/medical); arXiv 2604.16368 (Polish, Apple Silicon, J/token) | Energy accounting of spec. decoding is published; no multilingual / per-word / Indic measurement found. Novelty is only the language axis |
| N4 | EAGLE-3 / DFlash on consumer Blackwell sm_120 | "EAGLE-3 DFlash RTX 5090 sm_120 speculative decoding profiling vLLM SGLang consumer Blackwell" (standard) | NVIDIA DFlash blog (B200); LMSYS DFlash/Spec V2 blog 2026-06-15 (Qwen3-4B, datacenter); Baseten DFlash blog; vLLM AMD blog | No consumer sm_120 results found |
| — | (found while searching) | — | SSSD arXiv 2411.05894: training-free n-gram + hardware-aware speculation, reports more uniform speed-ups across languages | Candidate model-free baseline next to A1 / DictSpec |
| — | EAGLE-4 / low-rank draft head | "EAGLE-4 speculative decoding low-rank candidate head avoids full vocabulary LM head draft" (standard, 2026-11-25) | fal.ai "eagle 4 technical deepdive" (Oct 2026, only source naming EAGLE-4); SlimSpec arXiv 2605.10453 (low-rank draft LM head, different mechanism) | Possible A8 if Qwen3 weights are public; not verified |
