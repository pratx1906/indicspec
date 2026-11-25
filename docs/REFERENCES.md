# References

The base reference list (IDs `R-*`, URLs, status at project start) is
[PROJECT_PROMPT.md §14](../PROJECT_PROMPT.md#14-references). This file extends it using the same fields
(ID · reference · URL · status · note) and records licences. Statuses here reflect what *this repository*
verified; see [CLAIMS.md](CLAIMS.md) and [research/](research/).

## Added during the project

| ID | Reference | URL | Status | Licence | Note |
| --- | --- | --- | --- | --- | --- |
| R-UCD | Unicode Character Database (as bundled with CPython `unicodedata`) | https://www.unicode.org/ucd/ | MEASURED (via test) | Unicode licence | Block ranges, C21 |

Data-source references (FLORES via IndicGenBench, XQuAD-IN, Belebele, Aya evaluation suite, IN22) are
described with their licences in [research/data_sources.md](research/data_sources.md) and will be moved
here with IDs once the benchmark design is frozen (WS2).

### Benchmark sources (WS2, pinned; licences checked from README front matter at the revision)

| ID | Reference | URL | Status | Licence | Note |
| --- | --- | --- | --- | --- | --- |
| R-DATA-FLORES-IN | IndicGenBench Flores-IN (FLORES-200 dev/devtest), rev `f8650438298df086750ff4973661bb58a201a5ee` | https://huggingface.co/datasets/google/IndicGenBench_flores_in | OPENED (files) | CC-BY-SA-4.0 | translation, translation_xx2en; canary: no pre-training |
| R-DATA-XQUAD-IN | IndicGenBench XQuAD-IN, rev `32026f0bcb0eed9aea0418694448c143c98acb0a` | https://huggingface.co/datasets/google/IndicGenBench_xquad_in | OPENED (files) | CC-BY-SA-4.0 | qa |
| R-DATA-BELEBELE | Belebele, rev `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` | https://huggingface.co/datasets/facebook/belebele | OPENED (files) | CC-BY-SA-4.0 | summarisation passages; card: test use only |
| R-DATA-AYA-EVAL | Aya evaluation suite, `dolly_machine_translated`, rev `e69f652e99144bf9e5019caa6c940fa1e9a24ba3` | https://huggingface.co/datasets/CohereLabs/aya_evaluation_suite | OPENED (files) | Apache-2.0 (Dolly upstream CC-BY-SA-3.0) | chat; NLLB-translated |
| R-DATA-IN22-CONV | IN22-Conv | https://huggingface.co/datasets/ai4bharat/IN22-Conv | NOT DOWNLOADED (gated) | CC-BY-4.0 (card) | planned held-out domain |
