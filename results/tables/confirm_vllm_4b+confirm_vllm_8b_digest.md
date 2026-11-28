# Digest: confirm_vllm_4b+confirm_vllm_8b (draft, copy numbers into CLAIMS.md)

PREREG commit 6023a78; requests kept 60500; excluded cells 7.

- **H1**: confirmed
  - A3.k3@qwen3-8b: verdict confirmed; per_language ta confirmed, te confirmed, kn confirmed, ml confirmed
  - A4.k3@qwen3-8b: verdict confirmed; per_language ta confirmed, te confirmed, kn confirmed, ml confirmed
- **H2**: inconclusive
  - A1.k8@qwen3-4b: verdict confirmed; per_language ta confirmed, te confirmed, kn confirmed, ml confirmed
  - A5.k15@qwen3-4b: verdict falsified; per_language ta falsified, te falsified, kn falsified, ml falsified
  - A5.k15@qwen3-8b: verdict falsified; per_language ta falsified, te falsified, kn falsified, ml falsified
- **H3**: inconclusive
  - A1.k8@qwen3-4b: verdict confirmed; diff_ci [0.432, 0.491]
  - A5.k15@qwen3-4b: verdict falsified; diff_ci [-0.249, -0.216]
  - A5.k15@qwen3-8b: verdict falsified; diff_ci [-0.341, -0.312]
- **H4**: inconclusive
  - A1.k8@qwen3-4b: verdict confirmed; per_language ta confirmed, te confirmed, kn confirmed, ml confirmed; crossover en 8, ta >32, te >32, kn >32, ml >32
  - A5.k15@qwen3-4b: verdict confirmed; per_language ta falsified, te confirmed, kn confirmed, ml confirmed; crossover en 32, ta 16, te 8, kn 8, ml 8
  - A5.k15@qwen3-8b: verdict inconclusive; per_language ta falsified, te confirmed, kn falsified, ml confirmed; crossover en >8, ta 8, te 4, kn 8, ml 1
- **G1**: YES
  - A1.k8@qwen3-4b: hit True
  - A3.k3@qwen3-8b: hit True
  - A4.k3@qwen3-8b: hit True
  - A5.k15@qwen3-4b: hit True
  - A5.k15@qwen3-8b: hit True
- **G2**: NO
  - A1.k8@qwen3-4b: a True; b False
  - A5.k15@qwen3-4b: a True; b True
  - A5.k15@qwen3-8b: a False; b True
- **G3**: not evaluated (no new drafter yet)

Speed-up vs no speculation, one user, qwen3-4b (95% CI):

| Arm | en | hi | ta | te | kn | ml |
|---|---|---|---|---|---|---|
| A1.k8 | 1.40 (1.31–1.48) | 2.31 (2.14–2.47) | 3.11 (2.85–3.36) | 3.00 (2.75–3.25) | 3.11 (2.85–3.38) | 2.97 (2.73–3.21) |
| A5.k15 | 2.26 (2.11–2.42) | 1.82 (1.74–1.91) | 1.56 (1.49–1.62) | 1.50 (1.46–1.54) | 1.30 (1.26–1.34) | 1.11 (1.09–1.13) |

Speed-up vs no speculation, one user, qwen3-8b (95% CI):

| Arm | en | hi | ta | te | kn | ml |
|---|---|---|---|---|---|---|
| A3.k3 | 1.95 (1.90–2.00) | 1.18 (1.16–1.19) | 1.00 (0.99–1.01) | 1.03 (1.02–1.04) | 1.02 (1.01–1.03) | 0.97 (0.96–0.98) |
| A4.k3 | 1.34 (1.31–1.37) | 1.75 (1.73–1.78) | 0.99 (0.98–1.00) | 1.12 (1.11–1.13) | 1.07 (1.06–1.08) | 0.97 (0.96–0.97) |
| A5.k15 | 2.28 (2.15–2.42) | 1.70 (1.65–1.76) | 1.29 (1.26–1.32) | 1.13 (1.11–1.15) | 1.24 (1.21–1.26) | 0.96 (0.95–0.97) |

