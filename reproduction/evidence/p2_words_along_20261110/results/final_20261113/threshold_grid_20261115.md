# Threshold robustness grid (11-15)

Rows: threshold of the self-index criterion (80% is the paper's). Groups SR self-reference, IO self-index only, N neither. P one-sided permutation (seeds of the paper's scripts); nesting P exact.

| threshold | main: with index / SR / IO / N | nesting P (main) | natural: with index / SR / IO / N | main KL self/ext rho (P) | main share kept own-ext rho (P) | role-general rho, KL / share kept | main medians KL self/ext (N / IO / SR) | main medians share kept own-ext (N / IO / SR) | natural rho KL (P) | natural rho share kept (P) | pooled rho KL (P) | pooled rho share kept (P) | pooled levels N / index one order / index both / SR one order / SR both |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70% | 16 / 7 / 9 / 9 | C(16,7)/C(25,7) = 0.0238 | 14 / 5 / 9 / 3 | 0.74 (0.0000) | 0.55 (0.0036) | -0.23 / -0.10 | 0.33 / 1.05 / 5.15 | -0.04 / +0.03 / +0.14 | 0.59 (0.0175) | 0.73 (0.0023) | 0.70 (0.0027) | 0.54 (0.0238) | 1 / 4 / 5 / 1 / 3 |
| 80% | 13 / 7 / 6 / 12 | C(13,7)/C(25,7) = 0.0036 | 13 / 5 / 8 / 4 | 0.68 (0.0001) | 0.61 (0.0009) | -0.22 / -0.27 | 0.46 / 1.27 / 5.15 | -0.04 / +0.09 / +0.14 | 0.67 (0.0080) | 0.79 (0.0009) | 0.73 (0.0021) | 0.70 (0.0037) | 2 / 5 / 4 / 0 / 3 |
| 90% | 13 / 7 / 6 / 12 | C(13,7)/C(25,7) = 0.0036 | 13 / 5 / 8 / 4 | 0.68 (0.0001) | 0.61 (0.0009) | -0.22 / -0.27 | 0.46 / 1.27 / 5.15 | -0.04 / +0.09 / +0.14 | 0.67 (0.0080) | 0.79 (0.0009) | 0.73 (0.0021) | 0.70 (0.0037) | 2 / 5 / 4 / 0 / 3 |

No model recognizes its own messages without a self-index at any threshold, in either order.

| threshold | order | model | group at 80% | group at this threshold | deciding layers beyond L/3 (0-based index: share) |
|---|---|---|---|---|---|
| 70% | main | Gemma-4-31B | neither | self-index only | content 57: 0.707, 58: 0.773 |
| 70% | main | OLMo-2-13B | neither | self-index only | content 21: 0.760, 22: 0.733, 25: 0.707, 26: 0.707, 28: 0.733, 29: 0.733, 30: 0.720, 31: 0.720, 32: 0.707, 33: 0.707, 34: 0.707 |
| 70% | main | Moonlight-16B-A3B | neither | self-index only | content 20: 0.707 |
| 70% | natural | Seed-OSS-36B | neither | self-index only | onset 22: 0.707 |
