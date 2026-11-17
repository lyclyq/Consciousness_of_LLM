| depth | Spearman rho | p (one-sided) | slope of log KL ratio on -log R_A | 95% interval | Pearson (log-log) |
|---|---|---|---|---|---|
| 0.25 | +0.84 | 0.0000 | 2.07 | [1.60, 2.96] | +0.83 |
| 0.50 | +0.92 | 0.0000 | 1.81 | [1.41, 2.88] | +0.88 |
| 0.75 | +0.97 | 0.0000 | 1.66 | [1.36, 2.22] | +0.93 |
| 1.00 | +0.94 | 0.0000 | 1.57 | [1.25, 2.13] | +0.92 |

share of layers with the self-index on message content: rho = +0.82

| model | group | KL ratio | R_A last layer | 1 / R_A^2 |
|---|---|---|---|---|
| GLM-4.6V-Flash | self-index only | 64.84 | 0.131 | 58.29 |
| gpt-oss-20b | self-reference | 34.67 | 0.066 | 227.39 |
| EXAONE-4.0.1-32B | self-reference | 12.77 | 0.064 | 247.17 |
| gpt-oss-120b | self-reference | 9.66 | 0.420 | 5.67 |
| MiMo-V2.6-Distill-Qwen-9B | self-reference | 5.15 | 0.283 | 12.48 |
| Qwen3.8-27B | self-reference | 2.69 | 0.355 | 7.94 |
| phi-4 | self-index only | 2.34 | 0.528 | 3.59 |
| GLM-4.7-Flash | self-index only | 2.00 | 0.490 | 4.17 |
| GLM-5.3-Flash | self-reference | 1.13 | 0.537 | 3.47 |
| OLMo-2-7B | neither | 1.11 | 0.940 | 1.13 |
| Gemma-4-31B | neither | 1.07 | 0.962 | 1.08 |
| OLMo-2-13B | neither | 1.05 | 0.974 | 1.05 |
| Qwen3.8-Flash-Next | self-reference | 1.01 | 0.791 | 1.60 |
| Moonlight-16B-A3B | neither | 0.86 | 0.948 | 1.11 |
| Granite-4.2-30B | self-index only | 0.80 | 1.241 | 0.65 |
| Granite-4.1-30B | self-index only | 0.64 | 1.181 | 0.72 |
| MiMo-7B-RL | neither | 0.57 | 1.579 | 0.40 |
| Qwen3.5-27B | neither | 0.54 | 1.110 | 0.81 |
| Apertus-8B | neither | 0.39 | 1.528 | 0.43 |
| Qwen3.6-27B | neither | 0.33 | 0.933 | 1.15 |
| ERNIE-4.5-21B-A3B | neither | 0.31 | 1.591 | 0.40 |
| Qwen3.6-35B-A3B | neither | 0.22 | 1.640 | 0.37 |
| MiMo-VL-7B-RL-2508 | neither | 0.20 | 1.907 | 0.27 |
| Seed-OSS-36B | neither | 0.16 | 1.870 | 0.29 |
| DeepSeek-R1-Distill-Llama-70B | self-index only | 0.05 | 1.826 | 0.30 |
