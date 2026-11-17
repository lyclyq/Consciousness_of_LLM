# Group decomposition (11-13)

## main, retention

| quantity | n | neither | self-index only | self-reference | Spearman with group (p) | Jonckheere p | self-reference above the rest, p |
|---|---|---|---|---|---|---|---|
| self - words | 25 | +0.16 | +0.23 | +0.40 | +0.47 (0.0093) | 0.0075 | 0.0101 |
| self - U-S (self-specific) | 25 | -0.04 | +0.09 | +0.14 | +0.61 (0.0009) | 0.0014 | 0.0043 |
| U-S - words (role-general) | 25 | +0.26 | +0.16 | +0.20 | -0.27 (0.9059) | 0.9244 | 0.7888 |

## main, kl

| quantity | n | neither | self-index only | self-reference | Spearman with group (p) | Jonckheere p | self-reference above the rest, p |
|---|---|---|---|---|---|---|---|
| log10 self / words | 25 | +0.10 | +0.51 | +1.00 | +0.56 (0.0032) | 0.0050 | 0.0032 |
| log10 self / U||S (self-specific) | 25 | -0.34 | +0.10 | +0.71 | +0.68 (0.0001) | 0.0001 | 0.0006 |
| log10 U||S / words (role-general) | 25 | +0.46 | +0.09 | +0.37 | -0.22 (0.8589) | 0.8709 | 0.5314 |

## natural, retention

| quantity | n | neither | self-index only | self-reference | Spearman with group (p) | Jonckheere p | self-reference above the rest, p |
|---|---|---|---|---|---|---|---|
| self - words | 17 | +0.05 | +0.42 | +0.46 | +0.31 (0.1143) | 0.1187 | 0.2540 |
| self - U-S (self-specific) | 13 | -0.23 | +0.11 | +0.23 | +0.79 (0.0009) | 0.0009 | 0.0382 |
| U-S - words (role-general) | 13 | +0.32 | +0.16 | +0.09 | -0.15 (0.7164) | 0.7094 | 0.7703 |

## natural, kl

| quantity | n | neither | self-index only | self-reference | Spearman with group (p) | Jonckheere p | self-reference above the rest, p |
|---|---|---|---|---|---|---|---|
| log10 self / words | 17 | -0.06 | +0.21 | +0.54 | +0.61 (0.0050) | 0.0076 | 0.0144 |
| log10 self / U||S (self-specific) | 13 | -0.53 | +0.31 | +0.37 | +0.67 (0.0080) | 0.0081 | 0.1431 |
| log10 U||S / words (role-general) | 13 | +0.61 | +0.03 | +0.17 | -0.24 (0.8035) | 0.7884 | 0.7186 |

## Per model, next-token KL (log10)

| order | model | group | self / words | self / U||S | U||S / words | self pair |
|---|---|---|---|---|---|---|
| main | Qwen3.8-27B | self-reference | +1.22 | +0.43 | +0.79 | A-S |
| main | Qwen3.8-Flash-Next | self-reference | +0.73 | +0.01 | +0.72 | A-S |
| main | gpt-oss-20b | self-reference | +1.91 | +1.54 | +0.37 | A-U |
| main | gpt-oss-120b | self-reference | +1.42 | +0.98 | +0.43 | A-U |
| main | GLM-5.3-Flash | self-reference | +0.36 | +0.05 | +0.31 | A-S |
| main | EXAONE-4.0.1-32B | self-reference | +0.46 | +1.11 | -0.65 | A-S |
| main | MiMo-V2.6-Distill-Qwen-9B | self-reference | +1.00 | +0.71 | +0.28 | A-S |
| main | GLM-4.6V-Flash | self-index only | +1.68 | +1.81 | -0.13 | A-U |
| main | GLM-4.7-Flash | self-index only | +0.30 | +0.30 | +0.00 | A-S |
| main | DeepSeek-R1-Distill-Llama-70B | self-index only | -1.10 | -1.27 | +0.17 | A-U |
| main | phi-4 | self-index only | +0.79 | +0.37 | +0.42 | A-S |
| main | Granite-4.2-30B | self-index only | +0.72 | -0.09 | +0.81 | A-S |
| main | Granite-4.1-30B | self-index only | -0.97 | -0.19 | -0.78 | A-S |
| main | OLMo-2-13B | neither | +0.65 | +0.02 | +0.63 | A-S |
| main | OLMo-2-7B | neither | +0.74 | +0.05 | +0.69 | A-U |
| main | Apertus-8B | neither | +0.51 | -0.40 | +0.91 | A-U |
| main | Moonlight-16B-A3B | neither | +0.26 | -0.06 | +0.32 | A-U |
| main | MiMo-7B-RL | neither | +0.20 | -0.25 | +0.45 | A-U |
| main | MiMo-VL-7B-RL-2508 | neither | -0.12 | -0.70 | +0.58 | A-U |
| main | ERNIE-4.5-21B-A3B | neither | -0.09 | -0.51 | +0.42 | A-U |
| main | Seed-OSS-36B | neither | -0.19 | -0.80 | +0.61 | A-U |
| main | Gemma-4-31B | neither | +0.34 | +0.03 | +0.31 | A-S |
| main | Qwen3.6-27B | neither | -0.15 | -0.49 | +0.34 | A-U |
| main | Qwen3.6-35B-A3B | neither | -0.18 | -0.65 | +0.47 | A-U |
| main | Qwen3.5-27B | neither | +0.00 | -0.27 | +0.27 | A-U |
| natural | Qwen3.8-27B | self-reference | +1.50 | - | - | A-U |
| natural | gpt-oss-20b | self-reference | +2.00 | - | - | A-U |
| natural | EXAONE-4.0.1-32B | self-reference | +0.37 | +0.77 | -0.40 | A-U |
| natural | Gemma-4-31B | self-reference | +0.27 | -0.17 | +0.45 | A-S |
| natural | DeepSeek-V4.1-Flash | self-reference | +0.54 | +0.37 | +0.17 | A-U |
| natural | GLM-4.6V-Flash | self-index only | +1.06 | +0.31 | +0.75 | A-S |
| natural | Granite-4.1-30B | self-index only | +0.40 | +0.37 | +0.03 | A-S |
| natural | MiMo-7B-RL | self-index only | -0.21 | +0.31 | -0.52 | A-S |
| natural | Moonlight-16B-A3B | self-index only | +0.13 | +0.43 | -0.30 | A-S |
| natural | phi-4 | self-index only | +0.66 | +0.45 | +0.21 | A-S |
| natural | ERNIE-4.5-21B-A3B | self-index only | -0.07 | +0.18 | -0.25 | A-S |
| natural | DeepSeek-R1-Distill-Llama-70B | self-index only | -0.90 | - | - | A-U |
| natural | OLMo-2-7B | self-index only | +0.28 | -0.42 | +0.71 | A-S |
| natural | Seed-OSS-36B | neither | -0.69 | -0.53 | -0.16 | A-S |
| natural | Apertus-8B | neither | +0.24 | - | - | A-U |
| natural | Ministral-3-14B-Instruct | neither | -0.19 | -0.86 | +0.67 | A-S |
| natural | Ministral-3-14B-Reasoning | neither | +0.08 | -0.53 | +0.61 | A-S |
