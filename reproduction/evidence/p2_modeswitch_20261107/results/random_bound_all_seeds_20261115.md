# The random control over all seeds (eight random directions per layer, all positions)

Change of the rate of "you" from the unedited condition. Per readout: the largest absolute change over its runs.

| readout | runs | random removal, max abs | random installation, max abs | fitted removal (r05q_all) |
|---|---|---|---|---|
| Qwen3.8-27B, single pass | 6 | 0.015 | 0.004 | -0.758 |
| Qwen3.8-27B, thinking | 6 | 0.067 | 0.000 | -0.960 |
| gpt-oss-20b | 6 | 0.047 | 0.007 | -0.760 |
| gpt-oss-120b | 6 | 0.047 | 0.007 | -0.960 |
| MiMo-V2.6-Distill-Qwen-9B | 5 | 0.147 | 0.013 | -0.407 |
| EXAONE-4.0.1-32B | 5 | 0.053 | 0.027 | -0.174 |
| GLM-5.3-Flash | 1 | 0.007 | 0.000 | -0.379 |

Overall: random at most 0.147 (removal 0.147, installation 0.027); fitted removal at least 0.174 in absolute value.
