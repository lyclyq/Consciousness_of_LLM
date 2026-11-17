# 保持率实验汇总

M1 = 第一个三分之一之后的索引层数，依次为 k = 1、2、3、4、5、完整；M2 = A–U 共同部分大小保持 [95% 区间]；M3 = 方向余弦。
"按原文" = 协议第 5 节（主表分组）；"自然顺序重分组" = 协议修订第 6 条与第三步预注册第 5 节的读法。

| 步 | 顺序 | 设置 | 主表分组 | 留出 | M1 | M2 | M3 | 按原文 | 自然顺序重分组 |
|---|---|---|---|---|---|---|---|---|---|
| 1-2 | main | exaone4_32b_en | index | 75 | 42 42 42 42 42 42 | 0.697 [0.643, 0.771] | 0.705 | FAILS |  |
| 1-2 | main | glm46v_flash | index | 75 | 26 26 26 26 26 26 | 1.044 [0.999, 1.107] | 0.765 | holds |  |
| 1-2 | main | gpt_oss_20b_en | index | 75 | 15 16 16 16 16 16 | 1.027 [1.001, 1.043] | 0.932 | holds |  |
| 1-2 | main | granite41_30b_en | index | 75 | 14 14 15 14 15 15 | 0.821 [0.754, 0.862] | 0.778 | holds |  |
| 1-2 | main | mimo_v26_distill_qwen9b | index | 75 | 21 21 21 21 21 21 | 0.947 [0.895, 1.007] | 0.861 | holds |  |
| 1-2 | main | phi4_en | index | 75 | 26 26 26 26 26 26 | 1.077 [1.001, 1.153] | 0.808 | holds |  |
| 1-2 | main | qwen38_27b | index | 75 | 40 42 42 42 42 42 | 0.976 [0.856, 1.112] | 0.673 | holds |  |
| 1-2 | main | qwen38_27b_think | index | 75 | 42 42 42 42 42 42 | 1.115 [1.086, 1.142] | 0.787 | holds |  |
| 1-2 | main | r1_distill_llama70b | index | 75 | 53 53 53 53 53 53 | 1.098 [1.075, 1.118] | 0.819 | holds |  |
| 1-2 | main | apertus8b_en | no_index | 75 | 0 0 0 0 0 0 | 0.807 [0.754, 0.912] | 0.703 | holds |  |
| 1-2 | main | ernie45_21b_a3b_thinking | no_index | 75 | 0 0 0 0 0 0 | 0.918 [0.845, 0.990] | 0.711 | holds |  |
| 1-2 | main | gemma4_31b_en | no_index | 75 | 0 0 0 0 0 0 | 0.888 [0.768, 1.049] | 0.527 | holds |  |
| 1-2 | main | mimo7b_rl_0530 | no_index | 75 | 0 0 0 0 0 0 | 1.385 [1.314, 1.486] | 0.617 | holds |  |
| 1-2 | main | mimo_vl7b_rl_2508 | no_index | 75 | 0 0 0 0 0 0 | 0.338 [0.216, 0.676] | 0.683 | holds |  |
| 1-2 | main | moonlight16b_a3b_instruct | no_index | 75 | 0 0 0 0 0 0 | 0.957 [0.851, 1.072] | 0.671 | holds |  |
| 1-2 | main | seed_oss_36b | no_index | 75 | 0 0 0 0 0 0 | 0.915 [0.808, 1.061] | 0.537 | holds |  |
| 1-2 | natural | exaone4_32b_en | index | 75 | 42 42 42 42 42 42 | 1.082 [1.020, 1.111] | 0.778 | holds | holds |
| 1-2 | natural | glm46v_flash | index | 75 | 6 9 6 5 7 3 | 1.502 [1.398, 1.624] | 0.814 | holds | holds |
| 1-2 | natural | gpt_oss_20b_en | index | 75 | - - - - - - | 0.934 [0.921, 0.947] | 0.943 | holds | holds |
| 1-2 | natural | granite41_30b_en | index | 75 | 42 41 42 41 42 42 | 1.067 [1.029, 1.110] | 0.868 | holds | holds |
| 1-2 | natural | mimo_v26_distill_qwen9b | index | 75 | 21 21 21 21 21 21 | 1.109 [1.062, 1.171] | 0.832 | holds | holds |
| 1-2 | natural | phi4_en | index | 75 | 26 26 26 26 26 26 | 1.005 [0.979, 1.036] | 0.909 | holds | holds |
| 1-2 | natural | qwen38_27b | index | 75 | - - - - - - | 1.368 [1.266, 1.486] | 0.853 | holds | holds |
| 1-2 | natural | qwen38_27b_think | index | 75 | - - - - - - | 1.065 [1.054, 1.075] | 0.856 | holds | holds |
| 1-2 | natural | r1_distill_llama70b | index | 75 | - - - - - - | 1.072 [1.049, 1.095] | 0.847 | holds | holds |
| 1-2 | natural | apertus8b_en | no_index | 75 | - - - - - - | 1.060 [1.011, 1.106] | 0.853 | n/a | n/a |
| 1-2 | natural | ernie45_21b_a3b_thinking | no_index | 75 | 18 18 18 18 18 18 | 0.979 [0.935, 1.024] | 0.730 | FAILS | holds |
| 1-2 | natural | gemma4_31b_en | no_index | 75 | 35 35 36 35 36 38 | 0.957 [0.897, 1.084] | 0.798 | FAILS | holds |
| 1-2 | natural | mimo7b_rl_0530 | no_index | 75 | 24 24 24 24 24 18 | 0.942 [0.907, 0.979] | 0.691 | FAILS | holds |
| 1-2 | natural | mimo_vl7b_rl_2508 | no_index | 75 | 0 0 0 1 1 1 | 0.472 [0.322, 0.856] | 0.296 | FAILS | FAILS |
| 1-2 | natural | moonlight16b_a3b_instruct | no_index | 75 | 18 18 18 18 18 18 | 1.081 [0.970, 1.194] | 0.825 | FAILS | holds |
| 1-2 | natural | seed_oss_36b | no_index | 75 | 2 0 0 0 1 0 | 1.076 [0.970, 1.152] | 0.728 | FAILS | FAILS |
| 3 | main | gpt_oss_20b_en | index | 200 | 15 16 16 16 16 16 | 0.976 [0.965, 0.988] | 0.929 | holds |  |
| 3 | main | mimo_v26_distill_qwen9b | index | 200 | 21 21 21 21 21 21 | 0.992 [0.963, 1.018] | 0.764 | holds |  |
| 3 | main | qwen38_27b | index | 200 | 42 42 42 42 42 42 | 0.859 [0.826, 0.895] | 0.483 | holds |  |
| 3 | main | qwen38_27b_think | index | 200 | 42 42 42 42 42 42 | 1.105 [1.096, 1.114] | 0.793 | holds |  |
| 3 | main | apertus8b_en | no_index | 200 | 0 0 0 0 0 0 | 0.864 [0.830, 0.901] | 0.624 | holds |  |
| 3 | main | mimo_vl7b_rl_2508 | no_index | 200 | 0 0 0 0 0 0 | 0.586 [0.476, 0.694] | 0.477 | holds |  |
| 3 | natural | gpt_oss_20b_en | index | 200 | - - - - - - | 0.949 [0.943, 0.955] | 0.927 | holds | holds |
| 3 | natural | mimo_v26_distill_qwen9b | index | 200 | 21 21 21 21 21 21 | 1.206 [1.177, 1.234] | 0.763 | holds | holds |
| 3 | natural | qwen38_27b | index | 200 | - - - - - - | 1.222 [1.197, 1.248] | 0.521 | holds | holds |
| 3 | natural | qwen38_27b_think | index | 200 | - - - - - - | 1.064 [1.060, 1.068] | 0.830 | holds | holds |
| 3 | natural | apertus8b_en | no_index | 200 | - - - - - - | 0.953 [0.920, 0.987] | 0.611 | n/a | n/a |
| 3 | natural | mimo_vl7b_rl_2508 | no_index | 200 | 0 0 0 1 1 1 | 0.615 [0.532, 0.730] | 0.580 | FAILS | FAILS |
