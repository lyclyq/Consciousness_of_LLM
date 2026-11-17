# Own-message markers, content geometry and self-attribution (make_template_markers.py)

| model | Assistant header | User header | own-message marker | geometry (mode): layers apart / all, median R_A | P(you) own / User, no thinking | "you" own / User, thinking |
|---|---|---|---|---|---|---|
| gemma4_31b | `'<|turn>model\n'` | `'<|turn>user\n'` | role word only | (non_thinking) 0 / 60, 1.06 | 0.00 / 0.00 | 0.03 / 0.00 |
| granite41_30b | `'<|start_of_role|>assistant<|end_of_role|>'` | `'<|start_of_role|>user<|end_of_role|>'` | role word only | (non_thinking) 17 / 64, 1.27 | 0.03 / 0.04 | - |
| qwen35_122b | `'<|im_start|>assistant\n'` | `'<|im_start|>user\n'` | role word only | (non_thinking) 2 / 48, 1.19 | 0.15 / 0.04 | - |
| qwen35_27b | `'<|im_start|>assistant\n'` | `'<|im_start|>user\n'` | role word only | (non_thinking) 8 / 64, 1.30 | 0.06 / 0.04 | 0.02 / 0.00 |
| qwen36_27b | `'<|im_start|>assistant\n'` | `'<|im_start|>user\n'` | role word only | (non_thinking) 6 / 64, 1.51 | 0.05 / 0.03 | 0.00 / 0.00 |
| qwen36_35b | `'<|im_start|>assistant\n'` | `'<|im_start|>user\n'` | role word only | (non_thinking) 1 / 40, 1.43 | 0.04 / 0.04 | 0.00 / 0.00 |
| r1_llama70b | `'<｜Assistant｜>'` | `'<｜User｜>'` | role word only | (thinking) 0 / 80, 1.77 | - | 0.09 / 0.14 |
| seed_oss_36b | `'<seed:bos>assistant\n'` | `'<seed:bos>user\n'` | role word only | (non_thinking) 0 / 64, 1.72 | 0.00 / 0.00 | 0.00 / 0.00 |
| exaone4_32b | `'[|assistant|]\n<think>\n\n</think>\n\n'` | `'[|user|]\n'` | '<think>\n\n</think>\n\n' | (non_thinking) 64 / 64, 0.09 | 0.13 / 0.16 | 0.17 / 0.03 |
| glm46v_flash | `'<|assistant|>\n<think></think>\n'` | `'<|user|>\n'` | '<think></think>\n' | (non_thinking) 39 / 40, 0.10 | 0.37 / 0.36 | 0.09 / 0.00 |
| glm47_flash | `'<|assistant|></think>'` | `'<|user|>'` | '</think>' | (non_thinking) 27 / 47, 0.64 | 0.27 / 0.23 | 0.08 / 0.01 |
| glm53_flash | `'<|assistant|><think></think>'` | `'<|user|>'` | '<think></think>' | (thinking) 1 / 45, 0.88 | - | 0.46 / 0.01 |
| gpt_oss_120b | `'<|start|>assistant<|channel|>final<|message|>'` | `'<|start|>user<|message|>'` | '<|channel|>final' | (thinking) 19 / 36, 0.62 | - | 0.97 / 0.00 |
| gpt_oss_20b | `'<|start|>assistant<|channel|>final<|message|>'` | `'<|start|>user<|message|>'` | '<|channel|>final' | (thinking) 19 / 24, 0.43 | - | 0.78 / 0.00 |
| granite42_30b | `'<|im_start|>assistant\n<think></think>'` | `'<|im_start|>user\n'` | '<think></think>' | (non_thinking) 22 / 64, 0.83 | 0.06 / 0.01 | 0.00 / 0.00 |
| qwen38_27b | `'<|im_start|>assistant\n<think>\n\n</think>\n\n'` | `'<|im_start|>user\n'` | '<think>\n\n</think>\n\n' | (non_thinking) 63-64 / 64, 0.44-0.48 | 0.72 / 0.05 | 0.95 / 0.00 |
| qwen38_flashnext | `'<|im_start|>assistant\n<think>\n\n</think>\n\n'` | `'<|im_start|>user\n'` | '<think>\n\n</think>\n\n' | (non_thinking) 20 / 48, 0.72 | 0.19 / 0.10 | 0.66 / 0.00 |
