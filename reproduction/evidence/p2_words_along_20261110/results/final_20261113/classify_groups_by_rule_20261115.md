# Groups by rule (11-15)

Rule: 0-based layer i (output of block i+1) qualifies when i >= L/3 and the share of held-out texts with
R_A smallest is at least the threshold, on the message text or at the response onset. 80% is the paper's threshold.

| threshold | order | self-reference | self-index only | neither | with index | nesting P (main) | changed from the paper |
|---|---|---|---|---|---|---|---|
| 70% | main | 7 | 9 | 9 | 16 | C(16,7)/C(25,7) = 0.0238 | gemma4_31b_en neither -> self-index only; olmo2_13b_en neither -> self-index only; moonlight16b_a3b_instruct neither -> self-index only |
| 70% | natural | 5 | 9 | 3 | 14 | - | seed_oss_36b neither -> self-index only |
| 80% | main | 7 | 6 | 12 | 13 | C(13,7)/C(25,7) = 0.0036 | none |
| 80% | natural | 5 | 8 | 4 | 13 | - | none |
| 90% | main | 7 | 6 | 12 | 13 | C(13,7)/C(25,7) = 0.0036 | none |
| 90% | natural | 5 | 8 | 4 | 13 | - | none |
