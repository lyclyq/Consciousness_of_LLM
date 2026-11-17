r"""11-10 (user: "这个比值一开始设置就不太合理，我们还是猜测有一种关系，一开始设置一个完整的阈值判断不太好"): the English KL results
subsection is rewritten as a relation between the next-token separation (the smaller of the two content ratios) and the share of
layers carrying the self-index on the message content, without a pass/fail threshold, counts, or a permutation test. Numbers from
evidence/p2_modeswitch_20261107/results/kl_ratio_distribution_20261110.md (kl_ratio_distribution_20261110.py): Spearman 0.82 over
25 models, 0.78 within the 13 with a self-index, 0.87 over the 13 natural-order models. Replaces the text between the first
sentence of the subsection and the next \section. CRLF kept.
usage: fix_v5_kl_relation_en_20261110.py PAPER_SOURCE"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / 'self_en_body.tex'
t = p.read_text(encoding='utf-8')
i = t.index('The attribution question asks for the source. The model also processes its own message')
j = t.index('\\section{Causal Tests and Alternative Explanations}', i)
assert 'label{sec:res_nexttoken}' in t[i - 200:i] and j - i < 7000
NEW = r"""The attribution question asks for the source. The model also processes its own message when no question asks for it: at every token of the message, it predicts the next one. We compared these predictions between the three roles of the same held-out body, on content, by the smaller of the two ratios of \(\mathrm{KL}(A\Vert U)\) and \(\mathrm{KL}(A\Vert S)\) to \(\mathrm{KL}(U\Vert S)\), the divergence between the two external sources (Section~\ref{sec:setup_nexttoken}; Appendix~\ref{app:nexttoken}).

Under the main probe, this ratio grows with the share of layers at which the model's own message is singled out on its content (Figure~\ref{fig:self_kl}a, Table~\ref{tab:self_kl}), with a Spearman correlation of \(\rho = 0.82\) over the twenty-five models and of 0.78 among the thirteen with a self-index. The eight models that carry the index on the content of the message at 69--100\% of their layers have ratios from 2.0 to 65: GLM-4.7-Flash (2.0), phi-4 (2.3), Qwen3.8-27B (2.7), MiMo-V2.6-Distill-Qwen-9B (5.2), gpt-oss-120b (9.7), EXAONE-4.0.1-32B (12.8), gpt-oss-20b (35), and GLM-4.6V-Flash (65). The models without a self-index, and DeepSeek-R1-Distill-Llama-70B and the two Granite models, which carry it on the content at fewer than half of their layers, have ratios from 0.05 to 1.11, so that their own message differs from an external one at most about as much as the two external ones differ from each other. Qwen3.8-Flash-Next and GLM-5.3-Flash, whose content share is a mean over four residual streams, lie at 1.01 and 1.13.

In natural conversation order the relation is the same (Figure~\ref{fig:self_kl}b, Table~\ref{tab:self_kl_natural}), with \(\rho = 0.87\) over thirteen models. The models that carry the index on the content of the message at more than half of their layers have ratios from 1.6 to 5.9, and those that carry it there at five layers or fewer from 0.14 to 0.72. MiMo-7B-RL has no self-index under the main probe and one at 34 of its 36 layers in natural order, and its ratio rises from 0.57 to 2.0. DeepSeek-V4.1-Flash, which accepts only natural order and carries the index at nearly every layer, has a ratio of 2.4, and Ministral-3-14B, which carries none, ratios of 0.14 and 0.29 in its two versions. Within one model, the separation changes with the self-index. Without the empty reasoning block that its template adds to its own messages, which weakens the self-index of Qwen3.8-27B (Section~\ref{sec:res_inducer}), the full ratio of the model without thinking falls from 308 to 1.4 (interval 0.8 to 3.0).

Most of the full divergence lies in the control tokens in the Qwen models. The body of a User message predicts the end of the turn at nearly every position in Qwen3.8-27B (0.99 without thinking), and without the control tokens the ratios fall from 308 to 4.0 without thinking and from 40 to 8.5 with thinking. Qwen3.6-27B, which has no self-index, has a full ratio of 39 through these tokens alone, against 0.33 on content. In the gpt-oss models, MiMo-V2.6-Distill-Qwen-9B, and the other models, the control tokens carry almost no probability on the body, and the difference lies in the predictions of content.

This difference follows the first level, not the second. GLM-4.6V-Flash, GLM-4.7-Flash, and phi-4 carry a self-index without recognizing their own messages, and their ratios (2.0 to 65) lie among those of the models that do. Qwen3.8-Flash-Next and GLM-5.3-Flash, which recognize their own messages after reasoning, lie at 1.0 and 1.1. The next-token prediction thus reads the index on the message itself, before and apart from any question, and more strongly the more of the depth carries the index on the message, whether or not the model recognizes its own messages. Section~\ref{sec:res_mechanism} shows that the coordinates of the index carry this difference.

"""
t = t[:i] + NEW.replace('\n', '\n') + t[j:]
p.write_text(t, encoding='utf-8', newline='\r\n')
print('ok')
