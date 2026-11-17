r"""11-10 (user: KL as corroborating evidence for the account, stated positively, no hard threshold; the 2D figure, the statistics,
and the edge cases and architecture stated clearly). Edits the KL results subsection in both languages: the readout reading of the
account, the permutation p values (kl_boundary_stats_20261110.md: main p = 1.0e-05, natural p = 1.9e-04), the single-stream
separation, and the two four-stream models. CRLF kept; every old string must match once.
usage: fix_v5_kl_support_20261110.py PAPER_SOURCE"""
import sys
from pathlib import Path

S = Path(sys.argv[1])
E = {
    'en': [
        ('to \\(\\mathrm{KL}(U\\Vert S)\\), the divergence between the two external sources (Section~\\ref{sec:setup_nexttoken}; Appendix~\\ref{app:nexttoken}).',
         'to \\(\\mathrm{KL}(U\\Vert S)\\), the divergence between the two external sources (Section~\\ref{sec:setup_nexttoken}; Appendix~\\ref{app:nexttoken}). On the account of Section~\\ref{sec:theory}, this projection is one more readout of the history (Figure~\\ref{fig:self_readout}a, bottom row). Where it is aligned with \\(u\\), it sets the model\'s own message apart without any question, so the next-token prediction gives evidence on the self-index that does not depend on the attribution question.'),
        ('with a Spearman correlation of \\(\\rho = 0.82\\) over the twenty-five models and of 0.78 among the thirteen with a self-index.',
         'with a Spearman correlation of \\(\\rho = 0.82\\) over the twenty-five models (one-sided permutation \\(p = 1\\times10^{-5}\\)) and of 0.78 among the thirteen with a self-index.'),
        ('Qwen3.8-Flash-Next and GLM-5.3-Flash, whose content share is a mean over four residual streams, lie at 1.01 and 1.13.',
         'Among the twenty-three models with a single residual stream, dense or mixture-of-experts, every model that carries the index on the content at 69\\% of its layers or more has a higher ratio than every model that carries it at 47\\% or less, and no model lies between 1.11 and 2.0. The two models with four residual streams lie at the lower end: Qwen3.8-Flash-Next at 1.01, although its four streams carry the index on the content at 79\\% of the layers on average, and GLM-5.3-Flash at 1.13 (40\\%). In these models the content share is a mean over the four streams, whereas the next token is predicted from a mixture of the streams.'),
        ('with \\(\\rho = 0.87\\) over thirteen models.', 'with \\(\\rho = 0.87\\) over thirteen models (\\(p = 2\\times10^{-4}\\)).')],
    'zh': [
        ('之比中较小的一个来概括（第~\\ref{sec:setup_nexttoken} 节，附录~\\ref{app:nexttoken}）。',
         '之比中较小的一个来概括（第~\\ref{sec:setup_nexttoken} 节，附录~\\ref{app:nexttoken}）。按第~\\ref{sec:theory} 节的解释，这个投影也是读取历史的一条读出（图~\\ref{fig:self_readout}a 最下一行）。它与 \\(u\\) 对齐时，不需要任何问题就把模型自己的消息区分开，因此下一个 token 预测给出一条不依赖归属问题的、关于自我索引的证据。'),
        ('二十五个模型上的 Spearman 相关为 \\(\\rho = 0.82\\)，在十三个有自我索引的模型中为 0.78。',
         '二十五个模型上的 Spearman 相关为 \\(\\rho = 0.82\\)（单侧置换检验 \\(p = 1\\times10^{-5}\\)），在十三个有自我索引的模型中为 0.78。'),
        ('Qwen3.8-Flash-Next 与 GLM-5.3-Flash 的内容覆盖是四条残差流的平均，它们的比值为 1.01 与 1.13。',
         '在 23 个只有一条残差流的模型（dense 或 MoE）中，凡在 69\\% 或更多的层上于内容上带有索引的模型，比值都高于在 47\\% 或更少的层上带有索引的每一个模型，没有模型落在 1.11 与 2.0 之间。两个有四条残差流的模型落在低端：Qwen3.8-Flash-Next 为 1.01，尽管它的四条流平均在 79\\% 的层上于内容上带有索引，GLM-5.3-Flash 为 1.13（40\\%）。在这两个模型中，内容覆盖是四条流的平均，而下一个 token 是由各条流的混合预测的。'),
        ('十三个模型上 \\(\\rho = 0.87\\)。', '十三个模型上 \\(\\rho = 0.87\\)（\\(p = 2\\times10^{-4}\\)）。')]}
for lang, reps in E.items():
    p = S / f'self_{lang}_body.tex'
    t = p.read_text(encoding='utf-8')
    for a, b in reps:
        assert t.count(a) == 1, (lang, a[:40])
        t = t.replace(a, b)
    p.write_text(t, encoding='utf-8', newline='\r\n')
    print(lang, 'ok')
