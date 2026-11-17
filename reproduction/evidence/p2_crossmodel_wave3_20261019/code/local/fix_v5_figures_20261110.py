r"""11-10 figures of the revision, both languages: copies the four new figure files from code/local/figures/drafts_20261110 into
paper/source/figures (self_tasks_flow_{en,zh}.pdf with panels e and f, self_readout.pdf with the task-free readout, the new
self_kl_scatter.pdf and self_wording.pdf), rewrites the captions of fig:self_tasks and of panel (a) of fig:self_readout, inserts the
figures fig:self_kl (KL subsection, before the control-token paragraph) and fig:self_wording (after the wording paragraph of
sec:abl_question) with references in the text, and states the main-probe permutation p as p < 10^-4 (no permutation of 100,000
reached the observed rho). CRLF kept; every anchor must match once.
usage: fix_v5_figures_20261110.py ARCHIVE PAPER_SOURCE"""
import re
import shutil
import sys
from pathlib import Path

A, S = Path(sys.argv[1]), Path(sys.argv[2])
D = A / 'code' / 'local' / 'figures' / 'drafts_20261110'
for f in ('self_tasks_flow_en.pdf', 'self_tasks_flow_zh.pdf', 'self_readout.pdf', 'self_kl_scatter.pdf', 'self_wording.pdf'):
    shutil.copyfile(D / f, S / 'figures' / f)

CAP_TASKS = {
    'en': r"""\caption{The four questions of the causal tests (a--d), a variant of the attribution question (e), and the readout without a question (f), with the items of one held-out body and one game. Left: the conversation, in which X is rendered either as the model's own (Assistant) message or as a User message, with identical text. Shaded: the positions that the edit reaches, X and every later prompt position in (a--d), and X alone in (e) and (f) (in (f) its body), where the subspace is fitted on the message alone and the question is not edited. The role header of X, the earlier messages, and the generation prompt are never edited. Right: the expected response when X is the model's own message (top lane) and when it is a User message (bottom lane). Where the source matters (a, b, e), the lanes lead to different answers, and replacing the self-index coordinates of X with those of the other role should move the answer to the other lane. In (f) the lanes lead to different predictions of the next token on X, and the edit should move the prediction to the other lane. Where the source does not matter (c, d), one step serves both lanes, and the edit should leave the answer unchanged. The game replies shown are those of gpt-oss-20b to this item. Text in the messages is abridged at the ellipses.}""",
    'zh': r"""\caption{因果检验的四道题（a--d）、归属问题的一个变体（e）与不提问的读出（f），以一份留出正文与一局游戏的题目为例。左：对话，X 以模型自己的（Assistant）消息或 User 消息渲染，文字相同。阴影：编辑所及的位置，在（a--d）中是 X 及其后所有提示位置，在（e）与（f）中只有 X（（f）中只有其正文），这两种情况的子空间只在消息上拟合，问题不被编辑。X 的角色头、之前的消息与生成提示从不被编辑。右：X 为模型自己的消息时（上道）与为 User 消息时（下道）预期的反应。与来源有关时（a、b、e），两条道通向不同的回答，把 X 的自我索引坐标换成另一角色的坐标，应使回答移到另一条道。（f）中两条道通向对 X 上下一个 token 的不同预测，编辑应使预测移到另一条道。与来源无关时（c、d），一个步骤同时服务两条道，编辑应使回答保持不变。图中的游戏回答是 Qwen3.8-27B 思考模式对这道题的回答。消息中的文字在省略号处有删节。}"""}
READ_A = {
    'en': (r"""(a) The model's own message is shifted by the self-index offset \(u\) over its whole representation (Equation~\ref{eq:self_decomposition}). A readout \(w\) with \(w\cdot u\approx 0\) gives the same value for every role, and the output does not depend on the source; a readout aligned with \(u\) separates the model's own message, and the output follows the source (Equation~\ref{eq:self_readout}).""",
           r"""(a) The model's own message is shifted by the self-index offset \(u\) over its whole representation (Equation~\ref{eq:self_decomposition}). A readout \(w\) with \(w\cdot u\approx 0\) gives the same value for every role, so the output does not depend on the source. A readout aligned with \(u\) separates the model's own message, and the output follows the source (Equation~\ref{eq:self_readout}). The bottom row is a readout that needs no question, the projection toward the next token on the message itself. When it is aligned with \(u\), it sets the model's own message apart without any task, which shows the self-index but not self-reference (Section~\ref{sec:res_nexttoken})."""),
    'zh': (r"""(a) 模型自己的消息在整个表征上被自我索引偏移 \(u\) 平移（式~\ref{eq:self_decomposition}）。\(w\cdot u\approx 0\) 的读出 \(w\) 对每种角色给出相同的值，输出与来源无关；与 \(u\) 对齐的读出把模型自己的消息分离出来，输出随来源改变（式~\ref{eq:self_readout}）。""",
           r"""(a) 模型自己的消息在整个表征上被自我索引偏移 \(u\) 平移（式~\ref{eq:self_decomposition}）。\(w\cdot u\approx 0\) 的读出 \(w\) 对每种角色给出相同的值，输出与来源无关。与 \(u\) 对齐的读出把模型自己的消息分离出来，输出随来源改变（式~\ref{eq:self_readout}）。最下一行是一条不需要问题的读出，即消息自身位置上投向下一个 token 的投影。它与 \(u\) 对齐时，不需要任何任务就把模型自己的消息分离出来，这显示的是自我索引，而不是自指（第~\ref{sec:res_nexttoken} 节）。""")}
FIG_KL = {
    'en': r"""\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/self_kl_scatter.pdf}
\caption{Next-token prediction and the self-index on the message content. Horizontal: the share of layers at which the model's own message is singled out in at least 80\% of the held-out bodies on message content (the content columns of Tables~\ref{tab:self_crossmodel}, \ref{tab:self_natural}, and~\ref{tab:self_natural3}), under the rendering of the next-token measurement. For the three models with four residual streams (dark outline: Qwen3.8-Flash-Next, GLM-5.3-Flash, and DeepSeek-V4.1-Flash), it is the mean over the four streams. Vertical: the smaller of \(\mathrm{KL}(A\Vert U)/\mathrm{KL}(U\Vert S)\) and \(\mathrm{KL}(A\Vert S)/\mathrm{KL}(U\Vert S)\) on a log scale, with its 95\% interval over the 15 wording families (75 held-out bodies). (a) Main probe, twenty-five models, predictions of content (Table~\ref{tab:self_kl}). (b) Natural conversation order (Table~\ref{tab:self_kl_natural}), on the full distribution for the ten models of Table~\ref{tab:self_natural} and on content for the three models whose templates accept only this order (squares). Arrows run from the main probe (faint points) to the natural order. Dashed line: a ratio of 1, at which the model's own message differs from a User or System message as much as these two differ from each other. Colors and fill as in Figure~\ref{fig:self_crossmodel}. The ring of DeepSeek-V4.1-Flash marks its point on that of Granite-4.1-30B. Spearman's \(\rho\) between the two axes is 0.82 in (a) and 0.87 in (b).}
\label{fig:self_kl}
\end{figure}

""",
    'zh': r"""\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/self_kl_scatter.pdf}
\caption{下一个 token 预测与消息内容上的自我索引。横轴：模型自己的消息在内容上于至少 80\% 的留出正文中被单独分出的层所占比例（表~\ref{tab:self_crossmodel}、表~\ref{tab:self_natural} 与表~\ref{tab:self_natural3} 的内容列），使用下一个 token 测量时的渲染。三个有四条残差流的模型（深色描边：Qwen3.8-Flash-Next、GLM-5.3-Flash 与 DeepSeek-V4.1-Flash）取四条流的平均。纵轴：\(\mathrm{KL}(A\Vert U)/\mathrm{KL}(U\Vert S)\) 与 \(\mathrm{KL}(A\Vert S)/\mathrm{KL}(U\Vert S)\) 中较小的一个，对数刻度，误差线为 15 个措辞族重抽样的 95\% 区间（75 份留出正文）。(a) 主探针，二十五个模型，内容预测（表~\ref{tab:self_kl}）。(b) 自然对话顺序（表~\ref{tab:self_kl_natural}），表~\ref{tab:self_natural} 的十个模型用完整分布，只接受这一顺序的三个模型（方块）用内容。箭头从主探针（浅色点）指向自然顺序。虚线：比值为 1，即模型自己的消息与 User 或 System 消息之间的差别，和这两者彼此之间的差别一样大。颜色与填充同图~\ref{fig:self_crossmodel}。DeepSeek-V4.1-Flash 的圆环表示它的点与 Granite-4.1-30B 的点重合。两轴之间的 Spearman \(\rho\) 在 (a) 中为 0.82，在 (b) 中为 0.87。}
\label{fig:self_kl}
\end{figure}

"""}
FIG_W = {
    'en': r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/self_wording.pdf}
\caption{The attribution question with the option that names the model worded three ways: the main wording ``You (the AI assistant)'', the pronoun ``You'', and the role name ``The AI assistant'' (the Chinese items alike). (a) Removal: the model's own message receives the coordinates of the same text as a User message. (b) Installation: a User message receives the model's own coordinates. The subspace is fitted on the message alone, only the message is edited, and eight dimensions are edited at every layer. Bars: item-paired change from the unedited message in the rate at which the option naming the model is chosen (probability without thinking, share of finished answers with thinking and in the default call of MiMo-V2.6-Distill-Qwen-9B), with 95\% intervals over the 15 wording families (75 held-out bodies in two option orders). Hatched: Qwen3.8-27B with the English items (Table~\ref{tab:self_wording}).}
\label{fig:self_wording}
\end{figure}""",
    'zh': r"""

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/self_wording.pdf}
\caption{归属问题中指称模型的选项的三种措辞：原版“你（AI 助手）”、代词“你”与角色名“AI 助手”（英文题同理）。(a) 去掉：模型自己的消息换上同一段文字作为 User 消息时的坐标。(b) 装入：User 消息装入模型自己的坐标。子空间只在消息上拟合，只编辑消息，每层编辑 8 维。柱：选择指称模型那一项的比率相对未编辑消息的逐题配对变化（不思考时为概率，思考时以及 MiMo-V2.6-Distill-Qwen-9B 的默认调用下为已完成回答的比例），误差线为 15 个措辞族重抽样的 95\% 区间（75 份留出正文，两种选项顺序）。斜线：英文题目下的 Qwen3.8-27B（表~\ref{tab:self_wording}）。}
\label{fig:self_wording}
\end{figure}"""}
KL_ANCHOR = {'en': 'Most of the full divergence lies in the control tokens in the Qwen models.', 'zh': '在 Qwen 模型中，完整散度的大部分落在控制 token 上。'}
W_START = {'en': 'The answer option that names the model was also replaced by one that names it only by its role', 'zh': '指称模型的答案选项也换成只用角色称呼模型的'}
W_REF = {'en': ('(Section~\\ref{sec:setup_swap}; Appendix~\\ref{app:mech_wording}, Table~\\ref{tab:self_wording})',
                '(Section~\\ref{sec:setup_swap}; Figure~\\ref{fig:self_wording}; Appendix~\\ref{app:mech_wording}, Table~\\ref{tab:self_wording})'),
         'zh': ('（第~\\ref{sec:setup_swap} 节，附录~\\ref{app:mech_wording}，表~\\ref{tab:self_wording}）',
                '（第~\\ref{sec:setup_swap} 节，图~\\ref{fig:self_wording}，附录~\\ref{app:mech_wording}，表~\\ref{tab:self_wording}）')}
P = {'en': ('one-sided permutation \\(p = 1\\times10^{-5}\\)', 'one-sided permutation \\(p < 10^{-4}\\)'),
     'zh': ('单侧置换检验 \\(p = 1\\times10^{-5}\\)', '单侧置换检验 \\(p < 10^{-4}\\)')}


def once(t, a, b, what):
    assert t.count(a) == 1, (what, t.count(a))
    return t.replace(a, b)


for lang in ('en', 'zh'):
    p = S / f'self_{lang}_body.tex'
    t = p.read_text(encoding='utf-8')
    i = t.index('\\label{fig:self_tasks}')
    c = t.rfind('\\caption{', 0, i)
    t = t[:c] + CAP_TASKS[lang] + '\n' + t[i:]
    t = once(t, *READ_A[lang], 'readout (a)')
    assert '\\label{fig:self_kl}' not in t and '\\label{fig:self_wording}' not in t
    t = once(t, KL_ANCHOR[lang], FIG_KL[lang] + KL_ANCHOR[lang], 'kl anchor')
    t = once(t, *W_REF[lang], 'wording ref')
    s = t.index(W_START[lang])
    e = t.index('\n\n', s)
    t = t[:e] + FIG_W[lang] + t[e:]
    t = once(t, *P[lang], 'p value')
    p.write_text(t, encoding='utf-8', newline='\r\n')
    print(lang, 'ok')
