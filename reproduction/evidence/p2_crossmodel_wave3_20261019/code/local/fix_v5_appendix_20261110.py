r"""11-10 appendix revision, both languages (user: KL, MiMo-V2.6-Distill-Qwen-9B as the fifth readout, the wording test). Uses
stats/v5_appendix_tex.json (make_v5_appendix_tex_20261110.py); no number is typed here.
  1. a new section "Next-Token Prediction" (app:nexttoken) before "Further Causal Tests", with Tables tab:self_kl, tab:self_kl_natural,
     tab:self_kl_swap;
  2. a new subsection "How the Question Names the Model" (app:mech_wording, Table tab:self_wording) before app:mech_ablation;
  3. a MiMo column in tab:self_seeds, tab:self_cluster, tab:self_question (colspec, the two header rows, \multicolumn{5} -> {6},
     one cell per data row; the number of data rows must equal the number of cells).
Each anchor must match exactly once. The appendix files are LF. 11-10: the rationale clauses were removed afterwards (paper rule: design choices as facts only).
usage: fix_v5_appendix_20261110.py ARCHIVE PAPER_SOURCE"""
import json
import re
import sys
from pathlib import Path

A, S = Path(sys.argv[1]), Path(sys.argv[2])
J = json.load(open(A / 'stats' / 'v5_appendix_tex.json', encoding='utf-8'))

KL_EN = r"""\clearpage
\section{Next-Token Prediction}
\label{app:nexttoken}

Section~\ref{sec:res_nexttoken} compares the model's predictions of the next token on the body of a message between its three roles. Each held-out body is rendered under the main probe as an Assistant, a User, and a System message, followed by a user turn that asks for a reply (in natural order, after the user request of Appendix~\ref{app:natural}), and each rendering is run once, teacher-forced, with the model's own logits. At every body position common to the three renderings except the last, we take the predicted distribution over the next token and the Kullback--Leibler divergence between two roles, \(\mathrm{KL}(A\Vert U)\), \(\mathrm{KL}(A\Vert S)\), and \(\mathrm{KL}(U\Vert S)\), and its median over the positions of a body. A ratio is the sum of these divergences over the 75 held-out bodies divided by that of \(\mathrm{KL}(U\Vert S)\), and its 95\% interval resamples the 15 wording families with all their bodies (2,000 resamples). For the predictions of content, the control tokens of the template are removed from both distributions and the rest is renormalized: the special tokens of the tokenizer, the tokens that end or open a turn, and the tags of reasoning and tool calls. P(control) is the probability that the model puts on these tokens, as a median over the positions of a body and a mean over bodies. Table~\ref{tab:self_kl} gives the main probe for the twenty-five models, Table~\ref{tab:self_kl_natural} the natural order, and Table~\ref{tab:self_kl_swap} the replacement of the self-index coordinates in the five readouts of Section~\ref{sec:res_mechanism}.

The replacement uses the subspace fitted on the message alone (Section~\ref{sec:abl_question}). At every layer, the coordinates of the donor rendering along its eight directions, at the aligned body positions, are written into the same positions of the recipient. Only the body is edited. Left is the divergence between the edited own message and the external one as a share of the unedited divergence, \(\mathrm{KL}(A{\leftarrow}U\Vert U)/\mathrm{KL}(A\Vert U)\). Closed is the share of the divergence between an external message and the model's own that the installation removes, \(1-\mathrm{KL}(U{\leftarrow}A\Vert A)/\mathrm{KL}(U\Vert A)\).

\begin{table}[htbp]
\centering
\caption{Next-token prediction under the main probe, in the order of Table~\ref{tab:self_crossmodel}. Self-index: as grouped in Section~\ref{sec:res_crossmodel}. P(control): probability of the control tokens on the body of the model's own, a User, and a System message. Ratios of \(\mathrm{KL}(A\Vert U)\) and \(\mathrm{KL}(A\Vert S)\) to \(\mathrm{KL}(U\Vert S)\), on the full distribution and on content, with 95\% intervals over the 15 wording families (75 held-out bodies).}
\label{tab:self_kl}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
& & & Full & \multicolumn{2}{c}{Content} \\
\cmidrule(lr){4-4}\cmidrule(lr){5-6}
Model & Self-index & P(control), A / U / S & A--U / U--S & A--U / U--S & A--S / U--S \\
\midrule
%<<tab:self_kl
%>>tab:self_kl
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Next-token prediction in natural conversation order. Content index: layers at which the model's own message is singled out on its content in this order (Tables~\ref{tab:self_natural} and~\ref{tab:self_natural3}). Upper part: full distribution, on which the control tokens of these models carry almost no probability under the main probe (Table~\ref{tab:self_kl}). Lower part: the models that accept only natural order, content. The templates of Qwen3.8-27B, gpt-oss-20b, Apertus-8B, and DeepSeek-R1-Distill-Llama-70B keep no System message after a user turn, so these models have no reference \(\mathrm{KL}(U\Vert S)\) in this order.}
\label{tab:self_kl_natural}
\footnotesize
\begin{tabular}{lccc}
\toprule
Model & Content index & A--U / U--S & A--S / U--S \\
\midrule
%<<tab:self_kl_natural
%>>tab:self_kl_natural
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Replacement of the eight self-index coordinates on the body, next-token prediction of content (75 held-out bodies, 95\% intervals over the 15 wording families). Left: share of the divergence between the model's own and the external message that remains after the own message receives the coordinates of the external one. Closed: share of the divergence between an external message and the model's own that the installation of the model's own coordinates removes. Random: the same edits in a random subspace of the same dimension, left after removal and closed by installation. Last column: \(\mathrm{KL}(A{\leftarrow}U\Vert U)/\mathrm{KL}(U\Vert S)\) after removal.}
\label{tab:self_kl_swap}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Left} & \multicolumn{2}{c}{Closed} & Random & After removal \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Readout & own \(\leftarrow\) User & own \(\leftarrow\) System & User \(\leftarrow\) own & System \(\leftarrow\) own & left / closed & A--U / U--S \\
\midrule
%<<tab:self_kl_swap
%>>tab:self_kl_swap
\bottomrule
\end{tabular}
\end{table}

"""

KL_ZH = r"""\clearpage
\section{下一个 token 的预测}
\label{app:nexttoken}

第~\ref{sec:res_nexttoken} 节比较模型在消息正文上对下一个 token 的预测在三种角色之间的差别。每份留出正文在主探针下分别渲染为 Assistant、User 和 System 消息，后接一个请模型回应的用户轮次（自然顺序下放在附录~\ref{app:natural} 的用户请求之后），每种渲染以教师强制方式前向一次，取模型自己的 logits。在三种渲染共有的每个正文位置（最后一个除外），取模型对下一个 token 的预测分布，计算两种角色之间的 Kullback--Leibler 散度 \(\mathrm{KL}(A\Vert U)\)、\(\mathrm{KL}(A\Vert S)\) 和 \(\mathrm{KL}(U\Vert S)\)，并在一份正文的各位置上取中位数。比值是 75 份留出正文上这些散度之和与 \(\mathrm{KL}(U\Vert S)\) 之和的比，95\% 区间对 15 个措辞族连同其全部正文做重抽样（2,000 次）。内容预测从两个分布中去掉模板的控制 token，再在其余 token 上重新归一化。控制 token 包括分词器的特殊 token、结束或开启一个轮次的 token，以及推理和工具调用的标签。P（控制）是模型放在这些 token 上的概率，在一份正文的各位置上取中位数，再在正文之间取平均。表~\ref{tab:self_kl} 给出二十五个模型在主探针下的结果，表~\ref{tab:self_kl_natural} 给出自然顺序，表~\ref{tab:self_kl_swap} 给出第~\ref{sec:res_mechanism} 节五个读出中替换自我索引坐标的结果。

替换沿用只在消息上拟合的子空间（第~\ref{sec:abl_question} 节）。在每一层，把供体渲染在对齐的正文位置上沿 8 个方向的坐标写入接收者的同一位置。只改正文。剩余是改动后自己的消息与外部消息之间的散度占改动前的比例，即 \(\mathrm{KL}(A{\leftarrow}U\Vert U)/\mathrm{KL}(A\Vert U)\)。拉近是装入自己的坐标后，外部消息与自己消息之间的散度被去掉的比例，即 \(1-\mathrm{KL}(U{\leftarrow}A\Vert A)/\mathrm{KL}(U\Vert A)\)。

\begin{table}[htbp]
\centering
\caption{主探针下的下一个 token 预测，模型顺序同表~\ref{tab:self_crossmodel}。自我索引：按第~\ref{sec:res_crossmodel} 节的分组。P（控制）：模型自己的消息、User 消息和 System 消息正文上控制 token 的概率。\(\mathrm{KL}(A\Vert U)\) 和 \(\mathrm{KL}(A\Vert S)\) 与 \(\mathrm{KL}(U\Vert S)\) 之比，分别在完整分布和内容上计算，括号内为 15 个措辞族重抽样的 95\% 区间（75 份留出正文）。}
\label{tab:self_kl}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccc}
\toprule
& & & 完整分布 & \multicolumn{2}{c}{内容} \\
\cmidrule(lr){4-4}\cmidrule(lr){5-6}
模型 & 自我索引 & P（控制），A / U / S & A--U / U--S & A--U / U--S & A--S / U--S \\
\midrule
%<<tab:self_kl
%>>tab:self_kl
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{自然对话顺序下的下一个 token 预测。内容索引：在这一顺序下，模型自己的消息在内容上被单独分出的层（表~\ref{tab:self_natural} 和表~\ref{tab:self_natural3}）。上半部分：完整分布，这些模型在主探针下控制 token 上几乎没有概率（表~\ref{tab:self_kl}）。下半部分：只接受自然顺序的模型，内容。Qwen3.8-27B、gpt-oss-20b、Apertus-8B 和 DeepSeek-R1-Distill-Llama-70B 的模板在用户轮次之后不保留 System 消息，因此这些模型在这一顺序下没有参照 \(\mathrm{KL}(U\Vert S)\)。}
\label{tab:self_kl_natural}
\footnotesize
\begin{tabular}{lccc}
\toprule
模型 & 内容索引 & A--U / U--S & A--S / U--S \\
\midrule
%<<tab:self_kl_natural
%>>tab:self_kl_natural
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{在正文上替换 8 个自我索引坐标后的内容预测（75 份留出正文，15 个措辞族重抽样的 95\% 区间）。剩余：自己的消息换上外部消息的坐标后，它与外部消息之间的散度剩下的比例。拉近：外部消息装入模型自己的坐标后，它与模型自己消息之间的散度被去掉的比例。随机：同维数随机子空间中的同样改动，去掉后剩余的比例与装入后拉近的比例。最后一列：去掉后的 \(\mathrm{KL}(A{\leftarrow}U\Vert U)/\mathrm{KL}(U\Vert S)\)。}
\label{tab:self_kl_swap}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{剩余} & \multicolumn{2}{c}{拉近} & 随机 & 去掉后 \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
读出 & 自己 \(\leftarrow\) User & 自己 \(\leftarrow\) System & User \(\leftarrow\) 自己 & System \(\leftarrow\) 自己 & 剩余 / 拉近 & A--U / U--S \\
\midrule
%<<tab:self_kl_swap
%>>tab:self_kl_swap
\bottomrule
\end{tabular}
\end{table}

"""

W_EN = r"""\FloatBarrier
\subsection{How the Question Names the Model}
\label{app:mech_wording}

Table~\ref{tab:self_wording} gives the wording test of Section~\ref{sec:abl_question}. Everything is as in the message rows of Table~\ref{tab:self_question}: the subspace fitted on the message alone, the edit of the message alone, eight dimensions at every layer, and the 75 held-out bodies in two option orders. Only the text of the answer options changes, and the option letters and their two orders are kept. For the gpt-oss models, whose template writes the current date into the system message, the date of the runs of Table~\ref{tab:self_question} is used. The main wording of Qwen3.8-27B with the English items was run here as well, together with the two others.

\begin{table}[htbp]
\centering
\caption{Wording of the answer option that names the model, attribution question, subspace fitted on the message alone and only the message edited, eight dimensions at every layer (150 items per condition). Main: ``You (the AI assistant) / The user / The system''. Pronoun: ``You / Me / The system''. Role name: ``The AI assistant / The user / The system'' (the Chinese items alike). Unedited: share of finished items answering with the option that names the model (probability without thinking), for the model's own and the User message. Removal (own \(\leftarrow\) User) and installation (User \(\leftarrow\) own): item-paired change with a 95\% interval over the 15 wording families, and in parentheses the share of the unedited gap that the edit closes (not given for MiMo-V2.6-Distill-9B, whose unedited gap is small). Random: the larger absolute change of the two random-subspace edits. Main wording: the message rows of Table~\ref{tab:self_question} (without random conditions), except Qwen3.8-27B with the English items.}
\label{tab:self_wording}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{Unedited} & & & \\
\cmidrule(lr){2-3}
Readout, wording & own & User & Removal & Installation & Random \\
\midrule
%<<tab:self_wording
%>>tab:self_wording
\bottomrule
\end{tabular}
\end{table}

"""

W_ZH = r"""\FloatBarrier
\subsection{问题如何称呼模型}
\label{app:mech_wording}

表~\ref{tab:self_wording} 给出第~\ref{sec:abl_question} 节的措辞检验。其余设置与表~\ref{tab:self_question} 中只改消息的各行相同：子空间只在消息上拟合，只改消息，每层 8 维，75 份留出正文、两种选项顺序。只换答案选项的文字，选项字母和两种顺序不变。gpt-oss 的模板会把当天日期写进 system 消息，这里用表~\ref{tab:self_question} 那几次运行的日期。Qwen3.8-27B 英文题的原版措辞也在这里与另两种一起运行。

\begin{table}[htbp]
\centering
\caption{答案中指称模型的选项的措辞，归属题，子空间只在消息上拟合、只改消息，每层 8 维（每个条件 150 题）。原版：“你（AI 助手）/ 用户 / 系统”。代词：“你 / 我 / 系统”。角色名：“AI 助手 / 用户 / 系统”（英文题同理）。未编辑：模型自己的消息与 User 消息中，选择指称模型那一项的已完成题目比例（不思考时为概率）。去掉（自己 \(\leftarrow\) User）和装入（User \(\leftarrow\) 自己）：逐题配对的变化，括号前为 15 个措辞族重抽样的 95\% 区间，括号内为改动关掉的未编辑差距的比例（MiMo-V2.6-Distill-9B 的未编辑差距小，不给出）。随机：两种随机子空间改动中绝对值较大的变化。原版措辞：表~\ref{tab:self_question} 中只改消息的各行（没有随机条件），Qwen3.8-27B 英文题除外。}
\label{tab:self_wording}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{未编辑} & & & \\
\cmidrule(lr){2-3}
读出，措辞 & 自己 & User & 去掉 & 装入 & 随机 \\
\midrule
%<<tab:self_wording
%>>tab:self_wording
\bottomrule
\end{tabular}
\end{table}

"""
HEADS = {'en': ('& Qwen3.8-27B & Qwen3.8-27B & gpt-oss-20b & gpt-oss-120b \\\\', ' & MiMo-V2.6-9B', 'thinking \\\\', ' & default call'),
         'zh': ('& Qwen3.8-27B & Qwen3.8-27B & gpt-oss-20b & gpt-oss-120b \\\\', ' & MiMo-V2.6-9B', '思考 \\\\', ' & 默认调用')}


def fill(t, label, rows):
    pat = re.compile(r'(%<<' + re.escape(label) + r'\n)(.*?)(%>>' + re.escape(label) + r'\n)', re.S)
    assert len(pat.findall(t)) == 1, label
    return pat.sub(lambda m: m.group(1) + '\n'.join(rows) + '\n' + m.group(3), t)


def add_column(t, label, cells, lang):
    i, j = t.index(f'%<<{label}\n'), t.index(f'%>>{label}\n')
    body = t[i + len(f'%<<{label}\n'):j].split('\n')
    k = 0
    for n, line in enumerate(body):
        if line.startswith('\\multicolumn{5}'):
            body[n] = line.replace('\\multicolumn{5}', '\\multicolumn{6}', 1)
        elif line.strip().endswith('\\\\'):
            body[n] = line.rstrip()[:-2].rstrip() + f' & {cells[k]} \\\\'
            k += 1
    assert k == len(cells), (label, k, len(cells))
    t = t[:i + len(f'%<<{label}\n')] + '\n'.join(body) + t[j:]
    h1, a1, h2, a2 = HEADS[lang]
    b = t.rfind('\\begin{tabular}{lcccc}', 0, i)
    seg = t[b:i]
    assert seg.count(h1) == 1 and seg.rstrip().count('\\midrule') >= 1, (label, 'header')
    seg = seg.replace('\\begin{tabular}{lcccc}', '\\begin{tabular}{lccccc}', 1).replace(h1, h1[:-2].rstrip() + a1 + ' \\\\', 1)
    lines = seg.split('\n')
    idx = [n for n, ln in enumerate(lines) if ln.rstrip().endswith(h2)]
    assert len(idx) == 1, (label, 'second header', idx)
    lines[idx[0]] = lines[idx[0]].rstrip()[:-2].rstrip() + a2 + ' \\\\'
    return t[:b] + '\n'.join(lines) + t[i:]


for lang, kl, w in (('en', KL_EN, W_EN), ('zh', KL_ZH, W_ZH)):
    p = S / f'self_{lang}_appendix.tex'
    t = p.read_text(encoding='utf-8')
    assert '\r' not in t
    anchor = '\\clearpage\n\\section{' + ('Further Causal Tests' if lang == 'en' else '更多因果检验') + '}\n\\label{app:mechanism}'
    assert t.count(anchor) == 1 and 'app:nexttoken' not in t, (lang, 'kl anchor')
    t = t.replace(anchor, kl + anchor, 1)
    m = re.search(r'\\FloatBarrier\n\\subsection\{[^}]*\}\n\\label\{app:mech_ablation\}', t)
    assert m and 'app:mech_wording' not in t, (lang, 'wording anchor')
    t = t[:m.start()] + w + t[m.start():]
    t = fill(t, 'tab:self_kl', J[f'kl_main_{lang}'])
    t = fill(t, 'tab:self_kl_natural', J[f'kl_nat_{lang}'])
    t = fill(t, 'tab:self_kl_swap', J[f'kl_swap_{lang}'])
    t = fill(t, 'tab:self_wording', J[f'wording_{lang}'])
    for lab, key in (('tab:self_seeds', 'mimo_seeds'), ('tab:self_cluster', 'mimo_cluster'), ('tab:self_question', 'mimo_question')):
        t = add_column(t, lab, J[key], lang)
    p.write_text(t, encoding='utf-8', newline='\n')
    print(lang, 'ok')
