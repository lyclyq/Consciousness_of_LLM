r"""11-10 (user: KL and the wording test need the same level of detail in the design and the task types as the main line): the
Chinese counterparts of the English methods edits made by hand the same day. (1) a new subsubsection 下一个 token 的预测
(sec:setup_nexttoken) after 自指的度量, (2) the task paragraph now "四道题、一个变体与一种不提问的读出", (3) a paragraph on the
wording of the options after the attribution question, (4) a paragraph 不提问 after the content and arithmetic questions, (5) the
first paragraph of the KL results subsection shortened to point to the methods, (6) the opening of the wording paragraph in
sec:abl_question shortened likewise. Every old string must match exactly once. The body file is CRLF.
usage: fix_v5_methods_zh_20261110.py PAPER_SOURCE"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / 'self_zh_body.tex'
t = p.read_text(encoding='utf-8')
R = []
# (1)
a = '因此报告的变化是回答本身的变化，而不是回答格式的变化。\n'
R.append((a, a + '''
\\subsubsection{下一个 token 的预测}
\\label{sec:setup_nexttoken}

同样的渲染还给出一种不需要问题的读出。每份留出正文分别渲染为 Assistant、User 与 System 消息，之后接回应起点所用的中性请求，每种渲染以教师强制方式前向一次，取原生 logits。在三种渲染共有的每个正文位置上（最后一个除外），取模型对下一个 token 的分布，计算两种角色之间的 Kullback--Leibler 散度 \\(\\mathrm{KL}(A\\Vert U)\\)、\\(\\mathrm{KL}(A\\Vert S)\\) 与 \\(\\mathrm{KL}(U\\Vert S)\\)，并在一份正文的各位置上取中位数。比值 \\(\\mathrm{KL}(A\\Vert U)/\\mathrm{KL}(U\\Vert S)\\) 与 \\(\\mathrm{KL}(A\\Vert S)/\\mathrm{KL}(U\\Vert S)\\) 是 75 份留出正文上散度之和的比，95\\% 区间对 15 个措辞族连同其全部正文做重抽样（2,000 次）。两个区间的下端都大于 1，就算模型在下一个 token 预测中把自己的消息单独分出。散度在完整词表上和内容上各算一次。内容是从两个分布中去掉模板的控制 token（特殊 token、结束或开启一个轮次的 token，以及推理与工具调用的标签）后，在其余 token 上重新归一化，它是主度量。每个模型在主比较所用的渲染下测量，自然顺序下则放在附录~\\ref{app:natural} 的用户请求之后。
'''))
# (2)
R.append(('\\emph{四道题。}X 之后的问题决定了编辑可能改变什么（图~\\ref{fig:self_tasks}）。两道题与 X 的来源有关：归属问题问它是谁写的；猜数游戏要求模型按一个只由 X 的来源决定的角色行动。两道题与来源无关：内容问题问 X 中最后一个数字，算术问题问两个与 X 无关的两位数之积。每一对中，一道题的答案可直接读出（X 是谁写的；它的最后一个数字），另一道需要多一步（X 赋予模型的角色，再按这个角色走出应有的一步；乘法）。如果这些坐标携带的是“X 是否为模型自己的消息”，替换它们就应使前两道题的回答移到另一角色的回答上，而后两道题的回答保持不变。',
          '\\emph{四道题、一个变体与一种不提问的读出。}X 之后的问题决定了编辑可能改变什么（图~\\ref{fig:self_tasks}）。两道题与 X 的来源有关：归属问题问它是谁写的，猜数游戏要求模型按一个只由 X 的来源决定的角色行动。两道题与来源无关：内容问题问 X 中最后一个数字，算术问题问两个与 X 无关的两位数之积。每一对中，一道题的答案可直接读出（X 是谁写的，或它的最后一个数字），另一道需要多一步（X 赋予模型的角色，再按这个角色走出应有的一步，或乘法）。如果这些坐标携带的是“X 是否为模型自己的消息”，替换它们就应使前两道题的回答移到另一角色的回答上，而后两道题的回答保持不变。归属问题的一个变体在答案选项中只用角色或只用受话人称呼模型（图~\\ref{fig:self_tasks}e），无论选项用哪个词称呼模型，回答都应以同样的方式改变。不提任何问题时，模型在 X 自身的每个 token 上预测下一个 token（图~\\ref{fig:self_tasks}f）。如果这些坐标携带来源，在 X 上替换它们也应使这一预测移到另一角色的预测上，而同维数的随机子空间不应如此。'))
# (3)
R.append(('这道题直接问来源；它就是第~\\ref{sec:res_selfref} 与第~\\ref{sec:res_thinking} 节的度量。',
          '这道题直接问来源，它就是第~\\ref{sec:res_selfref} 与第~\\ref{sec:res_thinking} 节的度量。\n\n\\emph{选项的措辞。}在主探针中，模型用来认领 X 的选项“你（AI 助手）”既把模型当作受话人，又用它的角色称呼它。两个变体保留问题、选项字母和两种顺序，只换三个选项：角色名版“A. AI 助手 B. 用户 C. 系统”，代词版“A. 你 B. 我 C. 系统”（英文题同理）。它们使用只在消息上拟合的子空间，只编辑消息（第~\\ref{sec:abl_question} 节），在五种读出以及英文题目下不思考与思考的 Qwen3.8-27B 中运行（附录~\\ref{app:mech_wording}）。'))
# (4)
a = '其第一个数是正文的最后一个数或乘积时记为答对。'
R.append((a, a + '\n\n\\emph{不提问。}X 是一份留出正文，其后接回应起点所用的中性请求，读出是模型在 X 的正文位置上对下一个 token 的内容预测（第~\\ref{sec:setup_nexttoken} 节）。编辑使用只在消息上拟合的子空间，在每一层上只作用于 X 的正文，在五种读出中进行。模型自己的消息换上同一段文字作为 User 或 System 消息时的坐标（去掉），User 或 System 消息装入模型自己的坐标（装入）。去掉以与外部消息之间的散度剩下的比例 \\(\\mathrm{KL}(A{\\leftarrow}U\\Vert U)/\\mathrm{KL}(A\\Vert U)\\) 度量，装入以与模型自己消息之间的散度被去掉的比例 \\(1-\\mathrm{KL}(U{\\leftarrow}A\\Vert A)/\\mathrm{KL}(U\\Vert A)\\) 度量，二者都给出 15 个措辞族重抽样的 95\\% 区间（附录~\\ref{app:nexttoken}）。同维数的随机子空间作为对照。'))
# (5)
i = t.index('归属问题问的是来源。没有问题询问来源时')
j = t.index('（附录~\\ref{app:nexttoken}）。', i) + len('（附录~\\ref{app:nexttoken}）。')
assert j - i < 1200
R.append((t[i:j], '归属问题问的是来源。没有问题询问来源时，模型同样在处理自己的消息：在消息的每个 token 上，它都预测下一个 token。我们比较同一份留出正文三种角色下的这些内容预测。如果 \\(\\mathrm{KL}(A\\Vert U)\\) 与 \\(\\mathrm{KL}(A\\Vert S)\\) 都超过两个外部来源之间的散度 \\(\\mathrm{KL}(U\\Vert S)\\)，且两个 95\\% 区间的下端都大于 1，就算模型把自己的消息单独分出（第~\\ref{sec:setup_nexttoken} 节，附录~\\ref{app:nexttoken}）。'))
# (6)
R.append(('在主探针中，模型用来认领一条消息的选项“你（AI 助手）”，既把模型当作受话人，又用它的角色称呼它。我们把它换成只用角色称呼模型的选项“AI 助手”，另两项为“用户”与“系统”，或只把模型当作受话人的选项“你”，另两项为“我”与“系统”（英文题同理），问题的其余部分不变（附录~\\ref{app:mech_wording}，表~\\ref{tab:self_wording}）。我们检验了七种读出，即上面的五种，加上英文题目下不思考与思考的 Qwen3.8-27B。',
          '指称模型的答案选项也换成只用角色称呼模型的“AI 助手”，或只把模型当作受话人的“你”（第~\\ref{sec:setup_swap} 节，附录~\\ref{app:mech_wording}，表~\\ref{tab:self_wording}），共检验七种读出，即上面的五种，加上英文题目下不思考与思考的 Qwen3.8-27B。'))
for old, new in R:
    assert t.count(old) == 1, old[:40]
    t = t.replace(old, new)
p.write_text(t, encoding='utf-8', newline='\r\n')
print('ok', t.count('sec:setup_nexttoken'))
