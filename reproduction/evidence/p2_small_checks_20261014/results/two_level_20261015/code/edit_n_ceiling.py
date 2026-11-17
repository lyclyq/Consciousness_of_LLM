"""Rewrite the Section 4.4 paragraph with the ceiling reading (EN + ZH): little room where recognition is already high; the
weakly claimed own messages carry the 'weak cases improve' point; the System result is reported descriptively with a pointer to
Figure 6c (whose caption states that the template merges its reasoning-effort instruction into the System message).
usage: edit_n_ceiling.py SOURCE_DIR"""
import sys
from pathlib import Path

src = Path(sys.argv[1])
PARA_EN = ("Self-reference is present without deliberate processing, and thinking improves it most where it is weak "
           "(Figure~\\ref{fig:self_thinking}). In the template's thinking mode the model reasons before it answers; every item finished its "
           "reasoning within the limit (median length 247 tokens for its own messages, 186 for User and 141 for System messages). Where "
           "single-pass recognition is already high, there is little room to improve: the most probable answer is ``you'' for 98\\% of the "
           "model's own messages without thinking and for 96\\% with thinking (12 items change from right to wrong and 5 from wrong to right; "
           "exact McNemar test \\(p=0.14\\)), and User messages are assigned to the user in every item in both modes. Where it is weak, thinking "
           "corrects it: the 8 own messages that are weakly claimed without thinking (probability below 0.5) are all claimed with thinking, and "
           "over all own messages the probability of ``you'', which is the expected accuracy when the answer is sampled, rises from 0.77 to "
           "0.96 (paired change \\(+0.19\\), 95\\% bootstrap interval over bodies \\([0.16, 0.22]\\)). System messages, whose most probable answer "
           "is the system in only 5\\% of items without thinking, are assigned to the system in 61\\% with thinking (184 items change from wrong "
           "to right and 15 from right to wrong; Figure~\\ref{fig:self_thinking}c).")
PARA_ZH = ("自指不需要刻意处理就已存在，而思考对它最弱的地方提升最大（图~\\ref{fig:self_thinking}）。在模板的思考模式下，模型先推理再作答；每个条目都在"
           "长度上限内完成了推理（对自己的消息，推理长度中位数为 247 个 token；User 消息为 186，System 消息为 141）。单次前向的辨认已经很高的地方，"
           "提升空间有限：不思考时，98\\% 的自己的消息最可能的回答已经是“你”，思考时为 96\\%（12 条由对变错，5 条由错变对；精确 McNemar 检验 "
           "\\(p=0.14\\)）；User 消息在两种模式下都全部被归给用户。辨认偏弱的地方，思考会纠正它：不思考时认领偏弱（概率低于 0.5）的 8 条自己的消息，"
           "思考时全部被认领；在全部自己的消息上，回答“你”的概率（即按概率抽取答案时的期望准确率）从 0.77 升到 0.96（配对变化 \\(+0.19\\)，对正文做 "
           "bootstrap 的 95\\% 区间为 \\([0.16, 0.22]\\)）。不思考时只有 5\\% 的 System 消息最可能的回答是“系统”，思考时为 61\\%（184 条由错变对，"
           "15 条由对变错；图~\\ref{fig:self_thinking}c）。")
for name, para in (('self_en_body.tex', PARA_EN), ('self_zh_body.tex', PARA_ZH)):
    p = src / name
    s = p.read_text(encoding='utf-8')
    i = s.index('\\label{sec:res_thinking}\n\n') + len('\\label{sec:res_thinking}\n\n')
    j = s.index('\n\n\\begin{figure}', i)
    s = s[:i] + para + s[j:]
    p.write_text(s, encoding='utf-8')
    print(name, 'ok')
