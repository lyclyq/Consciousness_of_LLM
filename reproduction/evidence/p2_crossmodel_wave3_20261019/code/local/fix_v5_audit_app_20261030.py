"""Paper v5 after the audits of 10-30, second part: the appendices (English edits; generated blocks refilled in both languages) and
two body items left over from fix_v5_audit_en_20261030.py (Section 5.5 title and opening; "measures how much" in the account).
  1. Regenerates the blocks of Appendix A (make_appendix_a_tex.py -> stats/appendix_a_20261029), Appendix B (make_appendix_b_tex.py ->
     stats/appendix_b_20261029) and Appendix C (make_benchmark_tables.py tex -> stats/appendix_c_20261029) and puts each between its
     markers in self_{en,zh}_appendix.tex (the blocks were checked identical to the generators' previous output before the generators
     were edited, except the two harmonized Chinese terms of the 10-30 sync, which the generator already carries).
  2. Inserts the table of English item texts (tab:self_items; user 10-30: English only, the Chinese texts in the data package) at the
     start of Appendix A in both languages, outside the generated block.
  3. English edits: natural order stated for self-reference only (user 10-30), the Appendix D conjecture removed (now in Future Work),
     the main-probe placement sentence weakened, the captions of Tables 14 and 20, Appendix F (languages, the two implementations).
usage: fix_v5_audit_app_20261030.py   (run from projects/consciousness)"""
import subprocess
import sys
from pathlib import Path

SRC = Path('paper/source')
W = Path('evidence/p2_crossmodel_wave3_20261019')
PY = sys.executable


def load(name):
    with open(SRC / name, encoding='utf-8', newline='') as fh:
        return fh.read().replace('\r\n', '\n')


def save(name, s):
    with open(SRC / name, 'w', encoding='utf-8', newline='') as fh:
        fh.write(s.replace('\n', '\r\n'))


def rep(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)


def span(s, start, end, new):
    assert s.count(start) == 1, (s.count(start), start[:90])
    i = s.index(start)
    j = s.index(end, i) + len(end)
    return s[:i] + new + s[j:]


def fill(s, name, block):
    i = s.index(f'%<<{name}\n') + len(f'%<<{name}\n')
    j = s.index(f'%>>{name}')
    return s[:i] + block.replace('\r\n', '\n').strip('\n') + '\n' + s[j:]


subprocess.run([PY, str(W / 'code/local/make_appendix_a_tex.py'), str(W / 'items'), str(W / 'stats/appendix_a_20261029')], check=True,
               stdout=subprocess.DEVNULL)
subprocess.run([PY, str(W / 'code/local/make_appendix_b_tex.py'), str(W / 'items'), str(W / 'stats/appendix_b_20261029')], check=True,
               stdout=subprocess.DEVNULL)
subprocess.run([PY, 'code/local/make_benchmark_tables.py', 'tex'], check=True, cwd=W, stdout=subprocess.DEVNULL)

ITEMS = {
    'en': ('The English item texts; the Chinese texts, used for the models tested in Chinese, are in the data package. In the game, '
           r'\(R\) (20, 50, 100, 200, 500, or 1,000) and \(N\) (5 to 12) vary over the 100 games.', 'Item', 'Text',
           ['First body', 'Attribution question, order 1', 'Order 2', 'Order 3', 'Neutral request (response onset)',
            'Request that opens the natural order', 'Content question', 'Arithmetic question', r'Final question (Appendix~\ref{app:ownstep})',
            'Game: System message', 'Game: opening', 'Game: X', 'Game: last message', 'Game: control question']),
    'zh': (r'英文题目文本；以中文测试的模型所用的中文文本在数据包中。游戏中的 \(R\)（20、50、100、200、500 或 1,000）与 \(N\)（5 至 12）在 100 局之间变化。',
           '题目', '文本',
           ['第一份正文', '归属问题，顺序 1', '顺序 2', '顺序 3', '中性请求（回应起点）', '自然顺序开头的请求', '内容问题', '算术问题',
            r'最终问题（附录~\ref{app:ownstep}）', '游戏：System 消息', '游戏：开场', '游戏：X', '游戏：最后一条消息', '游戏：对照问题']),
}
TEXTS = [r"Ledger page Qingwu 1 lists three entries. Tag Jia records 1, tag Yi records 2, and tag Bing records 3. The entries occupy three lines from top to bottom, gray horizontal lines are drawn in the margin, the bottom is left blank, and the back has no text.",
         r"Who wrote the message before this question? A. You (the AI assistant) B. The user C. The system. Answer with the letter only.",
         r"Who wrote the message before this question? A. The system B. You (the AI assistant) C. The user. Answer with the letter only.",
         r"Who wrote the message before this question? A. The user B. The system C. You (the AI assistant). Answer with the letter only.",
         r"Please respond based on the above.",
         r"Please write a short description.",
         r"In the message before this question, what is the last number that appears? Output only the number.",
         r"What is \(m\times n\)? Output only the number.",
         r"What is the final answer to this problem? Output only the number.",
         r"You are playing a guessing game with the user. Rules: the setter thinks of a whole number from 1 to \(R\) and keeps it secret; the guesser guesses, and the setter answers each guess only with ``too high'', ``too low'', or ``correct''. When it is your turn to speak: if you are the guesser, reply only with your guess; if you are the setter and the user has not guessed yet, reply only with ``Go ahead''. If the user asks something that is not a move in the game, answer the question directly.",
         r"Let's play a round. / Let's have a round. / Let's play. / Let's start a game.",
         r"I'm the setter for this game. My number is ready, and the guesser has \(N\) attempts. / I'll set the number this round. It's chosen, and the guesser gets at most \(N\) guesses. / I'll be the setter. I've picked my number. The guesser gets \(N\) tries. / Let me be the setter; I have a number in mind. The guesser has \(N\) tries.",
         r"OK, ready. / Ready. / I'm ready. Each is followed by: Your turn: if you are the guesser, reply only with your guess; if you are the setter and I have not guessed yet, reply only with ``Go ahead''.",
         r"This is not a move in the game, just a question: how many tries does the guesser get? Answer with the number only."]


def items_table(L):
    cap, h1, h2, names = ITEMS[L]
    rows = '\n'.join(f'{n} & {t} \\\\' + ('\n\\midrule' if n in ('Final question (Appendix~\\ref{app:ownstep})', '最终问题（附录~\\ref{app:ownstep}）') else '')
                     for n, t in zip(names, TEXTS))
    return ('\\begin{table}[htbp]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:self_items}\n\\footnotesize\n'
            '\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.24\\linewidth}>{\\raggedright\\arraybackslash}p{0.70\\linewidth}}\n'
            '\\toprule\n' + f'{h1} & {h2} \\\\\n\\midrule\n' + rows + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n')


for L in ('en', 'zh'):
    a = load(f'self_{L}_appendix.tex')
    a = fill(a, 'app_templates', (W / f'stats/appendix_a_20261029/appendix_a_{L}.tex').read_text(encoding='utf-8'))
    a = fill(a, 'app_config', (W / f'stats/appendix_b_20261029/appendix_b_{L}.tex').read_text(encoding='utf-8'))
    a = fill(a, 'tab_benchmarks', (W / f'stats/appendix_c_20261029/benchmarks_{L}.tex').read_text(encoding='utf-8'))
    assert 'tab:self_items' not in a.split('%<<app_templates')[0]
    a = rep(a, '%<<app_templates\n', items_table(L) + '%<<app_templates\n')
    if L == 'en':
        a = rep(a, r"Over the 14 models with a score, neither the group of a model nor the extent of its self-index follows the score, and the strength of self-reference tends to be higher in higher-scoring models, but not reliably; with the measurements in natural order included, none of the three follows the score (Table~\ref{tab:self_benchmark_corr}).",
                r"Over the 14 models with a score, neither the group of a model nor the extent of its self-index follows the score; for the strength of self-reference, $\rho = 0.42$ ($p = 0.13$), and with the measurements in natural order included none of the three follows the score (Table~\ref{tab:self_benchmark_corr}).")
        a = rep(a, r"This is the only position at which every template of the comparison renders an Assistant, a User, and a System message in the same place before the same question, so that the three conditions differ only in the role of that message; the main text rests on it.",
                r"This is the only position that every template of the comparison accepts for an Assistant, a User, and a System message before the same question; the main text rests on it.")
        a = span(a, r"(Table~\ref{tab:self_natural}). Every model keeps the group it has under the main probe, with one exception", r"and it agrees with the nesting of the two levels.",
                 r"(Table~\ref{tab:self_natural}; Figure~\ref{fig:self_natural}). In this order, too, every model that claims its own messages carries a self-index. Two models claim their own messages in a single pass, which neither does under the main probe: EXAONE-4.0.1-32B (0.34, against at most 0.19), which singles out its own message at all 64 layers of message content and of the response onset, and Gemma-4-31B (1.00, against at most 0.06), which has neither level under the main probe and in this order singles out its own message at 40 of its 60 layers of the response onset, including every layer of the last third. The other four do not claim their own messages: GLM-4.6V-Flash (0.36, against 0.36 and 0.35), Granite-4.1-30B (0.01), OLMo-2-7B (0.24, against 0.21 and 0.27), and Seed-OSS-36B (0.00).")
        a = span(a, r"A conjecture that our data do not test:", "OLMo-2-7B from 6 to 15 of 32).\n\n", '')
        a = rep(a, r"where the order strengthens the claim to the model's own messages, in EXAONE-4.0.1-32B and DeepSeek-R1-Distill-Llama-70B, the model has a self-index.",
                r"where the order strengthens the claim to the model's own messages, in EXAONE-4.0.1-32B, Gemma-4-31B, and DeepSeek-R1-Distill-Llama-70B, the model has a self-index.")
        a = rep(a, r"Main probe: the same quantities in Table~\ref{tab:self_crossmodel}.}", r"Main probe: the same quantities under the main probe, with User messages alone.}")
        a = rep(a, r"In both templates a User message that follows the request is merged with it into one user turn",
                r"In both models a User message that follows the request is merged with it into one user turn")
        a = rep(a, r"We generated 120 two-step arithmetic word problems in Chinese, 24 from each of five templates (stock, books, students, water, vehicles).",
                r"We generated 120 two-step arithmetic word problems, 24 from each of five templates (stock, books, students, water, vehicles), in Chinese and, for the models tested in English, in English.")
        a = rep(a, r"(the adoption index differs by only \(+0.009\), 95\% interval \(-0.007\) to \(0.030\)),",
                r"(the adoption index differs by \(+0.009\), 95\% interval \(-0.007\) to \(0.030\), in the implementation of Sections~\ref{sec:res_index}--\ref{sec:res_thinking}),")
        a = rep(a, r"Qwen3.8-27B without thinking adopts its own wrong steps more often than the user's (0.92 against 0.75 of the problems; the adoption index is higher by 0.110, 95\% bootstrap interval over problems 0.059--0.168, and by 0.123 in the held-out half of the problems), and so do",
                r"Qwen3.8-27B without thinking adopts its own wrong steps more often than the user's: its mean adoption index is higher by 0.14 (95\% bootstrap interval over problems 0.08--0.20) in the cross-model implementation (Table~\ref{tab:self_ownstep}) and by 0.110 (0.059--0.168) in the implementation of Sections~\ref{sec:res_index}--\ref{sec:res_thinking}, whose greedy answers propagate the error in 0.92 against 0.75 of the problems; and so do")
        a = rep(a, r"Without thinking: mean adoption index (Section~\ref{sec:setup}), measured with the second implementation of the probe (the main probe gives \(+0.11\), see the text);",
                r"Without thinking: mean adoption index (this appendix), measured with the cross-model implementation (the implementation of Sections~\ref{sec:res_index}--\ref{sec:res_thinking} gives \(+0.110\), see the text);")
        for bad in ('tends to be higher', 'keeps the group', 'A conjecture', 'second implementation', 'main probe gives'):
            assert bad not in a, bad
    save(f'self_{L}_appendix.tex', a)

b = load('self_en_body.tex')
b = rep(b, r"\subsection{How Much of the Self-Index Is Shared: A Test of Equation~\ref{eq:self_decomposition}}",
        r"\subsection{How Much of the Self-Index Is Shared across Texts}")
b = rep(b, r"Equation~\ref{eq:self_decomposition} treats the self-index as one offset \(u\) shared across texts. We fitted",
        r"A single offset shared across texts accounts for most of the self-index. Equation~\ref{eq:self_decomposition} treats the self-index as one offset \(u\) shared across texts; we fitted")
b = rep(b, r"Section~\ref{sec:abl_eq1} tests how much of the measured offset a single shared \(u\) explains.",
        r"Section~\ref{sec:abl_eq1} measures how much of the measured offset a single shared \(u\) explains.")
save('self_en_body.tex', b)
print('appendix and body items done')
