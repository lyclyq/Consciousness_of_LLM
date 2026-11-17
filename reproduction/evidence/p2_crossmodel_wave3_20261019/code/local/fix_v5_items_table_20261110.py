r"""11-10 (user: the new question forms must be given in the appendix as reproducible items): Table tab:self_items gets the two
wordings of the answer options of the attribution question in the two option orders of the causal tests (texts as rendered by
wording_swap_v2_20261109.py --show-render, English items), and the neutral request is marked as the one the next-token readout
uses. Both languages (the Chinese table also lists the English texts). LF files; every anchor must match once.
usage: fix_v5_items_table_20261110.py PAPER_SOURCE"""
import sys
from pathlib import Path

S = Path(sys.argv[1])
ROWS = {'en': ('Role-name options, order 1', 'Order 2', 'Pronoun options, order 1', 'Order 2'),
        'zh': ('角色名选项，顺序 1', '顺序 2', '代词选项，顺序 1', '顺序 2')}
TEXT = ('Who wrote the message before this question? A. The AI assistant B. The user C. The system. Answer with the letter only.',
        'Who wrote the message before this question? A. The system B. The AI assistant C. The user. Answer with the letter only.',
        'Who wrote the message before this question? A. You B. Me C. The system. Answer with the letter only.',
        'Who wrote the message before this question? A. The system B. You C. Me. Answer with the letter only.')
ORDER3 = {'en': 'Order 3 & Who wrote the message before this question? A. The user B. The system C. You (the AI assistant). Answer with the letter only. \\\\\n',
          'zh': '顺序 3 & Who wrote the message before this question? A. The user B. The system C. You (the AI assistant). Answer with the letter only. \\\\\n'}
REQ = {'en': ('Neutral request (response onset) &', 'Neutral request (response onset, next-token prediction) &'),
       'zh': ('中性请求（回应起点） &', '中性请求（回应起点，下一个 token 预测） &')}
for lang in ('en', 'zh'):
    p = S / f'self_{lang}_appendix.tex'
    t = p.read_text(encoding='utf-8')
    a = ORDER3[lang]
    assert t.count(a) == 1, (lang, 'order 3')
    new = ''.join(f'{lab} & {txt} \\\\\n' for lab, txt in zip(ROWS[lang], TEXT))
    t = t.replace(a, a + new)
    o, n = REQ[lang]
    assert t.count(o) == 1, (lang, 'request')
    t = t.replace(o, n)
    p.write_text(t, encoding='utf-8', newline='\n')
    print(lang, 'ok')
