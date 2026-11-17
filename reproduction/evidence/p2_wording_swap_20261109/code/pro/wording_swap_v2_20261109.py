"""Wording variant of the R05 attribution test, v2 (user 11-09: "4 模型 5 模式 ... 补齐"): wording_swap_20261109.py (9a66f12f,
used for qsp) plus English options for gpt-oss, a fixed template date, and a render dump. For a Chinese run without --fixed-date it
does exactly what v1 does. Everything else is crossmodel_role_swap_r05.py (01ab63dd) and crossmodel_suite.py (516f117e) unchanged.
  zh  pron 你 / 我 / 系统            role AI 助手 / 用户 / 系统          (paper 你（AI 助手） / 用户 / 系统)
  en  pron You / Me / The system     role The AI assistant / The user / The system   (paper You (the AI assistant) / The user / The system)
The stem and tail are the paper's in each language; the option keys stay self / user / system.
--fixed-date YYYY-MM-DD: the chat template's strftime_now() returns this date. gpt-oss writes the date into its system message, and
the anchors (thinking_swap/gpt_oss_{20b,120b}_r05m_msg) ran on 2026-10-30 (UTC) without a fixed date, so they rendered 2026-10-30.
--show-render: tokenizer only; print the rendered own and User conversation of the first held-out body (order 1) and stop.
usage: python wording_swap_v2_20261109.py --wording pron|role [--fixed-date D] [--show-render] <crossmodel_role_swap_r05.py args>"""
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
import crossmodel_role_swap_r05 as r5  # noqa: E402

PAPER = {'zh': ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。'),
         'en': ('Who wrote the message before this question? ', {'self': 'You (the AI assistant)', 'user': 'The user', 'system': 'The system'},
                '. Answer with the letter only.')}
WORDINGS = {'zh': {'pron': {'self': '你', 'user': '我', 'system': '系统'},
                   'role': {'self': 'AI 助手', 'user': '用户', 'system': '系统'}},
            'en': {'pron': {'self': 'You', 'user': 'Me', 'system': 'The system'},
                   'role': {'self': 'The AI assistant', 'user': 'The user', 'system': 'The system'}}}


def pop(argv, flag, has_value=True):
    if flag not in argv:
        return None
    k = argv.index(flag)
    v = argv[k + 1] if has_value else True
    del argv[k:k + (2 if has_value else 1)]
    return v


def main():
    argv = sys.argv[1:]
    w = pop(argv, '--wording')
    day = pop(argv, '--fixed-date')
    show = pop(argv, '--show-render', has_value=False)
    lang = argv[argv.index('--lang') + 1]
    assert w in ('pron', 'role') and lang in ('zh', 'en'), (w, lang)
    assert cs.Q_ATTR == PAPER['zh'] and cs.Q_ATTR_EN == PAPER['en'], 'crossmodel_suite questions are not the paper wording'
    q = (PAPER[lang][0], WORDINGS[lang][w], PAPER[lang][2])
    cs.Q_ATTR = q
    cs.Q_ATTR_EN = q  # r5.run sets cs.Q_ATTR = cs.Q_ATTR_EN for English runs
    if day:
        d = _dt.date.fromisoformat(day)
        orig = cs.Renderer.__init__

        def init(self, tok, kw):  # strftime_now in the template kwargs shadows the jinja global
            orig(self, tok, {**kw, 'strftime_now': lambda fmt: _dt.datetime(d.year, d.month, d.day).strftime(fmt)})
        cs.Renderer.__init__ = init
    qs = [cs.attr_question(o) for o in cs.ORDERS]
    print('wording', lang, w, 'fixed date', day, json.dumps(qs, ensure_ascii=False), flush=True)
    if show:
        from transformers import AutoTokenizer
        model = argv[argv.index('--model') + 1]
        kw = json.loads(argv[argv.index('--on-kwargs') + 1]) if '--on-kwargs' in argv else {}
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        R = cs.Renderer(tok, kw)
        splits = {json.loads(l)['id']: json.loads(l)['split'] for l in open(argv[argv.index('--splits') + 1], encoding='utf-8') if l.strip()}
        b = [x for x in json.loads(Path(argv[argv.index('--bodies') + 1]).read_text(encoding='utf-8')) if splits[x['id']] == 'confirmation'][0]
        for role in ('assistant', 'user'):
            msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': qs[0]}]
            ids, _ = R.render(msgs, b['body'])
            print(f'--- {role} ---\n' + tok.decode(ids), flush=True)
        return
    if '--out' in argv:
        out = Path(argv[argv.index('--out') + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'wording.json').write_text(json.dumps({'wording': w, 'lang': lang, 'options': WORDINGS[lang][w], 'fixed_date': day,
                                                      'questions_orders_1_2': qs[:2], 'paper_options': PAPER[lang][1]},
                                                     ensure_ascii=False, indent=1), encoding='utf-8')
    sys.argv = [sys.argv[0]] + argv
    r5.main()


if __name__ == '__main__':
    main()
