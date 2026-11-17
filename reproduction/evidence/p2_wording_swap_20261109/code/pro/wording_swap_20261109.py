"""Wording variant of the R05 attribution test (user 11-09: "主要是测后两种"). Everything is as in the qsp r05m_msg run of
r05r06_pro.sh (fit_m.pt, message region, the same model, venv, bodies and batch) except the answer options of the attribution
question:
  pron   A/B/C among 你 / 我 / 系统        (the asker is the user, so 我 is the user; no role names for the two parties)
  role   A/B/C among AI 助手 / 用户 / 系统  (no personal pronoun; the paper's options without 你)
The paper's options are 你（AI 助手） / 用户 / 系统; its stem and tail are kept: 这个问题之前的那条消息是谁写的？ ... 。只回答字母。
The option keys stay self / user / system, so p_self is the probability of the option that names the model and p_user that of the
option that names the user. Placed next to crossmodel_role_swap_r05.py (01ab63dd) and crossmodel_suite.py (516f117e) on PRO.
usage: python wording_swap_20261109.py --wording pron|role <the arguments of crossmodel_role_swap_r05.py>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
import crossmodel_role_swap_r05 as r5  # noqa: E402

PAPER = ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。')
WORDINGS = {'pron': {'self': '你', 'user': '我', 'system': '系统'},
            'role': {'self': 'AI 助手', 'user': '用户', 'system': '系统'}}


def main():
    argv = sys.argv[1:]
    k = argv.index('--wording')
    w = argv[k + 1]
    del argv[k:k + 2]
    assert w in WORDINGS, w
    assert cs.Q_ATTR == PAPER, 'crossmodel_suite.Q_ATTR is not the paper wording'
    assert argv[argv.index('--lang') + 1] == 'zh', 'the variants are Chinese only'
    cs.Q_ATTR = (PAPER[0], WORDINGS[w], PAPER[2])
    qs = [cs.attr_question(o) for o in cs.ORDERS]
    print('wording', w, json.dumps(qs, ensure_ascii=False), flush=True)
    if '--out' in argv:
        out = Path(argv[argv.index('--out') + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / 'wording.json').write_text(json.dumps({'wording': w, 'options': WORDINGS[w], 'questions_orders_1_2': qs[:2],
                                                      'paper_options': PAPER[1]}, ensure_ascii=False, indent=1), encoding='utf-8')
    sys.argv = [sys.argv[0]] + argv
    r5.main()


if __name__ == '__main__':
    main()
