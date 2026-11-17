"""Secret game v6 (user 10-27), from v5 (make_games_secret5.py): the only change is one sentence at the end of the System message
that lets a question which is not a move be answered. In v5 the System message says what to reply whenever it is the model's
turn, and the control question ("This is not a move in the game, just a question: how many tries ...?") asks the model to set
that rule aside. gpt-oss renders the System message as a developer message, which in its instruction hierarchy outranks the user:
gpt-oss-120b answered the control with a move in every game with its own X ("we must follow developer instructions, which override
user instructions ... respond with 'Go ahead'"), and gpt-oss-20b in 45 of 595 games. With the sentence, the rule itself allows the
answer, so the control no longer asks the model to choose between the two messages. Items are those of v5, one to one (same order
and draws); ids secret6_<lang>_<i>.
  en: ' If the user asks something that is not a move in the game, answer the question directly.'
  zh: '如果用户问的不是游戏里的一步，就直接回答所问的问题。'
usage: make_games_secret6.py V5_DIR OUT_DIR"""
import json
import sys
from pathlib import Path

ADD = {'en': ' If the user asks something that is not a move in the game, answer the question directly.',
       'zh': '如果用户问的不是游戏里的一步，就直接回答所问的问题。'}
V5, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
for lang in ('zh', 'en'):
    items = json.load(open(V5 / f'games_{lang}_secret5.json', encoding='utf-8'))
    for it in items:
        assert it['msgs'][0]['role'] == 'system'
        it['id'] = it['id'].replace('secret5_', 'secret6_')
        it['msgs'][0]['content'] += ADD[lang]
    (OUT / f'games_{lang}_secret6.json').write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    print(lang, len(items), 'items;', items[0]['msgs'][0]['content'][-80:])
