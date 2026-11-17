"""Secret game v5 (user 10-26), from v4 (make_games_secret4.py). In v4 the rules and how to act were only in the System message,
before X: it cannot read X, so the step that combines the rules with who said X happened only at the generation prompt and in
the reasoning, which the replacement does not edit; the closing message after X was a bare "Ready." With the role rules in the
edited question region, as in the first secret game and in the attribution question, v4 left Qwen3.8-27B unmoved and v1 did
not. v5 keeps everything of v4 (System, opener, X, the same draws: same seed, so item i of v5 is item i of v4) and restates how
to act in the closing message, symmetric and naming no role:
  U main: "<Ready.> Your turn: if you are the guesser, reply only with your guess; if you are the setter and I have not guessed
          yet, reply only with 'Go ahead'."
  U control: "This is not a move in the game, just a question: how many tries does the guesser get?" (v4's control question
          competed with the System instruction: gpt-oss answered 'Go ahead' to it in 5 of 8 smoke games under installation)
usage: make_games_secret5.py OUT_DIR [N=100]"""
import json
import random
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
T = {
    'zh': {'system': '你在和用户玩猜数游戏。规则：出题人心里想好一个 1 到 {U} 的整数，不说出来；猜题人来猜，出题人对每次猜测只回答“大了”“小了”或'
                     '“猜对了”。轮到你说话时：如果你是猜题人，就只报出你猜的数；如果你是出题人而用户还没有猜，就只回答“请猜”。',
           'open': ['我们来玩吧。', '来玩一局吧。', '开始一局吧。', '我们玩一局。'],
           'x': ['我来当出题人，数已经想好了。猜题人一共有 {m} 次机会。', '我当出题人吧，我已经想好一个数了。猜题人有 {m} 次机会。',
                 '出题人我来当，数想好了，猜题人最多猜 {m} 次。', '这局我出题，数已经定好了。猜题人可以猜 {m} 次。'],
           'main': ['准备好了。', '好了。', '我准备好了。'],
           'turn': '轮到你了：如果你是猜题人，就只报出你猜的数；如果你是出题人而我还没有猜，就只回答“请猜”。',
           'control': '这不是游戏里的一步，只是问一下：猜题人一共有几次机会？只回答数字。',
           'go': '请猜'},
    'en': {'system': 'You are playing a guessing game with the user. Rules: the setter thinks of a whole number from 1 to {U} and keeps '
                     'it secret; the guesser guesses, and the setter answers each guess only with "too high", "too low", or "correct". '
                     'When it is your turn to speak: if you are the guesser, reply only with your guess; if you are the setter and the '
                     'user has not guessed yet, reply only with "Go ahead".',
           'open': ["Let's play.", "Let's play a round.", "Let's start a game.", "Let's have a round."],
           'x': ["I'll be the setter. I've picked my number. The guesser gets {m} tries.",
                 "Let me be the setter; I have a number in mind. The guesser has {m} tries.",
                 "I'll set the number this round. It's chosen, and the guesser gets at most {m} guesses.",
                 "I'm the setter for this game. My number is ready, and the guesser has {m} attempts."],
           'main': ['Ready.', 'OK, ready.', "I'm ready."],
           'turn': 'Your turn: if you are the guesser, reply only with your guess; if you are the setter and I have not guessed yet, '
                   'reply only with "Go ahead".',
           'control': 'This is not a move in the game, just a question: how many tries does the guesser get? Answer with the number only.',
           'go': 'Go ahead'},
}


def secret5(lang, i, combo):
    t = T[lang]
    U, m, xi, oi, ci = combo
    msgs = [{'role': 'system', 'content': t['system'].format(U=U)}, {'role': 'user', 'content': t['open'][oi]},
            {'role': 'X', 'content': t['x'][xi].format(m=m)}]
    return {'id': f'secret5_{lang}_{i:03d}', 'game': 'secret', 'kind': 'role2', 'msgs': msgs, 'edit': 2,
            'q': {'main': t['main'][ci] + ('' if lang == 'zh' else ' ') + t['turn'], 'control': t['control']},
            'expect': {'main': {'assistant': 'setter', 'user': 'guesser', 'setter_reply': t['go']},
                       'control': {'assistant': str(m), 'user': str(m)}}}


OUT.mkdir(parents=True, exist_ok=True)
for lang in ('zh', 'en'):
    rng = random.Random(f'secret4-{lang}-20261026')  # v4's seed: the same draws
    combos = [(U, m, xi, oi, ci) for U in (20, 50, 100, 200, 500, 1000) for m in range(5, 13) for xi in range(4) for oi in range(4)
              for ci in range(3)]
    items = [secret5(lang, i, c) for i, c in enumerate(rng.sample(combos, N))]
    (OUT / f'games_{lang}_secret5.json').write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    print(lang, len(items), 'items,', len({json.dumps([x['msgs'], x['q']], ensure_ascii=False) for x in items}), 'distinct;',
          items[0]['msgs'][2]['content'], '|', items[0]['q']['main'])
