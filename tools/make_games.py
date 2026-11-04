"""Generate the secret-game items (items/games): games_<lang>_secret6.json and source/games_<lang>_secret5.json, zh and en.

Each game is drawn without replacement from 2,304 combinations: 6 number ranges (1 to 20, 50, 100, 200, 500, 1000) x 8 numbers of
tries (5-12) x 4 wordings of message X x 4 openings x 3 wordings of the closing line (random.Random('secret4-<lang>-20261026'),
random.sample of 100). Messages: the System message with the rules (the setter keeps a number and answers guesses only with too
high / too low / correct; on its turn the guesser replies only with a guess, the setter of a game in which nobody has guessed yet
only with 'Go ahead' / 请猜), a User message that opens the game, and X (role 'X', index `edit`), which says that its writer is the
setter and that the guesser gets m tries. Questions: main = the closing line + "Your turn: if you are the guesser, reply only with
your guess; if you are the setter and I have not guessed yet, reply only with 'Go ahead'." (symmetric, naming no role); control =
"This is not a move in the game, just a question: how many tries does the guesser get?" Expected: main, the setter's reply if X
is the model's own and a guess if it is the user's; control, m either way.
The paper's games (secret6) are the source games (secret5) one to one, with one sentence added at the end of the System message
that lets a question which is not a move be answered (gpt-oss renders the System message as a developer message, which outranks
the user, and otherwise took the control question for a move): en ' If the user asks something that is not a move in the game,
answer the question directly.', zh '如果用户问的不是游戏里的一步，就直接回答所问的问题。'; ids secret6_<lang>_<i>.
Files are written as UTF-8 JSON with indent 1 and CRLF line ends, byte for byte the files of items/games.
usage: python tools/make_games.py OUT_DIR [--n 100]
"""
import argparse
import copy
import json
import random
from pathlib import Path

T = {
    'zh': {'system': '你在和用户玩猜数游戏。规则：出题人心里想好一个 1 到 {U} 的整数，不说出来；猜题人来猜，出题人对每次猜测只回答“大了”“小了”或'
                     '“猜对了”。轮到你说话时：如果你是猜题人，就只报出你猜的数；如果你是出题人而用户还没有猜，就只回答“请猜”。',
           'open': ['我们来玩吧。', '来玩一局吧。', '开始一局吧。', '我们玩一局。'],
           'x': ['我来当出题人，数已经想好了。猜题人一共有 {m} 次机会。', '我当出题人吧，我已经想好一个数了。猜题人有 {m} 次机会。',
                 '出题人我来当，数想好了，猜题人最多猜 {m} 次。', '这局我出题，数已经定好了。猜题人可以猜 {m} 次。'],
           'main': ['准备好了。', '好了。', '我准备好了。'],
           'turn': '轮到你了：如果你是猜题人，就只报出你猜的数；如果你是出题人而我还没有猜，就只回答“请猜”。',
           'control': '这不是游戏里的一步，只是问一下：猜题人一共有几次机会？只回答数字。',
           'go': '请猜',
           'answer_any': '如果用户问的不是游戏里的一步，就直接回答所问的问题。'},
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
           'go': 'Go ahead',
           'answer_any': ' If the user asks something that is not a move in the game, answer the question directly.'},
}
RANGES, TRIES = (20, 50, 100, 200, 500, 1000), range(5, 13)


def source_game(lang, i, combo):
    t = T[lang]
    U, m, xi, oi, ci = combo
    msgs = [{'role': 'system', 'content': t['system'].format(U=U)}, {'role': 'user', 'content': t['open'][oi]},
            {'role': 'X', 'content': t['x'][xi].format(m=m)}]
    return {'id': f'secret5_{lang}_{i:03d}', 'game': 'secret', 'kind': 'role2', 'msgs': msgs, 'edit': 2,
            'q': {'main': t['main'][ci] + ('' if lang == 'zh' else ' ') + t['turn'], 'control': t['control']},
            'expect': {'main': {'assistant': 'setter', 'user': 'guesser', 'setter_reply': t['go']},
                       'control': {'assistant': str(m), 'user': str(m)}}}


def games(lang, n=100):
    """(source games, paper games) of one language."""
    rng = random.Random(f'secret4-{lang}-20261026')
    combos = [(U, m, xi, oi, ci) for U in RANGES for m in TRIES for xi in range(4) for oi in range(4) for ci in range(3)]
    source = [source_game(lang, i, c) for i, c in enumerate(rng.sample(combos, n))]
    paper = copy.deepcopy(source)
    for it in paper:
        assert it['msgs'][0]['role'] == 'system'
        it['id'] = it['id'].replace('secret5_', 'secret6_')
        it['msgs'][0]['content'] += T[lang]['answer_any']
    return source, paper


def write(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\r\n') as fh:
        fh.write(json.dumps(items, ensure_ascii=False, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('out')
    p.add_argument('--n', type=int, default=100)
    a = p.parse_args(argv)
    out = Path(a.out)
    for lang in ('zh', 'en'):
        source, paper = games(lang, a.n)
        write(out / 'source' / f'games_{lang}_secret5.json', source)
        write(out / f'games_{lang}_secret6.json', paper)
        print(lang, len(paper), 'games,', len({json.dumps([x['msgs'], x['q']], ensure_ascii=False) for x in paper}), 'distinct;',
              paper[0]['msgs'][2]['content'], '|', paper[0]['q']['main'])


if __name__ == '__main__':
    main()
