"""The items of the paper (items/ at the repository root, or the folder named by the environment variable SELFINDEX_ITEMS).

Questions are built from items/templates.json exactly as the experiments built them: head + "A. <option> B. <option> C. <option>"
+ tail, the options in the given order and separated by single spaces.
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path

ROLES = ('assistant', 'user', 'system')


def items_dir() -> Path:
    return Path(os.environ.get('SELFINDEX_ITEMS') or Path(__file__).resolve().parents[1] / 'items')


@lru_cache(maxsize=None)
def templates() -> dict:
    return json.loads((items_dir() / 'templates.json').read_text(encoding='utf-8'))


def orders():
    """The three cyclic option orders; single-pass self-reference uses all three, thinking mode and the causal tests the first two."""
    return [tuple(o) for o in templates()['attribution']['orders']]


def attr_question(order, lang='zh', kind='attribution'):
    """The attribution question in one option order. kind: 'attribution' (the main wording), 'attribution_other_wording' (the held-out
    wording of the stricter tasks: other head, same options, letters and tail) or 'attribution_numbered' (options numbered 1-3)."""
    t = templates()
    a = t['attribution']
    head, opt, tail = a[lang]['head'], a[lang]['options'], a[lang]['tail']
    marks = a['letters']
    if kind == 'attribution_other_wording':
        head = t['attribution_other_wording'][lang]
    elif kind == 'attribution_numbered':
        marks, tail = t['attribution_numbered']['digits'], t['attribution_numbered']['tail'][lang]
    elif kind != 'attribution':
        raise ValueError(kind)
    return head + ' '.join(f'{marks[i]}. {opt[k]}' for i, k in enumerate(order)) + tail


def yes_no_question(which, lang='zh', order=0):
    """The yes/no questions of the stricter tasks: which = 'yn_self' or 'yn_user'; order 0 or 1 is the order of the answer words."""
    t = templates()['yes_no']
    return t['questions'][lang][which] + t['tails'][lang][order]


def reply_request(lang='zh'):
    return templates()['reply_request'][lang]


def final_answer_question(lang='zh'):
    return templates()['final_answer'][lang]


def content_question(lang='zh'):
    return templates()['content'][lang]


def arithmetic_question(body_id, lang='zh'):
    """One two-digit multiplication per body, fixed by the body id. Returns (question, expected answer)."""
    rng = random.Random(body_id)
    x, y = rng.randint(13, 97), rng.randint(13, 97)
    q = templates()['arithmetic'][lang].format(x=x, y=y)
    return q, str(x * y)


def natural_order_request(lang='zh'):
    return templates()['natural_order']['request'][lang]


@lru_cache(maxsize=None)
def bodies(lang='zh'):
    """The 150 bodies with id, category (wording family), split ('discovery' or 'confirmation', the held-out half) and body text."""
    rows = [json.loads(l) for l in open(items_dir() / 'bodies' / 'role_bodies_zh.jsonl', encoding='utf-8') if l.strip()]
    if lang == 'en':
        en = json.loads((items_dir() / 'bodies' / 'bodies_en_150.json').read_text(encoding='utf-8'))
        rows = [{**r, 'body': en[r['id']]} for r in rows]
    return tuple(rows)


def pilot_bodies(lang='zh'):
    return json.loads((items_dir() / 'bodies' / f'pilot_bodies_{lang}.json').read_text(encoding='utf-8'))


def problems(lang='zh'):
    return json.loads((items_dir() / 'problems' / f'problems_{lang}_120.json').read_text(encoding='utf-8'))


def games(lang='zh'):
    return json.loads((items_dir() / 'games' / f'games_{lang}_secret6.json').read_text(encoding='utf-8'))
