"""Checks and assembles the phase-3 body bank (40 domains x 10 bodies, Chinese + English) written for the persistence
experiment. Rules (the brief given to the writers): Chinese 70-110 characters, 6-10 clauses ('，' + '。'), only '，' and '。',
ends with '。', third-person declarative, no first/second person, no role or conversation words, no Latin letters; English
ends with '.', only ',' and '.', none of the listed English words. Also: unique ids, no duplicate or near-duplicate texts.
Split: in every domain, of the two bodies of each genre the first goes to discovery and the second to confirmation.
Each body is its own resampling unit (category = id); domain and genre are kept for a domain-level check.
usage: build_phase3_bodies_20261106.py SET_DIR OUT_DIR   (writes phase3_bodies_zh.jsonl, phase3_bodies_en.json, phase3_check.json)"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BAN_ZH = ['我', '你', '您', '咱', '自己', '本人', '用户', '助手', '系统', '模型', '人工智能', '消息', '对话', '回复', '回答', '聊天', '写',
          '请', '要求', '必须', '应该', '需要', '吗']
BAN_EN = ['i', 'me', 'my', 'we', 'our', 'us', 'you', 'your', 'user', 'users', 'assistant', 'system', 'systems', 'model', 'models', 'ai',
          'message', 'messages', 'chat', 'reply', 'replies']
GENRES = ['记录', '描述', '过程', '事件', '说明']
rows, problems = [], []
for f in sorted(SRC.glob('set*.jsonl')):
    for ln, line in enumerate(f.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception as e:
            problems.append(f'{f.name}:{ln} bad json {e}')
            continue
        r['_src'] = f'{f.name}:{ln}'
        rows.append(r)


def check(r):
    p = []
    z, e = r.get('body_zh', ''), r.get('body_en', '')
    if not 70 <= len(z) <= 110:
        p.append(f'zh length {len(z)}')
    nc = z.count('，') + z.count('。')
    if not 6 <= nc <= 10:
        p.append(f'zh clauses {nc}')
    if not z.endswith('。'):
        p.append('zh does not end with 。')
    bad = sorted({ch for ch in z if not (ch.isalnum() or ch in '，。')})
    if bad:
        p.append(f'zh punctuation {bad}')
    if re.search('[A-Za-z]', z):
        p.append('zh has Latin letters')
    hits = [w for w in BAN_ZH if w in z]
    if hits:
        p.append(f'zh banned {hits}')
    if not e.endswith('.'):
        p.append('en does not end with .')
    bad_en = sorted({ch for ch in e if not (ch.isalnum() or ch in " ,.-'")})
    if bad_en:
        p.append(f'en punctuation {bad_en}')
    toks = re.findall(r"[A-Za-z]+", e.lower())
    hits_en = sorted({w for w in toks if w in BAN_EN})
    if hits_en:
        p.append(f'en banned {hits_en}')
    if r.get('genre') not in GENRES:
        p.append(f'genre {r.get("genre")}')
    return p


ok, bad_rows = [], []
for r in rows:
    p = check(r)
    (bad_rows if p else ok).append((r, p))
ids = Counter(r['id'] for r, _ in ok)
dup_ids = [i for i, c in ids.items() if c > 1]
seen, near = {}, []
for r, _ in ok:
    key = r['body_zh'][:12]
    if key in seen:
        near.append((seen[key], r['id']))
    seen[key] = r['id']
exact = [i for i, c in Counter(r['body_zh'] for r, _ in ok).items() if c > 1]
by_dom = defaultdict(list)
for r, _ in ok:
    by_dom[r['domain']].append(r)
out_zh, out_en = [], {}
for dom in sorted(by_dom):
    rs = sorted(by_dom[dom], key=lambda x: x['id'])
    for g in GENRES:
        gr = [x for x in rs if x['genre'] == g]
        for j, x in enumerate(gr):
            split = 'discovery' if j % 2 == 0 else 'confirmation'
            bid = f'p3_{x["id"]}'
            out_zh.append({'id': bid, 'body': x['body_zh'], 'split': split, 'category': bid, 'domain': dom, 'genre': g,
                           'body_sha256_utf8': hashlib.sha256(x['body_zh'].encode()).hexdigest()})
            out_en[bid] = x['body_en']
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'phase3_bodies_zh.jsonl').write_bytes(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out_zh).encode('utf-8'))
(OUT / 'phase3_bodies_en.json').write_bytes(json.dumps(out_en, ensure_ascii=False, indent=1).encode('utf-8') + b'\n')
rep = {'rows_read': len(rows), 'rows_ok': len(ok), 'rows_failed': len(bad_rows), 'json_problems': problems,
       'failed': [{'id': r.get('id'), 'src': r['_src'], 'problems': p} for r, p in bad_rows],
       'duplicate_ids': dup_ids, 'exact_duplicates': exact, 'same_first_12_chars': near,
       'domains': {d: len(v) for d, v in sorted(by_dom.items())},
       'split': dict(Counter(r['split'] for r in out_zh)), 'genre': dict(Counter(r['genre'] for r in out_zh)),
       'zh_chars_min_max': [min(len(r['body']) for r in out_zh), max(len(r['body']) for r in out_zh)] if out_zh else None}
(OUT / 'phase3_check.json').write_bytes(json.dumps(rep, ensure_ascii=False, indent=1).encode('utf-8') + b'\n')
print(json.dumps({k: rep[k] for k in ('rows_read', 'rows_ok', 'rows_failed', 'duplicate_ids', 'exact_duplicates', 'split', 'zh_chars_min_max')}, ensure_ascii=False))
print('domains', len(rep['domains']), 'failed examples', [(x['id'], x['problems']) for x in rep['failed'][:12]])
