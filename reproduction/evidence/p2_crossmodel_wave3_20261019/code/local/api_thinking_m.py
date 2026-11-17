"""Own-step trust WITH thinking through a developer's API (user 10-21), for the models whose thinking answers come from an API:
GLM-5.3-Flash (Z.ai) and DeepSeek-V4.1-Flash (DeepSeek, model name deepseek-flash). Items and messages as crossmodel_thinking_m.py:
120 problems x step correct / wrong x ROLE assistant / user, [user: problem] [ROLE: step] [user: final-answer question] (Chinese).
Request settings are those of each model's thinking-900 run: Z.ai {temperature 0, max_tokens 3072, thinking enabled}
(api_thinking.py); DeepSeek {thinking enabled, reasoning_effort high, temperature 1.0, top_p 1.0, max_tokens 3072}
(api_thinking_ds.py --top-p 1.0 --temperature 1.0). prompt_tokens must equal the count under the official template
(expected_tokens_m.py); every item records whether it does. The answer is read from the final content (the API returns the
reasoning separately): every number in it, classed as in crossmodel_thinking_m.py. Resumable; spend guard; the DeepSeek balance
is logged before and after; keys are read from the local key file, never printed.
usage: api_thinking_m.py --provider zai|deepseek --problems problems_zh_120.json --expected F --out DIR [--workers 6] [--limit N]"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

KEYS = Path(r'<home>\.config\p2_api\keys.env')
Q_FINAL = '这道题的最终答案是多少？只输出数字。'
NUM = re.compile(r'-?\d+(?:\.\d+)?')
PROV = {'zai': ('https://api.z.ai/api/paas/v4', 'ZAI_API_KEY', 'glm-5.3-flash',
                {'temperature': 0, 'max_tokens': 3072, 'thinking': {'type': 'enabled'}}),
        'deepseek': ('https://api.deepseek.com', 'DEEPSEEK_API_KEY', 'deepseek-flash',
                     {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high', 'top_p': 1.0, 'max_tokens': 3072,
                      'stream': False, 'temperature': 1.0})}


def key(name):
    for line in KEYS.read_text(encoding='utf-8').splitlines():
        if line.startswith(name + '='):
            return line.split('=', 1)[1].strip()
    raise SystemExit(f'{name} missing in {KEYS}')


def request(base, k, path, body=None, tries=8):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 method='GET' if body is None else 'POST',
                                 headers={'Authorization': 'Bearer ' + k, 'Content-Type': 'application/json'})
    delay = 5
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')[:300]
            if e.code in (429, 500, 502, 503, 504) and t < tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f'HTTP {e.code}: {msg}')
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if t < tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f'network: {e}')


def classify(nums, p):
    c, a = str(p['answer']), str(p['answer_if_adopted'])
    if c in nums and a not in nums:
        return 'correct'
    if a in nums and c not in nums:
        return 'adopted'
    return 'other' if nums else 'none'


class Spend:
    def __init__(self, pin, pout, budget):
        self.pin, self.pout, self.budget, self.usd, self.lock = pin, pout, budget, 0.0, threading.Lock()

    def add(self, u):
        with self.lock:
            self.usd += (u.get('prompt_tokens', 0) * self.pin + u.get('completion_tokens', 0) * self.pout) / 1e6

    def over(self):
        with self.lock:
            return self.budget is not None and self.usd >= self.budget


def run_item(a, base, k, model, extra, expected, spend, item):
    f, p, which, role = item
    if spend.over():
        return 'skipped (budget)'
    msgs = [{'role': 'user', 'content': p['problem']}, {'role': role, 'content': p[f'step_{which}']}, {'role': 'user', 'content': Q_FINAL}]
    body = {'model': model, 'messages': msgs, **extra}
    t0 = time.time()
    o = request(base, k, '/chat/completions', body)
    ch = o['choices'][0]
    m = ch['message']
    content = m.get('content') or ''
    u = o.get('usage', {})
    spend.add(u)
    finished = ch.get('finish_reason') == 'stop' and bool(content.strip())
    nums = NUM.findall(re.sub(r'<[^>]+>', ' ', content).replace('**', '').replace(',', '')) if finished else []
    exp = expected.get(f"{p['id']}__{which}__{role}")
    rec = {'id': p['id'], 'family': p['family'], 'split': p['split'], 'step': which, 'role': role, 'answer': p['answer'],
           'answer_if_adopted': p['answer_if_adopted'], 'numbers': nums, 'class': classify(nums, p) if finished else 'none',
           'finished': finished, 'finish_reason': ch.get('finish_reason'), 'content': content,
           'reasoning_content': m.get('reasoning_content') or '', 'usage': u, 'expected_prompt_tokens': exp,
           'prompt_tokens_match': u.get('prompt_tokens') == exp, 'model_requested': model, 'model_returned': o.get('model'),
           'request': {kk: v for kk, v in body.items() if kk != 'messages'}, 'messages': msgs,
           'seconds': round(time.time() - t0, 1), 'time': time.strftime('%F %T')}
    tmp = Path(str(f) + '.part')
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    tmp.replace(f)
    return f"{which} {role} -> {rec['class']} ({u.get('completion_tokens')} tok, match {rec['prompt_tokens_match']})"


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {'n': len(recs), 'prompt_tokens_match': sum(r['prompt_tokens_match'] for r in recs),
         'unfinished': sum(not r['finished'] for r in recs), 'models_returned': dict(Counter(r['model_returned'] for r in recs))}
    for which in ('correct', 'wrong'):
        for role in ('assistant', 'user'):
            rs = [r for r in recs if r['step'] == which and r['role'] == role and r['finished']]
            c = Counter(r['class'] for r in rs)
            S[f'{which}_{role}'] = {'n_finished': len(rs), **dict(c), 'adopted_share': round(c.get('adopted', 0) / max(1, len(rs)), 4),
                                    'correct_share': round(c.get('correct', 0) / max(1, len(rs)), 4)}
    (out / 'summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(S, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', choices=PROV, required=True)
    ap.add_argument('--problems', required=True)
    ap.add_argument('--expected', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--price-in', type=float, default=1.0)
    ap.add_argument('--price-out', type=float, default=4.0)
    ap.add_argument('--budget-usd', type=float, default=5.0)
    a = ap.parse_args()
    base, kname, model, extra = PROV[a.provider]
    k = key(kname)
    expected = json.loads(Path(a.expected).read_text(encoding='utf-8'))[model]
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    problems = json.loads(Path(a.problems).read_text(encoding='utf-8'))
    todo = [(out / 'items' / f"{p['id']}__{w}__{r}.json", p, w, r) for p in problems for w in ('correct', 'wrong')
            for r in ('assistant', 'user') if not (out / 'items' / f"{p['id']}__{w}__{r}.json").exists()]
    if a.limit:
        todo = todo[:a.limit]
    if a.provider == 'deepseek':
        with open(out / 'balance.log', 'a', encoding='utf-8') as fh:
            fh.write(f"{time.strftime('%F %T')} before {len(todo)} items: {json.dumps(request(base, k, '/user/balance'))}\n")
    spend = Spend(a.price_in, a.price_out, a.budget_usd)
    print(f'{len(todo)} items on {a.provider}:{model}', flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_item, a, base, k, model, extra, expected, spend, it) for it in todo]
        for n, fu in enumerate(cf.as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:  # the item stays pending for a rerun
                res = f'ERROR {str(e)[:200]}'
            print(f'{n}/{len(todo)} {res} | est. ${spend.usd:.3f}', flush=True)
    if a.provider == 'deepseek':
        with open(out / 'balance.log', 'a', encoding='utf-8') as fh:
            fh.write(f"{time.strftime('%F %T')} after: {json.dumps(request(base, k, '/user/balance'))}\n")
    summarize(out)


if __name__ == '__main__':
    main()
