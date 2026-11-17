"""Thinking-mode self-reference through a first-party OpenAI-compatible API (Z.ai GLM; Alibaba Model Studio Qwen).

Same items, question wording and option orders as crossmodel_thinking_b.py (pilot bodies x 3 roles x the first two cyclic
orders): the conversation is [ROLE: body][user: question] and the provider applies the model's chat template server-side.
Check of the template: the API's reported prompt_tokens must equal the count of the official template rendered locally
(expected_tokens.py); every item records whether it does. The answer is the first option letter in the final content
(after the reasoning, which the API returns separately). Greedy where the API allows it (temperature 0); max_tokens
3072 as the local runs (reasoning plus answer). One JSON per item; reruns skip finished items. A spend guard stops new
requests once the running cost estimate reaches --budget-usd. The API key is read from a local key file, never printed.
usage: api_thinking.py --provider zai --model glm-5.3 --bodies pilot_bodies_zh.json --expected expected_tokens.json
                       --out DIR [--workers 4] [--price-in 1.4 --price-out 4.4 --budget-usd 3]"""
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
PROVIDERS = {'zai': ('https://api.z.ai/api/paas/v4', 'ZAI_API_KEY'),
             'dashscope': ('https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'DASHSCOPE_API_KEY')}
Q = ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。')
ORDERS = (('self', 'user', 'system'), ('system', 'self', 'user'))
LETTERS = ('A', 'B', 'C')
ROLES = ('assistant', 'user', 'system')


def question(order):
    return Q[0] + ' '.join(f'{LETTERS[i]}. {Q[1][k]}' for i, k in enumerate(order)) + Q[2]


def key_for(name):
    for line in KEYS.read_text(encoding='utf-8').splitlines():
        if line.startswith(name + '='):
            return line.split('=', 1)[1].strip()
    raise SystemExit(f'{name} missing in {KEYS}')


class Spend:
    def __init__(self, pin, pout, budget):
        self.pin, self.pout, self.budget, self.usd, self.lock = pin, pout, budget, 0.0, threading.Lock()

    def add(self, usage):
        with self.lock:
            self.usd += (usage.get('prompt_tokens', 0) * self.pin + usage.get('completion_tokens', 0) * self.pout) / 1e6

    def over(self):
        with self.lock:
            return self.budget is not None and self.usd >= self.budget


def call(base, key, body, tries=8):
    req = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(), method='POST',
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
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


def run_item(args, base, key, expected, spend, item):
    path, b, r, oi, order = item
    if spend.over():
        return 'skipped (budget)'
    msgs = [{'role': r, 'content': b['body']}, {'role': 'user', 'content': question(order)}]
    body = {'model': args.model, 'messages': msgs, 'temperature': 0, 'max_tokens': args.max_tokens}
    if args.provider == 'zai':
        body['thinking'] = {'type': 'enabled'}
    else:
        body['enable_thinking'] = True
    t0 = time.time()
    out = call(base, key, body)
    ch = out['choices'][0]
    msg = ch['message']
    content = msg.get('content') or ''
    reasoning = msg.get('reasoning_content') or ''
    usage = out.get('usage', {})
    spend.add(usage)
    m = re.search(r'[ABC]', content.replace('**', ''))
    letter = m.group(0) if m else None
    exp = expected.get(f"{b['id']}__{r}__o{oi}")
    rec = {'id': b['id'], 'role': r, 'order': list(order), 'model_requested': args.model, 'model_returned': out.get('model'),
           'provider': args.provider, 'on_letter': letter, 'on_answer': order[LETTERS.index(letter)] if letter else None,
           'finish_reason': ch.get('finish_reason'), 'finished': ch.get('finish_reason') == 'stop' and bool(content.strip()),
           'content': content, 'reasoning_content': reasoning, 'usage': usage, 'expected_prompt_tokens': exp,
           'prompt_tokens_match': usage.get('prompt_tokens') == exp, 'request': {k: v for k, v in body.items() if k != 'messages'},
           'seconds': round(time.time() - t0, 1), 'time': time.strftime('%F %T')}
    tmp = Path(str(path) + '.part')
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)
    return f"{r} o{oi} -> {rec['on_answer']} ({usage.get('completion_tokens')} tok, match {rec['prompt_tokens_match']})"


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {'n': len(recs), 'prompt_tokens_match': sum(x['prompt_tokens_match'] for x in recs),
         'completion_tokens': sum(x['usage'].get('completion_tokens', 0) for x in recs)}
    for r in ROLES:
        rs = [x for x in recs if x['role'] == r]
        fin = [x for x in rs if x['finished']]
        c = Counter(x['on_answer'] for x in fin)
        S[r] = {'n': len(rs), 'finished': len(fin), 'answers': dict(c), 'you_share': round(c.get('self', 0) / max(1, len(fin)), 4)}
    (out / 'summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(S, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--provider', choices=PROVIDERS, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--expected', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--price-in', type=float, default=0.0)
    p.add_argument('--price-out', type=float, default=0.0)
    p.add_argument('--budget-usd', type=float, default=None)
    p.add_argument('--limit', type=int, default=None, help='run only the first N pending items (pilot)')
    args = p.parse_args()
    base, kname = PROVIDERS[args.provider]
    key = key_for(kname)
    expected = json.loads(Path(args.expected).read_text(encoding='utf-8'))[args.model]
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    bodies = json.loads(Path(args.bodies).read_text(encoding='utf-8'))
    todo = []
    for b in bodies:
        for oi, order in enumerate(ORDERS):
            for r in ROLES:
                f = out / 'items' / f"{b['id']}__{r}__o{oi}.json"
                if not f.exists():
                    todo.append((f, b, r, oi, order))
    if args.limit:
        todo = todo[:args.limit]
    spend = Spend(args.price_in, args.price_out, args.budget_usd)
    print(f'{len(todo)} items to run on {args.provider}:{args.model}', flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_item, args, base, key, expected, spend, it): it for it in todo}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:  # recorded, the item stays pending for a rerun
                res = f'ERROR {str(e)[:200]}'
            print(f'{k}/{len(todo)} {res} | est. ${spend.usd:.3f}', flush=True)
    summarize(out)


if __name__ == '__main__':
    main()
