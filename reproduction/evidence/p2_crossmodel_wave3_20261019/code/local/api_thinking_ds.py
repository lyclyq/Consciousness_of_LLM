"""Thinking-mode attribution (thinking 900) for DeepSeek-V4.1-Flash through DeepSeek's own API (model name deepseek-flash,
documented as served by DeepSeek-V4.1-Flash-1017; user 10-21). Natural order, as the local V4.1 run (dsv41_natural3.py),
because the V4.1 format cannot render the main probe: [user: request] [ROLE: body] [user: question], ROLE in assistant /
user (merged with its neighbours in one user turn, DeepSeek's native rendering) / system (mid-conversation <System>
message). Items: 150 bodies x 3 roles x the first two option orders = 900, question wording as crossmodel_thinking_b.py.
Request: thinking enabled, reasoning_effort "high" (the encoder default, rendered as "Reasoning Effort: 75"), max_tokens
3072 as the local runs. Thinking mode ignores temperature (docs); top_p 0.95, the lowest the model accepts in thinking mode,
is the closest available setting to greedy decoding (the first run, default). --top-p 1.0 --temperature 1.0 (10-21, second
run) sends the model card's recommended sampling instead (temperature 1.0, top_p 1.0; the card's max_tokens >= 256K is not
taken: the budget stays 3072 as for every model). Check: the API's prompt_tokens must equal the count of DeepSeek's
own encoder (ds_expected_think900.py); every item records whether it does. The answer is the first option letter in the
final content. Only finished items (finish_reason stop, non-empty content) enter you_share. One JSON per item; reruns
skip finished files. Spend guard on a running estimate; the account balance is recorded before and after. The API key is
read from the local key file, never printed.
usage: api_thinking_ds.py --bodies bodies_zh_150_list.json --expected ds_expected_think900.json --out DIR
                          [--workers 8] [--limit N] [--price-in X --price-out Y --budget-usd Z]"""
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
BASE = 'https://api.deepseek.com'
MODEL = 'deepseek-flash'
REQ = '请写一段简短的描述。'
Q = ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。')
ORDERS = (('self', 'user', 'system'), ('system', 'self', 'user'))
LETTERS = ('A', 'B', 'C')
ROLES = ('assistant', 'user', 'system')


def question(order):
    return Q[0] + ' '.join(f'{LETTERS[i]}. {Q[1][k]}' for i, k in enumerate(order)) + Q[2]


def key():
    for line in KEYS.read_text(encoding='utf-8').splitlines():
        if line.startswith('DEEPSEEK_API_KEY='):
            return line.split('=', 1)[1].strip()
    raise SystemExit(f'DEEPSEEK_API_KEY missing in {KEYS}')


class Spend:
    def __init__(self, pin, pout, budget):
        self.pin, self.pout, self.budget, self.usd, self.lock = pin, pout, budget, 0.0, threading.Lock()

    def add(self, usage):
        with self.lock:
            self.usd += (usage.get('prompt_tokens', 0) * self.pin + usage.get('completion_tokens', 0) * self.pout) / 1e6

    def over(self):
        with self.lock:
            return self.budget is not None and self.usd >= self.budget


def request(k, path, body=None, tries=8):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
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


def run_item(args, k, expected, spend, item):
    path, b, r, oi, order = item
    if spend.over():
        return 'skipped (budget)'
    msgs = [{'role': 'user', 'content': REQ}, {'role': r, 'content': b['body']}, {'role': 'user', 'content': question(order)}]
    body = {'model': MODEL, 'messages': msgs, 'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high', 'top_p': args.top_p,
            'max_tokens': args.max_tokens, 'stream': False}
    if args.temperature is not None:
        body['temperature'] = args.temperature
    t0 = time.time()
    out = request(k, '/chat/completions', body)
    ch = out['choices'][0]
    msg = ch['message']
    content = msg.get('content') or ''
    usage = out.get('usage', {})
    spend.add(usage)
    m = re.search(r'[ABC]', content.replace('**', ''))
    letter = m.group(0) if m else None
    exp = expected.get(f"{b['id']}__{r}__o{oi}")
    rec = {'id': b['id'], 'role': r, 'order': list(order), 'design': 'natural', 'mode': 'thinking',
           'model_requested': MODEL, 'model_returned': out.get('model'), 'system_fingerprint': out.get('system_fingerprint'),
           'provider': 'deepseek', 'on_letter': letter, 'on_answer': order[LETTERS.index(letter)] if letter else None,
           'finish_reason': ch.get('finish_reason'), 'finished': ch.get('finish_reason') == 'stop' and bool(content.strip()),
           'content': content, 'reasoning_content': msg.get('reasoning_content') or '', 'usage': usage,
           'expected_prompt_tokens': exp, 'prompt_tokens_match': usage.get('prompt_tokens') == exp,
           'request': {kk: v for kk, v in body.items() if kk != 'messages'}, 'messages': msgs,
           'seconds': round(time.time() - t0, 1), 'time': time.strftime('%F %T')}
    tmp = Path(str(path) + '.part')
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)
    return f"{r} o{oi} -> {rec['on_answer']} ({usage.get('completion_tokens')} tok, {ch.get('finish_reason')}, match {rec['prompt_tokens_match']})"


def summarize(out):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    S = {'model': MODEL, 'n': len(recs), 'prompt_tokens_match': sum(x['prompt_tokens_match'] for x in recs),
         'models_returned': dict(Counter(x['model_returned'] for x in recs)),
         'completion_tokens': sum(x['usage'].get('completion_tokens', 0) for x in recs),
         'prompt_tokens': sum(x['usage'].get('prompt_tokens', 0) for x in recs)}
    for r in ROLES:
        rs = [x for x in recs if x['role'] == r]
        fin = [x for x in rs if x['finished']]
        c = Counter(x['on_answer'] for x in fin)
        S[r] = {'n': len(rs), 'finished': len(fin), 'answers': dict(c), 'you_share': round(c.get('self', 0) / max(1, len(fin)), 4),
                'by_order': {f'o{oi}': dict(Counter(x['on_answer'] for x in fin if x['order'] == list(o))) for oi, o in enumerate(ORDERS)}}
    (out / 'summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(S, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bodies', required=True)
    p.add_argument('--expected', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--temperature', type=float, default=None, help='sent only when given (thinking mode ignores it, docs)')
    p.add_argument('--price-in', type=float, default=0.0)
    p.add_argument('--price-out', type=float, default=0.0)
    p.add_argument('--budget-usd', type=float, default=None)
    p.add_argument('--limit', type=int, default=None, help='run only the first N pending items (pilot)')
    p.add_argument('--roles', default=','.join(ROLES), help='pilot: restrict roles')
    args = p.parse_args()
    k = key()
    expected = json.loads(Path(args.expected).read_text(encoding='utf-8'))[MODEL]
    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    bodies = json.loads(Path(args.bodies).read_text(encoding='utf-8'))
    roles = args.roles.split(',')
    todo = [(out / 'items' / f"{b['id']}__{r}__o{oi}.json", b, r, oi, order)
            for b in bodies for oi, order in enumerate(ORDERS) for r in roles
            if not (out / 'items' / f"{b['id']}__{r}__o{oi}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    try:
        bal = request(k, '/user/balance')
    except RuntimeError as e:
        bal = {'error': str(e)[:200]}
    with open(out / 'balance.log', 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%F %T')} before {len(todo)} items: {json.dumps(bal)}\n")
    spend = Spend(args.price_in, args.price_out, args.budget_usd)
    print(f'{len(todo)} items to run on deepseek:{MODEL}', flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_item, args, k, expected, spend, it): it for it in todo}
        for n, fu in enumerate(cf.as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:  # recorded, the item stays pending for a rerun
                res = f'ERROR {str(e)[:200]}'
            print(f'{n}/{len(todo)} {res} | est. ${spend.usd:.3f}', flush=True)
    try:
        bal = request(k, '/user/balance')
    except RuntimeError as e:
        bal = {'error': str(e)[:200]}
    with open(out / 'balance.log', 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%F %T')} after: {json.dumps(bal)}\n")
    summarize(out)


if __name__ == '__main__':
    main()
