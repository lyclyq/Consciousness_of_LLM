"""Thinking-mode answers through the developers' own APIs: the attribution question and the reasoning problems for the models
whose thinking-mode answers come from an API (GLM-5.3-Flash through Z.ai, DeepSeek-V4.1-Flash through DeepSeek; Alibaba Model
Studio (DashScope) is supported as well). The provider applies the model's chat template server-side.

Tasks (--task):
  attribution  the attribution question in thinking mode, the first two cyclic option orders, all bodies (or the 30 pilot bodies)
               x 3 roles: 900 items (180).
               Z.ai, DashScope: the main probe [ROLE: body][user: question].
               DeepSeek: the natural conversation order [user: request][ROLE: body][user: question], because the DeepSeek-V4.1
               format cannot render the main probe (an opening Assistant message gets no role token). DeepSeek's encoder merges a
               User body with its neighbours into one user turn (its native rendering of user text) and keeps a System body in
               place as a mid-conversation System message.
               The answer is the first option letter in the final content (the API returns the reasoning separately).
  problems     the 120 two-step problems, [user: problem][ROLE: step][user: final-answer question], ROLE assistant or user, step
               correct or wrong: 480 items. Every number in the final content is read; the item is 'correct' if only the correct
               answer occurs, 'adopted' if only the error-propagating answer occurs, otherwise 'other' (or 'none': no number).
Request settings (those of the paper's runs, the defaults): Z.ai and DashScope: temperature 0, max_tokens 3072, thinking enabled.
DeepSeek: thinking enabled, reasoning_effort "high", temperature 1.0 and top_p 1.0 (the model card's recommended sampling; the
thinking mode offers no greedy decoding), max_tokens 3072, model deepseek-flash (documented by the provider as served by
DeepSeek-V4.1-Flash-1017). --top-p 0.95 --no-temperature sends the earlier DeepSeek setting (top_p 0.95, the lowest the thinking
mode accepts; temperature, which thinking mode ignores, not sent).

Check of the template: the API's reported prompt_tokens must equal the count of the item under the model's own template rendered
locally; every item records whether it does. The counts come from --expected (a JSON file {model: {item: count}}), or are computed
from --tokenizer: the Hugging Face chat template of that tokenizer folder (Z.ai, DashScope), or DeepSeek's own encoder (thinking
mode, reasoning effort high; the copy in selfindex/dsv41/official, or --encoding) with the tokenizer.json of that folder (DeepSeek).
--count-tokens writes these counts to the file --out instead of running (with DeepSeek's attribution items also <out>_rendered.json,
the rendered prompt of the first body for each role).

Only finished items (finish_reason stop, non-empty content) enter the summary shares. One JSON per item in <out>/items; reruns skip
finished items, and an item whose request fails stays pending. A spend guard stops new requests once the running cost estimate
(--price-in, --price-out: USD per million tokens) reaches --budget-usd. With DeepSeek the account balance is logged before and after
(balance.log). The API key is read from the environment (ZAI_API_KEY, DASHSCOPE_API_KEY, DEEPSEEK_API_KEY), sent only in the
Authorization header and never written anywhere.
usage: python -m selfindex.api --task attribution|problems --provider zai|dashscope|deepseek (--expected F | --tokenizer DIR)
       --out DIR [--model NAME] [--bodies all|pilot] [--roles R,R] [--workers N] [--limit N] [--max-tokens N] [--top-p P]
       [--temperature T | --no-temperature] [--price-in X --price-out Y --budget-usd Z]
       python -m selfindex.api --count-tokens --task attribution|problems --provider P --tokenizer DIR --out FILE [--model NAME]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from . import items as I
from .io import atomic
from .render import ids_of

LANG = 'zh'  # the API runs use the Chinese items
PROVIDERS = {'zai': {'base': 'https://api.z.ai/api/paas/v4', 'key_env': 'ZAI_API_KEY', 'model': 'glm-5.3-flash'},
             'dashscope': {'base': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'key_env': 'DASHSCOPE_API_KEY', 'model': None},
             'deepseek': {'base': 'https://api.deepseek.com', 'key_env': 'DEEPSEEK_API_KEY', 'model': 'deepseek-flash'}}
LETTERS = ('A', 'B', 'C')
ROLES = I.ROLES
NUM = re.compile(r'-?\d+(?:\.\d+)?')
# defaults that differ by task and provider: (workers, price_in, price_out, budget_usd)
DEFAULTS = {('attribution', 'zai'): (4, 0.0, 0.0, None), ('attribution', 'dashscope'): (4, 0.0, 0.0, None),
            ('attribution', 'deepseek'): (8, 0.0, 0.0, None), ('problems', 'zai'): (6, 1.0, 4.0, 5.0),
            ('problems', 'deepseek'): (6, 1.0, 4.0, 5.0)}
ENCODING_PY = Path(__file__).resolve().parent / 'dsv41' / 'official' / 'encoding' / 'encoding.py'


# ----------------------------------------------------------------------------------------------- items
def orders():
    return I.orders()[:2]  # thinking mode uses the first two cyclic orders


def bodies(which='all'):
    return list(I.bodies(LANG)) if which == 'all' else list(I.pilot_bodies(LANG))


def attribution_items(provider, which='all', roles=ROLES):
    """[(key, body, role, order index, order, messages)], in the order of the runs."""
    natural = provider == 'deepseek'
    out = []
    for b in bodies(which):
        for oi, order in enumerate(orders()):
            q = I.attr_question(order, LANG)
            for r in roles:
                msgs = ([{'role': 'user', 'content': I.natural_order_request(LANG)}] if natural else []) + \
                       [{'role': r, 'content': b['body']}, {'role': 'user', 'content': q}]
                out.append((f"{b['id']}__{r}__o{oi}", b, r, oi, order, msgs))
    return out


def problem_items():
    """[(key, problem, step, role, messages)], in the order of the runs."""
    q = I.final_answer_question(LANG)
    return [(f"{p['id']}__{w}__{r}", p, w, r, [{'role': 'user', 'content': p['problem']}, {'role': r, 'content': p[f'step_{w}']},
                                               {'role': 'user', 'content': q}])
            for p in I.problems(LANG) for w in ('correct', 'wrong') for r in ('assistant', 'user')]


# ----------------------------------------------------------------------------------------------- prompt-token expectation
def load_encoder(path=None):
    """DeepSeek-V4.1's prompt encoder (encoding.py of the release)."""
    spec = importlib.util.spec_from_file_location('ds_encoding', Path(path or ENCODING_PY))
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


def expected_tokens(task, provider, tokenizer_dir, which='all', encoding=None):
    """{item key: prompt tokens} of every item under the model's own template, rendered locally, and for DeepSeek's attribution
    items the rendered prompt of the first body for each role ({} otherwise)."""
    shown = {}
    if provider == 'deepseek':
        E = load_encoder(encoding)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(tokenizer_dir) / 'tokenizer.json'))

        def render(m):
            return E.encode_messages(m, thinking_mode='thinking', reasoning_effort='high')

        def count(m):
            return len(tok.encode(render(m), add_special_tokens=False).ids)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_dir)

        def count(m):
            return len(ids_of(tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True)))
    out = {}
    if task == 'problems':
        for key, p, w, r, msgs in problem_items():
            out[key] = count(msgs)
        return out, shown
    first = bodies(which)[0]['id']
    for key, b, r, oi, order, msgs in attribution_items(provider, which):
        if provider == 'deepseek':
            s = render(msgs)
            if r == 'system':  # the body must stay between the request and the question, in its own System turn
                assert s.index(msgs[0]['content']) < s.index(E.SYSTEM_SP_TOKEN + b['body']) < s.rindex(msgs[-1]['content']), 'system body moved'
            out[key] = len(tok.encode(s, add_special_tokens=False).ids)
            if b['id'] == first and oi == 0:
                shown[r] = s
        else:
            out[key] = count(msgs)
    return out, shown


def write_expected(path, model, counts, shown=None):
    path = Path(path)
    json.dump({model: counts}, open(path, 'w', encoding='utf-8'), indent=0)
    if shown:
        json.dump(shown, open(str(path).replace('.json', '_rendered.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------------------------- requests
def api_key(provider):
    name = PROVIDERS[provider]['key_env']
    k = os.environ.get(name, '').strip()
    if not k:
        raise SystemExit(f'{name} is not set')
    return k


class Spend:
    def __init__(self, pin, pout, budget):
        self.pin, self.pout, self.budget, self.usd, self.lock = pin, pout, budget, 0.0, threading.Lock()

    def add(self, usage):
        with self.lock:
            self.usd += (usage.get('prompt_tokens', 0) * self.pin + usage.get('completion_tokens', 0) * self.pout) / 1e6

    def over(self):
        with self.lock:
            return self.budget is not None and self.usd >= self.budget


def request(base, k, path, body=None, tries=8):
    """POST body as JSON (GET without body); retries rate limits, server errors and network errors with growing delays."""
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


def request_body(a, msgs):
    """The request of one item, in the field order of the runs."""
    if a.provider == 'deepseek':
        body = {'model': a.model, 'messages': msgs, 'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high', 'top_p': a.top_p,
                'max_tokens': a.max_tokens, 'stream': False}
        if a.temperature is not None:
            body['temperature'] = a.temperature
        return body
    body = {'model': a.model, 'messages': msgs, 'temperature': a.temperature, 'max_tokens': a.max_tokens}
    if a.provider == 'zai':
        body['thinking'] = {'type': 'enabled'}
    else:
        body['enable_thinking'] = True
    return body


def first_letter(content):
    m = re.search(r'[ABC]', content.replace('**', ''))
    return m.group(0) if m else None


def classify(nums, p):
    c, a = str(p['answer']), str(p['answer_if_adopted'])
    if c in nums and a not in nums:
        return 'correct'
    if a in nums and c not in nums:
        return 'adopted'
    return 'other' if nums else 'none'


def run_attribution_item(a, base, k, expected, spend, item):
    path, (key, b, r, oi, order, msgs) = item
    if spend.over():
        return 'skipped (budget)'
    body = request_body(a, msgs)
    t0 = time.time()
    out = request(base, k, '/chat/completions', body)
    ch = out['choices'][0]
    msg = ch['message']
    content = msg.get('content') or ''
    usage = out.get('usage', {})
    spend.add(usage)
    letter = first_letter(content)
    exp = expected.get(key)
    if a.provider == 'deepseek':
        rec = {'id': b['id'], 'role': r, 'order': list(order), 'design': 'natural', 'mode': 'thinking',
               'model_requested': a.model, 'model_returned': out.get('model'), 'system_fingerprint': out.get('system_fingerprint'),
               'provider': 'deepseek', 'on_letter': letter, 'on_answer': order[LETTERS.index(letter)] if letter else None,
               'finish_reason': ch.get('finish_reason'), 'finished': ch.get('finish_reason') == 'stop' and bool(content.strip()),
               'content': content, 'reasoning_content': msg.get('reasoning_content') or '', 'usage': usage,
               'expected_prompt_tokens': exp, 'prompt_tokens_match': usage.get('prompt_tokens') == exp,
               'request': {kk: v for kk, v in body.items() if kk != 'messages'}, 'messages': msgs,
               'seconds': round(time.time() - t0, 1), 'time': time.strftime('%F %T')}
        tail = f"{ch.get('finish_reason')}, "
    else:
        rec = {'id': b['id'], 'role': r, 'order': list(order), 'model_requested': a.model, 'model_returned': out.get('model'),
               'provider': a.provider, 'on_letter': letter, 'on_answer': order[LETTERS.index(letter)] if letter else None,
               'finish_reason': ch.get('finish_reason'), 'finished': ch.get('finish_reason') == 'stop' and bool(content.strip()),
               'content': content, 'reasoning_content': msg.get('reasoning_content') or '', 'usage': usage, 'expected_prompt_tokens': exp,
               'prompt_tokens_match': usage.get('prompt_tokens') == exp, 'request': {kk: v for kk, v in body.items() if kk != 'messages'},
               'seconds': round(time.time() - t0, 1), 'time': time.strftime('%F %T')}
        tail = ''
    atomic(path, rec)
    return f"{r} o{oi} -> {rec['on_answer']} ({usage.get('completion_tokens')} tok, {tail}match {rec['prompt_tokens_match']})"


def run_problem_item(a, base, k, expected, spend, item):
    f, (key, p, which, role, msgs) = item
    if spend.over():
        return 'skipped (budget)'
    body = request_body(a, msgs)
    t0 = time.time()
    o = request(base, k, '/chat/completions', body)
    ch = o['choices'][0]
    m = ch['message']
    content = m.get('content') or ''
    u = o.get('usage', {})
    spend.add(u)
    finished = ch.get('finish_reason') == 'stop' and bool(content.strip())
    nums = NUM.findall(re.sub(r'<[^>]+>', ' ', content).replace('**', '').replace(',', '')) if finished else []
    exp = expected.get(key)
    rec = {'id': p['id'], 'family': p['family'], 'split': p['split'], 'step': which, 'role': role, 'answer': p['answer'],
           'answer_if_adopted': p['answer_if_adopted'], 'numbers': nums, 'class': classify(nums, p) if finished else 'none',
           'finished': finished, 'finish_reason': ch.get('finish_reason'), 'content': content,
           'reasoning_content': m.get('reasoning_content') or '', 'usage': u, 'expected_prompt_tokens': exp,
           'prompt_tokens_match': u.get('prompt_tokens') == exp, 'model_requested': a.model, 'model_returned': o.get('model'),
           'request': {kk: v for kk, v in body.items() if kk != 'messages'}, 'messages': msgs,
           'seconds': round(time.time() - t0, 1), 'time': time.strftime('%F %T')}
    atomic(f, rec)
    return f"{which} {role} -> {rec['class']} ({u.get('completion_tokens')} tok, match {rec['prompt_tokens_match']})"


# ----------------------------------------------------------------------------------------------- summaries
def summarize_attribution(out, provider, model):
    recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((out / 'items').glob('*.json'))]
    if provider == 'deepseek':
        S = {'model': model, 'n': len(recs), 'prompt_tokens_match': sum(x['prompt_tokens_match'] for x in recs),
             'models_returned': dict(Counter(x['model_returned'] for x in recs)),
             'completion_tokens': sum(x['usage'].get('completion_tokens', 0) for x in recs),
             'prompt_tokens': sum(x['usage'].get('prompt_tokens', 0) for x in recs)}
    else:
        S = {'n': len(recs), 'prompt_tokens_match': sum(x['prompt_tokens_match'] for x in recs),
             'completion_tokens': sum(x['usage'].get('completion_tokens', 0) for x in recs)}
    for r in ROLES:
        rs = [x for x in recs if x['role'] == r]
        fin = [x for x in rs if x['finished']]
        c = Counter(x['on_answer'] for x in fin)
        S[r] = {'n': len(rs), 'finished': len(fin), 'answers': dict(c), 'you_share': round(c.get('self', 0) / max(1, len(fin)), 4)}
        if provider == 'deepseek':
            S[r]['by_order'] = {f'o{oi}': dict(Counter(x['on_answer'] for x in fin if x['order'] == list(o))) for oi, o in enumerate(orders())}
    (out / 'summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(S, ensure_ascii=False))


def summarize_problems(out):
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


# ----------------------------------------------------------------------------------------------- run
def resolve(a):
    """Fill the defaults that depend on task and provider; check the combination."""
    if (a.task, a.provider) not in DEFAULTS:
        raise SystemExit(f'--task {a.task} is not available for --provider {a.provider}')
    a.model = a.model or PROVIDERS[a.provider]['model']
    if not a.model:
        raise SystemExit(f'--model is required for --provider {a.provider}')
    w, pin, pout, bud = DEFAULTS[a.task, a.provider]
    a.workers = w if a.workers is None else a.workers
    a.price_in = pin if a.price_in is None else a.price_in
    a.price_out = pout if a.price_out is None else a.price_out
    a.budget_usd = bud if a.budget_usd is None else a.budget_usd
    if a.provider != 'deepseek' and (a.no_temperature or a.top_p is not None):
        raise SystemExit('--top-p and --no-temperature apply to --provider deepseek only')
    if a.no_temperature:
        a.temperature = None
    elif a.temperature is None:
        a.temperature = 1.0 if a.provider == 'deepseek' else 0
    if a.top_p is None:
        a.top_p = 1.0
    return a


def balance(base, k, out, what):
    try:
        bal = request(base, k, '/user/balance')
    except RuntimeError as e:
        bal = {'error': str(e)[:200]}
    with open(out / 'balance.log', 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%F %T')} {what}: {json.dumps(bal)}\n")


def run(a):
    a = resolve(a)
    if a.count_tokens:
        if not a.tokenizer:
            raise SystemExit('--count-tokens needs --tokenizer')
        counts, shown = expected_tokens(a.task, a.provider, a.tokenizer, a.bodies, a.encoding)
        write_expected(a.out, a.model, counts, shown)
        print(a.model, len(counts), 'items, tokens min/max', min(counts.values()), max(counts.values()))
        return
    base = PROVIDERS[a.provider]['base']
    k = api_key(a.provider)
    if a.expected:
        expected = json.loads(Path(a.expected).read_text(encoding='utf-8'))[a.model]
    elif a.tokenizer:
        expected = expected_tokens(a.task, a.provider, a.tokenizer, a.bodies, a.encoding)[0]
    else:
        raise SystemExit('give --expected or --tokenizer (the prompt-token check)')
    out = Path(a.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    if a.task == 'attribution':
        roles = tuple(a.roles.split(','))
        todo = [(out / 'items' / f'{it[0]}.json', it) for it in attribution_items(a.provider, a.bodies, roles)]
        work = run_attribution_item
    else:
        todo = [(out / 'items' / f'{it[0]}.json', it) for it in problem_items()]
        work = run_problem_item
    todo = [t for t in todo if not t[0].exists()]
    if a.limit:
        todo = todo[:a.limit]
    if a.provider == 'deepseek':
        balance(base, k, out, f'before {len(todo)} items')
    spend = Spend(a.price_in, a.price_out, a.budget_usd)
    print(f'{len(todo)} items to run on {a.provider}:{a.model}', flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, a, base, k, expected, spend, it) for it in todo]
        for n, fu in enumerate(cf.as_completed(futs), 1):
            try:
                res = fu.result()
            except Exception as e:  # the item stays pending for a rerun
                res = f'ERROR {str(e)[:200]}'
            print(f'{n}/{len(todo)} {res} | est. ${spend.usd:.3f}', flush=True)
    if a.provider == 'deepseek':
        balance(base, k, out, 'after')
    if a.task == 'attribution':
        summarize_attribution(out, a.provider, a.model)
    else:
        summarize_problems(out)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--task', choices=('attribution', 'problems'), required=True)
    p.add_argument('--provider', choices=tuple(PROVIDERS), required=True)
    p.add_argument('--model', default=None, help='model name of the provider (default: glm-5.3-flash on Z.ai, deepseek-flash on DeepSeek)')
    p.add_argument('--out', required=True, help='the run folder; with --count-tokens the JSON file to write')
    p.add_argument('--expected', default=None, help='JSON {model: {item: prompt tokens}} of the local renderings')
    p.add_argument('--tokenizer', default=None, help='tokenizer folder for the local prompt-token counts')
    p.add_argument('--encoding', default=None, help="DeepSeek's encoding.py (default: the copy in selfindex/dsv41/official)")
    p.add_argument('--count-tokens', action='store_true', help='write the prompt-token counts of the items to --out and stop')
    p.add_argument('--bodies', choices=('all', 'pilot'), default='all', help='attribution: the 150 bodies or the 30 pilot bodies')
    p.add_argument('--roles', default=','.join(ROLES), help='attribution: restrict the roles (pilot runs)')
    p.add_argument('--workers', type=int, default=None, help='parallel requests (default 4 / 8 on DeepSeek / 6 for problems)')
    p.add_argument('--limit', type=int, default=0, help='run only the first N pending items (pilot)')
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--top-p', type=float, default=None, help='DeepSeek only (default 1.0)')
    p.add_argument('--temperature', type=float, default=None, help='default 0 (Z.ai, DashScope) or 1.0 (DeepSeek)')
    p.add_argument('--no-temperature', action='store_true', help='DeepSeek: send no temperature')
    p.add_argument('--price-in', type=float, default=None, help='USD per million prompt tokens (default 0; problems 1.0)')
    p.add_argument('--price-out', type=float, default=None, help='USD per million completion tokens (default 0; problems 4.0)')
    p.add_argument('--budget-usd', type=float, default=None, help='stop new requests at this estimate (default none; problems 5.0)')
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
