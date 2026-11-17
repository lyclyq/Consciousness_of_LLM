"""Print the decoded renderings of one body (main and natural, every role) for a model, as mode_switch_20261107.py builds them.
Read-only check before a run (no model, no GPU).  usage: render_dump_20261108.py --model DIR --tmpl-kwargs JSON --bodies F
       [--lang en --bodies-en F] [--fixed-date D]"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossmodel_suite import checked_tokenizer  # noqa: E402
from persistence_readout_20261106 import natural_system_ok  # noqa: E402
from mode_switch_20261107 import REQ, renderings  # noqa: E402
import crossmodel_suite as cs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--tmpl-kwargs', default='{}')
ap.add_argument('--bodies', required=True)
ap.add_argument('--lang', choices=('zh', 'en'), default='zh')
ap.add_argument('--bodies-en', default=None)
ap.add_argument('--fixed-date', default=None)
a = ap.parse_args()
kw = json.loads(a.tmpl_kwargs)
if a.fixed_date:
    d = dt.date.fromisoformat(a.fixed_date)
    kw['strftime_now'] = lambda fmt: dt.datetime(d.year, d.month, d.day).strftime(fmt)
b = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()][0]
if a.lang == 'en':
    b = {**b, 'body': json.loads(Path(a.bodies_en).read_text(encoding='utf-8'))[b['id']]}
from transformers import AutoTokenizer  # noqa: E402
tok = checked_tokenizer(AutoTokenizer.from_pretrained(a.model, trust_remote_code=True), a.model)
q = cs.Q_REPLY_EN if a.lang == 'en' else cs.Q_REPLY
sys_ok = natural_system_ok(tok, kw, q, REQ[a.lang], b['body'])
print(json.dumps({'model': a.model, 'body': b['id'], 'tmpl_kwargs': a.tmpl_kwargs, 'natural_system_ok': sys_ok}, ensure_ascii=False))
R = renderings(tok, kw, b['body'], a.lang, sys_ok)
for order, rr in R.items():
    for rn, v in rr.items():
        print(f'===== {order} {rn}: {len(v["ids"])} tokens, body tokens {len(v["common"])}, close {tok.decode([v["close"]])!r}')
        print(tok.decode(v['ids']))
