"""Diagnostic (11-08): gpt-oss-120b body-token accuracy ~1% in mode_switch. Is the per-position logit path broken?
(1) full forward on one main-probe A rendering: acc / lp_next on body positions;
(2) the same positions from truncated prefixes (logits at the last position of ids[:t+1]) -> do they agree with (1)?
(3) greedy generation of 40 tokens after a plain user prompt (coherence).  usage: python g120_logit_diag_20261108.py MODEL [fixed_date]"""
import argparse, json, sys, datetime as dt
from pathlib import Path
sys.path.insert(0, '/root/p2_scale_20261015/crossmodel')
import crossmodel_suite as cs
from crossmodel_suite import load
from mode_switch_20261107 import renderings
import torch
m = sys.argv[1]; day = sys.argv[2] if len(sys.argv) > 2 else None
kw = {}
if day:
    d = dt.date.fromisoformat(day); kw['strftime_now'] = lambda fmt: dt.datetime(d.year, d.month, d.day).strftime(fmt)
torch_, tok, model, cfg, placement, cls = load(argparse.Namespace(model=m, max_memory_gib=90, cpu_offload_gib=0))
dev = next(model.parameters()).device
print('class', cls, 'placement', str(placement)[:200], 'dtype', next(model.parameters()).dtype, flush=True)
qc = getattr(model.config, 'quantization_config', None); print('quant', str(qc)[:300], flush=True)
en = json.loads(Path('/root/p2_scale_20261015/crossmodel/bodies_en_150.json').read_text(encoding='utf-8'))
bid = sorted(en)[0]; body = en[bid]
R = renderings(tok, kw, body, 'en', False)
for rn in ('A', 'U'):
    v = R['main'][rn]; ids = v['ids']; c = v['common']
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        lg = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True, logits_to_keep=0).logits[0].float()
    pos = c[:-1]; nxt = [ids[i + 1] for i in pos]
    lp = torch.log_softmax(lg[pos], -1)
    acc = float((lp.argmax(-1) == torch.tensor(nxt, device=lp.device)).float().mean())
    print(rn, 'n_tok', len(ids), 'body', len(c), 'acc', round(acc, 3), 'lp_next', round(float(lp[range(len(pos)), nxt].mean()), 3), flush=True)
    print('  argmax first 12:', repr(tok.decode(lp.argmax(-1)[:12].tolist())), '| actual:', repr(tok.decode(nxt[:12])), flush=True)
    diffs = []
    for t in pos[:6] + pos[len(pos) // 2: len(pos) // 2 + 3]:
        xt = torch.tensor([ids[:t + 1]], device=dev)
        with torch.no_grad():
            lt = model(input_ids=xt, attention_mask=torch.ones_like(xt), use_cache=False, return_dict=True, logits_to_keep=1).logits[0, -1].float()
        a, b = torch.log_softmax(lg[t], -1), torch.log_softmax(lt, -1)
        diffs.append((t, int(a.argmax()), int(b.argmax()), round(float((a.exp() * (a - b)).sum()), 4), ids[t + 1]))
    print('  full-vs-prefix (t, argmax_full, argmax_prefix, KL, actual):', diffs, flush=True)
msgs = [{'role': 'user', 'content': 'What is the capital of France? Answer in one sentence.'}]
p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', **kw).to(dev)
with torch.no_grad():
    g = model.generate(p, max_new_tokens=60, do_sample=False)
print('gen:', repr(tok.decode(g[0, p.shape[1]:])), flush=True)
