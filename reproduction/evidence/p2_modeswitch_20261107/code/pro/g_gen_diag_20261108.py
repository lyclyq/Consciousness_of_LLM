"""Diagnostic 2 (11-08): does gpt-oss (20b / 120b) generate coherently here, and predict plain text?  usage: python g_gen_diag_20261108.py MODEL"""
import argparse, sys, datetime as dt
sys.path.insert(0, '/root/p2_scale_20261015/crossmodel')
from crossmodel_suite import load
import torch
m = sys.argv[1]
torch_, tok, model, cfg, placement, cls = load(argparse.Namespace(model=m, max_memory_gib=90, cpu_offload_gib=0))
dev = next(model.parameters()).device
kw = {'strftime_now': lambda fmt: dt.datetime(2026, 10, 19).strftime(fmt)}
txt = 'The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is'
ids = tok(txt, return_tensors='pt')['input_ids'].to(dev)
with torch.no_grad():
    lg = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0].float()
lp = torch.log_softmax(lg[:-1], -1); nxt = ids[0, 1:]
print('plain acc', round(float((lp.argmax(-1) == nxt).float().mean()), 3), 'lp', round(float(lp[range(len(nxt)), nxt].mean()), 3))
print('plain argmax:', repr(tok.decode(lp.argmax(-1).tolist())))
print('next after Spain is:', repr(tok.decode([int(lg[-1].argmax())])))
msgs = [{'role': 'user', 'content': 'What is the capital of France? Answer in one sentence.'}]
s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
p = tok(s, return_tensors='pt', add_special_tokens=False)['input_ids'].to(dev)
with torch.no_grad():
    g = model.generate(input_ids=p, attention_mask=torch.ones_like(p), max_new_tokens=120, do_sample=False)
print('gen:', repr(tok.decode(g[0, p.shape[1]:])))
