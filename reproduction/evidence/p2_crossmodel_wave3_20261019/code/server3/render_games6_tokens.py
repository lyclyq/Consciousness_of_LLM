"""Token-level view of one secret game item under one readout's template (user 10-27: show exactly what is rendered and which
positions the replacement writes), with the runner's own functions: crossmodel_games.make_render (rendering, X span, generation
prompt) and crossmodel_role_swap.align (recipient and donor positions that are edited). For X as the model's own message (recipient
assistant, donor user: removal) and as a User message (recipient user, donor assistant: installation), main question and control,
every recipient token is printed with its index and E (edited; with the aligned donor token) or . (left as it is). Tokenizer only.
usage: render_games6_tokens.py MODEL_DIR TEMPLATE_KWARGS ITEMS ITEM_INDEX OUT_TXT"""
import json
import sys
from pathlib import Path

model, kw, items, k, out = sys.argv[1:6]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_games as G  # noqa: E402
from crossmodel_role_swap import align  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
R = G.Renderer(tok, json.loads(kw))
cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>', '<|turn>')]
b = json.loads(Path(items).read_text(encoding='utf-8'))[int(k)]
L = [f'##### {model} | template kwargs {kw} | {items} item {b["id"]}']
for q in ('main', 'control'):
    render = G.make_render(R, cands, q)
    for rec, don in (('assistant', 'user'), ('user', 'assistant')):
        r, d = render(rec, b), render(don, b)
        ri, di = align(r, d)
        pair = dict(zip(ri, di))
        ids = r[0]
        L.append(f'\n=== question {q}: X = {rec} (recipient), donor = same conversation with X = {don}; '
                 f'{len(ri)} of {len(ids)} prompt positions edited; X body {r[1]}-{r[2] - 1}, generation prompt from {r[3]}')
        for i, t in enumerate(ids):
            s = tok.decode([t]).replace('\n', '\\n')
            if i in pair:
                L.append(f'{i:4d} E  {s!r:24s} <- donor {pair[i]:4d} {tok.decode([d[0][pair[i]]]).replace(chr(10), chr(92) + "n")!r}')
            else:
                L.append(f'{i:4d} .  {s!r}')
Path(out).write_text('\n'.join(L) + '\n', encoding='utf-8')
print('written', out, len(L), 'lines')
