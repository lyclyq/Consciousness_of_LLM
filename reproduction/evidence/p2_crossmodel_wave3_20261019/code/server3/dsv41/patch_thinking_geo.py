"""One-off patch (10-21): add --thinking and --geometry-only to dsv41_natural3.py (from sha 1e5e3fb9). usage: patch_thinking_geo.py FILE"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding='utf-8')


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''Attribution: probability of the option letters at the answer position, three cyclic option orders.
usage: dsv41_natural3.py --model-dir D --bodies role_bodies.jsonl --out DIR [--limit N] [--wave-seqs N] [--check-single]"""''',
    '''Attribution: probability of the option letters at the answer position, three cyclic option orders.
--thinking (2026-10-21): render in DeepSeek's thinking mode (reasoning effort "high", the encoder default) instead of chat
mode, the counterpart of the thinking-rendering geometry of the other models; with this encoder the two renderings differ only
in the "Reasoning Effort" line that opens the conversation and in the last token (<think> instead of </think>).
--geometry-only: the geometry sequences only, no attribution sequences. Without these flags nothing changes.
usage: dsv41_natural3.py --model-dir D --bodies role_bodies.jsonl --out DIR [--limit N] [--wave-seqs N] [--check-single]
                         [--thinking] [--geometry-only]"""''')
rep('''    def __init__(self, E, tok):
        self.E, self.tok = E, tok
        self.backend = tok.backend_tokenizer''',
    '''    def __init__(self, E, tok, thinking=False):
        self.E, self.tok = E, tok
        self.backend = tok.backend_tokenizer
        self.kw = {'thinking_mode': 'thinking', 'reasoning_effort': 'high'} if thinking else {'thinking_mode': 'chat'}''')
rep("        text = self.E.encode_messages(msgs, thinking_mode='chat')", '        text = self.E.encode_messages(msgs, **self.kw)')
rep('''    R = Renderer(E, tok)
    inf = Path(a.model_dir) / 'inference'
''', '''    R = Renderer(E, tok, a.thinking)
    inf = Path(a.model_dir) / 'inference'
''')
rep("'not_built': 'vision tower, aligner, DSpark/MTP layers (text-only prefill)', 'thinking_mode': 'chat',",
    "'not_built': 'vision tower, aligner, DSpark/MTP layers (text-only prefill)', **R.kw, 'geometry_only': a.geometry_only,")
rep('''            items.append({'key': f"{b['id']}__{c}__geo", 'ids': geo[c][0], 'capture': spans[c]})
            for oi, order in enumerate(ORDERS):''', '''            items.append({'key': f"{b['id']}__{c}__geo", 'ids': geo[c][0], 'capture': spans[c]})
            for oi, order in enumerate(() if a.geometry_only else ORDERS):''')
rep("        pick = [geo_items[0], [it for it in items if it['capture'] is None][0]]",
    "        pick = [geo_items[0]] + [it for it in items if it['capture'] is None][:1]")
rep('''    assemble(out, meta)
    atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0, 'expert_s': stats['expert_s']})''',
    '''    assemble(out, meta, a.geometry_only)
    atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0, 'expert_s': stats['expert_s']})''')
rep('''    R = Renderer(E, tok)
    bodies = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()]''',
    '''    R = Renderer(E, tok, a.thinking)
    bodies = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()]''')
rep("            for k in ['geo'] + [f'o{i}' for i in range(len(ORDERS))]:",
    "            for k in ['geo'] + ([] if a.geometry_only else [f'o{i}' for i in range(len(ORDERS))]):")
rep("        for k in ('script_sha256', 'kernel_sha256', 'official_model_py_sha256', 'encoding_py_sha256', 'tokenizer_json_sha256', 'config'):",
    """        for k in ('script_sha256', 'kernel_sha256', 'official_model_py_sha256', 'encoding_py_sha256', 'tokenizer_json_sha256', 'config',
                  'thinking_mode', 'geometry_only'):""")
rep('''    assemble(out, meta)
    atomic(out / 'status.json', {'state': 'complete', 'shards': st,''', '''    assemble(out, meta, a.geometry_only)
    atomic(out / 'status.json', {'state': 'complete', 'shards': st,''')
rep('''def assemble(out, meta):
    """Per body: N/<id>.json and N/<id>.npz in the crossmodel_natural3 layout (user copied from user_merged)."""''',
    '''def assemble(out, meta, geometry_only=False):
    """Per body: N/<id>.json and N/<id>.npz in the crossmodel_natural3 layout (user copied from user_merged); with
    geometry_only the records carry no attribution fields."""''')
rep('''            arrs[f'content_{c}'], arrs[f'onset_{c}'] = z['content'], z['onset']
            ps = []''', '''            arrs[f'content_{c}'], arrs[f'onset_{c}'] = z['content'], z['onset']
            if geometry_only:
                rec['conds'][c] = {'n_body_tokens': len(m['spans'][c]), 'n_tokens': m['n_tokens'][c]}
                continue
            ps = []''')
rep('''    S['attr'] = {c: {k: float(np.mean([r['conds'][c]['mean'][k] for r in recs])) for k in ('self', 'user', 'system')} for c in CONDS}
    S['attr_bodies_own_above_external'] = int(sum(r['conds']['assistant']['mean']['self'] >
                                                  max(r['conds']['user']['mean']['self'], r['conds']['system']['mean']['self']) for r in recs))
    S['letter_mass_min'] = float(min(r['conds'][c]['letter_mass_min'] for r in recs for c in CONDS))
''', '''    has_attr = all('mean' in r['conds'][c] for r in recs for c in CONDS)
    if has_attr:
        S['attr'] = {c: {k: float(np.mean([r['conds'][c]['mean'][k] for r in recs])) for k in ('self', 'user', 'system')} for c in CONDS}
        S['attr_bodies_own_above_external'] = int(sum(r['conds']['assistant']['mean']['self'] >
                                                      max(r['conds']['user']['mean']['self'], r['conds']['system']['mean']['self']) for r in recs))
        S['letter_mass_min'] = float(min(r['conds'][c]['letter_mass_min'] for r in recs for c in CONDS))
''')
rep("    print('answers:', {c: {k: round(v, 3) for k, v in d.items()} for c, d in S['attr'].items()}, flush=True)",
    """    if has_attr:
        print('answers:', {c: {k: round(v, 3) for k, v in d.items()} for c, d in S['attr'].items()}, flush=True)""")
rep("    p.add_argument('--finalize', action='store_true', help='after all shards: assemble and summarize (no model loaded)')",
    """    p.add_argument('--finalize', action='store_true', help='after all shards: assemble and summarize (no model loaded)')
    p.add_argument('--thinking', action='store_true', help='render in thinking mode (reasoning effort high) instead of chat mode')
    p.add_argument('--geometry-only', action='store_true', help='geometry sequences only, no attribution sequences')""")
p.write_text(s, encoding='utf-8', newline='\n')
print('patched', p)
