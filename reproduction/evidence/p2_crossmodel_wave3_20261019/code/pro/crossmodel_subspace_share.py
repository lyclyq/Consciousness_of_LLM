"""Is the manipulated subspace the shared self-index or text-specific noise (user 10-23: the epsilon check for the exact subspace
used in the interventions, both models, every fit)? The k-dimensional subspace V_l of an intervention fit (fit.pt of
crossmodel_thinking_ablate2.py or crossmodel_singlepass_ablate.py: top eigenvectors of the Assistant-User differences of the 75
DISCOVERY bodies) is applied to the 75 HELD-OUT bodies x 2 option orders, rendered under the same template setting as the fit.
At every layer, at the aligned positions of each pair of renderings (the body token by token and the template tokens after it up
to the generation prompt, as in the interventions: crossmodel_role_swap.align), with D_XY = h_X - h_Y per position:
  share_V_AU  = sum ||D_AU V||^2 / sum ||D_AU||^2   the share of a new text's own-User difference inside the fitted k dims
  share_R_AU  = the same for the random R_l of the fit (expected k / d for a direction unrelated to the difference)
  share_V_AS  = the same for own-System differences (System never entered the fit)
  us_over_au_V = sum ||D_US V||^2 / sum ||D_AU V||^2   User-System distance inside the subspace relative to own-User
pooled over positions and items, also per item (median, 10th percentile) for share_V_AU, and separately for body and post-body
positions. If the difference were text-specific noise, share_V_AU would sit near share_R_AU.
usage: crossmodel_subspace_share.py --model DIR --on-kwargs JSON --fit-from fit.pt --bodies F --splits F --lang zh|en --out F.json
       [--k 8] [--band 0-63] [--max-memory-gib 44]"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import Renderer, load, attr_question, ORDERS  # noqa: E402
from crossmodel_role_swap import align  # noqa: E402  (68a3c3e4 or later: two-segment alignment)

ROLES = ('assistant', 'user', 'system')


def run(a):
    import numpy as np
    import torch
    if a.lang == 'en':
        cs.Q_ATTR = cs.Q_ATTR_EN
    torch, tok, model, cfg, placement, cls = load(a)
    _, layers = cs.find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in (a.band or f'0-{nL - 1}').split('-'))
    band = tuple(range(b0, b1 + 1))
    R = Renderer(tok, json.loads(a.on_kwargs))
    splits = {json.loads(l)['id']: json.loads(l)['split'] for l in open(a.splits, encoding='utf-8') if l.strip()}
    bodies = json.loads(Path(a.bodies).read_text(encoding='utf-8'))
    held = [b for b in bodies if splits[b['id']] == 'confirmation']
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]
    F = torch.load(a.fit_from)
    V = {l: F['V'][l][:, :a.k].contiguous() for l in band}
    Rr = {l: F['R'][l][:, :a.k].contiguous() for l in band}
    state = {'cap': None, 'sl': None}

    def make_hook(l):
        def hook(module, inputs, output):
            if state['cap'] is not None:
                h = output[0] if isinstance(output, tuple) else output
                state['cap'][l] = h[0, state['sl']].float()  # stays on the layer's device
            return output
        return hook
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]

    def render(role, b, oi):
        ids, span = R.render([{'role': role, 'content': b['body']}, {'role': 'user', 'content': attr_question(ORDERS[oi])}], b['body'])
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)
        return ids, span[0], span[-1] + 1, g0

    keys = ('AU_V', 'AU_R', 'AU', 'AS_V', 'AS', 'US_V')
    tot = {reg: {k: np.zeros(nL) for k in keys} for reg in ('all', 'body', 'post')}
    per_item = {l: [] for l in band}
    t0 = time.time()
    n_items = 0
    for b in held:
        for oi in (0, 1):
            rend = {r: render(r, b, oi) for r in ROLES}
            # each pair of roles aligned as in the interventions (crossmodel_role_swap.align: the body token by token, the
            # template tokens after it on their common suffix; aligned body positions come first in each list)
            pairs = {}
            for x_, y_ in (('assistant', 'user'), ('assistant', 'system'), ('user', 'system')):
                ix, iy = align(rend[x_], rend[y_])
                pairs[(x_, y_)] = (ix, iy, sum(1 for i in ix if i < rend[x_][2]))
            H = {}
            for r in ROLES:
                ids = rend[r][0]
                state.update(cap={}, sl=slice(None))
                x = torch.tensor([ids], device=dev0)
                with torch.no_grad():
                    model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
                H[r] = state['cap']
                state.update(cap=None)
            for l in band:
                dev = H['assistant'][l].device
                Vl, Rl = V[l].to(dev), Rr[l].to(dev)
                D = {}
                for (x_, y_), (ix, iy, nb) in pairs.items():
                    D[x_[0].upper() + y_[0].upper()] = (H[x_][l][torch.tensor(ix, device=dev)] - H[y_][l][torch.tensor(iy, device=dev)], nb)
                (dAU, nAU), (dAS, nAS), (dUS, nUS) = D['AU'], D['AS'], D['US']
                q = {'AU_V': (((dAU @ Vl) ** 2).sum(1), nAU), 'AU_R': (((dAU @ Rl) ** 2).sum(1), nAU), 'AU': ((dAU ** 2).sum(1), nAU),
                     'AS_V': (((dAS @ Vl) ** 2).sum(1), nAS), 'AS': ((dAS ** 2).sum(1), nAS), 'US_V': (((dUS @ Vl) ** 2).sum(1), nUS)}
                for k, (v, nb) in q.items():
                    tot['all'][k][l] += float(v.sum())
                    tot['body'][k][l] += float(v[:nb].sum())
                    tot['post'][k][l] += float(v[nb:].sum())
                per_item[l].append(float(q['AU_V'][0].sum() / q['AU'][0].sum()))
            n_items += 1
        print(f'{n_items} items {time.time() - t0:.0f}s', flush=True)
    for hd in handles:
        hd.remove()
    d = V[band[0]].shape[0]
    res = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'model': a.model, 'on_kwargs': a.on_kwargs,
           'fit_from': a.fit_from, 'k': a.k, 'hidden': d, 'random_expected': a.k / d, 'n_items': n_items, 'band': [b0, b1], 'by_region': {}}
    for reg in tot:
        t = tot[reg]
        res['by_region'][reg] = {
            'share_V_AU': [round(t['AU_V'][l] / t['AU'][l], 4) if t['AU'][l] else None for l in range(nL)],
            'share_R_AU': [round(t['AU_R'][l] / t['AU'][l], 5) if t['AU'][l] else None for l in range(nL)],
            'share_V_AS': [round(t['AS_V'][l] / t['AS'][l], 4) if t['AS'][l] else None for l in range(nL)],
            'us_over_au_V': [round(t['US_V'][l] / t['AU_V'][l], 4) if t['AU_V'][l] else None for l in range(nL)]}
    res['per_item_share_V_AU'] = {str(l): {'median': round(float(np.median(v)), 4), 'p10': round(float(np.percentile(v, 10)), 4),
                                           'min': round(float(np.min(v)), 4)} for l, v in per_item.items()}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('k', 'hidden', 'random_expected', 'n_items')}), flush=True)
    for reg in ('all', 'body', 'post'):
        r = res['by_region'][reg]
        print(reg, 'share_V_AU', r['share_V_AU'][b0:b1 + 1:max(1, (b1 - b0) // 8)], 'share_R_AU max', max(x for x in r['share_R_AU'] if x is not None),
              'us_over_au_V', r['us_over_au_V'][b0:b1 + 1:max(1, (b1 - b0) // 8)], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{}')
    p.add_argument('--fit-from', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--lang', choices=('en', 'zh'), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--band', default='')
    p.add_argument('--limit-bodies', type=int, default=0)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--cpu-offload-gib', type=float, default=0)
    run(p.parse_args())


if __name__ == '__main__':
    main()
