"""The shared displacement in the edited subspace: is the subspace of the causal tests the shared self-index of new texts, or
text-specific noise of the fitted ones? (template-native renderings only)

The k-dimensional subspace V_l of a fit (fit.pt of selfindex.fits or fit_q.pt of selfindex.role_swap --make-fits: the leading
eigenvectors of the Assistant-User differences of the discovery bodies) is applied to the held-out bodies x the first two option
orders (attribution question), rendered in the template setting of the fit. Each body is run unedited under the three roles; at
every layer of --band, at the aligned positions of each pair of renderings (render.align: the body token by token and the
template tokens after it up to the generation prompt, as in the causal tests), with D_XY = h_X - h_Y per position:
  share_V_AU   = sum ||D_AU V||^2 / sum ||D_AU||^2   the share of a new text's own-User difference inside the fitted k dimensions
  share_R_AU   = the same for the random R_l of the fit (k / d expected for a direction unrelated to the difference)
  share_V_AS   = the same for own-System differences (System never entered the fit)
  us_over_au_V = sum ||D_US V||^2 / sum ||D_AU V||^2   User-System distance inside the subspace relative to own-User
pooled over positions and items, for all aligned positions and separately for the body and the template tokens after it (post),
and per item (median, 10th percentile, minimum) for share_V_AU. If the difference were text-specific noise, share_V_AU would sit
near share_R_AU.
Output: one JSON file (--out): by_region.{all,body,post} with one value per layer (None outside the band), per_item_share_V_AU per
layer of the band, k, hidden size, random_expected = k / d, the number of items.
usage: python -m selfindex.subspace_share --model DIR --on-kwargs JSON --lang zh|en --fit-from FIT --out F.json [--k 8] [--band A-B]
         [--limit-bodies N] [--body-ids FILE]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import __version__
from . import items as I
from .load import find_layers, load_model
from .render import Renderer, align
from .role_swap import selected_bodies

ROLES = I.ROLES
TURN_STARTS = ('<|start|>', '<|im_start|>')


def run(a, loaded=None):
    import numpy as np
    L_ = loaded or load_model(a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib)
    torch, tok, model = L_.torch, L_.tok, L_.model
    _, layers = find_layers(model)
    nL = len(layers)
    b0, b1 = (int(x) for x in (a.band or f'0-{nL - 1}').split('-'))
    band = tuple(range(b0, b1 + 1))
    R = Renderer(tok, json.loads(a.on_kwargs))
    ORDERS = I.orders()
    held = [b for b in selected_bodies(a) if b['split'] == 'confirmation']
    if a.limit_bodies:
        held = held[:a.limit_bodies]
    dev0 = model.get_input_embeddings().weight.device
    cands = [tok.convert_tokens_to_ids(t) for t in TURN_STARTS]
    F = torch.load(a.fit_from, weights_only=True)
    V = {l: F['V'][l][:, :a.k].contiguous() for l in band}
    Rr = {l: F['R'][l][:, :a.k].contiguous() for l in band}
    st = {'cap': None, 'sl': None}

    def make_hook(l):
        def hook(module, inputs, output):
            if st['cap'] is not None:
                h = output[0] if isinstance(output, tuple) else output
                st['cap'][l] = h[0, st['sl']].float()  # stays on the layer's device
            return output
        return hook

    def render(role, b, oi):
        msgs = [{'role': role, 'content': b['body']}, {'role': 'user', 'content': I.attr_question(ORDERS[oi], a.lang)}]
        ids, span = R.render(msgs, b['body'])
        g0 = max(i for i in range(len(ids)) if ids[i] in cands)
        return ids, span[0], span[-1] + 1, g0

    keys = ('AU_V', 'AU_R', 'AU', 'AS_V', 'AS', 'US_V')
    tot = {reg: {k: np.zeros(nL) for k in keys} for reg in ('all', 'body', 'post')}
    per_item = {l: [] for l in band}
    handles = [layers[l].register_forward_hook(make_hook(l)) for l in band]
    t0 = time.time()
    n_items = 0
    try:
        for b in held:
            for oi in (0, 1):
                rend = {r: render(r, b, oi) for r in ROLES}
                pairs = {}  # each pair aligned as in the causal tests; the aligned body positions come first in each list
                for x_, y_ in (('assistant', 'user'), ('assistant', 'system'), ('user', 'system')):
                    ix, iy = align(rend[x_], rend[y_])
                    pairs[(x_, y_)] = (ix, iy, sum(1 for i in ix if i < rend[x_][2]))
                H = {}
                for r in ROLES:
                    st.update(cap={}, sl=slice(None))
                    x = torch.tensor([rend[r][0]], device=dev0)
                    with torch.no_grad():
                        model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
                    H[r] = st['cap']
                    st.update(cap=None)
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
    finally:
        for hd in handles:
            hd.remove()
    d = V[band[0]].shape[0]
    res = {'selfindex_version': __version__, 'model': a.model, 'on_kwargs': a.on_kwargs, 'fit_from': a.fit_from, 'k': a.k, 'hidden': d,
           'random_expected': a.k / d, 'n_items': n_items, 'band': [b0, b1], 'by_region': {}}
    for reg in tot:
        t = tot[reg]
        res['by_region'][reg] = {
            'share_V_AU': [round(t['AU_V'][l] / t['AU'][l], 4) if t['AU'][l] else None for l in range(nL)],
            'share_R_AU': [round(t['AU_R'][l] / t['AU'][l], 5) if t['AU'][l] else None for l in range(nL)],
            'share_V_AS': [round(t['AS_V'][l] / t['AS'][l], 4) if t['AS'][l] else None for l in range(nL)],
            'us_over_au_V': [round(t['US_V'][l] / t['AU_V'][l], 4) if t['AU_V'][l] else None for l in range(nL)]}
    res['per_item_share_V_AU'] = {str(l): {'median': round(float(np.median(v)), 4), 'p10': round(float(np.percentile(v, 10)), 4),
                                           'min': round(float(np.min(v)), 4)} for l, v in per_item.items()}
    res['lang'], res['body_ids'] = a.lang, a.body_ids
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('k', 'hidden', 'random_expected', 'n_items')}), flush=True)
    for reg in ('all', 'body', 'post'):
        r = res['by_region'][reg]
        print(reg, 'share_V_AU', r['share_V_AU'][b0:b1 + 1:max(1, (b1 - b0) // 8)], 'share_R_AU max', max(x for x in r['share_R_AU'] if x is not None),
              'us_over_au_V', r['us_over_au_V'][b0:b1 + 1:max(1, (b1 - b0) // 8)], flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--on-kwargs', default='{}', help='chat-template keyword arguments of the rendering of the fit (JSON)')
    p.add_argument('--lang', choices=('zh', 'en'), required=True)
    p.add_argument('--fit-from', required=True, help='fit file with the subspaces V and R')
    p.add_argument('--out', required=True, help='output JSON file')
    p.add_argument('--k', type=int, default=8, help='dimensions of the subspace (the leading directions of V and R)')
    p.add_argument('--band', default='', help='first-last layer, e.g. 0-63 (default: every layer)')
    p.add_argument('--limit-bodies', type=int, default=0, help='the first N held-out bodies only')
    p.add_argument('--body-ids', default='', help='JSON list of body ids (or objects with an id): only these bodies, in this order')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU memory')
    return p


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
