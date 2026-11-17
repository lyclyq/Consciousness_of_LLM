"""Output-side breadth of the 18-condition self-space contrasts (PREREG_output_breadth_20261107_zh.md).

For every stored body token at L63 (the output of the last decoder layer), apply the model's final norm (Qwen3_5RMSNorm:
x / rms * (1 + w)) and the unembedding, and compare conditions pairwise:
  dz  = logits_X - logits_Y, centred over the vocabulary (softmax is invariant to a constant);
  dp  = softmax(logits_X) - softmax(logits_Y);
  PR(v) = (sum v^2)^2 / sum v^4, the effective number of vocabulary entries that carry the change;
  top-k share = share of sum v^2 in the k largest entries; KL(X || Y) per token.
Per body: median over its body tokens (all, first third, last third). Across bodies: mean over the held-out bodies, 95% interval
by resampling the 15 wording families (2,000 resamples). Shared change: the mean of dz over all held-out tokens, its PR, its
share |mean|^2 / mean |dz|^2, and the vocabulary entries with the largest positive and negative values.
Sanity check: top-1 next-token accuracy of condition A against the actual next body token.
No forward passes; reads the stored states only.
usage: output_breadth_20261107.py --run DIR --model DIR --out JSON [--device cuda:0] [--limit N]"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PAIRS = [('A', 'U'), ('A', 'S'), ('A', 'A_switch'), ('U', 'S'), ('A_switch', 'U'), ('U_p0', 'U'), ('A_p0', 'A'), ('S_p0', 'S'),
         ('A_p0', 'U_p0'), ('A_p0', 'S_p0'), ('U_you', 'U_me'), ('U_hospital', 'U_school'), ('U_yesterday', 'U_tomorrow'),
         ('U_ming', 'U_hong'), ('U_pleasant', 'U_painful'), ('U_p0', 'U_p0alt')]
KS = (10, 100, 1000)
NB = 2000


def prt(v):
    s2 = (v * v).sum(-1)
    return s2 * s2 / (v ** 4).sum(-1)


def topshare(v):
    e = v * v
    tot = e.sum(-1)
    cs = torch.cumsum(torch.topk(e, max(KS), dim=-1).values, -1)
    return {k: cs[..., k - 1] / tot for k in KS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA: this script does not fall back to the CPU')
    run, M = Path(args.run), Path(args.model)
    ident = json.loads((run / 'identity.json').read_text(encoding='utf-8'))
    names = [c['name'] for c in ident['conditions']]
    ci = {n: i for i, n in enumerate(names)}
    keep = list(ident['keep_layers'])
    assert ident['n_layers'] == 64 and keep[-1] == 63, (ident['n_layers'], keep)
    li = keep.index(63)
    cfg = json.loads((M / 'config.json').read_text(encoding='utf-8'))
    tc = cfg.get('text_config', cfg)
    eps = float(tc['rms_norm_eps'])
    wm = json.loads((M / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']

    def get(k):
        with safe_open(str(M / wm[k]), framework='pt', device='cpu') as f:
            return f.get_tensor(k)
    dev = torch.device(args.device)
    wn = get('model.language_model.norm.weight').float().to(dev)
    W = get('lm_head.weight').to(dev).float()
    V = W.shape[0]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M))
    metas = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(run.glob('*.json')) if p.name != 'identity.json']
    if args.limit:
        metas = metas[:args.limit]
    rows, fam_sum, fam_sq = [], {}, {}
    acc_hit = acc_n = 0
    t0 = time.time()
    for bi, m in enumerate(metas):
        assert m['conditions'] == names, m['id']
        z = np.load(run / f"{m['id']}.npz")
        h = torch.from_numpy(np.ascontiguousarray(z['tokens'][:, li])).to(dev).float()      # (C, T, d)
        T = h.shape[1]
        assert T == m['n_body_tokens'] == len(m['body_ids']), m['id']
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + wn)
        logits = hn @ W.T                                                                     # (C, T, V)
        nxt = torch.tensor(m['body_ids'][1:], device=dev)
        acc_hit += int((logits[ci['A'], :T - 1].argmax(-1) == nxt).sum())
        acc_n += T - 1
        lp = torch.log_softmax(logits, -1)
        a3 = max(1, T // 3)
        first, last = slice(0, a3), slice(T - a3, T)
        row = {'id': m['id'], 'split': m['split'], 'family': m['category'], 'T': T, 'pairs': {}}
        for X, Y in PAIRS:
            dz = logits[ci[X]] - logits[ci[Y]]
            dz = dz - dz.mean(-1, keepdim=True)
            pX = lp[ci[X]].exp()
            dp = pX - lp[ci[Y]].exp()
            kl = (pX * (lp[ci[X]] - lp[ci[Y]])).sum(-1)
            a, b = prt(dz), prt(dp)
            tz, tp = topshare(dz), topshare(dp)

            def md(x, s=slice(None)):
                return float(x[s].median())
            rec = {'pr_dz': md(a), 'pr_dz_first': md(a, first), 'pr_dz_last': md(a, last),
                   'pr_dp': md(b), 'pr_dp_first': md(b, first), 'pr_dp_last': md(b, last), 'kl': md(kl),
                   'dz_norm': md(dz.norm(dim=-1))}
            rec.update({f'top{k}_dz': md(tz[k]) for k in KS})
            rec.update({f'top{k}_dp': md(tp[k]) for k in KS})
            row['pairs'][f'{X}-{Y}'] = rec
            if m['split'] == 'confirmation':
                key = f'{X}-{Y}'
                s, n = fam_sum.setdefault(key, {}).get(m['category'], (torch.zeros(V, device=dev), 0))
                fam_sum[key][m['category']] = (s + dz.sum(0), n + T)
                q = fam_sq.setdefault(key, {})
                q[m['category']] = q.get(m['category'], 0.0) + float((dz * dz).sum())
        rows.append(row)
        del logits, lp, h, hn
        if bi % 10 == 0 or bi == len(metas) - 1:
            print(f'{bi + 1}/{len(metas)} bodies, {time.time() - t0:.0f}s, A top-1 next-token accuracy so far '
                  f'{acc_hit / max(acc_n, 1):.3f}', flush=True)

    conf = [r for r in rows if r['split'] == 'confirmation']
    disc = [r for r in rows if r['split'] != 'confirmation']
    fams = sorted({r['family'] for r in conf})
    rng = np.random.default_rng(20261107)
    draws = [rng.integers(0, len(fams), len(fams)) for _ in range(NB)]
    out = {'experiment': 'P2_selfspace_output_breadth_20261107',
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'run_identity_sha256': hashlib.sha256((run / 'identity.json').read_bytes()).hexdigest(),
           'model': str(M), 'layer': 63, 'eps': eps, 'vocab': V, 'gaussian_pr_reference': V / 3,
           'n_confirmation': len(conf), 'n_discovery': len(disc), 'families': len(fams), 'bootstrap': NB,
           'A_next_token_top1_accuracy': acc_hit / max(acc_n, 1), 'pairs': {}}
    for X, Y in PAIRS:
        p = f'{X}-{Y}'
        res = {}
        for key in rows[0]['pairs'][p]:
            fm = np.array([sum(r['pairs'][p][key] for r in conf if r['family'] == f) for f in fams])
            fn = np.array([sum(1 for r in conf if r['family'] == f) for f in fams])
            bs = [fm[d].sum() / fn[d].sum() for d in draws]
            res[key] = {'confirmation': float(fm.sum() / fn.sum()),
                        'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                        'discovery': float(np.mean([r['pairs'][p][key] for r in disc])) if disc else None}
        S = torch.stack([fam_sum[p][f][0] for f in fams])
        Nf = torch.tensor([fam_sum[p][f][1] for f in fams], device=dev, dtype=torch.float32)
        Q = torch.tensor([fam_sq[p][f] for f in fams], device=dev, dtype=torch.float64)
        sh = S.sum(0) / Nf.sum()
        bpr, bsh = [], []
        for d in draws:
            di = torch.as_tensor(d, device=dev)
            v = S[di].sum(0) / Nf[di].sum()
            bpr.append(float(prt(v)))
            bsh.append(float((v * v).sum()) / float(Q[di].sum() / Nf[di].sum()))
        res['shared'] = {'pr': float(prt(sh)), 'pr_ci95': [float(np.quantile(bpr, 0.025)), float(np.quantile(bpr, 0.975))],
                         'share': float((sh * sh).sum()) / float(Q.sum() / Nf.sum()),
                         'share_ci95': [float(np.quantile(bsh, 0.025)), float(np.quantile(bsh, 0.975))],
                         'norm': float(sh.norm())}
        up, dn = torch.topk(sh, 15), torch.topk(-sh, 15)
        res['shared']['up'] = [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(up.values, up.indices)]
        res['shared']['down'] = [[tok.decode([int(i)]), round(float(-v), 4)] for v, i in zip(dn.values, dn.indices)]
        out['pairs'][p] = res
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('written', args.out, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
