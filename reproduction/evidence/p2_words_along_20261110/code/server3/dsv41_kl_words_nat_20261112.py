"""P2 ordinary words in next-token prediction for DeepSeek-V4.1-Flash-1017, natural order only (paper Table tab:self_natural3), 11-12.
Exploratory and internal, like the rest of this archive: not for the release or the data bundle.
The measurement of kl_words_orders_20261111.py (HF models; order natural/natural3), on the paper's DeepSeek runner
dsv41_natural3.py (official inference code, pure-PyTorch kernels, release weights read by memory map) with the grouped forward of
the paper-run KL port dsv41_modeswitch_nat3_20261108.py (46e977c4, forward_all: every sequence on its own, layer by layer over a
group, attention state handed down per sequence, the model's own final step applied at the body positions):
  roles   A, U, S rendered as that KL run (dsv41_modeswitch_nat3_20261108.render_body: [user: REQ][ROLE: body][user: Q_REPLY] in chat
          mode, the encoder merging the user turns), their common body positions (dsv41_natural3.common_body); KL(A||U), KL(A||S),
          KL(U||S), full and content (control tokens removed, each renormalised), median over the body positions but the last
          (dsv41_modeswitch_nat3_20261108.body_record, unchanged)
  words   the words-along DeepSeek renderings (dsv41_words_along_20261110.render: the body as a User message preceded by the
          frame 以下是{w}的记录：, Chinese, the 32 pairs of words_v2.json), their common body positions; KL in both directions,
          full and content, median over the positions but the last (as kl_words_orders_20261111.py)
Check (as kl_words_orders_20261111.py): every body's role KLs against the paper-run KL of this model
(modeswitch/nat3/dsv41_flash/bodies/<id>.json, pairs[pn][kl_body|kl_content]), relative difference <= 0.02, recorded per body.
The smoke (--smoke) measures the first 2 held-out bodies; a shard run (--shard k --n-shards 4, one process per GPU on every 4th
held-out body, as the paper's runs) requires the smoke's passed check.json of this script. --finalize (no model): every body on
disk, summary (family resampling, ratios to U-S and A-U, as kl_words_orders_20261111.summarize), DONE.
Output in the format of kl_words_orders_20261111.py: OUT/bodies/<id>.json, OUT/check.json, OUT/identity.json, OUT/summary.json.
usage: dsv41_kl_words_nat_20261112.py --model-dir D --bodies role_bodies.jsonl --words words_v2.json --out DIR --check-ref DIR
       (--smoke | --shard k --n-shards n | --finalize | --dry-run) [--group-bodies 1] [--max-seq 1024] [--stop-file F]"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import dsv41_natural3 as D  # noqa: E402
import dsv41_modeswitch_nat3_20261108 as KN  # noqa: E402
import dsv41_words_along_20261110 as WA  # noqa: E402

ROLES = KN.ROLES
NB = 2000
CHECK_REL = 0.02


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def held_out(a):
    bodies = [json.loads(l) for l in open(a.bodies, encoding='utf-8') if l.strip()]
    bodies = [b for b in bodies if b.get('split') == 'confirmation']
    assert len(bodies) == 75, len(bodies)
    return bodies


def plan(R, b, pairs):
    """sequences of one body: the three roles (the KL run's rendering), then w1, w2 of every pair (the words-along rendering);
    per sequence the positions whose next-token distribution is compared (common body positions but the last)"""
    Rb = KN.render_body(R, b['body'])
    seqs, poss, meta = [], [], {'roles': Rb, 'pairs': {}}
    for rn, _ in ROLES:
        seqs.append(Rb[rn]['ids'])
        poss.append(Rb[rn]['common'][:-1])
    body = b['body'].rstrip()
    for p in pairs:
        r2 = {f'w{i + 1}': WA.render(R, 'user', p['frame'].format(w=w) + body, body) for i, w in enumerate(p['w'])}
        c2 = D.common_body(r2)
        meta['pairs'][p['id']] = {'i': len(seqs), 'n_positions': len(c2['w1']) - 1,
                                  'token_counts': [len(r2['w1'][0]), len(r2['w2'][0])]}
        for w in ('w1', 'w2'):
            seqs.append(r2[w][0])
            poss.append(c2[w][:-1])
    return seqs, poss, meta


def kl_med(a, b):
    return float((a.exp() * (a - b)).sum(-1).median())


def content(lp, mask):
    lc = lp[:, ~mask]
    return lc - torch.logsumexp(lc, -1, keepdim=True)


def body_record(b, meta, lps, mask, ref_dir):
    """lps: per sequence of plan(), log-probabilities [positions + 1, vocab] on the host (forward_all; last row = last position)"""
    rr = KN.body_record(b, meta['roles'], {rn: lps[k] for k, (rn, _) in enumerate(ROLES)}, mask)  # the KL run's role record
    rec = {'id': b['id'], 'split': b.get('split'), 'category': b.get('category'), 'order': 'natural3', 'roles': rr['pairs'],
           'role_stats': rr['roles'], 'words': {}}
    for pid, m in meta['pairs'].items():
        l1, l2 = lps[m['i']][:-1], lps[m['i'] + 1][:-1]
        c1, c2 = content(l1, mask), content(l2, mask)
        rec['words'][pid] = {'kl_body_12': kl_med(l1, l2), 'kl_body_21': kl_med(l2, l1),
                             'kl_content_12': kl_med(c1, c2), 'kl_content_21': kl_med(c2, c1),
                             'n_positions': m['n_positions'], 'token_counts': m['token_counts']}
    ref = json.loads((Path(ref_dir) / 'bodies' / f'{b["id"]}.json').read_text(encoding='utf-8'))['pairs']
    assert set(ref) == set(rec['roles']), ('role pairs differ from the reference', sorted(ref), sorted(rec['roles']))
    rels = [abs(rec['roles'][pn][k] - ref[pn][k]) / max(abs(ref[pn][k]), 1e-12) for pn in ref for k in ('kl_body', 'kl_content')]
    rec['check'] = {'max_rel': float(max(rels)), 'ok': bool(max(rels) <= CHECK_REL), 'kinds': ['kl_body', 'kl_content']}
    return rec


def summarize(out, pair_ids):
    """kl_words_orders_20261111.summarize (2c9d0922), unchanged but for D.atomic"""
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    fams = sorted({r['category'] for r in rows})
    fidx = np.array([fams.index(r['category']) for r in rows])
    rng = np.random.default_rng(20261111)
    draws = [rng.integers(0, len(fams), len(fams)) for _ in range(NB)]

    def fs(v):
        return np.array([v[fidx == f].sum() for f in range(len(fams))])

    def ratio(a, b):
        fa, fb = fs(a), fs(b)
        bs = np.array([fa[d].sum() / fb[d].sum() for d in draws])
        return {'ratio': float(a.sum() / b.sum()), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}, bs
    res = {'n_bodies': len(rows), 'families': len(fams), 'roles_pairs': list(rows[0]['roles']), 'kinds': {}}
    bases = [b for b in ('U-S', 'A-U') if b in rows[0]['roles']]
    for k in ('kl_content', 'kl_body'):
        V = {pn: np.array([r['roles'][pn][k] for r in rows]) for pn in rows[0]['roles']}
        for pid in pair_ids:
            V[f'{pid}:12'] = np.array([r['words'][pid][f'{k}_12'] for r in rows])
            V[f'{pid}:21'] = np.array([r['words'][pid][f'{k}_21'] for r in rows])
        o = {'mean_per_body': {n: float(v.mean()) for n, v in V.items()}}
        bsd = {}
        for base in bases:
            key = 'to_US' if base == 'U-S' else 'to_AU'
            o[key], bsd[key] = {}, {}
            for n, v in V.items():
                o[key][n], bsd[key][n] = ratio(v, V[base])
        for d in ('12', '21'):
            names = [f'{pid}:{d}' for pid in pair_ids]
            for key in bsd:
                med = np.median(np.stack([bsd[key][n] for n in names]), 0)
                o[f'word_median_{key}_{d}'] = {'ratio': float(np.median([o[key][n]['ratio'] for n in names])),
                                               'ci95': [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))],
                                               'max': float(max(o[key][n]['ratio'] for n in names)),
                                               'min': float(min(o[key][n]['ratio'] for n in names))}
        res['kinds'][k] = o
    res['check_max_rel'] = float(max(r['check']['max_rel'] for r in rows))
    D.atomic(out / 'summary.json', res)
    c = res['kinds']['kl_content']
    s = []
    if 'to_US' in c:
        s.append('A-U/U-S {:.2f}, A-S/U-S {:.2f}; words/U-S median {:.3f} [{:.3f}, {:.3f}] (max {:.3f})'.format(
            c['to_US']['A-U']['ratio'], c['to_US']['A-S']['ratio'], c['word_median_to_US_12']['ratio'],
            *c['word_median_to_US_12']['ci95'], c['word_median_to_US_12']['max']))
    s.append('words/A-U median {:.4f} [{:.4f}, {:.4f}] (max {:.4f})'.format(
        c['word_median_to_AU_12']['ratio'], *c['word_median_to_AU_12']['ci95'], c['word_median_to_AU_12']['max']))
    print(f'summary (content, {out.name}): ' + '; '.join(s) + '; check max rel {:.2e}'.format(res['check_max_rel']), flush=True)


def dry_run(a):
    E, tok = KN.encoder_and_tok(a.model_dir)
    R = D.Renderer(E, tok, False)
    W, pairs = WA.load_words(a.words)
    bodies = held_out(a)
    n_seq, lmax, npos = 0, 0, []
    for b in bodies:
        seqs, poss, meta = plan(R, b, pairs)
        wa_seqs, _, _ = WA.plan(R, b['body'], pairs, check_runner=True)   # the words-along run's sequences
        assert [s for s in seqs[3:]] == [s[0] for s in wa_seqs[3:]], f'{b["id"]}: word renderings differ from the words-along run'
        n_seq += len(seqs)
        lmax = max(lmax, max(len(s) for s in seqs))
        npos += [len(p) for p in poss]
    ref = json.loads((Path(a.check_ref) / 'identity.json').read_text(encoding='utf-8'))
    print(json.dumps({'bodies': len(bodies), 'order': 'natural3', 'lang': 'zh', 'rows': ['A-U', 'A-S', 'U-S'] + [p['id'] for p in pairs],
                      'forwards': n_seq, 'max_tokens': lmax, 'positions_min_max': [min(npos), max(npos)], 'thinking_mode': R.kw,
                      'n_control_ids': len(KN.control_ids(tok)), 'ref_n_control_ids': ref.get('n_control_ids'),
                      'ref_script': ref.get('script_sha256', '')[:8]}, ensure_ascii=False))
    assert len(KN.control_ids(tok)) == ref.get('n_control_ids'), 'control ids differ from the reference run'
    print('DRY RUN OK')


def run(a):
    out = Path(a.out)
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    me = sha(__file__)
    if not a.smoke:
        g = json.loads((out / 'check.json').read_text(encoding='utf-8'))
        assert g.get('ok') and g.get('script_sha256') == me, f'no passed smoke check of this script in {out}/check.json'
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: this script does not fall back to the CPU')
    dev = torch.device('cuda')
    W, pairs = WA.load_words(a.words)
    pair_ids = [p['id'] for p in pairs]
    M, E = D.load_official(a.model_dir)
    t_load = time.time()
    model, tok, args, ck, loaded, n_host = D.build(M, a.model_dir, 1, a.max_seq, dev)
    torch.set_default_device(dev)  # as dsv41_natural3.run: only after the model is built on the host
    stats = {'expert_s': 0.0}
    D.patch_runtime(M, dev, stats)
    R = D.Renderer(E, tok, False)
    ctrl = KN.control_ids(tok)
    V = args.vocab_size
    mask = torch.zeros(V, dtype=torch.bool, device='cpu')  # on the CPU with the log-probabilities (as the KL port)
    mask[torch.tensor([i for i in ctrl if i < V], dtype=torch.long, device='cpu')] = True
    mask[len(tok):] = True
    rid = json.loads((Path(a.check_ref) / 'identity.json').read_text(encoding='utf-8'))
    assert rid['n_control_ids'] == len(ctrl), ('control ids differ from the reference run', rid['n_control_ids'], len(ctrl))
    inf = Path(a.model_dir) / 'inference'
    ident = {'experiment': 'P2_kl_words_dsv41_nat3_20261112', 'order': 'natural3', 'script_sha256': me,
             'runner_sha256': sha(D.__file__), 'kernel_sha256': sha(D.K.__file__), 'kl_port_sha256': sha(KN.__file__),
             'words_along_sha256': sha(WA.__file__), 'official_model_py_sha256': sha(inf / 'model.py'),
             'official_engram_py_sha256': sha(inf / 'engram.py'), 'encoding_py_sha256': sha(Path(a.model_dir) / 'encoding' / 'encoding.py'),
             'tokenizer_json_sha256': sha(Path(a.model_dir) / 'tokenizer.json'), 'model': a.model_dir, 'tmpl_kwargs': R.kw,
             'lang': 'zh', 'request': D.REQ, 'reply_question': D.Q_REPLY, 'natural_system_ok': True, 'words_sha256': sha(a.words),
             'words_version': W['version'], 'pair_ids': pair_ids, 'n_bodies': 75, 'bodies_sha256': sha(a.bodies),
             'check_ref': str(a.check_ref), 'check_kind': 'nat', 'n_control_ids': len(ctrl), 'len_tokenizer': len(tok), 'vocab': V,
             'torch': torch.__version__, 'cuda': torch.version.cuda, 'group_bodies': a.group_bodies}
    shard_meta = {'shard': a.shard, 'n_shards': a.n_shards, 'smoke': a.smoke, 'gpu': torch.cuda.get_device_name(0),
                  'host': socket.gethostname(), 'load_s': time.time() - t_load, 'n_host_memory_tensors': n_host}
    D.atomic(out / f'identity_{"smoke" if a.smoke else f"shard{a.shard}"}.json', {**ident, **shard_meta})
    bodies = held_out(a)
    if a.smoke:
        bodies = bodies[:2]
    elif a.n_shards > 1:
        bodies = bodies[a.shard::a.n_shards]
    todo = [b for b in bodies if not (out / 'bodies' / f'{b["id"]}.json').exists()]
    print(f'{len(bodies)} bodies, {len(todo)} to run', flush=True)
    t0, done, chk = time.time(), 0, []
    for i in range(0, len(todo), a.group_bodies):
        if a.stop_file and Path(a.stop_file).exists():
            print(f'[stop] {a.stop_file} exists: stopping (rerun resumes)', flush=True)
            break
        grp = todo[i:i + a.group_bodies]
        tb = time.time()
        plans = [plan(R, b, pairs) for b in grp]
        seqs = [s for p_ in plans for s in p_[0]]
        poss = [q for p_ in plans for q in p_[1]]
        assert max(len(s) for s in seqs) + 2 <= a.max_seq, 'raise --max-seq'
        lps = KN.forward_all(M, model, seqs, poss, dev)
        off = 0
        for b, (s_, _, meta) in zip(grp, plans):
            rec = body_record(b, meta, lps[off:off + len(s_)], mask, a.check_ref)
            off += len(s_)
            rec['seconds'] = round((time.time() - tb) / len(grp), 1)
            if a.smoke:
                chk.append({'id': b['id'], **rec['check'], 'ours': rec['roles']})
                if not rec['check']['ok']:
                    continue
            elif not rec['check']['ok']:
                raise SystemExit(f'CHECK FAILED on {b["id"]}: {rec["check"]}')
            D.atomic(out / 'bodies' / f'{b["id"]}.json', rec)
            done += 1
        del lps
        print(f'{done}/{len(todo)} bodies, {time.time() - t0:.0f}s, experts {stats["expert_s"]:.0f}s; last {grp[-1]["id"]} '
              f'({time.time() - tb:.0f}s), check max rel {rec["check"]["max_rel"]:.2e}', flush=True)
    if a.smoke:
        ok = len(chk) == 2 and all(c['ok'] for c in chk)
        D.atomic(out / 'check.json', {'criterion_rel': CHECK_REL, 'bodies': chk, 'ok': ok, 'script_sha256': me})
        print('check (natural3):', json.dumps([{k: c[k] for k in ('id', 'max_rel', 'ok')} for c in chk]), flush=True)
        if not ok:
            raise SystemExit('CHECK FAILED (natural3): the role KLs differ from the reference run')
    n_all = sum(1 for b in bodies if (out / 'bodies' / f'{b["id"]}.json').exists())
    print(f'{"smoke" if a.smoke else f"shard {a.shard}"}: {n_all}/{len(bodies)} bodies on disk', flush=True)


def finalize(a):
    out = Path(a.out)
    W, pairs = WA.load_words(a.words)
    bodies = held_out(a)
    missing = [b['id'] for b in bodies if not (out / 'bodies' / f'{b["id"]}.json').exists()]
    assert not missing, f'{len(missing)} bodies missing, e.g. {missing[:3]}'
    idents = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('identity_*.json'))]
    keys = ('script_sha256', 'runner_sha256', 'kernel_sha256', 'kl_port_sha256', 'words_along_sha256', 'official_model_py_sha256',
            'encoding_py_sha256', 'tokenizer_json_sha256', 'words_sha256', 'tmpl_kwargs', 'n_control_ids')
    for other in idents[1:]:
        for k in keys:
            assert other[k] == idents[0][k], f'runs differ in {k}'
    ident = {k: v for k, v in idents[0].items() if k not in ('shard', 'n_shards', 'smoke', 'gpu', 'host', 'load_s', 'n_host_memory_tensors')}
    ident.update(runs=[{k: d.get(k) for k in ('shard', 'smoke', 'gpu', 'host', 'load_s')} for d in idents], finalize_script_sha256=sha(__file__))
    D.atomic(out / 'identity.json', ident)
    summarize(out, [p['id'] for p in pairs])
    (out / 'DONE').write_text(time.strftime('%F %T') + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--words', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--check-ref', required=True, help="the paper-run KL of this model (modeswitch/nat3/dsv41_flash)")
    p.add_argument('--smoke', action='store_true', help='the first 2 held-out bodies and the check')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--group-bodies', type=int, default=1, help='bodies per group (67 sequences each, as the words-along run)')
    p.add_argument('--max-seq', type=int, default=1024)
    p.add_argument('--stop-file', default=None)
    p.add_argument('--dry-run', action='store_true', help='tokenizer and encoder only: renderings, words-along identity, control ids')
    p.add_argument('--finalize', action='store_true')
    a = p.parse_args()
    if a.dry_run:
        dry_run(a)
        return
    assert a.out, '--out is required'
    if a.finalize:
        finalize(a)
        return
    try:
        run(a)
    except BaseException as exc:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        D.atomic(Path(a.out) / f'failed_{"smoke" if a.smoke else f"shard{a.shard}"}.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
