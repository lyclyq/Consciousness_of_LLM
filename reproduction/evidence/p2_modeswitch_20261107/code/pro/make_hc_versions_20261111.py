"""Write the *_hc_20261111 versions of the three mechanism scripts (11-11) from the originals by exact text replacements, each of
which must occur exactly the stated number of times. The changes: turn-start tokens from hc_edit_20261111.turn_start_ids, and the
subspace algebra through proj/back/gram/top_eig/random_basis/share/overlap, which equal the original expressions on single-stream
models and act per stream on four-stream layer outputs [batch, tokens, streams, hidden].
usage: make_hc_versions_20261111.py SRC_DIR OUT_DIR"""
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
IMPORT = "from hc_edit_20261111 import turn_start_ids, proj, back, gram, top_eig, random_basis, share as hc_share, overlap as hc_overlap  # noqa: E402\n"
OLD_CANDS = "cands = [tok.convert_tokens_to_ids(t) for t in ('<|start|>', '<|im_start|>')]"

R05 = [
    ("from crossmodel_suite import Renderer, load, attr_question, ORDERS, LETTERS  # noqa: E402\n",
     "from crossmodel_suite import Renderer, load, attr_question, ORDERS, LETTERS  # noqa: E402\n" + IMPORT, 1),
    (OLD_CANDS, "cands = turn_start_ids(tok)", 2),
    ("                    state['coords'][row][l] = (x @ B).clone()\n", "                    state['coords'][row][l] = proj(x, B).clone()\n", 1),
    ("                    delta = (c - x @ B) @ B.T\n", "                    delta = back(c - proj(x, B), B)\n", 1),
    ("(V if which == 'V' else Rr)[l][:, :k].contiguous()", "(V if which == 'V' else Rr)[l][..., :k].contiguous()", 1),
    ("                    Gq[l] = Gq.get(l, 0) + Dq.T @ Dq\n                    Gm[l] = Gm.get(l, 0) + Dm.T @ Dm\n",
     "                    Gq[l] = Gq.get(l, 0) + gram(Dq)\n                    Gm[l] = Gm.get(l, 0) + gram(Dm)\n", 1),
    ("""    def share(G, V8):  # fraction of the trace of D^T D in the 8-dim space
        return float((V8.double().T @ G @ V8.double()).trace() / G.trace())
    for l in band:
        for G, V in ((Gq[l], Vq), (Gm[l], Vm)):
            w, U = torch.linalg.eigh(G)
            V[l] = U[:, -K_MAX:].flip(-1).float().contiguous()
        d = Gq[l].shape[0]
        for j in range(5):
            g = torch.Generator().manual_seed(SEED0 + j + l)
            Rs[j][l] = torch.linalg.qr(torch.randn(d, K_MAX, generator=g)).Q.float().contiguous()
        q8, m8 = Vq[l][:, :8], Vm[l][:, :8]
        diag['layers'][l] = {'share_q_in_q': round(share(Gq[l], q8), 4), 'share_m_in_m': round(share(Gm[l], m8), 4),
                             'share_m_in_q': round(share(Gq[l], m8), 4), 'share_q_in_m': round(share(Gm[l], q8), 4),
                             'overlap_q_m': round(float((torch.linalg.svdvals(q8.T @ m8) ** 2).mean()), 4),
                             'share_random_in_q': round(share(Gq[l], Rs[0][l][:, :8]), 5)}
""", """    def rd(v, n):  # four-stream fits: the mean over the streams, the streams under per_stream
        return round(sum(v) / len(v), n) if isinstance(v, list) else round(v, n)
    eig_dev = 'cuda:0' if Gq[band[0]].dim() == 3 else None  # four-stream Gram matrices: eigh per stream on the GPU
    diag['streams'] = int(Gq[band[0]].shape[0]) if Gq[band[0]].dim() == 3 else 1
    for l in band:
        for G, V in ((Gq[l], Vq), (Gm[l], Vm)):
            V[l] = top_eig(G, K_MAX, eig_dev)
        for j in range(5):
            g = torch.Generator().manual_seed(SEED0 + j + l)
            Rs[j][l] = random_basis(g, Gq[l], K_MAX)
        q8, m8 = Vq[l][..., :8], Vm[l][..., :8]
        vals = {'share_q_in_q': (hc_share(Gq[l], q8), 4), 'share_m_in_m': (hc_share(Gm[l], m8), 4),
                'share_m_in_q': (hc_share(Gq[l], m8), 4), 'share_q_in_m': (hc_share(Gm[l], q8), 4),
                'overlap_q_m': (hc_overlap(q8, m8), 4), 'share_random_in_q': (hc_share(Gq[l], Rs[0][l][..., :8]), 5)}
        diag['layers'][l] = {k: rd(v, n) for k, (v, n) in vals.items()}
        if diag['streams'] > 1:
            diag['layers'][l]['per_stream'] = {k: [round(x, n) for x in v] for k, (v, n) in vals.items()}
""", 1),
]
GAMES = [
    ("from crossmodel_role_swap import align, CLOSE  # noqa: E402\n", "from crossmodel_role_swap import align, CLOSE  # noqa: E402\n" + IMPORT, 1),
    (OLD_CANDS, "cands = turn_start_ids(tok)", 2),
    ("                    state['coords'][row][l] = (x @ B).clone()\n", "                    state['coords'][row][l] = proj(x, B).clone()\n", 1),
    ("                    h[row, sl] = (x + (c - x @ B) @ B.T).to(h.dtype)\n", "                    h[row, sl] = (x + back(c - proj(x, B), B)).to(h.dtype)\n", 1),
    ("[l][:, :k]", "[l][..., :k]", 1),
]
SWAP = [
    ("    cands = [t for t in (tok.convert_tokens_to_ids(x) for x in ('<|start|>', '<|im_start|>')) if isinstance(t, int) and t >= 0]\n",
     "    cands = turn_start_ids(tok)\n", 1),
    ("F[w][l][:, :a.k].contiguous()", "F[w][l][..., :a.k].contiguous()", 1),
    ("                    st['coords'][w][l] = (x @ Bs[w][l].to(h.device)).cpu()\n", "                    st['coords'][w][l] = proj(x, Bs[w][l].to(h.device)).cpu()\n", 1),
    ("                delta = (c - x @ B) @ B.T\n", "                delta = back(c - proj(x, B), B)\n", 1),
]


def make(name, out_name, reps, anchor_import=None):
    s = (SRC / name).read_bytes().decode('utf-8')
    crlf = '\r\n' in s  # crossmodel_games.py is CRLF: replace on LF text, write back with the original line ends
    s = s.replace('\r\n', '\n')
    for old, new, n in reps:
        c = s.count(old)
        assert c == n, f'{name}: {c} occurrences of {old[:70]!r}, expected {n}'
        s = s.replace(old, new)
    if anchor_import:
        a, b = anchor_import
        assert s.count(a) == 1, f'{name}: import anchor'
        s = s.replace(a, a + b)
    head = (f'"""{out_name}: {name} with hc_edit_20261111 (turn-start tokens of every template; one subspace per stream on four-stream\n'
            f'models). Identical on single-stream models with <|start|> / <|im_start|> templates. Original docstring follows.\n')
    assert s.startswith('"""')
    s = head + s[3:]
    if crlf:
        s = s.replace('\n', '\r\n')
    (OUT / out_name).write_bytes(s.encode('utf-8'))
    print('wrote', out_name)


OUT.mkdir(parents=True, exist_ok=True)
make('crossmodel_role_swap_r05.py', 'crossmodel_role_swap_r05_hc_20261111.py', R05)
make('crossmodel_games.py', 'crossmodel_games_hc_20261111.py', GAMES)
swap_src = (SRC / 'mode_switch_swap_20261108.py').read_bytes().decode('utf-8')
imp_anchor = next(l for l in swap_src.splitlines(True) if l.startswith('from crossmodel_role_swap') or l.startswith('from crossmodel_suite import'))
make('mode_switch_swap_20261108.py', 'mode_switch_swap_hc_20261111.py', SWAP, (imp_anchor, IMPORT))
