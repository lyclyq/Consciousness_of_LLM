"""Helpers for the mechanism scripts on four-stream models and on templates without <|start|> / <|im_start|> (11-11, user: GLM-5.3-Flash
"按流" and EXAONE in the mechanism group). Used by the *_hc_20261111 versions of crossmodel_role_swap_r05.py, crossmodel_games.py and
mode_switch_swap_20261108.py. On a single-stream model and a <|start|> / <|im_start|> template every function reduces to the
original expression (x @ B, d @ B.T, the same turn-start ids), so those readouts are computed exactly as before.
  turn_start_ids(tok)  ids of the turn-start tokens of the template: <|start|> (gpt-oss), <|im_start|> (Qwen, MiMo), <|system|>,
                       <|user|>, <|assistant|> (GLM), [|system|], [|user|], [|assistant|] (EXAONE); only tokens that exist as single
                       tokens of the vocabulary. The generation prompt starts at the last of them.
  proj(x, B), back(d, B)  coordinates and their embedding: B [d, k] for a layer state [n, d] (x @ B, d @ B.T); B [S, d, k] for a
                       four-stream state [n, S, d], one subspace per stream (einsum over each stream separately).
  gram(D)              D^T D in float64: [d, d] for [n, d]; [S, d, d] (per stream) for [n, S, d].
  top_eig(G, k)        the k leading eigenvectors (descending) of each Gram matrix: [d, k] or [S, d, k].
  random_basis(gen, shape_of_G, k)  QR of a Gaussian: [d, k] from torch.randn(d, k) (as before) or [S, d, k] from torch.randn(S, d, k).
"""
from __future__ import annotations

TURN_TOKENS = ('<|start|>', '<|im_start|>', '<|system|>', '<|user|>', '<|assistant|>', '[|system|]', '[|user|]', '[|assistant|]')


def turn_start_ids(tok):
    out = []
    for t in TURN_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and tok.convert_ids_to_tokens(i) == t:
            out.append(i)
    assert out, 'no turn-start token of the template found'
    return out


def proj(x, B):
    if B.dim() == 2:
        return x @ B
    import torch
    return torch.einsum('nsd,sdk->nsk', x, B)


def back(d, B):
    if B.dim() == 2:
        return d @ B.T
    import torch
    return torch.einsum('nsk,sdk->nsd', d, B)


def gram(D):
    D = D.double()
    if D.dim() == 2:
        return D.T @ D
    import torch
    return torch.einsum('nsd,nse->sde', D, D)


def top_eig(G, k, device=None):
    import torch
    if G.dim() == 2:
        w, U = torch.linalg.eigh(G)
        return U[:, -k:].flip(-1).float().contiguous()
    w, U = torch.linalg.eigh(G.to(device) if device is not None else G)
    return U[..., -k:].flip(-1).float().cpu().contiguous()


def random_basis(gen, G, k):
    import torch
    if G.dim() == 2:
        return torch.linalg.qr(torch.randn(G.shape[0], k, generator=gen)).Q.float().contiguous()
    return torch.linalg.qr(torch.randn(G.shape[0], G.shape[1], k, generator=gen)).Q.float().contiguous()


def share(G, V8):
    """fraction of the trace of D^T D in the 8-dim space; per stream for four-stream fits (list)"""
    if G.dim() == 2:
        return float((V8.double().T @ G @ V8.double()).trace() / G.trace())
    return [float((V8[s].double().T @ G[s] @ V8[s].double()).trace() / G[s].trace()) for s in range(G.shape[0])]


def overlap(A8, B8):
    """mean squared cosine of the principal angles of two 8-dim spaces; per stream for four-stream fits (list)"""
    import torch
    if A8.dim() == 2:
        return float((torch.linalg.svdvals(A8.T @ B8) ** 2).mean())
    return [float((torch.linalg.svdvals(A8[s].T @ B8[s]) ** 2).mean()) for s in range(A8.shape[0])]
