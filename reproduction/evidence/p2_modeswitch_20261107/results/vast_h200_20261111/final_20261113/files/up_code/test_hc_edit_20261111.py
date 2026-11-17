"""Tests of hc_edit_20261111 (11-11; CPU, synthetic). Single-stream: every helper equals the original expression bit for bit.
Four-stream: proj/back/gram/top_eig act on each stream exactly as the single-stream helper on that stream alone; an edit with the
recipient's own coordinates is the identity; the edit changes only the k coordinates of each stream.
usage: test_hc_edit_20261111.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hc_edit_20261111 import proj, back, gram, top_eig, random_basis, share  # noqa: E402

torch.manual_seed(0)
n, S, d, k = 37, 4, 96, 8
ok = []

# single stream: the original expressions
x = torch.randn(n, d)
B = torch.linalg.qr(torch.randn(d, k)).Q
c = torch.randn(n, k)
ok.append(('proj 2d', torch.equal(proj(x, B), x @ B)))
ok.append(('back 2d', torch.equal(back(c - proj(x, B), B), (c - x @ B) @ B.T)))
D = torch.randn(n, d)
ok.append(('gram 2d', torch.equal(gram(D), D.double().T @ D.double())))
G = gram(D)
w, U = torch.linalg.eigh(G)
ok.append(('top_eig 2d', torch.equal(top_eig(G, k), U[:, -k:].flip(-1).float().contiguous())))
g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
ok.append(('random 2d', torch.equal(random_basis(g1, G, 64), torch.linalg.qr(torch.randn(d, 64, generator=g2)).Q.float().contiguous())))

# four streams: per stream equals the single-stream computation on that stream
X = torch.randn(n, S, d)
Bs = torch.stack([torch.linalg.qr(torch.randn(d, k)).Q for _ in range(S)])
C = torch.randn(n, S, k)
P = proj(X, Bs)
ok.append(('proj 4s', all(torch.allclose(P[:, s], X[:, s] @ Bs[s], atol=1e-5) for s in range(S))))
E = back(C - P, Bs)
ok.append(('back 4s', all(torch.allclose(E[:, s], (C[:, s] - X[:, s] @ Bs[s]) @ Bs[s].T, atol=1e-5) for s in range(S))))
ok.append(('identity edit', torch.equal(back(P - proj(X, Bs), Bs), torch.zeros_like(X))))
Y = X + back(C - proj(X, Bs), Bs)
ok.append(('edit sets coords', torch.allclose(proj(Y, Bs), C, atol=1e-4)))
Q = torch.stack([torch.eye(d) - Bs[s] @ Bs[s].T for s in range(S)])
ok.append(('edit keeps complement', all(torch.allclose(Y[:, s] @ Q[s], X[:, s] @ Q[s], atol=1e-4) for s in range(S))))
D4 = torch.randn(n, S, d)
G4 = gram(D4)
ok.append(('gram 4s', all(torch.allclose(G4[s], D4[:, s].double().T @ D4[:, s].double()) for s in range(S))))
V4 = top_eig(G4, k)
ok.append(('top_eig 4s', all(min(float((V4[s][:, j] @ top_eig(G4[s], k)[:, j]).abs()), 1.0) > 0.9999 for s in range(S) for j in range(k))))
R4 = random_basis(torch.Generator().manual_seed(7), G4, 64)
ok.append(('random 4s orthonormal', all(torch.allclose(R4[s].T @ R4[s], torch.eye(64), atol=1e-4) for s in range(S))))
ok.append(('share 4s', len(share(G4, V4)) == S and all(0 < v <= 1 for v in share(G4, V4))))
for name, v in ok:
    print(('PASS' if v else 'FAIL'), name)
print('ALL PASS' if all(v for _, v in ok) else 'SOME FAILED')
sys.exit(0 if all(v for _, v in ok) else 1)
