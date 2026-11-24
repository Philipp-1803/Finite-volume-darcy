import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

N = 4
a = 0.0
b = 1.0
h = (b - a) / N

K = 1
p_links = 1
p_rechts = 0

zellmittelpunkte = a + (np.arange(N) + 0.5) * h
zellgrenzen = a + np.arange(N + 1) * h

t_innen = K / h
t_rand = 2 * K / h

richtungen = {"links": -1, "rechts": 1}
randdruck = {"links": p_links, "rechts": p_rechts}

A = lil_matrix((N, N), dtype=float)
b_vec = np.zeros(N)

for ix in range(N):
    for richtung, dix in richtungen.items():
        ix_nachbar = ix + dix

        if 0 <= ix_nachbar < N:
            A[ix, ix] += t_innen
            A[ix, ix_nachbar] -= t_innen
        else:
            A[ix, ix] += t_rand
            b_vec[ix] += t_rand * randdruck[richtung]

A = A.tocsr()
druck = spsolve(A, b_vec)

flussgitter_x = np.zeros(N + 1)

flussgitter_x[0] = t_rand * (p_links - druck[0])
flussgitter_x[N] = t_rand * (druck[-1] - p_rechts)

for ix in range(N - 1):
    flussgitter_x[ix + 1] = t_innen * (druck[ix] - druck[ix + 1])

print("Zellmittelpunkte:", zellmittelpunkte)
print("Zellgrenzen:", zellgrenzen)
print("Matrix A:\n", A.toarray())
print("Rechte Seite:", b_vec)
print("Druck:", druck)
print("Fluss an den Zellgrenzen:", flussgitter_x)
