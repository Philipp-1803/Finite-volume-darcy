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
u_unten = 2
u_oben = 0

anzahl_zellen = N * N
punkte = np.zeros((anzahl_zellen, 2))
A = lil_matrix((anzahl_zellen, anzahl_zellen), dtype=float)
b_vec = np.zeros(anzahl_zellen)

t_innen = K
t_rand = 2 * K

richtungen = {
    "links": (0, -1),
    "rechts": (0, 1),
    "unten": (-1, 0),
    "oben": (1, 0),
}
randdruck = {"links": p_links, "rechts": p_rechts}
randfluss = {"unten": u_unten, "oben": u_oben}

for k in range(anzahl_zellen):
    j, i = divmod(k, N)
    punkte[k] = (a + (i + 0.5) * h, a + (j + 0.5) * h)

    for richtung, (dj, di) in richtungen.items():
        j_nachbar = j + dj
        i_nachbar = i + di

        if 0 <= j_nachbar < N and 0 <= i_nachbar < N:
            nachbar = j_nachbar * N + i_nachbar
            A[k, k] += t_innen
            A[k, nachbar] -= t_innen

        elif richtung in randdruck:
            A[k, k] += t_rand
            b_vec[k] += t_rand * randdruck[richtung]

        elif richtung in randfluss:
            if richtung == "unten":
                b_vec[k] -= h * randfluss[richtung]
            if richtung == "oben":
                b_vec[k] += 0

A = A.tocsr()
druck = spsolve(A, b_vec)
druckgitter = druck.reshape(N, N)

flussgitter_x = lil_matrix((N, N + 1), dtype=float)
flussgitter_y = lil_matrix((N + 1, N), dtype=float)

for zell_index in range(N):
    for kanten_index in range(N + 1):
        if 0 < kanten_index < N:
            flussgitter_x[zell_index, kanten_index] = (
                K
                / h
                * (
                    druckgitter[zell_index, kanten_index - 1]
                    - druckgitter[zell_index, kanten_index]
                )
            )
            flussgitter_y[kanten_index, zell_index] = (
                K
                / h
                * (
                    druckgitter[kanten_index - 1, zell_index]
                    - druckgitter[kanten_index, zell_index]
                )
            )
        if kanten_index == 0:
            flussgitter_x[zell_index, kanten_index] = (
                2 * K / h * (p_links - druckgitter[zell_index, 0])
            )
            flussgitter_y[kanten_index, zell_index] = -u_unten

        if kanten_index == N:
            flussgitter_x[zell_index, kanten_index] = (
                2 * K / h * (druckgitter[zell_index, N - 1] - p_rechts)
            )
            flussgitter_y[kanten_index, zell_index] = u_oben


# print("Zellmittelpunkte (x, y):\n", punkte)
# print("Matrix A:\n", A)
# print("Rechte Seite:\n", b_vec)
print("Druckgitter (unterste Gitterzeile zuerst):\n", druckgitter)
print("Flussgitter in x-Richtung:\n", flussgitter_x.toarray())
print("Flussgitter in y-Richtung:\n", flussgitter_y.toarray())
