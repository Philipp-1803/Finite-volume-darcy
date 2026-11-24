import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

N = 4
a = 0.0
b = 1.0
h = (b - a) / N

K = 1
p_links = 1
p_rechts = 0
u_unten = 2
u_oben = 0

anzahl_zellen = N * N * N
punkte = np.zeros((anzahl_zellen, 3))
A = lil_matrix((anzahl_zellen, anzahl_zellen), dtype=float)
b_vec = np.zeros(anzahl_zellen)


def f(x, y, z):
    return 8 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z)


t_innen = K * h
t_rand = 2 * K * h

richtungen = {
    "links": (0, 0, -1),
    "rechts": (0, 0, 1),
    "unten": (0, -1, 0),
    "oben": (0, 1, 0),
    "vorne": (-1, 0, 0),
    "hinten": (1, 0, 0),
}

randdruck = {"links": p_links, "rechts": p_rechts}
randfluss = {"unten": u_unten, "oben": u_oben}

for k in range(anzahl_zellen):
    l, hilfsvariable = divmod(k, N * N)
    j, i = divmod(hilfsvariable, N)

    punkte[k] = (a + (i + 0.5) * h, a + (j + 0.5) * h, a + (l + 0.5) * h)

    x, y, z = punkte[k]
    b_vec[k] = h**3 * f(x, y, z)

    for richtung, (dl, dj, di) in richtungen.items():
        l_nachbar = l + dl
        j_nachbar = j + dj
        i_nachbar = i + di

        if 0 <= l_nachbar < N and 0 <= j_nachbar < N and 0 <= i_nachbar < N:
            nachbar = l_nachbar * N * N + j_nachbar * N + i_nachbar
            A[k, k] += t_innen
            A[k, nachbar] -= t_innen

        elif richtung in randdruck:
            A[k, k] += t_rand
            b_vec[k] += t_rand * randdruck[richtung]

        elif richtung in randfluss:
            if richtung == "unten":
                b_vec[k] -= h**2 * randfluss[richtung]
            if richtung == "oben":
                b_vec[k] += 0

A = A.tocsr()
druck = spsolve(A, b_vec)
druckgitter = druck.reshape(N, N, N)

flussgitter_x = np.zeros((N, N, N + 1))
flussgitter_y = np.zeros((N, N + 1, N))
flussgitter_z = np.zeros((N + 1, N, N))

flussgitter_x[:, :, 0] = 2 * K / h * (p_links - druckgitter[:, :, 0])
flussgitter_x[:, :, N] = 2 * K / h * (druckgitter[:, :, -1] - p_rechts)

flussgitter_y[:, 0, :] = -u_unten
flussgitter_y[:, N, :] = u_oben

for iz in range(N):
    for iy in range(N):
        for ix in range(N):
            druck_zelle = druckgitter[iz, iy, ix]

            if ix + 1 < N:
                flussgitter_x[iz, iy, ix + 1] = (
                    K / h * (druck_zelle - druckgitter[iz, iy, ix + 1])
                )

            if iy + 1 < N:
                flussgitter_y[iz, iy + 1, ix] = (
                    K / h * (druck_zelle - druckgitter[iz, iy + 1, ix])
                )

            if iz + 1 < N:
                flussgitter_z[iz + 1, iy, ix] = (
                    K / h * (druck_zelle - druckgitter[iz + 1, iy, ix])
                )

print("Zellmittelpunkte (x, y, z):\n", punkte)
print("Matrix A:\n", A)
print("Rechte Seite:\n", b_vec)
print("Druckgitter (z, y, x):\n", druckgitter)
print("Flussgitter in x-Richtung:\n", flussgitter_x)
print("Flussgitter in y-Richtung:\n", flussgitter_y)
print("Flussgitter in z-Richtung:\n", flussgitter_z)


# Gewählte z-Schicht: 0 bis N-1
schicht = 0
z_schnitt = a + (schicht + 0.5) * h

# Zellgrenzen und Zellmittelpunkte
grenzen = np.linspace(a, b, N + 1)
mitten = a + (np.arange(N) + 0.5) * h
X, Y = np.meshgrid(mitten, mitten)

# Flussdichten auf die Zellmittelpunkte mitteln
ux = 0.5 * (flussgitter_x[schicht, :, :-1] + flussgitter_x[schicht, :, 1:])
uy = 0.5 * (flussgitter_y[schicht, :-1, :] + flussgitter_y[schicht, 1:, :])

fig, ax = plt.subplots(figsize=(7, 6))

farben = ax.pcolormesh(
    grenzen,
    grenzen,
    druckgitter[schicht],
    cmap="viridis",
    shading="flat",
    edgecolors="gray",
    linewidth=0.5,
    vmin=druckgitter.min(),
    vmax=druckgitter.max(),
)

ax.quiver(X, Y, ux, uy, angles="xy", color="white")

fig.colorbar(farben, ax=ax, label="Druck p")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Druck und Strömung bei z = {z_schnitt:.3f}")
ax.set_aspect("equal")
ax.set_xlim(a, b)
ax.set_ylim(a, b)

plt.tight_layout()
plt.show()
